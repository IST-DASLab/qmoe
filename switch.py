# Copyright (C) QMoE.2023 Elias Frantar (elias.frantar@ist.ac.at)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#         http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
# In case you want a different HuggingFace home for downloading massive models
# os.environ['HF_HOME'] = ''


import argparse
import collections
import sys
import time
import types

import torch
import torch.nn as nn
import transformers

from datautils import *
from gptq import *
from quant import *


# Memory manager for lazily loading weights from different model shards.
class ShardLoader:

  def __init__(self, index, max_shards=10):
    self.index = index
    self.loaded = collections.OrderedDict()
    self.max_shards = max_shards # maximum number of shards kept in memory

  def get(self, name):
    shard = self.index[name]
    if shard not in self.loaded:
      self.loaded[shard] = torch.load(shard)
      # Free least recently used shard
      if len(self.loaded) > self.max_shards:
        self.loaded.popitem(last=False)
    return self.loaded[shard][name]

  def load(self, module, root='', dev=None):
    sd = module.state_dict()
    for name in sd:
      sd[name] = self.get(root + '.' + name if root else name)
    module.load_state_dict(sd)
    module.to(dev)


# Linear layer where weights must be loaded explicitly.
class LazyLinear(nn.Module):

  def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
    super().__init__()
    self._in_features = in_features
    self._out_features = out_features
    self._bias = bias
    self._device = device
    self._dtype = dtype
    # HF SwitchTransformer fails if there is no weight attribute
    self.weight = None

  # Register from where to load weights.
  def set_resources(self, name, loader):
    self.name = name
    self.loader = loader

  def load(self, dev):
    self.linear = nn.Linear(
      self._in_features, self._out_features, self._bias, self._device, self._dtype
    )
    self.loader.load(self.linear, root=self.name, dev=dev)

  def free(self):
    del self.linear

  def forward(self, inp):
    return self.linear(inp)


# Find all layers of a certain type in a given module.
def find_layers(module, layers=[LazyLinear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def load_switch(name):
  # Extract HuggingFace sharding map required for memory management 
  archive = transformers.utils.cached_file(name, transformers.utils.WEIGHTS_INDEX_NAME)
  archive, meta = transformers.utils.hub.get_checkpoint_shard_files(name, archive)
  archive = {f.split('/')[-1]: f for f in archive}
  for key in meta['weight_map']:
    meta['weight_map'][key] = archive[meta['weight_map'][key]]
  index = meta['weight_map']

  linear = nn.Linear
  # Do not explicitly allocate any linear layers when creating a model
  setattr(nn, 'Linear', LazyLinear)
  with transformers.modeling_utils.no_init_weights():
    config = transformers.SwitchTransformersConfig.from_pretrained(name)
    # We don't want any randomness during inference
    config.router_jitter_noise = 0
    config.expert_capacity = 1024
    # There are some bugs in loading the largest Switch that we need to work around
    if 'switch-c-2048' in name:
      config.torch_dtype = torch.bfloat16
      config.num_decoder_layers = 15
      config.num_layers = 15
      config.tie_word_embeddings = False # causes different last layer handling 
      import_path = transformers.models.switch_transformers.modeling_switch_transformers
      # The c-2048 model has only sparse layers which HuggingFace cannot currently deal with
      class AlwaysSparse(import_path.SwitchTransformersBlock):
        def __init__(self, config, has_relative_attention_bias=False, is_sparse=False):
            super().__init__(
                config, has_relative_attention_bias=has_relative_attention_bias, is_sparse=True
            )
      setattr(import_path, 'SwitchTransformersBlock', AlwaysSparse)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.torch_dtype) # ensure we load in correct type
    model = transformers.SwitchTransformersForConditionalGeneration(config)
    torch.set_default_dtype(default_dtype)
  setattr(nn, 'Linear', linear)

  # Set up shard loader with correct resource pointers.
  loader = ShardLoader(index)
  for layername, lazy in find_layers(model).items():
    lazy.set_resources(layername, loader)
    if 'router' in layername:
      lazy._dtype = torch.float
    else:
      lazy._dtype = config.torch_dtype

  # Large checkpoints store embeddings differently which leads to loading problems
  if name in ['google/switch-xxl-128', 'google/switch-c-2048']:
    shared_embed = loader.get('shared.weight')
    lm_head = loader.get('decoder.lm_head.weight')
    embeds = {
      'encoder.embed_tokens.weight': shared_embed,
      'decoder.embed_tokens.weight': shared_embed,
      'lm_head.weight': lm_head,
    }
    path = '%s_embeds.pt' % name.replace('google/', '')
    if not os.path.exists(path):
      torch.save(embeds, path)
    for name in embeds:
      loader.index[name] = path

  model.eval()
  return model, loader


# Context manager for temporarily loading lazy layers.
class LazyLoad:

  def __init__(self, lazys, dev):
    self.lazys = lazys
    self.dev = dev

  def __enter__(self):
    for lazy in self.lazys:
      lazy.load(self.dev)

  def __exit__(self, exc_type, exc_val, exc_tb):
    for lazy in self.lazys:
      lazy.free()


# Move entire dict contents to given device.
def dict_to(d, dev):
  d = dict(**d) # copy original dict
  for k in d:
    if isinstance(d[k], torch.Tensor):
      d[k] = d[k].to(dev)
  return d


# List buffer datastructure for efficient per-sample and full-mask access.
class ListBuffer:

  def __init__(self, sizes, dim=None, dtype=None, dev=None):
    self.slices = []
    tot = 0
    for size in sizes:
      self.slices.append((tot, tot + size))
      tot += size 
    self.buffer = torch.empty((tot, dim) if dim else tot, dtype=dtype, device=dev)

  def __len__(self):
    return len(self.slices)

  def __getitem__(self, key):
    i, j = self.slices[key]
    # We expect a batch-dimension of 1
    return self.buffer[i:j].unsqueeze(0)

  def __setitem__(self, key, value):
    i, j = self.slices[key]
    self.buffer[i:j] = value.squeeze(0)


# Run the model until a given layer and capture results.
def run_until(model, layer, inps, kwargs, dev, outs=None):
  if not outs:
    outs = inps
  cache = {'i': 0}

  # Break out of forward pass with this exception
  class StopInference(Exception):
    pass

  def new_forward(self, *args, **kwargs1):
    outs[cache['i']] = args[0].cpu()
    cache['i'] += 1
    raise StopInference

  forward = layer.forward
  layer.forward = types.MethodType(new_forward, layer) 

  for i in range(len(inps)):
    try:
      model(inps[i].to(dev), **dict_to(kwargs[i], dev))
    except StopInference:
      pass

  layer.forward = forward


@torch.no_grad()
def switch_forward(
  model, loader, data, decoder_data, trainsamples, valmeta, dev,
  par_exp=16, max_tokens_mul=4
):
  use_cache = model.config.use_cache
  model.config.use_cache = False # avoid any extra memory usage

  if args.save:
    from sub1 import Sub1CheckpointManager
    checkpointer = Sub1CheckpointManager(args.save, model, loader, find_layers(model))

  for root in ['encoder', 'decoder']:
    part = getattr(model, root)

    skip_mask = None
    if root == 'encoder':
      inps = list(data)
      kwargs = [{} for _ in inps]
      loader.load(part.embed_tokens, root=root + '.embed_tokens', dev=dev)
      buffer = ListBuffer([inp.shape[1] for inp in inps], dim=model.config.d_model, dtype=model.config.torch_dtype)
      if not args.no_mask_special:
        skip_mask = ListBuffer([inp.shape[1] for inp in inps], dim=None, dtype=torch.bool)
        for i in range(len(inps)):
          skip_mask[i] = inps[i] < 32000 # simply skip mask token in encoder input
      run_until(part, part.block[0], inps, kwargs, dev, outs=buffer)
    else:
      # Decoder inference requires encoder results
      kwargs = [{'encoder_hidden_states': inps[i]} for i in range(len(inps))]
      inps = list(decoder_data)
      loader.load(part.embed_tokens, root=root + '.embed_tokens', dev=dev)
      buffer = ListBuffer([inp.shape[1] for inp in inps], dim=model.config.d_model, dtype=model.config.torch_dtype)
      if not args.no_mask_special:
        skip_mask = ListBuffer([inp.shape[1] for inp in inps], dim=None, dtype=torch.bool)
        for i in range(len(inps)):
          # Skip tokens >>before<< mask tokens in decoder output as they are used to predict the latter
          skip_mask[i] = inps[i] >= 32000
          skip_mask[i] = ~torch.cat([skip_mask[i][:, 1:], torch.BoolTensor([[True]])], 1)
      run_until(part, part.block[0], inps, kwargs, dev, outs=buffer)
    inps = buffer

    for i, layer in enumerate(part.block):
      print(i)
      loader.load(layer, root=root + '.block.%d' % i, dev=dev)

      if i != 0:
        # Attention bias must be copied from the first layer in each model part 
        attn = layer.layer[0].SelfAttention
        attn.has_relative_attention_bias = True
        attn.relative_attention_bias = part.block[0].layer[0].SelfAttention.relative_attention_bias

      if root == 'decoder':
        # For decoder inference we need to pass encoder outputs and attention masks
        def run(inp, **kwargs):
          mask = torch.ones(inp.shape[0], inp.shape[1], device=dev)
          kwargs = dict(kwargs)
          kwargs['attention_mask'] = model.decoder.get_extended_attention_mask(mask, inp.shape[:2])
          return layer(inp, **kwargs)
      else:
        run = layer

      if not layer.is_sparse:
        # Simply run through the entire dense layer
        with LazyLoad(find_layers(layer).values(), dev):
          for j in range(len(inps)):
            inps[j] = run(inps[j].to(dev), **dict_to(kwargs[j], dev))[0].cpu()
      else:
        def scoped(): # make sure all memory is freed after this call
          nonexpert = [v for k, v in find_layers(layer).items() if 'expert' not in k]
          ffn = layer.layer[1 if root == 'encoder' else 2]

          # Run through dense part of the block and collect router information for each token 
          with LazyLoad(nonexpert, dev):
            run_until(run, ffn, inps, kwargs, dev)

            sizes = [inps[i].shape[1] for i in range(len(inps))]
            expert_index = ListBuffer(sizes, dim=None, dtype=torch.long)
            probs = ListBuffer(sizes, 1, dev=dev)
            for j in range(len(inps)):
              inp = ffn.layer_norm(inps[j].to(dev))
              mask, prob = ffn.mlp.router(inp)[:2]
              expert_index[j] = torch.argmax(mask, -1).cpu()
              probs[j] = prob

          traintokens = inps.slices[trainsamples - 1][1] # number of training tokens among all tokens
          experts = list(ffn.mlp.experts.values())
          # Maximum tokens used per expert for compression
          # This is to avoid OOM in rare edge cases when massive token counts are sent to a single expert
          max_tokens = int(max_tokens_mul * traintokens / len(experts))

          # Process multiple experts in parallel
          for j1 in range(0, len(experts), par_exp):
            tick1 = time.time()
            j2 = j1 + par_exp

            # Fetch data corresponding to current set of experts
            tick = time.time()
            expert_tokens_idx = []
            expert_inps = []
            expert_skip_mask = []
            for j in range(j1, j2):
              # Vectorized access to full buffer behind sample list
              expert_tokens_idx.append(torch.nonzero(expert_index.buffer == j).flatten())
              expert_inps.append(inps.buffer[expert_tokens_idx[-1][:max_tokens], :])
              expert_inps[-1] = expert_inps[-1].to(dev)
              if skip_mask:
                expert_skip_mask.append(skip_mask.buffer[expert_tokens_idx[-1][:max_tokens]])
                expert_skip_mask[-1] = expert_skip_mask[-1].to(dev)
            torch.cuda.synchronize()
            print([e.shape[0] for e in expert_tokens_idx])
            print('Extract to GPU', time.time() - tick)

            load = sum([list(find_layers(e).values()) for e in experts[j1:j2]], [])
            with LazyLoad(load, dev):
              subsets_lazy = [find_layers(e) for e in experts[j1:j2]]
              subsets = [{n: l.linear for n, l in s.items()} for s in subsets_lazy]

              if args.wbits < 16 and root != args.skip:
                tick = time.time()

                order = [['wi', 'wo']] if not args.true_sequential else [['wi'], ['wo']]
                for names in order:
                  # Calculate Hessians separately for each expert.
                  Hs = []
                  for j, subset in enumerate(subsets):
                    Hs.append({})
                    def calc_hessian(name):
                      def tmp(layer, inp, out):
                        Hs[-1][name] = hessian(inp[0].data, baseline=args.nearest)
                      return tmp
                    handles = []
                    for name in subset:
                      if name in names:
                        handles.append(subset[name].register_forward_hook(calc_hessian(name)))
                    # CRITICAL: Avoid leaking any valtokens into the Hessians used for compression!
                    valcount = torch.sum((expert_index.buffer[traintokens:] == (j1 + j)).int())
                    dropped_tokens = max(expert_tokens_idx[j].shape[0] - max_tokens, 0)
                    valstart = len(expert_inps[j]) - max(valcount - dropped_tokens, 0)
                    tmp = expert_inps[j][:valstart, :]
                    if skip_mask:
                      tmp = tmp[expert_skip_mask[j][:valstart]]
                    experts[j1 + j](ffn.layer_norm(tmp))
                  torch.cuda.synchronize()
                  print('Compute Hessians', time.time() - tick)
                  for h in handles:
                    h.remove()

                  # Compress all layers in parallel across all experts.
                  for name in subsets[0]:
                    if name not in names:
                      continue
                    # Stack to 3D tensors to call batch GPTQ implementation 
                    W = torch.stack([s[name].weight.data for s in subsets])
                    H = torch.stack([h[name] for h in Hs])
                    quantizer = Quantizer()
                    quantizer.configure(args.wbits, sym=False)
                    Q = batch_gptq(
                      W, H, quantizer, percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.actorder
                    )
                    for j in range(Q.shape[0]):
                      subsets[j][name].weight.data = Q[j]
                      if args.separate_eval:
                        name1 = subsets_lazy[j][name].name + '.weight'
                        loader.loaded[loader.index[name1]][name1] = Q[j].cpu()

                torch.cuda.synchronize()
                print('GPTC', time.time() - tick)

              # Now run all tokens through compressed (or uncompressed) experts
              tick = time.time()
              for j in range(len(expert_inps)):
                if expert_tokens_idx[j].shape[0] <= max_tokens:
                  expert_inps[j] = experts[j1 + j](ffn.layer_norm(expert_inps[j]))
                  expert_inps[j] *= probs.buffer[expert_tokens_idx[j], :]
              torch.cuda.synchronize()
              print('Run through compressed', time.time() - tick)

              if args.save:
                for j in range(j1, j2):
                  name = '%s.block.%d.layer.%d.mlp.experts.expert_%d' % (
                    root, i, 1 if root == 'encoder' else 2, j
                  )
                  checkpointer.add_expert(name, experts[j])

              # Write results back into central buffer.
              tick = time.time()
              for j in range(len(expert_inps)):
                if expert_tokens_idx[j].shape[0] > max_tokens:
                  # In case we could not load all samples for an expert initially due to memory concerns, we need
                  # to process them explicitly in batches now.
                  expert_inps[j] = None # free memory
                  for k1 in range(0, expert_tokens_idx[j].shape[0], max_tokens):
                    k2 = k1 + max_tokens
                    inp = inps.buffer[expert_tokens_idx[j][k1:k2], :].to(dev)
                    inp = experts[j1 + j](ffn.layer_norm(inp))
                    inp *= probs.buffer[expert_tokens_idx[j][k1:k2], :]
                    inp = inp.cpu()
                    inps.buffer[expert_tokens_idx[j][k1:k2], :] += inp
                else:
                  expert_inps[j] = expert_inps[j].cpu()
                  inps.buffer[expert_tokens_idx[j], :] += expert_inps[j]
              torch.cuda.synchronize()
              print('Residual store', time.time() - tick)

            torch.cuda.synchronize()
            print(time.time() - tick1)

        scoped()

        if args.save:
          checkpointer.save_experts()

      if i != 0 and args.separate_eval:
        attn.has_relative_attention_bias = False
        del attn.relative_attention_bias

    loader.load(part.final_layer_norm, root=root + '.final_layer_norm', dev=dev)
    for i in range(len(inps)):
      inps[i] = part.final_layer_norm(inps[i].to(dev)).cpu()

  # Compute loss only on validation data
  with LazyLoad([model.lm_head], dev):
    totsum = 0
    totlen = 0
    for i in range(trainsamples, len(inps)):
      hidden_states = inps[i].to(dev)
      # For large models this is true
      if model.config.tie_word_embeddings:
        hidden_states *= (model.model_dim ** -.5)
      lm_logits = model.lm_head(hidden_states)
      shift_logits = lm_logits[:, :-1, :].contiguous()
      shift_labels = decoder_data[i].to(dev)[:, 1:]
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
      totsum += loss.float() * hidden_states.shape[1]
      totlen += hidden_states.shape[1]
      if (i - trainsamples + 1) == valmeta[0][1]:
        print(valmeta[0][0] + ':', (totsum / totlen).item())
        totsum = 0
        totlen = 0
        valmeta.pop(0)
  
  if args.save:
    checkpointer.finalize()

  model.config.use_cache = use_cache


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
    'model', type=str,
    help='Switch model to load; pass `google/switch-X`.'
  )
  parser.add_argument(
    '--trainsamples', type=int, default=128,
    help='Number of calibration data samples.'
  )
  parser.add_argument(
    '--valsamples', type=int, default=128,
    help='Number of validation data samples.'
  )
  parser.add_argument(
    '--percdamp', type=float, default=.1,
    help='Percent of the average Hessian diagonal to use for dampening.'
  )
  parser.add_argument(
    '--nearest', action='store_true',
    help='Whether to run the RTN baseline.'
  ) 
  parser.add_argument(
    '--wbits', type=float, default=16, choices=[1.5, 2, 16],
    help='#bits to use for quantization; use 16 for evaluating base model.'
  )
  parser.add_argument(
    '--groupsize', type=int, default=-1,
    help='Groupsize to use for quantization; default uses full row.'
  )
  parser.add_argument(
    '--actorder', action='store_true',
    help='Whether or not to use the activation order heuristic.'
  )
  parser.add_argument(
    '--true-sequential', action='store_true',
    help='Whether or not to run in true sequential mode.'
  )
  parser.add_argument(
    '--skip', default='',
    help='Whether to skip pruning the encoder or the decoder.'
  )
  parser.add_argument(
    '--no-mask-special', action='store_true',
    help='Do not skip special tokens for reconstruction.'
  )
  parser.add_argument(
    '--separate-eval', action='store_true',
    help='Perform a separate evaluation pass for verification.'
  )
  parser.add_argument(
    '--detaileval', action='store_true',
    help='Whether to perform evaluation on additional datasets.'
  )
  parser.add_argument(
    '--save', type=str, default='',
    help='Where to store the model.'
  )

  args = parser.parse_args() 


  if args.save and args.wbits != 1.5:
    raise ValueError('Only saving ternary models is supported.')


  model, loader = load_switch(args.model)
  data, decoder_data, valmeta = get_c4(
    args.model, args.trainsamples, args.valsamples, detaileval=args.detaileval
  )
  if args.nearest or args.wbits == 16:
    data = data[args.trainsamples:]
    decoder_data = decoder_data[args.trainsamples:]
    args.trainsamples = 0

  dev = torch.device('cuda:0')

  # This was only used for verification that there is no validation data leakage.
  if args.separate_eval:
    switch_forward(
      model, loader, data[:args.trainsamples], decoder_data[:args.trainsamples], valmeta, args.trainsamples, dev
    )
    args.wbits = 16
    switch_forward(
      model, loader, data[args.trainsamples:], decoder_data[args.trainsamples:], valmeta, 0, dev
    )
    exit()

  tick = time.time()
  switch_forward(model, loader, data, decoder_data, args.trainsamples, valmeta, dev)
  print('Time:', time.time() - tick)
