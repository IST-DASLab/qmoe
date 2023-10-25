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


import argparse
import copy
import heapq
import numpy as np
import os
import time
import torch
import torch.nn as nn
import transformers

import sub1_cuda


def setup():
  COUNT = 2 ** 16 # dictionary size
  ZEROS = .885 # probability of sampling 0

  probs = [ZEROS] + [(1 - ZEROS) / 2] * 2

  # Generate pairs and corresponding probabilities
  probs1 = []
  def gen(prob, count):
    if count == 0:
      probs1.append(prob)
    else:
      for i in range(len(probs)):
        gen(prob * probs[i], count - 1)
  gen(1., 2)
  probs = probs1

  res = []

  # Generate maximum probability sequences for dictionary
  pq = [(-1., [])]
  while len(res) < COUNT:
    top = heapq.heappop(pq)
    if top[0] != -1.:
      res.append(top)
    if len(top[1]) == 14:
      continue
    for i in range(len(probs)):
      heapq.heappush(pq, (top[0] * probs[i], top[1] + [i]))

  # Encode dictionary data in QMoE table format
  dec = np.zeros(2 * COUNT, dtype=np.uint32)
  for i in range(COUNT):
    for j, r in enumerate(res[i][1][:7]):
      dec[2 * i + 0] |= (r  % 3) << (4 * j + 0)
      dec[2 * i + 0] |= (r // 3) << (4 * j + 2)
    dec[2 * i + 0] <<= 4
    dec[2 * i + 0] |= len(res[i][1])
    for j, r in enumerate(res[i][1][7:]):
      dec[2 * i + 1] |= (r  % 3) << (4 * j + 0)
      dec[2 * i + 1] |= (r // 3) << (4 * j + 2)
    dec[2 * i + 1] <<= 4
    dec[2 * i + 1] |= len(res[i][1])
  dec = dec.astype(np.int32)

  def trie_add(trie, seq, idx, i=0):
    if i == len(seq):
      trie[-1] = idx
    else:
      trie[seq[i]] = trie_add(trie.get(seq[i], {}), seq, idx, i + 1)
    return trie

  # Build dictionary trie for encoding
  trie = {}
  for i, r in enumerate(res):
    trie = trie_add(trie, r[1], i)

  trie_arr = [[-1] * len(probs) for _ in range(COUNT + 1)]

  def make_trie_arr(trie):
    idx = trie.get(-1, COUNT)
    for i in trie:
      if i != -1:
        trie_arr[idx][i] = trie[i][-1]
        make_trie_arr(trie[i])

  # Turn trie into an array to use in the encoding CUDA kernel
  make_trie_arr(trie)
  trie_arr = np.array(trie_arr, dtype=np.int32)

  dec = torch.from_numpy(dec)
  trie_arr = torch.from_numpy(trie_arr)

  return dec, trie_arr

dec, trie = setup()


GPUS = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
# Default GPU; last in list as it will receive the least amount of layers due to rounding up
DEV = GPUS[-1]
DEC = [dec.to(gpu) for gpu in GPUS]
TRIE = trie.to(DEV)


class Sub1Linear(nn.Module):

  def __init__(self, height, comp_size):
    super().__init__()
    self.register_buffer(
      'w_comp', torch.zeros(comp_size, dtype=torch.int16)
    )
    self.register_buffer(
      'row_off', torch.zeros(height + 1, dtype=torch.int32)
    )
    self.register_buffer(
      'ter_minmax', torch.zeros(2 * height, dtype=torch.bfloat16)
    )
    # HuggingFace Switch does not run without a `weight`
    self.register_buffer(
      'weight', torch.zeros(0, dtype=torch.bfloat16)
    )

  def forward(self, x):
    y = torch.zeros(
      (x.shape[0], self.row_off.shape[0] - 1), dtype=torch.bfloat16, device=x.device
    )
    dec = DEC[self.w_comp.device.index]
    # Currently we naively run individual matrix-vector products also for larger tokens counts.
    # This could be easily improved.
    for i in range(y.shape[0]):
      sub1_cuda.sub1matvec(
        dec, self.w_comp, self.row_off, self.ter_minmax, x[i], y[i]
      )
    return y

  @staticmethod
  def make(w):
    w = w.bfloat16()

    ter_minmax = torch.column_stack((w.min(1)[0], w.max(1)[0])).flatten()
    row_off = torch.zeros(
      w.shape[0] + 1, dtype=torch.int32, device=w.device
    )
    w_tern = torch.zeros(
      w.shape, dtype=torch.int32, device=w.device
    )
    # We load min-max as a `half2` in the kernel
    w_tern[w == ter_minmax[0::2].reshape((-1, 1))] = 1
    w_tern[w == ter_minmax[1::2].reshape((-1, 1))] = 2
    w_comp = sub1_cuda.sub1pack(TRIE, w_tern, row_off)

    linear = Sub1Linear(w.shape[0], w_comp.shape[0])
    linear.w_comp.data = w_comp
    linear.row_off.data = row_off
    linear.ter_minmax = ter_minmax

    return linear


# Handles building and saving actual compressed checkpoints.
class Sub1CheckpointManager:

  def __init__(self, path, model, loader, lazys):
    loader.load(model, dev='cpu')
    self.sd_noexp = model.state_dict() 

    for name, layer in lazys.items():
      if 'expert' in name:
        continue
      layer.load('cpu')
      for n, p in layer.linear.state_dict().items():
        # BFLOAT16 casting is mostly for base-128 which is for some reason stored in FP32 by HuggingFace
        self.sd_noexp[name + '.' + n] = p.bfloat16()
      layer.linear = None

    os.makedirs(path, exist_ok=True)
    config = copy.deepcopy(model.config)
    config.torch_dtype = 'bfloat16' # also for base-128 
    config.save_pretrained(path)
    torch.save(self.sd_noexp, os.path.join(path, 'noexp.pt'))
    self.path = path

    self.sd_exp = [{}]
    self.sizes = {}

  # Add new packed expert to memory buffer.
  def add_expert(self, name, expert):
    wi = Sub1Linear.make(expert.wi.linear.weight)
    for n, p in wi.state_dict().items():
      self.sd_exp[-1][name + '.wi.' + n] = p.cpu()
    self.sizes[name + '.wi'] = wi.w_comp.numel()
    wo = Sub1Linear.make(expert.wo.linear.weight)
    for n, p in wo.state_dict().items():
      self.sd_exp[-1][name + '.wo.' + n] = p.cpu()
    self.sizes[name + '.wo'] = wo.w_comp.numel()

  # Write out all packed experts to disk and clear buffer.
  def save_experts(self):
    torch.save(self.sd_exp[-1], os.path.join(self.path, 'exp%02d.pt' % (len(self.sd_exp) - 1)))
    self.sd_exp[-1] = None
    self.sd_exp.append({})

  # Write out final metadata.
  def finalize(self):
    torch.save(self.sizes, os.path.join(self.path, 'sizes.pt'))


def load_sub1(path, simul=False):
  # We will load linear layers manually, hence we will temporarily overwrite them
  # with an empty placeholder to avoid problematic unnecessary memory allocations.

  class LinearShell(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
      super().__init__()
      self._in_features = in_features
      self._out_features = out_features
      self._bias = bias
      self._device = device
      self._dtype = dtype
      # HF SwitchTransformer fails if there is no weight attribute
      self.weight = None

  config = transformers.SwitchTransformersConfig.from_pretrained(path)
  default_dtype = torch.get_default_dtype()
  torch.set_default_dtype(config.torch_dtype)

  linear = nn.Linear
  setattr(nn, 'Linear', LinearShell)
  with transformers.modeling_utils.no_init_weights():
    if hasattr(config, 'is_full_sparse'): # for c-2048
      import_path = transformers.models.switch_transformers.modeling_switch_transformers
      class AlwaysSparse(import_path.SwitchTransformersBlock):
        def __init__(self, config, has_relative_attention_bias=False, is_sparse=False):
          super().__init__(
            config, has_relative_attention_bias=has_relative_attention_bias, is_sparse=True
          )
      setattr(import_path, 'SwitchTransformersBlock', AlwaysSparse)
    model = transformers.SwitchTransformersForConditionalGeneration(config)
  setattr(nn, 'Linear', linear)

  sizes = torch.load(os.path.join(path, 'sizes.pt'))
  singletons = {}

  # Replace linear placeholders with real layers for non-experts and compressed layers for experts
  def init_model(module, name=''):
    for attr in dir(module):
      tmp = getattr(module, attr)
      if isinstance(tmp, LinearShell):
        if 'expert' not in name:
          setattr(
            module, attr, nn.Linear(
              tmp._in_features, tmp._out_features, tmp._bias, tmp._device, tmp._dtype
            )
          )
        else:
          l = '.'.join(name.split('.')[:3])
          inf = config.d_model if attr == 'wi' else config.d_ff
          ouf = config.d_ff if attr == 'wi' else config.d_model
          if simul:
          # Make all experts in a layer point to the same singleton to simulate idealized
          # standard precision execution of extremely massive models.
            if (l, inf, ouf) not in singletons:
              singletons[(l, inf, ouf)] = nn.Linear(inf, ouf)
              singletons[(l, inf, ouf)].weight.data[:] = 0
            setattr(module, attr, singletons[(l, inf, ouf)])
          else:
            setattr(module, attr, Sub1Linear(ouf, sizes[name + '.' + attr]))
    for name1, child in module.named_children():
      init_model(child, name + '.' + name1 if name != '' else name1)

  with transformers.modeling_utils.no_init_weights():
    init_model(model)
  
  # Manually loaded saved weight data into initialized layers 
  sd = model.state_dict()
  for filename in os.listdir(path):
    if filename in ['sizes.pt', 'config.json', 'README.md', '.gitattributes', '.git']:
      continue
    if simul and filename != 'noexp.pt':
      continue
    print(filename)
    for k, v in torch.load(os.path.join(path, filename)).items():
      sd[k][:] = v

  torch.set_default_dtype(default_dtype)

  # Number of transformer blocks per GPU, round up
  per_gpu = (config.num_layers * 2 + len(GPUS) - 1) // len(GPUS)
  # MLP to fix major HF bottleneck and implement multi-GPU inference
  class FasterMLP(nn.Module):
    def __init__(self, mlp, gpu):
      super().__init__()
      mlp = mlp.to(gpu)
      self.router = mlp.router
      self.experts = nn.ModuleList(mlp.experts.values())
      self.gpu = gpu
    def forward(self, hidden_states):
      gpu = hidden_states.device
      # Move to this layer's GPU
      hidden_states = hidden_states.to(self.gpu)
      router_mask, router_probs, router_logits = self.router(hidden_states)
      expert_index = torch.argmax(router_mask, dim=-1)
      next_states = hidden_states.clone()
      # Only execute calls for experts to which at least one token is routed.
      # This is > 10x faster than the default HuggingFace implementation.
      for idx in torch.nonzero(torch.sum(router_mask, dim=(0, 1)), as_tuple=True)[0]:
        token_indices = router_mask[:, :, idx].bool()
        next_states[token_indices] = self.experts[idx](hidden_states[token_indices])
      hidden_states = router_probs * next_states
      # Move back to original input GPU
      hidden_states = hidden_states.to(gpu)
      return hidden_states, (router_logits, expert_index)
  # Store the small amount of non-expert layer data on the default GPU 
  for n, p in model.named_parameters():
    p.data = p.data.to(DEV)
  idx = 0
  for part, off in [(model.encoder, 1), (model.decoder, 2)]:
    for layer in part.block:
      if layer.is_sparse:
        gpu = GPUS[idx // per_gpu]
        layer.layer[off].mlp = FasterMLP(layer.layer[off].mlp, gpu)
        idx += 1

  model.eval()
  return model


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
    'checkpoint', type=str,
    help='Path to sub1 compressed model.'
  )
  parser.add_argument(
    '--gentokens', type=int, default=0,
    help='Number of tokens to time generating.'
  )
  parser.add_argument(
    '--valsamples', type=int, default=0
  )
  parser.add_argument(
    '--simul', action='store_true',
    help='Simulate BF16 model.'
  )
  parser.add_argument(
    '--detaileval', action='store_true',
    help='Whether to perform evaluation on additional datasets.'
  )

  args = parser.parse_args()

  from datautils import *
  model = load_sub1(args.checkpoint, simul=args.simul)

  if args.gentokens:
    data, decoder_data, _ = get_c4('google/switch-base-128', 0, 11)

    torch.cuda.synchronize()
    with torch.no_grad():
      times = []
      for i in range(len(data)):
        tick = time.time()
        y = model.generate(
          data[i].to(DEV),
          decoder_start_token_id=0,
          min_length=args.gentokens,
          max_length=args.gentokens,
          use_cache=True
        )
        # print(y)
        torch.cuda.synchronize()
        if i > 0: # warmup
          times.append(time.time() - tick)
          print(times[-1])
      print('Mean:', np.mean(times))

  if (args.valsamples or args.detaileval) and not args.simul:
    data, decoder_data, valmeta = get_c4(
      'google/switch-base-128', 0, args.valsamples, detaileval=args.detaileval
    )

    totsum = 0
    totlen = 0
    with torch.no_grad():
      for i in range(len(data)):
        res = model(
          input_ids=data[i].to(DEV),
          decoder_input_ids=decoder_data[i].to(DEV)
        )
        shift_logits = res.logits[:, :-1, :].contiguous()
        shift_labels = decoder_data[i].to(DEV)[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        totsum += loss.float() * res.logits.shape[1]
        totlen += res.logits.shape[1]
        if i + 1 == valmeta[0][1]:
          print(valmeta[0][0] + ':', totsum.item() / totlen)
          totsum = 0
          totlen = 0
          valmeta.pop(0)
