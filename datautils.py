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


import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer


def set_seed(seed):
  np.random.seed(seed)
  torch.random.manual_seed(seed)

# T5 span corruption implementation from:
# https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py

def random_spans_noise_mask(length, noise_density=.15, mean_noise_span_length=3.):
  orig_length = length

  num_noise_tokens = int(np.round(length * noise_density))
  # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
  num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
  num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

  # avoid degeneracy by ensuring positive number of noise spans
  num_noise_spans = max(num_noise_spans, 1)
  num_nonnoise_tokens = length - num_noise_tokens

  # pick the lengths of the noise spans and the non-noise spans
  def _random_segmentation(num_items, num_segments):
    mask_indices = np.arange(num_items - 1) < (num_segments - 1)
    np.random.shuffle(mask_indices)
    first_in_segment = np.pad(mask_indices, [[1, 0]])
    segment_id = np.cumsum(first_in_segment)
    # count length of sub segments assuming that list is sorted
    _, segment_length = np.unique(segment_id, return_counts=True)
    return segment_length

  noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
  nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

  interleaved_span_lengths = np.reshape(
    np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
  )
  span_starts = np.cumsum(interleaved_span_lengths)[:-1]
  span_start_indicator = np.zeros((length,), dtype=np.int8)
  span_start_indicator[span_starts] = True
  span_num = np.cumsum(span_start_indicator)
  is_noise = np.equal(span_num % 2, 1)

  return is_noise[:orig_length]

def get_c4(model, trainsamples=128, valsamples=128, detaileval=False):
  if 'xxl' in model:
    model = 'google/switch-base-128' # xxl tokenizer is broken
  tokenizer = AutoTokenizer.from_pretrained(model)

  traindata = load_dataset(
    'allenai/c4', 'allenai--c4', data_files={'train': ['en/c4-train.00000-of-01024.json.gz', 'en/c4-train.00001-of-01024.json.gz']}, split='train'
  )

  mask_tokens = [
    tokenizer('<extra_id_%d>' % i).input_ids[0] for i in range(100)
  ]

  def mlm(input_ids):
    mask = ~random_spans_noise_mask(len(input_ids))

    encoder = []
    decoder = [0] # must start with pad token

    i = 0
    segments = 0

    # Build encoder and decoder data for masked-language-modelling
    while i < len(input_ids):
      if mask[i]:
        encoder.append(input_ids[i])
        i += 1
      else:
        encoder.append(mask_tokens[segments])
        decoder.append(mask_tokens[segments])
        segments += 1
        while i < len(input_ids) and mask[i] == 0:
          decoder.append(input_ids[i])
          i += 1
    encoder.append(1) # must end with eos token
    decoder.append(mask_tokens[segments]) # must end with another mask token

    return encoder, decoder

  inputlen = 568 # corresponds to 512 after masking

  if detaileval:
    MAX = 128
    valdata = load_dataset('togethercomputer/RedPajama-Data-1T-Sample', split='train')
    valdata1 = []
    meta = valdata['meta']
    import random
    random.seed(0)
    counts = []
    for key in ['arxiv', "'source': 'github'", "'source': 'stackexchange'", 'https://en.wikipedia.org/wiki']:
      counts.append(0)
      for i in [i for i in range(len(valdata)) if key in meta[i]]:
        tokens = tokenizer(valdata[i]['text']).input_ids
        if not (1 < len(tokens) <= inputlen):
          continue
        valdata1.append(tokens)
        counts[-1] += 1
        if counts[-1] == MAX:
          break
    print(counts)
    valdata = valdata1
    tasks = ['arxiv', 'github', 'stack', 'wiki']
    valmeta = [(t, sum(counts[:(i + 1)])) for i, t in enumerate(tasks)]
    valsamples = len(valdata)
  else:
    valdata = load_dataset(
      'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    valmeta = [('c4', valsamples)]

  encoder = []
  decoder = []

  for data, nsamples in [
    (traindata, trainsamples), (valdata, trainsamples + valsamples)
  ]:
    set_seed(0)
    for i, sample in enumerate(data):
      if len(encoder) == nsamples:
        break
      if not detaileval or len(encoder) < trainsamples:
        sample = tokenizer(sample['text']).input_ids
      # For some reason larger models seem to have problems with clipped samples.
      # Either they were not used for training or require specific undocumented preprocessing.
      # Hence, we only consider samples that fit fully in the model context here.
      if len(sample) > inputlen:
        continue 
      enc, dec = mlm(sample)
      encoder.append(torch.LongTensor(enc).unsqueeze(0))
      decoder.append(torch.LongTensor(dec).unsqueeze(0))

  return encoder, decoder, valmeta
