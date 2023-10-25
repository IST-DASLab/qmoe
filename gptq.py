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


import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


# Ensure that there are no automatic TF32 operations which can mess with numerics
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def hessian(inp, baseline=False):
  nsamples = inp.shape[0]
  if nsamples == 0 or baseline:
    # Simulate RTN by returning and identity Hessian
    return torch.eye(inp.shape[-1], device=inp.device)
  inp = inp.float()
  inp = inp.reshape((-1, inp.shape[-1]))
  H = inp.t().matmul(inp)
  H /= 2 / nsamples
  return H

# Adapted from https://github.com/IST-DASLab/gptq

def batch_gptq(
  W, H, quantizer, blocksize=128, percdamp=.1, groupsize=-1, actorder=False
):
  dtype = W.dtype
  W = W.clone()
  W = W.float()

  rows, columns = W.shape[1:]
  dev = W.device

  quantizer.find_params(W)

  Losses = torch.zeros_like(W)
  Q = torch.zeros_like(W)

  diag = torch.arange(columns, device=dev)
  damp = percdamp * torch.mean(H[:, diag, diag], axis=-1, keepdim=True)
  damp = torch.maximum(damp, 1e-6 * torch.ones_like(damp)) # catch all zeros
  H[:, diag, diag] += damp

  if actorder:
    perm = torch.argsort(H[:, diag, diag], dim=1, descending=True)
    for i in range(W.shape[0]):
      W[i] = W[i, :, perm[i]]
      H[i] = H[i][perm[i]][:, perm[i]]
    invperm = torch.argsort(perm, dim=1)

  err = True
  while err:
    # We need to loop as batch operations only return the first error
    try:
      H1 = torch.linalg.cholesky(H)
      H1 = torch.cholesky_inverse(H1)
      H1 = torch.linalg.cholesky(H1, upper=True)
      H = H1
      err = False
    except RuntimeError as ex:
      print('Skip due to singularity.')
      idx = int(str(ex).replace('linalg.cholesky: (Batch element ', '').split('):')[0])
      # Do RTN for failed Hessians by turning them into identity
      H[idx] = torch.eye(columns, device=dev)
  Hinv = H

  for i1 in range(0, columns, blocksize):
    i2 = min(i1 + blocksize, columns)
    count = i2 - i1

    W1 = W[:, :, i1:i2].clone()
    Q1 = torch.zeros_like(W1)
    Err1 = torch.zeros_like(W1)
    Losses1 = torch.zeros_like(W1)
    Hinv1 = Hinv[:, i1:i2, i1:i2]

    for i in range(count):
      w = W1[:, :, i]
      d = Hinv1[:, i, i].unsqueeze(1)

      if groupsize != -1:
        if (i1 + i) % groupsize == 0:
          quantizer.find_params(W[:, :, (i1 + i):(i1 + i + groupsize)])

      q = quantize(
        w.unsqueeze(2), quantizer.scale, quantizer.zero, quantizer.maxq
      ).flatten(1)
      Q1[:, :, i] = q
      Losses1[:, :, i] = (w - q) ** 2 / d ** 2
      err1 = (w - q) / d
      W1[:, :, i:] -= torch.bmm(err1.unsqueeze(2), Hinv1[:, i, i:].unsqueeze(1))
      Err1[:, :, i] = err1

    Q[:, :, i1:i2] = Q1
    Losses[:, :, i1:i2] = Losses1 / 2

    W[:, :, i2:] -= torch.bmm(Err1, Hinv[:, i1:i2, i2:])

  torch.cuda.synchronize(device=dev)
  print('error', torch.sum(Losses.flatten(1), 1))
  print('Sparsity:', torch.mean((Q == 0).float()))

  if actorder:
    for i in range(W.shape[0]):
      Q[i] = Q[i, :, invperm[i]]

  return Q.to(dtype)


if __name__ == '__main__':
  import time

  D = 2048
  K = 8

  torch.random.manual_seed(0)
  X = torch.randn(128, 512, D).cuda()
  W = torch.randn(K, 768, D).cuda()
  quantizer = Quantizer()
  quantizer.configure(2)
  
  H = hessian(X).repeat(K, 1, 1)
  Q = batch_gptq(W, H, quantizer)
  tick = time.time()
  COUNT = 10
  for i in range(COUNT):
    H = hessian(X).repeat(K, 1, 1)
    Q = batch_gptq(W, H, quantizer)
    torch.cuda.synchronize()
  print((time.time() - tick) / COUNT)

  print(Q[0])

