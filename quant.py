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


# Adapted from https://github.com/IST-DASLab/gptq/blob/main/quant.py


import numpy as np
import torch
import torch.nn as nn


def quantize(x, scale, zero, maxq):
  if maxq < 0:
    return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
  q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
  return scale * (q - zero)

class Quantizer(nn.Module):

  def configure(
    self, bits, sym=False
  ):
    if bits == 1.5:
      self.maxq = torch.tensor(-1) # use -1 to identify ternary
    else:
      self.maxq = torch.tensor(2 ** int(bits) - 1)
    self.sym = sym

  def find_params(self, x):
    dev = x.device
    self.maxq = self.maxq.to(dev)

    tmp = torch.zeros(x.shape[:-1], device=dev)
    xmin = torch.minimum(x.min(-1)[0], tmp)
    xmax = torch.maximum(x.max(-1)[0], tmp)

    if self.sym:
      xmax = torch.maximum(torch.abs(xmin), xmax)
      tmp = xmin < 0
      if torch.any(tmp):
        xmin[tmp] = -xmax[tmp]
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1

    if self.maxq < 0:
      # For ternary, repurpose `scale` as max and `zero` as min to avoid interface changes
      self.scale = xmax
      self.zero = xmin
    else:
      self.scale = (xmax - xmin) / self.maxq
      if self.sym:
        self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
      else:
        self.zero = torch.round(-xmin / self.scale)

    self.scale = self.scale.unsqueeze(-1)
    self.zero = self.zero.unsqueeze(-1)

  def quantize(self, x):
    if self.ready():
      return quantize(x, self.scale, self.zero, self.maxq)
    return x

