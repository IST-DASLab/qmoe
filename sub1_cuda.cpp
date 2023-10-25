// Copyright (C) QMoE.2023 Elias Frantar (elias.frantar@ist.ac.at)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void sub1matvec_cuda(
  torch::Tensor dec,
  torch::Tensor w_comp,
  torch::Tensor row_off,
  torch::Tensor ter_minmax,
  torch::Tensor x,
  torch::Tensor y
);

torch::Tensor sub1pack_cuda(
  torch::Tensor trie,
  torch::Tensor w_tern,
  torch::Tensor row_off
);

void sub1matvec(
  torch::Tensor dec,
  torch::Tensor w_comp,
  torch::Tensor row_off,
  torch::Tensor ter_minmax,
  torch::Tensor x,
  torch::Tensor y
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(dec));
  sub1matvec_cuda(dec, w_comp, row_off, ter_minmax, x, y);
}

torch::Tensor sub1pack(
  torch::Tensor trie,
  torch::Tensor w_tern,
  torch::Tensor row_off
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(trie));
  return sub1pack_cuda(trie, w_tern, row_off);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sub1matvec", &sub1matvec, "Sub1 bit compressed matvec (CUDA)");
  m.def("sub1pack", &sub1pack, "Sub1 bit compressed packing (CUDA)");
}
