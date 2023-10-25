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
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <iostream>


#define CALL_IF(BLOCKHEIGHT, WARPS, HEIGHT, WIDTH) \
  else if (height == HEIGHT && width == WIDTH) { \
    dim3 blocks((height + BLOCKHEIGHT - 1) / BLOCKHEIGHT, 1); \
    dim3 threads(WARPS * 32); \
    Sub1MatVec<BLOCKHEIGHT, WARPS, HEIGHT, WIDTH><<<blocks, threads>>>( \
      (int*) dec.data_ptr(), \
      (ushort*) w_comp.data_ptr(), \
      (int*) row_off.data_ptr(), \
      (__nv_bfloat162*) ter_minmax.data_ptr(), \
      (__nv_bfloat16*) x.data_ptr(), \
      (__nv_bfloat16*) y.data_ptr() \
    ); \
  }


template <
  const int blockheight,
  const int threads,
  const int height,
  const int width
>
__global__ void Sub1MatVec(
  const            int* __restrict__ dec,
  const         ushort* __restrict__ w_comp,
  const            int* __restrict__ row_off,
  const __nv_bfloat162* __restrict__ ter_minmax,
  const  __nv_bfloat16* __restrict__ x,
         __nv_bfloat16* __restrict__ y
);

__global__ void Sub1PreparePack(
  const int* __restrict__ trie,
        int* __restrict__ w_tern,
        int* __restrict__ row_lens,
  int width
);

__global__ void Sub1Pack(
  const    int* __restrict__ w_prep,
  const    int* __restrict__ row_off,
        ushort* __restrict__ w_comp,
  int width
);


void sub1matvec_cuda(
  torch::Tensor dec,
  torch::Tensor w_comp,
  torch::Tensor row_off,
  torch::Tensor ter_minmax,
  torch::Tensor x,
  torch::Tensor y
) {
  int height = y.numel();
  int width = x.numel();

  if (false) {}
  CALL_IF(10, 10, 768, 3072)
  CALL_IF(38, 32, 3072, 768)
  CALL_IF(13, 13, 1024, 4096)
  CALL_IF(50, 32, 4096, 1024)
  CALL_IF(26, 26, 2080, 6144)
  CALL_IF(75, 32, 6144, 2080)
}

#define PACK_BLOCKHEIGHT 32

torch::Tensor sub1pack_cuda(
  torch::Tensor trie,
  torch::Tensor w_tern,
  torch::Tensor row_off
) {
  int height = w_tern.size(0);
  int width = w_tern.size(1);

  dim3 blocks(height / PACK_BLOCKHEIGHT, 1);
  dim3 threads(PACK_BLOCKHEIGHT);

  row_off.index_put_({torch::indexing::None}, 0);

  Sub1PreparePack<<<blocks, threads>>>(
    (int*) trie.data_ptr(),
    (int*) w_tern.data_ptr(),
    (int*) row_off.data_ptr(),
    width
  );

  row_off.index_put_({torch::indexing::None}, row_off.cumsum(0));

  torch::Tensor w_comp = torch::zeros(
    row_off[height].item<int>(),
    torch::TensorOptions().dtype(torch::kInt16).device(row_off.device())
  );
  Sub1Pack<<<blocks, threads>>>(
    (int*) w_tern.data_ptr(),
    (int*) row_off.data_ptr(),
    (ushort*) w_comp.data_ptr(),
    width
  );

  return w_comp;
}


template <
  const int blockheight,
  const int warps,
  const int height,
  const int width
>
__global__ void Sub1MatVec(
  const            int* __restrict__ dec,
  const         ushort* __restrict__ w_comp,
  const            int* __restrict__ row_off,
  const __nv_bfloat162* __restrict__ ter_minmax,
  const  __nv_bfloat16* __restrict__ x,
         __nv_bfloat16* __restrict__ y
) {
  int thread = threadIdx.x;
  int warp = thread / 32;
  int thread_in_warp = thread % 32;

  __shared__ float x_shared[width + 28];
  for (int i = thread; i < width; i += 32 * warps)
    x_shared[i] = __bfloat162float(x[i]);
  if (thread < 28)
    x_shared[width + thread] = 0;

  int num = thread_in_warp / 14;
  int dig = thread_in_warp % 14;

  __shared__ float deq[3][warps * 32];
  deq[0][thread] = 0;

  // Needs to be int to avoid bank conflicts on writing
  __shared__ int w_comp_block[warps][32];

  int startrow = blockheight * blockIdx.x;
  for (int row = startrow + warp; row < startrow + blockheight; row += warps) {
    if (row >= height) {
      __syncthreads();
      continue;
    }
    int off = row_off[row];
    int len = row_off[row + 1] - off;
    deq[1][thread] = __bfloat162float(ter_minmax[row].x);
    deq[2][thread] = __bfloat162float(ter_minmax[row].y);
    __syncthreads();

    float res = 0;
    int idx = 0;

    for (int i = 0; i < len; i += 32) {
      if (i + thread_in_warp < len)
        w_comp_block[warp][thread_in_warp] = w_comp[off + i + thread_in_warp];

      int filled = 32;
      if (len - i < 32)
        filled = len - i;

      if (thread_in_warp < 28) {
        for (int j = 0; j < filled; j++) {
          int enc = w_comp_block[warp][j];
          int wx14 = dec[2 * enc + num];
          int ter = (wx14 >> (4 + 2 * dig)) & 0x3;
          float w = deq[ter][thread];
          res += w * x_shared[idx + thread_in_warp];
          idx += 2 * (wx14 & 0xf);
        }
      }
    }

    for (int i = 16; i > 0; i /= 2)
      res += __shfl_down_sync(0xffffffff, res, i);
    if (thread_in_warp == 0)
      y[row] += __float2bfloat16(res);
  }
}


// The efficiency of packing is not so important, hence we can keep the corresponding kernels
// very simple.


#define TRIE_ROOT 65536

__global__ void Sub1PreparePack(
  const int* __restrict__ trie,
        int* __restrict__ w_tern,
        int* __restrict__ row_off,
  int width
) {
  int row = PACK_BLOCKHEIGHT * blockIdx.x + threadIdx.x;
  int off = width * row;

  int node = TRIE_ROOT;
  int res = off;

  for (int i = 0; i < width; i += 2) {
    int num = 0;
    for (int j = 1; j >= 0; j--) {
      if (i + j < width)
        num = 3 * num + w_tern[off + i + j];
    }

    if (trie[9 * node + num] == -1) {
      w_tern[res++] = node;
      node = TRIE_ROOT;
    }
    node = trie[9 * node + num];
  }
  w_tern[res++] = node;

  row_off[row + 1] = res - off;
}

__global__ void Sub1Pack(
  const    int* __restrict__ w_prep,
  const    int* __restrict__ row_off,
        ushort* __restrict__ w_comp,
  int width
) {
  int row = PACK_BLOCKHEIGHT * blockIdx.x + threadIdx.x;
  for (int i = 0; i < row_off[row + 1] - row_off[row]; i++)
    w_comp[row_off[row] + i] = ushort(w_prep[width * row + i]);
}
