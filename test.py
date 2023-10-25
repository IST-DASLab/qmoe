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


# Test file for individual layer benchmarks and theoretical compression rates.


import argparse
import heapq
import numpy as np
import random
import time
import torch

import sub1_cuda


COUNT = 2 ** 16
ZEROS = .885

probs = [ZEROS] + [(1 - ZEROS) / 2] * 2

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

pq = [(-1., [])]
while len(res) < COUNT:
    top = heapq.heappop(pq)
    if top[0] != -1.:
        res.append(top)
    if len(top[1]) == 14:
        continue
    for i in range(len(probs)):
        heapq.heappush(pq, (top[0] * probs[i], top[1] + [i]))

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

def trie_add(trie, seq, idx, i=0):
    if i == len(seq):
        trie[-1] = idx
    else:
        trie[seq[i]] = trie_add(trie.get(seq[i], {}), seq, idx, i + 1)
    return trie

trie = {}
for i, r in enumerate(res):
    trie = trie_add(trie, r[1], i)

def genseq(count):
    seq = random.choices(list(range(len(probs))), weights=probs, k=count // 2)
    if count % 2 != 0:
        seq.append(random.randint(0, 3 ** (count % 2) - 1))
    return seq

def greedy(seq):
    res = []

    i = 0
    curtrie = trie
    while i < len(seq):
        if seq[i] not in curtrie:
            res.append(curtrie[-1])
            curtrie = trie
        curtrie = curtrie[seq[i]]
        i += 1
    if -1 in curtrie:
        res.append(curtrie[-1])

    return res


def decompress(seq, width):
    dec = []
    row = 0
    for idx in seq:
        for tern in res[idx][1]:
            for _ in range(2):
                w = (tern % 3)
                dec.append([0, termin[row], termax[row]][w])
                if len(dec) % D2 == 0:
                    row += 1
                    break
                tern //= 3
    return np.array(dec, dtype=np.float32)

def benchmark(f, warmup=100, iter=1000):
    for _ in range(warmup):
        f()
        torch.cuda.synchronize()

    tick = time.time()
    for _ in range(iter):
        f()
        torch.cuda.synchronize()
    return (time.time() - tick) / iter


parser = argparse.ArgumentParser()
parser.add_argument(
  '--benchmark', action='store_true',
  help='Whether to run benchmarking.'
)
args = parser.parse_args()


dec = torch.from_numpy(dec.astype(np.int32)).cuda()

for D1, D2 in [
    (768, 3072), (3072, 768),
    (1024, 4096), (4096, 1024),
    (2080, 6144), (6144, 2080)
]:
    print((D1, D2))

    mat = []
    row_off = [0]
    for _ in range(D1):
        row = greedy(genseq(D2))
        mat.extend(row)
        row_off.append(len(mat))

    w_comp = np.array(mat, dtype=np.uint16)
    row_off = np.array(row_off, dtype=np.int32)

    termin = -np.random.uniform(size=D1).astype(np.float32)
    termax = +np.random.uniform(size=D1).astype(np.float32)
    terminmax = np.column_stack((termin, termax)).reshape(-1)
    x = np.random.uniform(size=(D2, 1)).astype(np.float32)
    y = np.zeros((D1, 1), dtype=np.float32)
    w = decompress(w_comp, D2).reshape((D1, D2))

    w_comp = torch.from_numpy(w_comp.astype(np.int16)).cuda()
    row_off = torch.from_numpy(row_off).cuda()
    ter_minmax = torch.from_numpy(terminmax).cuda()
    ter_minmax = ter_minmax.bfloat16()
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    w = torch.from_numpy(w).cuda()
    x = x.bfloat16()
    y = y.bfloat16()
    w = w.bfloat16()

    if args.benchmark:
        print(' Sub1:', benchmark(lambda: sub1_cuda.sub1matvec(dec, w_comp, row_off, ter_minmax, x, y)))
        print('Dense:', benchmark(lambda: torch.matmul(w, x, out=y)))
    else:
        print((16 * D1 * D2) / (16 * row_off[-1] + 64 * len(row_off)))

        print(torch.matmul(w, x))

        sub1_cuda.sub1matvec(dec, w_comp, row_off, ter_minmax, x, y)
        print(y)

        import sub1
        w = w.to(sub1.DEV)
        x = x.to(sub1.DEV)
        linear = sub1.Sub1Linear.make(w)
        print(linear(x.reshape((1, -1))))
