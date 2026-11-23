# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A two-stage custom all-reduce with vLLM's one-stage bits, for the batch-invariant sampler.

vLLM's batch-invariant mode runs every tensor-parallel all-reduce on the custom all-reduce's one-stage kernel
(``VLLM_CUSTOM_ALLREDUCE_ALGO=1stage``): each rank reads every rank's whole input and sums each element in rank order
0, 1, ... in fp32, then rounds once. Its two-stage kernel moves an eighth of the data at 8 ranks but starts each
rank's reduce-scatter at its own rank (``(rank + i) % ngpus``), so an element's order depends on which rank's chunk
it falls in, and so on the message size. This is vLLM's two-stage kernel with the reduce-scatter reading the ranks
in order 0, 1, ...: every element gets the one-stage arithmetic.

It runs on the custom all-reduce's own IPC buffers and signals: the input is copied into the rank's pre-registered
staging buffer, the kernel reads every rank's staging buffer and keeps its partial in the signal buffer's scratch
area, with vLLM's barriers on the same flags, so it interleaves with vLLM's own kernels. Its arguments are those
fixed buffers, so it also runs inside CUDA graphs. Compiled on first use (``torch.utils.cpp_extension``).
"""

from __future__ import annotations

import os
from typing import Any

import torch

# Chunks of at least this many bytes take the two-stage kernel (faster from 512 KiB at 8 H200s: 1.1x, 2.5x at 8 MiB); smaller ones keep
# the one-stage kernel, which gives the same bits.
MIN_BYTES_ENV = "ARCTIC_GLM5_FIXED_AR_MIN_BYTES"
DEFAULT_MIN_BYTES = 512 * 1024

_SOURCE = r"""
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

namespace arctic_far {
constexpr int kMaxRanks = 16;
constexpr int kMaxBlocks = 36;
using FlagType = uint32_t;

// vLLM 0.32's custom_collective_common.cuh layout: these structs alias the buffers vLLM allocated.
struct Signal {
  alignas(128) FlagType start[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType end[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType _flag[kMaxBlocks];
};
struct __align__(16) RankData { const void* ptrs[kMaxRanks]; };
struct __align__(16) RankSignals { Signal* signals[kMaxRanks]; };

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t { T data[sz]; using type = T; static constexpr int size = sz; };
template <typename T>
struct packed_t { using P = array_t<T, 16 / sizeof(T)>; using A = array_t<float, 16 / sizeof(T)>; };

#define DINLINE __device__ __forceinline__
DINLINE float upcast_s(nv_bfloat16 v) { return __bfloat162float(v); }
DINLINE float upcast_s(float v) { return v; }
template <typename T> DINLINE T downcast_s(float v);
template <> DINLINE nv_bfloat16 downcast_s(float v) { return __float2bfloat16(v); }
template <> DINLINE float downcast_s(float v) { return v; }

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  array_t<float, N> out;
#pragma unroll
  for (int i = 0; i < N; i++) out.data[i] = upcast_s(val.data[i]);
  return out;
}
template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  O out;
#pragma unroll
  for (int i = 0; i < O::size; i++) out.data[i] = downcast_s<typename O::type>(val.data[i]);
  return out;
}

static DINLINE void st_flag_release(FlagType* a, FlagType f) {
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(f), "l"(a));
}
static DINLINE FlagType ld_flag_acquire(FlagType* a) {
  FlagType f;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(f) : "l"(a));
  return f;
}
static DINLINE void st_flag_volatile(FlagType* a, FlagType f) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(f), "l"(a));
}
static DINLINE FlagType ld_flag_volatile(FlagType* a) {
  FlagType f;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(f) : "l"(a));
  return f;
}

template <int ngpus>
DINLINE void barrier_at_start(const RankSignals& sg, Signal* self_sg, int rank) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    st_flag_volatile(&sg.signals[threadIdx.x]->start[blockIdx.x][rank], flag);
    while (ld_flag_volatile(&self_sg->start[blockIdx.x][threadIdx.x]) != flag);
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int ngpus>
DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    st_flag_release(&sg.signals[threadIdx.x]->end[blockIdx.x][rank], flag);
    while (ld_flag_acquire(&self_sg->end[blockIdx.x][threadIdx.x]) != flag);
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// vLLM's packed_reduce: fp32 sum over ptrs[0], ptrs[1], ... in that order, one rounding.
template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    A v = upcast(ptrs[i][idx]);
#pragma unroll
    for (int j = 0; j < A::size; j++) tmp.data[j] += v.data[j];
  }
  return downcast<P>(tmp);
}

// vLLM's cross_device_reduce_2stage, with the reduce-scatter reading the ranks in order 0, 1, ... (vLLM rotates
// them to start at the own rank): the one-stage kernel's order for every element, whatever the size.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    fixed_order_2stage(RankData dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    ptrs[i] = (const P*)dp.ptrs[i];
    int target = (rank + i) % ngpus;
    tmps[i] = (P*)(sg.signals[target] + 1);
  }
  auto tmp_out = tmps[0];
  barrier_at_start<ngpus>(sg, self_sg, rank);
  for (int idx = start + tid; idx < end; idx += stride) tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  barrier_at_end<ngpus>(sg, self_sg, rank);
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = (rank + i) % ngpus;
      if (gather_from_rank == ngpus - 1 || idx < part) {
        ((P*)result)[gather_from_rank * part + idx] = tmps[i][idx];
      }
    }
  }
}
}  // namespace arctic_far

void fixed_order_allreduce(torch::Tensor inp, torch::Tensor out, std::vector<int64_t> buffers,
                           std::vector<int64_t> signals, int64_t rank, int64_t buffer_bytes) {
  using namespace arctic_far;
  int world = (int)buffers.size();
  TORCH_CHECK(world == (int)signals.size() && world <= kMaxRanks, "bad rank count");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(), "contiguous tensors only");
  auto bytes = inp.numel() * inp.element_size();
  TORCH_CHECK(bytes <= buffer_bytes, "input larger than the staging buffer");
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  RankData dp;
  RankSignals sg;
  for (int i = 0; i < world; i++) {
    dp.ptrs[i] = reinterpret_cast<const void*>(buffers[i]);
    sg.signals[i] = reinterpret_cast<Signal*>(signals[i]);
  }
  Signal* self_sg = sg.signals[rank];
  cudaMemcpyAsync(const_cast<void*>(dp.ptrs[rank]), inp.data_ptr(), bytes, cudaMemcpyDeviceToDevice, stream);
  int threads = 512;
  auto launch = [&](auto dummy) {
    using T = decltype(dummy);
    int d = packed_t<T>::P::size;
    int size = (int)(inp.numel() / d);
    TORCH_CHECK(inp.numel() % d == 0, "size must be a multiple of 16 bytes");
    int blocks = std::min(kMaxBlocks, (size + threads - 1) / threads);
    T* result = reinterpret_cast<T*>(out.data_ptr());
    switch (world) {
      case 2: fixed_order_2stage<T, 2><<<blocks, threads, 0, stream>>>(dp, sg, self_sg, result, rank, size); break;
      case 4: fixed_order_2stage<T, 4><<<blocks, threads, 0, stream>>>(dp, sg, self_sg, result, rank, size); break;
      case 8: fixed_order_2stage<T, 8><<<blocks, threads, 0, stream>>>(dp, sg, self_sg, result, rank, size); break;
      default: TORCH_CHECK(false, "2, 4 or 8 ranks");
    }
  };
  if (inp.scalar_type() == at::ScalarType::BFloat16) launch(nv_bfloat16{});
  else if (inp.scalar_type() == at::ScalarType::Float) launch(float{});
  else TORCH_CHECK(false, "bf16 or fp32 only");
}
"""

_DECLARATION = """
#include <torch/extension.h>
void fixed_order_allreduce(torch::Tensor inp, torch::Tensor out, std::vector<int64_t> buffers,
                           std::vector<int64_t> signals, int64_t rank, int64_t buffer_bytes);
"""
_MODULE: list = []


def _module() -> Any:
    if not _MODULE:
        from torch.utils.cpp_extension import load_inline

        if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        _MODULE.append(load_inline(name="arctic_fixed_order_allreduce", cpp_sources=_DECLARATION, cuda_sources=_SOURCE,
                                   functions=["fixed_order_allreduce"], extra_cuda_cflags=["-O3"], verbose=False))  # fmt: skip
    return _MODULE[0]


def fixed_order_all_reduce(ca: Any, inp: torch.Tensor, out: torch.Tensor) -> None:
    """``inp`` all-reduced into ``out`` on custom all-reduce ``ca`` (vLLM's ``CustomAllreduce``) by the fixed-order
    two-stage kernel: the one-stage kernel's bits."""
    _module().fixed_order_allreduce(inp, out, list(ca.buffer_ptrs), list(ca.meta_ptrs), int(ca.rank), int(ca.max_size))


def patch_custom_allreduce() -> None:
    """vLLM's ``CustomAllreduce`` chunks of :data:`MIN_BYTES_ENV` bytes and more on the fixed-order two-stage kernel,
    when vLLM's batch-invariant mode has pinned the one-stage kernel."""
    from vllm.distributed.device_communicators import custom_all_reduce

    import logging

    log = logging.getLogger(__name__)
    cls = custom_all_reduce.CustomAllreduce
    original = cls._all_reduce_chunk
    min_bytes = int(os.environ.get(MIN_BYTES_ENV, DEFAULT_MIN_BYTES))
    seen = {"fixed": False, "one_stage": False}

    def _all_reduce_chunk(self, inp, out, registered):
        if (getattr(self, "batch_invariant", False) and inp.dtype in (torch.bfloat16, torch.float32)
                and inp.numel() * inp.element_size() >= min_bytes and inp.is_contiguous() and out.is_contiguous()
                and self.world_size in (2, 4, 8)):  # fmt: skip
            if not seen["fixed"]:
                seen["fixed"] = True
                log.warning("fixed-order all-reduce: first call, %d bytes, rank %d", inp.numel() * inp.element_size(),
                            self.rank)  # fmt: skip
            fixed_order_all_reduce(self, inp, out)
            return
        if not seen["one_stage"]:
            seen["one_stage"] = True
            log.warning("fixed-order all-reduce: first one-stage call, %d bytes, batch_invariant=%s",
                        inp.numel() * inp.element_size(), getattr(self, "batch_invariant", None))  # fmt: skip
        original(self, inp, out, registered)

    cls._all_reduce_chunk = _all_reduce_chunk
    log.warning("fixed-order all-reduce: patched CustomAllreduce (chunks of %d bytes and more), pid %d", min_bytes,
                os.getpid())  # fmt: skip
