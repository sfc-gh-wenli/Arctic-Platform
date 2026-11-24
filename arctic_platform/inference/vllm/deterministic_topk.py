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

"""One Triton program per row for :func:`~arctic_platform.inference.vllm.glm5_batch_invariance.deterministic_topk`.

The row's scores become the same 64-bit keys (order-preserving score bits above the column), which are sorted in
registers; the first ``k`` keys' columns are sorted again and written. The selection is the composition's to the bit:
the keys are unique, so the top ``k`` set has one answer however it is found, and both write it in descending column
order. One launch replaces about ten kernels over int64 copies of the row.
"""

from __future__ import annotations

import torch

MAX_COLUMNS = 8192  # a row's keys stay in registers up to this many columns; wider rows take the composition


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def deterministic_topk_kernel(logits_ptr, starts_ptr, ends_ptr, out_ptr, cols, stride_row, stride_out, k,
                                  width, BLOCK: tl.constexpr, OUT_BLOCK: tl.constexpr):  # fmt: skip
        row = tl.program_id(0)
        start = tl.load(starts_ptr + row).to(tl.int64)
        end = tl.load(ends_ptr + row).to(tl.int64)
        col = tl.arange(0, BLOCK).to(tl.int64)
        valid = (col >= start) & (col < end) & (col < cols)
        score = tl.load(logits_ptr + row * stride_row + col, mask=col < cols, other=0.0).to(tl.float32)
        bits = score.to(tl.int32, bitcast=True).to(tl.int64)
        ordered = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
        key = tl.where(valid, (ordered << 32) | col, -9223372036854775808)
        key = tl.sort(key, descending=True)
        rank = tl.arange(0, BLOCK)
        chosen = tl.where((rank < k) & (key != -9223372036854775808), key & 0xFFFFFFFF, -1)
        chosen = tl.sort(chosen, descending=True)
        written = tl.where(chosen >= 0, chosen - start, -1)
        tl.store(out_ptr + row * stride_out + rank, written.to(tl.int32), mask=rank < width)
        if OUT_BLOCK > BLOCK:
            tail = tl.arange(0, OUT_BLOCK)
            tl.store(out_ptr + row * stride_out + tail, tl.full([OUT_BLOCK], -1, tl.int32),
                     mask=(tail >= BLOCK) & (tail < width))  # fmt: skip

    return deterministic_topk_kernel


_KERNEL = None


def supported(logits: torch.Tensor, out: torch.Tensor) -> bool:
    return (logits.is_cuda and logits.dim() == 2 and logits.stride(1) == 1 and out.dtype == torch.int32
            and out.stride(1) == 1 and logits.shape[1] <= MAX_COLUMNS)  # fmt: skip


def deterministic_topk_triton(logits: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, k: int,
                              out: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """``deterministic_topk`` for a CUDA row-major ``logits`` of at most ``MAX_COLUMNS`` columns and an int32 ``out``."""
    import triton

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _kernel()
    rows, cols = logits.shape
    if rows == 0:
        return out
    block = max(16, triton.next_power_of_2(cols))
    out_block = max(block, triton.next_power_of_2(out.shape[1]))
    _KERNEL[(rows,)](logits, starts, ends, out, cols, logits.stride(0), out.stride(0), min(k, cols), out.shape[1],
                     BLOCK=block, OUT_BLOCK=out_block, num_warps=8 if block >= 2048 else 4)  # fmt: skip
    return out
