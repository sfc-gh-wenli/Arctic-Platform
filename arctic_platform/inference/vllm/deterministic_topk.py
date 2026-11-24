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

""":func:`~arctic_platform.inference.vllm.glm5_batch_invariance.deterministic_topk` with its keys built in one Triton
pass.

The composition builds each row's 64-bit keys (order-preserving score bits above the column, ``INT64_MIN`` outside the
row's range) in about ten elementwise kernels over int64 copies of the scores. One kernel writes the same keys; the
same ``torch.topk`` then selects the same columns, and the final descending sort runs on their int32 values, which
orders them the same way.
"""

from __future__ import annotations

import torch

_BLOCK = 1024


def _keys_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def topk_keys_kernel(logits_ptr, starts_ptr, ends_ptr, keys_ptr, cols, stride_row, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        start = tl.load(starts_ptr + row).to(tl.int64)
        end = tl.load(ends_ptr + row).to(tl.int64)
        inside = col < cols
        valid = (col >= start) & (col < end) & inside
        score = tl.load(logits_ptr + row * stride_row + col, mask=inside, other=0.0).to(tl.float32)
        bits = score.to(tl.int32, bitcast=True).to(tl.int64)
        ordered = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
        key = tl.where(valid, (ordered << 32) | col, -9223372036854775808)
        tl.store(keys_ptr + row.to(tl.int64) * cols + col, key, mask=inside)

    return topk_keys_kernel


_KEYS_KERNEL = None


def deterministic_topk_keys(logits: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, k: int,
                            out: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """``deterministic_topk`` for a CUDA ``logits`` whose rows are contiguous."""
    import triton

    global _KEYS_KERNEL
    if _KEYS_KERNEL is None:
        _KEYS_KERNEL = _keys_kernel()
    rows, cols = logits.shape
    if rows == 0:
        return out
    keys = torch.empty((rows, cols), dtype=torch.int64, device=logits.device)
    _KEYS_KERNEL[(rows, triton.cdiv(cols, _BLOCK))](logits, starts, ends, keys, cols, logits.stride(0), BLOCK=_BLOCK)
    take = min(k, cols)
    top = torch.topk(keys, take, dim=-1, sorted=False).indices
    valid = keys.gather(1, top) != torch.iinfo(torch.int64).min
    picked = torch.where(valid, top - starts[:, None].long(), -1).to(torch.int32)
    out[:, :take] = picked.sort(dim=-1, descending=True).values.to(out.dtype)
    out[:, take:] = -1
    return out
