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

"""One Triton kernel for :func:`~arctic_platform.inference.vllm.glm5_batch_invariance.grouped_topk` with a single
expert group and a score-correction bias (GLM-5.3-Flash: sigmoid scores, 288 experts, top 8).

The caller computes the scores (``torch.sigmoid``, so they round as before). Per token the kernel takes the
largest biased score eight times, ties to the lower expert, gathers the unbiased scores of the chosen experts, sums
them in selection order as ``_ordered_sum`` does, divides with IEEE rounding and multiplies by the scaling factor.

Same experts, order and weights as the composition except among exactly equal biased scores: CUDA ``torch.topk``
orders those by its sorting network (eight equal values come out as experts 7, 6, 4, 5, 1, 0, 2, 3), this kernel by
expert. The MoE sums its experts in that order, so the sampler and the trainer must both take this kernel.
"""

from __future__ import annotations

import torch

MAX_EXPERTS = 1024


def _kernel():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def router_topk_kernel(scores_ptr, bias_ptr, weights_ptr, ids_ptr, tokens, experts, stride, scale,
                           TOPK: tl.constexpr, RENORM: tl.constexpr, SCALE: tl.constexpr, BLOCK_T: tl.constexpr,
                           BLOCK_E: tl.constexpr):  # fmt: skip
        t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        e = tl.arange(0, BLOCK_E)
        k = tl.arange(0, TOPK)
        live = t < tokens
        inside = live[:, None] & (e[None, :] < experts)
        score = tl.load(scores_ptr + t[:, None].to(tl.int64) * stride + e[None, :], mask=inside, other=0.0)
        bias = tl.load(bias_ptr + e, mask=e < experts, other=0.0)
        key = tl.where(inside, score + bias[None, :], float("-inf"))
        weights = tl.zeros([BLOCK_T, TOPK], tl.float32)
        ids = tl.zeros([BLOCK_T, TOPK], tl.int32)
        total = tl.zeros([BLOCK_T], tl.float32)
        for i in tl.static_range(TOPK):
            best = tl.max(key, axis=1)
            pick = tl.min(tl.where(key == best[:, None], e[None, :], BLOCK_E), axis=1)
            chosen = e[None, :] == pick[:, None]
            w = tl.sum(tl.where(chosen, score, 0.0), axis=1)
            total = w if i == 0 else total + w
            weights = tl.where(k[None, :] == i, w[:, None], weights)
            ids = tl.where(k[None, :] == i, pick[:, None], ids)
            key = tl.where(chosen, float("-inf"), key)
        if RENORM:
            weights = libdevice.div_rn(weights, total[:, None])
        if SCALE:
            weights = weights * scale
        out = t[:, None].to(tl.int64) * TOPK + k[None, :]
        tl.store(weights_ptr + out, weights, mask=live[:, None])
        tl.store(ids_ptr + out, ids, mask=live[:, None])

    return router_topk_kernel


_KERNEL = None


def supported(scores: torch.Tensor, topk: int, num_expert_group: int, topk_group: int, bias) -> bool:
    return (scores.is_cuda and scores.dtype == torch.float32 and scores.dim() == 2 and scores.stride(1) == 1
            and bias is not None and num_expert_group <= 1 and topk_group <= 1 and scores.shape[1] <= MAX_EXPERTS
            and topk in (1, 2, 4, 8, 16))  # fmt: skip


def router_topk(scores: torch.Tensor, bias: torch.Tensor, topk: int, renormalize: bool,
                routed_scaling_factor: float) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """``(weights fp32, ids int32)`` of ``[tokens, topk]`` from fp32 ``scores`` and the correction ``bias``."""
    import triton

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _kernel()
    tokens, experts = scores.shape
    weights = torch.empty((tokens, topk), dtype=torch.float32, device=scores.device)
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=scores.device)
    if tokens == 0:
        return weights, ids
    block_e = triton.next_power_of_2(experts)
    block_t = 4 if block_e >= 512 else 8
    _KERNEL[(triton.cdiv(tokens, block_t),)](scores, bias.float().contiguous(), weights, ids, tokens, experts,
                                             scores.stride(0), float(routed_scaling_factor), TOPK=topk,
                                             RENORM=bool(renormalize), SCALE=routed_scaling_factor != 1.0,
                                             BLOCK_T=block_t, BLOCK_E=block_e, num_warps=4)  # fmt: skip
    return weights, ids
