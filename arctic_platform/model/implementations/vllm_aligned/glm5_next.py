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
"""vLLM-aligned logprobs for GLM-5.3-Flash (``glm5_next``). Work in progress: the mixers and the MoE are not yet in.

The target is the sampler :mod:`arctic_platform.inference.vllm.glm5_batch_invariance` makes batch-invariant
(``VLLM_BATCH_INVARIANT=1``, ``ARCTIC_GLM5_BATCH_INVARIANT=1``, enforce_eager, no MTP, no prefix caching, prompts
prefilled whole), at tensor parallelism 1. Every per-token operator of that sampler gives a token the same bits
whatever else the step holds, so the trainer calls those kernels once on its packed tokens; the operators that mix
tokens (KDA, the sparse indexer and sparse attention) replay the sampler's prefill and decode structure.

Each decoder layer is one :class:`_AlignedLayer`: its forward runs the vLLM kernels without autograd on the layer's
input residual streams, and its backward recomputes the trainer's stock layer on those streams and backpropagates
through it, as reentrant activation checkpointing does. The logprobs are then vLLM's, and the gradient is the stock
layer's at the activations the sampler's kernels produced.

vLLM defers a layer's last hyper-connection post into the next layer's fused post + pre-norm. With the fused kernel
off (as the batch-invariant sampler runs it) that op is the standalone post kernel followed by the pre-norm GEMM and
the pre kernel on the same per-token data, so materializing the residual streams at every layer boundary, as the
stock layers carry them, gives the same bits.

Kernels the sampler runs (its kernel inventory, ``layer_digest.kernel_inventory``, on the 8-layer proxy at TP1):

* every unquantized linear (all KDA projections, MLA ``fused_qkv_a``/``q_b``/``kv_b``, the indexer's ``wq_b`` and
  ``wk_weights_proj``, every ``o_proj``): ``linear_batch_invariant``, whose tile is chosen by ``(N, K)``, so the
  trainer passes vLLM's merged weights (KDA's ``[q; k; v; b; f_a; g_a]``, MLA's ``[q_a; kv_a]``);
* dense MLP and shared-expert ``gate_up``/``down``: block-FP8 ``TritonFp8BlockScaledMMKernel`` (activations
  quantized per 128-group, row-major scales, no UE8M0) with ``SiluAndMulWithClamp``;
* routed experts: ``FusedMoEKernel`` with ``TritonExperts`` (clamp limit 10), routing by the batch-invariant
  grouped top-k on fp32 gate logits, ``routed_scaling_factor`` applied to the routed output;
* sparse attention ``FlashAttnMLASparseImpl`` (FA3, one split); KDA prefill FlashKDA, decode ``fused_recurrent_kda``;
* direct ``F.linear``/``torch.mm`` calls in model code (the indexer's kpool gate and head weights) bypass the linear
  layers; the head weights go through the batch-invariant matmul in the sampler patch.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import os
from typing import Any
from typing import Callable

import torch

from arctic_platform.model.implementations.vllm_aligned import ops

FP8_BLOCK = 128  # block-FP8 weight block (rows and columns)
# "0": run the KDA ops before o_proj once per emulated TP rank instead of once over all of them (same bits).
TP_BATCH_ENV = "ARCTIC_VLLM_ALIGNED_TP_BATCH"
# "0": a sequence-parallel rank computes its mixer partials at every token, not only at those it hands them on for.
SP_QUERIES_ENV = "ARCTIC_VLLM_ALIGNED_SP_QUERIES"
# "1"/"0": the backward ("vllm") takes the forward's mixer and MoE outputs instead of recomputing them (same values).
# Default: with sequence parallelism, where a rank keeps only its shard's rows.
KEEP_OUTPUTS_ENV = "ARCTIC_VLLM_ALIGNED_KEEP_OUTPUTS"
# "1": time the forward's and backward's phases with CUDA events; rank 0 logs the totals at the next call.
PROFILE_ENV = "ARCTIC_VLLM_ALIGNED_PROFILE"
_PHASE_PREFIX = contextvars.ContextVar("vllm_aligned_phase", default="fwd")
_PHASES: list = []
_PHASE_TOKENS: list = [0]

# Set while a layer's backward recomputes the aligned forward with every kernel call differentiable (``vllm_op``).
_DIFFERENTIABLE = contextvars.ContextVar("vllm_aligned_differentiable", default=False)


@contextlib.contextmanager
def differentiable():
    """Within: each vLLM kernel call of the aligned forward keeps its value and gets the gradient of its reference
    math at its inputs (:mod:`glm5_next_backward`)."""
    token = _DIFFERENTIABLE.set(True)
    try:
        yield
    finally:
        _DIFFERENTIABLE.reset(token)


@contextlib.contextmanager
def _phase(name: str):
    """With :data:`PROFILE_ENV`, times the block on the current stream (:func:`_log_phases` reports it)."""
    if os.environ.get(PROFILE_ENV) != "1":
        yield
        return
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        _PHASES.append((f"{_PHASE_PREFIX.get()}.{name}", start, end))


def _log_phases(tokens: int) -> None:
    """Rank 0's phase totals (seconds, nested phases counted in their parents too) since the last call, then starts
    timing a call of ``tokens`` packed tokens."""
    if _PHASES:
        import collections
        import json
        import logging

        torch.cuda.synchronize()
        totals: dict = collections.defaultdict(float)
        for name, start, end in _PHASES:
            totals[name] += start.elapsed_time(end) / 1000
        _PHASES.clear()
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logging.getLogger(__name__).warning("vllm_aligned profile: tokens=%d %s", _PHASE_TOKENS[0],
                                                json.dumps({k: round(v, 3) for k, v in sorted(totals.items())}))  # fmt: skip
    _PHASE_TOKENS[0] = tokens


def _bind(fn: Callable, static: dict) -> Callable:
    """``fn`` with the keyword arguments of ``static`` it takes."""
    import inspect

    params = inspect.signature(fn).parameters
    if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
        static = {k: v for k, v in static.items() if k in params}
    return functools.partial(fn, **static)


def _op(kernel: Callable, reference: str, *inputs: torch.Tensor, **static: Any):
    """``kernel(*inputs, **static)``; in :func:`differentiable` mode with the gradient of the named reference."""
    if not _DIFFERENTIABLE.get():
        return _bind(kernel, static)(*inputs)
    from arctic_platform.model.implementations.vllm_aligned import glm5_next_backward as bw

    return bw.vllm_op(_bind(kernel, static), _bind(getattr(bw, reference), static), *inputs)


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _op(ops.linear_kernel, "ref_linear", x, weight)


def tp_shard(tensor: torch.Tensor, tp: int, rank: int, dim: int = 0) -> torch.Tensor:
    """Tensor-parallel rank ``rank``'s contiguous ``1 / tp`` of ``tensor`` along ``dim``, as vLLM's loaders shard."""
    if tp == 1:
        return tensor
    size = tensor.shape[dim]
    if size % tp:
        raise ValueError(f"{size} rows do not split across {tp} tensor-parallel ranks")
    return tensor.narrow(dim, rank * (size // tp), size // tp)


def tp_sum(partials: list[torch.Tensor]) -> torch.Tensor:
    """vLLM's batch-invariant all-reduce: the ranks' partials added in fp32 in rank order, rounded once."""
    if len(partials) == 1:
        return partials[0]
    total = partials[0].float()
    for partial in partials[1:]:
        total = total + partial.float()
    return total.to(partials[0].dtype)


def _shard(weight: torch.Tensor, scale: torch.Tensor | None, tp: int, rank: int,
           dim: int) -> tuple[torch.Tensor, torch.Tensor | None]:  # fmt: skip
    """A rank's shard of a weight along ``dim``: of a bf16 weight (``scale`` None), or of a block-FP8 one with its
    block scales (:func:`_fp8_shard`)."""
    if scale is None:
        return tp_shard(weight, tp, rank, dim).contiguous(), None
    return _fp8_shard(weight, scale, tp, rank, dim)


def _fp8_shard(weight: torch.Tensor, scale: torch.Tensor, tp: int, rank: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A rank's shard of a block-FP8 weight and its block scales along ``dim`` (whole 128-blocks per rank)."""
    if tp == 1:
        return weight, scale
    if (weight.shape[dim] // tp) % FP8_BLOCK:
        raise ValueError(f"{weight.shape[dim]} FP8 rows do not split across {tp} ranks in {FP8_BLOCK}-blocks")
    return tp_shard(weight, tp, rank, dim).contiguous(), tp_shard(scale, tp, rank, dim).contiguous()


def _column_lora(factors: tuple | None, tp: int, rank: int) -> tuple | None:
    """vLLM's column-parallel LoRA shard: the whole ``A``, the rank's rows of ``B``."""
    return factors if factors is None or tp == 1 else (factors[0], tp_shard(factors[1], tp, rank).contiguous())


def _row_lora(factors: tuple | None, tp: int, rank: int) -> tuple | None:
    """vLLM's row-parallel LoRA shard: the rank's input columns of ``A``, the whole ``B``."""
    return factors if factors is None or tp == 1 else (tp_shard(factors[0], tp, rank, 1).contiguous(), factors[1])


def _mhc_constants():
    from vllm.models.glm5next.common.model import _MHC_POST_MULT_VALUE

    return _MHC_POST_MULT_VALUE


def hc_pre(streams: torch.Tensor, hc: Any, norm: Any, config: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """vLLM's ``mhc_pre`` (TileLang, one split) with the sublayer's RMSNorm fused: ``(post, comb, layer_input)``.

    ``streams`` is ``[tokens, hc_mult, hidden]`` bf16; ``hc`` the stock ``Glm5NextTextHyperConnection`` and ``norm``
    the RMSNorm that follows it in the stock layer.
    """
    return _op(_mhc_pre_kernel, "ref_mhc_pre", streams, hc.fn, hc.scale, hc.base, norm.weight,
               rms_eps=config.rms_norm_eps, pre_eps=config.hc_eps, sinkhorn_eps=config.hc_eps,
               post_mult=_mhc_constants(), iters=config.hc_sinkhorn_iters, norm_eps=norm.variance_epsilon)  # fmt: skip


def _mhc_pre_kernel(streams, fn, scale, base, norm_weight, *, rms_eps, pre_eps, sinkhorn_eps, post_mult, iters,
                    norm_eps):  # fmt: skip
    return torch.ops.vllm.mhc_pre_tilelang(streams, fn.float(), scale.float(), base.float(), rms_eps, pre_eps,
                                           sinkhorn_eps, post_mult, iters, 1, norm_weight, norm_eps)  # fmt: skip


def hc_post(x: torch.Tensor, streams: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """vLLM's ``mhc_post``: ``post_j * x + sum_i comb_ij * streams_i``, the residual streams after a sublayer."""
    return _op(torch.ops.vllm.mhc_post_tilelang, "ref_mhc_post", x, streams, post, comb)


class _AlignedLayer(torch.autograd.Function):
    """One decoder layer: the aligned (vLLM-kernel) forward, the stock layer's backward."""

    @staticmethod
    def forward(ctx, streams: torch.Tensor, run_aligned: Callable, run_stock: Callable) -> torch.Tensor:
        ctx.run_stock = run_stock
        ctx.save_for_backward(streams)
        with torch.no_grad():
            return run_aligned(streams)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (streams,) = ctx.saved_tensors
        # Index and scatter accumulations accumulate with atomics by default (0.2-0.8% run to run at a MoE layer);
        # torch's deterministic implementations make the layer's backward bitwise reproducible. Its fill of new
        # tensors stays off: DeepEP writes its receive buffers from its own stream, which the fill would race.
        import torch.utils.deterministic as deterministic

        previous = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                    deterministic.fill_uninitialized_memory)  # fmt: skip
        torch.use_deterministic_algorithms(True, warn_only=True)
        deterministic.fill_uninitialized_memory = False
        phase = _PHASE_PREFIX.set("bwd")
        try:
            with torch.enable_grad(), _phase("recompute"):
                replay = streams.detach().requires_grad_(streams.requires_grad)
                out = ctx.run_stock(replay)
            with _phase("autograd"):
                torch.autograd.backward(out, grad_out)
        finally:
            _PHASE_PREFIX.reset(phase)
            torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
            deterministic.fill_uninitialized_memory = previous[2]
        return (replay.grad if streams.requires_grad else None), None, None


KDA_O_NORM_EPS = 1e-5  # vLLM builds the gated output norm with FusedRMSNormGated's default eps, not rms_norm_eps


def _length_bucket(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def _kda_conv_kernel(qkv: torch.Tensor, conv_weight: torch.Tensor, *, layout: Any, trace: dict | None = None,
                     **_) -> torch.Tensor:  # fmt: skip
    """vLLM's q|k|v causal conv with SiLU over ``layout.steps`` (:func:`_kda_steps_layout`): ``causal_conv1d_fn`` for
    a prefill step (each request from its conv state slot, or zero), ``causal_conv1d_update`` for a decode step.
    ``[tokens, 3 * projection]``.

    A decode step runs every request's decoded tokens in one varlen ``causal_conv1d_update``: its per-token
    arithmetic is a decode step's, the window carried in registers instead of the state slot (bitwise equal to a call
    per step). A step's ``imports`` load a saved state into a slot before it runs, its ``exports`` save one after."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    # State slot 0 is vLLM's null block, which the conv kernel treats as padding, so the slots start at 1.
    conv_states = qkv.new_zeros((layout.num_slots + 1, qkv.shape[-1], conv_weight.shape[-1] - 1))
    if layout.single:
        (step,) = layout.steps
        out = causal_conv1d_fn(qkv.transpose(0, 1), conv_weight, None, activation="silu", conv_states=conv_states,
                               has_initial_state=step.has_initial_state, cache_indices=step.slots,
                               query_start_loc=step.cu).transpose(0, 1)  # fmt: skip
        if trace is not None:
            trace.update(conv_in=qkv.transpose(0, 1), conv_weight=conv_weight, conv_out=out)
        return out
    conv, saved = qkv.new_empty(qkv.shape), {}
    for step in layout.steps:
        for slot, key in step.imports:
            conv_states[slot] = saved[key]
        x = qkv[step.rows]
        if step.decode:
            # The length bound is a compile-time constant of the kernel; bucketing it bounds the recompiles.
            conv[step.rows] = causal_conv1d_update(x, conv_states, conv_weight, None, activation="silu",
                                                   conv_state_indices=step.slots, query_start_loc=step.cu,
                                                   max_query_len=_length_bucket(step.max_len))  # fmt: skip
        else:
            conv[step.rows] = causal_conv1d_fn(x.transpose(0, 1), conv_weight, None, activation="silu",
                                               conv_states=conv_states, has_initial_state=step.has_initial_state,
                                               cache_indices=step.slots, query_start_loc=step.cu).transpose(0, 1)  # fmt: skip
        for slot, key in step.exports:
            saved[key] = conv_states[slot].clone()
    return conv


def _kda_core_kernel(q, k, v, g, beta, a_log, dt_bias, *, layout: Any, lower_bound: float,
                     trace: dict | None = None, **_) -> torch.Tensor:  # fmt: skip
    """vLLM's KDA core on ``[1, tokens, heads, head_dim]`` q/k/v/g and ``[tokens, heads]`` beta logits over
    ``layout.steps``: FlashKDA for a prefill step, from the fp32 state in each request's slot (zero for a new
    request), and ``fused_recurrent_kda`` (gate and beta sigmoid in the kernel) for a decode step.

    A decode step runs every request's decoded tokens in one varlen ``fused_recurrent_kda``: the kernel carries the
    fp32 state in registers through a decode step's per-token arithmetic, where the steps reload it from the fp32
    slot (lossless), so the outputs are bitwise a call per step's. The state indices are per token (the request's
    slot each time), so each token's state store lands in its own request's slot. ``imports`` and ``exports`` load
    and save states as in :func:`_kda_conv_kernel`."""
    import vllm._flashkda_C  # noqa: F401
    from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda

    heads, head_dim = q.shape[2:]
    # The fp32 recurrent state of each request's slot; slot 0 is the null block.
    recurrent = torch.zeros((layout.num_slots + 1, heads, head_dim, head_dim), dtype=torch.float32, device=q.device)

    def prefill(step, q_p, k_p, v_p, g_p, beta_p):
        tokens, seqs, slots = q_p.shape[1], step.slots.numel(), step.slots.long()
        state = recurrent[slots]
        final_state = torch.empty_like(state)
        workspace = torch.empty(torch.ops._flashkda_C.get_workspace_size(tokens, heads, seqs), dtype=torch.uint8,
                                device=q.device)  # fmt: skip
        out = torch.empty((1, tokens, heads, head_dim), dtype=q.dtype, device=q.device)
        torch.ops._flashkda_C.fwd(
            q_p.contiguous(), k_p.contiguous(), v_p.contiguous(), g_p.contiguous(), beta_p.unsqueeze(0),
            head_dim**-0.5, out, workspace, a_log, dt_bias.view(-1, head_dim), lower_bound,
            state, final_state, step.cu.contiguous(), None, None,
        )  # fmt: skip
        recurrent[slots] = final_state
        return out

    if layout.single:
        if trace is not None:
            trace.update(q=q, k=k, v=v, g=g, beta=beta.unsqueeze(0))
        return prefill(layout.steps[0], q, k, v, g, beta)
    core, saved = torch.empty(q.shape, dtype=q.dtype, device=q.device), {}
    for step in layout.steps:
        for slot, key in step.imports:
            recurrent[slot] = saved[key]
        rows = step.rows
        if step.decode:
            # Contiguous q/k/v (token stride heads * head_dim), as the sampler passes them (``kda_decode_strides``):
            # the kernel's compiled variant, and its rounding, depends on whether that stride is a multiple of 16.
            slots = step.slots[:, None].expand(step.slots.numel(), max(step.max_len, 1)).contiguous()
            out, _ = fused_recurrent_kda(
                q=q[:, rows].contiguous(), k=k[:, rows].contiguous(), v=v[:, rows].contiguous(), g=g[:, rows],
                beta=beta[rows].unsqueeze(0), initial_state=recurrent, use_qk_l2norm_in_kernel=True,
                cu_seqlens=step.cu, ssm_state_indices=slots, sigmoid_beta=True, a_log=a_log.reshape(1, 1, heads, 1),
                g_bias=dt_bias, compute_gate=True, lower_bound=lower_bound,
            )  # fmt: skip
            core[:, rows] = out.to(core.dtype)
        else:
            core[:, rows] = prefill(step, q[:, rows], k[:, rows], v[:, rows], g[:, rows], beta[rows])
        for slot, key in step.exports:
            saved[key] = recurrent[slot].clone()
    return core


def _gated_norm_kernel(x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

    # It writes its output over a contiguous ``x``; the per-op backward keeps ``x`` for the reference.
    return rms_norm_gated(x.clone(), g, weight, None, "sigmoid", eps=eps)


def kda_prefill(attn: Any, x: torch.Tensor, cu_seqlens: torch.Tensor, config: Any, trace: dict | None = None) -> torch.Tensor:
    """:func:`kda_forward` with every token prefilled."""
    return kda_forward(attn, x, cu_seqlens, config, trace=trace)


def _kda_weights(attn: Any, tp: int = 1, rank: int = 0) -> Any:
    """Tensor-parallel rank ``rank``'s KDA weights as vLLM holds them: its heads of q/k/v/b, ``f_b``/``g_b``, the conv,
    ``A_log`` and ``dt_bias``, the replicated ``f_a``/``g_a``, and its input columns of ``o_proj`` (and of its LoRA A)."""
    import types

    gate = attn.forget_gate
    projection = attn.num_heads * attn.head_dim
    conv = attn.conv1d.weight.float().reshape(3 * projection, -1)
    merged = [tp_shard(t, tp, rank) for t in (attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight,
                                               attn.b_proj.weight)]  # fmt: skip
    return types.SimpleNamespace(
        heads=attn.num_heads // tp,
        head_dim=attn.head_dim,
        merged=torch.cat([*merged, gate.f_a_proj.weight, attn.g_a_proj.weight]),
        f_b=tp_shard(gate.f_b_proj.weight, tp, rank),
        g_b=tp_shard(attn.g_b_proj.weight, tp, rank),
        conv=torch.cat([tp_shard(c, tp, rank) for c in conv.split(projection)]),
        a_log=tp_shard(gate.A_log.float().reshape(-1), tp, rank),
        dt_bias=tp_shard(gate.dt_bias.float().reshape(-1), tp, rank),
        o_norm=attn.o_norm.weight,
        o_weight=tp_shard(bf16_weight(attn.o_proj), tp, rank, 1).contiguous(),
        o_lora=_row_lora(lora_factors(attn.o_proj), tp, rank),
    )


def kda_forward(attn: Any, x: torch.Tensor, cu_seqlens: torch.Tensor, config: Any,
                prompt_lens: torch.Tensor | None = None, trace: dict | None = None, tp: int = 1,
                prefill_chunk: int = 0, layout: Any = None) -> torch.Tensor:  # fmt: skip
    """vLLM's ``Glm5NextLinearAttention`` on packed requests: each prompt prefilled (whole, or ``prefill_chunk``
    tokens at a time as the sampler's ``ARCTIC_PREFILL_CHUNK_ALIGN`` cuts it), then the rest decoded.

    ``x`` is ``[tokens, hidden]`` (the layer input after the hyper-connection pre-norm), ``cu_seqlens`` the int32
    request boundaries and ``prompt_lens`` how many leading tokens of each request were prefilled (all by default).
    The merged ``[q; k; v; b; f_a; g_a]`` projection; over each prompt the merged q|k|v causal conv with vLLM's fp32
    conv weights and FlashKDA on the raw gate and beta logits; over each later token, one decode step at a time, the
    conv update and the recurrent kernel from the conv state and the fp32 recurrent state the request has reached;
    then the sigmoid-gated RMSNorm and ``o_proj``. ``trace`` (debug) collects the prefill's intermediate tensors.

    At ``tp`` > 1 every tensor-parallel rank's share runs on its own weights (:func:`_kda_weights`), and the ranks'
    ``o_proj`` partials are all-reduced (:func:`tp_sum`). ``layout`` (:func:`_kda_steps_layout`) replaces the
    single-turn rule with a turn plan's spans.
    """
    return tp_sum(kda_partials(attn, x, cu_seqlens, config, prompt_lens, trace, tp, prefill_chunk, layout, range(tp)))


def kda_partials(attn: Any, x: torch.Tensor, cu_seqlens: torch.Tensor, config: Any, prompt_lens: torch.Tensor | None,
                 trace: dict | None, tp: int, prefill_chunk: int, layout: Any, ranks) -> list[torch.Tensor]:  # fmt: skip
    """The ``o_proj`` partials of tensor-parallel ranks ``ranks`` (:func:`kda_forward` before its all-reduce)."""
    ranks = list(ranks)
    if trace is None and len(ranks) > 1 and os.environ.get(TP_BATCH_ENV, "1") != "0":
        return _kda_ranks([_kda_weights(attn, tp, rank) for rank in ranks], x, cu_seqlens, config, prompt_lens,
                          prefill_chunk, layout)  # fmt: skip
    return [_kda_rank(_kda_weights(attn, tp, rank), x, cu_seqlens, config, prompt_lens, trace if rank == 0 else None,
                      prefill_chunk, layout) for rank in ranks]  # fmt: skip


def _kda_ranks(ws: list, x: torch.Tensor, cu_seqlens: torch.Tensor, config: Any, prompt_lens: torch.Tensor | None,
               prefill_chunk: int = 0, layout: Any = None) -> list[torch.Tensor]:  # fmt: skip
    """:func:`_kda_rank`'s ``o_proj`` partials of several ranks: every op before ``o_proj`` is column- or
    head-parallel and runs once over all the ranks' columns or heads, which gives each rank the bits of its own
    call; ``o_proj`` (row-parallel, its partials summed in rank order) runs per rank."""
    n, heads, head_dim = len(ws), ws[0].heads, ws[0].head_dim
    projection = heads * head_dim
    tokens = x.shape[0]
    if layout is None:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        prompt_lens = lengths if prompt_lens is None else prompt_lens.long()
        if prefill_chunk > 0:
            prompt_lens = prompt_lens - ((prompt_lens > 1) & (prompt_lens % prefill_chunk == 1)).long()
        layout = _single_turn_kda_layout(cu_seqlens, prompt_lens, prefill_chunk, x.device)
    rows = [w.merged[: 3 * projection] for w in ws] + [w.merged[3 * projection : 3 * projection + heads] for w in ws]
    projected = _linear(x, torch.cat([*rows, ws[0].merged[3 * projection + heads :]]))
    qkv, beta, f_a, g_a = projected.split([3 * projection * n, heads * n, head_dim, head_dim], dim=-1)
    g1 = _linear(f_a, torch.cat([w.f_b for w in ws])).reshape(1, tokens, heads * n, head_dim)
    g2 = _linear(g_a, torch.cat([w.g_b for w in ws])).reshape(tokens, heads * n, head_dim)
    conv = _op(_kda_conv_kernel, "ref_causal_conv_silu", qkv, torch.cat([w.conv for w in ws]), layout=layout,
               trace=None, cu_seqlens=cu_seqlens)  # fmt: skip
    q, k, v = (t.reshape(1, tokens, heads * n, head_dim) for t in conv.view(tokens, n, 3, projection).unbind(2))
    core = _op(_kda_core_kernel, "ref_kda_core", q, k, v, g1, beta, torch.cat([w.a_log for w in ws]),
               torch.cat([w.dt_bias for w in ws]), layout=layout, lower_bound=config.linear_lower_bound, trace=None,
               cu_seqlens=cu_seqlens)  # fmt: skip
    core = _op(_gated_norm_kernel, "ref_gated_rms_norm", core, g2, ws[0].o_norm, eps=KDA_O_NORM_EPS)
    return [add_lora(_linear(c.reshape(tokens, -1), w.o_weight), c.reshape(tokens, -1), [w.o_lora],
                     (w.o_weight.shape[0],)) for c, w in zip(core.split(heads, dim=2), ws)]  # fmt: skip


def _kda_step(pieces: list[tuple[int, int, int]], *, decode: bool, resumed: set, device: torch.device) -> Any:
    """One batched KDA call over ``pieces`` ``(slot, lo, hi)`` (packed token ranges): its token rows, ``cu``
    boundaries, state slots, whether each request starts from its slot's state, and the longest piece."""
    import types

    sizes = torch.tensor([hi - lo for _, lo, hi in pieces], dtype=torch.int32)
    return types.SimpleNamespace(
        decode=decode,
        rows=torch.cat([torch.arange(lo, hi) for _, lo, hi in pieces]).to(device),
        cu=torch.cat([sizes.new_zeros(1), torch.cumsum(sizes, 0, dtype=torch.int32)]).to(device),
        slots=torch.tensor([slot for slot, _, _ in pieces], dtype=torch.int32, device=device),
        has_initial_state=torch.tensor([slot in resumed for slot, _, _ in pieces], dtype=torch.bool, device=device),
        max_len=int(sizes.max()) if len(pieces) else 0,
        imports=[],
        exports=[],
    )  # fmt: skip


def _kda_steps_layout(spans: list[tuple[int, list[tuple[bool, int, int]], tuple[int, int] | None]],
                      device: torch.device) -> Any:
    """The KDA calls that replay how the sampler computed ``spans``: per span (its slot is its index + 1) its packed
    start, its segments ``(decode, lo, hi)`` relative to that start in the order the request ran them, and the
    ``(span, offset)`` whose state it resumed from (``None``: a new request).

    Spans run level by level (a span after the one it resumes from), the ``k``-th segments of a level's spans
    batched into one prefill and one decode call; a resumed span's slot is loaded with the state its provider had
    after the segment ending at the offset. A one-token prefill segment is a decode step, as in vLLM."""
    import types

    normalized = []
    for start, segments, resume in spans:
        merged: list[list] = []
        for decode, lo, hi in segments:
            decode = decode or hi - lo == 1
            if merged and decode and merged[-1][0]:
                merged[-1][2] = hi
            else:
                merged.append([decode, lo, hi])
        normalized.append((start, [tuple(m) for m in merged], resume))
    levels: list[int] = []
    for start, segments, resume in normalized:
        if resume is not None and not 0 <= resume[0] < len(levels):
            raise ValueError(f"a span resumes from span {resume[0]}, which does not come before it")
        levels.append(0 if resume is None else levels[resume[0]] + 1)
    wanted = {resume for _, _, resume in normalized if resume is not None}
    for span, offset in wanted:
        if offset not in {hi for _, _, hi in normalized[span][1]}:
            raise ValueError(f"a span resumes at offset {offset} of span {span}, where none of its segments ends")
    resumed = {i + 1 for i, (_, _, resume) in enumerate(normalized) if resume is not None}
    steps = []
    for level in range(max(levels, default=-1) + 1):
        members = [i for i, value in enumerate(levels) if value == level]
        first = len(steps)
        for k in range(max(len(normalized[i][1]) for i in members)):
            for decode in (False, True):
                pieces = [(i, lo, hi) for i in members if k < len(normalized[i][1])
                          for seg_decode, lo, hi in (normalized[i][1][k],) if seg_decode == decode]  # fmt: skip
                if not pieces:
                    continue
                step = _kda_step([(i + 1, normalized[i][0] + lo, normalized[i][0] + hi) for i, lo, hi in pieces],
                                 decode=decode, resumed=resumed | {i + 1 for i, lo, _ in pieces if lo > 0},
                                 device=device)  # fmt: skip
                step.exports = [(i + 1, (i, hi)) for i, _, hi in pieces if (i, hi) in wanted]
                steps.append(step)
        steps[first].imports = [(i + 1, normalized[i][2]) for i in members if normalized[i][2] is not None]
    return types.SimpleNamespace(num_slots=len(spans), steps=steps, single=False,
                                 spans=[(start, segments[-1][2], resume) for start, segments, resume in normalized])


def _single_turn_kda_layout(cu_seqlens: torch.Tensor, prompt_lens: torch.Tensor, prefill_chunk: int,
                            device: torch.device) -> Any:
    """:func:`_kda_steps_layout` for packed requests whose first ``prompt_lens`` tokens the sampler prefilled
    (whole, or ``prefill_chunk`` tokens at a time from the request's start), the rest decoded."""
    import types

    starts, lengths = cu_seqlens[:-1].tolist(), (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    prompts = prompt_lens.tolist()
    if all(p == n for p, n in zip(prompts, lengths)) and not (prefill_chunk and max(prompts) > prefill_chunk):
        num_seqs = len(starts)
        step = types.SimpleNamespace(decode=False, rows=None, cu=cu_seqlens,
                                     slots=torch.arange(1, num_seqs + 1, dtype=torch.int32, device=device),
                                     has_initial_state=torch.zeros(num_seqs, dtype=torch.bool, device=device),
                                     max_len=0, imports=[], exports=[])  # fmt: skip
        return types.SimpleNamespace(num_slots=num_seqs, steps=[step], single=True)
    chunk = prefill_chunk or max(max(prompts), 1)
    rounds = max(-(-p // chunk) for p in prompts)
    steps = []
    for r in range(rounds):
        pieces = [(s + 1, starts[s] + r * chunk, starts[s] + min((r + 1) * chunk, p))
                  for s, p in enumerate(prompts) if p > r * chunk]  # fmt: skip
        steps.append(_kda_step(pieces, decode=False, resumed=set() if r == 0 else {s for s, _, _ in pieces},
                               device=device))  # fmt: skip
    if any(p < n for p, n in zip(prompts, lengths)):
        pieces = [(s + 1, starts[s] + p, starts[s] + n) for s, (p, n) in enumerate(zip(prompts, lengths))]
        steps.append(_kda_step(pieces, decode=True, resumed={s for s, _, _ in pieces}, device=device))
    return types.SimpleNamespace(num_slots=len(starts), steps=steps, single=False)


def _kda_rank(w: Any, x: torch.Tensor, cu_seqlens: torch.Tensor, config: Any, prompt_lens: torch.Tensor | None,
              trace: dict | None, prefill_chunk: int = 0, layout: Any = None) -> torch.Tensor:  # fmt: skip
    heads, head_dim = w.heads, w.head_dim
    projection = heads * head_dim
    tokens = x.shape[0]
    if layout is None:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        prompt_lens = lengths if prompt_lens is None else prompt_lens.long()
        if prefill_chunk > 0:
            # vLLM runs a one-token chunk as a decode step.
            prompt_lens = prompt_lens - ((prompt_lens > 1) & (prompt_lens % prefill_chunk == 1)).long()
        layout = _single_turn_kda_layout(cu_seqlens, prompt_lens, prefill_chunk, x.device)
    projected = _linear(x, w.merged)
    qkv, beta, f_a, g_a = projected.split([3 * projection, heads, head_dim, head_dim], dim=-1)
    g1 = _linear(f_a, w.f_b).reshape(1, tokens, heads, head_dim)
    g2 = _linear(g_a, w.g_b).reshape(tokens, heads, head_dim)

    conv = _op(_kda_conv_kernel, "ref_causal_conv_silu", qkv, w.conv, layout=layout, trace=trace,
               cu_seqlens=cu_seqlens)  # fmt: skip
    q, k, v = (t.reshape(1, tokens, heads, head_dim) for t in conv.split(projection, dim=-1))
    core = _op(_kda_core_kernel, "ref_kda_core", q, k, v, g1, beta, w.a_log, w.dt_bias, layout=layout,
               lower_bound=config.linear_lower_bound, trace=trace, cu_seqlens=cu_seqlens)  # fmt: skip
    if trace is not None:
        trace.update(core=core, g2=g2)
    core = _op(_gated_norm_kernel, "ref_gated_rms_norm", core, g2, w.o_norm, eps=KDA_O_NORM_EPS).reshape(tokens, -1)
    if trace is not None:
        trace.update(normed=core)
    return add_lora(_linear(core, w.o_weight), core, [w.o_lora], (w.o_weight.shape[0],))


SPARSE_TOPK_BLOCK = 128  # sparse MLA tiles the top-k index buffer in 128 columns


def topk_buffer_width(config: Any) -> int:
    """vLLM's top-k index buffer width: ``index_topk`` plus the incomplete pool tail, rounded up to 128."""
    width = config.index_topk + (config.index_kpool - 1 if config.index_kpool > 1 else 0)
    return -(-width // SPARSE_TOPK_BLOCK) * SPARSE_TOPK_BLOCK


def causal_topk(positions: torch.Tensor, width: int) -> torch.Tensor:
    """vLLM's short-prefill top-k: every earlier token of the request, ``-1`` past the query's position."""
    columns = torch.arange(width, device=positions.device, dtype=torch.int32)
    rows = columns.expand(positions.shape[0], width).clone()
    rows[columns[None, :] > positions.to(torch.int32)[:, None]] = -1
    return rows


INDEXER_K_NORM_EPS = 1e-6
INDEXER_PADDED_HEADS = 32


def indexer_topk(indexer: Any, x: torch.Tensor, q_c: torch.Tensor, positions: torch.Tensor, cu_seqlens: torch.Tensor,
                 config: Any, width: int, keys: tuple[torch.Tensor, torch.Tensor] | None = None,
                 queries: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
    """vLLM's kpool sparse-indexer top-k for whole prompts: ``[tokens, width]`` request-relative token ids, ``-1`` padded.

    ``keys`` (``key_index``, ``key_cu``) gives each request's key positions as packed token indices (the request's own
    tokens by default); with a turn plan a request's keys before its cache hit are the tokens of the request that
    wrote them, and the returned ids are key positions. ``queries`` (packed token indices) limits the rows to those
    tokens' queries; the keys are still every token's.

    A token whose position is below ``index_topk`` gets its causal row (every earlier token). The others score the
    request's complete pools (``(pos + 1) // index_kpool`` of them) as the sampler's prefill does: ``wq_b`` and the
    fused ``[wk; weights_proj]`` projection by the batch-invariant matmul, the fp32 head weights by the bf16-operand
    batch-invariant matmul, the key LayerNorm, the Hadamard-rotated FP8 query, the softmax(gate + ape)-weighted pool
    compression to FP8, DeepGEMM's FP8 MQA logits, the deterministic top ``index_topk // index_kpool`` pools, and
    their expansion to tokens with the incomplete pool's tail appended.

    A decoded token gets the same row: the sampler's decode step compresses each pool with the same arithmetic on
    the same bf16 keys and scores, and its paged FP8 MQA logits equal the prefill kernel's.
    """
    import torch.nn.functional as F
    from vllm.models.glm5next.nvidia.ops import kpool_compress as kpool_ops
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    from arctic_platform.inference.vllm.batch_invariant_fp32_logits import matmul_fp32_out
    from arctic_platform.inference.vllm.glm5_batch_invariance import deterministic_topk

    topk_tokens, kpool = config.index_topk, config.index_kpool
    query_positions = positions if queries is None else positions[queries]
    out = causal_topk(query_positions, width)
    long_rows = torch.nonzero(query_positions >= topk_tokens).flatten()
    if long_rows.numel() == 0:
        return out
    if getattr(config, "index_rope_head_dim", 0) or getattr(config, "qk_rope_head_dim", 0):
        raise NotImplementedError("vLLM-aligned sparse indexer covers mla_use_nope (no rotary embedding)")
    tokens = x.shape[0]
    head_dim, n_head = indexer.wk.weight.shape[0], indexer.weights_proj.weight.shape[0]
    x_q, q_c = (x, q_c) if queries is None else (x[queries], q_c[queries])
    rows_q = q_c.shape[0]

    q = ops.linear_kernel(q_c, indexer.wq_b.weight).view(rows_q, n_head, head_dim)
    kw = ops.linear_kernel(x, torch.cat([indexer.wk.weight, indexer.weights_proj.weight]))
    weights = matmul_fp32_out(x_q, indexer.weights_proj.weight.t().contiguous())
    k = F.layer_norm(kw[:, :head_dim].float(), (head_dim,), indexer.k_norm.weight.float(),
                     indexer.k_norm.bias.float(), INDEXER_K_NORM_EPS).to(kw.dtype)  # fmt: skip
    q_fp8, q_scale = kpool_ops.fwht128_quant_fp8(q.reshape(-1, head_dim))
    q_fp8, q_scale = q_fp8.view(rows_q, n_head, head_dim), q_scale.view(rows_q, n_head, 1)
    weights = (weights.unsqueeze(-1) * q_scale * (head_dim**-0.5 * n_head**-0.5)).squeeze(-1)
    gate_score = F.linear(x, indexer.index_kpool_compress_gate)
    if n_head < INDEXER_PADDED_HEADS:
        pad = INDEXER_PADDED_HEADS - n_head
        q_fp8 = torch.cat([q_fp8, q_fp8.new_zeros(rows_q, pad, head_dim)], dim=1)
        weights = torch.cat([weights, weights.new_zeros(rows_q, pad)], dim=1)

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    key_index, key_cu = keys if keys is not None else (torch.arange(tokens, device=x.device), cu_seqlens)
    key_starts, key_lengths = key_cu[:-1].long(), (key_cu[1:] - key_cu[:-1]).long()
    pools = key_lengths // kpool
    pool_starts = torch.cumsum(pools, 0) - pools
    seq = torch.repeat_interleave(torch.arange(pools.numel(), device=x.device), pools)
    first = key_starts[seq] + kpool * (torch.arange(int(pools.sum()), device=x.device) - pool_starts[seq])
    slots = key_index[first[:, None] + torch.arange(kpool, device=x.device)[None, :]]
    k_pool, k_scale = kpool_ops.kpool_compress_and_write_cache(
        torch.empty((1, 64, head_dim + 4), dtype=torch.uint8, device=x.device),
        k[slots], gate_score[slots], indexer.index_kpool_compress_ape.float(),
        torch.zeros(slots.shape[0], dtype=torch.int64, device=x.device), pool_size=kpool, head_dim=head_dim,
        round_scale=True, return_compressed=True, write_cache=False,
    )  # fmt: skip

    long_tokens = long_rows if queries is None else queries[long_rows]
    seq_of_token = torch.repeat_interleave(torch.arange(lengths.numel(), device=x.device), lengths)[long_tokens]
    ks = pool_starts[seq_of_token].to(torch.int32)
    ke = ks + ((query_positions[long_rows] + 1) // kpool).to(torch.int32)
    logits = fp8_fp4_mqa_logits((q_fp8[long_rows], None), (k_pool, k_scale), weights[long_rows], ks, ke,
                                clean_logits=False)  # fmt: skip
    pool_topk = torch.empty((long_rows.numel(), topk_tokens // kpool), dtype=torch.int32, device=x.device)
    deterministic_topk(logits, ks, ke, topk_tokens // kpool, pool_topk)
    expanded = kpool_ops.expand_pools_and_append_tail(pool_topk.to(torch.int64),
                                                      (query_positions[long_rows] + 1).to(torch.int32), kpool)  # fmt: skip
    rows = torch.full((long_rows.numel(), width), -1, dtype=torch.int32, device=x.device)
    rows[:, : expanded.shape[-1]] = expanded.to(torch.int32)
    out[long_rows] = rows
    return out


def mla_kv_up_weights(kv_b_proj: Any, config: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's absorbed MLA weights from ``kv_b_proj``: ``W_UK_T [N, P, L]`` and ``W_UV [N, L, V]``."""
    heads, latent = config.num_attention_heads, config.kv_lora_rank
    nope, value = config.qk_nope_head_dim, config.v_head_dim
    weight = kv_b_proj.weight.t().reshape(latent, heads, nope + value)
    w_uk, w_uv = weight.split([nope, value], dim=-1)
    return w_uk.permute(1, 2, 0).contiguous(), w_uv.transpose(0, 1).contiguous()


def dequant_fp8_block(weight: torch.Tensor, scale_inv: torch.Tensor, block: int = FP8_BLOCK) -> torch.Tensor:
    """vLLM's load-time dequantization of a block-FP8 weight it keeps in bf16 (``_dequant_fp8_block``)."""
    rows, cols = weight.shape
    padded = torch.nn.functional.pad(weight, (0, (-cols) % block, 0, (-rows) % block))
    scale = scale_inv.to(torch.float32).repeat_interleave(block, 0).repeat_interleave(block, 1)
    return (padded.to(torch.float32) * scale).to(torch.bfloat16)[:rows, :cols].contiguous()


def base_linear(module: Any) -> Any:
    """The projection under a PEFT LoRA wrapper, or the projection itself."""
    return getattr(module, "base_layer", module)


def bf16_weight(linear: Any) -> torch.Tensor:
    """A projection's weight as vLLM serves it in bf16: dequantized if the trainer holds it in block FP8."""
    linear = base_linear(linear)
    scale_inv = getattr(linear, "weight_scale_inv", None)
    if linear.weight.element_size() == 1 and scale_inv is not None:
        return dequant_fp8_block(linear.weight, scale_inv)
    return linear.weight


def lora_factors(module: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    """A projection's active PEFT LoRA factors as vLLM holds the synced adapter: ``A`` and ``B`` cast to bf16, then
    ``B`` multiplied by ``lora_alpha / r`` (``LoRALayerWeights.optimize``)."""
    lora_a = getattr(module, "lora_A", None)
    if not lora_a or getattr(module, "disable_adapters", False) or getattr(module, "merged", False):
        return None
    names = [name for name in module.active_adapters if name in lora_a]
    if not names:
        return None
    if len(names) > 1:
        raise NotImplementedError("vLLM serves one LoRA adapter at a time")
    a = lora_a[names[0]].weight.to(torch.bfloat16)
    b = module.lora_B[names[0]].weight.to(torch.bfloat16)
    scaling = module.scaling[names[0]]
    return a, (b if scaling == 1 else b * scaling)


def add_lora(y: torch.Tensor, x: torch.Tensor, adapters: list, output_slices: tuple[int, ...]) -> torch.Tensor:
    """vLLM's ``add_lora_linear`` with every token on one adapter: the shrink ``x @ A.T`` into an fp32
    ``[slices, tokens, r]`` buffer, then the expand adding ``buffer @ B.T`` onto each output slice of ``y`` in place.

    A slice without an adapter, or with a ``B`` narrower than the slice (``kv_a`` against its zero rope rows), gets
    zero rows, as in vLLM's stacked buffers.
    """
    present = [f for f in adapters if f is not None]
    if not present or x.shape[0] == 0:
        return y
    rank = present[0][0].shape[0]
    if any(a.shape[0] != rank for a, _ in present):
        raise NotImplementedError("the vLLM-aligned LoRA covers one rank across a merged projection")
    factors = []
    for pair, width in zip(adapters, output_slices):
        a, b = pair if pair is not None else (x.new_zeros(rank, x.shape[1]), x.new_zeros(width, rank))
        if b.shape[0] < width:
            b = torch.cat([b, b.new_zeros(width - b.shape[0], rank)])
        factors += [a, b]
    if _DIFFERENTIABLE.get():
        return _op(_add_lora_kernel, "ref_add_lora", y, x, *factors, slices=tuple(output_slices))
    return _add_lora_kernel(y, x, *factors, out=y)


def _add_lora_kernel(y: torch.Tensor, x: torch.Tensor, *factors: torch.Tensor, out: torch.Tensor | None = None,
                     **_) -> torch.Tensor:  # fmt: skip
    """The shrink and expand of :func:`add_lora` (``factors`` = A0, B0, A1, B1, ...) adding onto ``out`` (a copy of
    ``y`` by default)."""
    from vllm.lora.ops.triton_ops import LoRAKernelMeta
    from vllm.lora.ops.triton_ops import lora_expand
    from vllm.lora.ops.triton_ops import lora_shrink
    from vllm.lora.ops.triton_ops import utils as lora_utils

    out = y.clone() if out is None else out
    tokens, rank = x.shape[0], factors[0].shape[0]
    a_stack = [a.unsqueeze(0).contiguous() for a in factors[0::2]]
    b_stack = [b.unsqueeze(0).contiguous() for b in factors[1::2]]
    # vLLM caches each weight list's pointers and strides by ``data_ptr`` alone; these tensors are new every call.
    lora_utils._LORA_A_PTR_DICT.clear()
    lora_utils._LORA_B_PTR_DICT.clear()
    meta = LoRAKernelMeta.make(1, tokens, x.device)
    meta.prepare_tensors(torch.zeros(tokens, dtype=torch.int32, device=x.device))
    buffer = torch.empty((len(a_stack), tokens, rank), dtype=torch.float32, device=x.device)
    lora_shrink(x.contiguous(), a_stack, buffer, *meta.meta_args(tokens, False), 1.0)
    lora_expand(buffer, b_stack, out, *meta.meta_args(tokens, False), offset_start=0, add_inputs=True)
    return out


def lora_linear(x: torch.Tensor, module: Any, weight: torch.Tensor | None = None) -> torch.Tensor:
    """A projection as vLLM serves it with the synced adapter: the batch-invariant bf16 matmul, then the LoRA delta."""
    y = _linear(x, bf16_weight(module) if weight is None else weight)
    return add_lora(y, x, [lora_factors(module)], (y.shape[-1],))


def mla_attention(attn: Any, x: torch.Tensor, positions: torch.Tensor, cu_seqlens: torch.Tensor,
                  config: Any, record: dict | None = None, tp: int = 1,
                  keys: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:  # fmt: skip
    """:func:`mla_partials` of every tensor-parallel rank, all-reduced as vLLM reduces them (:func:`tp_sum`)."""
    return tp_sum(mla_partials(attn, x, positions, cu_seqlens, config, record, tp, keys, range(tp)))


def mla_partials(attn: Any, x: torch.Tensor, positions: torch.Tensor, cu_seqlens: torch.Tensor,
                 config: Any, record: dict | None, tp: int, keys: tuple[torch.Tensor, torch.Tensor] | None,
                 ranks, queries: torch.Tensor | None = None) -> list[torch.Tensor]:  # fmt: skip
    """vLLM's ``Glm5NextMLAAttention`` (sparse MLA with ``mla_use_nope``) on packed requests.

    Each token attends to the rows :func:`indexer_topk` selects (every earlier token below ``index_topk``, the
    indexer's pools from there on); ``record`` keeps them for the backward's stock recompute. With ``mla_use_nope``
    vLLM skips the rotary embedding and pads ``kv_a_proj_with_mqa`` with
    zero rows up to ``kv_lora_rank + qk_rope_head_dim``, so the rope part of every key is zero. The block-FP8
    ``q_a``/``kv_a``/``q_b``/``o`` projections are dequantized to bf16 as vLLM does at load. Then: the merged
    ``[q_a; kv_a]`` projection, the fused q/kv RMSNorm, ``q_b_proj``, the query absorbed into the latent space by
    ``W_UK_T``, FA3 sparse attention over one-token pages of ``[kv_c | k_pe]`` rows (QV form), the value
    up-projection by ``W_UV`` and ``o_proj``. A LoRA adapter on ``q_a``/``kv_a``/``q_b``/``o`` adds its delta with
    vLLM's kernels (:func:`add_lora`).

    At ``tp`` > 1 the replicated ``[q_a; kv_a]``, norms and indexer run once; each tensor-parallel rank of ``ranks``
    then runs its heads (its rows of ``q_b_proj`` and ``kv_b_proj``, its input columns of ``o_proj``) into its
    ``o_proj`` partial (:func:`mla_attention` all-reduces them). ``keys`` maps each request's key positions to packed
    tokens (:func:`indexer_topk`). ``queries`` (packed token indices) limits the partials, and the top-k ``record``
    keeps, to those tokens' rows; every token is still a key. Each row has the bits of the full call: the per-query
    kernels' tiles do not depend on how many rows a call has.
    """
    if not getattr(config, "mla_use_nope", True) and config.qk_rope_head_dim:
        raise NotImplementedError("vLLM-aligned sparse MLA covers mla_use_nope (no rotary embedding)")
    starts, lengths = cu_seqlens[:-1], cu_seqlens[1:] - cu_seqlens[:-1]
    heads, latent, rope = config.num_attention_heads, config.kv_lora_rank, config.qk_rope_head_dim
    nope, value = config.qk_nope_head_dim, config.v_head_dim
    tokens = x.shape[0]

    kv_a = bf16_weight(attn.kv_a_proj_with_mqa)
    if kv_a.shape[0] == latent and rope:
        kv_a = torch.cat([kv_a, kv_a.new_zeros((rope, kv_a.shape[1]))])
    qkv_lora = _linear(x, torch.cat([bf16_weight(attn.q_a_proj), kv_a]))
    qkv_lora = add_lora(qkv_lora, x, [lora_factors(attn.q_a_proj), lora_factors(attn.kv_a_proj_with_mqa)],
                        (config.q_lora_rank, latent + rope))  # fmt: skip
    q_c, kv_lora = qkv_lora.split([config.q_lora_rank, latent + rope], dim=-1)
    kv_c, k_pe = kv_lora.split([latent, rope], dim=-1)
    q_c, kv_c = _op(_q_kv_norm_kernel, "ref_q_kv_rmsnorm", q_c, kv_c, attn.q_a_layernorm.weight,
                    attn.kv_a_layernorm.weight, eps=attn.q_a_layernorm.variance_epsilon)  # fmt: skip

    seq_of_token = torch.repeat_interleave(torch.arange(starts.numel(), device=x.device), lengths)
    with torch.no_grad():  # the selection is discrete: no gradient through the indexer
        topk = indexer_topk(attn.indexer, x, q_c, positions, cu_seqlens, config, topk_buffer_width(config), keys,
                            queries)  # fmt: skip
    if queries is not None:
        q_c, seq_of_token, tokens = q_c[queries], seq_of_token[queries], queries.numel()
    if keys is None:
        rows = torch.where(topk >= 0, topk + starts.to(torch.int32)[seq_of_token, None], topk)
    else:
        key_index, key_cu = keys
        at = (key_cu[:-1].long()[seq_of_token, None] + topk.long()).clamp(0, key_index.numel() - 1)
        rows = torch.where(topk >= 0, key_index[at].to(torch.int32), topk)
    if record is not None:
        record["topk"] = rows
    valid = (topk >= 0).sum(-1).to(torch.int32)
    kv_rows = torch.cat([kv_c, k_pe], dim=-1)
    v_cache = kv_rows[:, :latent].unsqueeze(1).unsqueeze(1)

    w_uk_t_all, w_uv_all = mla_kv_up_weights(attn.kv_b_proj, config)
    q_b, q_b_lora = bf16_weight(attn.q_b_proj), lora_factors(attn.q_b_proj)
    o, o_lora = bf16_weight(attn.o_proj), lora_factors(attn.o_proj)
    heads_r = heads // tp
    q_ropes, q_latents = [], []
    for rank in ranks:
        q_b_r = tp_shard(q_b, tp, rank)
        q = add_lora(_linear(q_c, q_b_r), q_c, [_column_lora(q_b_lora, tp, rank)], (q_b_r.shape[0],))
        q = q.view(tokens, heads_r, nope + rope)
        q_nope, q_pe = q.split([nope, rope], dim=-1)
        q_latents.append(_op(_head_bmm_kernel, "ref_head_bmm", q_nope.transpose(0, 1), tp_shard(w_uk_t_all, tp, rank)))
        q_ropes.append(q_pe if rope else q.new_zeros((tokens, heads_r, 64)))
    if rope:
        k_cache = kv_rows[:, latent:].unsqueeze(1).unsqueeze(1)
    else:  # FA3's QV path needs a 64-wide Q/K: a zero query against the first 64 latent columns
        k_cache = v_cache[..., :64]
    # One call over every rank's heads: FA3 gives each head the bits of the rank's own call (heads are independent),
    # and the reference's FlashMLA pads a call's heads to 64, so per-rank calls of a few heads would each pay for 64.
    outs = _op(_sparse_mla_kernel, "ref_sparse_mla", torch.cat(q_ropes, 1), k_cache, v_cache, torch.cat(q_latents, 1),
               rows, valid, softmax_scale=(nope + rope) ** -0.5)  # fmt: skip
    partials = []
    for i, rank in enumerate(ranks):
        out = outs[:, i * heads_r : (i + 1) * heads_r].contiguous()
        up = _op(_head_bmm_kernel, "ref_head_bmm", out.transpose(0, 1), tp_shard(w_uv_all, tp, rank))
        up = up.view(tokens, heads_r * value)
        o_r = tp_shard(o, tp, rank, 1).contiguous()
        partials.append(add_lora(_linear(up, o_r), up, [_row_lora(o_lora, tp, rank)], (o_r.shape[0],)))
    return partials


def _q_kv_norm_kernel(q: torch.Tensor, kv: torch.Tensor, q_weight: torch.Tensor, kv_weight: torch.Tensor, *,
                      eps: float) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    from vllm.models.common.ops import fused_q_kv_rmsnorm

    return fused_q_kv_rmsnorm(q, kv, q_weight, kv_weight, eps)


def _head_bmm_kernel(a: torch.Tensor, b: torch.Tensor, **_) -> torch.Tensor:
    """``a [heads, tokens, i] @ b [heads, i, o]`` written into a ``[tokens, heads, o]`` tensor, as vLLM's absorbed
    MLA writes its per-head matmuls."""
    out = a.new_empty((a.shape[1], a.shape[0], b.shape[2]))
    torch.bmm(a, b, out=out.transpose(0, 1))
    return out


def _sparse_mla_kernel(q_rope, k_cache, v_cache, q_v, rows, valid, *, softmax_scale: float, **_) -> torch.Tensor:
    from vllm.v1.attention.backends.mla import flashattn_mla_sparse

    tokens = q_v.shape[0]
    return flashattn_mla_sparse.flash_attn_varlen_func(
        q=q_rope,
        k=k_cache,
        v=v_cache,
        q_v=q_v,
        max_seqlen_q=1,
        cu_seqlens_q=torch.arange(tokens + 1, dtype=torch.int32, device=q_v.device),
        max_seqlen_k=rows.shape[1],
        seqused_k=valid,
        block_table=rows,
        softmax_scale=softmax_scale,
        causal=True,
        fa_version=3,
    )


def _gather_tokens(group: Any, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """The sequence-parallel ranks' token shards of ``x`` along ``dim``, concatenated in rank order (no gradient)."""
    dtype = x.dtype
    x = x.to(torch.uint8) if dtype == torch.bool else x.contiguous()
    parts = [torch.empty_like(x) for _ in range(torch.distributed.get_world_size(group))]
    torch.distributed.all_gather(parts, x, group=group)
    return torch.cat(parts, dim=dim).to(dtype)


def _gather_rows(group: Any, x: torch.Tensor, chunk: int, total: int) -> torch.Tensor:
    """The ranks' rows of ``x`` (each at most ``chunk``, padded to it for the collective) concatenated in rank order,
    ``total`` rows. With gradients on (the per-op backward) the gather is autograd-aware: its backward sums the ranks'
    gradients and hands each rank its rows' (``gather_sequence``)."""
    from arctic_platform.model.implementations.gpu.sp.collectives import gather_sequence

    if x.shape[0] < chunk:
        x = torch.nn.functional.pad(x, [0, 0] * (x.dim() - 1) + [0, chunk - x.shape[0]])
    if torch.is_grad_enabled():
        return gather_sequence(x.unsqueeze(0), group)[0][:total]
    return _gather_tokens(group, x)[:total]


def _cu_from_positions(positions: torch.Tensor) -> torch.Tensor:
    """Packed request boundaries from per-row positions (each request restarts at 0; a pad token is its own)."""
    pos = positions.reshape(-1)
    starts = torch.nonzero(pos == 0).flatten()
    if starts.numel() == 0 or int(starts[0]) != 0:
        starts = torch.cat([starts.new_zeros(1), starts])
    return torch.cat([starts, starts.new_tensor([pos.numel()])]).to(torch.int32)


def _sp_tp_split(sp: int, tp: int) -> tuple[Callable[[int], list[int]], Callable[[int, int], int]]:
    """How sequence-parallel ranks share a mixer's tensor-parallel work: ``owned(s)``, the sampler TP ranks whose
    partials SP rank ``s`` computes (over the whole sequence), and ``source(r, o)``, the SP rank that hands TP rank
    ``r``'s partial to the owner of token shard ``o``. With ``sp`` dividing ``tp`` each SP rank computes ``tp / sp``
    ranks; with ``tp`` dividing ``sp`` each TP rank is computed by ``sp / tp`` SP ranks, which split the owners;
    otherwise every SP rank computes every rank."""
    if sp <= tp and tp % sp == 0:
        per = tp // sp
        return (lambda s: list(range(s * per, (s + 1) * per))), (lambda r, o: r // per)
    if sp > tp and sp % tp == 0:
        copies = sp // tp
        return (lambda s: [s // copies]), (lambda r, o: r * copies + o % copies)
    return (lambda s: list(range(tp))), (lambda r, o: o)


class _AllToAllRows(torch.autograd.Function):
    """``all_to_all_single`` of ``[rows, ...]`` blocks with ``send_rows[i]`` rows to rank ``i`` and ``recv_rows[i]``
    from it; the backward sends each gradient row back the way it came."""

    @staticmethod
    def forward(ctx, group, flat: torch.Tensor, send_rows: list[int], recv_rows: list[int]) -> torch.Tensor:
        ctx.group, ctx.send_rows, ctx.recv_rows = group, send_rows, recv_rows
        out = flat.new_empty((sum(recv_rows), *flat.shape[1:]))
        torch.distributed.all_to_all_single(out, flat.contiguous(), recv_rows, send_rows, group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = grad.new_empty((sum(ctx.send_rows), *grad.shape[1:]))
        torch.distributed.all_to_all_single(out, grad.contiguous(), ctx.send_rows, ctx.recv_rows, group=ctx.group)
        return None, out, None, None


def _pieces(spans: list[tuple[int, int]], lo: int, hi: int) -> list[tuple[int, int, int]]:
    """The parts of ``spans`` (sorted token intervals whose rows are concatenated) inside ``[lo, hi)``: each part's
    row offset in the concatenation, first token and length."""
    out, offset = [], 0
    for a, b in spans:
        first, last = max(a, lo), min(b, hi)
        if first < last:
            out.append((offset + first - a, first, last - first))
        offset += b - a
    return out


def _route_partials(group: Any, partials: dict[int, torch.Tensor], owned: Callable[[int], list[int]],
                    spans_of: Callable[[int], list[tuple[int, int]]], tp: int, chunk: int,
                    shard: slice) -> torch.Tensor:  # fmt: skip
    """Each token shard's owner receives every TP rank's partial rows for its tokens (one all-to-all) and adds them
    as vLLM's all-reduce does (:func:`tp_sum`, rank order): the owner's rows of the all-reduced mixer output. SP rank
    ``s`` holds the partials of its TP ranks ``owned(s)`` at the tokens ``spans_of(s)``; the SP ranks holding one TP
    rank cover every token once between them."""
    size, me = torch.distributed.get_world_size(group), torch.distributed.get_rank(group)
    sample = next(iter(partials.values()))
    send, send_rows = [], []
    for owner in range(size):
        parts = _pieces(spans_of(me), owner * chunk, (owner + 1) * chunk)
        send += [partials[r][offset : offset + n] for r in sorted(partials) for offset, _, n in parts]
        send_rows.append(len(partials) * sum(n for _, _, n in parts))
    arriving, recv_rows = [], []
    for s in range(size):
        parts = _pieces(spans_of(s), shard.start, shard.stop)
        arriving += [(r, first, n) for r in sorted(owned(s)) for _, first, n in parts]
        recv_rows.append(len(owned(s)) * sum(n for _, _, n in parts))
    flat = torch.cat(send) if send else sample[:0]
    received = _AllToAllRows.apply(group, flat, send_rows, recv_rows)
    by_rank: dict[int, list] = {r: [] for r in range(tp)}
    offset = 0
    for r, first, n in arriving:
        by_rank[r].append((first, received[offset : offset + n]))
        offset += n
    pieces = []
    for r in range(tp):
        parts = [rows for _, rows in sorted(by_rank[r], key=lambda part: part[0])]
        if sum(rows.shape[0] for rows in parts) != shard.stop - shard.start:
            raise RuntimeError(f"TP rank {r}'s partials do not cover tokens {shard.start}-{shard.stop} once")
        pieces.append(torch.cat(parts) if parts else received[:0])
    return tp_sum(pieces)


def _sp_unit_bounds(ctx: Any) -> list[int]:
    """The packed-token boundaries of the pieces whose KDA recurrences are independent: the requests, or with a turn
    plan its rows (a span resumes only from its own row's)."""
    cache = ctx.sp_cache
    if "units" not in cache:
        turns = getattr(ctx, "turns", None)
        cache["units"] = list(turns.row_bounds) if turns is not None else ctx.cu_seqlens.tolist()
    return cache["units"]


def _sp_spans(ctx: Any, size: int, kda: bool) -> Callable[[int], list[tuple[int, int]]]:
    """The packed-token intervals at which each sequence-parallel rank computes its TP ranks' mixer partials
    (:func:`_route_partials`): the shards of the owners it hands them to (:func:`_sp_tp_split`). Where several SP ranks
    compute one TP rank KDA instead cuts the sequence between them at the request (turn-plan row) boundaries nearest
    even shares, so that each runs the recurrences of its own requests."""
    key = ("spans", kda)
    if key not in ctx.sp_cache:
        tp, chunk, tokens = getattr(ctx, "tp", 1), ctx.sp_chunk, ctx.sp_tokens
        owned, source = _sp_tp_split(size, tp)
        if kda and size > tp and size % tp == 0 and os.environ.get(SP_QUERIES_ENV, "1") != "0":
            copies, bounds = size // tp, _sp_unit_bounds(ctx)
            cuts = [0] + [min(bounds, key=lambda b: abs(b - tokens * c / copies)) for c in range(1, copies)] + [tokens]
            cuts = [max(cuts[: i + 1]) for i in range(len(cuts))]
            table = [[(cuts[s % copies], cuts[s % copies + 1])] if cuts[s % copies] < cuts[s % copies + 1] else []
                     for s in range(size)]  # fmt: skip
        else:
            table = []
            for s in range(size):
                spans: list[list[int]] = []
                for o in range(size):
                    lo, hi = min(o * chunk, tokens), min((o + 1) * chunk, tokens)
                    if lo == hi or not any(source(r, o) == s for r in owned(s)):
                        continue
                    if spans and spans[-1][1] == lo:
                        spans[-1][1] = hi
                    else:
                        spans.append([lo, hi])
                table.append([tuple(span) for span in spans])
        ctx.sp_cache[key] = table
    return ctx.sp_cache[key].__getitem__


def _sequence_parallel_mixer(layer: Any, x: torch.Tensor, ctx: Any) -> torch.Tensor:
    """The mixer of a sequence-parallel rank's token shard. The layer input is gathered from every rank; the rank
    computes the ``o_proj`` partials of its share of the sampler's tensor-parallel ranks (:func:`_sp_tp_split`) with
    the same calls as without sequence parallelism, at the tokens it hands them on for (:func:`_sp_spans`: MLA's
    queries, KDA's requests); one all-to-all hands each shard's owner every rank's partial rows, which it adds in rank
    order, so the output has the same bits. The sparse top-k rows the backward's stock recompute replays are the
    shard's queries' (global key indices, as the stock context-parallel attention reads them). Differentiated (the
    per-op backward), the gather and the all-to-all carry the gradients back to the ranks that computed each part."""
    import types

    group = ctx.sp_group
    size, me, tp = torch.distributed.get_world_size(group), torch.distributed.get_rank(group), getattr(ctx, "tp", 1)
    owned, _ = _sp_tp_split(size, tp)
    mine = owned(me)
    if getattr(ctx, "sp_cache", None) is None:
        ctx.sp_cache = {}
    spans_of = _sp_spans(ctx, size, layer.block_type == "linear_attention")
    record = _record(ctx, layer)
    whole = types.SimpleNamespace(**{**vars(ctx), "sp_group": None, "replay": None if record is None else {}})
    with _phase("sp.gather"):
        x = _gather_rows(group, x, ctx.sp_chunk, ctx.sp_tokens)
    with _phase(f"sp.partials.{layer.block_type}"):
        partials = _mixer_partials(layer, x, whole, mine, spans_of(me))
    if record is not None:
        rows = whole.replay.get(id(layer), {}).get("topk")
        if rows is not None:
            parts = _pieces(spans_of(me), ctx.shard.start, ctx.shard.stop)
            start = parts[0][0] if parts else 0
            record["topk"] = rows[start : start + ctx.shard.stop - ctx.shard.start]
    with _phase("sp.route"):
        return _route_partials(group, dict(zip(mine, partials)), owned, spans_of, tp, ctx.sp_chunk, ctx.shard)


def _mixer_partials(layer: Any, x: torch.Tensor, ctx: Any, ranks,
                    spans: list[tuple[int, int]] | None = None) -> list[torch.Tensor]:  # fmt: skip
    """The mixer's ``o_proj`` partials of the sampler's tensor-parallel ranks ``ranks``, before the all-reduce, at the
    packed tokens ``spans`` (sorted intervals, their rows concatenated; every token by default). MLA runs its queries
    there; KDA runs the requests covering them (:func:`_kda_partials_between`)."""
    tp, turns, tokens = getattr(ctx, "tp", 1), getattr(ctx, "turns", None), x.shape[0]
    if spans == [(0, tokens)]:
        spans = None
    if spans is not None and not spans:
        return [x[:0] * 0 for _ in ranks]  # keeps this rank's gather and all-to-all in the backward
    queries = None if spans is None else torch.cat([torch.arange(a, b) for a, b in spans]).to(x.device)
    restrict = queries is not None and os.environ.get(SP_QUERIES_ENV, "1") != "0"
    if layer.block_type == "linear_attention":
        lo, hi = 0, tokens
        if restrict:
            bounds = _sp_unit_bounds(ctx)
            lo, hi = max(b for b in bounds if b <= spans[0][0]), min(b for b in bounds if b >= spans[-1][1])
        if (lo, hi) == (0, tokens):
            partials = kda_partials(layer.self_attn, x, ctx.cu_seqlens, ctx.config, getattr(ctx, "prompt_lens", None),
                                    None, tp, getattr(ctx, "prefill_chunk", 0), turns and turns.kda, ranks)  # fmt: skip
        else:
            partials = _kda_partials_between(layer.self_attn, x, ctx, ranks, lo, hi)
        if queries is None or spans == [(lo, hi)]:
            return partials
        return [p[queries - lo] for p in partials]
    record = _record(ctx, layer)
    partials = mla_partials(layer.self_attn, x, ctx.positions, ctx.cu_seqlens, ctx.config, record, tp,
                            turns and (turns.key_index, turns.key_cu), ranks, queries if restrict else None)  # fmt: skip
    if queries is None or restrict:
        return partials
    if record is not None and "topk" in record:
        record["topk"] = record["topk"][queries]
    return [p[queries] for p in partials]


def _kda_partials_between(attn: Any, x: torch.Tensor, ctx: Any, ranks, lo: int, hi: int) -> list[torch.Tensor]:
    """:func:`kda_partials` of the requests (turn-plan rows) at packed tokens ``[lo, hi)``, whole ones: the same bits
    as in the call over every request."""
    tp, turns = getattr(ctx, "tp", 1), getattr(ctx, "turns", None)
    key = ("kda", lo, hi)
    if key not in ctx.sp_cache:
        bounds = _sp_unit_bounds(ctx)
        first, last = bounds.index(lo), bounds.index(hi)
        if turns is None:
            prompt_lens = getattr(ctx, "prompt_lens", None)
            ctx.sp_cache[key] = (ctx.cu_seqlens[first : last + 1] - lo, None,
                                 None if prompt_lens is None else prompt_lens[first:last])  # fmt: skip
        else:
            j0, j1 = turns.row_spans[first], turns.row_spans[last]
            spans = [(start - lo, segments, None if resume is None else (resume[0] - j0, resume[1]))
                     for start, segments, resume in turns.kda_spans[j0:j1]]  # fmt: skip
            ctx.sp_cache[key] = (turns.cu[j0 : j1 + 1] - lo, _kda_steps_layout(spans, x.device), None)
    cu, layout, prompt_lens = ctx.sp_cache[key]
    return kda_partials(attn, x[lo:hi], cu, ctx.config, prompt_lens, None, tp, getattr(ctx, "prefill_chunk", 0),
                        layout, ranks)  # fmt: skip


def _mixer(layer: Any, x: torch.Tensor, ctx: Any) -> torch.Tensor:
    if getattr(ctx, "sp_group", None) is not None:
        return _sequence_parallel_mixer(layer, x, ctx)
    return tp_sum(_mixer_partials(layer, x, ctx, range(getattr(ctx, "tp", 1))))


def fp8_block_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale_inv: torch.Tensor) -> torch.Tensor:
    """vLLM's block-FP8 linear (``TritonFp8BlockScaledMMKernel``): per-128-group activation quant, Triton GEMM."""
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.platforms import current_platform

    x2d = x.reshape(-1, x.shape[-1])
    q, scale = fp8_utils.per_token_group_quant_fp8(
        x2d,
        group_size=FP8_BLOCK,
        column_major_scales=False,
        tma_aligned_scales=False,
        dtype=current_platform.fp8_dtype(),
        use_ue8m0=False,
    )
    out = torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
        q, weight, scale, weight_scale_inv, [FP8_BLOCK, FP8_BLOCK], x.dtype
    )
    return out.view(*x.shape[:-1], weight.shape[0])


def silu_and_mul_with_clamp(x: torch.Tensor, limit: float) -> torch.Tensor:
    """vLLM's ``SiluAndMulWithClamp``: ``silu(min(gate, limit)) * clamp(up, -limit, limit)`` on ``[gate | up]``."""
    ops._custom_ops()
    out = torch.empty(*x.shape[:-1], x.shape[-1] // 2, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul_with_clamp(out, x.contiguous(), float(limit), 1.0, 0.0)
    return out


def _merged_fp8(*linears: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's merged column-parallel FP8 weight and block scales: the pieces stacked along the output rows."""
    for linear in linears[:-1]:
        if linear.weight.shape[0] % FP8_BLOCK:
            raise ValueError(f"merging FP8 projections needs {FP8_BLOCK}-row pieces, got {linear.weight.shape[0]}")
    return torch.cat([linear.weight for linear in linears]), torch.cat([linear.weight_scale_inv for linear in linears])


def _mlp_partials(x: torch.Tensor, gate: tuple, up: tuple, down: tuple, swiglu_limit: float,
                  tp: int) -> list[torch.Tensor]:  # fmt: skip
    """vLLM's ``Glm5NextMLP`` on each tensor-parallel rank, before the all-reduce: its rows of the merged ``gate_up``,
    the clamped SwiGLU, its input columns of ``down``. ``gate``/``up``/``down`` are ``(weight, scale)``: block FP8
    (``TritonFp8BlockScaledMMKernel``), or bf16 with ``scale`` None (the batch-invariant linear on the merged
    weight, whose tile depends on its ``(N, K)``)."""
    partials = []
    for rank in range(tp):
        (gate_w, gate_s), (up_w, up_s) = (_shard(*pair, tp, rank, 0) for pair in (gate, up))
        down_w, down_s = _shard(*down, tp, rank, 1)
        if gate_s is None:
            gate_up = _linear(x, torch.cat([gate_w, up_w]))
            hidden = _op(silu_and_mul_with_clamp, "ref_silu_and_mul_with_clamp", gate_up, limit=swiglu_limit)
            partials.append(_linear(hidden, down_w))
            continue
        if gate_w.shape[0] % FP8_BLOCK:
            raise ValueError(f"merging FP8 projections needs {FP8_BLOCK}-row pieces, got {gate_w.shape[0]}")
        gate_up = _op(fp8_block_linear, "ref_fp8_linear", x, torch.cat([gate_w, up_w]), torch.cat([gate_s, up_s]))
        hidden = _op(silu_and_mul_with_clamp, "ref_silu_and_mul_with_clamp", gate_up, limit=swiglu_limit)
        partials.append(_op(fp8_block_linear, "ref_fp8_linear", hidden, down_w, down_s))
    return partials


def dense_mlp(mlp: Any, x: torch.Tensor, tp: int = 1) -> torch.Tensor:
    """vLLM's ``Glm5NextMLP``: merged ``gate_up``, the clamped SwiGLU, ``down`` (all-reduced at ``tp`` > 1)."""
    pieces = [(p.weight, getattr(p, "weight_scale_inv", None)) for p in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)]
    return tp_sum(_mlp_partials(x, *pieces, mlp.swiglu_limit, tp))


def shared_expert_partials(ffn: Any, x: torch.Tensor, swiglu_limit: float, tp: int = 1) -> list[torch.Tensor]:
    """vLLM's shared-expert ``Glm5NextMLP`` (``reduce_results=False``) on the trainer's ``BCFeedForward`` (``w1`` gate,
    ``w3`` up, ``w2`` down): each tensor-parallel rank's partial."""
    pieces = [(getattr(ffn, w), getattr(ffn, f"{w}_scale_inv", None)) for w in ("w1", "w3", "w2")]
    return _mlp_partials(x, *pieces, swiglu_limit, tp)


def shared_expert_mlp(ffn: Any, x: torch.Tensor, swiglu_limit: float) -> torch.Tensor:
    """:func:`shared_expert_partials` at TP 1."""
    return shared_expert_partials(ffn, x, swiglu_limit)[0]


def _full(tensor: torch.Tensor, num_experts: int, ep_group: Any = None) -> torch.Tensor:
    """All ``num_experts`` experts of an expert-parallel weight: a ``DTensor``'s full value, or the ranks' local
    slices (contiguous expert blocks, as DeepSpeed holds them) gathered in rank order over ``ep_group``.

    vLLM at TP 1 sums a token's top-k expert outputs in one kernel; splitting that sum over expert-parallel ranks
    would round differently, so every rank runs all of a token's experts on the gathered weights.
    """
    full_tensor = getattr(tensor, "full_tensor", None)
    if callable(full_tensor):
        return full_tensor()
    if tensor.shape[0] == num_experts:
        return tensor
    if ep_group is None:
        raise ValueError(f"the aligned MoE needs the expert-parallel group to gather {num_experts} experts")
    world = torch.distributed.get_world_size(ep_group)
    if tensor.shape[0] * world != num_experts:
        raise ValueError(f"{tensor.shape[0]} local experts on {world} ranks do not make {num_experts}")
    gathered = tensor.new_empty((num_experts, *tensor.shape[1:]))
    torch.distributed.all_gather_into_tensor(gathered, tensor.contiguous(), group=ep_group)
    return gathered


def _triton_experts(num_experts: int, top_k: int, hidden: int, intermediate: int, swiglu_limit: float, w13_scale,
                    w2_scale, device):  # fmt: skip
    """vLLM's routed-expert kernel for GLM-5.3-Flash at TP 1: ``FusedMoEKernel`` with ``TritonExperts``, block FP8, or
    bf16 when ``w13_scale`` is None (``UnquantizedFusedMoEMethod``'s quant config: the SwiGLU clamp only)."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import make_fp8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize

    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=device,
        routing_method=RoutingMethodType.DeepSeekV3,
        swiglu_limit=swiglu_limit,
    )
    quant_config = FusedMoEQuantConfig.make(gemm1_clamp_limit=swiglu_limit) if w13_scale is None else make_fp8_moe_quant_config(
        fp8_backend=Fp8MoeBackend.TRITON,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        a1_scale=None,
        a2_scale=None,
        block_shape=[FP8_BLOCK, FP8_BLOCK],
        swiglu_limit=swiglu_limit,
        gemm1_alpha=None,
        gemm1_beta=None,
        layer=None,
    )
    prepare = maybe_make_prepare_finalize(
        moe=moe_config, quant_config=quant_config, allow_new_interface=True, use_monolithic=False
    )
    return FusedMoEKernel(prepare, TritonExperts(moe_config, quant_config)), MoEActivation.SILU


def replay_mask(routes: torch.Tensor, n_experts: int) -> torch.Tensor:
    """``[tokens]``: which rows of ``routes`` (``[tokens, top_k]``) hold a captured route: distinct expert ids in
    range. DSS fills a position no capture covers with zeros and a best-effort fresh row with ``ROUTER_REPLAY_FRESH``;
    neither is a valid top-k."""
    ordered = routes.sort(dim=-1).values
    distinct = (ordered[:, 1:] != ordered[:, :-1]).all(-1) if routes.shape[-1] > 1 else torch.ones_like(routes[:, 0], dtype=torch.bool)
    return distinct & (routes >= 0).all(-1) & (routes < n_experts).all(-1)


def replayed_weights(logits: torch.Tensor, ids: torch.Tensor, config: Any) -> torch.Tensor:
    """The gate weights of the experts ``ids`` (``[tokens, top_k]``), in :func:`grouped_topk`'s arithmetic: the
    unbiased scores of those experts, renormalized by the ordered sum, then scaled. Equal to its weights, bit for bit,
    when ``ids`` are the experts it selects, in its order."""
    from arctic_platform.inference.vllm.glm5_batch_invariance import _ordered_sum

    scores = logits.sigmoid() if config.scoring_func == "sigmoid" else torch.softmax(logits, dim=-1)
    weights = scores.gather(1, ids.long())
    if config.norm_topk_prob:
        weights = weights / _ordered_sum(weights)
    if config.routed_scaling_factor != 1.0:
        weights = weights * config.routed_scaling_factor
    return weights.to(torch.float32)


def moe(mlp: Any, x: torch.Tensor, config: Any, ep_group: Any = None, record: dict | None = None,
        tp: int = 1, keep_out: bool = False, routes: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
    """vLLM's ``Glm5NextMoE`` on the trainer's ``MoE``: fp32 router logits by the batch-invariant matmul, the
    batch-invariant grouped top-k (renormalized and scaled by ``routed_scaling_factor`` there, as vLLM's router does
    when the runner does not scale the output), the Triton routed experts, then ``shared + routed`` in bf16.

    ``routes`` (``[tokens, top_k]``, router replay): each token with a captured route takes those experts, with
    their gate weights from this forward's logits (:func:`replayed_weights`); the others route themselves.

    At ``tp`` > 1 (experts on every rank, split by intermediate size) each tensor-parallel rank runs the experts on its
    shard of every expert's ``gate``/``up`` rows and ``down`` columns, adds its shared-expert partial, and the ranks'
    sums are all-reduced, as vLLM's MoE runner reduces ``shared + routed`` once (:func:`tp_sum`)."""
    from vllm.v1.worker.workspace import init_workspace_manager
    from vllm.v1.worker.workspace import is_workspace_manager_initialized

    from arctic_platform.inference.vllm.batch_invariant_fp32_logits import linear_fp32_out
    from arctic_platform.inference.vllm.glm5_batch_invariance import grouped_topk

    if not is_workspace_manager_initialized():  # the fused-MoE kernels take their scratch from vLLM's workspace
        init_workspace_manager(x.device)
    logits = _op(linear_fp32_out, "ref_linear_fp32_out", x, mlp.router.gate.weight)

    def route(logits, **_):
        return grouped_topk(x, logits, topk=config.num_experts_per_tok, renormalize=config.norm_topk_prob,
                            num_expert_group=config.n_group, topk_group=config.topk_group,
                            scoring_func=config.scoring_func, routed_scaling_factor=config.routed_scaling_factor,
                            e_score_correction_bias=mlp.expert_bias)  # fmt: skip

    with torch.no_grad():
        weights, ids = route(logits)
        if routes is not None:
            captured = replay_mask(routes, config.n_routed_experts)
            ids = torch.where(captured[:, None], routes.to(ids.dtype), ids)
            weights = torch.where(captured[:, None], replayed_weights(logits, ids, config), weights)
    if _DIFFERENTIABLE.get():
        if config.scoring_func != "sigmoid":
            raise NotImplementedError("the per-op vLLM backward covers sigmoid routing")

        def weights_of(logits, ids, **_):
            return route(logits)[0] if routes is None else replayed_weights(logits, ids, config)

        weights = _op(weights_of, "ref_router_weights", logits, ids,
                      renormalize=config.norm_topk_prob, scaling=config.routed_scaling_factor)  # fmt: skip
    if record is not None:
        record["experts"] = ids.to(torch.int64)
    if _DIFFERENTIABLE.get() and record is not None and "out" in record:
        # The per-op recompute takes the forward's exact output: re-running vLLM's routed experts would gather every
        # expert onto every rank, which the full model has no memory left for in its backward.
        shared = shared_expert_partials(mlp.shared_expert, x, config.swiglu_limit, tp)
        reference = tp_sum(shared) + stock_routed_experts(mlp, x, ids, weights).to(x.dtype)
        return substitute(reference, record.pop("out"))
    routed = _op(_routed_experts_ranks, "ref_routed_experts_ep", x, weights, ids, experts=mlp.experts, moe=mlp,
                 total=config.n_routed_experts, ep_group=ep_group, tp=tp, top_k=config.num_experts_per_tok,
                 limit=config.swiglu_limit)  # fmt: skip
    shared = shared_expert_partials(mlp.shared_expert, x, config.swiglu_limit, tp)
    out = tp_sum([s + r for s, r in zip(shared, routed)])
    if keep_out and record is not None:
        record["out"] = out.detach()
    return out


def _routed_experts_ranks(x, weights, ids, *, experts: Any, total: int, ep_group: Any, tp: int, top_k: int,
                          limit: float) -> tuple[torch.Tensor, ...]:  # fmt: skip
    """Each tensor-parallel rank's routed-expert output: vLLM's Triton experts on that rank's shard of every expert's
    ``gate``/``up`` rows and ``down`` columns, the experts gathered from every expert-parallel rank."""
    with _phase("moe.gather_experts"):
        w1, w2, w3 = (_full(t, total, ep_group) for t in (experts.w1, experts.w2, experts.w3))
        scales = [getattr(experts, f"{w}_scale_inv", None) for w in ("w1", "w2", "w3")]
        s1, s2, s3 = (None, None, None) if scales[0] is None else (_full(t, total, ep_group) for t in scales)
    out = []
    for rank in range(tp):
        (w1_r, s1_r), (w3_r, s3_r) = _shard(w1, s1, tp, rank, 1), _shard(w3, s3, tp, rank, 1)
        w2_r, s2_r = _shard(w2, s2, tp, rank, 2)
        w13 = torch.cat([w1_r, w3_r], dim=1)
        w13_scale = None if s1_r is None else torch.cat([s1_r, s3_r], dim=1)
        out.append(_routed_experts_kernel(x, weights, ids, w13, w13_scale, w2_r, s2_r, top_k=top_k, limit=limit))
    return tuple(out)


def stock_routed_experts(moe: Any, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """The stock MoE's routed experts alone (no shared expert) at the given expert ids and routing weights, through
    its own expert-parallel dispatch, with vLLM's FP8 activation quantization (:func:`_vllm_mlp_quantization`)."""
    import types as _types

    from arctic_platform.model.implementations.moe.distributed.ep_backend import uses_dispatch_ep

    ids = ids.long()
    saved = moe.shared_expert, moe.combine_replay
    moe.shared_expert, moe.combine_replay = None, None
    try:
        with _vllm_mlp_quantization(_types.SimpleNamespace(mlp=moe)):
            if uses_dispatch_ep(moe.ep_comm_backend):
                return moe._run_deepep_routed_experts(x, ids, weights)
            scores, tokens, counts = moe.reorderer(weights, ids)
            routed = moe._run_routed_experts(x, tokens, counts, scores)
            index = tokens.reshape(-1, 1).expand(-1, x.shape[-1])
            return torch.zeros_like(x, dtype=routed.dtype).scatter_add(0, index, routed)
    finally:
        moe.shared_expert, moe.combine_replay = saved


def _routed_experts_kernel(x, weights, ids, w13, w13_scale, w2, w2_scale, *, top_k: int, limit: float) -> torch.Tensor:
    """vLLM's Triton routed experts on the given expert weight shard (``w13 [E, 2I, H]``, ``w2 [E, H, I]``)."""
    num_experts, intermediate2, hidden = w13.shape
    # The kernel captures the scales in its quant config, so it is rebuilt for every call (cheap, host-side only).
    kernel, activation = _triton_experts(num_experts, top_k, hidden, intermediate2 // 2, limit, w13_scale, w2_scale,
                                         x.device)  # fmt: skip
    return kernel.apply(x.contiguous(), w13, w2, weights, ids, activation=activation, global_num_experts=num_experts,
                        expert_map=None, apply_router_weight_on_input=False)  # fmt: skip


def _mlp(layer: Any, x: torch.Tensor, ctx: Any) -> torch.Tensor:
    tp = getattr(ctx, "tp", 1)
    if hasattr(layer.mlp, "gate_proj"):
        return dense_mlp(layer.mlp, x, tp)
    routes = getattr(ctx, "routes", None)
    return moe(layer.mlp, x, ctx.config, getattr(ctx, "ep_group", None), _record(ctx, layer), tp,
               keep_out=getattr(ctx, "keep_moe_out", False),
               routes=None if routes is None else routes[:, layer._replay_layer_idx])  # fmt: skip


def _record(ctx: Any, layer: Any) -> dict | None:
    """Where the forward keeps ``layer``'s discrete choices (expert ids, sparse top-k rows) for the stock recompute."""
    replay = getattr(ctx, "replay", None)
    return None if replay is None else replay.setdefault(id(layer), {})


def aligned_layer(layer: Any, streams: torch.Tensor, config: Any, ctx: Any) -> torch.Tensor:
    """The sampler's computation of one decoder layer on ``[tokens, hc_mult, hidden]`` residual streams."""
    with _phase("hc"):
        post, comb, x = hc_pre(streams, layer.attn_hc, layer.input_layernorm, config)
    with _phase(f"mixer.{layer.block_type}"):
        mixed = _mixer(layer, x, ctx)
    if getattr(ctx, "keep_mixer_out", False) and not _DIFFERENTIABLE.get():
        _record(ctx, layer)["mixer_out"] = mixed
    with _phase("hc"):
        streams = hc_post(mixed, streams, post, comb)
        post, comb, x = hc_pre(streams, layer.ffn_hc, layer.post_attention_layernorm, config)
    with _phase("mlp"):
        out = _mlp(layer, x, ctx)
    with _phase("hc"):
        return hc_post(out, streams, post, comb)


def aligned_hidden(backbone: Any, input_ids: torch.Tensor, config: Any, ctx: Any, run_stock: Callable) -> torch.Tensor:
    """The sampler's final hidden states (after ``hc_contract`` and the final RMSNorm) for packed ``input_ids``.

    ``run_stock(layer, streams)`` runs the stock layer on ``[tokens, hc_mult, hidden]`` streams for the backward.
    """
    embeds = ops.tensor_parallel_embedding(backbone.embed_tokens(input_ids.reshape(-1)), getattr(ctx, "tp", 1))
    streams = embeds.unsqueeze(1).expand(-1, config.hc_mult, -1).contiguous()
    for layer in backbone.layers:
        streams = _AlignedLayer.apply(
            streams,
            lambda s, layer=layer: aligned_layer(layer, s, config, ctx),
            lambda s, layer=layer: run_stock(layer, s),
        )
    hidden = streams.mean(dim=1)
    return ops.rms_norm(hidden, backbone.norm.weight, backbone.norm.variance_epsilon, fused=False)


def check_supported_config(config: Any) -> None:
    for name in ("hc_mult", "hc_sinkhorn_iters", "hc_eps", "layer_types", "index_topk", "index_kpool"):
        if getattr(config, name, None) is None:
            raise ValueError(f"the vLLM-aligned GLM-5.3-Flash forward needs config.{name}")
    if getattr(config, "qk_rope_head_dim", 0) and not getattr(config, "mla_use_nope", False):
        raise ValueError("the vLLM-aligned GLM-5.3-Flash forward covers sparse MLA with mla_use_nope")


REPLAY_TOPK_ATTR = "_ap_replay_topk_indices"
# The block-FP8 projections vLLM dequantizes to bf16 at load (see ``bf16_weight``); the trainer runs them W8A8.
MLA_DEQUANTIZED_PROJECTIONS = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj")


def _bf16_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    """A ``BlockFp8Linear`` as vLLM serves it: a bf16 matmul on the dequantized weight (plus a LoRA delta)."""
    from arctic_platform.model.implementations.fp8 import original_param_and_lora_delta

    weight, delta = original_param_and_lora_delta(self, "weight")
    weight = dequant_fp8_block(weight, self.weight_scale_inv, self.block_size)
    if delta is not None:
        weight = weight + delta.to(weight.dtype)
    return torch.nn.functional.linear(x, weight.to(x.dtype))


@contextlib.contextmanager
def _stock_replay(layer: Any, record: dict, kwargs: dict, vllm_mla_weights: bool):
    """The stock layer set up to follow the aligned forward: its sparse attention takes the forward's top-k rows, its
    MLA projections (with ``vllm_mla_weights``) run as bf16 matmuls on vLLM's dequantized weights. Yields the extra
    layer kwargs (the forward's expert ids, ``routed_experts``, for router replay)."""
    import types as _types

    swapped = []
    if vllm_mla_weights and layer.block_type == "deepseek_sparse_attention":
        for name in MLA_DEQUANTIZED_PROJECTIONS:
            # Under PEFT the wrapper keeps adding the adapter's ``lora_B(lora_A(x))`` to the swapped base.
            module = base_linear(getattr(layer.self_attn, name, None))
            if module is not None and getattr(module, "weight_scale_inv", None) is not None:
                module.forward = _types.MethodType(_bf16_linear_forward, module)
                swapped.append(module)
    extra = {}
    experts = record.get("experts")  # the aligned forward's, replayed routes included
    if experts is not None:
        index = layer._replay_layer_idx
        routes = experts.new_zeros((1, experts.shape[0], index + 1, experts.shape[1]))
        routes[0, :, index] = experts
        extra["routed_experts"] = routes
    topk = record.get("topk")
    if topk is not None:
        setattr(layer.self_attn, REPLAY_TOPK_ATTR, topk.unsqueeze(0))
    try:
        yield extra
    finally:
        if topk is not None:
            delattr(layer.self_attn, REPLAY_TOPK_ATTR)
        for module in swapped:
            del module.forward


def _stock_layer_runner(backbone: Any, input_ids: torch.Tensor, position_ids: torch.Tensor, kwargs: dict,
                        replay: dict | None = None, vllm_mla_weights: bool = True) -> Callable:  # fmt: skip
    """``run_stock(layer, streams)``: the stock decoder layer on ``[tokens, hc_mult, hidden]`` streams, packed as the
    stock model forward packs them (one row, the attention-mask mapping and the packing kwargs it passes down).

    With ``replay`` (the aligned forward's records), the stock layer reuses the forward's discrete choices instead of
    making its own: its router takes the forward's expert ids (router replay) and its sparse attention the forward's
    top-k rows, so the backward differentiates the selection the forward computed. With ``vllm_mla_weights``, the
    sparse-attention projections vLLM serves dequantized run as bf16 matmuls instead of the trainer's W8A8 GEMMs.
    """
    mask = torch.ones(1, input_ids.numel(), dtype=torch.bool, device=input_ids.device)

    def run(layer: Any, streams: torch.Tensor) -> torch.Tensor:
        record = {} if replay is None else replay.get(id(layer), {})
        with _stock_replay(layer, record, kwargs, vllm_mla_weights) as extra:
            out, _ = layer(
                streams.unsqueeze(0),
                attention_mask=mask,
                position_ids=position_ids.reshape(1, -1),
                position_embeddings=None,
                input_ids=input_ids.reshape(1, -1),
                prev_topk_indices=None,
                **{**kwargs, **extra},
            )
        return out.squeeze(0)

    return run


class _Substitute(torch.autograd.Function):
    """``exact`` as the value, ``reference``'s graph for the gradient: a sublayer's output with the bits vLLM's kernels
    computed and the derivative of the same math, evaluated where vLLM's forward actually was."""

    @staticmethod
    def forward(ctx, reference: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
        ctx.dtype = reference.dtype
        return exact.reshape(reference.shape).clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad.to(ctx.dtype), None


def substitute(reference: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    return _Substitute.apply(reference, exact.detach())


class _FakeFp8(torch.autograd.Function):
    """vLLM's per-128-group dynamic FP8 activation quantization (``per_token_group_quant_fp8``: scale ``amax / 448``,
    a power of two with ``ue8m0``), dequantized, with a straight-through gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, ue8m0: bool) -> torch.Tensor:
        if x.shape[-1] % FP8_BLOCK or x.numel() == 0:
            return x.clone()
        groups = x.float().reshape(-1, x.shape[-1] // FP8_BLOCK, FP8_BLOCK)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-10) / fp8_max
        if ue8m0:
            scale = torch.exp2(torch.ceil(torch.log2(scale)))
        q = (groups / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return (q.float() * scale).reshape(x.shape).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


def fake_fp8(x: torch.Tensor, ue8m0: bool) -> torch.Tensor:
    return _FakeFp8.apply(x, ue8m0)


@contextlib.contextmanager
def _vllm_mlp_quantization(layer: Any):
    """The stock MLPs with vLLM's FP8 activation quantization (straight-through): the routed experts' two GEMM inputs
    with power-of-two scales (DeepGEMM's E8M0 mode, which vLLM's MoE quantizer follows), the dense and shared-expert
    block-FP8 linears with plain scales (``TritonFp8BlockScaledMMKernel``, ``use_ue8m0=False``)."""
    import types as _types

    from arctic_platform.model.implementations import fp8
    from arctic_platform.model.implementations.fp8 import dequantize_from_fp8_blockwise
    from arctic_platform.model.implementations.moe.layers import moe as stock_moe

    def experts_decoded(self, x, num_tokens_per_expert, weights, scales, ab, scores=None):
        if ab:
            raise NotImplementedError("the vLLM backward covers experts without LoRA")
        from arctic_platform.model.implementations.fp8 import grouped_fp8_decoded_mm

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        rows, block = int(offsets[-1]), self.fp8_block_size
        (w1, w2, w3), (s1, s2, s3) = weights, scales
        xq = fake_fp8(x, True)
        gate = grouped_fp8_decoded_mm(xq, w1, s1, offsets, rows, block)
        up = grouped_fp8_decoded_mm(xq, w3, s3, offsets, rows, block)
        hidden = fake_fp8(stock_moe._apply_swiglu(gate, up, self.swiglu_limit), True)
        out = grouped_fp8_decoded_mm(hidden, w2, s2, offsets, rows, block)
        return out if scores is None else stock_moe.apply_routing_weights(out, scores)

    def fp8_linear(x, weight, scale, block_size, delta=None):
        out = torch.nn.functional.linear(fake_fp8(x, False), dequantize_from_fp8_blockwise(weight, scale, dtype=x.dtype))
        return out if delta is None else out + torch.nn.functional.linear(x, delta.to(x.dtype))

    def block_linear(self, x):
        weight = dequantize_from_fp8_blockwise(self.weight, self.weight_scale_inv, dtype=x.dtype)
        return torch.nn.functional.linear(fake_fp8(x, False), weight)

    saved = [(stock_moe.GroupedExperts, "_forward_fp8_decoded"), (fp8, "fp8_linear_with_lora_delta")]
    originals = [getattr(owner, name) for owner, name in saved]
    stock_moe.GroupedExperts._forward_fp8_decoded = experts_decoded
    fp8.fp8_linear_with_lora_delta = fp8_linear
    swapped = [m for m in layer.mlp.modules() if isinstance(m, torch.nn.Linear)
               and getattr(m, "weight_scale_inv", None) is not None and m.weight.element_size() == 1]  # fmt: skip
    for module in swapped:
        module.forward = _types.MethodType(block_linear, module)
    try:
        yield
    finally:
        for (owner, name), original in zip(saved, originals):
            setattr(owner, name, original)
        for module in swapped:
            del module.forward


def _vllm_backward_runner(input_ids: torch.Tensor, position_ids: torch.Tensor, kwargs: dict, ctx: Any,
                          per_op: frozenset = frozenset()) -> Callable:  # fmt: skip
    """``run(layer, streams)`` for the backward: the layer as a chain of sublayers whose values are the aligned
    forward's (vLLM's kernels, recomputed) and whose gradients are the stock sublayers' made to compute vLLM's math
    (replayed expert ids and top-k rows, bf16 MLA projections, vLLM's FP8 activation quantization), each evaluated on
    the exact input vLLM's forward gave it (:func:`substitute`). Sublayers: the attention-site hyper-connection pre
    (with its norm), the mixer, the post; the FFN site's pre, the MLP or MoE, the post.

    ``per_op`` names sublayer groups (``"hc"``: both sites' pre and post; ``"mixer"``; ``"mlp"``) that take the
    per-op backward instead (vLLM's kernels under :func:`differentiable`), for comparing the two group by group."""
    from arctic_platform.model.implementations.glm53 import modeling_glm5_next as modeling

    config = ctx.config
    mask = torch.ones(1, input_ids.numel(), dtype=torch.bool, device=input_ids.device)
    # What the stock decoder layer passes on to its attention: the packing kwargs without its own named arguments.
    named = ("attention_mask", "position_ids", "past_key_values", "use_cache", "position_embeddings",
             "prev_topk_indices", "routed_experts")  # fmt: skip
    layer_kwargs = {**{k: v for k, v in kwargs.items() if k not in named}, "input_ids": input_ids.reshape(1, -1)}

    def pre(layer, hc, norm, streams):
        if "hc" in per_op:
            with differentiable():
                post, comb, x = hc_pre(streams, hc, norm, config)
            return None, post, comb, x, (post.detach(), comb.detach(), x.detach())
        with torch.no_grad():
            exact = hc_pre(streams, hc, norm, config)
        res = streams.unsqueeze(0)
        fused = modeling.fused_hyper_connection_applies(hc, res)
        post, comb, x = modeling.hyper_connection(hc, res) if fused else hc(res)
        x = norm(x)
        post, comb, x = (substitute(r, e) for r, e in zip((post, comb, x), exact))
        return fused, post, comb, x.squeeze(0), exact

    def post_(layer, fused, post, comb, out, out_exact, streams, exact):
        if "hc" in per_op:
            with differentiable():
                return hc_post(out, streams, post, comb)
        with torch.no_grad():
            value = hc_post(out_exact, streams, exact[0], exact[1])
        mixed = modeling._remix(fused, layer._fp32_parity, post, comb, out.unsqueeze(0), streams.unsqueeze(0))
        return substitute(mixed.squeeze(0), value)

    def run(layer: Any, streams: torch.Tensor) -> torch.Tensor:
        record = _record(ctx, layer) or {}
        fused, post, comb, x, exact = pre(layer, layer.attn_hc, layer.input_layernorm, streams)
        with _stock_replay(layer, record, kwargs, vllm_mla_weights=True) as extra:
            if "mixer" in per_op:
                with differentiable(), _phase(f"per_op.mixer.{layer.block_type}"):
                    mixed = _mixer(layer, x, ctx)
                mixed_exact = mixed.detach()
            else:
                mixed_exact = record.pop("mixer_out", None)  # the forward's (ctx.keep_mixer_out)
                if mixed_exact is None:
                    with torch.no_grad(), _phase(f"exact.mixer.{layer.block_type}"):
                        mixed_exact = _mixer(layer, exact[2], ctx)
                # The stock layer takes these by name before passing the rest of the packing kwargs down.
                with _phase(f"stock.mixer.{layer.block_type}"):
                    if layer.block_type == "linear_attention":
                        mixed = layer.self_attn(**{**layer_kwargs, "hidden_states": x.unsqueeze(0),
                                                   "cache_params": None, "attention_mask": mask})  # fmt: skip
                    else:
                        mixed = layer.self_attn(**{**layer_kwargs, "hidden_states": x.unsqueeze(0),
                                                   "attention_mask": mask, "position_ids": position_ids.reshape(1, -1),
                                                   "past_key_values": None, "use_cache": False,
                                                   "position_embeddings": None, "prev_topk_indices": None})[0]  # fmt: skip
                mixed = substitute(mixed.squeeze(0), mixed_exact)
            streams = post_(layer, fused, post, comb, mixed, mixed_exact, streams, exact)
            fused, post, comb, x, exact = pre(layer, layer.ffn_hc, layer.post_attention_layernorm, streams)
            if "mlp" in per_op:
                with differentiable():
                    out = _mlp(layer, x, ctx)
                return post_(layer, fused, post, comb, out, out.detach(), streams, exact)
            mlp_exact = record.pop("out", None)  # the forward's MoE output (ctx.keep_moe_out)
            if mlp_exact is None:
                with torch.no_grad(), _phase("exact.mlp"):
                    mlp_exact = _mlp(layer, exact[2], ctx)
            with _vllm_mlp_quantization(layer), _phase("stock.mlp"):
                if "routed_experts" in extra:
                    out = layer.mlp(x.unsqueeze(0), routed_experts=extra["routed_experts"][:, :, layer._replay_layer_idx])
                else:
                    out = layer.mlp(x.unsqueeze(0))
            out = substitute(out.squeeze(0), mlp_exact)
            return post_(layer, fused, post, comb, out, mlp_exact, streams, exact)

    return run


def _per_op_backward_runner(ctx: Any) -> Callable:
    """``run(layer, streams)`` for the backward: the aligned layer itself, recomputed with every vLLM kernel call
    differentiable (:func:`differentiable`). Each kernel keeps its value; its gradient is that of its reference math
    (:mod:`glm5_next_backward`) at the inputs the kernel got, so every op's derivative is taken at vLLM's values."""

    def run(layer: Any, streams: torch.Tensor) -> torch.Tensor:
        with differentiable():
            return aligned_layer(layer, streams, ctx.config, ctx)

    return run


def turn_layout(plan: torch.Tensor, cu_seqlens: torch.Tensor, *, block_size: int, device: torch.device) -> Any:
    """A packed batch with a turn plan (:mod:`~arctic_platform.model.implementations.vllm_aligned.turn_plan`, the
    ``[tokens, K, 3]`` plan of every row) laid out request by request: each row's columns (the sampler requests that
    computed it) as spans of an expanded token sequence.

    ``expand_index`` is each span token's packed position, ``positions`` its position in its row, ``cu`` the spans'
    boundaries, ``output_index`` the span token whose logprob each packed position returns (``-1`` for none: the
    first span computing the position, or the one the plan marks), ``replay_index`` the token of the first of the
    row's own spans (those returning logprobs; every span of a row where none does) computing it, ``-1`` if none
    does. A span's keys are its row's positions up to its end: before its cache hit, page by page, the tokens of the
    span that wrote the page, then its own
    (``key_index`` over the expanded tokens, ``key_cu`` its boundaries). ``kda`` replays every span's prefill
    chunks and decode runs, resuming from the state its provider had at the hit (:func:`_kda_steps_layout`, built
    from ``kda_spans``). A span resumes only from its own row's: ``row_spans`` and ``row_bounds`` are the rows' first
    spans and first span tokens (and the totals), the independent pieces of the KDA replay."""
    import types

    from arctic_platform.model.implementations.vllm_aligned.turn_plan import TurnSpan
    from arctic_platform.model.implementations.vllm_aligned.turn_plan import parse_row

    plan = plan.to("cpu", torch.long)
    bounds = cu_seqlens.to(torch.long).tolist()
    spans: list = []
    row_starts: list[int] = []
    for row, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        row_spans = parse_row(plan[lo:hi], row=row, block_size=block_size)
        if not row_spans:
            row_spans = [TurnSpan(row, 0, 0, hi - lo, ((False, 0, hi - lo),), (), None, ())]
        spans += row_spans
        row_starts += [lo] * len(row_spans)
    span_id = {(span.row, span.column): i for i, span in enumerate(spans)}
    lengths = [span.end - span.start for span in spans]
    starts = torch.cumsum(torch.tensor([0] + lengths), 0).tolist()
    expand_index = torch.cat([torch.arange(r + span.start, r + span.end) for span, r in zip(spans, row_starts)])
    positions = torch.cat([torch.arange(span.start, span.end) for span in spans])
    output_index = torch.full((bounds[-1],), -1, dtype=torch.long)
    for i in reversed(range(len(spans))):
        span, r = spans[i], row_starts[i]
        output_index[r + span.start : r + span.end] = torch.arange(starts[i], starts[i] + lengths[i])
    rows_with_outputs = {span.row for span in spans if span.outputs}
    replay_index = torch.full((bounds[-1],), -1, dtype=torch.long)
    for i in reversed(range(len(spans))):
        span, r = spans[i], row_starts[i]
        if span.outputs or span.row not in rows_with_outputs:
            replay_index[r + span.start : r + span.end] = torch.arange(starts[i], starts[i] + lengths[i])
    for i, (span, r) in enumerate(zip(spans, row_starts)):
        if span.outputs:
            outputs = torch.tensor(span.outputs)
            output_index[r + outputs] = starts[i] + outputs - span.start
    key_index = []
    for i, span in enumerate(spans):
        for page, column in enumerate(span.page_columns):
            writer = span_id[(span.row, column)]
            key_index.append(starts[writer] + page * block_size - spans[writer].start + torch.arange(block_size))
        key_index.append(torch.arange(starts[i], starts[i] + lengths[i]))
    key_lengths = torch.tensor([span.end for span in spans])
    row_spans = [0] + [i + 1 for i in range(len(spans)) if i + 1 == len(spans) or spans[i + 1].row != spans[i].row]
    kda = []
    for i, span in enumerate(spans):
        resume = None
        if span.resume_column is not None:
            provider = span_id[(span.row, span.resume_column)]
            resume = (provider, span.start - spans[provider].start)
        segments = [(decode, lo - span.start, hi - span.start) for decode, lo, hi in span.segments]
        kda.append((starts[i], segments, resume))
    return types.SimpleNamespace(
        expand_index=expand_index.to(device),
        positions=positions.to(device),
        cu=torch.tensor(starts, dtype=torch.int32, device=device),
        output_index=output_index.to(device),
        replay_index=replay_index.to(device),
        key_index=torch.cat(key_index).to(device),
        key_cu=torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(key_lengths, 0)]).to(torch.int32).to(device),
        kda=_kda_steps_layout(kda, device),
        kda_spans=kda,
        row_spans=row_spans,
        row_bounds=[starts[i] for i in row_spans],
    )


def plan_routes(routes: torch.Tensor, turns: Any) -> torch.Tensor:
    """Router-replay routes (``[tokens, layers, top_k]``, one per packed position) for the span tokens of
    :func:`turn_layout`.

    The trajectory's last capture holds one route per row position, that of the trajectory's first request to compute
    it (Arctic-Platform #473). Only that request's span replays it. Every other span gets ``-1``, which
    :func:`replay_mask` rejects, so it routes itself as the sampler's did: a later request's recompute of the position
    (a re-prefilled page tail), and another trajectory's request whose cached pages the row read, for which the
    capture holds the routes of whichever request first published those pages, not necessarily that one."""
    span_tokens = torch.arange(turns.expand_index.numel(), device=turns.expand_index.device)
    first = turns.replay_index[turns.expand_index] == span_tokens
    return torch.where(first[:, None, None], routes[turns.expand_index].to(torch.int32), -1)


def plan_block_size(plan: torch.Tensor, cu_seqlens: torch.Tensor) -> int:
    """The cache page of a packed turn plan, as far as the plan shows it: the greatest common divisor of where its
    rows' columns start and where the column a span reads from changes (both are page boundaries); the longest row
    if nothing resumes."""
    import math

    plan = plan.reshape(-1, *plan.shape[-2:]).to("cpu")
    bounds = cu_seqlens.to(torch.long).tolist()
    boundaries = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        role = plan[lo:hi, :, 0]
        for column in range(role.shape[1]):
            where = torch.nonzero(role[:, column] > 0).flatten()
            if where.numel() and int(where[0]) > 0:
                boundaries.append(int(where[0]))
            changes = torch.nonzero(role[1:, column] != role[:-1, column]).flatten() + 1
            boundaries += [int(p) for p in changes.tolist() if int(role[p - 1, column]) < 0 and int(role[p, column]) < 0]
    return math.gcd(*boundaries) if boundaries else max(hi - lo for lo, hi in zip(bounds[:-1], bounds[1:]))


def _sequence_of_tokens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    return torch.repeat_interleave(torch.arange(lengths.numel(), device=cu_seqlens.device), lengths)


def prompt_lengths(trained: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """How many leading tokens of each packed request the sampler prefilled: through its first trained position, whose
    logits come from the prefill (every token when nothing is trained); the tokens after it were decoded."""
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    seq = _sequence_of_tokens(cu_seqlens)
    offset = torch.arange(seq.numel(), device=seq.device) - cu_seqlens[:-1].long()[seq]
    first = torch.where(trained.reshape(-1).bool(), offset, lengths[seq])
    first = torch.full_like(lengths, torch.iinfo(torch.long).max).scatter_reduce(0, seq, first, "amin")
    return torch.minimum(first + 1, lengths)


def _log_memory(model: Any, tokens: int) -> None:
    """Debug (the experimental per-op backward): rank 0's CUDA memory at each aligned forward, and the live CUDA
    tensors by shape once allocated memory has grown 2 GiB past the first call's."""
    import collections
    import gc
    import logging

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    log = logging.getLogger(__name__)
    allocated = torch.cuda.memory_allocated()
    first = getattr(model, "_vllm_aligned_first_allocated", None)
    if first is None:
        model._vllm_aligned_first_allocated = first = allocated
    log.warning("vllm_aligned memory: tokens=%d allocated=%.2f GiB peak=%.2f GiB growth=%.2f GiB", tokens,
                allocated / 2**30, torch.cuda.max_memory_allocated() / 2**30, (allocated - first) / 2**30)  # fmt: skip
    torch.cuda.reset_peak_memory_stats()
    if allocated - first < 2 * 2**30:
        return
    sizes: collections.Counter = collections.Counter()
    counts: collections.Counter = collections.Counter()
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.is_cuda and not isinstance(obj, torch.nn.Parameter):
            key = (tuple(obj.shape), str(obj.dtype), obj.requires_grad)
            sizes[key] += obj.untyped_storage().nbytes()
            counts[key] += 1
    for key, nbytes in sizes.most_common(12):
        log.warning("vllm_aligned live tensors: %s x%d = %.2f GiB", key, counts[key], nbytes / 2**30)


def aligned_logprobs(model: Any, input_ids: torch.Tensor, position_ids: torch.Tensor, labels: torch.Tensor,
                     trained: torch.Tensor, cu_seqlens: torch.Tensor | None, temperature: float,
                     kwargs: dict, plan: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
    """The sampler's logprob of each ``labels`` token (``[tokens]``, fp32): each request's prompt prefilled whole
    through its first ``trained`` position, the tokens after it decoded; with a turn ``plan`` each row replayed
    request by request as the sampler computed it (:func:`turn_layout`), and ``trained`` not used."""
    import types

    backbone = model.model.language_model
    config = backbone.config
    if position_ids.dim() == 3:
        position_ids = position_ids[0]
    routes = kwargs.get("routed_experts")  # router replay: [1, tokens (or more, padded), layers, top_k]
    if routes is not None:
        routes = routes.reshape(-1, *routes.shape[-2:]).to(input_ids.device)
    mode = getattr(model, "_vllm_aligned_backward", "stock")
    group, local = getattr(model, "_vllm_aligned_sp_group", None), input_ids.numel()
    if group is not None:
        # Each rank holds a contiguous window of the packed call; concatenated in rank order they are the call.
        if routes is not None and routes.shape[0] < local * torch.distributed.get_world_size(group):
            routes = _gather_tokens(group, routes[:local])
        input_ids, position_ids, labels, trained = (_gather_tokens(group, t.reshape(1, -1), dim=1)
                                                    for t in (input_ids, position_ids, labels, trained))  # fmt: skip
        plan = None if plan is None else _gather_tokens(group, plan.reshape(1, -1, *plan.shape[-2:]), dim=1)
        cu_seqlens = _cu_from_positions(position_ids)
    layout = ops.PackedLayout.build(position_ids, block_size=1, cu_seqlens=cu_seqlens)
    cu = layout.cu_seqlens.to(torch.int32)
    if routes is not None:
        routes = routes[: input_ids.numel()]
    turns = None
    if plan is not None:
        if mode == "stock" and torch.is_grad_enabled():
            raise NotImplementedError("the stock backward cannot resume a request from another's state: with a turn "
                                      "plan use vllm_aligned_backward 'vllm' or 'vllm_op'")  # fmt: skip
        plan = plan.reshape(-1, *plan.shape[-2:])
        block_size = getattr(model, "_vllm_aligned_kv_block_size", 0) or plan_block_size(plan, cu)
        turns = turn_layout(plan, cu, block_size=block_size, device=input_ids.device)
        scored = torch.zeros(turns.expand_index.numel(), dtype=torch.bool, device=input_ids.device)
        scored[turns.output_index[turns.output_index >= 0]] = True
        input_ids = input_ids.reshape(-1)[turns.expand_index]
        routes = None if routes is None else plan_routes(routes, turns)
        position_ids = turns.positions
        labels = torch.where(scored, labels.reshape(-1)[turns.expand_index], -100)
    tokens, shard, chunk = input_ids.numel(), None, 0
    if group is not None:
        # The rank's rows for the per-token operators: its own window, or with a plan an even share of the expanded
        # spans (whose count need not divide by sp_size; the gathers pad each share to ``chunk`` rows).
        rank, size = torch.distributed.get_rank(group), torch.distributed.get_world_size(group)
        chunk = local if turns is None else -(-tokens // size)
        shard = slice(min(rank * chunk, tokens), min((rank + 1) * chunk, tokens))
        if turns is None:
            # The stock context-parallel attention (the backward's recompute) reads the global boundaries the
            # stock backbone forward would have set.
            from arctic_platform.model.implementations.glm53 import context_parallel

            for layer in backbone.layers:
                setattr(layer.self_attn, context_parallel._GLOBAL_POSITION_IDS, position_ids.reshape(1, -1))
                setattr(layer.self_attn, context_parallel._GLOBAL_CU_SEQLENS, cu)
    local_ids = input_ids if shard is None else input_ids.reshape(-1)[shard].reshape(1, -1)
    local_positions = position_ids if shard is None else position_ids.reshape(-1)[shard].reshape(1, -1)
    ctx = types.SimpleNamespace(config=config, positions=layout.positions if turns is None else turns.positions,
                                cu_seqlens=cu if turns is None else turns.cu, turns=turns,
                                routes=routes if shard is None or routes is None else routes[shard],
                                prompt_lens=None if turns is not None else prompt_lengths(trained, cu),
                                replay={} if torch.is_grad_enabled() else None,
                                ep_group=getattr(model, "_vllm_aligned_ep_group", None),
                                tp=getattr(model, "_vllm_aligned_tensor_parallel", 1),
                                prefill_chunk=getattr(model, "_vllm_aligned_prefill_chunk", 0),
                                sp_group=group, shard=shard, sp_chunk=chunk, sp_tokens=tokens)  # fmt: skip
    _log_phases(tokens)
    if mode == "vllm_op":
        _log_memory(model, input_ids.numel())
        ctx.keep_moe_out = ctx.replay is not None
        run_backward = _per_op_backward_runner(ctx)
    elif mode == "vllm" and ctx.replay is not None:
        # With a turn plan the mixers take the per-op backward: the stock KDA cannot start from another request's
        # state, and the per-op references follow the plan's spans.
        per_op = frozenset({"mixer"}) if turns is not None else frozenset()
        keep = os.environ.get(KEEP_OUTPUTS_ENV, "1" if group is not None else "0") == "1"
        ctx.keep_moe_out, ctx.keep_mixer_out = keep, keep and "mixer" not in per_op
        run_backward = _vllm_backward_runner(local_ids, local_positions, kwargs, ctx, per_op=per_op)
    else:
        run_backward = _stock_layer_runner(backbone, local_ids, local_positions, kwargs, ctx.replay)
    with _phase("layers"):
        hidden = aligned_hidden(backbone, local_ids, config, ctx, run_backward)
    with _phase("lm_head"):
        logprobs = ops.lm_head_logprobs(
            hidden,
            model.lm_head.weight,
            labels.reshape(-1) if shard is None else labels.reshape(-1)[shard],
            tensor_parallel=ctx.tp,
            temperature=temperature,
            token_chunk_size=model._vllm_aligned_token_chunk_size,
            fp32=model._vllm_aligned_lm_head_fp32,
        )
    if turns is None:
        return logprobs
    if group is not None:
        logprobs = _gather_rows(group, logprobs, chunk, tokens)
    logprobs = torch.where(turns.output_index >= 0, logprobs[turns.output_index.clamp(min=0)], 0.0)
    if group is not None:
        rank = torch.distributed.get_rank(group)
        return logprobs[rank * local : (rank + 1) * local]
    return logprobs


def _causal_lm_forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, temperature=None,
                       dss_compute_logprobs: bool = False, **kwargs):  # fmt: skip
    from arctic_platform.model.implementations.vllm_aligned.qwen3 import _scalar_temperature
    from arctic_platform.model.implementations.vllm_aligned.qwen3_5 import LOSS_MASK_KWARG
    from arctic_platform.model.implementations.vllm_aligned.turn_plan import PLAN_KWARG

    loss_mask = kwargs.pop(LOSS_MASK_KWARG, None)
    plan = kwargs.pop(PLAN_KWARG, None)
    if not dss_compute_logprobs:
        return self._vllm_aligned_stock_forward(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels, **kwargs
        )
    if labels is None or input_ids is None:
        raise ValueError("dss_compute_logprobs requires input_ids and labels")
    if position_ids is None:
        position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).expand_as(input_ids)
    labels = labels.to(input_ids.device)
    trained = loss_mask if loss_mask is not None else labels != -100
    logprobs = aligned_logprobs(
        self,
        input_ids,
        position_ids,
        labels,
        trained.to(input_ids.device),
        kwargs.get("cu_seq_lens_q"),
        _scalar_temperature(temperature),
        kwargs,
        plan=None if plan is None else plan.to(input_ids.device),
    )
    return {"logprobs": logprobs.view(input_ids.shape)}


def check_tensor_parallel(config: Any, tp: int, fp8: bool = True) -> None:
    """The sampler's tensor-parallel size has to split the heads, and the MLPs and experts (block-FP8 ones in whole
    blocks)."""
    if tp < 1:
        raise ValueError("tensor_parallel must be positive")
    if tp == 1:
        return
    for name in ("num_attention_heads", "linear_num_heads"):
        if getattr(config, name) % tp:
            raise ValueError(f"config.{name}={getattr(config, name)} does not split across {tp} ranks")
    sizes = {"intermediate_size": config.intermediate_size, "moe_intermediate_size": config.moe_intermediate_size,
             "shared expert": config.moe_intermediate_size * (config.n_shared_experts or 1)}  # fmt: skip
    block = FP8_BLOCK if fp8 else 1
    for name, size in sizes.items():
        if size % (tp * block):
            raise ValueError(f"{name} {size} does not split across {tp} ranks" + (f" in {block}-blocks" if fp8 else ""))


def apply_vllm_aligned_glm5_next(model: Any, *, token_chunk_size: int, lm_head_fp32: bool = False,
                                 ep_group: Any = None, tensor_parallel: int = 1, backward: str = "vllm",
                                 prefill_chunk: int = 0, kv_block_size: int = 0, sp_group: Any = None) -> None:  # fmt: skip
    """Make a GLM-5.3-Flash model's ``dss_compute_logprobs`` forward compute the batch-invariant sampler's logprobs.

    ``tensor_parallel`` is the sampler's TP size: every rank's share of the split layers runs on its own weight shard,
    and the partial sums are reduced as vLLM's batch-invariant all-reduce reduces them. ``backward`` is ``"vllm"``
    (each sublayer's gradient is taken at the aligned forward's exact values, :func:`_vllm_backward_runner`; about
    0.7% from the per-op backward per layer, at about a third of its time), ``"stock"`` (each layer's backward
    recomputes the stock layer on the layer's input) or ``"vllm_op"`` (each vLLM kernel's gradient is its reference
    math's at its exact inputs, :func:`_per_op_backward_runner`). ``prefill_chunk`` is the
    sampler's ``ARCTIC_PREFILL_CHUNK_ALIGN`` with a token budget of that size, so every prompt is prefilled that many
    tokens at a time from the sequence start (the KDA layers resume each chunk from the state the previous one left).
    ``kv_block_size`` is the sampler's hybrid cache page, which a turn plan's hits are multiples of (inferred from
    the plan by default).

    ``sp_group`` runs the forward sequence-parallel: each rank's contiguous window of the packed call goes through
    the per-token operators (hyper-connections, norms, MLP and MoE, LM head) on that rank, and every mixer (KDA,
    sparse MLA with its indexer) gathers its input from the group and runs on the whole sequence, so the logprobs
    are those of sp_size 1. The backward recomputes the stock context-parallel layers on the rank's window; the
    per-op backward differentiates each mixer on the whole sequence (the gather sums the ranks' gradients). With a
    turn plan the per-token operators take an even share of the expanded spans and the logprobs are gathered back.
    """
    if prefill_chunk < 0:
        raise ValueError("prefill_chunk must be non-negative")
    if backward not in ("stock", "vllm", "vllm_op"):
        raise ValueError(f"backward must be 'stock', 'vllm' or 'vllm_op', got {backward!r}")
    if sp_group is not None and torch.distributed.get_world_size(sp_group) > 1:
        model._vllm_aligned_sp_group = sp_group
    import types

    from arctic_platform.inference.vllm.glm5_batch_invariance import ensure_glm5_batch_invariance

    config = model.model.language_model.config
    check_supported_config(config)
    fp8 = any(hasattr(m, "weight_scale_inv") or hasattr(m, "w1_scale_inv") for m in model.modules())
    check_tensor_parallel(config, int(tensor_parallel), fp8=fp8)
    ops.require_batch_invariant_env()
    import os

    os.environ.setdefault("ARCTIC_GLM5_BATCH_INVARIANT", "1")
    ensure_glm5_batch_invariance()
    model._vllm_aligned_ep_group = ep_group
    model._vllm_aligned_tensor_parallel = int(tensor_parallel)
    model._vllm_aligned_backward = backward
    model._vllm_aligned_prefill_chunk = int(prefill_chunk)
    model._vllm_aligned_kv_block_size = int(kv_block_size)
    model._vllm_aligned_token_chunk_size = int(token_chunk_size)
    model._vllm_aligned_lm_head_fp32 = bool(lm_head_fp32)
    model._vllm_aligned_stock_forward = model.forward
    model.forward = types.MethodType(_causal_lm_forward, model)
    model._vllm_aligned = True
    model._vllm_aligned_takes_loss_mask = True


__all__ = ["aligned_hidden", "aligned_layer", "apply_vllm_aligned_glm5_next", "hc_post", "hc_pre"]
