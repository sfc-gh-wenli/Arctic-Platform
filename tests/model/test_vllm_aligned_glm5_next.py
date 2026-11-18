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

import torch

from arctic_platform.model.implementations.vllm_aligned.glm5_next import _AlignedLayer


def test_aligned_layer_returns_the_aligned_value_with_the_stock_gradient():
    torch.manual_seed(0)
    stock = torch.nn.Linear(6, 6)
    streams = torch.randn(5, 6, requires_grad=True)
    nudge = torch.full((5, 6), 1e-3)

    out = _AlignedLayer.apply(streams, lambda s: torch.tanh(stock(s)) + nudge, lambda s: torch.tanh(stock(s)))
    assert torch.equal(out, torch.tanh(stock(streams)).detach() + nudge)
    weights = torch.randn(5, 6)
    (out * weights).sum().backward()

    want_streams = streams.detach().clone().requires_grad_(True)
    reference = torch.nn.Linear(6, 6)
    reference.load_state_dict(stock.state_dict())
    (torch.tanh(reference(want_streams)) * weights).sum().backward()
    assert torch.equal(streams.grad, want_streams.grad)
    assert torch.equal(stock.weight.grad, reference.weight.grad)
    assert torch.equal(stock.bias.grad, reference.bias.grad)


def test_aligned_layers_chain_like_checkpointed_layers():
    torch.manual_seed(1)
    layers = [torch.nn.Linear(4, 4) for _ in range(3)]
    x = torch.randn(3, 4, requires_grad=True)
    h = x
    for layer in layers:
        h = _AlignedLayer.apply(h, lambda s, layer=layer: torch.relu(layer(s)), lambda s, layer=layer: torch.relu(layer(s)))
    h.sum().backward()
    y = x.detach().clone().requires_grad_(True)
    clones = [torch.nn.Linear(4, 4) for _ in layers]
    for clone, layer in zip(clones, layers):
        clone.load_state_dict(layer.state_dict())
    g = y
    for clone in clones:
        g = torch.relu(clone(g))
    g.sum().backward()
    assert torch.equal(x.grad, y.grad)
    for clone, layer in zip(clones, layers):
        assert torch.equal(clone.weight.grad, layer.weight.grad)


def test_merged_fp8_stacks_weights_and_block_scales():
    import types

    import pytest

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import _merged_fp8

    def linear(rows, cols, fill):
        return types.SimpleNamespace(weight=torch.full((rows, cols), fill), weight_scale_inv=torch.full((rows // 128, 1), fill))

    weight, scale = _merged_fp8(linear(256, 128, 1.0), linear(128, 128, 2.0))
    assert weight.shape == (384, 128) and scale.shape == (3, 1)
    assert scale[:, 0].tolist() == [1.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        _merged_fp8(linear(100, 128, 1.0), linear(128, 128, 2.0))


def test_causal_topk_and_buffer_width_follow_vllm():
    import types

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import causal_topk
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import topk_buffer_width

    assert topk_buffer_width(types.SimpleNamespace(index_topk=2048, index_kpool=4)) == 2176
    rows = causal_topk(torch.tensor([0, 2]), 4)
    assert rows.tolist() == [[0, -1, -1, -1], [0, 1, 2, -1]]


def test_mla_kv_up_weights_split_kv_b_proj_per_head():
    import types

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import mla_kv_up_weights

    config = types.SimpleNamespace(num_attention_heads=2, kv_lora_rank=3, qk_nope_head_dim=4, v_head_dim=5)
    weight = torch.arange(2 * 9 * 3, dtype=torch.float32).view(2 * 9, 3)
    w_uk_t, w_uv = mla_kv_up_weights(types.SimpleNamespace(weight=weight), config)
    assert w_uk_t.shape == (2, 4, 3) and w_uv.shape == (2, 3, 5)
    per_head = weight.view(2, 9, 3)
    assert torch.equal(w_uk_t[1], per_head[1, :4])
    assert torch.equal(w_uv[1], per_head[1, 4:].t())


def test_dequant_fp8_block_matches_vllm_formula_with_partial_blocks():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import dequant_fp8_block

    torch.manual_seed(2)
    weight = torch.randn(200, 130).to(torch.float8_e4m3fn)
    scale = torch.rand(2, 2) + 0.5
    got = dequant_fp8_block(weight, scale)
    full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:200, :130]
    assert got.dtype == torch.bfloat16 and torch.equal(got, (weight.float() * full).to(torch.bfloat16))


def test_prompt_lengths_run_through_the_first_trained_position():
    import torch

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import prompt_lengths

    cu = torch.tensor([0, 5, 8, 12], dtype=torch.int32)
    trained = torch.zeros(12, dtype=torch.bool)
    trained[[2, 3]] = True  # request 0: prompt of 3 tokens, then decoded
    trained[5] = True  # request 1: trained from its first token
    assert prompt_lengths(trained, cu).tolist() == [3, 1, 4]


def test_stock_recompute_replays_the_forward_choices_and_vllm_mla_weights():
    import types

    import torch

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
            self.weight_scale_inv = torch.nn.Parameter(torch.ones(1, 1), requires_grad=False)
            self.block_size = 128

        def forward(self, x):
            return "fp8"

    seen = {}

    class Layer(torch.nn.Module):
        block_type = "deepseek_sparse_attention"
        _replay_layer_idx = 2

        def __init__(self):
            super().__init__()
            self.self_attn = types.SimpleNamespace(q_a_proj=Projection(), kv_a_proj_with_mqa=Projection(),
                                                   q_b_proj=Projection(), o_proj=Projection())  # fmt: skip

        def forward(self, streams, **kwargs):
            seen["routes"] = kwargs.get("routed_experts")
            seen["topk"] = getattr(self.self_attn, glm5_next.REPLAY_TOPK_ATTR, None)
            seen["projection"] = self.self_attn.q_a_proj.forward
            return streams, None

    layer = Layer()
    experts = torch.tensor([[1, 2], [3, 4], [5, 6]])
    topk = torch.tensor([[0, -1], [0, 1], [0, 1]], dtype=torch.int32)
    replay = {id(layer): {"experts": experts, "topk": topk}}
    run = glm5_next._stock_layer_runner(None, torch.zeros(1, 3, dtype=torch.long), torch.arange(3)[None], {}, replay)
    run(layer, torch.zeros(3, 4, 8))

    assert seen["routes"].shape == (1, 3, 3, 2) and torch.equal(seen["routes"][0, :, 2], experts)
    assert torch.equal(seen["topk"][0], topk)
    assert seen["projection"].__func__ is glm5_next._bf16_linear_forward
    assert not hasattr(layer.self_attn, glm5_next.REPLAY_TOPK_ATTR)
    assert layer.self_attn.q_a_proj(torch.zeros(1)) == "fp8"


def _peft_projection(rows: int = 6, cols: int = 8, r: int = 2, alpha: int = 4):
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear as LoraLinear

    class Projection(torch.nn.Linear):
        def __init__(self):
            super().__init__(cols, rows, bias=False)
            self.weight_scale_inv = torch.nn.Parameter(torch.full((1, 1), 0.5), requires_grad=False)
            self.block_size = 128

        def forward(self, x):
            raise AssertionError("the W8A8 forward runs")

    wrapped = LoraLinear(Projection(), "default", LoraConfig(r=r, lora_alpha=alpha), r=r, lora_alpha=alpha)
    torch.nn.init.normal_(wrapped.lora_B["default"].weight)
    fp8 = torch.randn(rows, cols).to(torch.float8_e4m3fn)
    wrapped.base_layer.weight = torch.nn.Parameter(fp8, requires_grad=False)
    return wrapped


def test_lora_factors_cast_to_bf16_then_fold_the_scaling_like_vllm():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import bf16_weight
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import lora_factors

    torch.manual_seed(3)
    projection = _peft_projection(r=2, alpha=4)
    a, b = lora_factors(projection)
    assert a.dtype == b.dtype == torch.bfloat16
    assert torch.equal(a, projection.lora_A["default"].weight.to(torch.bfloat16))
    assert torch.equal(b, projection.lora_B["default"].weight.to(torch.bfloat16) * 2.0)
    assert torch.equal(bf16_weight(projection), (projection.base_layer.weight.float() * 0.5).to(torch.bfloat16))
    projection.enable_adapters(False)
    assert lora_factors(projection) is None
    assert lora_factors(torch.nn.Linear(2, 2)) is None


def test_add_lora_stacks_merged_slices_like_vllm(monkeypatch):
    import sys
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    def lora_shrink(x, a_stack, buffer, *meta_and_scale):
        assert len(meta_and_scale) == 8
        for i, a in enumerate(a_stack):
            buffer[i] = (x.float() @ a[0].float().t()) * meta_and_scale[-1]

    def lora_expand(buffer, b_stack, y, *meta, offset_start, add_inputs):
        assert add_inputs and offset_start == 0
        offset = 0
        for i, b in enumerate(b_stack):
            width = b.shape[1]
            y[:, offset : offset + width] += (buffer[i].to(b.dtype) @ b[0].t()).to(y.dtype)
            offset += width

    class Meta:
        @staticmethod
        def make(max_loras, tokens, device):
            return Meta()

        def prepare_tensors(self, mapping):
            assert mapping.tolist() == [0] * mapping.numel()

        def meta_args(self, tokens, specialize):
            return (None,) * 7

    ops = types.ModuleType("vllm.lora.ops.triton_ops")
    utils = types.ModuleType("vllm.lora.ops.triton_ops.utils")
    utils._LORA_A_PTR_DICT, utils._LORA_B_PTR_DICT = {(1,): "stale"}, {(1,): "stale"}
    ops.LoRAKernelMeta, ops.lora_shrink, ops.lora_expand, ops.utils = Meta, lora_shrink, lora_expand, utils
    for name, module in (("vllm", types.ModuleType("vllm")), ("vllm.lora", types.ModuleType("vllm.lora")),
                         ("vllm.lora.ops", types.ModuleType("vllm.lora.ops")), (ops.__name__, ops),
                         (utils.__name__, utils)):  # fmt: skip
        monkeypatch.setitem(sys.modules, name, module)

    torch.manual_seed(4)
    x = torch.randn(5, 8, dtype=torch.bfloat16)
    y = torch.randn(5, 7, dtype=torch.bfloat16)
    q = (torch.randn(2, 8, dtype=torch.bfloat16), torch.randn(3, 2, dtype=torch.bfloat16))
    kv = (torch.randn(2, 8, dtype=torch.bfloat16), torch.randn(2, 2, dtype=torch.bfloat16))  # 2 of 4 rows: no rope rows
    out = glm5_next.add_lora(y.clone(), x, [q, kv], (3, 4))

    def delta(a, b):
        return ((x.float() @ a.float().t()).to(torch.bfloat16) @ b.t()).to(torch.bfloat16)

    assert torch.equal(out[:, :3], y[:, :3] + delta(*q))
    assert torch.equal(out[:, 3:5], y[:, 3:5] + delta(*kv))
    assert torch.equal(out[:, 5:], y[:, 5:])
    assert not utils._LORA_A_PTR_DICT and not utils._LORA_B_PTR_DICT
    only_kv = glm5_next.add_lora(y.clone(), x, [None, kv], (3, 4))
    assert torch.equal(only_kv[:, :3], y[:, :3]) and torch.equal(only_kv[:, 3:5], out[:, 3:5])
    assert glm5_next.add_lora(y, x, [None, None], (3, 4)) is y


def test_stock_recompute_keeps_the_peft_adapter_on_the_bf16_projection():
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    torch.manual_seed(5)
    projection = _peft_projection()
    x = torch.randn(3, 8)
    seen = {}

    class Layer(torch.nn.Module):
        block_type = "deepseek_sparse_attention"

        def __init__(self):
            super().__init__()
            self.self_attn = types.SimpleNamespace(q_a_proj=projection)

        def forward(self, streams, **kwargs):
            seen["out"] = self.self_attn.q_a_proj(x)
            return streams, None

    run = glm5_next._stock_layer_runner(None, torch.zeros(1, 3, dtype=torch.long), torch.arange(3)[None], {})
    run(Layer(), torch.zeros(3, 4, 8))

    lora = projection.lora_B["default"](projection.lora_A["default"](x)) * projection.scaling["default"]
    base = torch.nn.functional.linear(x, glm5_next.bf16_weight(projection).float())
    assert torch.allclose(seen["out"], base + lora)
    seen["out"].sum().backward()
    assert projection.lora_B["default"].weight.grad.abs().sum() > 0
    assert "forward" not in vars(projection) and "forward" not in vars(projection.base_layer)


def test_substitute_takes_the_exact_value_and_the_reference_gradient():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import substitute

    w = torch.tensor([2.0, 3.0], requires_grad=True)
    reference = (w * 1.5).unsqueeze(0)
    exact = torch.tensor([3.25, 4.25], dtype=torch.float64)
    out = substitute(reference, exact)
    assert out.shape == (1, 2) and out.dtype == torch.float64 and out.tolist() == [[3.25, 4.25]]
    out.sum().backward()
    assert w.grad.tolist() == [1.5, 1.5]


def test_fake_fp8_is_vllms_group_quantization_with_a_straight_through_gradient():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import fake_fp8

    torch.manual_seed(9)
    x = (torch.randn(3, 256) * 5).to(torch.bfloat16).requires_grad_(True)
    groups = x.detach().float().reshape(3, 2, 128)
    scale = groups.abs().amax(-1, keepdim=True) / 448.0
    expected = ((groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).reshape(3, 256)
    assert torch.equal(fake_fp8(x, False), expected.to(torch.bfloat16))
    pow2 = torch.exp2(torch.ceil(torch.log2(scale)))
    expected_ue8m0 = ((groups / pow2).to(torch.float8_e4m3fn).float() * pow2).reshape(3, 256)
    out = fake_fp8(x, True)
    assert torch.equal(out, expected_ue8m0.to(torch.bfloat16)) and not torch.equal(out, fake_fp8(x, False))
    out.float().sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    odd = torch.randn(2, 100)
    assert torch.equal(fake_fp8(odd, True), odd)


def test_vllm_backward_chains_the_exact_sublayer_values_with_stock_gradients(monkeypatch):
    import contextlib
    import sys
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    n, hidden, tokens = 2, 4, 3

    def remix(fused, fp32, post, comb, out, res):
        return post[..., None] * out[..., None, :] + comb.transpose(-1, -2) @ res

    modeling = types.ModuleType("arctic_platform.model.implementations.glm53.modeling_glm5_next")
    modeling.fused_hyper_connection_applies = lambda hc, res: False
    modeling._remix = remix
    package = types.ModuleType("arctic_platform.model.implementations.glm53")
    package.modeling_glm5_next = modeling
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)

    class HC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(1.2))

        def forward(self, res):
            post = torch.sigmoid(self.a) * torch.ones(1, tokens, n)
            return post, torch.eye(n).expand(1, tokens, n, n) * self.a, res.mean(2) * self.a

    class Scale(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(value))

        def forward(self, hidden_states, **kwargs):
            return hidden_states * self.w

    class Layer(torch.nn.Module):
        block_type, _fp32_parity, _replay_layer_idx = "linear_attention", True, 0

        def __init__(self):
            super().__init__()
            self.attn_hc, self.ffn_hc = HC(), HC()
            self.input_layernorm, self.post_attention_layernorm = Scale(0.9), Scale(1.1)
            self.self_attn, self.mlp = Scale(1.7), Scale(2.3)

    def exact_pre(streams, hc, norm, config):
        return torch.full((tokens, n), 0.5), torch.eye(n).expand(tokens, n, n) * 1.25, streams.mean(1) * 1.5

    def exact_post(x, streams, post, comb):
        return remix(False, True, post, comb, x, streams)

    monkeypatch.setattr(glm5_next, "hc_pre", exact_pre)
    monkeypatch.setattr(glm5_next, "hc_post", exact_post)
    monkeypatch.setattr(glm5_next, "_mixer", lambda layer, x, ctx: x * 2.0)
    monkeypatch.setattr(glm5_next, "_mlp", lambda layer, x, ctx: x * 3.0)
    monkeypatch.setattr(glm5_next, "_vllm_mlp_quantization", lambda layer: contextlib.nullcontext())

    layer = Layer()
    ctx = types.SimpleNamespace(config=types.SimpleNamespace(), replay={})
    run = glm5_next._vllm_backward_runner(torch.zeros(tokens, dtype=torch.long), torch.arange(tokens), {}, ctx)
    torch.manual_seed(10)
    streams = torch.randn(tokens, n, hidden, requires_grad=True)
    out = run(layer, streams)

    with torch.no_grad():
        post, comb, x = exact_pre(streams, None, None, None)
        s1 = exact_post(x * 2.0, streams, post, comb)
        post, comb, x = exact_pre(s1, None, None, None)
        expected = exact_post(x * 3.0, s1, post, comb)
    assert torch.equal(out, expected)
    out.sum().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert streams.grad is not None and streams.grad.abs().sum() > 0


def test_tp_sum_adds_rank_partials_in_fp32_in_rank_order_and_rounds_once():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import tp_sum

    torch.manual_seed(6)
    partials = [torch.randn(5, 7).to(torch.bfloat16) for _ in range(4)]
    expected = partials[0].float()
    for p in partials[1:]:
        expected = expected + p.float()
    out = tp_sum(partials)
    assert out.dtype == torch.bfloat16 and torch.equal(out, expected.to(torch.bfloat16))
    assert tp_sum(partials[:1]) is partials[0]


def test_fp8_and_lora_shards_follow_vllm_tensor_parallel_layouts():
    import pytest

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    weight = torch.arange(512 * 3).reshape(512, 3)
    scale = torch.arange(4 * 1).reshape(4, 1).float()
    w, s = glm5_next._fp8_shard(weight, scale, 2, 1, 0)
    assert torch.equal(w, weight[256:]) and torch.equal(s, scale[2:])
    with pytest.raises(ValueError, match="128-blocks"):
        glm5_next._fp8_shard(weight, scale, 8, 0, 0)
    experts = torch.arange(2 * 3 * 512).reshape(2, 3, 512)
    e, es = glm5_next._fp8_shard(experts, torch.zeros(2, 1, 4), 4, 3, 2)
    assert torch.equal(e, experts[:, :, 384:]) and es.shape == (2, 1, 1)

    a, b = torch.randn(2, 8), torch.randn(6, 2)
    col_a, col_b = glm5_next._column_lora((a, b), 3, 2)
    assert torch.equal(col_a, a) and torch.equal(col_b, b[4:])
    row_a, row_b = glm5_next._row_lora((a, b), 4, 1)
    assert torch.equal(row_a, a[:, 2:4]) and torch.equal(row_b, b)
    assert glm5_next._row_lora(None, 4, 1) is None and glm5_next._column_lora((a, b), 1, 0) == (a, b)


def test_tensor_parallel_check_needs_whole_heads_and_fp8_blocks():
    import types

    import pytest

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import check_tensor_parallel

    config = types.SimpleNamespace(num_attention_heads=64, linear_num_heads=32, intermediate_size=12288,
                                   moe_intermediate_size=1536, n_shared_experts=1)  # fmt: skip
    check_tensor_parallel(config, 4)
    with pytest.raises(ValueError, match="moe_intermediate_size"):
        check_tensor_parallel(config, 8)
    with pytest.raises(ValueError, match="linear_num_heads"):
        check_tensor_parallel(types.SimpleNamespace(**{**vars(config), "linear_num_heads": 6}), 4)


def test_kda_rank_weights_are_vllm_tensor_parallel_shards():
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    heads, head_dim, hidden, kernel = 4, 2, 5, 3
    projection = heads * head_dim

    def lin(rows, cols):
        return types.SimpleNamespace(weight=torch.randn(rows, cols))

    torch.manual_seed(7)
    gate = types.SimpleNamespace(f_a_proj=lin(head_dim, hidden), f_b_proj=lin(projection, head_dim),
                                 A_log=torch.randn(heads), dt_bias=torch.randn(projection))  # fmt: skip
    attn = types.SimpleNamespace(num_heads=heads, head_dim=head_dim, q_proj=lin(projection, hidden),
                                 k_proj=lin(projection, hidden), v_proj=lin(projection, hidden), b_proj=lin(heads, hidden),
                                 g_a_proj=lin(head_dim, hidden), g_b_proj=lin(projection, head_dim), forget_gate=gate,
                                 conv1d=types.SimpleNamespace(weight=torch.randn(3 * projection, 1, kernel)),
                                 o_norm=types.SimpleNamespace(weight=torch.randn(head_dim)), o_proj=lin(hidden, projection))  # fmt: skip
    full = glm5_next._kda_weights(attn)
    assert full.heads == heads and full.merged.shape == (3 * projection + heads + 2 * head_dim, hidden)
    w = glm5_next._kda_weights(attn, tp=2, rank=1)
    half = slice(projection // 2, projection)
    expected = torch.cat([attn.q_proj.weight[half], attn.k_proj.weight[half], attn.v_proj.weight[half],
                          attn.b_proj.weight[2:], gate.f_a_proj.weight, attn.g_a_proj.weight])  # fmt: skip
    assert w.heads == 2 and torch.equal(w.merged, expected)
    conv = attn.conv1d.weight.reshape(3 * projection, kernel)
    assert torch.equal(w.conv, torch.cat([conv[half], conv[projection:][half], conv[2 * projection:][half]]))
    assert torch.equal(w.f_b, gate.f_b_proj.weight[half]) and torch.equal(w.g_b, attn.g_b_proj.weight[half])
    assert torch.equal(w.a_log, gate.A_log[2:]) and torch.equal(w.dt_bias, gate.dt_bias[half])
    assert torch.equal(w.o_weight, attn.o_proj.weight[:, half]) and w.o_lora is None


def test_lm_head_logits_are_the_ranks_vocabulary_slices(monkeypatch):
    import types

    from arctic_platform.model.implementations.vllm_aligned import ops

    shapes = []

    def head(x, weight):
        shapes.append(tuple(weight.shape))
        return x @ weight.t()

    monkeypatch.setattr(ops, "_batch_invariant", lambda: types.SimpleNamespace(linear_batch_invariant=head))
    monkeypatch.setattr(ops, "_token_logprobs_kernel",
                        lambda logits, ids: torch.log_softmax(logits.float(), -1).gather(1, ids[:, None])[:, 0])  # fmt: skip
    torch.manual_seed(8)
    hidden, weight = torch.randn(3, 4), torch.randn(100, 4)
    labels = torch.tensor([0, 50, 99])
    one = ops.lm_head_logprobs_kernel(hidden, weight, labels, 1.0, 8, 64, tensor_parallel=1)
    assert shapes == [(128, 4)]
    shapes.clear()
    four = ops.lm_head_logprobs_kernel(hidden, weight, labels, 1.0, 8, 64, tensor_parallel=4)
    assert shapes == [(32, 4)] * 4
    assert torch.allclose(one, four)


def test_aligned_layer_backward_runs_with_deterministic_algorithms_and_restores_the_setting():
    seen = []

    def stock(s):
        import torch.utils.deterministic

        seen.append((torch.are_deterministic_algorithms_enabled(), torch.utils.deterministic.fill_uninitialized_memory))
        return s * 2

    streams = torch.randn(3, 4, requires_grad=True)
    assert not torch.are_deterministic_algorithms_enabled()
    _AlignedLayer.apply(streams, lambda s: s * 2, stock).sum().backward()
    assert seen == [(True, False)]
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.utils.deterministic.fill_uninitialized_memory
    assert torch.equal(streams.grad, torch.full((3, 4), 2.0))

def _steps(layout):
    return [(st.decode, st.rows.tolist(), st.cu.tolist(), st.slots.tolist(), st.has_initial_state.tolist(), st.imports,
             st.exports) for st in layout.steps]  # fmt: skip


def test_single_turn_kda_layout_cuts_prompts_every_chunk_tokens_then_decodes():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import _single_turn_kda_layout

    cu = torch.tensor([0, 5, 8, 18], dtype=torch.int32)
    layout = _single_turn_kda_layout(cu, torch.tensor([4, 3, 10]), 4, torch.device("cpu"))
    assert not layout.single and layout.num_slots == 3
    assert _steps(layout) == [
        (False, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11], [0, 4, 7, 11], [1, 2, 3], [False] * 3, [], []),
        (False, [12, 13, 14, 15], [0, 4], [3], [True], [], []),
        (False, [16, 17], [0, 2], [3], [True], [], []),
        (True, [4], [0, 1, 1, 1], [1, 2, 3], [True] * 3, [], []),
    ]
    whole = _single_turn_kda_layout(cu, torch.tensor([5, 3, 10]), 0, torch.device("cpu"))
    assert whole.single and whole.steps[0].rows is None


def test_kda_steps_resume_from_the_providers_state_and_run_one_token_chunks_as_decode():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import _kda_steps_layout

    spans = [(0, [(False, 0, 8), (False, 8, 12), (True, 12, 14)], None),
             (14, [(False, 0, 4), (True, 4, 6)], (0, 8)),
             (20, [(False, 0, 5), (False, 5, 6)], None)]  # fmt: skip
    layout = _kda_steps_layout(spans, torch.device("cpu"))
    a, b, c = (list(range(lo, hi)) for lo, hi in ((0, 8), (14, 18), (20, 25)))
    assert _steps(layout) == [
        (False, a + c, [0, 8, 13], [1, 3], [False, False], [], [(1, (0, 8))]),
        (False, [8, 9, 10, 11], [0, 4], [1], [True], [], []),
        (True, [25], [0, 1], [3], [True], [], []),
        (True, [12, 13], [0, 2], [1], [True], [], []),
        (False, b, [0, 4], [2], [True], [(2, (0, 8))], []),
        (True, [18, 19], [0, 2], [2], [True], [], []),
    ]


def test_turn_layout_reads_cached_pages_from_their_writer_and_resumes_its_state():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import plan_block_size
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import turn_layout
    from arctic_platform.model.implementations.vllm_aligned.turn_plan import encode_turns

    plan = encode_turns(12, [{"start": 0, "chunk_ends": [8], "prompt_end": 8, "end": 10, "outputs": [7, 8]},
                             {"start": 8, "chunk_ends": [11], "prompt_end": 11, "end": 12, "outputs": [10]}],
                        block_size=4)  # fmt: skip
    cu = torch.tensor([0, 12], dtype=torch.int32)
    assert plan_block_size(plan, cu) == 8
    # Hits at 4 and 8, but the third request's [4, 8) comes from two writers (pages 2 and 3): the page is 2.
    later = encode_turns(16, [{"start": 0, "chunk_ends": [4, 6], "prompt_end": 6, "end": 7, "outputs": [5]},
                              {"start": 4, "chunk_ends": [8, 9], "prompt_end": 9, "end": 10, "outputs": [8]},
                              {"start": 8, "chunk_ends": [13], "prompt_end": 13, "end": 14, "outputs": [12]}],
                         block_size=2)  # fmt: skip
    assert plan_block_size(later, torch.tensor([0, 16], dtype=torch.int32)) == 2
    turns = turn_layout(plan, cu, block_size=4, device=torch.device("cpu"))
    assert turns.expand_index.tolist() == list(range(10)) + [8, 9, 10, 11]
    assert turns.positions.tolist() == list(range(10)) + [8, 9, 10, 11]
    assert turns.cu.tolist() == [0, 10, 14]
    assert turns.key_index.tolist() == list(range(10)) + list(range(8)) + [10, 11, 12, 13]
    assert turns.key_cu.tolist() == [0, 10, 22]
    assert turns.output_index.tolist() == list(range(10)) + [12, 13]
    assert _steps(turns.kda) == [
        (False, list(range(8)), [0, 8], [1], [False], [], [(1, (0, 8))]),
        (True, [8, 9], [0, 2], [1], [True], [], []),
        (False, [10, 11, 12], [0, 3], [2], [True], [(2, (0, 8))], []),
        (True, [13], [0, 1], [2], [True], [], []),
    ]
    assert turns.replay_index.tolist() == list(range(10)) + [12, 13]


def test_plan_routes_replay_on_the_first_request_to_compute_a_position_only():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import plan_routes
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import replay_mask
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import turn_layout
    from arctic_platform.model.implementations.vllm_aligned.turn_plan import encode_turns

    # The first request decodes positions 8 and 9; the second re-prefills them from its hit at 8 (a page tail).
    plan = encode_turns(12, [{"start": 0, "chunk_ends": [8], "prompt_end": 8, "end": 10, "outputs": [7, 8]},
                             {"start": 8, "chunk_ends": [11], "prompt_end": 11, "end": 12, "outputs": [10]}],
                        block_size=4)  # fmt: skip
    turns = turn_layout(plan, torch.tensor([0, 12], dtype=torch.int32), block_size=4, device=torch.device("cpu"))
    routes = torch.stack([torch.arange(4) + 4 * (p % 3) for p in range(12)]).to(torch.uint8)[:, None, :]  # 1 layer
    out = plan_routes(routes, turns)
    assert out.dtype == torch.int32 and out.shape == (14, 1, 4)
    replayed = replay_mask(out[:, 0], 16)
    # span tokens 10 and 11 are the second request's recompute of positions 8 and 9
    assert replayed.tolist() == [True] * 10 + [False, False, True, True]
    assert torch.equal(out[replayed, 0], routes[turns.expand_index[replayed], 0].to(torch.int32))


def test_plan_routes_leave_another_trajectorys_request_to_route_itself():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import plan_routes
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import replay_mask
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import turn_layout
    from arctic_platform.model.implementations.vllm_aligned.turn_plan import encode_turns

    # Another conversation's request wrote pages [0, 8) (no logprobs here); the row's own request hit them at 8.
    plan = encode_turns(12, [{"start": 0, "chunk_ends": [8], "prompt_end": 8, "end": 9, "outputs": []},
                             {"start": 8, "chunk_ends": [11], "prompt_end": 11, "end": 12, "outputs": [10]}],
                        block_size=4)  # fmt: skip
    turns = turn_layout(plan, torch.tensor([0, 12], dtype=torch.int32), block_size=4, device=torch.device("cpu"))
    assert turns.expand_index.tolist() == list(range(9)) + [8, 9, 10, 11]
    assert turns.replay_index.tolist() == [-1] * 8 + [9, 10, 11, 12]
    routes = torch.stack([torch.arange(4) + 4 * (p % 3) for p in range(12)]).to(torch.uint8)[:, None, :]
    replayed = replay_mask(plan_routes(routes, turns)[:, 0], 16)
    assert replayed.tolist() == [False] * 9 + [True] * 4
    # a row where no span returns logprobs (a plan-free row in a call with plans) replays on its first spans
    bare = encode_turns(12, [{"start": 0, "chunk_ends": [8], "prompt_end": 8, "end": 12, "outputs": []}], block_size=4)
    bare_turns = turn_layout(bare, torch.tensor([0, 12], dtype=torch.int32), block_size=4, device=torch.device("cpu"))
    assert replay_mask(plan_routes(routes, bare_turns)[:, 0], 16).all()


def _router_config():
    import types

    return types.SimpleNamespace(scoring_func="sigmoid", norm_topk_prob=True, routed_scaling_factor=2.5,
                                 n_routed_experts=16, n_group=4, topk_group=2, num_experts_per_tok=4, swiglu_limit=7.0)  # fmt: skip


def test_replayed_weights_equal_the_grouped_topk_at_its_own_experts():
    from arctic_platform.inference.vllm.glm5_batch_invariance import grouped_topk
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import replayed_weights

    config, gen = _router_config(), torch.Generator().manual_seed(0)
    logits = torch.randn(64, config.n_routed_experts, generator=gen)
    bias = torch.randn(config.n_routed_experts, generator=gen) * 0.1
    weights, ids = grouped_topk(None, logits, topk=config.num_experts_per_tok, renormalize=True,
                                num_expert_group=config.n_group, topk_group=config.topk_group, scoring_func="sigmoid",
                                routed_scaling_factor=config.routed_scaling_factor, e_score_correction_bias=bias)  # fmt: skip
    assert torch.equal(replayed_weights(logits, ids, config), weights)


def test_replay_mask_takes_distinct_in_range_routes_only():
    from arctic_platform.model import ROUTER_REPLAY_FRESH
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import replay_mask

    routes = torch.tensor([[3, 1, 7, 12], [0, 0, 0, 0], [ROUTER_REPLAY_FRESH] * 4, [3, 3, 7, 12], [3, 1, 7, 16]])
    assert replay_mask(routes, 16).tolist() == [True, False, False, False, False]


def test_moe_replays_captured_experts_and_routes_the_rest(monkeypatch):
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    config, gen = _router_config(), torch.Generator().manual_seed(1)
    x = torch.randn(3, 8, generator=gen)
    gate = torch.randn(config.n_routed_experts, 8, generator=gen)
    mlp = types.SimpleNamespace(router=types.SimpleNamespace(gate=types.SimpleNamespace(weight=gate)),
                                expert_bias=torch.zeros(config.n_routed_experts), shared_expert=None, experts=None)  # fmt: skip
    seen = {}

    def routed(x, weights, ids, **_):
        seen.update(weights=weights, ids=ids)
        return torch.zeros_like(x)

    monkeypatch.setattr(glm5_next, "_routed_experts_ranks", routed)
    monkeypatch.setattr(glm5_next, "shared_expert_partials", lambda *a, **k: [torch.zeros_like(x)])
    monkeypatch.setattr(glm5_next, "tp_sum", lambda parts: parts[0])
    import sys

    workspace = types.ModuleType("vllm.v1.worker.workspace")
    workspace.is_workspace_manager_initialized, workspace.init_workspace_manager = (lambda: True), (lambda device: None)
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.workspace", workspace)
    fp32_logits = types.ModuleType("arctic_platform.inference.vllm.batch_invariant_fp32_logits")
    fp32_logits.linear_fp32_out = lambda x, w: x.float() @ w.float().t()
    monkeypatch.setitem(sys.modules, fp32_logits.__name__, fp32_logits)
    glm5_next.moe(mlp, x, config)
    own_ids, own_weights = seen["ids"].clone(), seen["weights"].clone()
    replay = torch.tensor([[15, 14, 13, 12], [0, 0, 0, 0], own_ids[2].tolist()])
    glm5_next.moe(mlp, x, config, routes=replay)
    assert seen["ids"][0].tolist() == [15, 14, 13, 12]
    assert torch.equal(seen["ids"][1:], own_ids[1:]) and torch.equal(seen["weights"][1:], own_weights[1:])
    logits = x.float() @ gate.float().t()
    assert torch.equal(seen["weights"][0], glm5_next.replayed_weights(logits[:1], seen["ids"][:1], config)[0])


def test_bf16_mlp_partials_merge_gate_up_per_rank_and_sum_to_the_unsharded_mlp(monkeypatch):
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next
    from arctic_platform.model.implementations.vllm_aligned.glm5_next_backward import ref_silu_and_mul_with_clamp

    seen = []

    def linear(x, w):
        seen.append(tuple(w.shape))
        return x.double() @ w.double().t()

    monkeypatch.setattr(glm5_next, "_linear", linear)
    monkeypatch.setattr(glm5_next, "silu_and_mul_with_clamp", ref_silu_and_mul_with_clamp)
    gen = torch.Generator().manual_seed(2)
    hidden, inter, limit = 16, 8, 7.0
    gate, up, down = (torch.randn(*shape, generator=gen) for shape in ((inter, hidden), (inter, hidden), (hidden, inter)))
    mlp = types.SimpleNamespace(gate_proj=types.SimpleNamespace(weight=gate), up_proj=types.SimpleNamespace(weight=up),
                                down_proj=types.SimpleNamespace(weight=down), swiglu_limit=limit)  # fmt: skip
    x = torch.randn(5, hidden, generator=gen)
    whole = glm5_next.dense_mlp(mlp, x, tp=1)
    assert seen == [(2 * inter, hidden), (hidden, inter)]
    seen.clear()
    split = glm5_next.dense_mlp(mlp, x, tp=2)
    assert seen == [(inter, hidden), (hidden, inter // 2)] * 2
    assert torch.allclose(split.double(), whole.double(), atol=1e-6)


def test_bf16_routed_experts_hand_each_rank_its_shards_without_scales(monkeypatch):
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    calls = []

    def kernel(x, weights, ids, w13, w13_scale, w2, w2_scale, *, top_k, limit):
        calls.append((w13, w13_scale, w2, w2_scale))
        return torch.zeros_like(x)

    monkeypatch.setattr(glm5_next, "_routed_experts_kernel", kernel)
    experts_n, hidden, inter = 4, 16, 8
    experts = types.SimpleNamespace(w1=torch.randn(experts_n, inter, hidden), w3=torch.randn(experts_n, inter, hidden),
                                    w2=torch.randn(experts_n, hidden, inter))  # fmt: skip
    x, weights, ids = torch.randn(3, hidden), torch.rand(3, 2), torch.tensor([[0, 1], [2, 3], [1, 2]])
    glm5_next._routed_experts_ranks(x, weights, ids, experts=experts, total=experts_n, ep_group=None, tp=2, top_k=2,
                                    limit=7.0)  # fmt: skip
    assert len(calls) == 2
    for rank, (w13, s13, w2, s2) in enumerate(calls):
        half = slice(rank * inter // 2, (rank + 1) * inter // 2)
        assert s13 is None and s2 is None
        assert torch.equal(w13, torch.cat([experts.w1[:, half], experts.w3[:, half]], dim=1))
        assert torch.equal(w2, experts.w2[:, :, half])


def test_tensor_parallel_check_drops_the_fp8_block_rule_for_bf16_checkpoints():
    import types

    import pytest

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import check_tensor_parallel

    config = types.SimpleNamespace(num_attention_heads=64, linear_num_heads=64, intermediate_size=12288 + 64,
                                   moe_intermediate_size=2048, n_shared_experts=1)  # fmt: skip
    check_tensor_parallel(config, 8, fp8=False)
    with pytest.raises(ValueError):
        check_tensor_parallel(config, 8, fp8=True)


def test_packed_boundaries_from_gathered_positions_make_each_pad_token_its_own_request():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import _cu_from_positions

    positions = torch.tensor([[0, 1, 2, 0, 1, 0, 0]])
    assert _cu_from_positions(positions).tolist() == [0, 3, 5, 6, 7]
    assert _cu_from_positions(torch.tensor([[3, 4, 0]])).tolist() == [0, 2, 3]


def test_uneven_shards_are_padded_for_the_gather_and_trimmed_after(monkeypatch):
    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    sent = []

    def gather(group, x, dim=0):
        sent.append(x.clone())
        return torch.cat([torch.full_like(x, 7.0), x])  # rank 0's full chunk, then this rank's padded rows

    monkeypatch.setattr(glm5_next, "_gather_tokens", gather)
    with torch.no_grad():
        out = glm5_next._gather_rows(None, torch.ones(2, 3), chunk=4, total=6)
    assert sent[0].shape == (4, 3) and torch.equal(sent[0][2:], torch.zeros(2, 3))
    assert torch.equal(out, torch.cat([torch.full((4, 3), 7.0), torch.ones(2, 3)]))


def test_sequence_parallel_mixer_runs_the_whole_sequence_and_keeps_the_shards_rows(monkeypatch):
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    whole = torch.arange(8.0).reshape(8, 1)
    seen = {}

    def sparse_attention(attn, x, positions, cu_seqlens, config, record=None, tp=1, keys=None):
        seen["tokens"] = x.shape[0]
        if record is not None:
            record["topk"] = torch.arange(8 * 2).reshape(8, 2)
        return x * 10

    def partials(attn, x, positions, cu_seqlens, config, record, tp, keys, ranks, queries=None):
        assert queries is None
        return [sparse_attention(attn, x, positions, cu_seqlens, config, record, tp, keys)]

    def all_to_all_single(output, input, output_split_sizes, input_split_sizes, group=None):
        # Rank 0 sends TP rank 0's rows of either shard; stand-in for rank 1's TP rank 1: its partial of tokens 4-7.
        assert input_split_sizes == [4, 4] and output_split_sizes == [4, 4]
        output.copy_(input)

    # SP rank 0 of 2 against TP 2: it computes TP rank 0 over the whole sequence and owns tokens 0-3.
    monkeypatch.setattr(glm5_next, "_gather_rows", lambda group, x, chunk, total: whole)
    monkeypatch.setattr(glm5_next, "mla_partials", partials)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", all_to_all_single)
    layer = types.SimpleNamespace(block_type="deepseek_sparse_attention", self_attn=None)
    ctx = types.SimpleNamespace(config=None, positions=None, cu_seqlens=None, turns=None, replay={}, sp_group=object(),
                                shard=slice(0, 4), sp_chunk=4, sp_tokens=8, tp=2)  # fmt: skip
    with torch.no_grad():
        out = glm5_next._mixer(layer, whole[:4], ctx)
    assert seen["tokens"] == 8
    assert torch.equal(out, (whole[:4] + whole[4:]) * 10)
    assert torch.equal(ctx.replay[id(layer)]["topk"], torch.arange(16).reshape(8, 2)[:4])


def test_sequence_parallel_ranks_sharing_a_tensor_parallel_rank_run_mla_only_at_their_owners_queries(monkeypatch):
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    # Two SP ranks, one TP rank: each computes it, rank 1 for owner 1 alone (the last, short shard: tokens 3 and 4).
    whole = torch.arange(5.0).reshape(5, 1)
    seen = {}

    def partials(attn, x, positions, cu_seqlens, config, record, tp, keys, ranks, queries=None):
        seen["queries"] = queries.tolist()
        record["topk"] = torch.arange(5 * 2).reshape(5, 2)[queries]
        return [x[queries] * 10]

    def all_to_all_single(output, input, output_split_sizes, input_split_sizes, group=None):
        assert output_split_sizes == [0, 2] and input_split_sizes == [0, 2]
        output.copy_(input)

    monkeypatch.setattr(glm5_next, "_gather_rows", lambda group, x, chunk, total: whole)
    monkeypatch.setattr(glm5_next, "mla_partials", partials)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", all_to_all_single)
    layer = types.SimpleNamespace(block_type="deepseek_sparse_attention", self_attn=None)
    ctx = types.SimpleNamespace(config=None, positions=None, cu_seqlens=None, turns=None, replay={}, sp_group=object(),
                                shard=slice(3, 5), sp_chunk=3, sp_tokens=5, tp=1)  # fmt: skip
    with torch.no_grad():
        out = glm5_next._mixer(layer, whole[3:], ctx)
    assert seen["queries"] == [3, 4]
    assert torch.equal(out, whole[3:] * 10)
    assert torch.equal(ctx.replay[id(layer)]["topk"], torch.arange(10).reshape(5, 2)[3:])


def test_sequence_parallel_ranks_sharing_a_tensor_parallel_rank_split_kda_at_request_boundaries(monkeypatch):
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next

    # Two SP ranks, one TP rank, requests [0, 2), [2, 7), [7, 10): the cut nearest half the tokens is 7. Rank 1 owns
    # tokens [5, 10), runs the last request only and gets [5, 7) from rank 0.
    whole = torch.arange(10.0).reshape(10, 1)
    seen = {}

    def partials(attn, x, cu_seqlens, config, prompt_lens, trace, tp, prefill_chunk, layout, ranks):
        seen.update(x=x.flatten().tolist(), cu=cu_seqlens.tolist(), prompt_lens=prompt_lens.tolist(), layout=layout)
        return [x * 10]

    def all_to_all_single(output, input, output_split_sizes, input_split_sizes, group=None):
        assert output_split_sizes == [2, 3] and input_split_sizes == [0, 3]
        output[:2] = torch.tensor([[50.0], [60.0]])
        output[2:] = input

    monkeypatch.setattr(glm5_next, "_gather_rows", lambda group, x, chunk, total: whole)
    monkeypatch.setattr(glm5_next, "kda_partials", partials)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", all_to_all_single)
    layer = types.SimpleNamespace(block_type="linear_attention", self_attn=None)
    ctx = types.SimpleNamespace(config=None, positions=None, cu_seqlens=torch.tensor([0, 2, 7, 10], dtype=torch.int32),
                                prompt_lens=torch.tensor([2, 4, 3]), turns=None, replay=None, sp_group=object(),
                                shard=slice(5, 10), sp_chunk=5, sp_tokens=10, tp=1)  # fmt: skip
    with torch.no_grad():
        out = glm5_next._mixer(layer, whole[5:], ctx)
    assert seen == {"x": [7.0, 8.0, 9.0], "cu": [0, 3], "prompt_lens": [3], "layout": None}
    assert torch.equal(out, whole[5:] * 10)


def test_sp_ranks_split_kda_only_if_each_gets_requests_and_mla_only_if_each_gets_queries():
    import types

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import _sp_spans

    def spans(cu, tokens, kda):
        ctx = types.SimpleNamespace(tp=1, sp_chunk=-(-tokens // 2), sp_tokens=tokens, sp_cache={}, turns=None,
                                    cu_seqlens=torch.tensor(cu, dtype=torch.int32))  # fmt: skip
        spans_of, restrict = _sp_spans(ctx, 2, kda)
        return [spans_of(s) for s in range(2)], restrict

    assert spans([0, 2, 7, 10], 10, kda=True) == ([[(0, 7)], [(7, 10)]], True)
    # One request: no boundary to cut at. Each rank keeps its owner's rows (the run covers the whole request).
    assert spans([0, 10], 10, kda=True) == ([[(0, 5)], [(5, 10)]], True)
    # One token: rank 1 would compute nothing, so both compute everything.
    assert spans([0, 1], 1, kda=False) == ([[(0, 1)], []], False)


def test_a_turn_plan_rows_kda_replay_cut_out_of_the_batch_is_the_rows_own():
    import types

    from arctic_platform.model.implementations.vllm_aligned import glm5_next
    from arctic_platform.model.implementations.vllm_aligned.turn_plan import encode_turns

    cpu = torch.device("cpu")
    first = encode_turns(8, [{"start": 0, "chunk_ends": [8], "prompt_end": 8, "end": 8, "outputs": []}], block_size=4)
    second = encode_turns(12, [{"start": 0, "chunk_ends": [8], "prompt_end": 8, "end": 10, "outputs": [7, 8]},
                               {"start": 8, "chunk_ends": [11], "prompt_end": 11, "end": 12, "outputs": [10]}],
                          block_size=4)  # fmt: skip
    first = torch.cat([first, torch.zeros_like(first)], 1)  # a column of no request, as rows of fewer turns pad
    turns = glm5_next.turn_layout(torch.cat([first, second]), torch.tensor([0, 8, 20], dtype=torch.int32),
                                  block_size=4, device=cpu)  # fmt: skip
    assert turns.row_spans == [0, 1, 3] and turns.row_bounds == [0, 8, 22]
    alone = glm5_next.turn_layout(second, torch.tensor([0, 12], dtype=torch.int32), block_size=4, device=cpu)
    seen = {}

    def partials(attn, x, cu_seqlens, config, prompt_lens, trace, tp, prefill_chunk, layout, ranks):
        seen.update(x=x.flatten().tolist(), cu=cu_seqlens.tolist(), layout=layout)
        return [x]

    ctx = types.SimpleNamespace(turns=turns, sp_cache={}, config=None, tp=1)
    glm5_next.kda_partials, original = partials, glm5_next.kda_partials
    try:
        glm5_next._kda_partials_between(None, torch.arange(22.0).reshape(22, 1), ctx, [0], 8, 22)
    finally:
        glm5_next.kda_partials = original
    assert seen["x"] == [float(t) for t in range(8, 22)] and seen["cu"] == alone.cu.tolist()
    assert _steps(seen["layout"]) == _steps(alone.kda)


def test_sequence_parallel_ranks_split_the_tensor_parallel_work_and_every_owner_gets_each_partial_once():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import _sp_tp_split

    for sp, tp, per_rank in ((4, 8, 2), (8, 8, 1), (16, 8, 1), (2, 1, 1), (3, 8, 8)):
        owned, source = _sp_tp_split(sp, tp)
        assert all(len(owned(s)) == per_rank for s in range(sp))
        for owner in range(sp):
            for r in range(tp):
                assert r in owned(source(r, owner)), (sp, tp, owner, r)
        # Ranks computing the same TP rank split its owners evenly.
        if sp > tp:
            loads = [sum(source(owned(s)[0], o) == s for o in range(sp)) for s in range(sp)]
            assert len(set(loads)) == 1


def test_the_loader_sets_only_the_aligned_forwards_switches():
    import pytest
    from pydantic import ValidationError

    from arctic_platform.model.loaders.flash_moe import Glm5NextOptions

    options = Glm5NextOptions.model_validate({"vllm_aligned": True,
                                              "vllm_aligned_env": {"ARCTIC_VLLM_ALIGNED_SP_QUERIES": "0"}})  # fmt: skip
    assert options.vllm_aligned_env == {"ARCTIC_VLLM_ALIGNED_SP_QUERIES": "0"}
    with pytest.raises(ValidationError, match="ARCTIC_VLLM_ALIGNED_"):
        Glm5NextOptions.model_validate({"vllm_aligned_env": {"NCCL_ALGO": "Ring"}})
