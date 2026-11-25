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

"""Debug: profile a window of engine steps in a running sampler (TP rank 0) and log one summary line.

``ARCTIC_GLM5_SAMPLER_PROFILE=skip,steps,kernel_steps`` skips ``skip`` engine steps, then for ``steps`` steps records
each step's scheduled tokens and requests, whether it decodes only, its GPU time (CUDA events from the start of
``execute_model`` to the end of ``sample_tokens``, read once at the end) and its wall time and the gap before it; the
last ``kernel_steps`` steps also run under ``torch.profiler``. The summary (per step kind and batch size, and the
top kernels) is logged as ``arctic sampler profile: {json}``.
"""

from __future__ import annotations

import json
import logging
import os
import time

import torch

ENV = "ARCTIC_GLM5_SAMPLER_PROFILE"
log = logging.getLogger(__name__)


def _bucket(meta: dict) -> str:
    if meta["decode"]:
        reqs = meta["reqs"]
        edge = next((e for e in (8, 16, 32, 64, 128) if reqs <= e), 256)
        return f"decode<={edge} rows"
    tokens = meta["tokens"]
    edge = next((e for e in (512, 2048, 4096, 8192) if tokens <= e), 16384)
    return f"prefill/mixed<={edge} tokens"


def _summary(rows: list, prof) -> dict:
    torch.cuda.synchronize()
    out: dict = {"steps": len(rows), "kinds": {}}
    previous_end = None
    gaps = []
    for meta, start, end, t0, t1 in rows:
        gpu = start.elapsed_time(end)
        kind = out["kinds"].setdefault(_bucket(meta), {"steps": 0, "tokens": 0, "reqs": 0, "gpu_ms": 0.0, "wall_ms": 0.0})
        kind["steps"] += 1
        kind["tokens"] += meta["tokens"]
        kind["reqs"] += meta["reqs"]
        kind["gpu_ms"] += gpu
        kind["wall_ms"] += (t1 - t0) * 1e3
        if previous_end is not None:
            gaps.append((t0 - previous_end) * 1e3)
        previous_end = t1
    for kind in out["kinds"].values():
        kind["gpu_ms"], kind["wall_ms"] = round(kind["gpu_ms"], 1), round(kind["wall_ms"], 1)
    span = (rows[-1][4] - rows[0][3]) * 1e3 if rows else 0.0
    out.update(span_ms=round(span, 1), gpu_ms=round(sum(k["gpu_ms"] for k in out["kinds"].values()), 1),
               gap_ms=round(sum(gaps), 1), tokens=sum(r[0]["tokens"] for r in rows))  # fmt: skip
    if prof is not None:
        events = sorted(prof.key_averages(), key=lambda e: -getattr(e, "device_time_total", 0))
        out["kernels"] = [{"name": e.key[:80], "calls": e.count, "ms": round(e.device_time_total / 1e3, 1)}
                          for e in events[:40] if getattr(e, "device_time_total", 0) > 0]  # fmt: skip
    return out


def install() -> None:
    spec = os.environ.get(ENV)
    if not spec:
        return
    skip, steps, kernel_steps = (int(v) for v in spec.split(","))
    from vllm.distributed import get_tp_group
    from vllm.v1.worker import gpu_worker

    cls = gpu_worker.Worker
    original_execute, original_sample = cls.execute_model, cls.sample_tokens
    state: dict = {"n": 0, "rows": [], "open": None, "prof": None, "done": False}

    def close(now: float) -> None:
        if state["open"] is not None:
            meta, start, t0 = state["open"]
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            state["rows"].append((meta, start, end, t0, now))
            state["open"] = None

    def execute_model(self, scheduler_output):
        if state["done"] or get_tp_group().rank_in_group != 0:
            return original_execute(self, scheduler_output)
        close(time.perf_counter())
        n = state["n"]
        state["n"] += 1
        if skip <= n < skip + steps and scheduler_output.total_num_scheduled_tokens > 0:
            if n >= skip + steps - kernel_steps and state["prof"] is None:
                state["prof"] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
                state["prof"].__enter__()
            counts = scheduler_output.num_scheduled_tokens
            meta = {"tokens": int(scheduler_output.total_num_scheduled_tokens), "reqs": len(counts),
                    "decode": all(v == 1 for v in counts.values())}  # fmt: skip
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            state["open"] = (meta, start, time.perf_counter())
        elif n >= skip + steps and not state["done"]:
            state["done"] = True
            prof = state["prof"]
            if prof is not None:
                prof.__exit__(None, None, None)
            log.warning("arctic sampler profile: %s", json.dumps(_summary(state["rows"], prof)))
            state["rows"] = []
        return original_execute(self, scheduler_output)

    def sample_tokens(self, grammar_output):
        out = original_sample(self, grammar_output)
        if not state["done"] and get_tp_group().rank_in_group == 0:
            close(time.perf_counter())
        return out

    cls.execute_model = execute_model
    cls.sample_tokens = sample_tokens
    log.warning("arctic sampler profile: installed (skip %d, %d steps, kernels on the last %d)", skip, steps,
                kernel_steps)  # fmt: skip
