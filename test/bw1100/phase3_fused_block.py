#!/usr/bin/env python3
"""BW1100 Phase3 H4 block-chain measurement on DTK/HIP.

The chain launches the same *dtype and shapes* as Phase2 in one HIP timed
region, but deliberately fixes all FP8 Linear kernels to gfx938 Triton so the
whole chain remains in-process.  Phase2 ``auto`` may select the external
hipBLASLt executable for a few skinny shapes; treat that backend difference as
part of the H4 sensitivity, not as a pure fusion delta.
It measures scheduling/launch interaction without CUDA _scaled_mm or graphs.
For MoE experts, active-equivalent full-token GEMMs are an explicit abstract
proxy; it is not a production grouped-expert kernel.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from triton_fp8_gemm import _fp8_gemm


@triton.jit
def _grouped_expert_fp8_kernel(
        a, w, c, counts, m_stride, n: tl.constexpr, k: tl.constexpr,
        sa_e: tl.constexpr, sa_m: tl.constexpr, sa_k: tl.constexpr,
        sw_e: tl.constexpr, sw_k: tl.constexpr, sw_n: tl.constexpr,
        sc_e: tl.constexpr, sc_m: tl.constexpr, sc_n: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, GRID_M: tl.constexpr,
        GRID_N: tl.constexpr):
    """One-launch grouped GEMM with distinct weights and ragged token counts."""
    pid = tl.program_id(0)
    expert = pid // (GRID_M * GRID_N)
    # A full 256-expert tensor has offsets beyond INT32_MAX (for example
    # 255*7168*2048 FP8 elements).  DTK Triton otherwise performs the expert
    # stride product in int32 and wraps the pointer into unmapped VRAM.
    expert64 = expert.to(tl.int64)
    local = pid - expert * GRID_M * GRID_N
    pm, pn = local // GRID_N, local % GRID_N
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    count = tl.load(counts + expert64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for ko in range(0, k, BLOCK_K):
        kk = ko + rk
        av = tl.load(a + expert64 * sa_e + rm[:, None] * sa_m
                     + kk[None, :] * sa_k,
                     mask=(rm[:, None] < count) & (kk[None, :] < k), other=0.)
        wv = tl.load(w + expert64 * sw_e + kk[:, None] * sw_k
                     + rn[None, :] * sw_n,
                     mask=(kk[:, None] < k) & (rn[None, :] < n), other=0.)
        acc += tl.dot(av, wv)
    tl.store(c + expert64 * sc_e + rm[:, None] * sc_m + rn[None, :] * sc_n,
             acc.to(tl.bfloat16),
             mask=(rm[:, None] < count) & (rn[None, :] < n))


def measure_grouped_expert(m_tokens: int, n: int, k: int,
                           routed_experts: int = 256, topk: int = 8,
                           shared_experts: int = 1, warmup: int = 5,
                           iters: int = 20, min_ms: float = 100.0,
                           max_iters: int = 100):
    """Measure routing-faithful grouped FP8 GEMM; return latency and metadata.

    Counts are balanced and deterministic. Routed groups collectively process
    ``m_tokens*topk`` rows and each shared expert processes ``m_tokens`` rows.
    All groups use different weights, so HBM traffic represents distinct expert
    residency rather than repeated reuse of one matrix.
    """
    routed_rows = m_tokens * topk
    base, rem = divmod(routed_rows, routed_experts)
    host_counts = [base + (i < rem) for i in range(routed_experts)]
    max_m = max(host_counts)
    storage = torch.float8_e4m3fn
    # Padded expert-major activations make expert offsets compile-time cheap;
    # masks ensure padding performs no logical work.
    # Allocate FP8 directly.  ``torch.ones(..., bf16).to(fp8)`` temporarily
    # requires both BF16 and FP8 copies and adds a 7 GiB peak for Gate/Up.
    a = torch.empty(routed_experts, max_m, k, device='cuda', dtype=storage)
    a.fill_(1)
    w = torch.empty(routed_experts, k, n, device='cuda', dtype=storage)
    w.fill_(1)
    c = torch.empty(routed_experts, max_m, n, device='cuda',
                    dtype=torch.bfloat16)
    counts = torch.tensor(host_counts, device='cuda', dtype=torch.int32)
    bm, bn, bk = 32, 64, 64
    gm, gn = triton.cdiv(max_m, bm), triton.cdiv(n, bn)
    grid = (routed_experts * gm * gn,)
    shared_runs = [_fp8_run(m_tokens, n, k) for _ in range(shared_experts)]
    def run():
        _grouped_expert_fp8_kernel[grid](
            a, w, c, counts, max_m, n, k,
            a.stride(0), a.stride(1), a.stride(2),
            w.stride(0), w.stride(1), w.stride(2),
            c.stride(0), c.stride(1), c.stride(2),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            GRID_M=gm, GRID_N=gn, num_warps=4, num_stages=1)
        for shared_run in shared_runs:
            shared_run()
    latency = _event_time(run, warmup, iters, min_ms, max_iters)
    meta = {
        'routed_experts': routed_experts,
        'shared_experts': shared_experts,
        'distinct_experts': routed_experts + shared_experts,
        'routed_assignments': routed_rows,
        'shared_assignments': m_tokens * shared_experts,
        'max_tokens_per_expert': max_m,
        'backend': 'triton_grouped_fp8_one_launch',
    }
    del a, w, c, counts
    torch.cuda.empty_cache()
    return latency, meta


def _event_time(run: Callable[[], None], warmup: int, iters: int,
                min_ms: float, max_iters: int) -> float:
    for _ in range(max(1, warmup)): run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); run(); end.record(); end.synchronize()
    n = max(1, iters, math.ceil(min_ms / max(start.elapsed_time(end), .01)))
    n = min(n, max_iters)
    values = []
    for _ in range(3):
        start.record()
        for _ in range(n): run()
        end.record(); end.synchronize()
        values.append(start.elapsed_time(end) / 1000.0 / n)
    values.sort()
    return values[len(values) // 2]


def _fp8_run(m: int, n: int, k: int, repeats: int = 1) -> Callable[[], None]:
    storage = torch.float8_e4m3fn
    a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16).to(storage)
    b = torch.ones(n, k, device="cuda", dtype=torch.bfloat16).to(storage).t()
    c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    bm = 64 if m < 128 else 128; bn = 64 if n < 128 else 128
    bk = 32 if k <= 32 else 64 if k <= 64 else 128
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    def run():
        for _ in range(repeats):
            _fp8_gemm[grid](a, b, c, m, n, k, a.stride(0), a.stride(1),
                b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=4, num_stages=1)
    return run


def _bmm_run(batch: int, m: int, n: int, k: int, repeats: int = 1):
    a = torch.randn(batch, m, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(batch, n, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(batch, m, k, device="cuda", dtype=torch.bfloat16)
    def run():
        for _ in range(repeats): torch.bmm(a, b, out=out)
    return run


def _norm_run(elements: int, width: int):
    x = torch.randn(max(1, elements // width), width, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(width, device="cuda", dtype=torch.bfloat16)
    return lambda: F.rms_norm(x, (width,), w)


def _sigmoid_run(elements: int):
    x = torch.randn(elements, device="cuda", dtype=torch.bfloat16)
    return lambda: torch.sigmoid(x)


def build_kernel_chain(catalog_rows, app, stage: str, active_equiv: float):
    runs = []; counts = {"linear": 0, "bmm": 0, "norm": 0, "vector": 0}
    moe_proxy = False
    for r in catalog_rows:
        if r.flops <= 0 and r.pred_max_s <= 0: continue
        if r.cls == "Linear" and r.c_in and r.c_out:
            if stage == "fw": m, n, k = r.batch_seq, r.c_out, r.c_in
            elif stage == "agrad": m, n, k = r.batch_seq, r.c_in, r.c_out
            else: m, n, k = r.c_in, r.c_out, r.batch_seq
            repeats = 1
            if r.group == "G4" and "MlpBlock_MoE_" in r.name:
                repeats = max(1, round(active_equiv)); moe_proxy = True
            runs.append(_fp8_run(m, n, k, repeats)); counts["linear"] += 1
        elif r.cls == "BatchMatMul" and r.bmm_batch and stage != "wgrad":
            runs.append(_bmm_run(r.bmm_batch, r.bmm_m, r.bmm_n, r.bmm_k,
                                  2 if stage == "agrad" else 1)); counts["bmm"] += 1
        elif r.cls in ("LayerNorm", "RMSNorm") and r.act_size:
            width = max(1, r.act_size // max(r.batch_seq, 1))
            runs.append(_norm_run(r.act_size, width)); counts["norm"] += 1
        elif r.cls == "RouterSigmoid":
            runs.append(_sigmoid_run(max(1, int(r.flops // 4)))); counts["vector"] += 1
    return runs, counts, moe_proxy


def measure_fused_block(app, block: str, stage: str, microbatch: int = 1,
                        warmup: int = 10, iters: int = 50,
                        min_ms: float = 200.0, max_iters: int = 500,
                        fused_impl: str = "kernel_chain", catalog_rows: Optional[Sequence] = None,
                        matrix_dtype: str = "float8", active_equiv: float = 9.0):
    if fused_impl != "kernel_chain":
        return None, "BW1100 supports kernel_chain only; no production fused absorb kernel is installed"
    if stage not in ("fw", "agrad", "wgrad") or catalog_rows is None:
        return None, "invalid stage or missing catalog_rows"
    if matrix_dtype != "float8":
        return None, "BW1100 Phase3 kernel_chain currently validates FP8 matrix path only"
    runs, counts, moe_proxy = build_kernel_chain(catalog_rows, app, stage, active_equiv)
    if not runs: return None, "empty kernel chain"
    def run_all():
        for run in runs: run()
    try:
        latency = _event_time(run_all, warmup, iters, min_ms, max_iters)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        torch.cuda.empty_cache(); return None, f"{type(exc).__name__}: {str(exc)[:160]}"
    note = (f"HIP kernel_chain {counts}; fixed Triton FP8 + torch.bmm BF16 + RMSNorm; "
            "Phase2 auto may use hipBLASLt for skinny shapes")
    if moe_proxy: note += "; MoE=active-equivalent full-token proxy (not grouped routing)"
    return latency, note
