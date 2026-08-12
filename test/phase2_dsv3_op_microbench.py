#!/usr/bin/env python3
"""Phase2: DeepSeek-V3 single-op microbench (H2) vs Calculon catalog preds.

Measures isolated CUDA kernels that correspond to compiled DS-V3 layers and
compares against pred_max = max(flops_time, mem_time) from H20.json.

Groups (see phase2 design):
  G1 MLA Linears   — FP8 GEMM
  G2 absorb BMMs   — FP8 batched GEMM (default) or dtype-matched BF16 bmm
  G3 Dense SwiGLU  — FP8 GEMM + SiLU
  G4 MoE           — G4a physical expert×active_equiv vs G4b abstract pred
  G5 vector        — RMSNorm / Softmax / SiLU
  G6 WUK/WUV       — assert fw_flops==0 (no GEMM)

Example:
  # Catalog-only (no CUDA measure)
  python test/phase2_dsv3_op_microbench.py --predict-only --groups G1 G3 G6

  # Main path on H20
  python test/phase2_dsv3_op_microbench.py --groups G1 G3 G6 \\
      --csv test/phase2_g1g3.csv

  # Absorb BMM (FP8 meas vs float8 pred; optional bf16 matched pair)
  python test/phase2_dsv3_op_microbench.py --groups G2 --seq-size 1024
  python test/phase2_dsv3_op_microbench.py --groups G2 --seq-size 1024 \\
      --g2-kernel both --csv test/phase2_g2.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Sequence, Tuple

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TEST_DIR)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from calibrate_h20_common import (  # noqa: E402
    normalize_dtype,
    torch_storage_dtype,
)
from calibrate_h20_matrix_efficiency import (  # noqa: E402
    _align,
    _gemm_fp8,
    benchmark_shape,
)
from phase2_dsv3_op_catalog import (  # noqa: E402
    CatalogRow,
    build_catalog,
    compile_dsv3,
)
from calculon.system import System  # noqa: E402
from calculon.llm.layers import BatchMatMul  # noqa: E402

# Deduplicate: MoE block repeats MLA ops already covered by dense.
_DEDUP_ATTN_IN_MOE = True


@dataclass
class BenchRow:
    group: str
    track: str                 # abstract | physical | assert
    block: str
    name: str
    cls: str
    stage: str
    kernel: str
    # shape summary
    m: int
    n: int
    k: int
    batch: int
    flops: float
    bytes: float
    pred_f_s: float
    pred_m_s: float
    pred_max_s: float
    bound: str
    flop_mult: float
    weight_mult: float
    meas_s: Optional[float]
    err_max_pct: Optional[float]
    skipped: str
    notes: str


def _rel_err_pct(pred: float, meas: float) -> float:
    if meas <= 0:
        return float('nan')
    return 100.0 * (pred - meas) / meas


def _mape(errs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [abs(e) for e in errs if e is not None and not math.isnan(e)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _cuda_time(run: Callable[[], None], warmup: int, iters: int, min_ms: float,
               max_iters: int = 500) -> float:
    import torch
    for _ in range(max(1, warmup)):
        run()
    torch.cuda.synchronize()
    # probe
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        run()
    end.record()
    torch.cuda.synchronize()
    lat_ms = start.elapsed_time(end) / 5.0
    n = max(1, iters)
    if min_ms > 0:
        n = max(n, int(min_ms / max(lat_ms, 0.01)) + 1)
    n = min(n, max_iters)
    start.record()
    for _ in range(n):
        run()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / 1000.0) / n


def bench_linear_gemm(
    batch_seq: int, c_in: int, c_out: int, dtype: str,
    warmup: int, iters: int, min_ms: float, max_iters: int,
) -> float:
    """Linear A[m,c_in]@W[c_in,c_out] → benchmark_shape(m, n=c_out, k=c_in)."""
    m, n, k = _align(batch_seq), _align(c_out), _align(c_in)
    _, _, lat, _ = benchmark_shape(
        m, n, k, dtype, warmup, iters, min_ms, max_iters=max_iters)
    return lat


def resolve_g2_kernels(g2_kernel: str, matrix_dtype: str) -> List[str]:
    """Return meas/pred dtype tags for G2: 'float8' and/or 'bfloat16'.

    auto → bfloat16 matched pair (native ``torch.bmm`` + ``matrix.bfloat16``
    pred). FP8 has no batched API on PyTorch; ``fp8`` uses serial
    ``_scaled_mm`` under a CUDA graph (launch-heavy vs Calculon's aggregate
    BMM efficiency) — use for sensitivity only, or ``both``.
    """
    key = g2_kernel.strip().lower()
    if key == 'auto':
        return ['bfloat16']
    if key == 'fp8' or key == 'float8':
        return ['float8']
    if key in ('bf16', 'bfloat16'):
        return ['bfloat16']
    if key == 'both':
        return ['float8', 'bfloat16']
    raise SystemExit(
        f'Unknown --g2-kernel={g2_kernel!r}. Use auto|fp8|bf16|both.')


def predict_bmm(
    syst: System,
    batch: int,
    m: int,
    n: int,
    k: int,
    pred_dtype: str,
    vector_dtype: str,
    layer_name: str = 'g2_tmp',
) -> Tuple[float, float, float, float, float]:
    """Calculon BatchMatMul pred under pred_dtype (restores syst dtypes).

    Applies ``bmm_time_scale`` from H20.json based on ``layer_name``
    (attn_score vs absorb), matching Llm._append_bmm.
    """
    prev_override = getattr(syst, '_bmm_dtype_override', None)
    try:
        # Force BMM peak/mem width to pred_dtype (Phase2 G2 sensitivity).
        syst._bmm_dtype_override = pred_dtype
        kind = System.bmm_scale_kind(layer_name)
        scale = syst.get_bmm_time_scale(kind)
        layer = BatchMatMul(
            layer_name, syst, batch, m, n, k, time_scale=scale)
        layer.set_bytes_per_element(System.TypeSizes[pred_dtype])
        flops = float(layer.get_fw_flops())
        nbytes = float(layer.get_fw_mem_accessed())
        pred_f = float(layer.compute_flops_time('fw'))
        pred_m = float(layer.compute_mem_time('fw'))
        pred_max = float(syst.get_processing_time(pred_f, pred_m))
        return flops, nbytes, pred_f, pred_m, pred_max
    finally:
        syst._bmm_dtype_override = prev_override


def bench_bmm_bf16(
    batch: int, m: int, n: int, k: int,
    warmup: int, iters: int, min_ms: float, max_iters: int,
) -> Tuple[float, int, int, int]:
    """A[B,M,N] @ B[B,N,K] via torch.bmm (BF16). Returns (lat, m, n, k)."""
    import torch
    m, n, k = _align(m), _align(n), _align(k)
    device = torch.device('cuda')
    a = torch.randn(batch, m, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(batch, n, k, device=device, dtype=torch.bfloat16)
    out = torch.empty(batch, m, k, device=device, dtype=torch.bfloat16)

    def run():
        torch.bmm(a, b, out=out)

    return _cuda_time(run, warmup, iters, min_ms, max_iters), m, n, k


def bench_bmm_bf16_bwd(
    batch: int, m: int, n: int, k: int,
    warmup: int, iters: int, min_ms: float, max_iters: int,
) -> Tuple[float, int, int, int]:
    """BF16 bmm *backward* (≈2× matmul; matches BatchMatMul agrad_flops)."""
    import torch
    m, n, k = _align(m), _align(n), _align(k)
    device = torch.device('cuda')
    a0 = torch.randn(batch, m, n, device=device, dtype=torch.bfloat16)
    b0 = torch.randn(batch, n, k, device=device, dtype=torch.bfloat16)

    def run():
        a = a0.detach().requires_grad_(True)
        b = b0.detach().requires_grad_(True)
        torch.bmm(a, b).sum().backward()

    return _cuda_time(run, warmup, iters, min_ms, max_iters), m, n, k


def linear_gemm_mnk(stage: str, batch_seq: int, c_in: int, c_out: int,
                    ) -> Tuple[int, int, int]:
    """Return (m, n, k) for Linear fw / agrad / wgrad GEMM shapes."""
    if stage == 'fw':
        return batch_seq, c_out, c_in
    if stage == 'agrad':
        # dX = dY @ W.T
        return batch_seq, c_in, c_out
    if stage == 'wgrad':
        # dW = X.T @ dY
        return c_in, c_out, batch_seq
    raise ValueError(stage)


def bench_bmm_fp8(
    batch: int, m: int, n: int, k: int,
    warmup: int, iters: int, min_ms: float, max_iters: int,
) -> Tuple[float, int, int, int]:
    """FP8 batched GEMM matching Calculon BMM flops (B·2·M·N·K).

    No native FP8 bmm: run per-batch ``_scaled_mm`` under a CUDA graph so
    launch overhead does not dominate (plain Python loops are 10×+ too slow).
    """
    import torch
    m, n, k = _align(m), _align(n), _align(k)
    device = torch.device('cuda')
    storage = torch_storage_dtype('float8')
    a = torch.randn(batch, m, n, device=device, dtype=torch.bfloat16).to(storage)
    b = torch.randn(batch, n, k, device=device, dtype=torch.bfloat16)
    b_nk = b.transpose(-2, -1).contiguous().to(storage)  # [B, K, N]
    scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
    scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
    out = torch.empty(batch, m, k, device=device, dtype=torch.bfloat16)
    aux = torch.empty((), device=device, dtype=torch.float32)

    def _one_batch_pass():
        for i in range(batch):
            _gemm_fp8(a[i], b_nk[i], scale_a, scale_b, out_pair=(out[i], aux))

    # Warmup + try CUDA graph (removes per-slice Python launch tax).
    for _ in range(max(3, warmup)):
        _one_batch_pass()
    torch.cuda.synchronize()

    graph = None
    static_run = _one_batch_pass
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _one_batch_pass()
        graph = g

        def static_run():
            graph.replay()
    except Exception:
        # Graph capture unsupported for this shape/path — fall back to loop.
        static_run = _one_batch_pass

    return _cuda_time(static_run, 0, iters, min_ms, max_iters), m, n, k


def bench_rms_norm(
    act_size: int, hidden: int, warmup: int, iters: int, min_ms: float,
    max_iters: int,
) -> float:
    import torch
    device = torch.device('cuda')
    tokens = max(1, act_size // max(hidden, 1))
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    w = torch.ones(hidden, device=device, dtype=torch.bfloat16)
    eps = 1e-6
    if hasattr(torch.nn.functional, 'rms_norm'):
        def run():
            torch.nn.functional.rms_norm(x, (hidden,), w, eps)
    else:
        def run():
            # PyTorch < 2.4 fallback
            y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            y.mul_(w)

    return _cuda_time(run, warmup, iters, min_ms, max_iters)


def bench_softmax(
    act_size: int, heads: int, seq: int, warmup: int, iters: int,
    min_ms: float, max_iters: int,
) -> float:
    import torch
    device = torch.device('cuda')
    # act_size ≈ heads * S * S * mbs
    mbs = max(1, act_size // max(heads * seq * seq, 1))
    x = torch.randn(mbs, heads, seq, seq, device=device, dtype=torch.float32)

    def run():
        torch.softmax(x, dim=-1)

    return _cuda_time(run, warmup, iters, min_ms, max_iters)


def bench_silu(
    act_size: int, warmup: int, iters: int, min_ms: float, max_iters: int,
) -> float:
    import torch
    device = torch.device('cuda')
    x = torch.randn(act_size, device=device, dtype=torch.bfloat16)

    def run():
        torch.nn.functional.silu(x)

    return _cuda_time(run, warmup, iters, min_ms, max_iters)


def select_rows(
    catalog: Sequence[CatalogRow],
    groups: Sequence[str],
    stage: str,
    blocks: Sequence[str],
) -> List[CatalogRow]:
    out: List[CatalogRow] = []
    seen_names = set()
    for r in catalog:
        if r.stage != stage:
            continue
        if r.group not in groups:
            continue
        if r.block not in blocks:
            continue
        if _DEDUP_ATTN_IN_MOE and r.block == 'moe' and r.name.startswith('AttnBlock_'):
            # Prefer dense copy of identical MLA ops.
            continue
        # Prefer dense for shared G5 norms when both exist; keep MoE-only ops.
        key = (r.name, r.stage)
        if key in seen_names and r.block == 'moe' and r.group != 'G4':
            continue
        seen_names.add(key)
        out.append(r)
    return out


def measure_row(
    r: CatalogRow,
    matrix_dtype: str,
    app_seq: int,
    app_hidden: int,
    app_heads: int,
    active_equiv: float,
    warmup: int,
    iters: int,
    min_ms: float,
    max_iters: int,
    measure: bool,
    syst: Optional[System] = None,
    vector_dtype: str = 'bfloat16',
    g2_kernels: Optional[Sequence[str]] = None,
) -> List[BenchRow]:
    """Return one or more BenchRows (G4 emits abstract + physical)."""
    def make(
        track: str, kernel: str, meas: Optional[float],
        skipped: str = '', notes: str = '',
        pred_max: Optional[float] = None,
        pred_f: Optional[float] = None,
        pred_m: Optional[float] = None,
        flops: Optional[float] = None,
        bytes_: Optional[float] = None,
        bound: Optional[str] = None,
        m: Optional[int] = None, n: Optional[int] = None, k: Optional[int] = None,
        batch: Optional[int] = None,
    ) -> BenchRow:
        pm = r.pred_max_s if pred_max is None else pred_max
        pf = r.pred_f_s if pred_f is None else pred_f
        pmem = r.pred_m_s if pred_m is None else pred_m
        return BenchRow(
            group=r.group, track=track, block=r.block, name=r.name, cls=r.cls,
            stage=r.stage, kernel=kernel,
            m=r.batch_seq if m is None else m,
            n=(r.c_out or r.bmm_k) if n is None else n,
            k=(r.c_in or r.bmm_n) if k is None else k,
            batch=r.bmm_batch if batch is None else batch,
            flops=r.flops if flops is None else flops,
            bytes=r.bytes if bytes_ is None else bytes_,
            pred_f_s=pf, pred_m_s=pmem, pred_max_s=pm,
            bound=bound if bound is not None else r.bound,
            flop_mult=r.flop_mult, weight_mult=r.weight_mult,
            meas_s=meas,
            err_max_pct=_rel_err_pct(pm, meas) if meas is not None else None,
            skipped=skipped, notes=notes or r.notes,
        )

    if r.group == 'G6':
        ok = r.flops == 0
        return [make(
            'assert', 'none', None,
            skipped='' if ok else 'FAIL_nonzero_flops',
            notes=('PASS fw_flops==0' if ok else f'fw_flops={r.flops}'),
        )]

    # Fused SoftMax / SiLU: assert even under --predict-only.
    if r.cls == 'SoftMax' and r.flops == 0:
        return [make(
            'assert', 'softmax_fused', None,
            notes='PASS fused into flash-attn (KNOWN_GAP vs isolated '
                  'torch.softmax; not charged in block time)')]
    if r.cls in ('GeLU', 'SiLU') and r.flops == 0:
        return [make(
            'assert', 'fused', None,
            notes='PASS fused activation (no standalone compute)')]

    # G2 before predict-only early-return so dtype-matched pred always runs.
    if r.cls == 'BatchMatMul' and r.bmm_batch > 0:
        if r.stage == 'wgrad':
            return [make(
                'assert', 'bmm_no_wgrad', None,
                notes='PASS BatchMatMul has no weights (wgrad=0)')]
        kernels = list(g2_kernels or resolve_g2_kernels('auto', matrix_dtype))
        if syst is None:
            return [make('abstract', 'bmm', None,
                         skipped='no_system_for_g2_pred')]
        out_rows: List[BenchRow] = []
        scale_kind = System.bmm_scale_kind(r.name)
        scale = syst.get_bmm_time_scale(scale_kind)
        for kd in kernels:
            # Catalog row preds are stage-correct (fw vs agrad=2×).
            # predict_bmm() is fw-only; use it for fw dtype match, else catalog.
            if r.stage == 'fw':
                flops, nbytes, pf, pm, pmax = predict_bmm(
                    syst, r.bmm_batch, r.bmm_m, r.bmm_n, r.bmm_k,
                    kd, vector_dtype, layer_name=r.name)
            else:
                flops, nbytes = r.flops, r.bytes
                pf, pm, pmax = r.pred_f_s, r.pred_m_s, r.pred_max_s
                # Re-price flops_time under kd if needed (catalog already bmm_dtype).
                if kd != (syst.get_bmm_dtype() if syst else kd):
                    flops, nbytes, pf, pm, pmax = predict_bmm(
                        syst, r.bmm_batch, r.bmm_m, r.bmm_n, r.bmm_k,
                        kd, vector_dtype, layer_name=r.name)
                    # agrad ≈ 2× fw matmul time under same peak/efficiency.
                    pf, pm, pmax = pf * 2.0, pm * 2.0, pmax * 2.0
                    flops, nbytes = flops * 2.0, nbytes * 2.0
            bound = 'compute' if pf >= pm else 'memory'
            if not measure:
                out_rows.append(make(
                    'abstract', f'bmm_{kd}', None,
                    skipped='predict-only',
                    notes=(f'stage={r.stage} pred_dtype={kd} '
                           f'bmm_scale={scale_kind}:{scale}'),
                    pred_max=pmax, pred_f=pf, pred_m=pm,
                    flops=flops, bytes_=nbytes, bound=bound,
                    m=r.bmm_m, n=r.bmm_k, k=r.bmm_n, batch=r.bmm_batch,
                ))
                continue
            import torch
            try:
                if r.stage == 'agrad':
                    # Calculon BatchMatMul.agrad_flops = 2×fw; validate that
                    # model with 2× forward kernel (same shape). Autograd bwd
                    # has extra overhead and is not what _block_agrad charges.
                    if kd == 'float8':
                        lat1, ma, na, ka = bench_bmm_fp8(
                            r.bmm_batch, r.bmm_m, r.bmm_n, r.bmm_k,
                            warmup, iters, min_ms, max_iters)
                    else:
                        lat1, ma, na, ka = bench_bmm_bf16(
                            r.bmm_batch, r.bmm_m, r.bmm_n, r.bmm_k,
                            warmup, iters, min_ms, max_iters)
                    lat = lat1 * 2.0
                    kern = f'bmm_{kd}_x2_agrad'
                    note = (f'agrad=2×fw_{kd}_bmm (matches agrad_flops=2×fw); '
                            f'bmm_scale={scale_kind}:{scale}')
                elif kd == 'float8':
                    lat, ma, na, ka = bench_bmm_fp8(
                        r.bmm_batch, r.bmm_m, r.bmm_n, r.bmm_k,
                        warmup, iters, min_ms, max_iters)
                    kern = 'bmm_fp8'
                    note = (f'FP8 batched _scaled_mm (CUDA graph); '
                            f'bmm_scale={scale_kind}:{scale}')
                else:
                    lat, ma, na, ka = bench_bmm_bf16(
                        r.bmm_batch, r.bmm_m, r.bmm_n, r.bmm_k,
                        warmup, iters, min_ms, max_iters)
                    kern = 'bmm_bf16'
                    note = (f'BF16 bmm; pred also bfloat16; '
                            f'bmm_scale={scale_kind}:{scale}')
                if r.stage == 'fw' and (ma, na, ka) != (r.bmm_m, r.bmm_n, r.bmm_k):
                    flops, nbytes, pf, pm, pmax = predict_bmm(
                        syst, r.bmm_batch, ma, na, ka, kd, vector_dtype,
                        layer_name=r.name)
                    bound = 'compute' if pf >= pm else 'memory'
                out_rows.append(make(
                    'physical', kern, lat, notes=note,
                    pred_max=pmax, pred_f=pf, pred_m=pm,
                    flops=flops, bytes_=nbytes, bound=bound,
                    m=ma, n=ka, k=na, batch=r.bmm_batch,
                ))
            except Exception as e:
                # OOM / CUDA / scaled_mm shape errors
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                out_rows.append(make(
                    'physical', f'bmm_{kd}', None,
                    skipped=f'{type(e).__name__}: {str(e)[:120]}',
                    pred_max=pmax, pred_f=pf, pred_m=pm,
                    flops=flops, bytes_=nbytes, bound=bound,
                    m=r.bmm_m, n=r.bmm_k, k=r.bmm_n, batch=r.bmm_batch,
                ))
        return out_rows

    if not measure:
        track = 'abstract'
        skipped = 'predict-only'
        notes = r.notes
        if r.group == 'G4' and 'MoE_' in r.name and r.flop_mult > 1.01:
            notes = (f'flop_mult={r.flop_mult:.2f} weight_mult={r.weight_mult:.1f}; '
                     f'run without --predict-only for G4a physical×active')
        return [make(track, 'none', None, skipped=skipped, notes=notes)]

    import torch

    try:
        # ----- Linear GEMM (G1 / G3 / G4) -----
        if r.cls == 'Linear' and r.c_in > 0 and r.c_out > 0:
            m_g, n_g, k_g = linear_gemm_mnk(
                r.stage, r.batch_seq, r.c_in, r.c_out)
            if r.group == 'G4' and 'MoE_' in r.name and r.flop_mult > 1.01:
                # G4b: abstract pred (no direct measure of 257-expert weight)
                abstract = make('abstract', 'catalog_pred', None,
                                skipped='abstract_no_direct_meas',
                                notes=f'flop_mult={r.flop_mult:.2f} '
                                      f'weight_mult={r.weight_mult:.1f}')
                # G4a: single-expert physical × active_equiv
                lat1 = bench_linear_gemm(
                    m_g, k_g, n_g, matrix_dtype,
                    warmup, iters, min_ms, max_iters)
                # bench_linear_gemm(batch_seq, c_in, c_out) expects
                # m=batch_seq, n=c_out, k=c_in — remap via (m, k, n).
                physical = make(
                    'physical', f'fp8_gemm_x_active_{r.stage}',
                    lat1 * active_equiv,
                    notes=(f'stage={r.stage} shape=({m_g},{n_g},{k_g}) '
                           f'1expert_lat={lat1*1e6:.1f}us * '
                           f'active_equiv={active_equiv:.2f}'),
                    pred_max=r.pred_max_s,
                    m=m_g, n=n_g, k=k_g,
                )
                physical.err_max_pct = _rel_err_pct(r.pred_max_s, physical.meas_s)
                return [abstract, physical]

            # bench_linear_gemm(batch_seq, c_in, c_out) → m,n=c_out,k=c_in
            # Pass m as batch_seq, c_in=k_g, c_out=n_g.
            lat = bench_linear_gemm(
                m_g, k_g, n_g, matrix_dtype,
                warmup, iters, min_ms, max_iters)
            return [make('physical', f'fp8_gemm_{r.stage}', lat,
                         m=m_g, n=n_g, k=k_g,
                         notes=f'stage={r.stage} GEMM({m_g},{n_g},{k_g})')]

        # ----- Vector (G5 / G3 SiLU) -----
        if r.cls in ('LayerNorm', 'RMSNorm'):
            if r.act_size <= 0 and r.flops == 0:
                return [make('abstract', 'norm', None, skipped='empty_norm')]
            # hidden width: Q/KV norms use c_in-like act/tokens; pass app_hidden
            # for pre-attn/mlp, else infer from name.
            hidden = app_hidden
            if 'QNorm' in r.name:
                hidden = max(1, r.act_size // max(app_seq, 1))
            elif 'KVNorm' in r.name:
                hidden = max(1, r.act_size // max(app_seq, 1))
            lat = bench_rms_norm(
                r.act_size, hidden, warmup, iters, min_ms, max_iters)
            note = ('Calculon RMSNorm vs F.rms_norm' if r.cls == 'RMSNorm'
                    else 'Calculon LayerNorm flops vs F.rms_norm')
            return [make('physical', 'rms_norm', lat, notes=note)]

        if r.cls == 'SoftMax':
            if r.flops == 0:
                return [make(
                    'assert', 'softmax_fused', None,
                    notes='PASS fused into flash-attn (KNOWN_GAP vs isolated '
                          'torch.softmax; not charged in block time)')]
            if r.act_size > 0:
                lat = bench_softmax(
                    r.act_size, app_heads, app_seq, warmup, iters, min_ms,
                    max_iters)
                return [make(
                    'physical', 'softmax_isolated', lat,
                    notes='KNOWN_GAP: isolated softmax; training uses fused attn')]

        if r.cls in ('GeLU', 'SiLU'):
            if r.flops == 0:
                return [make(
                    'assert', 'fused', None,
                    notes='PASS fused activation (no standalone compute)')]
            if r.act_size > 0:
                lat = bench_silu(
                    r.act_size, warmup, iters, min_ms, max_iters)
                return [make('physical', 'silu', lat)]

        return [make('abstract', 'unmapped', None,
                     skipped='no_kernel_mapping')]
    except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
        torch.cuda.empty_cache()
        return [make('physical', 'error', None,
                     skipped=f'{type(e).__name__}: {str(e)[:120]}')]


def write_csv(path: str, rows: Sequence[BenchRow]) -> None:
    if not rows:
        return
    fields = list(asdict(rows[0]).keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f'\nWrote CSV: {path}  ({len(rows)} rows)')


def print_table(rows: Sequence[BenchRow]) -> None:
    hdr = (f'{"grp":3s} {"trk":8s} {"name":34s} {"kern":16s} '
           f'{"pred_us":>9s} {"meas_us":>9s} {"err%":>8s}')
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        meas = f'{r.meas_s*1e6:.1f}' if r.meas_s is not None else '—'
        err = f'{r.err_max_pct:+.1f}' if r.err_max_pct is not None else '—'
        flag = f'  {r.skipped}' if r.skipped else ''
        note = f'  {r.notes}' if r.notes and not r.skipped else ''
        print(
            f'{r.group:3s} {r.track:8s} {r.name:34s} {r.kernel:16s} '
            f'{r.pred_max_s*1e6:9.1f} {meas:>9s} {err:>8s}{flag}{note}'
        )


def summarize(rows: Sequence[BenchRow]) -> None:
    print('\n=== Phase2 microbench summary (H2) ===')
    g6 = [r for r in rows if r.group == 'G6']
    if g6:
        ok = all('PASS' in (r.notes or '') for r in g6)
        print(f'G6 absorb WUK/WUV: {"PASS" if ok else "FAIL"}')
    fused = [r for r in rows if r.track == 'assert' and 'fused' in (r.kernel or '')]
    if fused:
        ok = all('PASS' in (r.notes or '') for r in fused)
        print(f'Fused SiLU/GeLU/SoftMax (no standalone compute): '
              f'{"PASS" if ok else "FAIL"} (N={len(fused)})')
    sm_fused = [r for r in rows if r.kernel == 'softmax_fused']
    if sm_fused:
        print('  SoftMax: treated as flash-attn fused — KNOWN_GAP vs isolated '
              'torch.softmax microbench (not in MAPE)')

    measured = [r for r in rows
                if r.meas_s is not None and r.err_max_pct is not None]
    if not measured:
        print(f'N measured: 0 / {len(rows)} (asserts/predict-only only).')
        return

    print(f'N measured: {len(measured)} / {len(rows)}')
    overall = _mape([r.err_max_pct for r in measured])
    if overall is not None:
        print(f'Overall MAPE(pred_max vs meas): {overall:.2f}%')

    for g in ('G1', 'G2', 'G3', 'G4', 'G5'):
        sub = [r for r in measured if r.group == g]
        if not sub:
            continue
        m = _mape([r.err_max_pct for r in sub])
        tracks = sorted({r.track for r in sub})
        print(f'  {g}: N={len(sub):2d}  MAPE={m:6.2f}%  tracks={",".join(tracks)}'
              if m is not None else f'  {g}: n/a')
        if g == 'G2':
            for kern in sorted({r.kernel for r in sub}):
                ksub = [r for r in sub if r.kernel == kern]
                km = _mape([r.err_max_pct for r in ksub])
                if km is not None:
                    print(f'       {kern}: N={len(ksub)}  MAPE={km:.2f}%')
            absorb = [r for r in sub if 'Absorb' in r.name]
            score = [r for r in sub if r not in absorb]
            for label, part in (('absorb', absorb), ('attn_score', score)):
                if not part:
                    continue
                pm = _mape([r.err_max_pct for r in part])
                if pm is not None:
                    gate = 'PASS' if pm < 20 else 'CHECK'
                    print(f'       {label}: N={len(part)}  MAPE={pm:.2f}%  [{gate} <20%]')
        if g == 'G5':
            norms = [r for r in sub if r.kernel == 'rms_norm']
            iso_sm = [r for r in sub if r.kernel == 'softmax_isolated']
            if norms:
                nm = _mape([r.err_max_pct for r in norms])
                if nm is not None:
                    gate = 'PASS' if nm < 30 else 'CHECK'
                    print(f'       rms_norm: N={len(norms)}  MAPE={nm:.2f}%  '
                          f'[{gate} <30%]')
            if iso_sm:
                sm = _mape([r.err_max_pct for r in iso_sm])
                if sm is not None:
                    print(f'       softmax_isolated: N={len(iso_sm)}  '
                          f'MAPE={sm:.2f}%  [KNOWN_GAP]')

    # G4 gap highlight
    g4p = [r for r in measured if r.group == 'G4' and r.track == 'physical']
    if g4p:
        m = _mape([r.err_max_pct for r in g4p])
        print(f'  G4a physical×active vs abstract pred MAPE={m:.2f}%'
              if m is not None else '')
        print('  (gap>25% → MoE model bias / H4, do not retune Phase0 curves)')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=os.path.join(_ROOT, 'models/deepseek-v3-671b.json'))
    p.add_argument('--system', default=os.path.join(_ROOT, 'systems/H20.json'))
    p.add_argument('--matrix-dtype', default='float8')
    p.add_argument('--vector-dtype', default='bfloat16')
    p.add_argument('--seq-size', type=int, default=None)
    p.add_argument('--microbatch-size', type=int, default=1)
    p.add_argument('--expert-par', type=int, default=1)
    p.add_argument('--groups', nargs='+', default=['G1', 'G3', 'G6'],
                   help='Subset of G1..G6 (default: main path)')
    p.add_argument('--blocks', nargs='+', default=['dense', 'moe'],
                   choices=['dense', 'moe'])
    p.add_argument('--stage', default='fw', choices=['fw', 'agrad', 'wgrad'])
    p.add_argument('--predict-only', action='store_true')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--min-ms', type=float, default=200.0)
    p.add_argument('--max-iters', type=int, default=500)
    p.add_argument(
        '--g2-kernel', default='auto',
        help='G2 meas/pred pair: auto|fp8|bf16|both. '
             'auto/bf16: torch.bmm + Calculon matrix.bfloat16 pred (recommended). '
             'fp8: CUDA-graph loop of _scaled_mm + float8 pred (no native FP8 bmm). '
             'both: emit both tracks.',
    )
    p.add_argument('--csv', default='')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    matrix_dtype = normalize_dtype(args.matrix_dtype)
    vector_dtype = normalize_dtype(args.vector_dtype)
    groups = [g.upper() for g in args.groups]
    g2_kernels = resolve_g2_kernels(args.g2_kernel, matrix_dtype)

    llm, app, syst, exe = compile_dsv3(
        args.model, args.system,
        matrix_dtype=matrix_dtype, vector_dtype=vector_dtype,
        seq_size=args.seq_size, microbatch_size=args.microbatch_size,
        expert_par=args.expert_par,
    )
    active_equiv = 1.0
    if app.is_moe:
        active_equiv = app.moe_topk / exe.expert_par + app.num_shared_experts

    print(f'System: {args.system}')
    print(f'Model:  {args.model}  seq={app.seq_size} mbs={exe.microbatch_size}')
    print(f'matrix={matrix_dtype} vector={vector_dtype}  groups={groups}')
    print(f'g2_kernel={args.g2_kernel} → {g2_kernels}')
    print(f'active_equiv={active_equiv:.2f}  measure={not args.predict_only}')

    catalog = build_catalog(
        llm, app, exe, syst, stages=[args.stage], blocks=args.blocks)
    selected = select_rows(catalog, groups, args.stage, args.blocks)
    print(f'Selected {len(selected)} catalog ops')

    measure = not args.predict_only
    if measure:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit('CUDA required unless --predict-only')

    rows: List[BenchRow] = []
    for i, r in enumerate(selected, 1):
        if not args.quiet:
            print(f'[{i}/{len(selected)}] {r.group} {r.name} ...', flush=True)
        rows.extend(measure_row(
            r, matrix_dtype, app.seq_size, app.hidden, app.attn_heads,
            active_equiv, args.warmup, args.iters, args.min_ms, args.max_iters,
            measure=measure, syst=syst, vector_dtype=vector_dtype,
            g2_kernels=g2_kernels,
        ))

    if not args.quiet:
        print()
        print_table(rows)
    summarize(rows)
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == '__main__':
    main()
