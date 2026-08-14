#!/usr/bin/env python3
"""Calibrate H20 matrix gflops_efficiency for DeepSeek-V3 shapes.

Phase1 follow-ups baked in:
  - Dense sampling of the small-gflops regime (<~80 GF) that drove high MAPE
  - Enforce non-increasing efficiency as gflops decreases (no sawtooth)
  - Optional matrix_launch_s floor estimated from tiny GEMM latencies

Default --dtype=float8 matches DeepSeek-V3 FP8 Tensor-Core GEMM on H20.

Note: H20 native bfloat16 cuBLAS GEMM SIGFPEs on some shapes (e.g. 768×512×K).
bfloat16 calibration times the FP16 Tensor-Core path (same 148 TF peak) and
writes the curve under matrix.bfloat16.

Example:
  python test/calibrate_h20_matrix_efficiency.py --dtype float8 --min-ms 500 \\
      --update-json systems/H20.json
  # Re-monotonicize + write launch floor from an existing JSON (no GPU):
  python test/calibrate_h20_matrix_efficiency.py --dtype float8 \\
      --refine-json systems/H20.json --t-launch-us 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence, Set, Tuple

import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from calibrate_h20_common import (  # noqa: E402
    DTYPE_NBYTES,
    H20_MATRIX_PEAK_TFLOPS,
    add_dtype_arg,
    dtype_banner,
    enforce_monotonic_efficiency,
    estimate_launch_s,
    merge_efficiency_bins,
    normalize_dtype,
    torch_compute_dtype,
    torch_storage_dtype,
    write_system_json,
)

# Large / DS-V3 reference shapes (m, n, k) with A[m,k] @ B[k,n].
LARGE_SHAPES: List[Tuple[int, int, int]] = [
    (8192, 8192, 8192),
    (4096, 4096, 4096),
    (2048, 2048, 2048),
    (1024, 1024, 1024),
    (512, 512, 512),
    (4096, 7168, 1536),    # WDQ
    (4096, 1536, 16384),   # WUQ TP=1
    (4096, 1536, 8192),    # WQR TP=1
    (4096, 7168, 512),     # WDKV
    (4096, 7168, 64),      # WKR
    (4096, 16384, 7168),   # WO TP=1
    (4096, 7168, 256),     # Router
    (4096, 7168, 2048),    # MoE expert w1
    (4096, 2048, 7168),    # MoE expert w2
    (4096, 7168, 18432),   # Dense SwiGLU w1
    (4096, 18432, 7168),   # Dense SwiGLU w2
    (8192, 7168, 18432),
    (2048, 7168, 18432),
    (1024, 7168, 18432),
]

# Dense small-square ladder (fills tiny/mid GF bins).
_SMALL_SQUARE = [
    32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256,
    320, 384, 448, 512, 640, 768, 896,
]

# Skinny / mid (n, k) pairs × m-sweep → Phase1 problem region.
_SKINNY_NK: List[Tuple[int, int]] = [
    (16, 7168),     # skinny_ext
    (64, 7168),     # WKR
    (256, 7168),    # Router
    (512, 7168),    # WDKV-ish
    (1536, 7168),   # WDQ-ish smaller m
    (2048, 7168),   # expert w1
    (2048, 2048),
    (1024, 1024),
]

_SMALL_M = [
    32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512,
    640, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096,
    6144, 8192, 12288, 16384,
]

# Keep densified shapes with op gflops at or below this (plus all LARGE_SHAPES).
_SMALL_GFLOPS_CAP = 80.0


def _align(x: int, align: int = 16) -> int:
    return max(align, (x // align) * align)


def build_shape_list(dense_small: bool = True) -> List[Tuple[int, int, int]]:
    seen: Set[Tuple[int, int, int]] = set()
    out: List[Tuple[int, int, int]] = []

    def add(m: int, n: int, k: int) -> None:
        t = (_align(m), _align(n), _align(k))
        if t not in seen:
            seen.add(t)
            out.append(t)

    for m, n, k in LARGE_SHAPES:
        add(m, n, k)

    if dense_small:
        for s in _SMALL_SQUARE:
            add(s, s, s)
        for n, k in _SKINNY_NK:
            for m in _SMALL_M:
                gflops = 2.0 * m * n * k / 1e9
                if gflops <= _SMALL_GFLOPS_CAP:
                    add(m, n, k)

    out.sort(key=lambda s: 2 * s[0] * s[1] * s[2], reverse=True)
    return out


def _shape_squareness(m: int, n: int, k: int) -> float:
    dims = sorted((m, n, k))
    return dims[0] / max(dims[2], 1)


def select_shapes_for_measure(
    shapes: Sequence[Tuple[int, int, int]],
    gflops_tol: float = 0.05,
) -> List[Tuple[int, int, int]]:
    """One representative per gflops bin (prefer squarer = usually higher eff).

    Dense skinny sweeps create many (m,n,k) with nearly identical 2mnk; measuring
    all of them with min_ms budgeting makes the job look hung.
    """
    if gflops_tol <= 0:
        return list(shapes)
    selected: List[Tuple[int, int, int]] = []
    selected_g: List[float] = []
    # Consider larger ops first; within a bin keep the squarer candidate.
    ordered = sorted(
        shapes,
        key=lambda s: (-2 * s[0] * s[1] * s[2], -_shape_squareness(*s)),
    )
    for m, n, k in ordered:
        g = 2.0 * m * n * k / 1e9
        dup = any(
            abs(g - sg) <= gflops_tol * max(g, sg, 1e-12) for sg in selected_g
        )
        if dup:
            continue
        selected.append((m, n, k))
        selected_g.append(g)
    selected.sort(key=lambda s: 2 * s[0] * s[1] * s[2], reverse=True)
    return selected


def _estimate_bytes(m: int, n: int, k: int, dtype: str) -> int:
    bpe = DTYPE_NBYTES[dtype]
    out_bpe = DTYPE_NBYTES['bfloat16'] if dtype == 'float8' else bpe
    return m * k * bpe + n * k * bpe + m * n * out_bpe


def _gemm_fp8(a, b_nk, scale_a, scale_b, out_pair=None):
    kwargs = dict(scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    if out_pair is not None:
        kwargs['out'] = out_pair
    result = torch._scaled_mm(a, b_nk.t(), **kwargs)
    return result[0] if isinstance(result, tuple) else result


def _gemm_dense(a, b_nk, out=None):
    return torch.matmul(a, b_nk.t(), out=out)


def _setup_gemm(m: int, n: int, k: int, dtype: str, device: torch.device):
    storage = torch_storage_dtype(dtype)
    compute = torch_compute_dtype(dtype)

    if dtype == 'float8':
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16).to(storage)
        b = torch.randn(n, k, device=device, dtype=torch.bfloat16).to(storage)
        scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
        out_c = torch.empty(m, n, device=device, dtype=compute)
        out_aux = torch.empty((), device=device, dtype=torch.float32)
        out_pair = (out_c, out_aux)

        def run():
            return _gemm_fp8(a, b, scale_a, scale_b, out_pair=out_pair)

        return run, [a, b, scale_a, scale_b, out_c, out_aux]

    # H20 / cuBLAS: native bfloat16 GEMM hits SIGFPE on some shapes
    # (repro: m=768,n=512,k=7168). FP16 Tensor Core has the same nominal peak
    # (148 TF) on H20 — measure via fp16 path and store under matrix.bfloat16.
    if dtype == 'bfloat16':
        a = torch.randn(m, k, device=device, dtype=torch.float16)
        b = torch.randn(n, k, device=device, dtype=torch.float16)
        out = torch.empty(m, n, device=device, dtype=torch.float16)

        def run():
            return _gemm_dense(a, b, out=out)

        return run, [a, b, out]

    a = torch.randn(m, k, device=device, dtype=storage)
    b = torch.randn(n, k, device=device, dtype=storage)
    out = torch.empty(m, n, device=device, dtype=compute)

    def run():
        return _gemm_dense(a, b, out=out)

    return run, [a, b, out]


def _probe_latency_ms(run, probes: int = 5) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(probes):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / probes


def _choose_iters(
    min_ms: float, lat_ms: float, base_iters: int, max_iters: int,
) -> int:
    """Scale GEMM count to cover min_ms, capped by max_iters.

    Critical: each timed iter is a Python→CUDA launch. Tiny GEMMs (~30us GPU)
    with min_ms=500 uncapped request ~15k iters → tens of seconds of host
    overhead per shape (looks hung). Cap aggressively; CUDA-event timing stays
    valid even when wall clock is host-bound.
    """
    iters_used = max(1, base_iters)
    if min_ms > 0:
        iters_used = max(iters_used, int(min_ms / max(lat_ms, 0.01)) + 1)
    return min(iters_used, max(1, max_iters))


def benchmark_shape(
    m: int, n: int, k: int, dtype: str, warmup: int, iters: int, min_ms: float,
    max_iters: int = 500,
) -> Tuple[float, float, float, int]:
    m, n, k = _align(m), _align(n), _align(k)
    device = torch.device('cuda')
    run, refs = _setup_gemm(m, n, k, dtype, device)

    for _ in range(max(1, warmup)):
        run()
    torch.cuda.synchronize()

    lat_ms = _probe_latency_ms(run) if min_ms > 0 else 0.0
    iters_used = _choose_iters(min_ms, lat_ms, iters, max_iters)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters_used):
        run()
    end.record()
    torch.cuda.synchronize()

    latency_s = (start.elapsed_time(end) / 1000.0) / iters_used
    flops = 2.0 * m * n * k
    del refs
    return flops / 1e9, flops / latency_s / 1e12, latency_s, iters_used


def finalize_curve(
    raw_points: Sequence[Tuple[float, float]],
    floor_eff: float,
) -> List[List[float]]:
    """Merge nearby bins → monotonic descending gflops curve ending at [0, floor]."""
    merged = merge_efficiency_bins(raw_points)
    return enforce_monotonic_efficiency(merged, floor_eff)


def build_efficiency_curve(
    shapes: Sequence[Tuple[int, int, int]],
    dtype: str,
    peak_tflops: float,
    warmup: int,
    iters: int,
    min_ms: float,
    floor_eff: float,
    max_iters: int = 500,
) -> Tuple[List[List[float]], List[float], List[float]]:
    """Returns (curve, gflops_list, latency_list) for launch estimation."""
    points = []
    gflops_list: List[float] = []
    lat_list: List[float] = []
    n_shapes = len(shapes)

    for i, (m, n, k) in enumerate(shapes, 1):
        m, n, k = _align(m), _align(n), _align(k)
        est_mib = _estimate_bytes(m, n, k, dtype) / (1024 ** 2)
        print(
            f'  [{i}/{n_shapes}] running m={m} n={n} k={k} '
            f'(~{est_mib:.0f} MiB) ...',
            flush=True,
        )
        try:
            gflops, tflops, lat, iters_used = benchmark_shape(
                m, n, k, dtype, warmup, iters, min_ms, max_iters=max_iters)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'  skip OOM  m={m} n={n} k={k} (est ~{est_mib:.0f} MiB)', flush=True)
            continue
        except RuntimeError as e:
            print(f'  skip error m={m} n={n} k={k}: {e}', flush=True)
            torch.cuda.empty_cache()
            continue

        eff = min(1.0, max(floor_eff, tflops / peak_tflops))
        capped = ' (iters capped)' if iters_used >= max_iters else ''
        print(
            f'  m={m:5d} n={n:5d} k={k:5d}  '
            f'op={gflops:10.3f} GFLOPS  '
            f'achieved={tflops:7.2f} TF  '
            f'eff={eff:.4f}  lat={lat*1e6:8.1f} us  '
            f'iters={iters_used}  ~mem={est_mib:.0f} MiB{capped}',
            flush=True,
        )
        points.append((gflops, eff))
        gflops_list.append(gflops)
        lat_list.append(lat)
        if i % 16 == 0:
            torch.cuda.empty_cache()

    curve = finalize_curve(points, floor_eff)
    return curve, gflops_list, lat_list


def update_json(
    path: str,
    dtype: str,
    curve: List[List[float]],
    peak_tflops: float,
    launch_s: Optional[float],
) -> None:
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault('matrix', {})[dtype] = {
        'tflops': peak_tflops,
        'gflops_efficiency': curve,
    }
    if launch_s is not None and launch_s > 0:
        cfg['matrix_launch_s'] = launch_s
    write_system_json(path, cfg)
    msg = f'\nUpdated matrix.{dtype} in {path}'
    if launch_s is not None and launch_s > 0:
        msg += f'  (matrix_launch_s={launch_s*1e6:.2f} us)'
    print(msg)


def refine_existing_json(
    path: str, dtype: str, floor_eff: float, launch_s: Optional[float],
) -> None:
    with open(path) as f:
        cfg = json.load(f)
    mat = cfg.get('matrix', {}).get(dtype)
    if not mat or 'gflops_efficiency' not in mat:
        raise SystemExit(f'No matrix.{dtype}.gflops_efficiency in {path}')
    raw = mat['gflops_efficiency']
    before = list(raw)
    curve = enforce_monotonic_efficiency(raw, floor_eff)
    mat['gflops_efficiency'] = curve
    if launch_s is not None:
        if launch_s > 0:
            cfg['matrix_launch_s'] = launch_s
        else:
            cfg.pop('matrix_launch_s', None)
    write_system_json(path, cfg)

    changed = sum(
        1 for (g1, e1), (g2, e2) in zip(before, curve)
        if abs(float(e1) - float(e2)) > 1e-12 or abs(float(g1) - float(g2)) > 1e-12
    )
    print(f'Refined matrix.{dtype} in {path}: '
          f'{len(before)} → {len(curve)} points, ~{changed} eff adjustments')
    if launch_s is not None and launch_s > 0:
        print(f'Set matrix_launch_s={launch_s*1e6:.2f} us')
    # Show any remaining / applied monotonic clamps
    prev = None
    clamps = 0
    for g, e in curve:
        if prev is not None and float(e) > prev + 1e-15:
            clamps += 1
        prev = float(e) if float(g) > 0 else prev
    print(f'Monotonic check (large→small non-increasing): '
          f'{"OK" if clamps == 0 else f"FAIL ({clamps})"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dtype_arg(parser, default='float8')
    parser.add_argument('--peak-tflops', type=float, default=None,
                        help='Override peak TFLOPS (default: H20 sheet for --dtype)')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--min-ms', type=float, default=500.0)
    parser.add_argument(
        '--max-iters', type=int, default=500,
        help='Cap timed GEMMs per shape (default 500). Tiny kernels otherwise '
             'request 10k+ Python launches and appear hung.',
    )
    parser.add_argument(
        '--dedupe-gflops-tol', type=float, default=0.05,
        help='Pre-select one shape per gflops bin (rel tol; prefer squarer). '
             '0 = measure all shapes. Default 0.05.',
    )
    parser.add_argument('--floor-eff', type=float, default=0.01)
    parser.add_argument('--update-json', type=str, default=None)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument(
        '--no-dense-small', action='store_true',
        help='Disable dense <80GF shape expansion (legacy LARGE_SHAPES only)',
    )
    parser.add_argument(
        '--no-launch-floor', action='store_true',
        help='Do not estimate/write matrix_launch_s',
    )
    parser.add_argument(
        '--t-launch-us', type=float, default=None,
        help='Override launch floor in microseconds (skip auto-estimate)',
    )
    parser.add_argument(
        '--refine-json', type=str, default=None,
        help='Only re-monotonicize existing matrix.<dtype> curve (no GPU). '
             'Use with --t-launch-us to set launch floor.',
    )
    args = parser.parse_args()

    dtype = normalize_dtype(args.dtype)
    peak = args.peak_tflops if args.peak_tflops is not None else H20_MATRIX_PEAK_TFLOPS[dtype]

    if args.refine_json:
        launch = None
        if args.t_launch_us is not None:
            launch = args.t_launch_us * 1e-6
        elif args.no_launch_floor:
            launch = 0.0
        refine_existing_json(args.refine_json, dtype, args.floor_eff, launch)
        return

    if not torch.cuda.is_available():
        print('CUDA is required (or use --refine-json)', file=sys.stderr)
        sys.exit(1)

    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    print(f'Device: {torch.cuda.get_device_name(args.device)}')
    print(f'Total mem: {props.total_memory / 1024**3:.1f} GiB')
    dtype_banner(dtype, 'matrix')
    print(f'Peak Tensor Core: {peak} TFLOPS')
    if dtype == 'bfloat16':
        print('NOTE: H20 native bf16 GEMM can SIGFPE; measuring via FP16 TC path '
              '(same peak), writing matrix.bfloat16')
    print(f'Per-shape min GPU time: {args.min_ms} ms  (max-iters={args.max_iters})')
    print('Post-process: merge bins + enforce monotonic eff; '
          f'dense_small={not args.no_dense_small}; '
          f'dedupe_gflops_tol={args.dedupe_gflops_tol}')
    print(f'Calibrating matrix.{dtype} gflops_efficiency...\n')

    shapes_all = build_shape_list(dense_small=not args.no_dense_small)
    shapes = select_shapes_for_measure(shapes_all, args.dedupe_gflops_tol)
    n_small = sum(1 for m, n, k in shapes if 2 * m * n * k / 1e9 <= _SMALL_GFLOPS_CAP)
    print(f'{len(shapes)} shapes to measure '
          f'(from {len(shapes_all)} candidates; '
          f'{n_small} with op≤{_SMALL_GFLOPS_CAP:g} GF), '
          f'min-ms={args.min_ms}, max-iters={args.max_iters}\n',
          flush=True)

    curve, gflops_list, lat_list = build_efficiency_curve(
        shapes, dtype, peak, args.warmup, args.iters, args.min_ms, args.floor_eff,
        max_iters=args.max_iters)

    launch_s: Optional[float] = None
    if args.t_launch_us is not None:
        launch_s = args.t_launch_us * 1e-6
    elif not args.no_launch_floor:
        launch_s = estimate_launch_s(lat_list, gflops_list, max_gflops=1.0)
        if launch_s is not None:
            print(f'\nEstimated matrix_launch_s = {launch_s*1e6:.2f} us '
                  f'(median of tiny ≤1 GF latencies)')

    print(f'\n# Paste into systems/H20.json -> matrix.{dtype}')
    fragment = {'tflops': peak, 'gflops_efficiency': curve}
    print(json.dumps({dtype: fragment}, indent=2))
    if launch_s:
        print(f'# matrix_launch_s: {launch_s}  # {launch_s*1e6:.2f} us')

    # Sanity: no ascending eff when walking large→small
    prev = 1.1
    bad = 0
    for g, e in curve:
        if g == 0:
            break
        if e > prev + 1e-12:
            bad += 1
        prev = e
    print(f'\nMonotonic check: {"OK" if bad == 0 else f"FAIL ({bad} rises)"}')

    if args.update_json:
        update_json(args.update_json, dtype, curve, peak,
                    None if args.no_launch_floor and args.t_launch_us is None
                    else launch_s)


if __name__ == '__main__':
    main()
