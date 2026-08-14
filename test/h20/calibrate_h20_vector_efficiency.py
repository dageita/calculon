#!/usr/bin/env python3
"""Calibrate H20 vector gflops_efficiency for DeepSeek-V3 elementwise ops.

Phase1-style follow-ups (aligned with matrix calib):
  - Dense sampling of small numel / tiny-gflops regimes
  - Enforce non-increasing efficiency as gflops decreases
  - Optional vector_launch_s floor from tiny kernel latencies

Default --dtype=float8 targets Calculon vector.float8. Elementwise is
HBM-bound: peak defaults to "auto" (= max measured TF).

For DeepSeek-V3 mixed precision (GEMM=FP8, norms≈BF16), also run:
  python test/calibrate_h20_vector_efficiency.py --dtype bfloat16 --update-json ...

Kernels by --dtype:
  float8    : FP8 storage -> cast to BF16 -> mul
  float16   : FP16 mul
  bfloat16  : BF16 mul
  float32   : FP32 mul

Example:
  python test/calibrate_h20_vector_efficiency.py --dtype float8 --min-ms 500 \\
      --update-json systems/H20.json
  python test/calibrate_h20_vector_efficiency.py --dtype bfloat16 \\
      --refine-json systems/H20.json --t-launch-us 5
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

# Large power-of-two footprints + DS-V3 reference activations.
LARGE_NUMELS: List[int] = [
    256 * 1024 * 1024,
    128 * 1024 * 1024,
    64 * 1024 * 1024,
    32 * 1024 * 1024,
    16 * 1024 * 1024,
    8 * 1024 * 1024,
    4 * 1024 * 1024,
    2 * 1024 * 1024,
    1 * 1024 * 1024,
    512 * 1024,
    256 * 1024,
    4096 * 7168,            # hidden (S=4096)
    4096 * 1536,            # q_lora
    4096 * 512,             # kv_lora
    128 * 4096 * 4096,      # attn scores TP=1
]

# Dense small ladder (fills launch-bound / tiny GF bins).
_SMALL_POW2 = [
    256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
    12 * 1024, 16 * 1024, 24 * 1024, 32 * 1024, 48 * 1024, 64 * 1024,
    96 * 1024, 128 * 1024, 192 * 1024,
]

# DS-V3-like (m × width) with m-sweep for Phase2 vector ops.
_DS_WIDTHS = [512, 1536, 7168, 18432]
_DS_M = [
    32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512,
    768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
]


def build_numel_list(dense_small: bool = True) -> List[int]:
    seen: Set[int] = set()
    out: List[int] = []

    def add(n: int) -> None:
        n = int(n)
        if n >= 256 and n not in seen:
            seen.add(n)
            out.append(n)

    for n in LARGE_NUMELS:
        add(n)
    if dense_small:
        for n in _SMALL_POW2:
            add(n)
        for w in _DS_WIDTHS:
            for m in _DS_M:
                add(m * w)
        # Softmax-ish: heads × S × S with smaller S (large S kept in LARGE).
        for heads, s in [(8, 512), (16, 512), (32, 512), (64, 512),
                         (8, 1024), (16, 1024), (32, 1024),
                         (8, 2048), (16, 2048), (32, 2048),
                         (64, 2048), (128, 2048)]:
            add(heads * s * s)
    out.sort(reverse=True)
    return out


def _estimate_bytes(numel: int, dtype: str) -> int:
    bpe = DTYPE_NBYTES[dtype]
    out_bpe = DTYPE_NBYTES['bfloat16'] if dtype == 'float8' else bpe
    extra = (2 + 2) if dtype == 'float8' else 0
    return numel * (bpe + bpe + out_bpe + extra)


def _make_inputs(numel: int, dtype: str, device: torch.device):
    storage = torch_storage_dtype(dtype)
    compute = torch_compute_dtype(dtype)
    out = torch.empty(numel, device=device, dtype=compute)
    if dtype == 'float8':
        a = torch.randn(numel, device=device, dtype=torch.bfloat16).to(storage)
        b = torch.randn(numel, device=device, dtype=torch.bfloat16).to(storage)
    else:
        a = torch.randn(numel, device=device, dtype=storage)
        b = torch.randn(numel, device=device, dtype=storage)
    return a, b, out


def vector_kernel(a, b, out, dtype: str):
    if dtype == 'float8':
        torch.mul(a.to(torch.bfloat16), b.to(torch.bfloat16), out=out)
    else:
        torch.mul(a, b, out=out)
    return out


def _probe_latency_ms(a, b, out, dtype: str, probes: int = 5) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(probes):
        vector_kernel(a, b, out, dtype)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / probes


def _choose_iters(
    min_ms: float, lat_ms: float, base_iters: int, max_iters: int,
) -> int:
    iters_used = max(1, base_iters)
    if min_ms > 0:
        iters_used = max(iters_used, int(min_ms / max(lat_ms, 0.01)) + 1)
    return min(iters_used, max(1, max_iters))


def benchmark_numel(
    numel: int, dtype: str, warmup: int, iters: int, min_ms: float,
    max_iters: int = 500,
) -> Tuple[float, float, float, int]:
    device = torch.device('cuda')
    a, b, out = _make_inputs(numel, dtype, device)

    for _ in range(max(1, warmup)):
        vector_kernel(a, b, out, dtype)
    torch.cuda.synchronize()

    lat_ms = _probe_latency_ms(a, b, out, dtype) if min_ms > 0 else 0.0
    iters_used = _choose_iters(min_ms, lat_ms, iters, max_iters)
    # Avoid CUDA Graph here (SIGFPE on some sizes); capped Python loop is enough.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters_used):
        vector_kernel(a, b, out, dtype)
    end.record()
    torch.cuda.synchronize()
    latency_s = (start.elapsed_time(end) / 1000.0) / iters_used
    flops = float(numel)  # Calculon ElementWise: 1 flop / elem
    del a, b, out
    return flops / 1e9, flops / latency_s / 1e12, latency_s, iters_used


def collect_measurements(
    numels: Sequence[int], dtype: str, warmup: int, iters: int, min_ms: float,
    max_iters: int = 4000,
) -> List[Tuple[int, float, float, float, int]]:
    rows = []
    n_total = len(numels)
    for i, n in enumerate(numels, 1):
        est_mib = _estimate_bytes(n, dtype) / (1024 ** 2)
        print(f'  [{i}/{n_total}] running numel={n} (~{est_mib:.0f} MiB) ...',
              flush=True)
        try:
            gflops, tflops, lat, iters_used = benchmark_numel(
                n, dtype, warmup, iters, min_ms, max_iters=max_iters)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'  skip OOM  numel={n} (est ~{est_mib:.0f} MiB)', flush=True)
            continue
        except RuntimeError as e:
            torch.cuda.empty_cache()
            print(f'  skip error numel={n}: {e}', flush=True)
            continue

        print(
            f'  numel={n:12d}  op={gflops:10.6f} GFLOPS  '
            f'achieved={tflops:7.4f} TF  '
            f'lat={lat*1e6:8.1f} us  iters={iters_used}  ~mem={est_mib:.0f} MiB',
            flush=True,
        )
        rows.append((n, gflops, tflops, lat, iters_used))
        if i % 16 == 0:
            torch.cuda.empty_cache()
    return rows


def finalize_curve(
    raw_points: Sequence[Tuple[float, float]], floor_eff: float,
) -> List[List[float]]:
    merged = merge_efficiency_bins(raw_points)
    return enforce_monotonic_efficiency(merged, floor_eff)


def build_efficiency_curve(rows, peak_tflops: float, floor_eff: float):
    print(f'\nUsing peak_tflops={peak_tflops:.6f} TF for efficiency\n', flush=True)
    points = []
    for _n, gflops, tflops, _lat, _it in rows:
        eff = min(1.0, max(floor_eff, tflops / peak_tflops))
        print(
            f'  op={gflops:10.6f} GFLOPS  achieved={tflops:7.4f} TF  eff={eff:.4f}',
            flush=True,
        )
        points.append((gflops, eff))
    return finalize_curve(points, floor_eff)


def resolve_peak(peak_arg: str, measured: Sequence[float]) -> float:
    if peak_arg.lower() == 'auto':
        if not measured:
            raise SystemExit('No successful measurements; cannot auto-set peak')
        peak = max(measured)
        print(f'Auto peak_tflops = {peak:.6f} TF '
              f'(max achieved over {len(measured)} points)', flush=True)
        return peak
    return float(peak_arg)


def update_json(
    path: str, dtype: str, curve, peak_tflops: float, launch_s: Optional[float],
) -> None:
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault('vector', {})[dtype] = {
        'tflops': round(peak_tflops, 6),
        'gflops_efficiency': curve,
    }
    if launch_s is not None and launch_s > 0:
        cfg['vector_launch_s'] = launch_s
    write_system_json(path, cfg)
    msg = f'\nUpdated vector.{dtype} in {path}'
    if launch_s is not None and launch_s > 0:
        msg += f'  (vector_launch_s={launch_s*1e6:.2f} us)'
    print(msg)


def refine_existing_json(
    path: str, dtype: str, floor_eff: float, launch_s: Optional[float],
) -> None:
    with open(path) as f:
        cfg = json.load(f)
    vec = cfg.get('vector', {}).get(dtype)
    if not vec or 'gflops_efficiency' not in vec:
        raise SystemExit(f'No vector.{dtype}.gflops_efficiency in {path}')
    before = list(vec['gflops_efficiency'])
    curve = enforce_monotonic_efficiency(before, floor_eff)
    vec['gflops_efficiency'] = curve
    if launch_s is not None:
        if launch_s > 0:
            cfg['vector_launch_s'] = launch_s
        else:
            cfg.pop('vector_launch_s', None)
    write_system_json(path, cfg)

    changed = sum(
        1 for (g1, e1), (g2, e2) in zip(before, curve)
        if abs(float(e1) - float(e2)) > 1e-12 or abs(float(g1) - float(g2)) > 1e-12
    )
    print(f'Refined vector.{dtype} in {path}: '
          f'{len(before)} → {len(curve)} points, ~{changed} eff adjustments')
    if launch_s is not None and launch_s > 0:
        print(f'Set vector_launch_s={launch_s*1e6:.2f} us')
    prev = 1.1
    bad = 0
    for g, e in curve:
        if g == 0:
            break
        if float(e) > prev + 1e-12:
            bad += 1
        prev = float(e)
    print(f'Monotonic check: {"OK" if bad == 0 else f"FAIL ({bad})"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_dtype_arg(parser, default='float8')
    parser.add_argument(
        '--peak-tflops', type=str, default='auto',
        help='Peak TFLOPS or "auto" (= max measured). Prefer auto for vector.')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--min-ms', type=float, default=500.0)
    parser.add_argument(
        '--max-iters', type=int, default=500,
        help='Cap timed kernel count per numel (default 500)',
    )
    parser.add_argument('--floor-eff', type=float, default=0.001)
    parser.add_argument('--update-json', type=str, default=None)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument(
        '--no-dense-small', action='store_true',
        help='Disable dense small-numel expansion (LARGE_NUMELS only)',
    )
    parser.add_argument(
        '--no-launch-floor', action='store_true',
        help='Do not estimate/write vector_launch_s',
    )
    parser.add_argument(
        '--t-launch-us', type=float, default=None,
        help='Override launch floor in microseconds (skip auto-estimate)',
    )
    parser.add_argument(
        '--refine-json', type=str, default=None,
        help='Only re-monotonicize existing vector.<dtype> curve (no GPU)',
    )
    parser.add_argument(
        '--refine-all-dtypes', action='store_true',
        help='With --refine-json: refine every dtype present under vector',
    )
    args = parser.parse_args()

    if args.refine_json:
        launch = None
        if args.t_launch_us is not None:
            launch = args.t_launch_us * 1e-6
        elif args.no_launch_floor:
            launch = 0.0
        with open(args.refine_json) as f:
            cfg = json.load(f)
        dtypes = list(cfg.get('vector', {}).keys()) if args.refine_all_dtypes else [
            normalize_dtype(args.dtype)
        ]
        for dt in dtypes:
            refine_existing_json(args.refine_json, dt, args.floor_eff, launch)
        return

    dtype = normalize_dtype(args.dtype)

    if not torch.cuda.is_available():
        print('CUDA is required (or use --refine-json)', file=sys.stderr)
        sys.exit(1)

    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    numels = build_numel_list(dense_small=not args.no_dense_small)
    n_small = sum(1 for n in numels if n / 1e9 <= 0.01)  # ≤0.01 GF (=1e7 elems)
    print(f'Device: {torch.cuda.get_device_name(args.device)}')
    print(f'Total mem: {props.total_memory / 1024**3:.1f} GiB')
    dtype_banner(dtype, 'vector')
    print(f'peak: {args.peak_tflops}')
    print(f'Per-numel min GPU time: {args.min_ms} ms')
    print('Post-process: merge bins + enforce monotonic eff; '
          f'dense_small={not args.no_dense_small}')
    print(f'Calibrating vector.{dtype} gflops_efficiency...\n')
    print(f'{len(numels)} numels ({n_small} with op≤0.01 GF), '
          f'min-ms={args.min_ms}, max-iters={args.max_iters}\n', flush=True)

    rows = collect_measurements(
        numels, dtype, args.warmup, args.iters, args.min_ms,
        max_iters=args.max_iters)
    if not rows:
        raise SystemExit('No measurements collected')

    peak = resolve_peak(args.peak_tflops, [r[2] for r in rows])
    curve = build_efficiency_curve(rows, peak, args.floor_eff)

    launch_s: Optional[float] = None
    if args.t_launch_us is not None:
        launch_s = args.t_launch_us * 1e-6
    elif not args.no_launch_floor:
        gflops_list = [r[1] for r in rows]
        lat_list = [r[3] for r in rows]
        # Vector ops are tiny in GF; use ≤1e-3 GF (~1e6 elems) for launch.
        launch_s = estimate_launch_s(
            lat_list, gflops_list, max_gflops=1e-3, min_samples=3)
        if launch_s is not None:
            print(f'\nEstimated vector_launch_s = {launch_s*1e6:.2f} us '
                  f'(median of tiny ≤0.001 GF latencies)')

    top_eff = curve[0][1] if curve else 0.0
    if top_eff <= args.floor_eff * 1.01:
        print(
            f'\nWARNING: top efficiency {top_eff} ≈ floor. '
            f'Peak {peak} TF is still too high for this kernel.',
            file=sys.stderr,
        )

    print(f'\n# Paste into systems/H20.json -> vector.{dtype}')
    fragment = {'tflops': round(peak, 6), 'gflops_efficiency': curve}
    print(json.dumps({dtype: fragment}, indent=2))
    if launch_s:
        print(f'# vector_launch_s: {launch_s}  # {launch_s*1e6:.2f} us')

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
        update_json(
            args.update_json, dtype, curve, peak,
            None if args.no_launch_floor and args.t_launch_us is None
            else launch_s,
        )


if __name__ == '__main__':
    main()
