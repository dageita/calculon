#!/usr/bin/env python3
"""Calibrate H20 mem1 MB_efficiency (HBM STREAM-like).

Calculon mem1 is dtype-agnostic: efficiency is keyed only by transfer size
(MB -> bytes). This script measures device-to-device copy bandwidth vs size
and writes mem1.{GiB, GBps, MB_efficiency}.

Peak HBM: 4022 GB/s, capacity 96 GiB.

Example:
  python test/calibrate_h20_mem_efficiency.py --update-json systems/H20.json
  python test/calibrate_h20_mem_efficiency.py --min-ms 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Sequence, Tuple

import torch

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from calibrate_h20_common import (  # noqa: E402
    H20_MEM_CAPACITY_GIB,
    H20_MEM_PEAK_GBPS,
    write_system_json,
)

# MB keys match calculon Memory.efficiency (bytes = MB * 1e6).
DEFAULT_SIZES_MB: List[float] = [
    2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.25, 0.125,
]


def _probe_latency_ms(src: torch.Tensor, dst: torch.Tensor, probes: int = 5) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(probes):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / probes


def benchmark_copy(
    nbytes: int, warmup: int, iters: int, min_ms: float,
) -> Tuple[float, float, int]:
    """Byte D2D copy; STREAM counts 2*nbytes."""
    device = torch.device('cuda')
    nelem = max(4096, nbytes)
    src = torch.empty(nelem, dtype=torch.uint8, device=device)
    dst = torch.empty(nelem, dtype=torch.uint8, device=device)
    nbytes_used = nelem

    for _ in range(max(1, warmup)):
        dst.copy_(src)
    torch.cuda.synchronize()

    iters_used = max(1, iters)
    if min_ms > 0:
        lat_ms = _probe_latency_ms(src, dst)
        iters_used = max(iters_used, int(min_ms / max(lat_ms, 1e-3)) + 1)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters_used):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()

    latency_s = (start.elapsed_time(end) / 1000.0) / iters_used
    achieved_gbps = (2.0 * nbytes_used) / latency_s / 1e9

    del src, dst
    return achieved_gbps, latency_s, iters_used


def build_efficiency_curve(
    sizes_mb: Sequence[float],
    peak_gbps: float,
    warmup: int,
    iters: int,
    min_ms: float,
    floor_eff: float,
) -> List[List[float]]:
    points = []
    for mb in sorted(sizes_mb, reverse=True):
        nbytes = max(4096, int(mb * 1e6))
        est_mib = 2.0 * nbytes / (1024 ** 2)
        try:
            gbps, lat, iters_used = benchmark_copy(
                nbytes, warmup, iters, min_ms)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'  skip OOM  {mb} MB (est ~{est_mib:.0f} MiB)', flush=True)
            continue
        except RuntimeError as e:
            torch.cuda.empty_cache()
            print(f'  skip error {mb} MB: {e}', flush=True)
            continue

        eff = min(1.0, max(floor_eff, gbps / peak_gbps))
        print(
            f'  size={mb:8.3f} MB  '
            f'achieved={gbps:8.1f} GB/s  eff={eff:.4f}  '
            f'lat={lat*1e6:.1f} us  iters={iters_used}  ~mem={est_mib:.0f} MiB',
            flush=True,
        )
        points.append((mb, eff, gbps))
        torch.cuda.empty_cache()

    bins = {}
    for mb, eff, _ in points:
        key = float(f'{mb:.6g}')
        bins[key] = max(bins.get(key, 0.0), eff)

    curve = [[m, bins[m]] for m in sorted(bins.keys(), reverse=True)]
    if not curve or curve[-1][0] != 0:
        curve.append([0, floor_eff])
    return curve


def update_json(path: str, curve: List[List[float]], peak_gbps: float,
                capacity_gib: float) -> None:
    with open(path) as f:
        cfg = json.load(f)
    mem1 = cfg.setdefault('mem1', {})
    mem1['GiB'] = capacity_gib
    mem1['GBps'] = peak_gbps
    mem1['MB_efficiency'] = curve
    write_system_json(path, cfg)
    print(f'\nUpdated mem1 in {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--peak-gbps', type=float, default=H20_MEM_PEAK_GBPS)
    parser.add_argument('--capacity-gib', type=float, default=H20_MEM_CAPACITY_GIB)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--min-ms', type=float, default=500.0)
    parser.add_argument('--floor-eff', type=float, default=0.02)
    parser.add_argument('--update-json', type=str, default=None)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('CUDA is required', file=sys.stderr)
        sys.exit(1)

    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    print(f'Device: {torch.cuda.get_device_name(args.device)}')
    print(f'Total mem: {props.total_memory / 1024**3:.1f} GiB')
    print(f'Peak HBM: {args.peak_gbps} GB/s, capacity {args.capacity_gib} GiB')
    print(f'Per-size min GPU time: {args.min_ms} ms')
    print('Calibrating mem1.MB_efficiency (uint8 D2D copy, STREAM 2x bytes)...\n')
    print(f'{len(DEFAULT_SIZES_MB)} sizes, ETA >= '
          f'{len(DEFAULT_SIZES_MB) * args.min_ms / 1000:.1f}s '
          f'at --min-ms={args.min_ms}\n', flush=True)

    curve = build_efficiency_curve(
        DEFAULT_SIZES_MB, args.peak_gbps, args.warmup, args.iters,
        args.min_ms, args.floor_eff)

    print('\n# Paste into systems/H20.json -> mem1')
    fragment = {
        'GiB': args.capacity_gib,
        'GBps': args.peak_gbps,
        'MB_efficiency': curve,
    }
    print(json.dumps(fragment, indent=2))

    if args.update_json:
        update_json(args.update_json, curve, args.peak_gbps, args.capacity_gib)


if __name__ == '__main__':
    main()
