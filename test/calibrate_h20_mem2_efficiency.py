#!/usr/bin/env python3
"""Calibrate H20 mem2 MB_efficiency (Host <-> GPU / PCIe offload path).

Calculon mem2 is the secondary memory tier used for weight / activation /
optimizer offload. Timing is:

    offload_time = size / mem2.throughput(size)
    throughput   = GBps * MB_efficiency(size)

This script measures pinned-host <-> CUDA transfers (H2D / D2H) vs size and
writes mem2.{GiB, GBps, MB_efficiency}. Bytes are dtype-agnostic (uint8).

Defaults (H20-class PCIe): peak 64 GB/s unidirectional, capacity 512 GiB host.

Example:
  python test/calibrate_h20_mem2_efficiency.py --update-json systems/H20.json
  python test/calibrate_h20_mem2_efficiency.py --direction both --min-ms 1000
  python test/calibrate_h20_mem2_efficiency.py --capacity-gib 256 --peak-gbps 64
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
    H20_MEM2_CAPACITY_GIB,
    H20_MEM2_PEAK_GBPS,
    write_system_json,
)

# MB keys match calculon Memory.efficiency (bytes = MB * 1e6).
# Cap at 512 MB by default: large enough for PCIe asymptote, safer for pinned RAM.
DEFAULT_SIZES_MB: List[float] = [
    512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625,
]


def _probe_latency_ms(
    src: torch.Tensor, dst: torch.Tensor, probes: int = 5,
) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(probes):
        dst.copy_(src, non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / probes


def benchmark_xfer(
    nbytes: int,
    direction: str,
    warmup: int,
    iters: int,
    min_ms: float,
) -> Tuple[float, float, int]:
    """Pinned host <-> device copy. Counts 1*nbytes (unidirectional offload)."""
    device = torch.device('cuda')
    nelem = max(4096, nbytes)
    host = torch.empty(nelem, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(nelem, dtype=torch.uint8, device=device)
    nbytes_used = nelem

    if direction == 'h2d':
        src, dst = host, dev
    elif direction == 'd2h':
        src, dst = dev, host
    else:
        raise ValueError(f'direction must be h2d or d2h, got {direction!r}')

    for _ in range(max(1, warmup)):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()

    iters_used = max(1, iters)
    if min_ms > 0:
        lat_ms = _probe_latency_ms(src, dst)
        iters_used = max(iters_used, int(min_ms / max(lat_ms, 1e-3)) + 1)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters_used):
        dst.copy_(src, non_blocking=True)
    end.record()
    torch.cuda.synchronize()

    latency_s = (start.elapsed_time(end) / 1000.0) / iters_used
    # Unidirectional: one transfer moves nbytes_used (not STREAM 2x).
    achieved_gbps = nbytes_used / latency_s / 1e9

    del host, dev
    return achieved_gbps, latency_s, iters_used


def build_efficiency_curve(
    sizes_mb: Sequence[float],
    peak_gbps: float,
    direction: str,
    warmup: int,
    iters: int,
    min_ms: float,
    floor_eff: float,
) -> List[List[float]]:
    """direction: h2d | d2h | both (use min of H2D/D2H per size)."""
    points = []
    dirs = ['h2d', 'd2h'] if direction == 'both' else [direction]

    for mb in sorted(sizes_mb, reverse=True):
        nbytes = max(4096, int(mb * 1e6))
        est_mib = nbytes / (1024 ** 2)
        per_dir = {}
        try:
            for d in dirs:
                gbps, lat, iters_used = benchmark_xfer(
                    nbytes, d, warmup, iters, min_ms)
                per_dir[d] = (gbps, lat, iters_used)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'  skip OOM  {mb} MB (est ~{est_mib:.0f} MiB)', flush=True)
            continue
        except RuntimeError as e:
            torch.cuda.empty_cache()
            print(f'  skip error {mb} MB: {e}', flush=True)
            continue

        # Conservative: bottleneck direction for bidirectional offload stacks.
        gbps = min(v[0] for v in per_dir.values())
        lat = max(v[1] for v in per_dir.values())
        iters_used = max(v[2] for v in per_dir.values())
        eff = min(1.0, max(floor_eff, gbps / peak_gbps))

        detail = '  '.join(
            f'{d}={per_dir[d][0]:.1f}' for d in dirs
        )
        print(
            f'  size={mb:8.3f} MB  achieved={gbps:7.2f} GB/s  eff={eff:.4f}  '
            f'({detail})  lat={lat*1e6:.1f} us  iters={iters_used}  '
            f'~xfer={est_mib:.1f} MiB',
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
    mem2 = cfg.setdefault('mem2', {})
    mem2['GiB'] = capacity_gib
    mem2['GBps'] = peak_gbps
    mem2['MB_efficiency'] = curve
    write_system_json(path, cfg)
    print(f'\nUpdated mem2 in {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--peak-gbps', type=float, default=H20_MEM2_PEAK_GBPS,
                        help='Unidirectional PCIe/host peak GB/s (default 64)')
    parser.add_argument('--capacity-gib', type=float,
                        default=H20_MEM2_CAPACITY_GIB,
                        help='Host DRAM capacity for offload (default 512 GiB)')
    parser.add_argument(
        '--direction', choices=('both', 'h2d', 'd2h'), default='both',
        help='Transfer direction; both uses min(H2D,D2H) per size (default)',
    )
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--min-ms', type=float, default=500.0)
    parser.add_argument('--floor-eff', type=float, default=0.05)
    parser.add_argument('--update-json', type=str, default=None)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('CUDA is required', file=sys.stderr)
        sys.exit(1)

    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    print(f'Device: {torch.cuda.get_device_name(args.device)}')
    print(f'GPU mem: {props.total_memory / 1024**3:.1f} GiB')
    print(f'Peak mem2 (PCIe/host): {args.peak_gbps} GB/s, '
          f'capacity {args.capacity_gib} GiB')
    print(f'Direction: {args.direction}  |  per-size min time: {args.min_ms} ms')
    print('Calibrating mem2.MB_efficiency '
          '(pinned uint8 H2D/D2H, unidirectional 1x bytes)...\n')
    n_passes = 2 if args.direction == 'both' else 1
    print(f'{len(DEFAULT_SIZES_MB)} sizes x {n_passes} dir(s), ETA >= '
          f'{len(DEFAULT_SIZES_MB) * n_passes * args.min_ms / 1000:.1f}s '
          f'at --min-ms={args.min_ms}\n', flush=True)

    curve = build_efficiency_curve(
        DEFAULT_SIZES_MB, args.peak_gbps, args.direction,
        args.warmup, args.iters, args.min_ms, args.floor_eff)

    print('\n# Paste into systems/H20.json -> mem2')
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
