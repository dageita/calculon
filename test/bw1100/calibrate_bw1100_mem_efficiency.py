#!/usr/bin/env python3
"""Calibrate BW1100 mem1: single-card HBM3 device-copy efficiency.

BW1100 physical parameters are 144 GiB HBM3 and 2400 GB/s peak bandwidth.
Calculon counts a device copy as one read plus one write (2 x bytes).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from calibrate_bw1100_common import SYSTEM_JSON, require_bw1100, update_memory_curve

# The product specification is 144 decimal GB.  Calculon stores GiB.
HBM_CAPACITY_GIB = 144.0 * 1e9 / 2**30
HBM_PEAK_GBPS = 2400.0
DEFAULT_SIZES_MB = (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, .5, .25, .125)


def choose_iters(latency_ms: float, requested: int, min_ms: float, max_iters: int) -> int:
    return min(max_iters, max(1, requested, int(min_ms / max(latency_ms, .001)) + 1))


def benchmark_copy(nbytes: int, warmup: int, iterations: int, min_ms: float,
                   max_iters: int, samples: int) -> tuple[float, float, int]:
    """STREAM-like D2D copy; median independent timing rejects HBM noise."""
    src = torch.empty(max(4096, nbytes), dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(max(1, warmup)): dst.copy_(src)
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); dst.copy_(src); stop.record(); stop.synchronize()
    actual_iters = choose_iters(start.elapsed_time(stop), iterations, min_ms, max_iters)
    timings = []
    for _ in range(samples):
        start.record()
        for _ in range(actual_iters): dst.copy_(src)
        stop.record(); stop.synchronize()
        timings.append(start.elapsed_time(stop) / 1000.0 / actual_iters)
    latency_s = statistics.median(timings)
    return 2.0 * src.numel() / latency_s / 1e9, latency_s, actual_iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mb", type=float, nargs="+", default=DEFAULT_SIZES_MB)
    parser.add_argument("--peak-gbps", type=float, default=HBM_PEAK_GBPS)
    parser.add_argument("--capacity-gib", type=float, default=HBM_CAPACITY_GIB)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=3,
                        help="independent timings per transfer size; use median (default: 3)")
    parser.add_argument("--min-ms", type=float, default=500.0)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--floor-eff", type=float, default=.02)
    parser.add_argument("--update-json", type=Path, nargs="?", const=SYSTEM_JSON)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.peak_gbps <= 0 or args.capacity_gib <= 0 or args.samples < 1: raise ValueError("peak, capacity, and samples must be positive")
    isa = require_bw1100()
    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name} ({isa}), visible HBM: {props.total_memory / 2**30:.1f} GiB", flush=True)
    print(f"Calibrating HBM3: peak={args.peak_gbps:.1f} GB/s, capacity={args.capacity_gib:.1f} GiB", flush=True)
    points: list[list[float]] = []
    for index, mb in enumerate(sorted(set(args.sizes_mb), reverse=True), 1):
        nbytes = int(mb * 1e6)
        print(f"[{index:2d}/{len(set(args.sizes_mb))}] START {mb:g} MB", flush=True)
        try:
            gbps, latency_s, iters = benchmark_copy(nbytes, args.warmup, args.iterations, args.min_ms, args.max_iters, args.samples)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  skip OOM {mb:g} MB", flush=True); continue
        efficiency = min(1.0, max(args.floor_eff, gbps / args.peak_gbps))
        points.append([float(mb), round(efficiency, 6)])
        print(f"[{index:2d}] DONE {mb:g} MB  {gbps:.2f} GB/s  eff={efficiency:.4f}  {latency_s*1e6:.1f} us  iters={iters}", flush=True)
        torch.cuda.empty_cache()
    curve = points + [[0.0, args.floor_eff]]
    fragment = {"GiB": args.capacity_gib, "GBps": args.peak_gbps, "MB_efficiency": curve}
    output = args.output or Path(__file__).resolve().parent / "bw1100_mem1.json"
    output.write_text(json.dumps(fragment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(fragment, indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    if args.update_json:
        update_memory_curve(args.update_json, "mem1", args.capacity_gib, args.peak_gbps, curve)
        print(f"updated {args.update_json}: mem1", flush=True)


if __name__ == "__main__": main()
