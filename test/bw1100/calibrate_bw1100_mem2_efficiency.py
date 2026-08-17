#!/usr/bin/env python3
"""Calibrate BW1100 PCIe mem2: pinned-host to HBM transfer efficiency.

mem2 is the offload tier, not HBM3. Its bandwidth and capacity are host-specific,
so capacity defaults to MemTotal. ``mem2.GBps`` is the nominal one-way PCIe
peak and ``MB_efficiency`` stores observed pinned-copy bandwidth / peak. PCIe
5.0 x16 is 128 GB/s bidirectional, hence the default one-way peak is 64 GB/s.
For direction=both, the slower direction is used for each size.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from calibrate_bw1100_common import SYSTEM_JSON, require_bw1100, update_memory_curve

DEFAULT_SIZES_MB = (512, 256, 128, 64, 32, 16, 8, 4, 2, 1, .5, .25, .125, .0625)


def host_capacity_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"): return int(line.split()[1]) * 1024 / 2**30
    raise RuntimeError("Could not read MemTotal")


def choose_iters(latency_ms: float, requested: int, min_ms: float, max_iters: int) -> int:
    return min(max_iters, max(1, requested, int(min_ms / max(latency_ms, .001)) + 1))


def benchmark(nbytes: int, direction: str, warmup: int, iterations: int, min_ms: float,
              max_iters: int, samples: int) -> tuple[float, float, int]:
    host = torch.empty(max(4096, nbytes), dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(host.numel(), dtype=torch.uint8, device="cuda")
    src, dst = (host, dev) if direction == "h2d" else (dev, host)
    for _ in range(max(1, warmup)): dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); dst.copy_(src, non_blocking=True); stop.record(); stop.synchronize()
    actual_iters = choose_iters(start.elapsed_time(stop), iterations, min_ms, max_iters)
    timings = []
    for _ in range(samples):
        start.record()
        for _ in range(actual_iters): dst.copy_(src, non_blocking=True)
        stop.record(); stop.synchronize()
        timings.append(start.elapsed_time(stop) / 1000.0 / actual_iters)
    latency_s = statistics.median(timings)
    return host.numel() / latency_s / 1e9, latency_s, actual_iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mb", type=float, nargs="+", default=DEFAULT_SIZES_MB)
    parser.add_argument("--direction", choices=("both", "h2d", "d2h"), default="both")
    parser.add_argument("--peak-bidirectional-gbps", type=float, default=128.0,
                        help="PCIe advertised duplex peak (default: 128 GB/s)")
    parser.add_argument("--peak-unidirectional-gbps", type=float, default=None,
                        help="override duplex/2 if the supplied spec is already one-way")
    parser.add_argument("--allow-overpeak", action="store_true",
                        help="allow observed copy rate > nominal one-way peak")
    parser.add_argument("--capacity-gib", type=float, default=None, help="default: host MemTotal")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--samples", type=int, default=3,
                        help="independent timings per size and direction; use median (default: 3)")
    parser.add_argument("--min-ms", type=float, default=500.0)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--floor-eff", type=float, default=.05)
    parser.add_argument("--update-json", type=Path, nargs="?", const=SYSTEM_JSON)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.samples < 1: raise ValueError("samples must be positive")
    isa = require_bw1100(); capacity = args.capacity_gib if args.capacity_gib is not None else host_capacity_gib()
    dirs = ("h2d", "d2h") if args.direction == "both" else (args.direction,)
    print(f"Device: BW1100 ({isa}); host capacity={capacity:.1f} GiB; direction={args.direction}", flush=True)
    rows: list[tuple[float, float]] = []
    for index, mb in enumerate(sorted(set(args.sizes_mb), reverse=True), 1):
        results = []
        print(f"[{index:2d}/{len(set(args.sizes_mb))}] START {mb:g} MB", flush=True)
        try:
            for direction in dirs: results.append((direction, *benchmark(int(mb * 1e6), direction, args.warmup, args.iterations, args.min_ms, args.max_iters, args.samples)))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            torch.cuda.empty_cache(); print(f"  skip {mb:g} MB: {exc}", flush=True); continue
        gbps = min(item[1] for item in results)
        detail = " ".join(f"{item[0]}={item[1]:.2f}" for item in results)
        print(f"[{index:2d}] DONE {mb:g} MB  bottleneck={gbps:.2f} GB/s ({detail})", flush=True)
        rows.append((float(mb), gbps)); torch.cuda.empty_cache()
    if not rows: raise RuntimeError("No successful host/HBM transfer measurements")
    peak = (args.peak_unidirectional_gbps if args.peak_unidirectional_gbps is not None
            else args.peak_bidirectional_gbps / 2.0)
    if peak <= 0 or capacity <= 0: raise ValueError("peak and capacity must be positive")
    observed_peak = max(gbps for _, gbps in rows)
    if observed_peak > peak and not args.allow_overpeak:
        raise RuntimeError(
            f'observed pinned-copy rate {observed_peak:.3f} GB/s exceeds nominal '
            f'one-way PCIe peak {peak:.3f} GB/s; verify the vendor specification')
    curve = [[mb, round(min(1.0, max(args.floor_eff, gbps / peak)), 6)] for mb, gbps in rows] + [[0.0, args.floor_eff]]
    fragment = {"GiB": capacity, "GBps": peak, "MB_efficiency": curve}
    output = args.output or Path(__file__).resolve().parent / "bw1100_mem2.json"
    output.write_text(json.dumps(fragment, indent=2) + "\n", encoding="utf-8")
    print(f"Nominal mem2 one-way peak: {peak:.3f} GB/s; best observed: {observed_peak:.3f} GB/s", flush=True)
    print(json.dumps(fragment, indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    if args.update_json:
        update_memory_curve(args.update_json, "mem2", capacity, peak, curve)
        print(f"updated {args.update_json}: mem2", flush=True)


if __name__ == "__main__": main()
