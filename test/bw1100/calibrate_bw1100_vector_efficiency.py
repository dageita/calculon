#!/usr/bin/env python3
"""Calibrate BW1100 elementwise-vector efficiency using the DTK PyTorch runtime.

Vector operations are bandwidth/launch bound, so unlike matrix GEMM they have
no useful published hardware-TFLOPS peak.  The default peak is therefore the
maximum measured throughput in the same calibration run.  This keeps the
resulting efficiency curve self-consistent for every storage dtype, including
FP8 and INT8.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from calibrate_bw1100_common import (
    SYSTEM_JSON, dtype_from_name, enforce_monotonic_efficiency, merge_efficiency_bins,
    require_bw1100, update_system_curve,
    update_vector_launch_floor,
)
from triton_fp8_vector import benchmark_fp8_vector

def default_elements() -> list[int]:
    """BW1100-native size ladder with deliberately dense small element counts."""
    tiny = range(256, 16 * 1024 + 1, 256)
    small = range(20 * 1024, 64 * 1024 + 1, 4 * 1024)
    medium = range(80 * 1024, 1024 * 1024 + 1, 64 * 1024)
    large = (2 << 20, 4 << 20, 8 << 20, 16 << 20, 32 << 20, 64 << 20, 128 << 20, 256 << 20)
    return sorted(set(tiny) | set(small) | set(medium) | set(large), reverse=True)


def bounded_iterations(count: int, requested: int, max_total_gflops: float) -> int:
    """Cap large-vector timing work while retaining enough small-vector repeats."""
    work_gflops = count / 1e9  # Calculon's ElementWise model: one operation per element.
    return max(1, min(requested, int(max_total_gflops / max(work_gflops, 1e-12))))


def make_inputs(count: int, dtype_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype_name == "float8":
        # This is only used by the explicitly selected diagnostic fallback.
        # BW1100/gfx938 uses the standard E4M3 encoding, not AMD's FNUZ variant.
        storage = torch.float8_e4m3fn
        return (
            torch.randn(count, device="cuda", dtype=torch.bfloat16).to(storage),
            torch.randn(count, device="cuda", dtype=torch.bfloat16).to(storage),
        )
    if dtype_name == "int8":
        # randn does not construct integral tensors.  Values are intentionally
        # small enough that x*y+x stays in the signed INT8 range, so timing
        # measures the requested storage/operator dtype rather than overflow.
        return (
            torch.randint(-8, 8, (count,), device="cuda", dtype=torch.int8),
            torch.randint(-8, 8, (count,), device="cuda", dtype=torch.int8),
        )
    dtype = dtype_from_name(dtype_name)
    return torch.randn(count, device="cuda", dtype=dtype), torch.randn(count, device="cuda", dtype=dtype)


def vector_kernel(x: torch.Tensor, y: torch.Tensor, dtype_name: str) -> torch.Tensor:
    if dtype_name == "float8":
        return x.to(torch.bfloat16) * y.to(torch.bfloat16) + x.to(torch.bfloat16)
    return x * y + x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float8", "int8", "float16", "bfloat16", "float32"), default="float8")
    parser.add_argument("--elements", type=int, nargs="+", help="optional quick sweep; default is the BW1100 dense sweep")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=3,
                        help="independent timings per size; their median is calibrated (default: 3)")
    parser.add_argument("--max-total-gflops", type=float, default=1000.0,
                        help="per-size work cap; large vectors automatically use fewer iterations (default: 1000)")
    parser.add_argument("--peak-tflops", type=float, default=None,
                        help="optional override; default is the maximum measured throughput in this run")
    parser.add_argument("--fp8-backend", choices=("triton", "bf16-fallback"), default="triton",
                        help="FP8 implementation (default: one fused gfx938 Triton kernel; "
                             "bf16-fallback is diagnostic only)")
    parser.add_argument("--update-json", type=Path, nargs="?", const=SYSTEM_JSON,
                        help="write the curve into this system JSON (default: systems/BW1100.json)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.dtype == "float8" and args.fp8_backend == "bf16-fallback" and args.update_json:
        parser.error("--fp8-backend bf16-fallback is diagnostic only and cannot be used with --update-json")
    isa = require_bw1100()
    elements = args.elements or default_elements()
    backend = args.fp8_backend if args.dtype == "float8" else "torch"
    print(f"Measuring {len(elements)} BW1100-native dense vector sizes (backend={backend}, isa={isa})", flush=True)
    samples: list[tuple[int, float, float]] = []
    for index, count in enumerate(elements, 1):
        iterations = bounded_iterations(count, args.iterations, args.max_total_gflops)
        warmup = min(args.warmup, max(1, iterations // 5))
        print(f"[{index:3d}/{len(elements)}] START {count:10d} elements  warmup={warmup} iters={iterations}", flush=True)
        timings: list[float] = []
        for _ in range(args.samples):
            if args.dtype == "float8" and args.fp8_backend == "triton":
                timings.append(benchmark_fp8_vector(count, warmup, iterations))
            else:
                x, y = make_inputs(count, args.dtype)
                for _ in range(warmup):
                    z = vector_kernel(x, y, args.dtype)
                torch.cuda.synchronize()
                start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                for _ in range(iterations):
                    z = vector_kernel(x, y, args.dtype)
                stop.record()
                stop.synchronize()
                timings.append(start.elapsed_time(stop) / 1000.0 / iterations)
                del x, y, z
        latency = statistics.median(timings)
        # Calculon's ElementWise model accounts for one operation per element.
        tflops = count / latency / 1e12
        samples.append((count, latency, tflops))
        print(f"[{index:3d}/{len(elements)}] DONE  {count:10d} elements  {latency * 1e6:9.2f} us  "
              f"{tflops * 1e3:8.2f} GFLOP/s  backend={backend}", flush=True)
    peak = args.peak_tflops if args.peak_tflops is not None else max(tflops for _, _, tflops in samples)
    if peak <= 0: raise ValueError("--peak-tflops must be positive")
    print(f"Vector peak_tflops={peak:.9f} ({'override' if args.peak_tflops is not None else 'auto: maximum measured'})", flush=True)
    output = args.output or Path(__file__).resolve().parent / f"bw1100_vector_{args.dtype}.json"
    # Keep every distinct native size. The dense tiny ladder captures the
    # launch-bound region and yields more points than H20's lookup curve.
    curve = enforce_monotonic_efficiency(
        merge_efficiency_bins(((n / 1e9, tflops / peak) for n, _, tflops in samples), rel_tol=0.0),
        floor_eff=0.001,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({args.dtype: {"tflops": peak, "gflops_efficiency": curve}}, indent=2) + "\n")
    print(json.dumps({args.dtype: {"tflops": peak, "gflops_efficiency": curve}}, indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    if args.update_json:
        update_system_curve(args.update_json, "vector", args.dtype, peak, curve)
        print(f"updated {args.update_json}: vector.{args.dtype}", flush=True)
        # The eager FP16/BF16/FP32 vector expression dispatches more than one
        # kernel.  Its stable tiny-size median is therefore the effective
        # Calculon launch floor, not a hardware one-kernel launch measurement.
        tiny = [latency for count, latency, _ in samples if count <= 16 * 1024]
        if tiny:
            launch = statistics.median(tiny)
            update_vector_launch_floor(args.update_json, launch)
            print(f"updated {args.update_json}: vector_launch_s={launch:.9g}", flush=True)


if __name__ == "__main__":
    main()
