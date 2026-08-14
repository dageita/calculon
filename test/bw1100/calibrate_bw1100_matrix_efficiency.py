#!/usr/bin/env python3
"""Calibrate BW1100 GEMM efficiency for Calculon.

FP16/BF16/FP32 use the DTK PyTorch build. INT8 uses hipBLASLt. FP8 automatically
selects the faster of a gfx938 Triton/HCU kernel and hipBLASLt for every shape;
DTK 26.04 hipBLASLt alone tops out far below the hardware on large FP8 GEMMs.
A separate du_mma compute-only probe verifies the physical FP8 engine.

BW1100's product peak is 708 TFLOPS for FP8. The update gate validates native
FP8 support with the compute-only du_mma probe, rather than requiring a GEMM to
reach an arbitrary fraction of peak. Use --allow-fallback-fp8 only for explicit
hipBLASLt fallback diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from calibrate_bw1100_common import (
    SYSTEM_JSON, configured_peak, dtype_from_name, enforce_monotonic_efficiency, merge_efficiency_bins,
    require_bw1100, update_linear_shape_model, update_system_curve,
)

ROOT = Path(__file__).resolve().parent
INT8_PEAK_TFLOPS = 708.0  # BW1100 product specification, same tensor peak as FP8.
def default_shapes() -> list[tuple[int, int, int]]:
    """BW1100-native dense sweep, independent of any H20/LLM model shape.

    The fine 16-wide ladder is intentional: small GEMMs are launch-bound and
    their efficiency changes quickly.  Projection-shaped K=4096 probes make
    the resulting one-dimensional Calculon curve representative of common
    Linear layers as well as square GEMMs.
    """
    # Stop at 4096.  The 4608--8192 square GEMMs are a distinct, less
    # efficient tiling regime on this DTK stack.  Including them as the
    # highest-work samples in a one-dimensional monotonic curve would cap the
    # measured 4096-wide Linear projections below their actual throughput.
    squares = (
        list(range(16, 257, 16))
        + list(range(288, 1025, 32))
        + list(range(1152, 4097, 128))
    )
    rectangular_base = list(range(64, 1025, 64))
    shapes = {(size, size, size) for size in squares}
    shapes.update((base, base, 2 * base) for base in rectangular_base)
    shapes.update((base, 2 * base, 4 * base) for base in (64, 128, 192, 256, 384, 512, 768, 1024))
    shapes.update((m, n, 4096) for n in (256, 1024, 4096) for m in
                  (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192))
    return sorted(shapes, key=lambda x: 2 * x[0] * x[1] * x[2], reverse=True)


def diagnostic_shapes() -> list[tuple[int, int, int]]:
    """Small-N probes reported separately from the 1-D Calculon curve.

    A gflops-only lookup cannot model the severe tile under-utilisation of
    N=16/64 GEMMs.  Keep these cases visible to validate a workload, but do
    not let their throughput alter the generic matrix efficiency curve.
    """
    return [(m, n, 4096) for n in (16, 64) for m in
            (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)]


def build_native_backend(dtype: str, isa: str) -> Path:
    if dtype not in ("float8", "int8"):
        raise ValueError(f"No native hipBLASLt backend for {dtype}")
    stem = "fp8" if dtype == "float8" else "int8"
    source = ROOT / f"hipblaslt_{stem}_gemm.cpp"
    binary = ROOT / f".hipblaslt_{stem}_gemm"
    if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(
            ["hipcc", "-O3", f"--offload-arch={isa}", str(source), "-o", str(binary), "-lhipblaslt", "-lamdhip64"],
            check=True,
        )
    return binary


def build_fp8_backend(isa: str) -> Path:
    """Compatibility import used by the BW1100 roofline script."""
    return build_native_backend("float8", isa)


def build_fp8_peak_probe(isa: str) -> Path:
    source = ROOT / "du_mma_fp8_peak.cpp"
    binary = ROOT / ".du_mma_fp8_peak"
    if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(
            ["hipcc", "-O3", f"--offload-arch={isa}", str(source), "-o", str(binary), "-lamdhip64"],
            check=True,
        )
    return binary


def fp8_instruction_peak(binary: Path) -> float:
    completed = subprocess.run([str(binary), "4096", "2048"], check=True, text=True, capture_output=True)
    return float(json.loads(completed.stdout)["tflops"])


def time_torch_gemm(m: int, n: int, k: int, dtype: torch.dtype, warmup: int, iterations: int) -> float:
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        torch.mm(a, b)
    stop.record(); stop.synchronize()
    return start.elapsed_time(stop) / 1000.0 / iterations


def time_native_gemm(binary: Path, m: int, n: int, k: int, warmup: int, iterations: int) -> float:
    completed = subprocess.run(
        [str(binary), str(m), str(n), str(k), str(warmup), str(iterations)],
        check=True, text=True, capture_output=True,
    )
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr, flush=True)
    return float(json.loads(completed.stdout)["latency_s"])


def time_fp8(binary: Path, m: int, n: int, k: int, warmup: int, iterations: int) -> float:
    """Compatibility import used by the BW1100 roofline script."""
    return time_native_gemm(binary, m, n, k, warmup, iterations)


def time_triton_fp8(m: int, n: int, k: int, warmup: int, iterations: int) -> float:
    """Time a real FP8 GEMM emitted through Triton's gfx938 HCU path."""
    try:
        from triton_fp8_gemm import benchmark_fp8_gemm
    except ImportError as exc:
        raise RuntimeError(
            "The optimized FP8 backend needs Triton from the BW1100 PyTorch image. "
            "Use --fp8-backend hipblaslt only for DTK fallback diagnostics."
        ) from exc
    return benchmark_fp8_gemm(m, n, k, warmup, iterations)


def bounded_iterations(m: int, n: int, k: int, requested: int,
                       max_total_gflops: float | None) -> int:
    """Apply an explicit user work cap without making it the default policy."""
    if max_total_gflops is None:
        return requested
    work_gflops = 2.0 * m * n * k / 1e9
    return max(1, min(requested, int(max_total_gflops / max(work_gflops, 1e-12))))


def iterations_for_latency(probe_latency: float, args: argparse.Namespace,
                           m: int, n: int, k: int) -> int:
    """Choose enough repetitions for a stable event-timed GPU average."""
    target = max(args.min_iterations, int((args.min_timing_ms / 1e3) / max(probe_latency, 1e-9) + 0.999))
    maximum = bounded_iterations(m, n, k, args.iterations, args.max_total_gflops)
    return max(1, min(target, maximum))


def robust_projection_latency(measure_once, repeats: int = 3) -> float:
    """Use a median for K=4096 Linear projections to reject scheduler spikes."""
    return statistics.median(measure_once() for _ in range(repeats))


def write_diagnostic_csv(path: Path, samples: list[tuple[tuple[int, int, int], float, float, bool]]) -> None:
    """Persist small-N measurements without changing Calculon's JSON schema."""
    diagnostics = [(shape, latency, tflops) for shape, latency, tflops, included in samples if not included]
    if not diagnostics:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("m", "n", "k", "gflops", "latency_us", "tflops"))
        writer.writeheader()
        for (m, n, k), latency, tflops in diagnostics:
            writer.writerow({"m": m, "n": n, "k": k, "gflops": 2.0 * m * n * k / 1e9,
                             "latency_us": latency * 1e6, "tflops": tflops})
    print(f"wrote small-N diagnostic CSV {path}", flush=True)


def matrix_peak(dtype: str, override: float | None) -> float:
    """Read configured peak, with a migration-safe default for new INT8."""
    if override is not None:
        return override
    try:
        return configured_peak("matrix", dtype)
    except RuntimeError:
        if dtype != "int8":
            raise
        print(
            f"matrix.int8.tflops is absent from {SYSTEM_JSON}; using BW1100 INT8 physical peak "
            f"{INT8_PEAK_TFLOPS:.1f} TFLOPS. --update-json will add it.",
            file=sys.stderr, flush=True,
        )
        return INT8_PEAK_TFLOPS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float8", "int8", "float16", "bfloat16", "float32"), default="float8")
    parser.add_argument("--sizes", type=int, nargs="+", help="optional square-only quick sweep; default is the BW1100 dense sweep")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=500,
                        help="maximum timed iterations per shape (default: 500)")
    parser.add_argument("--min-iterations", type=int, default=20,
                        help="minimum timed iterations, unless an explicit work cap is lower (default: 20)")
    parser.add_argument("--min-timing-ms", type=float, default=100.0,
                        help="target total event-timing duration per shape (default: 100 ms)")
    parser.add_argument("--max-total-gflops", type=float, default=None,
                        help="optional explicit per-shape work cap; disabled by default")
    parser.add_argument("--include-small-n-diagnostics", dest="include_small_n_diagnostics",
                        action="store_true", default=True,
                        help="measure N=16/64 K=4096 probes (default); report them but exclude them from the JSON curve")
    parser.add_argument("--no-small-n-diagnostics", dest="include_small_n_diagnostics", action="store_false",
                        help="skip the default N=16/64 K=4096 diagnostic probes")
    parser.add_argument("--diagnostic-csv", type=Path,
                        help="optional CSV destination for the excluded N=16/64 diagnostic measurements")
    parser.add_argument("--min-native-fp8-ratio", type=float, default=0.50,
                        help="minimum du_mma instruction/official FP8 peak proving native hardware support (default: 0.50)")
    parser.add_argument("--allow-fallback-fp8", action="store_true",
                        help="permit hipBLASLt-only FP8 diagnostics; never use that fallback curve as a native-FP8 result")
    parser.add_argument("--fp8-backend", choices=("auto", "triton", "hipblaslt"), default="auto",
                        help="FP8 backend; auto retains the faster valid result per shape (default: auto)")
    parser.add_argument("--peak-tflops", type=float, default=None,
                        help="override the dtype peak in systems/BW1100.json")
    parser.add_argument("--update-json", type=Path, nargs="?", const=SYSTEM_JSON,
                        help="write the curve into this system JSON (default: systems/BW1100.json)")
    parser.add_argument("--output", type=Path, default=None, help="write a Calculon matrix fragment")
    args = parser.parse_args()
    if args.iterations < 1 or args.min_iterations < 1 or args.min_timing_ms <= 0:
        parser.error("--iterations, --min-iterations and --min-timing-ms must be positive")
    isa = require_bw1100()
    binary = (build_native_backend(args.dtype, isa)
              if args.dtype == "int8" or (args.dtype == "float8" and args.fp8_backend in ("auto", "hipblaslt")) else None)
    instruction_tflops = None
    native_fp8_ok = True
    if args.dtype == "float8":
        peak_probe = build_fp8_peak_probe(isa)
        instruction_tflops = fp8_instruction_peak(peak_probe)
        peak_for_probe = matrix_peak(args.dtype, args.peak_tflops)
        native_fp8_ok = instruction_tflops / peak_for_probe >= args.min_native_fp8_ratio
        print(
            f"FP8 hardware self-test: {instruction_tflops:.3f} TFLOPS "
            f"({instruction_tflops / peak_for_probe:.1%} of {peak_for_probe:.1f}T), backend={args.fp8_backend}",
            flush=True,
        )
    curve_shapes = [(size, size, size) for size in args.sizes] if args.sizes else default_shapes()
    shapes = [(shape, True) for shape in curve_shapes]
    if args.include_small_n_diagnostics and not args.sizes:
        known = set(curve_shapes)
        shapes.extend((shape, False) for shape in diagnostic_shapes() if shape not in known)
    print(f"Measuring {len(shapes)} BW1100 GEMM shapes ({len(curve_shapes)} curve shapes; "
          f"small-N diagnostics={'on' if args.include_small_n_diagnostics and not args.sizes else 'off'})", flush=True)
    samples: list[tuple[tuple[int, int, int], float, float, bool]] = []
    for index, ((m, n, k), contributes_to_curve) in enumerate(shapes, 1):
        probe_iterations = min(5, args.iterations)
        probe_warmup = min(args.warmup, max(1, probe_iterations))
        print(f"[{index:3d}/{len(shapes)}] START {m}x{n}x{k}  curve={'yes' if contributes_to_curve else 'no'} "
              f"probe_iters={probe_iterations}", flush=True)
        selected_backend = args.dtype
        if args.dtype == "float8" and args.fp8_backend == "auto":
            triton_latency = time_triton_fp8(m, n, k, probe_warmup, probe_iterations)
            hipblaslt_latency = time_native_gemm(binary, m, n, k, probe_warmup, probe_iterations)
            probe_latency, selected_backend = min(
                ((triton_latency, "triton"), (hipblaslt_latency, "hipblaslt")), key=lambda item: item[0]
            )
        elif args.dtype == "float8" and args.fp8_backend == "triton":
            probe_latency = time_triton_fp8(m, n, k, probe_warmup, probe_iterations)
            selected_backend = "triton"
        elif binary:
            probe_latency = time_native_gemm(binary, m, n, k, probe_warmup, probe_iterations)
            selected_backend = "hipblaslt"
        else:
            probe_latency = time_torch_gemm(m, n, k, dtype_from_name(args.dtype), probe_warmup, probe_iterations)
        iterations = iterations_for_latency(probe_latency, args, m, n, k)
        warmup = min(args.warmup, max(1, iterations // 5))
        if args.dtype == "float8" and selected_backend == "triton":
            measure_once = lambda: time_triton_fp8(m, n, k, warmup, iterations)
        elif args.dtype == "float8" and selected_backend == "hipblaslt":
            measure_once = lambda: time_native_gemm(binary, m, n, k, warmup, iterations)
        elif binary:
            measure_once = lambda: time_native_gemm(binary, m, n, k, warmup, iterations)
        else:
            measure_once = lambda: time_torch_gemm(m, n, k, dtype_from_name(args.dtype), warmup, iterations)
        is_projection = k == 4096 and n in (16, 64, 256, 1024, 4096)
        latency = robust_projection_latency(measure_once) if is_projection else measure_once()
        tflops = 2.0 * m * n * k / latency / 1e12
        samples.append(((m, n, k), latency, tflops, contributes_to_curve))
        backend_note = f"  backend={selected_backend}" if args.dtype == "float8" else ""
        print(f"[{index:3d}/{len(shapes)}] DONE  {m}x{n}x{k}  {latency * 1e6:9.2f} us  "
              f"{tflops:8.3f} TFLOP/s  iters={iterations}"
              f"{' median=3' if is_projection else ''}{backend_note}", flush=True)
    peak = matrix_peak(args.dtype, args.peak_tflops)
    if peak <= 0: raise ValueError("--peak-tflops must be positive")
    best_tflops = max(tflops for _, _, tflops, _ in samples)
    if args.diagnostic_csv:
        write_diagnostic_csv(args.diagnostic_csv, samples)
    fallback_fp8 = args.dtype == "float8" and (not native_fp8_ok or args.fp8_backend == "hipblaslt")
    if fallback_fp8:
        message = (
            f"FP8 fallback detected: backend={args.fp8_backend}, GEMM best={best_tflops:.3f} TFLOPS, "
            f"du_mma self-test={instruction_tflops:.3f} TFLOPS. "
            "Use --fp8-backend auto/triton with a working native self-test for production calibration."
        )
        if not args.allow_fallback_fp8:
            raise RuntimeError(message)
        print(f"WARNING: {message}", file=sys.stderr, flush=True)
    points = ((2.0 * m * n * k / 1e9, tflops / peak)
              for (m, n, k), _, tflops, contributes_to_curve in samples if contributes_to_curve)
    curve = enforce_monotonic_efficiency(merge_efficiency_bins(points, rel_tol=0.0), floor_eff=0.01)
    output = args.output or ROOT / f"bw1100_matrix_{args.dtype}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({args.dtype: {"tflops": peak, "gflops_efficiency": curve}}, indent=2) + "\n")
    print(json.dumps({args.dtype: {"tflops": peak, "gflops_efficiency": curve}}, indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    if args.update_json:
        if fallback_fp8:
            raise RuntimeError("Refusing --update-json for a fallback FP8 measurement; omit --update-json for diagnostics.")
        update_system_curve(args.update_json, "matrix", args.dtype, peak, curve)
        print(f"updated {args.update_json}: matrix.{args.dtype}", flush=True)
        if update_linear_shape_model(args.update_json, args.dtype, 4096, samples):
            print(f"updated {args.update_json}: linear_shape.{args.dtype}", flush=True)


if __name__ == "__main__":
    main()
