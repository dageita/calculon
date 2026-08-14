#!/usr/bin/env python3
"""BW1100 Phase-1 roofline validation for isolated Linear/GEMM operators.

This draws the calibrated Calculon roofline and compares it with event-timed
BW1100 GEMMs across low-, transition-, and high-arithmetic-intensity shapes.
The default shape families are generic matrix aspect ratios, rather than H20 or
model-specific dimensions.  FP8 measurement uses the native hipBLASLt helper
from calibrate_bw1100_matrix_efficiency.py.

Examples:
  python test/bw1100/phase1_linear_ai_roofline.py --dtype float16 --plot /tmp/bw.png
  python test/bw1100/phase1_linear_ai_roofline.py --dtype float8 --predict-only
  python test/bw1100/phase1_linear_ai_roofline.py --dtype float16 --csv /tmp/bw.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from calibrate_bw1100_common import DTYPE_NBYTES, dtype_from_name, require_bw1100  # noqa: E402
from calibrate_bw1100_matrix_efficiency import (  # noqa: E402
    build_fp8_backend, time_fp8, time_torch_gemm, time_triton_fp8,
)


M_SWEEP = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
# name, c_in, c_out. These are deliberately generic to cover AI regimes.
FAMILIES = (
    ("wide", 4096, 4096),
    ("medium", 4096, 1024),
    ("narrow", 4096, 256),
    ("skinny", 4096, 64),
    ("tiny-n", 4096, 16),
)


@dataclass
class Row:
    family: str
    m: int
    k: int
    n: int
    ai: float
    gflops: float
    pred_tflops: float
    pred_f_us: float
    pred_m_us: float
    pred_roof_us: float
    pred_sum_us: float
    calibrated_bound: str
    calibrated_ridge_ai: float
    physical_bound: str
    physical_ridge_ai: float
    measured_us: float | None
    measured_tflops: float | None
    err_roof_pct: float | None
    err_sum_pct: float | None
    backend: str = ""
    skipped: str = ""


class CalibratedEngine:
    """One Calculon processor/memory curve, without communication simulation."""

    def __init__(self, cfg: dict, curve_key: str, scale: float) -> None:
        self._scale = scale
        self._items = {
            name: (entry["tflops"] * 1e12, [(x * scale, eff) for x, eff in entry[curve_key]])
            for name, entry in cfg.items()
        }

    def flops(self, dtype: str) -> float:
        return self._items[dtype][0]

    def throughput(self, dtype: str, work: float) -> float:
        peak, curve = self._items[dtype]
        for threshold, efficiency in curve:
            if work >= threshold:
                return peak * efficiency
        raise ValueError(f"{work} is below {dtype} calibration-curve coverage")


class CalibratedMemory:
    def __init__(self, cfg: dict) -> None:
        self.bandwidth = cfg["GBps"] * 1e9
        self._curve = [(mbytes * 1e6, efficiency) for mbytes, efficiency in cfg["MB_efficiency"]]

    def throughput(self, nbytes: float) -> float:
        for threshold, efficiency in self._curve:
            if nbytes >= threshold:
                return self.bandwidth * efficiency
        raise ValueError(f"{nbytes} is below memory calibration-curve coverage")


class BW1100RooflineSystem:
    """Minimal single-GPU Calculon roofline model.

    ``calculon.system.System`` also imports the multi-GPU C++ communication
    simulator.  A GEMM roofline has no communication, so this narrow reader
    intentionally consumes the same JSON curves without that optional .so.
    """

    def __init__(self, cfg: dict) -> None:
        self.matrix = CalibratedEngine(cfg["matrix"], "gflops_efficiency", 1e9)
        self.mem1 = CalibratedMemory(cfg["mem1"])
        self.linear_shape = cfg.get("linear_shape") or cfg.get("linear_small_n") or {}

    def linear_shape_time(self, m: int, k: int, n: int, dtype: str) -> float:
        model = self.linear_shape
        if int(k) != int(model.get("reference_k", -1)):
            return 0.0
        curves = model.get("latency_s", {}).get(dtype, {})
        if not isinstance(curves, dict):
            return 0.0
        bucket = str(int(n))
        if bucket not in curves:
            return 0.0
        points = curves[bucket]
        for min_m, latency_s in points:
            if m >= int(min_m):
                return float(latency_s)
        return float(points[-1][1]) if points else 0.0


TYPE_SIZES = {"float8": 1, "float16": 2, "bfloat16": 2, "float32": 4}


def align(value: int, multiple: int = 16) -> int:
    return max(multiple, (value // multiple) * multiple)


def prediction(system: BW1100RooflineSystem, m: int, k: int, n: int, dtype: str) -> tuple[float, float, float, float, float, float, float, str, float]:
    # This is exactly Linear.get_fw_flops()/get_fw_mem_accessed() for a
    # single GEMM: read A and B once, then write C once.
    flops = 2.0 * m * k * n
    nbytes = (m * k + k * n + m * n) * TYPE_SIZES[dtype]
    pred_f = system.linear_shape_time(m, k, n, dtype)
    if pred_f <= 0:
        pred_f = flops / system.matrix.throughput(dtype, flops)
    pred_m = nbytes / system.mem1.throughput(nbytes)
    roof, summed = max(pred_f, pred_m), pred_f + pred_m
    ai = flops / nbytes
    ridge = (flops / pred_f) / (nbytes / pred_m) if pred_f > 0 and pred_m > 0 else float("inf")
    physical_ridge = system.matrix.flops(dtype) / system.mem1.bandwidth
    physical_bound = "compute" if ai >= physical_ridge else "memory"
    return flops, ai, pred_f, pred_m, roof, summed, ridge, physical_bound, physical_ridge


def measure(m: int, k: int, n: int, dtype: str, fp8_binary: Path | None,
            fp8_backend: str, warmup: int, iterations: int) -> tuple[float, float, str]:
    if dtype != "float8":
        latency = time_torch_gemm(m, n, k, dtype_from_name(dtype), warmup, iterations)
        return latency, 2.0 * m * n * k / latency / 1e12, "torch"
    if fp8_backend == "triton":
        latency, backend = time_triton_fp8(m, n, k, warmup, iterations), "triton"
    elif fp8_backend == "hipblaslt":
        latency, backend = time_fp8(fp8_binary, m, n, k, warmup, iterations), "hipblaslt"
    else:
        triton = time_triton_fp8(m, n, k, warmup, iterations)
        hipblaslt = time_fp8(fp8_binary, m, n, k, warmup, iterations)
        latency, backend = min(((triton, "triton"), (hipblaslt, "hipblaslt")), key=lambda item: item[0])
    return latency, 2.0 * m * n * k / latency / 1e12, backend


def run_row(system: BW1100RooflineSystem, family: str, m: int, k: int, n: int, dtype: str,
            fp8_binary: Path | None, args: argparse.Namespace) -> Row:
    m, k, n = align(m), align(k), align(n)
    flops, ai, pred_f, pred_m, roof, summed, ridge, physical_bound, physical_ridge = prediction(system, m, k, n, dtype)
    measured_us = measured_tflops = err_roof = err_sum = None
    backend = ""
    skipped = ""
    if not args.predict_only:
        try:
            latency, measured_tflops, backend = measure(
                m, k, n, dtype, fp8_binary, args.fp8_backend, args.warmup, args.iterations)
            measured_us = latency * 1e6
            err_roof = 100 * (roof - latency) / latency
            err_sum = 100 * (summed - latency) / latency
        except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError) as exc:
            torch.cuda.empty_cache(); skipped = f"{type(exc).__name__}: {exc}"[:180]
    return Row(family, m, k, n, ai, flops / 1e9, flops / roof / 1e12,
               pred_f * 1e6, pred_m * 1e6, roof * 1e6, summed * 1e6,
               "compute" if pred_f >= pred_m else "memory", ridge, physical_bound, physical_ridge,
               measured_us, measured_tflops, err_roof, err_sum, backend, skipped)


def write_csv(path: Path, rows: Iterable[Row]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader(); writer.writerows(asdict(row) for row in rows)
    print(f"wrote {path}", flush=True)


def plot(path: Path, rows: list[Row], system: BW1100RooflineSystem, dtype: str) -> None:
    measured = [row for row in rows if row.measured_tflops is not None]
    if not measured:
        print("No measured rows: --plot needs measurement (omit --predict-only).", flush=True); return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plot requires matplotlib in the calibration image") from exc
    peak = system.matrix.flops(dtype) / 1e12
    bandwidth = system.mem1.bandwidth / 1e9
    xmin, xmax = min(row.ai for row in measured) / 1.5, max(row.ai for row in measured) * 1.5
    xs = [10 ** (math.log10(xmin) + i * (math.log10(xmax) - math.log10(xmin)) / 200) for i in range(201)]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(xs, [min(peak, ai * bandwidth / 1000) for ai in xs], "k-", lw=2,
            label=f"physical roofline: {peak:.0f} TF / {bandwidth:.0f} GB/s")
    ax.axvline(peak * 1000 / bandwidth, color="black", ls=":", lw=1,
               label=f"physical ridge: {peak * 1000 / bandwidth:.1f} FLOPs/byte")
    for family in sorted({row.family for row in measured}):
        sub = sorted((row for row in measured if row.family == family), key=lambda row: row.ai)
        ax.plot([row.ai for row in sub], [row.measured_tflops for row in sub], "o-", label=f"{family} measured")
        ax.plot([row.ai for row in sub], [row.pred_tflops for row in sub], "--", alpha=.75, label=f"{family} calibrated")
    ax.set_xscale("log"); ax.set_xlabel("Arithmetic intensity (FLOPs/byte)"); ax.set_ylabel("TFLOPS")
    ax.set_title(f"BW1100 {dtype} Linear roofline: measured vs Calculon")
    ax.grid(True, which="both", alpha=.25); ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=160); plt.close(fig)
    print(f"wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float8", "float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--system-json", type=Path, default=ROOT / "systems" / "BW1100.json")
    parser.add_argument("--m-sweep", type=int, nargs="+", default=M_SWEEP)
    parser.add_argument("--families", nargs="+", help="subset: " + ", ".join(name for name, _, _ in FAMILIES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--fp8-backend", choices=("auto", "triton", "hipblaslt"), default="auto",
                        help="FP8 measurement backend; auto matches matrix calibration (default: auto)")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    config = json.loads(args.system_json.read_text(encoding="utf-8")); system = BW1100RooflineSystem(config)
    selected = [item for item in FAMILIES if args.families is None or item[0] in args.families]
    if not selected: raise SystemExit("No valid families selected")
    if not args.predict_only:
        isa = require_bw1100(); torch.cuda.set_device(0)
        fp8_binary = (build_fp8_backend(isa)
                      if args.dtype == "float8" and args.fp8_backend in ("auto", "hipblaslt") else None)
        print(f"Device: {torch.cuda.get_device_name(0)} ({isa})", flush=True)
    else:
        fp8_binary = None
    print(f"System={args.system_json} dtype={args.dtype} matrix peak={system.matrix.flops(args.dtype)/1e12:.3f} TF "
          f"HBM={system.mem1.bandwidth/1e9:.1f} GB/s", flush=True)
    rows: list[Row] = []
    cases = [(name, m, k, n) for name, k, n in selected for m in args.m_sweep]
    for index, (family, m, k, n) in enumerate(cases, 1):
        print(f"[{index:2d}/{len(cases)}] {family}: M={m} K={k} N={n}", flush=True)
        rows.append(run_row(system, family, m, k, n, args.dtype, fp8_binary, args))
    physical_ridge = system.matrix.flops(args.dtype) / system.mem1.bandwidth
    print(f"Physical ridge AI={physical_ridge:.2f} FLOPs/byte (peak/HBM); calibrated boundary is shape/work dependent.", flush=True)
    header = "family     M      K      N      AI  cal-bound phy-bound  roof_us  measured_us  roof_err%"
    print("\n" + header + "\n" + "-" * len(header))
    for row in rows:
        measured = "-" if row.measured_us is None else f"{row.measured_us:11.2f}"
        error = "-" if row.err_roof_pct is None else f"{row.err_roof_pct:+8.1f}"
        backend = f" {row.backend}" if row.backend else ""
        print(f"{row.family:9s} {row.m:6d} {row.k:6d} {row.n:6d} {row.ai:7.2f} {row.calibrated_bound:9s} "
              f"{row.physical_bound:9s} {row.pred_roof_us:9.2f} {measured:>12s} {error:>9s}{backend}")
    measured_rows = [row for row in rows if row.err_roof_pct is not None]
    if measured_rows:
        roof_mape = sum(abs(row.err_roof_pct) for row in measured_rows) / len(measured_rows)
        sum_mape = sum(abs(row.err_sum_pct) for row in measured_rows) / len(measured_rows)
        print(f"\nMAPE roofline max={roof_mape:.2f}%  sum ablation={sum_mape:.2f}%", flush=True)
    if args.csv: write_csv(args.csv, rows)
    if args.plot: plot(args.plot, rows, system, args.dtype)


if __name__ == "__main__": main()
