#!/usr/bin/env python3
"""Validate BW1100 Calculon vector curves with the calibrated elementwise op.

The measured operator intentionally matches calibrate_bw1100_vector_efficiency:
``x * y + x`` is charged as one Calculon ElementWise FLOP per element.  FP8
uses the same fused gfx938 Triton implementation as calibration; other dtypes
use the same eager DTK PyTorch expression.  This validates vector efficiency,
HBM efficiency, and vector_launch_s together, independently of GEMM roofline.

Examples:
  python test/bw1100/phase1_vector_roofline.py --dtype float8
  python test/bw1100/phase1_vector_roofline.py --dtype float16 --csv /tmp/v.csv --plot /tmp/v.png
  python test/bw1100/phase1_vector_roofline.py --dtype float8 --elements 1024 1048576 16777216
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from calibrate_bw1100_common import dtype_from_name, require_bw1100  # noqa: E402
from calibrate_bw1100_vector_efficiency import (  # noqa: E402
    bounded_iterations, default_elements, make_inputs, vector_kernel,
)
from triton_fp8_vector import benchmark_fp8_vector  # noqa: E402


@dataclass
class Row:
    elements: int
    gflops: float
    payload_mb: float
    pred_tflops: float
    pred_compute_us: float
    pred_memory_us: float
    pred_launch_us: float
    pred_roof_us: float
    bound: str
    measured_us: float | None
    measured_tflops: float | None
    err_pct: float | None
    skipped: str = ""


class Engine:
    def __init__(self, cfg: dict) -> None:
        self._items = {
            name: (entry["tflops"] * 1e12, [(x * 1e9, eff) for x, eff in entry["gflops_efficiency"]])
            for name, entry in cfg.items()
        }

    def throughput(self, dtype: str, work_flops: float) -> float:
        peak, curve = self._items[dtype]
        for threshold, efficiency in curve:
            if work_flops >= threshold:
                return peak * efficiency
        raise ValueError(f"{work_flops} FLOPs is below {dtype} vector curve coverage")


class Memory:
    def __init__(self, cfg: dict) -> None:
        self.bandwidth = cfg["GBps"] * 1e9
        self.curve = [(mb * 1e6, eff) for mb, eff in cfg["MB_efficiency"]]

    def throughput(self, nbytes: float) -> float:
        for threshold, efficiency in self.curve:
            if nbytes >= threshold:
                return self.bandwidth * efficiency
        raise ValueError(f"{nbytes} bytes is below mem1 curve coverage")


TYPE_BYTES = {"float8": 1, "int8": 1, "float16": 2, "bfloat16": 2, "float32": 4}


def memory_bytes(elements: int, dtype: str) -> int:
    """Logical traffic for the exact measured implementation.

    FP8 is one fused kernel (two reads plus one FP8 write).  The eager PyTorch
    expression has a multiply temporary, so it performs two input reads, one
    temporary write/read, one additional x read, and one output write.
    """
    bpe = TYPE_BYTES[dtype]
    return elements * (3 if dtype == "float8" else 5) * bpe


def measure(elements: int, dtype: str, warmup: int, iterations: int) -> float:
    if dtype == "float8":
        return benchmark_fp8_vector(elements, warmup, iterations)
    x, y = make_inputs(elements, dtype)
    for _ in range(warmup):
        z = vector_kernel(x, y, dtype)
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        z = vector_kernel(x, y, dtype)
    stop.record()
    stop.synchronize()
    latency = start.elapsed_time(stop) / 1000.0 / iterations
    del x, y, z
    return latency


def predict(engine: Engine, memory: Memory, launch_s: float, elements: int, dtype: str) -> tuple[float, float, float, float, str]:
    flops = float(elements)  # Calculon ElementWise convention, shared with calibration.
    compute_s = flops / engine.throughput(dtype, flops)
    mem_s = memory_bytes(elements, dtype) / memory.throughput(memory_bytes(elements, dtype))
    roof_s = max(compute_s, mem_s, launch_s)
    bound = "compute" if roof_s == compute_s else "memory" if roof_s == mem_s else "launch"
    return compute_s, mem_s, launch_s, roof_s, bound


def write_csv(path: Path, rows: Iterable[Row]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    print(f"wrote {path}", flush=True)


def plot(path: Path, rows: list[Row], dtype: str) -> None:
    valid = [row for row in rows if row.measured_us is not None]
    if not valid:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plot requires matplotlib") from exc
    valid.sort(key=lambda row: row.elements)
    x = [row.elements for row in valid]
    measured = [row.measured_us for row in valid]
    predicted = [row.pred_roof_us for row in valid]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, measured, "o-", label="measured")
    ax.plot(x, predicted, "s--", label="Calculon roofline")
    ax.plot(x, [row.pred_compute_us for row in valid], ":", label="vector compute")
    ax.plot(x, [row.pred_memory_us for row in valid], ":", label="HBM")
    ax.axhline(valid[0].pred_launch_us, color="grey", ls=":", label="vector launch floor")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("elements"); ax.set_ylabel("latency (us)")
    ax.set_title(f"BW1100 {dtype} vector roofline: measured vs Calculon")
    ax.grid(True, which="both", alpha=.25); ax.legend(); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=160); plt.close(fig)
    print(f"wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float8", "int8", "float16", "bfloat16", "float32"), default="float8")
    parser.add_argument("--system-json", type=Path, default=ROOT / "systems" / "BW1100.json")
    parser.add_argument("--elements", type=int, nargs="+", help="optional quick sweep; default is calibration ladder")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=3,
                        help="independent timings per point; report their median (default: 3)")
    parser.add_argument("--max-total-gflops", type=float, default=1000.0)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    cfg = json.loads(args.system_json.read_text(encoding="utf-8"))
    engine, memory = Engine(cfg["vector"]), Memory(cfg["mem1"])
    launch_s = float(cfg.get("vector_launch_s", 0.0) or 0.0)
    elements = args.elements or default_elements()
    if not args.predict_only:
        isa = require_bw1100(); torch.cuda.set_device(0)
        print(f"Device: {torch.cuda.get_device_name(0)} ({isa})", flush=True)
    print(f"System={args.system_json} dtype={args.dtype} vector launch={launch_s * 1e6:.2f} us; "
          f"{len(elements)} element counts", flush=True)

    rows: list[Row] = []
    for index, count in enumerate(elements, 1):
        compute_s, mem_s, launch_s, roof_s, bound = predict(engine, memory, launch_s, count, args.dtype)
        measured_s = measured_tflops = err_pct = None
        if not args.predict_only:
            iterations = bounded_iterations(count, args.iterations, args.max_total_gflops)
            warmup = min(args.warmup, max(1, iterations // 5))
            try:
                measured_s = statistics.median(
                    measure(count, args.dtype, warmup, iterations)
                    for _ in range(args.samples)
                )
                measured_tflops = count / measured_s / 1e12
                err_pct = 100.0 * (roof_s - measured_s) / measured_s
            except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError) as exc:
                torch.cuda.empty_cache()
                skipped = f"{type(exc).__name__}: {exc}"[:180]
            else:
                skipped = ""
        else:
            skipped = ""
        row = Row(count, count / 1e9, memory_bytes(count, args.dtype) / 1e6,
                  count / roof_s / 1e12, compute_s * 1e6, mem_s * 1e6,
                  launch_s * 1e6, roof_s * 1e6, bound,
                  None if measured_s is None else measured_s * 1e6,
                  measured_tflops, err_pct, skipped)
        rows.append(row)
        measured = "-" if row.measured_us is None else f"{row.measured_us:9.2f}"
        error = "-" if row.err_pct is None else f"{row.err_pct:+7.1f}"
        print(f"[{index:3d}/{len(elements)}] {count:10d} elem {row.pred_roof_us:9.2f} us "
              f"{measured:>10s} us {bound:7s} err={error}", flush=True)

    measured_rows = [row for row in rows if row.err_pct is not None]
    if measured_rows:
        errors = sorted(abs(row.err_pct) for row in measured_rows)
        p50 = errors[(len(errors) - 1) // 2]
        p95 = errors[min(len(errors) - 1, math.ceil(.95 * len(errors)) - 1)]
        print(f"MAPE={sum(errors) / len(errors):.2f}% P50={p50:.2f}% "
              f"P95={p95:.2f}% max={errors[-1]:.2f}%", flush=True)
    if args.csv:
        write_csv(args.csv, rows)
    if args.plot:
        plot(args.plot, rows, args.dtype)


if __name__ == "__main__":
    main()
