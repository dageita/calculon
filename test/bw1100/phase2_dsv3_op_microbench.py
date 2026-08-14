#!/usr/bin/env python3
"""BW1100 Phase 2: isolated DeepSeek-V3 operator validation (H2).

Unlike H20's CUDA implementation, this uses DTK/HIP PyTorch timing and the
same FP8 GEMM dispatch as BW1100 Phase 0: gfx938 Triton by default, with an
optional hipBLASLt diagnostic path.  FP8 BMM has no trustworthy batched
PyTorch API on this stack, so BMM is benchmarked and predicted as BF16 by
default.  That is an explicit baseline, not a claimed FP8 BMM measurement.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = Path(__file__).resolve().parent
H20_CATALOG = ROOT / "test" / "h20" / "phase2_dsv3_op_catalog.py"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
from calibrate_bw1100_common import require_bw1100  # noqa: E402
from calibrate_bw1100_matrix_efficiency import (  # noqa: E402
    build_native_backend, time_native_gemm, time_triton_fp8,
)

spec = importlib.util.spec_from_file_location("bw_phase2_catalog_source", H20_CATALOG)
catalog = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = catalog
spec.loader.exec_module(catalog)


@dataclass
class Row:
    group: str; block: str; name: str; cls: str; stage: str; kernel: str
    m: int; n: int; k: int; batch: int; flops: float; bytes: float
    pred_compute_us: float; pred_memory_us: float; pred_roof_us: float
    bound: str; measured_us: Optional[float]; error_pct: Optional[float]
    comparable: bool = True; skipped: str = ""; notes: str = ""


def event_time(run: Callable[[], None], warmup: int, iters: int, min_ms: float,
               samples: int) -> float:
    for _ in range(max(1, warmup)):
        run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); run(); end.record(); end.synchronize()
    n = max(1, iters, math.ceil(min_ms / max(start.elapsed_time(end), .01)))
    latencies = []
    for _ in range(samples):
        start.record()
        for _ in range(n):
            run()
        end.record(); end.synchronize()
        latencies.append(start.elapsed_time(end) / 1000.0 / n)
    return statistics.median(latencies)


def torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}[name]


def bmm_time(batch: int, m: int, n: int, k: int, dtype: str, args) -> float:
    a = torch.randn(batch, m, n, device="cuda", dtype=torch_dtype(dtype))
    b = torch.randn(batch, n, k, device="cuda", dtype=torch_dtype(dtype))
    out = torch.empty(batch, m, k, device="cuda", dtype=torch_dtype(dtype))
    return event_time(lambda: torch.bmm(a, b, out=out), args.warmup, args.iters,
                      args.min_ms, args.samples)


def gemm_time(m: int, n: int, k: int, dtype: str, fp8_backend: str, native, args) -> tuple[float, str]:
    if dtype == "float8":
        if fp8_backend == "triton":
            return time_triton_fp8(m, n, k, args.warmup, args.iters), "triton_fp8"
        if fp8_backend == "hipblaslt":
            return time_native_gemm(native, m, n, k, args.warmup, args.iters), "hipblaslt_fp8"
        # The current BW1100 float8 efficiency curve was calibrated with
        # per-shape auto dispatch.  Validate it with the same policy; a fixed
        # backend is deliberately available for workload-specific studies.
        candidates = [
            (time_triton_fp8(m, n, k, args.warmup, args.iters), "triton_fp8"),
            (time_native_gemm(native, m, n, k, args.warmup, args.iters), "hipblaslt_fp8"),
        ]
        return min(candidates, key=lambda item: item[0])
    a = torch.randn(m, k, device="cuda", dtype=torch_dtype(dtype))
    b = torch.randn(k, n, device="cuda", dtype=torch_dtype(dtype))
    out = torch.empty(m, n, device="cuda", dtype=torch_dtype(dtype))
    return event_time(lambda: torch.mm(a, b, out=out), args.warmup, args.iters,
                      args.min_ms, args.samples), f"torch_mm_{dtype}"


def vector_time(kind: str, elements: int, hidden: int, heads: int, seq: int, args) -> float:
    if kind == "norm":
        x = torch.randn(max(1, elements // hidden), hidden, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
        return event_time(lambda: torch.nn.functional.rms_norm(x, (hidden,), w), args.warmup, args.iters, args.min_ms, args.samples)
    if kind == "softmax":
        batch = max(1, elements // max(1, heads * seq * seq))
        x = torch.randn(batch, heads, seq, seq, device="cuda", dtype=torch.float32)
        return event_time(lambda: torch.softmax(x, dim=-1), args.warmup, args.iters, args.min_ms, args.samples)
    x = torch.randn(max(1, elements), device="cuda", dtype=torch.bfloat16)
    return event_time(lambda: torch.nn.functional.silu(x), args.warmup, args.iters, args.min_ms, args.samples)


def sigmoid_time(elements: int, args) -> float:
    x = torch.randn(max(1, elements), device="cuda", dtype=torch.bfloat16)
    return event_time(lambda: torch.sigmoid(x), args.warmup, args.iters,
                      args.min_ms, args.samples)


def moe_expert_time(r, app, dtype: str, fp8_backend: str, native, args) -> tuple[float, str, str]:
    """Return a routing-faithful serial expert baseline for one MoE Linear.

    DeepSeek-V3 sends ``topk`` copies of each token across the routed experts,
    while every token also visits each shared expert.  The catalog represents
    this with flop_multiplier=topk+shared and weight_multiplier=experts+shared.
    Benchmarking one full-M expert is therefore not comparable.  Measure the
    two representative shapes and sum their latency multiplicities:

      routed: num_experts × GEMM(ceil(M*topk/num_experts), N, K)
      shared: num_shared_experts × GEMM(M, N, K)

    This is deliberately a serial/grouped-GEMM baseline.  A future production
    grouped kernel can replace it without changing the workload accounting.
    """
    if r.stage != "fw":
        raise RuntimeError("routing-faithful MoE measurement currently supports fw only")
    m, n, k = int(r.batch_seq), int(r.c_out), int(r.c_in)
    routed_tokens = math.ceil(m * app.moe_topk / app.num_experts)
    routed_lat, routed_backend = gemm_time(
        routed_tokens, n, k, dtype, fp8_backend, native, args)
    shared_lat = 0.0
    shared_backend = "none"
    if app.num_shared_experts:
        one_shared, shared_backend = gemm_time(
            m, n, k, dtype, fp8_backend, native, args)
        shared_lat = app.num_shared_experts * one_shared
    total = app.num_experts * routed_lat + shared_lat
    note = (f"routing-faithful serial baseline: {app.num_experts}xM={routed_tokens} "
            f"routed ({routed_backend}) + {app.num_shared_experts}xM={m} "
            f"shared ({shared_backend}); assignments="
            f"{app.num_experts*routed_tokens + app.num_shared_experts*m}")
    return total, "moe_expert_serial", note


def error(pred: float, measured: float) -> float:
    return 100.0 * (pred - measured) / measured


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "models" / "deepseek-v3-671b.json")
    p.add_argument("--system", type=Path, default=ROOT / "systems" / "BW1100.json")
    p.add_argument("--matrix-dtype", choices=("float8", "int8", "float16", "bfloat16", "float32"), default="float8")
    p.add_argument("--vector-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    p.add_argument("--bmm-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    p.add_argument("--fp8-backend", choices=("auto", "triton", "hipblaslt"), default="auto",
                   help="must match the Phase-0 FP8 curve policy; auto is the BW1100.json default")
    p.add_argument("--seq-size", type=int); p.add_argument("--microbatch-size", type=int, default=1)
    p.add_argument("--expert-par", type=int, default=1); p.add_argument("--groups", nargs="+", default=["G1", "G2", "G3", "G5", "G6"])
    p.add_argument("--blocks", nargs="+", choices=("dense", "moe"), default=["dense", "moe"])
    p.add_argument("--stage", choices=("fw", "agrad", "wgrad"), default="fw")
    p.add_argument("--warmup", type=int, default=10); p.add_argument("--iters", type=int, default=50)
    p.add_argument("--min-ms", type=float, default=100.0); p.add_argument("--samples", type=int, default=3)
    p.add_argument("--predict-only", action="store_true"); p.add_argument("--csv", type=Path)
    p.add_argument("--device", type=int, default=0,
                   help="HIP device index (default 0)")
    p.add_argument("--names", nargs="*", default=None,
                   help="Optional exact/substr layer-name filter for quick tests")
    args = p.parse_args()
    if args.samples < 1: p.error("--samples must be positive")
    groups = {g.upper() for g in args.groups}
    llm, app, syst, exe = catalog.compile_dsv3(str(args.model), str(args.system), args.matrix_dtype, args.vector_dtype, args.seq_size, args.microbatch_size, args.expert_par)
    rows0 = catalog.build_catalog(llm, app, exe, syst, stages=[args.stage], blocks=args.blocks)
    selected = [r for r in rows0 if r.group in groups]
    if args.names:
        selected = [r for r in selected
                    if any(x == r.name or x in r.name for x in args.names)]
    native = None
    if not args.predict_only:
        isa = require_bw1100(); torch.cuda.set_device(args.device)
        if args.matrix_dtype == "float8" and args.fp8_backend in ("auto", "hipblaslt"):
            native = build_native_backend("float8", isa)
        print(f"Device={torch.cuda.get_device_name(args.device)} ({isa}); selected={len(selected)} ops", flush=True)
    out: list[Row] = []
    for i, r in enumerate(selected, 1):
        measured = None; kernel = "assert"; skipped = ""; notes = r.notes
        comparable = True
        try:
            if r.group == "G6":
                notes = (notes + "; " if notes else "") + ("PASS fw_flops==0" if r.flops == 0 else "FAIL fw_flops!=0")
            elif r.cls == "SoftMax" and r.flops == 0:
                notes = "PASS fused SoftMax: intentionally not isolated"
            elif r.cls in ("SiLU", "GeLU") and r.flops == 0:
                notes = "PASS fused activation: intentionally not isolated"
            elif (not args.predict_only and r.group == "G4" and
                  r.cls == "Linear" and "MlpBlock_MoE_" in r.name and
                  r.c_in and r.c_out):
                measured, kernel, moe_note = moe_expert_time(
                    r, app, args.matrix_dtype, args.fp8_backend, native, args)
                notes = (notes + "; " if notes else "") + moe_note
                # This is a serial-kernel diagnostic.  Catalog predicts an
                # ideal aggregate/grouped MoE operator, so comparing their
                # latencies as one MAPE sample is dimensionally invalid.
                comparable = False
                notes += "; excluded_from_mape: serial experts vs aggregate grouped-op prediction"
            elif not args.predict_only and r.cls == "Linear" and r.c_in and r.c_out:
                m, n, k = r.batch_seq, r.c_out, r.c_in
                if args.stage == "agrad": m, n, k = r.batch_seq, r.c_in, r.c_out
                elif args.stage == "wgrad": m, n, k = r.c_in, r.c_out, r.batch_seq
                measured, kernel = gemm_time(m, n, k, args.matrix_dtype, args.fp8_backend, native, args)
            elif not args.predict_only and r.cls == "BatchMatMul" and r.bmm_batch:
                measured = bmm_time(r.bmm_batch, r.bmm_m, r.bmm_n, r.bmm_k, args.bmm_dtype, args); kernel = f"torch_bmm_{args.bmm_dtype}"
                notes = (notes + "; " if notes else "") + "BF16/FP16 BMM baseline; no native FP8 BMM claim"
            elif not args.predict_only and r.cls == "RouterSigmoid":
                # Catalog currently does not expose act_size for router helper
                # classes; RouterSigmoid charges four FLOPs per score.
                elements = max(1, int(r.flops // 4))
                measured = sigmoid_time(elements, args); kernel = "sigmoid"
                notes = (notes + "; " if notes else "") + f"router scores={elements}"
            elif not args.predict_only and r.cls in ("LayerNorm", "RMSNorm", "SoftMax", "SiLU", "GeLU") and r.act_size:
                kind = "norm" if r.cls in ("LayerNorm", "RMSNorm") else "softmax" if r.cls == "SoftMax" else "silu"
                # Norm reduction width is layer-specific: main RMSNorm=hidden,
                # QNorm=q_lora_rank, KVNorm=kv_lora_rank.  act_size/batch_seq
                # recovers all three directly from the compiled layer.
                reduction = (max(1, r.act_size // r.batch_seq)
                             if kind == "norm" and r.batch_seq else app.hidden)
                measured = vector_time(kind, r.act_size, reduction, app.attn_heads, app.seq_size, args); kernel = kind
                if kind == "norm":
                    notes = (notes + "; " if notes else "") + f"reduction_width={reduction}"
            else: skipped = "predict-only" if args.predict_only else "unmapped_or_empty"
        except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError) as exc:
            torch.cuda.empty_cache(); skipped = f"{type(exc).__name__}: {str(exc)[:120]}"
        err = None if measured is None or not comparable else error(r.pred_max_s, measured)
        out.append(Row(r.group, r.block, r.name, r.cls, r.stage, kernel, r.batch_seq or r.bmm_m, r.c_out or r.bmm_k, r.c_in or r.bmm_n, r.bmm_batch, r.flops, r.bytes, r.pred_f_s*1e6, r.pred_m_s*1e6, r.pred_max_s*1e6, r.bound, None if measured is None else measured*1e6, err, comparable, skipped, notes))
        print(f"[{i:2d}/{len(selected)}] {r.group} {r.name:34s} pred={r.pred_max_s*1e6:9.2f}us meas={'-' if measured is None else f'{measured*1e6:.2f}'} {kernel} {skipped}", flush=True)
    valid = [abs(r.error_pct) for r in out if r.error_pct is not None]
    print(f"Phase2 measured={len(valid)}/{len(out)} MAPE={sum(valid)/len(valid):.2f}% max={max(valid):.2f}%" if valid else "Phase2: no measured rows")
    for group in sorted(groups):
        vals = [abs(r.error_pct) for r in out if r.group == group and r.error_pct is not None]
        if vals: print(f"  {group}: N={len(vals)} MAPE={sum(vals)/len(vals):.2f}% max={max(vals):.2f}%")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(out[0]).keys())); w.writeheader(); w.writerows(asdict(r) for r in out)
        print(f"wrote {args.csv}")


if __name__ == "__main__": main()
