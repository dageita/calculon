#!/usr/bin/env python3
"""Run a fixed BW1100 DeepSeek-V3 supernode scaling sweep via /calculate.

The default ladder is deliberately not an optimiser: it fixes TP=4, PP=2,
DP=CP=1 and doubles EP from 1 to 16.  This makes the 8→128-card comparison
auditable: each point is the same model and batch, while expert placement is
the only scaling axis.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HEADERS = [
    "Model", "Network", "datatype", "TP", "PP", "DP", "EP", "CP",
    "Batch Size", "Microbatch Size", "Activation recompute",
    "Optimizer sharding", "Batch Time(s)", "Comm Time(s)",
    "Per-batch EP communication time(s)", "Comm Ratio", "Memory(GiB)",
    "MFU", "Linear Scaling Throughput (samples/s)",
]


def number(obj, *names):
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else None
        if value is not None:
            return value
    return None


def gib(value):
    """Normalise API memory strings (for example ``1.101 TiB``) to GiB."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        amount, unit = value.strip().split()[:2]
        scale = {"TiB": 1024.0, "GiB": 1.0, "MiB": 1.0 / 1024.0,
                 "KiB": 1.0 / (1024.0 ** 2), "B": 1.0 / (1024.0 ** 3)}
        return float(amount) * scale[unit]
    except (ValueError, KeyError):
        return None


def request_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"},
                      method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_payload(num_procs, ep, args):
    return {
        "gpu": {
            "name": "BW1100",
            "sparse_tensor_fp16_processing_power": 354,
            "sparse_tensor_fp32_processing_power": 44,
            "memory": 144,
            "memory_bandwidth": 2400,
            "bus_bandwidth": 158.51514702896458,
            "network_bandwidth": 25,
            "support_p2p": True,
            "num_procs": num_procs,
        },
        "network": {"network_bandwidth": 25,
                    "network_topology": "Single machine"},
        "model": {
            "name": "DeepSeek-V3 671B", "seq_size": 4096,
            "hidden": 7168, "feedforward": 18432, "attn_heads": 128,
            "kv_heads": None, "attn_size": 128, "rope_theta": None,
            "rms_norm": None, "qk_norm": None, "ffn_type": None,
            "untied_embeddings": None, "num_blocks": 61,
            "vocab_size": 129280, "num_experts": 256, "moe_topk": 8,
            "norm_topk_prob": None, "router_aux_loss_coef": None,
            "num_shared_experts": 1, "moe_feedforward": 2048,
            "first_k_dense": 3, "moe_layer_freq": 1, "kv_size": 576,
            "q_lora_rank": 1536, "kv_lora_rank": 512,
            "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        },
        "trainning_config": {
            "optimization_strategy": "Full recomputation",
            "activation_recompute": "full", "optimizer_sharding": False,
            "tensor_par": args.tp, "pipeline_par": args.pp,
            "data_par": 1, "expert_par": ep, "context_par": 1,
            "batch_size": args.batch_size,
            "microbatch_size": args.microbatch_size,
            "matrix_dtype": args.matrix_dtype,
            "vector_dtype": args.vector_dtype,
        },
    }


def row_from_result(payload, result):
    cfg = payload["trainning_config"]
    gpu, network, model = payload["gpu"], payload["network"], payload["model"]
    summary = result.get("summary", {})
    memory = result.get("memory_usage", {})
    comm = result.get("communication", {})
    batch_time = number(summary, "batch_total_time")
    comm_time = number(comm, "total_comm_time")
    return {
        "Model": model["name"], "Network": network["network_topology"],
        "datatype": f'{cfg["matrix_dtype"]}/{cfg["vector_dtype"]}',
        "TP": cfg["tensor_par"], "PP": cfg["pipeline_par"],
        "DP": cfg["data_par"], "EP": cfg["expert_par"],
        "CP": cfg["context_par"], "Batch Size": cfg["batch_size"],
        "Microbatch Size": cfg["microbatch_size"],
        "Activation recompute": cfg["activation_recompute"],
        "Optimizer sharding": cfg["optimizer_sharding"],
        "Batch Time(s)": batch_time,
        "Comm Time(s)": comm_time,
        "Per-batch EP communication time(s)": number(
            comm, "batch_ep_comm_time", "batch_ep_comm"),
        "Comm Ratio": (comm_time / batch_time
                       if isinstance(batch_time, (int, float)) and batch_time else None),
        "Memory(GiB)": gib(number(memory, "overall_usage")),
        "MFU": number(summary, "total_efficiency"),
        "Linear Scaling Throughput (samples/s)": number(
            summary, "linear_scaling_throughput"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=(
        "http://127.0.0.1:8000/llm_training_calculator/calculator/calculate"))
    parser.add_argument("--steps", default="8,16,32,64,128")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--matrix-dtype", default="float8")
    parser.add_argument("--vector-dtype", default="float8")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path,
                        default=Path("test/bw1100/supernode_scaling_results.json"))
    parser.add_argument("--csv", type=Path,
                        default=Path("test/bw1100/supernode_scaling_results.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    if not steps or sorted(steps) != steps or len(set(steps)) != len(steps):
        parser.error("--steps must be unique ascending positive integers")

    records = []
    base = args.tp * args.pp
    for num_procs in steps:
        if num_procs % base:
            parser.error(f"num_procs={num_procs} not divisible by TP*PP={base}")
        ep = num_procs // base
        if ep > 256 or 256 % ep:
            parser.error(f"EP={ep} must divide DeepSeek-V3's 256 routed experts")
        payload = build_payload(num_procs, ep, args)
        assert (args.tp * args.pp * ep) == num_procs
        if args.dry_run:
            print(json.dumps(payload, indent=2))
            continue
        print(f"running num_procs={num_procs}: TP={args.tp} PP={args.pp} DP=1 EP={ep} CP=1", flush=True)
        started = time.time()
        try:
            result = request_json(args.url, payload, args.timeout)
            row = row_from_result(payload, result)
            records.append({"num_procs": num_procs, "payload": payload,
                            "response": result, "row": row,
                            "elapsed_s": time.time() - started, "status": "ok"})
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            records.append({"num_procs": num_procs, "payload": payload,
                            "error": str(exc), "elapsed_s": time.time() - started,
                            "status": "error"})
            print(f"num_procs={num_procs} failed: {exc}", file=sys.stderr, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "design": {
            "family": "fixed TP=4, PP=2, DP=1, CP=1; EP doubles with scale",
            "reason": "strong-scaling ladder; every point keeps model, precision, "
                      "batch and recompute policy fixed",
            "steps": steps,
        },
        "records": records,
    }, indent=2) + "\n")
    rows = [r["row"] for r in records if r["status"] == "ok"]
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {args.output} and {args.csv}; successful={len(rows)}/{len(steps)}")
    if not rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
