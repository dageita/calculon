#!/usr/bin/env python3
"""Calibrate generic PyTorch/Megatron runtime scheduling costs.

The output describes a GPU + software-stack capability.  It does not read a
model, parallel configuration, trace, sequence length, or training result.

Run this after upgrading PyTorch, CUDA, Transformer Engine or Megatron.  The
values are deliberately separate from matrix/vector throughput calibration.
"""
from __future__ import annotations
import argparse
import json
import statistics
import time

import torch
from torch.utils.checkpoint import checkpoint
from torch.profiler import ProfilerActivity, profile


def median(values):
    return statistics.median(values)


def timed(fn, repeats):
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - begin)
    return median(samples)
def cuda_kernel_count(fn):
    """Count CUDA launches after warm-up without inspecting a model trace."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as events:
        fn()
    torch.cuda.synchronize()
    return sum(event.device_type == torch.autograd.DeviceType.CUDA
               for event in events.events())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--launches", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--checkpoint-nodes", type=int, default=64)
    parser.add_argument("--norm-backend",
                        choices=("torch", "transformer_engine"), default="torch",
                        help="Megatron norm backend on this software stack; TE transformer layers may still use Torch Norm when Apex is absent")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device(args.device)
    x = torch.zeros(1024, device=device)
    for _ in range(100):
        x.add_(1)
    torch.cuda.synchronize()

    # CPU queueing cost, not elapsed GPU execution: the stream is drained only
    # after recording the host submission timestamp.
    issue_samples = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(args.launches):
            x.add_(1)
        issue_samples.append((time.perf_counter() - begin) / args.launches)
        torch.cuda.synchronize()
    kernel_dispatch_s = median(issue_samples)

    scalar_samples = []
    for _ in range(args.repeats):
        x.add_(1)
        begin = time.perf_counter()
        x[0].item()
        scalar_samples.append(time.perf_counter() - begin)
    scalar_sync_s = median(scalar_samples)

    # Compare the same trivial graph with and without checkpoint.  The forward
    # recompute itself is excluded by subtracting the no-checkpoint graph; this
    # leaves Python/autograd checkpoint bookkeeping and scheduling.
    def run_graph(use_checkpoint):
        value = torch.ones(1024, device=device, requires_grad=True)
        for _ in range(args.checkpoint_nodes):
            if use_checkpoint:
                value = checkpoint(lambda y: y * 1.0001, value,
                                   use_reentrant=True)
            else:
                value = value * 1.0001
        value.sum().backward()

    base = timed(lambda: run_graph(False), args.repeats)
    ckpt = timed(lambda: run_graph(True), args.repeats)
    checkpoint_node_s = max(0.0, (ckpt - base) / args.checkpoint_nodes)

    # A chain of autograd nodes has identical GPU kernels in both measurements;
    # use its excess wall time per node as conservative Python/autograd dispatch.
    one = timed(lambda: (torch.ones(1024, device=device, requires_grad=True)
                         * 1.0001).sum().backward(), args.repeats)
    chain = timed(lambda: run_graph(False), args.repeats)
    autograd_node_s = max(0.0, (chain - one) / (args.checkpoint_nodes - 1))

    # Transformer Engine internally expands a logical module into several CUDA
    # launches.  Profile representative framework operators only; no user model
    # graph, shape contract, or measured iteration participates in this value.
    operator_events = {}
    try:
        import transformer_engine.pytorch as te

        def probe(module):
            value = torch.randn(1024, 1024, device=device, requires_grad=True)
            for _ in range(3):
                output = module(value)
                output = output[0] if isinstance(output, tuple) else output
                torch.autograd.backward(output, torch.ones_like(output))
                module.zero_grad(set_to_none=True); value.grad = None
            forward = cuda_kernel_count(lambda: module(value))
            output = module(value); output = output[0] if isinstance(output, tuple) else output
            grad = torch.ones_like(output)
            backward = cuda_kernel_count(lambda: torch.autograd.backward(output, grad))
            fw_host, bw_host = [], []
            for _ in range(args.repeats):
                value = torch.randn(1024, 1024, device=device, requires_grad=True)
                torch.cuda.synchronize(); begin = time.perf_counter()
                output = module(value)
                fw_host.append(time.perf_counter() - begin)
                output = output[0] if isinstance(output, tuple) else output
                grad = torch.ones_like(output); torch.cuda.synchronize()
                begin = time.perf_counter(); torch.autograd.backward(output, grad)
                bw_host.append(time.perf_counter() - begin)
                torch.cuda.synchronize(); module.zero_grad(set_to_none=True)
            return forward, backward, median(fw_host), median(bw_host)

        linear_fw, linear_bw, linear_fw_host, linear_bw_host = probe(te.Linear(1024, 1024, device=device))
        rms_fw, rms_bw, rms_fw_host, rms_bw_host = probe(te.RMSNorm(1024, device=device))
        operator_events = {
            "Linear": {
                "fw": {"kernels": linear_fw, "host_s": linear_fw_host},
                "agrad": {"kernels": 1, "host_s": linear_bw_host / linear_bw},
                "wgrad": {"kernels": max(0, linear_bw - 1), "host_s": linear_bw_host * (linear_bw - 1) / linear_bw}},
            "RMSNorm": {
                "fw": {"kernels": rms_fw, "host_s": rms_fw_host},
                "agrad": {"kernels": 1, "host_s": rms_bw_host / rms_bw},
                "wgrad": {"kernels": max(0, rms_bw - 1), "host_s": rms_bw_host * (rms_bw - 1) / rms_bw}},
        }
    except (ImportError, AttributeError):
        pass

    result = {
        "framework_runtime": {
            "norm_backend": args.norm_backend,
            "enabled": True,
            "kernel_dispatch_s": kernel_dispatch_s,
            "autograd_node_s": autograd_node_s,
            "checkpoint_node_s": checkpoint_node_s,
            "scalar_sync_s": scalar_sync_s,
            # Megatron fused Adam groups tensors.  This is a generic bucket
            # capacity, not a parameter-count-specific number.
            "optimizer_bucket_bytes": 134217728,
            "optimizer_bucket_dispatch_s": kernel_dispatch_s,
            "operator_events": operator_events,
        },
        "metadata": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "method": "standalone runtime microbench; no model/profile input",
        },
    }
    blob = json.dumps(result, indent=2)
    if args.output:
        open(args.output, "w").write(blob + "\n")
    print(blob)


if __name__ == "__main__":
    main()
