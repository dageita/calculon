#!/usr/bin/env python3
"""Small-memory MoE communication validation for a fitted RCCL model."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


def measure_ms(fn, warmup, iters, dev):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    values = []
    for _ in range(iters):
        dist.barrier()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record(); fn(); end.record(); end.synchronize()
        values.append(start.elapsed_time(end))
    result = torch.tensor([statistics.median(values)], device=dev,
                          dtype=torch.float64)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result.item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', type=Path,
                   default=Path('test/bw1100/phase4_rccl_latency_bw.json'))
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--hidden', type=int, default=7168)
    p.add_argument('--skew', type=float, default=0.5,
                   help='Fraction of every rank routed to rank 0')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--gate-pct', type=float, default=30.0)
    p.add_argument('--output', type=Path,
                   default=Path('test/bw1100/phase5_moe_comm_quick.json'))
    args = p.parse_args()
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    if not 2 <= world <= 4:
        raise SystemExit(f'requires 2-4 ranks, got {world}')
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device('cuda', local_rank)
    dtype = torch.bfloat16
    x = torch.randn(args.tokens, args.hidden, device=dev, dtype=dtype)

    equal = [args.tokens // world] * world
    for i in range(args.tokens % world):
        equal[i] += 1
    hot = min(args.tokens, max(0, round(args.tokens * args.skew)))
    skewed = [(args.tokens - hot) // max(1, world - 1)] * world
    skewed[0] = hot
    remainder = args.tokens - sum(skewed)
    for i in range(1, 1 + remainder):
        skewed[i % world] += 1

    def recv_splits(send_splits):
        local = torch.tensor(send_splits, device=dev, dtype=torch.int64)
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        return [int(gathered[src][rank].item()) for src in range(world)]

    def make_case(send):
        recv = recv_splits(send)
        output = torch.empty(sum(recv), args.hidden, device=dev, dtype=dtype)
        def exchange():
            dist.all_to_all_single(output, x, recv, send)
        return exchange, output

    equal_exchange, equal_out = make_case(equal)
    skew_exchange, skew_out = make_case(skewed)
    equal_ms = measure_ms(equal_exchange, args.warmup, args.iters, dev)
    skew_ms = measure_ms(skew_exchange, args.warmup, args.iters, dev)
    def two_phase_exchange():
        equal_exchange(); equal_exchange()
    two_phase_ms = measure_ms(two_phase_exchange, args.warmup, args.iters, dev)

    # Model dispatch + local route ordering + expert-like vector work + combine.
    expert_ids = torch.randint(0, max(8, world * 4), (args.tokens,), device=dev)
    def route_sort():
        torch.argsort(expert_ids)
    route_ms = measure_ms(route_sort, args.warmup, args.iters, dev)
    weight = torch.randn(args.hidden, device=dev, dtype=dtype)
    def expert_compute():
        torch.mul(x, weight)
    compute_ms = measure_ms(expert_compute, args.warmup, args.iters, dev)
    def serial_step():
        route_sort(); equal_exchange(); expert_compute(); equal_exchange()
    def overlap_step():
        route_sort()
        work = dist.all_to_all_single(equal_out, x, equal, equal, async_op=True)
        expert_compute(); work.wait(); equal_exchange()
    serial_ms = measure_ms(serial_step, args.warmup, args.iters, dev)
    overlap_ms = measure_ms(overlap_step, args.warmup, args.iters, dev)

    calibration = json.loads(args.calibration.read_text())
    fit = calibration.get('physical_link_fit')
    if fit is None:
        # Compatibility with pre-physical-model calibration files.
        fit = (calibration.get('flow_group_fits') or
               calibration.get('fits'))['ep']
    payload = x.numel() * x.element_size()
    # This is an end-to-end RCCL launch baseline, paid once per collective.
    # C++ accounts for participant scaling through Flow volume and contention.
    one_phase_ms = (fit['latency_s'] +
                    payload / fit['bandwidth_Bps']) * 1e3
    predicted_two_phase_ms = 2 * one_phase_ms
    error = 100.0 * (predicted_two_phase_ms - two_phase_ms) / two_phase_ms
    result = {
        'world_size': world,
        'tokens_per_rank': args.tokens,
        'hidden': args.hidden,
        'payload_bytes_per_rank': payload,
        'equal_dispatch_ms': equal_ms,
        'skewed_dispatch_ms': skew_ms,
        'skew_penalty_pct': 100.0 * (skew_ms - equal_ms) / equal_ms,
        'two_phase_communication_ms': two_phase_ms,
        'route_sort_ms': route_ms,
        'expert_compute_ms': compute_ms,
        'serial_moe_step_ms': serial_ms,
        'overlap_moe_step_ms': overlap_ms,
        'overlap_gain_pct': 100.0 * (serial_ms - overlap_ms) / serial_ms,
        'calibrated_two_phase_model_ms': predicted_two_phase_ms,
        'calibrated_two_phase_model_error_pct': error,
        'checks': {
            'communication_positive': equal_ms > 0 and skew_ms > 0,
            'model_within_gate': abs(error) <= args.gate_pct,
            'overlap_not_slower': overlap_ms <= serial_ms * 1.05,
        },
    }
    result['pass'] = all(result['checks'].values())
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + '.tmp')
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        os.replace(tmp, args.output)
        print(json.dumps(result, indent=2), flush=True)
    dist.barrier(); dist.destroy_process_group()
    raise SystemExit(0 if result['pass'] else 1)


if __name__ == '__main__':
    main()
