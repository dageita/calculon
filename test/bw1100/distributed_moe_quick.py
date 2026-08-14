#!/usr/bin/env python3
"""Physical 2-4 GPU BW1100 distributed quick validation.

Run with torchrun. Measures TP all-reduce, EP all-to-all, CP all-gather,
PP point-to-point microbatch pipeline, and async communication/compute overlap.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


def event_ms(fn, warmup, iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        dist.barrier()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); e.synchronize()
        vals.append(s.elapsed_time(e))
    return statistics.median(vals)


def global_max(value, device):
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tensor-mb', type=float, default=32.0)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--microbatches', type=int, default=8)
    p.add_argument('--compute-m', type=int, default=2048)
    p.add_argument('--gate-pct', type=float, default=25.0)
    p.add_argument('--output', type=Path,
                   default=Path('test/bw1100/distributed_quick.json'))
    p.add_argument('--system', type=Path, default=Path('systems/BW1100.json'))
    p.add_argument('--network-index', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.dry_run:
        print(json.dumps({'required_world_size': '2-4',
                          'tests': ['TP/all_reduce', 'EP/all_to_all_single',
                                    'CP/all_gather', 'PP/send_recv_pipeline',
                                    'async_overlap'],
                          'tensor_mb': args.tensor_mb,
                          'microbatches': args.microbatches}, indent=2))
        return

    backend = 'nccl' if dist.is_nccl_available() else 'gloo'
    dist.init_process_group(backend=backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    if not 2 <= world <= 4:
        raise SystemExit(f'distributed quick requires 2-4 ranks, got {world}')
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device('cuda', local_rank)
    elem = max(world, int(args.tensor_mb * 1024**2 / 2))
    elem -= elem % world
    x = torch.ones(elem, device=dev, dtype=torch.bfloat16)
    chunk = elem // world
    a2a_out = torch.empty_like(x)
    gather_out = torch.empty(elem * world, device=dev, dtype=x.dtype)

    def tp():
        y = x.clone(); dist.all_reduce(y)
    def ep():
        dist.all_to_all_single(a2a_out, x)
    def cp():
        dist.all_gather_into_tensor(gather_out, x)

    tp_ms = global_max(event_ms(tp, args.warmup, args.iters), dev)
    ep_ms = global_max(event_ms(ep, args.warmup, args.iters), dev)
    cp_ms = global_max(event_ms(cp, args.warmup, args.iters), dev)

    cm = args.compute_m
    ma = torch.randn(cm, cm, device=dev, dtype=torch.bfloat16)
    mb = torch.randn(cm, cm, device=dev, dtype=torch.bfloat16)
    def compute(): torch.mm(ma, mb)
    compute_ms = global_max(event_ms(compute, 2, args.iters), dev)

    def serial():
        y = x.clone(); dist.all_reduce(y); compute()
    def overlap():
        y = x.clone(); work = dist.all_reduce(y, async_op=True)
        compute(); work.wait()
    serial_ms = global_max(event_ms(serial, 2, args.iters), dev)
    overlap_ms = global_max(event_ms(overlap, 2, args.iters), dev)
    ideal_ms = max(tp_ms, compute_ms)
    overlap_overhead_pct = 100.0 * (overlap_ms - ideal_ms) / ideal_ms
    overlap_gain_pct = 100.0 * (serial_ms - overlap_ms) / serial_ms

    cfg=json.load(open(args.system)); net=cfg['networks'][args.network_index]
    bw=float(net['bandwidth'])*1e9*float(net['efficiency'])
    latency=float(net['latency'])
    nbytes=elem*2
    def net_pred(op):
        scalar,offset=net['ops'][op]
        scaled=nbytes*float(scalar)
        scaled += scaled/world*float(offset or 0)
        return (latency+scaled/bw)*1e3
    tp_pred=net_pred('all_reduce'); cp_pred=net_pred('all_gather')
    # Network config has no all_to_all primitive. EP comparison uses p2p as
    # an explicit approximation and is labelled accordingly.
    ep_pred=(latency+nbytes*float(net['ops']['p2p'][0])/bw)*1e3

    token = torch.empty(max(1024, chunk // 8), device=dev,
                        dtype=torch.bfloat16)
    def pp_once():
        for _ in range(args.microbatches):
            if rank > 0: dist.recv(token, rank - 1)
            compute()
            if rank + 1 < world: dist.send(token, rank + 1)
    pp_ms = global_max(event_ms(pp_once, 1, max(2, args.iters // 2)), dev)
    bubble_formula = (world - 1) / (args.microbatches + world - 1)
    # This blocking micro-pipeline includes fill/drain and is intentionally a
    # conservative physical sanity check, not a production 1F1B scheduler.
    pp_ideal_ms = args.microbatches * compute_ms
    pp_excess_pct = 100.0 * (pp_ms - pp_ideal_ms) / pp_ideal_ms

    results = {
        'world_size': world, 'backend': backend,
        'tensor_mb': args.tensor_mb,
        'tp_all_reduce_ms': tp_ms,
        'ep_all_to_all_ms': ep_ms,
        'cp_all_gather_ms': cp_ms,
        'tp_model_ms':tp_pred,'tp_model_error_pct':100*(tp_pred-tp_ms)/tp_ms,
        'cp_model_ms':cp_pred,'cp_model_error_pct':100*(cp_pred-cp_ms)/cp_ms,
        'ep_model_p2p_approx_ms':ep_pred,
        'ep_model_p2p_approx_error_pct':100*(ep_pred-ep_ms)/ep_ms,
        'compute_ms': compute_ms,
        'serial_comm_compute_ms': serial_ms,
        'overlap_ms': overlap_ms,
        'overlap_gain_pct': overlap_gain_pct,
        'overlap_overhead_vs_ideal_pct': overlap_overhead_pct,
        'pp_pipeline_ms': pp_ms,
        'pp_excess_vs_compute_pct': pp_excess_pct,
        'pp_analytic_bubble_fraction': bubble_formula,
        'checks': {
            'tp_positive': tp_ms > 0,
            'ep_positive': ep_ms > 0,
            'cp_positive': cp_ms > 0,
            'overlap_not_worse_than_serial': overlap_ms <= serial_ms * 1.05,
            'overlap_near_ideal': overlap_overhead_pct <= args.gate_pct,
            'pp_positive': pp_ms > 0,
        },
    }
    results['pass'] = all(results['checks'].values())
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + '.tmp')
        tmp.write_text(json.dumps(results, indent=2) + '\n')
        os.replace(tmp, args.output)
        print(json.dumps(results, indent=2), flush=True)
    dist.barrier(); dist.destroy_process_group()
    raise SystemExit(0 if results['pass'] else 1)


if __name__ == '__main__':
    main()
