#!/usr/bin/env python3
"""Calibrate BW1100 RCCL collectives against LLMFlowSimulator's BW-only model.

Run under torchrun with 2-4 ranks.  PyTorch calls the backend ``nccl`` on
ROCm/DTK, but the implementation is RCCL.  EP is measured with
all_to_all_single.  CP is measured as the same ring send/recv used by the C++
flow simulator; all-gather is retained only as a diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


def measure_ms(fn, warmup: int, iters: int, dev: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        dist.barrier()
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        vals.append(begin.elapsed_time(end))
    local = torch.tensor([statistics.median(vals)], dtype=torch.float64,
                         device=dev)
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return float(local.item())


def flow_reference_seconds(net_cfg: dict, op: str, world: int,
                           nbytes: int, reference_bw_gbps: float) -> float:
    """Run one isolated C++ flow and merge its timeline intervals."""
    from calculon.network import Network

    net = Network(copy.deepcopy(net_cfg))
    bw = reference_bw_gbps * 1e9
    net.flow_network_init(bw, bw, 'Single machine')
    kw = dict(
        pp=1, dp=1, tp=1, ep=1, cp=1,
        fwdCompTime=1e-9, bwdCompTime=1e-9,
        # Two-phase EP events are part of the layered MLA/FFN state machine.
        # Tiny non-zero compute durations select the real DSV3 path without
        # materially changing the communication interval being calibrated.
        fwd_mla_time=1e-9, fwd_ffn_time=1e-9,
        bwd_mla_time=1e-9, bwd_ffn_time=1e-9,
        microbatches=1,
        fwdTPSize=0, bwdTPSize=0, fwdPPSize=0, bwdPPSize=0, dpSize=0,
        fwd_ep_size=0, bwd_ep_size=0,
        fwd_ep_dispatch_size=0, fwd_ep_combine_size=0,
        bwd_ep_dispatch_size=0, bwd_ep_combine_size=0,
        fwd_cp_size=0, bwd_cp_size=0, enable_timeline=True)
    if op == 'ep':
        kw.update(ep=world, fwd_ep_dispatch_size=nbytes)
    elif op == 'cp':
        kw.update(cp=world, fwd_cp_size=nbytes)
    else:
        raise ValueError(op)
    result = net.total_flow_network_time(**kw)
    # Current summary counters can miss a minimal synthetic EP/CP workload.
    # Timeline out-parameters are authoritative and contain actual flow times.
    prefix = 'EP_' if op == 'ep' else 'CP_'
    intervals = []
    for idx in range(min(int(result[13]), len(result[15]))):
        raw = result[15][idx]
        name = (raw.decode('utf-8', errors='ignore') if isinstance(raw, bytes)
                else str(raw or ''))
        if name.startswith(prefix):
            begin, end = float(result[17][idx]), float(result[18][idx])
            if end > begin:
                intervals.append((begin, end))
    intervals.sort()
    merged = []
    for begin, end in intervals:
        if not merged or begin > merged[-1][1]:
            merged.append([begin, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    seconds = sum(end - begin for begin, end in merged)
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError(f'C++ flow reference for {op} is invalid: {seconds}')
    return seconds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes-mb', default='1,4,16,32,64,128')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--fit-min-mb', type=float, default=16.0,
                   help='Exclude latency-dominated samples below this size')
    p.add_argument('--fit-ops', default='ep,cp', choices=('ep', 'cp', 'ep,cp'))
    p.add_argument('--system', type=Path, default=Path('systems/BW1100.json'))
    p.add_argument('--network-index', type=int, default=0)
    p.add_argument('--reference-mb', type=float, default=32.0)
    p.add_argument('--output', type=Path,
                   default=Path('test/bw1100/phase4_rccl_flow_calibration.json'))
    p.add_argument('--update-json', action='store_true',
                   help='Set selected tier bandwidth to fitted effective BW and efficiency=1')
    args = p.parse_args()

    if not dist.is_nccl_available():
        raise SystemExit('PyTorch NCCL backend unavailable (DTK build must provide RCCL)')
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    if not 2 <= world <= 4:
        raise SystemExit(f'requires 2-4 ranks, got {world}')
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device('cuda', local_rank)
    dtype = torch.bfloat16
    itemsize = torch.empty((), dtype=dtype).element_size()
    sizes = sorted({float(v) for v in args.sizes_mb.split(',') if v.strip()})
    rows = []

    for size_mb in sizes:
        count = max(world, int(size_mb * 1024**2 / itemsize))
        count -= count % world
        nbytes = count * itemsize       # per-rank total buffer / send volume
        x = torch.ones(count, dtype=dtype, device=dev)
        y = torch.empty_like(x)
        gathered = torch.empty(count * world, dtype=dtype, device=dev)

        def all_reduce():
            y.copy_(x)
            dist.all_reduce(y)

        def all_to_all():
            dist.all_to_all_single(y, x)

        def cp_ring():
            peer_next, peer_prev = (rank + 1) % world, (rank - 1) % world
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, x, peer_next),
                dist.P2POp(dist.irecv, y, peer_prev),
            ])
            for req in reqs:
                req.wait()

        def all_gather():
            dist.all_gather_into_tensor(gathered, x)

        for name, fn in (('tp_all_reduce', all_reduce),
                         ('ep_all_to_all', all_to_all),
                         ('cp_ring', cp_ring),
                         ('all_gather_diagnostic', all_gather)):
            ms = measure_ms(fn, args.warmup, args.iters, dev)
            rows.append(dict(op=name, size_mb=size_mb, nbytes=nbytes,
                             measured_ms=ms,
                             payload_algbw_gbps=nbytes / (ms * 1e-3) / 1e9))
        del x, y, gathered

    if rank == 0:
        cfg = json.loads(args.system.read_text())
        net_cfg = cfg['networks'][args.network_index]
        current_effective = (float(net_cfg['bandwidth']) *
                             float(net_cfg['efficiency']))
        ref_bytes = int(args.reference_mb * 1024**2)
        reference_bw = 100.0
        flow_latency_s = float(net_cfg['latency'])
        sim_ref = {
            op: flow_reference_seconds(net_cfg, op, world, ref_bytes,
                                       reference_bw)
            for op in ('ep', 'cp')
        }
        key_to_op = {'ep_all_to_all': 'ep', 'cp_ring': 'cp'}
        for row in rows:
            flow_op = key_to_op.get(row['op'])
            if flow_op:
                ref_transfer_s = max(0.0, sim_ref[flow_op] - flow_latency_s)
                scaled_transfer_s = ref_transfer_s * row['nbytes'] / ref_bytes
                measured_transfer_s = row['measured_ms'] * 1e-3 - flow_latency_s
                row['flow_latency_ms'] = flow_latency_s * 1e3
                if measured_transfer_s <= 0:
                    row['flow_fit_error'] = (
                        'measured time <= configured latency; calibrate latency first')
                    continue
                fitted = reference_bw * scaled_transfer_s / measured_transfer_s
                row['flow_fitted_bw_gbps'] = fitted
                predicted_ms = (flow_latency_s + scaled_transfer_s *
                                reference_bw / current_effective) * 1e3
                row['current_flow_model_ms'] = predicted_ms
                row['current_flow_error_pct'] = (
                    100.0 * (predicted_ms - row['measured_ms']) /
                    row['measured_ms'])

        selected = [r for r in rows if 'flow_fitted_bw_gbps' in r
                    if key_to_op.get(r['op']) in args.fit_ops.split(',')
                    and r['size_mb'] >= args.fit_min_mb]
        if not selected:
            raise RuntimeError('no samples selected for fitting')
        # A geometric mean gives EP and CP equal relative-error weight.  The
        # per-op values expose the irreducible mismatch of a one-BW model.
        fitted_values = [r['flow_fitted_bw_gbps'] for r in selected]
        joint_bw = math.exp(statistics.mean(math.log(v) for v in fitted_values))
        per_op = {}
        for op in ('ep', 'cp'):
            vals = [r['flow_fitted_bw_gbps'] for r in selected
                    if key_to_op.get(r['op']) == op]
            if vals:
                per_op[op] = math.exp(statistics.mean(math.log(v) for v in vals))
        for row in rows:
            if 'flow_fitted_bw_gbps' in row:
                transfer_ms = max(0.0, row['current_flow_model_ms'] -
                                  flow_latency_s * 1e3)
                row['fitted_flow_model_ms'] = (flow_latency_s * 1e3 +
                                               transfer_ms * current_effective /
                                               joint_bw)
                row['fitted_flow_error_pct'] = (
                    100.0 * (row['fitted_flow_model_ms'] - row['measured_ms']) /
                    row['measured_ms'])

        nccl_version = getattr(torch.cuda, 'nccl', None)
        nccl_version = (nccl_version.version() if nccl_version is not None
                        else None)
        result = {
            'world_size': world,
            'backend': 'nccl (RCCL on DTK/ROCm)',
            'torch_version': torch.__version__,
            'hip_version': torch.version.hip,
            'nccl_compatible_version': nccl_version,
            'system': str(args.system),
            'network_index': args.network_index,
            'current_effective_bw_gbps': current_effective,
            'fit_min_mb': args.fit_min_mb,
            'fit_ops': args.fit_ops.split(','),
            'per_op_flow_bw_gbps': per_op,
            'recommended_flow_bw_gbps': joint_bw,
            'samples': rows,
        }
        if args.update_json:
            backup = args.system.with_suffix(args.system.suffix + '.before-flow-bw')
            if not backup.exists():
                backup.write_text(args.system.read_text())
            cfg['networks'][args.network_index]['bandwidth'] = joint_bw
            cfg['networks'][args.network_index]['efficiency'] = 1.0
            args.system.write_text(json.dumps(cfg, indent=2) + '\n')
            result['updated_json'] = str(args.system)
            result['backup_json'] = str(backup)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + '.tmp')
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        os.replace(tmp, args.output)
        print(json.dumps(result, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
