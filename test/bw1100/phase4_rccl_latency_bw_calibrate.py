#!/usr/bin/env python3
"""Fit physical point-to-point link latency/bandwidth on BW1100.

Launch with 2-4 processes.  The PyTorch backend name is ``nccl`` for API
compatibility; DTK/ROCm executes these calls with RCCL.
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


def timed_ms(fn, warmup: int, iters: int, dev: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    samples = []
    for _ in range(iters):
        dist.barrier()
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    value = torch.tensor([statistics.median(samples)], dtype=torch.float64,
                         device=dev)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def fit_latency_bw(rows: list[dict]) -> dict:
    """Robustly fit t = latency + wire_bytes / bandwidth."""
    xs = [float(row['wire_bytes']) for row in rows]
    ys = [float(row['measured_ms']) / 1e3 for row in rows]
    # Small-message measurements are often non-monotonic at sub-microsecond
    # resolution.  Use positive slopes from well-separated sample pairs so
    # launch jitter cannot turn the fitted bandwidth negative.
    separation = max(1, len(xs) // 2)
    slopes = [(ys[j] - ys[i]) / (xs[j] - xs[i])
              for i in range(len(xs)) for j in range(i + separation, len(xs))
              if ys[j] > ys[i]]
    if not slopes:
        raise RuntimeError(
            'message range is latency dominated; include larger --sizes-kb')
    slope = statistics.median(slopes)
    latency = max(0.0, statistics.median(y - slope*x for x, y in zip(xs, ys)))
    bandwidth = 1.0 / slope
    errors = []
    for row, x, y in zip(rows, xs, ys):
        predicted = latency + x / bandwidth
        row['model_ms'] = predicted * 1e3
        row['error_pct'] = 100.0 * (predicted - y) / y
        errors.append(abs(row['error_pct']))
    return {
        'latency_s': latency,
        'latency_us': latency * 1e6,
        'bandwidth_Bps': bandwidth,
        'bandwidth_GBps': bandwidth / 1e9,
        'median_abs_error_pct': statistics.median(errors),
        'max_abs_error_pct': max(errors),
    }


def fit_physical_pingpong(rows: list[dict]) -> dict:
    """Fit one-way link properties from two-way ping-pong samples.

    Small messages determine latency; the incremental slope of the two
    largest messages determines asymptotic bandwidth.  This avoids fitting a
    bandwidth to launch-jitter noise on the latency plateau.
    """
    ordered = sorted(rows, key=lambda row: row['wire_bytes'])
    small = ordered[:max(2, len(ordered) // 2)]
    roundtrip_latency = statistics.median(
        float(row['measured_ms']) / 1e3 for row in small)
    lo, hi = ordered[-2], ordered[-1]
    delta_bytes = float(hi['wire_bytes']) - float(lo['wire_bytes'])
    delta_time = (float(hi['measured_ms']) - float(lo['measured_ms'])) / 1e3
    if delta_time <= 0:
        raise RuntimeError('largest ping-pong samples do not have a positive slope')
    bandwidth = delta_bytes / delta_time
    errors = []
    for row in ordered:
        measured = float(row['measured_ms']) / 1e3
        predicted = roundtrip_latency + float(row['wire_bytes']) / bandwidth
        row['model_ms'] = predicted * 1e3
        row['error_pct'] = 100.0 * (predicted - measured) / measured
        errors.append(abs(row['error_pct']))
    return {
        'latency_s': roundtrip_latency / 2.0,
        'latency_us': roundtrip_latency * 5e5,
        'bandwidth_Bps': bandwidth,
        'bandwidth_GBps': bandwidth / 1e9,
        'median_abs_error_pct': statistics.median(errors),
        'max_abs_error_pct': max(errors),
        'measurement': ('two-way ping-pong; latency from small-message median/2, '
                        'bandwidth from largest-two incremental slope'),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes-kb', default='4,16,64,256,1024,4096,16384,65536')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--output', type=Path,
                   default=Path('test/bw1100/phase4_rccl_latency_bw.json'))
    p.add_argument('--system', type=Path, default=Path('systems/BW1100.json'))
    p.add_argument('--network-index', type=int, default=0)
    p.add_argument('--update-json', action='store_true',
                   help='Write per-group Flow bandwidth/latency into the selected network')
    p.add_argument('--max-fit-error-pct', type=float, default=15.0,
                   help='Reject --update-json if any median fit error exceeds this value')
    p.add_argument('--allow-nonlocal-tier', action='store_true',
                   help='Allow writing a tier other than the smallest-capacity intra-node tier')
    args = p.parse_args()
    if not dist.is_nccl_available():
        raise SystemExit('DTK PyTorch does not expose its RCCL implementation')
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    if not 2 <= world <= 4:
        raise SystemExit(f'requires 2-4 ranks, got {world}')
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device('cuda', local_rank)
    sizes = sorted({int(v) for v in args.sizes_kb.split(',') if v.strip()})
    physical_rows = []
    validation_rows = {name: [] for name in ('all_reduce', 'all_to_all')}

    for size_kb in sizes:
        nbytes = size_kb * 1024
        count = max(world, nbytes // 2)
        count -= count % world
        nbytes = count * 2
        x = torch.ones(count, dtype=torch.bfloat16, device=dev)
        y = torch.empty_like(x)

        def all_reduce():
            y.copy_(x)
            dist.all_reduce(y)

        def all_to_all():
            dist.all_to_all_single(y, x)

        def p2p_pingpong():
            # A full rank-0 <-> rank-1 round trip gives an unambiguous
            # completion point. Additional ranks remain idle, making the
            # physical-link measurement independent of torchrun world size.
            if rank == 0:
                dist.send(x, 1)
                dist.recv(y, 1)
            elif rank == 1:
                dist.recv(y, 0)
                dist.send(y, 0)

        measurements = {
            'all_reduce': timed_ms(all_reduce, args.warmup, args.iters, dev),
            'all_to_all': timed_ms(all_to_all, args.warmup, args.iters, dev),
            'p2p': timed_ms(p2p_pingpong, args.warmup, args.iters, dev),
        }
        physical_rows.append({
            'size_kb': size_kb, 'payload_bytes': nbytes,
            'wire_bytes': 2 * nbytes, 'measured_ms': measurements['p2p']})
        for primitive in ('all_reduce', 'all_to_all'):
            wire_factor = (2.0 * (world - 1) / world
                           if primitive == 'all_reduce' else 1.0)
            validation_rows[primitive].append({
                'size_kb': size_kb,
                'payload_bytes': nbytes,
                'wire_bytes': nbytes * wire_factor,
                'measured_ms': measurements[primitive],
            })
        del x, y

    physical_fit = fit_physical_pingpong(physical_rows)
    result = {
        'world_size': world,
        'backend': 'nccl API (RCCL implementation on DTK/ROCm)',
        'torch_version': torch.__version__,
        'hip_version': torch.version.hip,
        'time_model': 'physical_link_latency_s + bytes / physical_link_bandwidth_Bps',
        'physical_link_fit': physical_fit,
        'physical_link_samples': physical_rows,
        'collective_validation_samples': validation_rows,
        'collective_parameters_are_not_written': True,
    }
    if rank == 0:
        if args.update_json:
            if physical_fit['median_abs_error_pct'] > args.max_fit_error_pct:
                raise RuntimeError(
                    f'refusing to update {args.system}: physical P2P median '
                    f'error {physical_fit["median_abs_error_pct"]:.3f}% exceeds gate')
            cfg = json.loads(args.system.read_text())
            if not 0 <= args.network_index < len(cfg['networks']):
                raise IndexError(f'network index {args.network_index} is out of range')
            tier = cfg['networks'][args.network_index]
            if int(tier['size']) < world:
                raise RuntimeError(
                    f'network[{args.network_index}] size={tier["size"]} cannot '
                    f'contain this {world}-HCU calibration')
            intra_size = min(int(net['size']) for net in cfg['networks'])
            if (int(tier['size']) != intra_size and
                    not args.allow_nonlocal_tier):
                raise RuntimeError(
                    f'refusing to write network[{args.network_index}] with '
                    f'size={tier["size"]}; the inferred intra-node tier has '
                    f'size={intra_size}. Select that tier or pass '
                    '--allow-nonlocal-tier')
            updated = copy.deepcopy(cfg)
            updated_tier = updated['networks'][args.network_index]
            # One physical parameter set for every GroupType.  C++ derives
            # collective rounds, traffic expansion and contention.
            updated_tier.pop('flow_parameters', None)
            updated_tier['bandwidth'] = physical_fit['bandwidth_GBps']
            updated_tier['efficiency'] = 1.0
            updated_tier['latency'] = physical_fit['latency_s']
            backup = args.system.with_suffix(args.system.suffix + '.before-flow-calibration')
            if not backup.exists():
                backup.write_text(args.system.read_text())
            system_tmp = args.system.with_suffix(args.system.suffix + '.tmp')
            system_tmp.write_text(json.dumps(updated, indent=2) + '\n')
            os.replace(system_tmp, args.system)
            result['updated_system'] = str(args.system)
            result['network_index'] = args.network_index
            result['backup_system'] = str(backup)
            result['generic_network_parameters'] = {
                'bandwidth': physical_fit['bandwidth_GBps'],
                'efficiency': 1.0,
                'latency': physical_fit['latency_s'],
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + '.tmp')
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        os.replace(tmp, args.output)
        print(json.dumps(result, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
