#!/usr/bin/env python3
"""Calibrate H20 network ``efficiency`` for Calculon Flow simulator.

Flow (LLMFlowSimulator) only consumes fabric capacity as:

    effective_BW_Bps = networks[tier].bandwidth * 1e9 * networks[tier].efficiency

``latency`` / ``ops`` / ``processor_usage`` are unused by the C++ flow path, so
for timeline / batch communication times it is enough to calibrate
``efficiency`` (bandwidth stays the vendor peak in GB/s).

This script measures NCCL collective bus bandwidth via ``torch.distributed``
across message sizes, then sets:

    efficiency = clamp(busbw_GBps / peak_GBps, floor, 1.0)

Default recommendation uses the **large-message asymptote** (sizes >=
``--large-bytes``), which matches bandwidth-bound training traffic.

Collectives (NCCL busbw factor, same convention as nccl-tests):
  all_reduce : busbw = algbw * 2*(n-1)/n
  all_gather / reduce_scatter / all_to_all : busbw = algbw * (n-1)/n

Examples (on the target GPU cluster)::

  # Intra-node NVLink (8 GPUs / host) → networks[0]
  torchrun --standalone --nproc_per_node=8 \\
    test/calibrate_h20_network_efficiency.py --tier intra \\
    --update-json systems/H20.json

  # Inter-node NIC (2 hosts × 8) → networks[1]
  torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \\
    --rdzv_endpoint=$MASTER_ADDR:29500 \\
    test/calibrate_h20_network_efficiency.py --tier inter \\
    --collective all_to_all --update-json systems/H20.json

  # Dry-run table only (no JSON write)
  torchrun --standalone --nproc_per_node=8 \\
    test/calibrate_h20_network_efficiency.py --tier intra --collective all_reduce
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_TEST_DIR, '..'))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from calibrate_h20_common import (  # noqa: E402
    H20_INTER_PEAK_GBPS,
    H20_INTRA_PEAK_GBPS,
    write_system_json,
)

# Message sizes (bytes). Ascending so the first printed row appears quickly;
# large-message asymptote is still taken from sizes >= --large-bytes.
DEFAULT_SIZES_BYTES: List[int] = [
    1 << 12,  # 4 KiB
    1 << 14,  # 16 KiB
    1 << 16,  # 64 KiB
    1 << 18,  # 256 KiB
    1 << 20,  # 1 MiB
    1 << 22,  # 4 MiB
    1 << 24,  # 16 MiB
    1 << 25,  # 32 MiB
    1 << 26,  # 64 MiB
    1 << 27,  # 128 MiB
    1 << 28,  # 256 MiB
]


def _busbw_factor(collective: str, world_size: int) -> float:
    """Convert algorithmic bandwidth → bus bandwidth (nccl-tests convention)."""
    n = world_size
    if n <= 1:
        return 1.0
    if collective == 'all_reduce':
        return 2.0 * (n - 1) / n
    if collective in ('all_gather', 'reduce_scatter', 'all_to_all'):
        return (n - 1) / n
    raise ValueError(f'unknown collective: {collective}')


def _run_collective(collective: str, tensor: torch.Tensor,
                    workspace: Optional[torch.Tensor] = None) -> None:
    if collective == 'all_reduce':
        dist.all_reduce(tensor)
    elif collective == 'all_gather':
        # out size = world * local; reuse workspace
        assert workspace is not None
        dist.all_gather_into_tensor(workspace, tensor)
    elif collective == 'reduce_scatter':
        assert workspace is not None
        dist.reduce_scatter_tensor(workspace, tensor)
    elif collective == 'all_to_all':
        assert workspace is not None
        dist.all_to_all_single(workspace, tensor)
    else:
        raise ValueError(collective)


def _alloc_tensors(
    collective: str, nbytes: int, device: torch.device, world_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """Allocate input (/output) tensors. Returns (inp, workspace, counted_bytes).

    ``counted_bytes`` is the message size used for algbw (= NCCL ``size``).
    """
    # Use uint8 so nbytes maps 1:1 to element count.
    nelem = max(world_size, nbytes)  # all_gather/rs need rank-aligned chunks
    # Align to world_size for reduce_scatter / all_gather chunking.
    if nelem % world_size != 0:
        nelem += world_size - (nelem % world_size)

    if collective == 'all_reduce':
        t = torch.empty(nelem, dtype=torch.uint8, device=device)
        return t, None, nelem

    if collective == 'all_gather':
        # each rank contributes nelem // world_size, output is nelem
        chunk = nelem // world_size
        inp = torch.empty(chunk, dtype=torch.uint8, device=device)
        out = torch.empty(nelem, dtype=torch.uint8, device=device)
        return inp, out, chunk  # nccl-tests size = sendbytes per rank

    if collective == 'reduce_scatter':
        inp = torch.empty(nelem, dtype=torch.uint8, device=device)
        out = torch.empty(nelem // world_size, dtype=torch.uint8, device=device)
        return inp, out, nelem // world_size

    if collective == 'all_to_all':
        # full buffer exchanged; size = nbytes per rank
        t = torch.empty(nelem, dtype=torch.uint8, device=device)
        out = torch.empty_like(t)
        return t, out, nelem

    raise ValueError(collective)


@torch.no_grad()
def benchmark_size(
    collective: str,
    nbytes: int,
    warmup: int,
    iters: int,
    min_ms: float,
) -> Tuple[float, float, float, int]:
    """Return (latency_s, algbw_GBps, busbw_GBps, iters_used).

    Latency is MAX across ranks (slowest rank bounds the collective).
    Callers must ensure every rank enters with the same nbytes (no local skip).
    """
    world = dist.get_world_size()
    device = torch.device(f'cuda:{torch.cuda.current_device()}')

    # Sync alloc success before any collective — otherwise one-rank OOM hangs NCCL.
    ok = torch.ones(1, device=device, dtype=torch.int32)
    inp = workspace = None
    counted = 0
    try:
        inp, workspace, counted = _alloc_tensors(
            collective, nbytes, device, world)
        inp.fill_(1)
        if workspace is not None:
            workspace.zero_()
    except torch.cuda.OutOfMemoryError:
        ok[0] = 0
        torch.cuda.empty_cache()
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if int(ok.item()) == 0:
        del inp, workspace
        torch.cuda.empty_cache()
        raise torch.cuda.OutOfMemoryError(
            f'OOM (or peer OOM) allocating {nbytes} bytes for {collective}')

    warmup_used = warmup
    if nbytes >= (1 << 26):
        warmup_used = min(warmup, 5)

    for _ in range(max(1, warmup_used)):
        _run_collective(collective, inp, workspace)
    torch.cuda.synchronize()
    dist.barrier()

    # Probe locally, then agree on a SINGLE iters_used across ranks.
    # Using per-rank probe_ms alone desyncs the timed loop (one rank finishes
    # early and enters the numel=1 MAX all_reduce while peers are still on the
    # data collective) → NCCL timeout. Log smoking gun: Rank0 NumelIn=1 vs
    # others NumelIn=4096 at the same SeqNum.
    iters_used = max(1, iters)
    probe_ms = 0.0
    if min_ms > 0:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            _run_collective(collective, inp, workspace)
        end.record()
        torch.cuda.synchronize()
        probe_ms = start.elapsed_time(end) / 5.0

    probe_t = torch.tensor([probe_ms], device=device, dtype=torch.float64)
    dist.all_reduce(probe_t, op=dist.ReduceOp.MAX)
    probe_ms = float(probe_t.item())
    if min_ms > 0 and probe_ms > 0:
        iters_used = max(iters_used, int(min_ms / max(probe_ms, 1e-3)) + 1)
    # Hard caps: small msgs don't need hundreds of iters; large msgs stay short.
    iters_used = min(iters_used, 100)
    if nbytes >= (1 << 26):
        iters_used = min(iters_used, max(iters, 20))
    iters_t = torch.tensor([iters_used], device=device, dtype=torch.int64)
    dist.all_reduce(iters_t, op=dist.ReduceOp.MAX)
    iters_used = int(iters_t.item())

    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters_used):
        _run_collective(collective, inp, workspace)
    end.record()
    torch.cuda.synchronize()
    local_ms = start.elapsed_time(end) / iters_used

    t = torch.tensor([local_ms], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    latency_s = float(t.item()) / 1000.0

    algbw = (counted / latency_s) / 1e9 if latency_s > 0 else 0.0
    busbw = algbw * _busbw_factor(collective, world)
    del inp, workspace
    torch.cuda.empty_cache()
    return latency_s, algbw, busbw, iters_used


def recommend_efficiency(
    rows: Sequence[Tuple[int, float, float, float]],
    peak_gbps: float,
    large_bytes: int,
    floor_eff: float,
) -> Tuple[float, List[Tuple[int, float]]]:
    """Pick constant efficiency from large-message busbw/peak.

    Returns (efficiency, per_size_eff_table).
    """
    per_size: List[Tuple[int, float]] = []
    large_effs: List[float] = []
    for nbytes, _lat, _alg, busbw in rows:
        eff = busbw / peak_gbps if peak_gbps > 0 else 0.0
        per_size.append((nbytes, eff))
        if nbytes >= large_bytes:
            large_effs.append(eff)
    if large_effs:
        raw = sum(large_effs) / len(large_effs)
    elif per_size:
        raw = per_size[-1][1]  # largest size (list is ascending)
    else:
        raw = floor_eff
    eff = max(floor_eff, min(1.0, raw))
    return eff, per_size


def update_json(path: str, tier: str, efficiency: float,
                peak_gbps: Optional[float] = None) -> None:
    with open(path) as f:
        cfg = json.load(f)
    nets = cfg.setdefault('networks', [])
    idx = 0 if tier == 'intra' else 1
    while len(nets) <= idx:
        nets.append({})
    nets[idx]['efficiency'] = round(float(efficiency), 6)
    if peak_gbps is not None:
        nets[idx]['bandwidth'] = float(peak_gbps)
    write_system_json(path, cfg)
    print(f'\nUpdated networks[{idx}] ({tier}) efficiency={efficiency:.6f}'
          f'{f" bandwidth={peak_gbps}" if peak_gbps is not None else ""} '
          f'in {path}')


def _init_dist() -> Tuple[int, int, int]:
    if not dist.is_available():
        raise SystemExit('torch.distributed is not available')
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun / elastic
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        raise SystemExit(
            'Must launch with torchrun, e.g.\n'
            '  torchrun --standalone --nproc_per_node=8 '
            'test/calibrate_h20_network_efficiency.py --tier intra'
        )
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--tier', choices=('intra', 'inter'), required=True,
        help='Which systems JSON network tier to calibrate '
             '(intra=networks[0] NVLink, inter=networks[1] NIC)',
    )
    parser.add_argument(
        '--collective',
        choices=('all_reduce', 'all_gather', 'reduce_scatter', 'all_to_all'),
        default='all_reduce',
        help='NCCL collective to measure (default all_reduce; '
             'use all_to_all for MoE EP traffic)',
    )
    parser.add_argument(
        '--peak-gbps', type=float, default=None,
        help='Nominal unidirectional peak GB/s '
             f'(default intra={H20_INTRA_PEAK_GBPS}, inter={H20_INTER_PEAK_GBPS})',
    )
    parser.add_argument(
        '--large-bytes', type=int, default=1 << 26,
        help='Sizes >= this contribute to asymptotic efficiency (default 64MiB)',
    )
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--min-ms', type=float, default=200.0,
                        help='Min window per size to stabilize timing')
    parser.add_argument('--floor-eff', type=float, default=0.05)
    parser.add_argument(
        '--max-bytes', type=int, default=None,
        help='Skip sizes larger than this (useful if GPU mem is tight)',
    )
    parser.add_argument(
        '--write-peak', action='store_true',
        help='Also write --peak-gbps into networks[tier].bandwidth',
    )
    parser.add_argument('--update-json', type=str, default=None,
                        help='Path to systems/*.json to update')
    parser.add_argument(
        '--dump-csv', type=str, default=None,
        help='Optional path (rank0) to write per-size measurements',
    )
    args = parser.parse_args()

    peak = args.peak_gbps
    if peak is None:
        peak = H20_INTRA_PEAK_GBPS if args.tier == 'intra' else H20_INTER_PEAK_GBPS

    rank, world, local_rank = _init_dist()
    device = torch.device(f'cuda:{local_rank}')

    # CRITICAL: every rank must use the SAME size list. Per-GPU free memory
    # differs; filtering locally used to desync collectives → NCCL hang with
    # all GPUs at 100% util and no further log lines.
    free_b, total_b = torch.cuda.mem_get_info()
    free_t = torch.tensor([free_b], device=device, dtype=torch.int64)
    dist.all_reduce(free_t, op=dist.ReduceOp.MIN)
    free_min = int(free_t.item())
    mem_cap = max(1 << 20, free_min // 4)

    if rank == 0:
        print(
            f'Calibrating network efficiency | tier={args.tier} '
            f'collective={args.collective} world={world} '
            f'peak={peak} GB/s device=cuda:{local_rank}',
            flush=True,
        )
        print(
            f'GPU mem: min_free={free_min/1e9:.2f}GB / total~{total_b/1e9:.2f}GB '
            f'→ size cap={mem_cap/1e6:.0f}MB (global MIN free/4)',
            flush=True,
        )
        print(
            f'{"bytes":>12} {"lat_us":>12} {"algbw_GB/s":>12} '
            f'{"busbw_GB/s":>12} {"eff":>8} {"iters":>8}',
            flush=True,
        )

    sizes = list(DEFAULT_SIZES_BYTES)
    if args.max_bytes is not None:
        sizes = [s for s in sizes if s <= args.max_bytes]
    sizes = [s for s in sizes if s <= mem_cap]
    # Broadcast length then list so every rank agrees even if logic drifts.
    n_sizes = torch.tensor([len(sizes)], device=device, dtype=torch.int64)
    dist.broadcast(n_sizes, src=0)
    if rank == 0:
        sizes_t = torch.tensor(sizes, device=device, dtype=torch.int64)
    else:
        sizes_t = torch.empty(int(n_sizes.item()), device=device, dtype=torch.int64)
    if int(n_sizes.item()) == 0:
        if rank == 0:
            raise SystemExit('No message sizes fit in available GPU memory')
        dist.barrier()
        dist.destroy_process_group()
        return
    dist.broadcast(sizes_t, src=0)
    sizes = [int(x) for x in sizes_t.tolist()]

    rows: List[Tuple[int, float, float, float]] = []
    for nbytes in sizes:
        if rank == 0:
            print(f'... benchmarking {nbytes} bytes', flush=True)
        try:
            lat_s, algbw, busbw, iters_used = benchmark_size(
                args.collective, nbytes, args.warmup, args.iters, args.min_ms,
            )
        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f'{nbytes:>12}  skip OOM (any rank)', flush=True)
            dist.barrier()
            continue
        rows.append((nbytes, lat_s, algbw, busbw))
        if rank == 0:
            eff = busbw / peak if peak > 0 else 0.0
            print(
                f'{nbytes:>12} {lat_s*1e6:>12.1f} {algbw:>12.3f} '
                f'{busbw:>12.3f} {eff:>8.4f} {iters_used:>8}',
                flush=True,
            )

    dist.barrier()
    if rank == 0:
        eff, per_size = recommend_efficiency(
            rows, peak, args.large_bytes, args.floor_eff,
        )
        print('\nPer-size efficiency (busbw / peak):')
        for nbytes, e in per_size:
            mark = ' *' if nbytes >= args.large_bytes else ''
            print(f'  {nbytes:>12}  {e:.4f}{mark}')
        print(
            f'\nRecommended networks[{0 if args.tier == "intra" else 1}]'
            f'.efficiency = {eff:.6f} '
            f'(mean of sizes >= {args.large_bytes} bytes, '
            f'clamped to [{args.floor_eff}, 1])'
        )
        print(
            f'Flow effective BW = {peak} * {eff:.6f} = {peak * eff:.3f} GB/s'
        )

        if args.dump_csv:
            with open(args.dump_csv, 'w') as f:
                f.write('bytes,latency_s,algbw_GBps,busbw_GBps,efficiency\n')
                for nbytes, lat, alg, bus in rows:
                    f.write(
                        f'{nbytes},{lat},{alg},{bus},{bus / peak if peak else 0}\n'
                    )
            print(f'Wrote {args.dump_csv}')

        if args.update_json:
            path = args.update_json
            if not os.path.isabs(path):
                # try cwd then repo root
                if not os.path.exists(path):
                    alt = os.path.join(_ROOT, path)
                    if os.path.exists(alt):
                        path = alt
            update_json(
                path, args.tier, eff,
                peak_gbps=peak if args.write_peak else None,
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
