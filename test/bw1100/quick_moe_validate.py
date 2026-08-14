#!/usr/bin/env python3
"""One-card, low-memory BW1100 MoE simulator quick validation.

Physical on one GPU: representative forward/agrad/wgrad operators and router
sort. Analytic only: TP/EP/PP/CP communication formulas, timeline ordering and
pipeline bubbles. A one-card run cannot validate physical multi-GPU bandwidth.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BW = ROOT / 'test' / 'bw1100'


def run(cmd, log):
    print('+', ' '.join(map(str, cmd)), flush=True)
    with open(log, 'w') as f:
        p = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=f,
                           stderr=subprocess.STDOUT)
    if p.returncode:
        print(Path(log).read_text(errors='replace')[-3000:])
        raise SystemExit(f'FAIL rc={p.returncode}: {cmd[1]}')


def csv_gate(path: Path, limit: float, require=1):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    errs = [abs(float(r['error_pct'])) for r in rows
            if r.get('comparable', '').lower() == 'true'
            and r.get('error_pct') not in ('', None)]
    mape = sum(errs) / len(errs) if errs else float('inf')
    ok = len(errs) >= require and mape <= limit
    print(f'  {path.name}: comparable={len(errs)} MAPE={mape:.2f}% '
          f'{"PASS" if ok else "FAIL"}')
    return ok


def route_sort(seq, topk, experts, warmup=3, iters=20):
    ids = torch.randint(experts, (seq, topk), device='cuda', dtype=torch.int32)
    flat = ids.flatten()
    def op():
        order = torch.argsort(flat)
        counts = torch.bincount(flat.to(torch.int64), minlength=experts)
        return order, counts
    for _ in range(warmup): op()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters): op()
    end.record(); end.synchronize()
    us = start.elapsed_time(end) * 1000.0 / iters
    print(f'  route sort+bincount: seq={seq} topk={topk} experts={experts} '
          f'{us:.2f} us PASS')
    return us


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seq-size', type=int, default=256,
                   help='small physical-op test seq (default 256)')
    p.add_argument('--out-dir', type=Path, default=BW / 'quick')
    p.add_argument('--gate-pct', type=float, default=25.0)
    p.add_argument('--device', default='auto')
    p.add_argument('--skip-gpu', action='store_true')
    p.add_argument('--timeline', action='store_true')
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    passed = True

    if not args.skip_gpu:
        choices = [(torch.cuda.mem_get_info(i)[0], i)
                   for i in range(torch.cuda.device_count())]
        dev = max(choices)[1] if args.device == 'auto' else int(args.device)
        torch.cuda.set_device(dev)
        print(f'GPU physical checks on device {dev}')
        # One representative projection, BMM and reduction keep this suite
        # small. Each stage is a separate process to release VRAM.
        reps = ['AttnBlock_MLA_WO', 'AttnBlock_MLA_ScoreKV',
                'MlpBlock_LayerNorm']
        for stage in ('fw', 'agrad', 'wgrad'):
            csv_path = args.out_dir / f'phase2_{stage}.csv'
            cmd = [sys.executable, BW / 'phase2_dsv3_op_microbench.py',
                   '--seq-size', args.seq_size, '--stage', stage,
                   '--groups', 'G1', 'G2', 'G3', 'G5',
                   '--blocks', 'dense', '--names', *reps,
                   '--warmup', '2', '--iters', '5',
                   '--min-ms', '10', '--samples', '1', '--device', dev,
                   '--csv', csv_path]
            run(cmd, args.out_dir / f'phase2_{stage}.log')
            passed &= csv_gate(csv_path, args.gate_pct)
        route_us = route_sort(args.seq_size, 8, 256)
    else:
        dev, route_us = None, None

    parallel_ok = True
    parallel_logs = []
    # The C++ flow object retains large per-model state; isolate virtual
    # topologies in subprocesses to keep RSS low and avoid cross-case lifetime
    # bugs in libpycallclass.so.
    for case in ('quick_tp2', 'quick_ep4', 'quick_pp4', 'quick_cp2'):
        cmd = [sys.executable, BW / 'phase4_dsv3_parallel_validate.py',
               '--case', case, '--formula-only']
        if args.timeline:
            cmd.append('--timeline')
        log = args.out_dir / f'parallel_{case}.log'
        try:
            run(cmd, log)
        except SystemExit:
            parallel_ok = False
        parallel_logs.append(log)
    parallel_ok &= all('=== DONE === PASS' in p.read_text(errors='replace')
                       for p in parallel_logs)
    passed &= parallel_ok
    print('  analytic TP/EP/PP/CP/timeline:', 'PASS' if parallel_ok else 'FAIL')

    summary = {
        'pass': bool(passed), 'physical_gpu': dev,
        'physical_seq_size': args.seq_size, 'route_sort_us': route_us,
        'physical_coverage': ['fw', 'agrad', 'wgrad', 'route_sort'],
        'analytic_coverage': ['TP', 'EP', 'PP', 'CP', 'communication',
                              'pipeline_timeline_and_bubble'],
        'not_physically_validated_on_one_gpu': [
            'multi-GPU collective bandwidth/contention',
            'real distributed EP token exchange',
            'real PP runtime scheduling/bubble'],
    }
    (args.out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
