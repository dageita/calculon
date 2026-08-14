#!/usr/bin/env python3
"""Select BW1100 HCUs and run communication calibration or MoE validation."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
WORKERS = {
    'legacy': ROOT / 'test' / 'bw1100' / 'phase4_rccl_flow_calibrate.py',
    'latency-bw': ROOT / 'test' / 'bw1100' / 'phase4_rccl_latency_bw_calibrate.py',
    'moe-quick': ROOT / 'test' / 'bw1100' / 'phase5_moe_comm_validate.py',
}
MODES = {
    'calibrate': 'latency-bw',
    'moe': 'moe-quick',
    'legacy': 'legacy',
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nproc', type=int, choices=(2, 3, 4), default=2)
    p.add_argument('--mode', choices=tuple(MODES), default='calibrate',
                   help='calibrate writes latency/BW results; moe validates MoE communication')
    p.add_argument('--worker', choices=tuple(WORKERS), default=None,
                   help=argparse.SUPPRESS)
    p.add_argument('--devices', help='comma-separated physical HCU ids')
    p.add_argument('--min-free-gb', type=float, default=2.0)
    p.add_argument('--dry-run', action='store_true')
    args, rest = p.parse_known_args()
    if rest and rest[0] == '--':
        rest = rest[1:]
    if args.devices:
        devices = [int(x) for x in args.devices.split(',') if x.strip()]
        if len(devices) != args.nproc:
            p.error('--devices count must equal --nproc')
    else:
        available = []
        for idx in range(torch.cuda.device_count()):
            try:
                free, _ = torch.cuda.mem_get_info(idx)
                available.append((free, idx))
            except RuntimeError:
                pass
        devices = [idx for free, idx in sorted(available, reverse=True)
                   if free >= args.min_free_gb * 1024**3][:args.nproc]
        if len(devices) < args.nproc:
            raise SystemExit(f'need {args.nproc} HCUs, found {devices}')
    worker = args.worker or MODES[args.mode]
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           '--nproc-per-node', str(args.nproc), str(WORKERS[worker]), *rest]
    print(json.dumps({'physical_devices': devices, 'command': cmd}, indent=2),
          flush=True)
    if args.dry_run:
        return
    env = os.environ.copy()
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.pop('ROCR_VISIBLE_DEVICES', None)
    env['HIP_VISIBLE_DEVICES'] = ','.join(map(str, devices))
    probe = [sys.executable, '-c',
             ('import torch; print("visible_hcus", torch.cuda.device_count()); '
              f'assert torch.cuda.device_count() == {args.nproc}')]
    subprocess.run(probe, cwd=ROOT, env=env, check=True)
    raise SystemExit(subprocess.call(cmd, cwd=ROOT, env=env))


if __name__ == '__main__':
    main()
