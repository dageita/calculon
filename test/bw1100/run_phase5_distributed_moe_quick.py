#!/usr/bin/env python3
"""Select 2-4 free BW1100 HCUs and launch the Phase5 MoE closure test."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT/'test'/'bw1100'/'phase5_distributed_moe_quick.py'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nproc', type=int, choices=(2, 3, 4), default=2)
    p.add_argument('--devices', help='comma-separated physical HCU ids')
    p.add_argument('--min-free-gb', type=float, default=16.0,
                   help='default covers FP8 grouped fw/agrad/wgrad workspace')
    p.add_argument('--dry-run', action='store_true')
    args, rest = p.parse_known_args()
    if rest and rest[0] == '--': rest = rest[1:]
    if args.devices:
        devices = [int(x) for x in args.devices.split(',') if x.strip()]
        if len(devices) != args.nproc:
            p.error('--devices count must equal --nproc')
    else:
        available = []
        for i in range(torch.cuda.device_count()):
            try:
                free, _ = torch.cuda.mem_get_info(i)
                available.append((free, i))
            except RuntimeError:
                pass
        devices = [i for free, i in sorted(available, reverse=True)
                   if free >= args.min_free_gb*1024**3][:args.nproc]
        if len(devices) < args.nproc:
            raise SystemExit(f'need {args.nproc} HCUs with >= '
                             f'{args.min_free_gb} GiB free; found {devices}')
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           '--nproc-per-node', str(args.nproc), str(WORKER), *rest]
    print(json.dumps({'physical_devices': devices, 'command': cmd}, indent=2),
          flush=True)
    if args.dry_run:
        return
    env = os.environ.copy()
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.pop('ROCR_VISIBLE_DEVICES', None)
    env['HIP_VISIBLE_DEVICES'] = ','.join(map(str, devices))
    # Worker-side quality reporting must use physical HCU ids; after HIP
    # remapping local ranks are only 0..nproc-1.
    env['PHASE5_PHYSICAL_DEVICES'] = ','.join(map(str, devices))
    probe = [sys.executable, '-c',
             ('import torch; print(torch.cuda.device_count()); '
              f'assert torch.cuda.device_count()=={args.nproc}')]
    subprocess.run(probe, cwd=ROOT, env=env, check=True)
    raise SystemExit(subprocess.call(cmd, cwd=ROOT, env=env))

if __name__ == '__main__':
    main()
