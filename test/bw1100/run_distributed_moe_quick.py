#!/usr/bin/env python3
"""Select 2-4 free HCUs and launch distributed_moe_quick.py via torchrun."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
WORKER=ROOT/'test'/'bw1100'/'distributed_moe_quick.py'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nproc',type=int,choices=(2,3,4),default=2)
    p.add_argument('--devices',default=None,
                   help='comma-separated physical HCU ids; default=most free')
    p.add_argument('--min-free-gb',type=float,default=2.0)
    p.add_argument('--dry-run',action='store_true')
    args,rest=p.parse_known_args()
    if rest and rest[0] == '--': rest = rest[1:]
    if args.devices:
        devices=[int(x) for x in args.devices.split(',') if x.strip()]
        if len(devices)!=args.nproc: p.error('--devices count must equal --nproc')
    else:
        avail=[]
        for i in range(torch.cuda.device_count()):
            try: free,total=torch.cuda.mem_get_info(i); avail.append((free,i))
            except RuntimeError: pass
        devices=[i for free,i in sorted(avail,reverse=True)
                 if free>=args.min_free_gb*1024**3][:args.nproc]
        if len(devices)<args.nproc:
            raise SystemExit(f'need {args.nproc} HCUs with >= '
                             f'{args.min_free_gb} GiB free, found {devices}')
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone',
         '--nproc-per-node',str(args.nproc),str(WORKER),*rest]
    print(json.dumps({'physical_devices':devices,'command':cmd},indent=2), flush=True)
    if args.dry_run: return
    env=os.environ.copy()
    # This DTK runtime honours HIP_VISIBLE_DEVICES but ignores
    # ROCR_VISIBLE_DEVICES.  Setting both is unsafe: one visibility layer can
    # remap physical ids before the other filters them, leaving no GPUs.
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.pop('ROCR_VISIBLE_DEVICES', None)
    env['HIP_VISIBLE_DEVICES']=','.join(map(str,devices))

    probe=[sys.executable, '-c',
           ('import json, torch; '
            'print(json.dumps({"visible_hcus": torch.cuda.device_count()})); '
            f'assert torch.cuda.device_count() == {args.nproc}, '
            '"visible HCU count does not match --nproc"')]
    subprocess.run(probe, cwd=ROOT, env=env, check=True)
    raise SystemExit(subprocess.call(cmd,cwd=ROOT,env=env))

if __name__=='__main__': main()
