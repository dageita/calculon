#!/usr/bin/env python3
"""Auto-calibrate small/ridge/saturated grouped-MoE seq regimes."""
from __future__ import annotations
import argparse, csv, json, math, os, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BW=ROOT/'test'/'bw1100'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',type=Path,default=ROOT/'systems'/'BW1100.json')
    p.add_argument('--experts',type=int,default=256)
    p.add_argument('--topk',type=int,default=8)
    p.add_argument('--tokens-per-expert',type=int,nargs=3,default=[4,32,128],
                   metavar=('SMALL','RIDGE','SATURATED'))
    p.add_argument('--device',default='auto')
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--iters',type=int,default=10)
    p.add_argument('--output-dir',type=Path,default=BW/'grouped_seq_calibration')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    seqs=sorted({max(1,math.ceil(t*a.experts/a.topk))
                 for t in a.tokens_per_expert})
    print('auto seq anchors:',seqs,'for tokens/expert',a.tokens_per_expert)
    if a.dry_run: return
    a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seq in seqs:
        out=a.output_dir/f'stepB_seq{seq}.csv'
        log=a.output_dir/f'stepB_seq{seq}.log'
        cmd=[sys.executable,BW/'phase3_stepB_moe_weight_split.py',
             '--system',a.system,'--seq-size',seq,'--device',a.device,
             '--grouped-warmup',a.warmup,'--grouped-iters',a.iters,
             '--grouped-min-ms','50','--grouped-max-iters','50','--csv',out]
        print('+',' '.join(map(str,cmd)),flush=True)
        with open(log,'w') as f:
            subprocess.run(list(map(str,cmd)),cwd=ROOT,stdout=f,
                           stderr=subprocess.STDOUT,check=True)
        with open(out,newline='') as f:
            rows.extend(r for r in csv.DictReader(f)
                        if r.get('grouped_comparable','').lower()=='true')
    cfg=json.load(open(a.system)); table=cfg.setdefault(
        'grouped_moe_shape_latency_s',{}).setdefault('float8',{})
    for r in rows:
        name=r['name']; m=int(r['seq_size']); lat=float(r['grouped_meas_s'])
        # Derive K/N from the existing projection metadata; all seq anchors
        # share K/N, weight/flop multipliers and backend.
        old=table.get(name,{})
        shape=old.get('shape')
        if not shape: raise SystemExit(f'missing base shape for {name}')
        flops=2*m*shape[1]*shape[2]*float(r['flop_mult'])
        nbytes=float(r['total_decomposed_bytes'])
        anchor={'m':m,'tokens_per_expert':m*a.topk/a.experts,
                'latency_s':lat,'effective_bandwidth_Bps':nbytes/lat,
                'effective_flops_per_s':flops/lat}
        ent=table.setdefault(name,old); ent['anchors']=[
            x for x in ent.get('anchors',[]) if int(x['m'])!=m]+[anchor]
        ent['anchors']=sorted(ent['anchors'],key=lambda x:int(x['m']))
        ent['seq_model']='log_rate_interpolation_clamped'
    tmp=str(a.system)+'.tmp'
    with open(tmp,'w') as f: json.dump(cfg,f,indent=2); f.write('\n')
    os.replace(tmp,a.system)
    print('updated',a.system,'anchors',seqs)

if __name__=='__main__': main()
