#!/usr/bin/env python3
"""Fit compact hardware shape models from BW1100 legacy calibration tables."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np


def collect_linear(cfg):
    rows=[]
    for key in ('linear_shape', 'linear_small_n'):
        model=cfg.get(key) or {}; k=int(model.get('reference_k', 0))
        for dtype, curves in (model.get('latency_s') or {}).items():
            bpe={'float8':1,'int8':1,'float16':2,'bfloat16':2,'float32':4}.get(dtype,2)
            for ns, points in curves.items():
                n=int(ns)
                for m,t in points:
                    x=[1.,2*m*k*n/1e15,(m*k+k*n+m*n)*bpe/1e12,
                       ((m+63)//64)*((n+63)//64)/1e6]
                    rows.append((dtype,x,float(t)))
    return rows

def collect_bmm(cfg):
    rows=[]
    table=((cfg.get('operator_shape_latency_s') or {}).get('bmm') or {})
    for dtype, entries in table.items():
        bpe={'float16':2,'bfloat16':2,'float32':4}.get(dtype,2)
        for ent in entries.values():
            b,m,n,k=map(int,ent['shape']); t=float(ent['latency_s'])
            x=[1.,b*2*m*n*k/1e15,b*(m*n+n*k+m*k)*bpe/1e12,
               b*((m+63)//64)*((k+63)//64)/1e6]
            rows.append((dtype,x,t))
    return rows

def collect_operator_linear(cfg):
    rows=[]
    table=((cfg.get('operator_shape_latency_s') or {}).get('linear') or {})
    for dtype, entries in table.items():
        bpe={'float8':1,'int8':1,'float16':2,'bfloat16':2,'float32':4}.get(dtype,2)
        for ent in entries.values():
            m,k,n=map(int,ent['shape']); t=float(ent['latency_s'])
            x=[1.,2*m*k*n/1e15,(m*k+k*n+m*n)*bpe/1e12,
               ((m+63)//64)*((n+63)//64)/1e6]
            rows.append((dtype,x,t))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',type=Path,default=Path('systems/BW1100.json'))
    p.add_argument('--gate-pct',type=float,default=20.)
    p.add_argument('--update-json',action='store_true')
    p.add_argument('--prune-legacy',action='store_true',
                   help='After all fitted models pass, archive and remove '
                        'operator_shape_latency_s/linear_small_n/linear_shape')
    p.add_argument('--restore-legacy',action='store_true')
    a=p.parse_args(); cfg=json.load(open(a.system))
    if a.restore_legacy:
        archive_path=str(a.system)+'.legacy-shapes.json'
        archive=json.load(open(archive_path)); cfg.update(archive)
        tmp=str(a.system)+'.tmp'
        with open(tmp,'w') as f: json.dump(cfg,f,indent=2); f.write('\n')
        os.replace(tmp,a.system); print('restored',archive_path); return
    rows=collect_linear(cfg)
    out={'linear':{},'operator_linear':{},'bmm':{}}
    for kind, source in [('linear',rows),
                         ('operator_linear',collect_operator_linear(cfg)),
                         ('bmm',collect_bmm(cfg))]:
      for dtype in sorted({r[0] for r in source}):
        sub=[r for r in source if r[0]==dtype]
        X=np.asarray([r[1] for r in sub]); y=np.asarray([r[2] for r in sub])
        coef=np.linalg.lstsq(X,y,rcond=None)[0]
        pred=np.maximum(0.,X@coef)
        mape=float(np.mean(np.abs(pred-y)/np.maximum(y,1e-12))*100)
        enabled=bool(mape<=a.gate_pct)
        out[kind][dtype]={'enabled':enabled,
            'coefficients_s':[float(x) for x in coef],
            'features':['constant','flops_PF','bytes_TB','tiles64_M'],
            'fit_points':len(sub),'fit_mape_pct':mape,
            'gate_pct':a.gate_pct}
        if kind == 'linear':
            out[kind][dtype]['reference_k'] = int(
                (cfg.get('linear_shape') or cfg.get('linear_small_n'))
                .get('reference_k', 4096))
        print(kind,dtype, 'points',len(sub),'MAPE',f'{mape:.2f}%',
              'ENABLED' if enabled else 'FALLBACK_TO_TABLE')
    if a.update_json:
        cfg['parametric_shape_models']=out
        if a.prune_legacy:
            if not all(v.get('enabled',False) for group in out.values()
                       for v in group.values()):
                raise SystemExit('refusing --prune-legacy: a fitted model '
                                 'did not pass its MAPE gate')
            archive={k:cfg[k] for k in ('operator_shape_latency_s',
                     'linear_small_n','linear_shape') if k in cfg}
            archive_path=str(a.system)+'.legacy-shapes.json'
            with open(archive_path,'w') as f:
                json.dump(archive,f,indent=2); f.write('\n')
            for k in archive: cfg.pop(k,None)
            print('archived legacy tables to',archive_path)
        tmp=str(a.system)+'.tmp'
        with open(tmp,'w') as f: json.dump(cfg,f,indent=2); f.write('\n')
        os.replace(tmp,a.system); print('updated',a.system)

if __name__=='__main__': main()
