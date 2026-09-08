#!/usr/bin/env python3
"""Calibrate exact model-aware skinny Linear latency on NVIDIA L20."""
import argparse,json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];H20=ROOT/"test/h20";sys.path.insert(0,str(H20))
from calibrate_h20_common import normalize_dtype,write_system_json
from calibrate_h20_matrix_efficiency import benchmark_shape
DTYPES=("float8","float16","bfloat16","float32")
MODELS=(Path("/models/Qwen3-0.6B"),Path("/models/DeepSeek-V4-2.7B-tiny"))
DATASET=Path("/datasets/ShareGPT_V3_unfiltered_cleaned_split.json")
def shape(path):
 cp=path if path.is_file() else path/"config.json";c=json.loads(cp.read_text());h=int(c["hidden_size"]);heads=int(c.get("num_attention_heads") or 1);kv=int(c.get("num_key_value_heads") or heads);hd=int(c.get("head_dim") or h//heads)
 ns={h,int(c.get("intermediate_size") or 0),int(c.get("moe_intermediate_size") or 0),heads*hd,kv*hd,int(c.get("n_routed_experts") or c.get("num_experts") or 0)}
 return cp,h,sorted(n for n in ns if 0<n<=4096)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dtype",default="all");p.add_argument("--model-path",type=Path,action="append");p.add_argument("--dataset-path",type=Path,default=DATASET)
 p.add_argument("--update-json","--config",dest="system",type=Path,required=True);p.add_argument("--warmup",type=int,default=10);p.add_argument("--iters",type=int,default=30);p.add_argument("--min-ms",type=float,default=100);p.add_argument("--max-iters",type=int,default=200);p.add_argument("--quick",action="store_true");a=p.parse_args()
 models=a.model_path or list(MODELS);dtypes=DTYPES if a.dtype=="all" else (normalize_dtype(a.dtype),);ms=[32,64,128,256,512,1024,2048,4096]
 if a.quick:ms=[128,1024];a.warmup,a.iters,a.min_ms,a.max_iters=2,3,10,20
 cfg=json.loads(a.system.read_text());linear=cfg.setdefault("linear_shape",{});byk=linear.setdefault("latency_s_by_k",{});linear["metadata"]={"policy":"exact K and N only; generic matrix curve otherwise","model_paths":[str(x) for x in models],"dataset_path":str(a.dataset_path)};count=0
 for model in models:
  cp,k,ns=shape(model);print(f"model={cp} K={k} N={ns}",flush=True)
  for d in dtypes:
   kt=byk.setdefault(d,{}).setdefault(str(k),{})
   for n in ns:
    pts=[]
    for m in ms:
     try:_,tf,t,u=benchmark_shape(m,n,k,d,a.warmup,a.iters,a.min_ms,a.max_iters)
     except (RuntimeError,torch.cuda.OutOfMemoryError) as e:print(f"skip {d} {m}x{n}x{k}: {e}",flush=True);torch.cuda.empty_cache();continue
     pts.append([m,round(float(t),12)]);count+=1;print(f"{d} {m}x{n}x{k}: {t*1e6:.3f} us {tf:.3f} TF",flush=True)
    if pts:kt[str(n)]=sorted(pts,reverse=True)
 if not count:raise SystemExit("no measurements collected")
 write_system_json(str(a.system),cfg);print(f"updated {a.system}: {count} linear-small-n measurements")
if __name__=="__main__":main()
