#!/usr/bin/env python3
"""One-command L20 matrix/vector/network calibration."""
from __future__ import annotations
import argparse,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; H20=ROOT/"test/h20"; HERE=Path(__file__).resolve().parent
DTYPES=("float8","float16","bfloat16","float32")
PEAK={"float8":239.0,"float16":119.5,"bfloat16":119.5,"float32":59.8}
MODELS=(Path("/models/Qwen3-0.6B"),Path("/models/DeepSeek-V4-2.7B-tiny"))
DATASET=Path("/datasets/ShareGPT_V3_unfiltered_cleaned_split.json")
def phases(): return tuple([f"matrix-{d}" for d in DTYPES]+[f"vector-{d}" for d in DTYPES]+["linear-small-n","memory","network-ar","network-a2a"])
def seed(path,gpus):
 sys.path.insert(0,str(H20)); from calibrate_h20_common import write_system_json
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists(): shutil.copy2(ROOT/"systems/L20.json",path)
 cfg=json.loads(path.read_text()); cfg["mem1"].update(GiB=48,GBps=864)
 cfg["networks"][0].update(bandwidth=32,size=gpus,latency=5e-6,topology="Single machine")
 write_system_json(str(path),cfg)
def validate(gpus):
 import torch
 names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
 if len(names)<gpus or any("L20" not in x for x in names[:gpus]): raise SystemExit(f"expected {gpus} L20 GPUs, got {names!r}")
 if torch.version.cuda!="12.8": raise SystemExit(f"expected PyTorch cu128, got CUDA {torch.version.cuda}")
def command(phase,a):
 if phase.startswith("matrix-"):
  d=phase[7:]; c=[sys.executable,str(H20/"calibrate_h20_matrix_efficiency.py"),"--dtype",d,"--peak-tflops",str(PEAK[d])]
 elif phase.startswith("vector-"):
  d=phase[7:]; c=[sys.executable,str(H20/"calibrate_h20_vector_efficiency.py"),"--dtype",d,"--peak-tflops","auto"]
 elif phase=="linear-small-n":
  c=[sys.executable,str(HERE/"calibrate_linear_small_n.py"),"--update-json",str(a.system)]
  for m in a.model_path:c+=["--model-path",str(m)]
  if a.dataset_path:c+=["--dataset-path",str(a.dataset_path)]
 elif phase=="memory": c=[sys.executable,str(H20/"calibrate_h20_mem_efficiency.py"),"--peak-gbps","864","--capacity-gib","48"]
 else:
  op="all_reduce" if phase=="network-ar" else "all_to_all"
  c=["torchrun","--standalone",f"--nproc_per_node={a.gpus}",str(H20/"calibrate_h20_network_efficiency.py"),"--tier","intra","--collective",op,"--peak-gbps","32","--dump-csv",str(a.output_dir/f"{phase}.csv")]
 if phase!="linear-small-n":c+=["--update-json",str(a.system)]
 if a.quick:
  if phase.startswith(("matrix-","vector-")):c+=["--warmup","2","--iters","3","--min-ms","10","--max-iters","20","--no-dense-small"]
  elif phase=="linear-small-n":c+=["--quick"]
  elif phase.startswith("network-"):c+=["--warmup","2","--iters","3","--min-ms","10"]
 return c
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--phase",choices=("all",*phases()),default="all")
 p.add_argument("--system","--config",dest="system",type=Path,default=ROOT/"systems/L20.json",help="explicit JSON to update")
 p.add_argument("--gpus",type=int,default=4)
 p.add_argument("--model-path",type=Path,action="append",help="repeatable; defaults to Qwen3-0.6B and DeepSeek-V4 tiny")
 p.add_argument("--dataset-path",type=Path,default=DATASET)
 p.add_argument("--output-dir",type=Path,default=ROOT/"calibration/l20/results")
 p.add_argument("--execute",action="store_true");p.add_argument("--quick",action="store_true");a=p.parse_args()
 if a.gpus<2:p.error("--gpus must be >=2")
 a.system=a.system.resolve();a.output_dir=a.output_dir.resolve();a.model_path=a.model_path or list(MODELS)
 for m in a.model_path:
  if not (m/"config.json").is_file() and not m.is_file():p.error(f"bad --model-path: {m}")
 if a.dataset_path and not a.dataset_path.exists():p.error(f"bad --dataset-path: {a.dataset_path}")
 seed(a.system,a.gpus); selected=phases() if a.phase=="all" else (a.phase,); cmds=[command(x,a) for x in selected]
 print("\n".join(" ".join(map(str,x)) for x in cmds),flush=True);a.output_dir.mkdir(parents=True,exist_ok=True)
 manifest={"system":str(a.system),"gpus":a.gpus,"model_paths":[str(x) for x in a.model_path],"dataset_path":str(a.dataset_path),"phases":selected,"commands":cmds}
 (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 if a.execute:
  validate(a.gpus)
  for c in cmds:subprocess.run(c,cwd=ROOT,check=True)
if __name__=="__main__":main()
