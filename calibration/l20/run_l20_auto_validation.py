#!/usr/bin/env python3
"""Calibrate L20, run Megatron and Calculon, and report prediction error."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path("/src/Simulator");CALCULON=ROOT/"calculon";HERE=Path(__file__).resolve().parent
DEFAULT_MODEL=Path("/models/Qwen3-0.6B");DEFAULT_MOE=Path("/models/DeepSeek-V4-2.7B-tiny")
DEFAULT_DATASET=Path("/datasets/ShareGPT_V3_unfiltered_cleaned_split.json")
def run(cmd,env=None,log=None):
 print("+"," ".join(map(str,cmd)),flush=True)
 if log:
  log.parent.mkdir(parents=True,exist_ok=True)
  with log.open("w") as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 else:subprocess.run(cmd,cwd=ROOT,env=env,check=True)
def compare(log,sim,out,model):
 run([sys.executable,str(HERE/"compare_iteration_logs.py"),"--model",model,"--log",str(log),"--simulator-json",str(sim),"--output",str(out)])
 return json.loads(out.read_text())
def megatron_prefix(path):
 if not path or str(path).lower()=="mock":return None
 p=Path(path)
 if p.suffix in (".bin",".idx"):p=p.with_suffix("")
 return p if p.with_suffix(".bin").exists() and p.with_suffix(".idx").exists() else None
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--gpus",type=int,default=8);p.add_argument("--devices");p.add_argument("--iterations",type=int,default=30);p.add_argument("--seq-length",type=int,default=1024);p.add_argument("--global-batch",type=int,default=8);p.add_argument("--micro-batch",type=int,default=1)
 p.add_argument("--system","--config",dest="system",type=Path,default=CALCULON/"systems/L20.json",help="explicit calibration output JSON")
 p.add_argument("--model-path",type=Path,default=DEFAULT_MODEL);p.add_argument("--moe-model-path",type=Path,default=DEFAULT_MOE);p.add_argument("--dataset-path",type=Path,default=DEFAULT_DATASET)
 p.add_argument("--output-dir",type=Path,default=HERE/"results/l20_auto");p.add_argument("--quick-calibration",action="store_true");p.add_argument("--skip-hardware-calibration",action="store_true");a=p.parse_args()
 if a.gpus not in (4,8):raise SystemExit("validated GPU counts are 4 or 8")
 for x in (a.model_path,a.moe_model_path):
  if not (x/"config.json").is_file():raise SystemExit(f"model path has no config.json: {x}")
 if not a.dataset_path.exists():raise SystemExit(f"dataset path does not exist: {a.dataset_path}")
 env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=a.devices or ",".join(map(str,range(a.gpus)));env["MODEL_PATH"]=str(a.model_path);env["DATASET_PATH"]=str(a.dataset_path)
 prefix=megatron_prefix(a.dataset_path)
 if prefix:env["DATA_PATH"]=str(prefix);dataset_mode="preprocessed"
 else:env.pop("DATA_PATH",None);dataset_mode="mock (source path recorded; preprocess to .bin/.idx to include input pipeline)"
 a.system=a.system.resolve();a.output_dir=a.output_dir.resolve();a.output_dir.mkdir(parents=True,exist_ok=True)
 if not a.skip_hardware_calibration:
  cmd=[sys.executable,str(HERE/"calibrate_l20.py"),"--gpus",str(a.gpus),"--system",str(a.system),"--model-path",str(a.model_path),"--model-path",str(a.moe_model_path),"--dataset-path",str(a.dataset_path),"--output-dir",str(a.output_dir/"hardware"),"--execute"]
  if a.quick_calibration:cmd.append("--quick")
  run(cmd,env,a.output_dir/"hardware-calibration.log")
 elif not a.system.exists():raise SystemExit(f"missing {a.system}")
 report={"gpus":a.gpus,"system":str(a.system),"model_path":str(a.model_path),"moe_model_path":str(a.moe_model_path),"dataset_path":str(a.dataset_path),"dataset_mode":dataset_mode,"cases":{}}
 for case in ("qwen3_06b","moe_v4_tiny"):
  ep=a.gpus if case=="moe_v4_tiny" else 1;d=a.output_dir/case;d.mkdir(parents=True,exist_ok=True);e=env.copy();e["MODEL_PATH"]=str(a.moe_model_path if ep>1 else a.model_path)
  e.update({"NUM_GPUS":str(a.gpus),"TP":"1","PP":"1","EP":str(ep),"ITERATIONS":str(a.iterations),"SEQ_LEN":str(a.seq_length),"MICRO_BATCH":str(a.micro_batch),"GLOBAL_BATCH":str(a.global_batch),"OUT_DIR":str(d)})
  run(["bash",str(HERE/"run_megatron_case.sh"),case],e,d/"megatron-driver.log")
  base=d/"simulator-baseline.json";sc=[sys.executable,str(HERE/"run_calculon_case.py"),"--case",case,"--num-procs",str(a.gpus),"--dp",str(a.gpus),"--ep",str(ep),"--system",str(a.system),"--seq-length",str(a.seq_length),"--global-batch",str(a.global_batch),"--micro-batch",str(a.micro_batch)]
  run(sc+["--output",str(base)],env,d/"simulator.log");actual=d/f"{case}.log";prediction=compare(actual,base,d/"prediction-error.json",case)
  report["cases"][case]={"prediction":prediction};print(json.dumps(report["cases"][case],indent=2),flush=True)
 (a.output_dir/"summary.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
