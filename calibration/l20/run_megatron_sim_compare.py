#!/usr/bin/env python3
"""One-command L20 Megatron measurement vs Calculon prediction.

Uses Megatron dual rank generators: TP*CP*DP*PP = world and
ETP*EP*EDP*PP = world. EDP is derived when omitted.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from model_catalog import (CASE_CHOICES, application_from_hf,
                           default_model_path)
ROOT=Path("/src/Simulator");HERE=Path(__file__).resolve().parent
DEFAULT_MODELS={case: default_model_path(case) for case in CASE_CHOICES}
DEFAULT_DATASET=Path("/datasets/ShareGPT_V3_unfiltered_cleaned_split.json")
SYSTEMS=ROOT / "calculon/systems"

OPTIMIZER_FLAGS = {
 "--use-distributed-optimizer", "--use-precision-aware-optimizer",
 "--main-grads-dtype", "--main-params-dtype", "--exp-avg-dtype",
 "--exp-avg-sq-dtype", "--grad-reduce-in-bf16",
 "--optimizer-cpu-offload", "--optimizer-offload-fraction",
 "--use-torch-optimizer-for-cpu-offload",
 "--overlap-cpu-optimizer-d2h-h2d",
 "--no-pin-cpu-grads", "--no-pin-cpu-params",
}

def add_optimizer_arguments(parser):
 parser.add_argument("--use-distributed-optimizer",action="store_true")
 parser.add_argument("--use-precision-aware-optimizer",action="store_true")
 parser.add_argument("--main-grads-dtype",choices=("fp32","bf16"),default="fp32")
 parser.add_argument("--main-params-dtype",choices=("fp32","fp16"),default="fp32")
 parser.add_argument("--exp-avg-dtype",choices=("fp32","fp16","fp8"),default="fp32")
 parser.add_argument("--exp-avg-sq-dtype",choices=("fp32","fp16","fp8"),default="fp32")
 parser.add_argument("--grad-reduce-in-bf16",action="store_true")
 parser.add_argument("--optimizer-cpu-offload",action="store_true")
 parser.add_argument("--optimizer-offload-fraction",type=float,default=1.0)
 parser.add_argument("--use-torch-optimizer-for-cpu-offload",action="store_true")
 parser.add_argument("--overlap-cpu-optimizer-d2h-h2d",action="store_true")
 parser.add_argument("--pin-cpu-grads",action=argparse.BooleanOptionalAction,
                     default=True)
 parser.add_argument("--pin-cpu-params",action=argparse.BooleanOptionalAction,
                     default=True)

def optimizer_contract(a):
 return {
  "optimizer_sharding":a.use_distributed_optimizer,
  "use_precision_aware_optimizer":a.use_precision_aware_optimizer,
  "main_grads_dtype":a.main_grads_dtype,
  "main_params_dtype":a.main_params_dtype,
  "exp_avg_dtype":a.exp_avg_dtype,
  "exp_avg_sq_dtype":a.exp_avg_sq_dtype,
  "grad_reduce_in_bf16":a.grad_reduce_in_bf16,
  "optimizer_offload":a.optimizer_cpu_offload,
  "optimizer_offload_fraction":a.optimizer_offload_fraction,
  "use_torch_optimizer_for_cpu_offload":
   a.use_torch_optimizer_for_cpu_offload,
  "overlap_cpu_optimizer_d2h_h2d":
   a.overlap_cpu_optimizer_d2h_h2d,
  "pin_cpu_grads":a.pin_cpu_grads,
  "pin_cpu_params":a.pin_cpu_params,
 }

def optimizer_megatron_args(a):
 args=[]
 if a.use_distributed_optimizer:args.append("--use-distributed-optimizer")
 if a.use_precision_aware_optimizer:
  args.extend([
   "--use-precision-aware-optimizer",
   "--main-grads-dtype",a.main_grads_dtype,
   "--main-params-dtype",a.main_params_dtype,
   "--exp-avg-dtype",a.exp_avg_dtype,
   "--exp-avg-sq-dtype",a.exp_avg_sq_dtype])
 if a.grad_reduce_in_bf16:args.append("--grad-reduce-in-bf16")
 if a.optimizer_cpu_offload:
  args.extend(["--optimizer-cpu-offload","--optimizer-offload-fraction",
               str(a.optimizer_offload_fraction)])
 if a.use_torch_optimizer_for_cpu_offload:
  args.append("--use-torch-optimizer-for-cpu-offload")
 if a.overlap_cpu_optimizer_d2h_h2d:
  args.append("--overlap-cpu-optimizer-d2h-h2d")
 if not a.pin_cpu_grads:args.append("--no-pin-cpu-grads")
 if not a.pin_cpu_params:args.append("--no-pin-cpu-params")
 return args

def optimizer_calculon_args(a):
 args=[
  "--main-grads-dtype",a.main_grads_dtype,
  "--main-params-dtype",a.main_params_dtype,
  "--exp-avg-dtype",a.exp_avg_dtype,
  "--exp-avg-sq-dtype",a.exp_avg_sq_dtype,
  "--optimizer-offload-fraction",str(a.optimizer_offload_fraction)]
 for enabled,flag in (
  (a.use_distributed_optimizer,"--use-distributed-optimizer"),
  (a.use_precision_aware_optimizer,"--use-precision-aware-optimizer"),
  (a.grad_reduce_in_bf16,"--grad-reduce-in-bf16"),
  (a.optimizer_cpu_offload,"--optimizer-cpu-offload"),
  (a.use_torch_optimizer_for_cpu_offload,
   "--use-torch-optimizer-for-cpu-offload"),
  (a.overlap_cpu_optimizer_d2h_h2d,
   "--overlap-cpu-optimizer-d2h-h2d")):
  if enabled:args.append(flag)
 if not a.pin_cpu_grads:args.append("--no-pin-cpu-grads")
 if not a.pin_cpu_params:args.append("--no-pin-cpu-params")
 return args

def resolve_system(system, gpu):
 """Resolve simulator hardware independently from the measured GPU."""
 if system is not None:
  return system.resolve()
 if gpu is None:
  return (SYSTEMS / "L20.json").resolve()
 requested = str(gpu).lower()
 direct = Path(gpu)
 if direct.is_file():
  return direct.resolve()
 matches = [path for path in SYSTEMS.glob("*.json")
            if path.name.lower() == requested or
            path.stem.lower() == requested.removesuffix(".json")]
 if len(matches) == 1:
  return matches[0].resolve()
 available = ", ".join(path.stem for path in sorted(SYSTEMS.glob("*.json")))
 raise ValueError(f"unknown simulator GPU {gpu!r}; choose one of: {available}")

def run(cmd, env, log):
 print("+", " ".join(map(str, cmd)), flush=True)
 log.parent.mkdir(parents=True, exist_ok=True)
 try:
  with log.open("w") as stream:
   subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                  stderr=subprocess.STDOUT, check=True)
 except subprocess.CalledProcessError as exc:
  lines = log.read_text(errors="replace").splitlines()
  markers = ("OutOfMemoryError:", "AssertionError:", "ValueError:",
             "RuntimeError:", "Signal 9", "SIGKILL")
  causes = []
  for line in lines:
   if any(marker in line for marker in markers):
    cause = line.strip()
    if cause not in causes:
     causes.append(cause)
  if causes:
   print(f"--- {log} (detected root causes) ---", file=sys.stderr)
   print("\n".join(causes[-8:]), file=sys.stderr)
  tail = "\n".join(lines[-40:])
  if tail:
   print(f"--- {log} (last 40 lines) ---\n{tail}", file=sys.stderr)
  raise SystemExit(f"command failed with exit code {exc.returncode}; full log: {log}") from exc
def data_prefix(path):
 p=Path(path)
 if p.suffix in (".bin",".idx"):p=p.with_suffix("")
 return p if p.with_suffix(".bin").is_file() and p.with_suffix(".idx").is_file() else None
def proc_vmstat_counter(name):
 try:
  for line in Path("/proc/vmstat").read_text().splitlines():
   if line.startswith(name + " "):
    return int(line.split()[1])
 except (OSError, ValueError):
  pass
 return None
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--case",choices=CASE_CHOICES,required=True,help="architecture adapter; --model-path supplies the corresponding local model directory")
 p.add_argument("--model-path",type=Path,help="local Hugging Face model directory; defaults by --case")
 p.add_argument("--dataset-path",type=Path,default=DEFAULT_DATASET,help="Megatron indexed prefix (.bin/.idx); raw JSON is recorded and uses mock-data")
 systems=p.add_mutually_exclusive_group()
 systems.add_argument("--system",type=Path,help="simulator system JSON (advanced override)")
 systems.add_argument("--gpu",help="simulator GPU/system name, e.g. L20 or A100_80G_NVLINK")
 p.add_argument("--output-dir",type=Path,default=HERE/"results/compare")
 p.add_argument("--devices",help="CUDA_VISIBLE_DEVICES; default selects 0..world-1")
 p.add_argument("--tp",type=int,default=1);p.add_argument("--pp",type=int,default=1);p.add_argument("--dp",type=int,default=1);p.add_argument("--ep",type=int,default=1);p.add_argument("--etp",type=int,help="expert tensor parallel; defaults to TP like Megatron");p.add_argument("--edp",type=int,help="expert data parallel; derived and validated");p.add_argument("--cp",type=int,default=1)
 p.add_argument("--iterations",type=int,default=30);p.add_argument("--seq-length",type=int,default=1024);p.add_argument("--global-batch",type=int,default=8);p.add_argument("--micro-batch",type=int,default=1)
 p.add_argument("--optimization-strategy", choices=("none","attention-only","full"), default="none", help="activation checkpoint policy; mapped identically to Megatron and Calculon")
 p.add_argument("--sequence-parallel", action=argparse.BooleanOptionalAction,
                default=None, help="use TP reduce-scatter/all-gather; defaults on for MoE with TP>1 because Megatron requires it")
 p.add_argument("--precision",choices=("bf16","fp16"),default="bf16")
 p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--min-lr",type=float,default=1e-5);p.add_argument("--warmup-fraction",type=float,default=.01);p.add_argument("--weight-decay",type=float,default=.1);p.add_argument("--clip-grad",type=float,default=1.)
 p.add_argument("--extra-megatron-arg",action="append",default=[],help="repeatable non-performance Megatron diagnostic argument; modeled optimizer flags have dedicated options")
 add_optimizer_arguments(p)
 p.add_argument("--warmup-samples",type=int,default=5)
 a=p.parse_args()
 if not 0 <= a.optimizer_offload_fraction <= 1:
  p.error("--optimizer-offload-fraction must be in [0, 1]")
 if a.use_precision_aware_optimizer and not a.use_distributed_optimizer:
  p.error("--use-precision-aware-optimizer requires --use-distributed-optimizer")
 if a.optimizer_cpu_offload and not a.use_precision_aware_optimizer:
  p.error("--optimizer-cpu-offload requires --use-precision-aware-optimizer")
 if a.grad_reduce_in_bf16 and a.precision != "bf16":
  p.error("--grad-reduce-in-bf16 requires --precision bf16")
 raw_optimizer_flags=[
  value for value in a.extra_megatron_arg
  if value.split("=",1)[0] in OPTIMIZER_FLAGS]
 if raw_optimizer_flags:
  p.error("use the explicit comparison option instead of "
          f"--extra-megatron-arg for: {', '.join(raw_optimizer_flags)}")
 a.model_path=a.model_path or DEFAULT_MODELS[a.case]
 for name in ("tp","pp","dp","ep","cp","iterations","seq_length","global_batch","micro_batch"):
  if getattr(a,name)<=0:p.error(f"--{name.replace('_','-')} must be positive")
 if a.seq_length%(2*a.cp):p.error("--seq-length must be divisible by 2*--cp")
 if not (a.model_path/"config.json").is_file():p.error(f"--model-path has no config.json: {a.model_path}")
 try:
  app = application_from_hf(a.case, a.model_path, a.seq_length)
 except (KeyError, ValueError, json.JSONDecodeError) as exc:
  p.error(f"invalid model contract for {a.model_path}: {exc}")
 is_moe = bool(app.get("num_experts"))
 if a.sequence_parallel is None:
  a.sequence_parallel = is_moe and a.tp > 1
 if is_moe and a.tp > 1 and not a.sequence_parallel:
  p.error("MoE training with --tp > 1 requires --sequence-parallel in this Megatron version")
 world = a.tp * a.pp * a.cp * a.dp
 if not is_moe:
  # Dense models do not instantiate Megatron's expert rank generator. Ignore
  # expert-only CLI values and use a neutral internally valid rank shape.
  a.ep = a.etp = 1
  a.edp = world // a.pp
 else:
  a.etp = a.etp or a.tp
  if a.etp <= 0 or (a.edp is not None and a.edp <= 0):
   p.error("--etp and --edp must be positive")
  if app["num_experts"] % a.ep:
   p.error("model num_experts must be divisible by --ep")
  expert_factor = a.etp * a.ep * a.pp
  if world % expert_factor:
   p.error("--etp * --ep * --pp must divide dense world TP*CP*DP*PP")
  derived_edp = world // expert_factor
  if a.edp is not None and a.edp != derived_edp:
   p.error(f"--edp must equal {derived_edp}=world/(ETP*EP*PP)")
  a.edp = derived_edp
 if not a.dataset_path.exists() and not data_prefix(a.dataset_path):p.error(f"--dataset-path does not exist: {a.dataset_path}")
 devices=a.devices or ",".join(map(str,range(world)))
 if len(devices.split(","))!=world:p.error("--devices count must equal TP*PP*CP*DP")
 try:a.system=resolve_system(a.system,a.gpu)
 except ValueError as exc:p.error(str(exc))
 if not a.system.is_file():p.error(f"missing --system: {a.system}")
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update({"CUDA_VISIBLE_DEVICES":devices,"NUM_GPUS":str(world),"TP":str(a.tp),"PP":str(a.pp),"DP":str(a.dp),"EP":str(a.ep),"ETP":str(a.etp),"EDP":str(a.edp),"CP":str(a.cp),"SEQUENCE_PARALLEL":"1" if a.sequence_parallel else "0","ITERATIONS":str(a.iterations),"SEQ_LEN":str(a.seq_length),"GLOBAL_BATCH":str(a.global_batch),"MICRO_BATCH":str(a.micro_batch),"LR":str(a.lr),"MIN_LR":str(a.min_lr),"WARMUP_FRACTION":str(a.warmup_fraction),"WEIGHT_DECAY":str(a.weight_decay),"CLIP_GRAD":str(a.clip_grad),"OPTIMIZATION_STRATEGY":a.optimization_strategy,"PRECISION":a.precision,"MODEL_PATH":str(a.model_path),"OUT_DIR":str(out),"EXTRA_MEGATRON_ARGS":" ".join(a.extra_megatron_arg + optimizer_megatron_args(a))})
 prefix=data_prefix(a.dataset_path)
 if prefix:env["DATA_PATH"]=str(prefix);dataset_mode="preprocessed"
 else:env.pop("DATA_PATH",None);dataset_mode="mock; raw dataset retained as provenance"
 oom_before=proc_vmstat_counter("oom_kill")
 megatron_error=None
 try:
  run(["bash",str(HERE/"run_megatron_case.sh"),a.case],env,out/"megatron-driver.log")
 except SystemExit as exc:
  megatron_error=exc
 oom_after=proc_vmstat_counter("oom_kill")
 oom_delta=(oom_after-oom_before if None not in (oom_before,oom_after) else None)
 health={"host_oom_kills_before":oom_before,"host_oom_kills_after":oom_after,"host_oom_kills_during_run":oom_delta,"valid":None if oom_delta is None else oom_delta == 0}
 (out/"measurement-health.json").write_text(json.dumps(health,indent=2)+"\n")
 if oom_delta:
  raise SystemExit(f"invalid benchmark: host oom_kill increased by {oom_delta} during Megatron execution; see {out/'measurement-health.json'}")
 if megatron_error is not None:
  raise megatron_error
 sim=out/"simulator-baseline.json"
 calc=[sys.executable,str(HERE/"run_calculon_case.py"),"--case",a.case,"--model-path",str(a.model_path),"--num-procs",str(world),"--tp",str(a.tp),"--pp",str(a.pp),"--dp",str(a.dp),"--ep",str(a.ep),"--etp",str(a.etp),"--edp",str(a.edp),"--cp",str(a.cp),"--optimization-strategy",a.optimization_strategy,"--lr",str(a.lr),"--min-lr",str(a.min_lr),"--warmup-fraction",str(a.warmup_fraction),"--weight-decay",str(a.weight_decay),"--clip-grad",str(a.clip_grad),"--precision",a.precision,"--system",str(a.system),"--seq-length",str(a.seq_length),"--global-batch",str(a.global_batch),"--micro-batch",str(a.micro_batch),"--output",str(sim)] + (["--sequence-parallel"] if a.sequence_parallel else []) + optimizer_calculon_args(a)
 run(calc,env,out/"simulator.log")
 sim_contract=json.loads(sim.read_text()).get("experiment_contract", {})
 sim_strategy=sim_contract.get("optimization", {}).get("canonical")
 if sim_strategy != a.optimization_strategy:
  raise RuntimeError(f"optimization strategy mismatch: Megatron={a.optimization_strategy}, Calculon={sim_strategy}")
 expected_hparams={"lr":a.lr,"min_lr":a.min_lr,"warmup_fraction":a.warmup_fraction,"weight_decay":a.weight_decay,"clip_grad":a.clip_grad,"precision":a.precision,**optimizer_contract(a)}
 if sim_contract.get("training_hyperparameters") != expected_hparams:
  raise RuntimeError("Megatron and Calculon training-hyperparameter contracts differ")
 measured=out/f"{a.case}.log"
 def compare(simulator_json, destination):
  destination.unlink(missing_ok=True)
  completed=subprocess.run([sys.executable,str(HERE/"compare_iteration_logs.py"),"--model",a.case,"--log",str(measured),"--simulator-json",str(simulator_json),"--warmup",str(a.warmup_samples),"--output",str(destination)],cwd=ROOT,env=env,check=False)
  if not destination.is_file():
   raise SystemExit(f"comparison failed before producing {destination}")
  comparison=json.loads(destination.read_text())
  if completed.returncode and comparison.get("comparison_valid") is not False:
   raise SystemExit(f"comparison failed with exit code {completed.returncode}; see {destination}")
  return comparison
 prediction=compare(sim,out/"prediction-error.json")
 contract={"case":a.case,"model_path":str(a.model_path),"dataset_path":str(a.dataset_path),"dataset_mode":dataset_mode,"simulator_system":str(a.system),"simulator_gpu":a.system.stem,"world_size":world,"tp":a.tp,"pp":a.pp,"dp":a.dp,"ep":a.ep,"etp":a.etp,"edp":a.edp,"cp":a.cp,"optimization_strategy":a.optimization_strategy,"precision":a.precision,"iterations":a.iterations,"seq_length":a.seq_length,"global_batch":a.global_batch,"micro_batch":a.micro_batch,"lr":a.lr,"min_lr":a.min_lr,"warmup_fraction":a.warmup_fraction,"weight_decay":a.weight_decay,"clip_grad":a.clip_grad,"optimizer":optimizer_contract(a),"extra_megatron_args":a.extra_megatron_arg}
 report={"contract":contract,"prediction":prediction}
 (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
 if not prediction.get("comparison_valid", False):
  reasons="; ".join(prediction.get("measurement_invalid_reasons", []))
  raise SystemExit(f"comparison report generated, but measurement is not stable enough for accuracy evaluation: {reasons}; see {out/'summary.json'}")
if __name__=="__main__":main()
