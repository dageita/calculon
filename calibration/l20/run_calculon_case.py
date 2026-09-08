#!/usr/bin/env python3
"""Run the Calculon side of a shared local-HF model contract."""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path('/src/Simulator/calculon')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calculon.llm import Llm
from calculon.llm.runner import Runner
from calculon.system import System
from model_catalog import (CASE_CHOICES, application_from_hf,
                           canonical_case, default_model_path)

p = argparse.ArgumentParser()
p.add_argument('--case', choices=CASE_CHOICES, required=True)
p.add_argument('--model-path', type=Path)
p.add_argument('--seq-length', type=int, default=1024)
p.add_argument('--global-batch', type=int, default=8)
p.add_argument('--micro-batch', type=int, default=1)
p.add_argument('--num-procs', type=int, default=4)
p.add_argument('--tp', type=int, default=1)
p.add_argument('--pp', type=int, default=1)
p.add_argument('--dp', type=int)
p.add_argument('--ep', type=int)
p.add_argument('--etp', type=int)
p.add_argument('--edp', type=int)
p.add_argument('--cp', type=int, default=1)
p.add_argument('--optimization-strategy',
               choices=('none', 'attention-only', 'full'), default='none')
p.add_argument('--sequence-parallel', action=argparse.BooleanOptionalAction,
               default=False)
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--min-lr', type=float, default=1e-5)
p.add_argument('--warmup-fraction', type=float, default=.01)
p.add_argument('--weight-decay', type=float, default=.1)
p.add_argument('--clip-grad', type=float, default=1.)
p.add_argument('--precision', choices=('bf16', 'fp16'), default='bf16')
p.add_argument('--use-distributed-optimizer', action='store_true')
p.add_argument('--use-precision-aware-optimizer', action='store_true')
p.add_argument('--main-grads-dtype', choices=('fp32', 'bf16'), default='fp32')
p.add_argument('--main-params-dtype', choices=('fp32', 'fp16'), default='fp32')
p.add_argument('--exp-avg-dtype', choices=('fp32', 'fp16', 'fp8'), default='fp32')
p.add_argument('--exp-avg-sq-dtype', choices=('fp32', 'fp16', 'fp8'), default='fp32')
p.add_argument('--grad-reduce-in-bf16', action='store_true')
p.add_argument('--optimizer-cpu-offload', action='store_true')
p.add_argument('--optimizer-offload-fraction', type=float, default=1.0)
p.add_argument('--use-torch-optimizer-for-cpu-offload', action='store_true')
p.add_argument('--overlap-cpu-optimizer-d2h-h2d', action='store_true')
p.add_argument('--pin-cpu-grads', action=argparse.BooleanOptionalAction,
               default=True)
p.add_argument('--pin-cpu-params', action=argparse.BooleanOptionalAction,
               default=True)
p.add_argument('--system', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if not 0 <= a.optimizer_offload_fraction <= 1:
    p.error('--optimizer-offload-fraction must be in [0, 1]')
if a.use_precision_aware_optimizer and not a.use_distributed_optimizer:
    p.error('--use-precision-aware-optimizer requires --use-distributed-optimizer')
if a.optimizer_cpu_offload and not a.use_precision_aware_optimizer:
    p.error('--optimizer-cpu-offload requires --use-precision-aware-optimizer')
if a.grad_reduce_in_bf16 and a.precision != 'bf16':
    p.error('--grad-reduce-in-bf16 requires --precision bf16')
case = canonical_case(a.case)
model_path = a.model_path or default_model_path(case)
try:
    app = application_from_hf(case, model_path, a.seq_length)
except (KeyError, ValueError, json.JSONDecodeError) as exc:
    p.error(f'invalid model contract for {model_path}: {exc}')

strategy = {'none': 'none', 'attention-only': 'attn_only',
            'full': 'full'}[a.optimization_strategy]
is_moe = bool(app.get('num_experts'))
dp = a.dp if a.dp is not None else a.num_procs // (a.tp * a.pp * a.cp)
if a.num_procs != a.tp * a.pp * a.cp * dp:
    raise SystemExit('num_procs must equal tp*pp*cp*dp')
if a.cp < 1:
    raise SystemExit('cp must be positive')
if not is_moe:
    # Dense execution has no expert rank generator. EP/ETP/EDP are accepted as
    # inert compatibility arguments; use a neutral internal shape for Execution.
    ep = etp = 1
    edp = a.num_procs // a.pp
else:
    ep = a.ep if a.ep is not None else 1
    etp = a.etp if a.etp is not None else a.tp
    if app['num_experts'] % ep:
        p.error('model num_experts must be divisible by --ep')
    expert_factor = etp * ep * a.pp
    if a.num_procs % expert_factor:
        raise SystemExit('ETP*EP*PP must divide num_procs')
    edp = a.num_procs // expert_factor
dtype = 'bfloat16' if a.precision == 'bf16' else 'float16'
exe = dict(
    num_procs=a.num_procs, tensor_par=a.tp, pipeline_par=a.pp, data_par=dp,
    tensor_par_net=0, pipeline_par_net=0, data_par_net=0, expert_par=ep,
    expert_tensor_par=etp, expert_data_par=edp, context_par=a.cp, expert_par_net=0, context_par_net=0,
    batch_size=a.global_batch, microbatch_size=a.micro_batch, datatype=dtype,
    matrix_dtype=dtype, vector_dtype=dtype, fused_activation=True,
    attention_type='mla' if app.get('kv_lora_rank') else 'multihead',
    attention_kernel='flash', activation_recompute=strategy,
    pipeline_interleaving=1,
    optimizer_sharding=a.use_distributed_optimizer,
    tensor_par_comm_type='rs_ag' if a.sequence_parallel else 'ar',
    tensor_par_overlap='none', seq_par_ag_redo=False,
    data_par_overlap=False, weight_offload=False, activations_offload=False,
    optimizer_offload=a.optimizer_cpu_offload, training=True,
    use_precision_aware_optimizer=a.use_precision_aware_optimizer,
    main_grads_dtype=a.main_grads_dtype,
    main_params_dtype=a.main_params_dtype,
    exp_avg_dtype=a.exp_avg_dtype, exp_avg_sq_dtype=a.exp_avg_sq_dtype,
    grad_reduce_in_bf16=a.grad_reduce_in_bf16,
    optimizer_offload_fraction=a.optimizer_offload_fraction,
    use_torch_optimizer_for_cpu_offload=(
        a.use_torch_optimizer_for_cpu_offload),
    overlap_cpu_optimizer_d2h_h2d=a.overlap_cpu_optimizer_d2h_h2d,
    pin_cpu_grads=a.pin_cpu_grads, pin_cpu_params=a.pin_cpu_params)
log = logging.getLogger('l20-case')
model = Llm(Llm.Application(app), log)
system_path = a.system or ROOT / f'systems/L20_{a.num_procs}GPU.json'
if not system_path.exists():
    system_path = ROOT / 'systems/L20.json'
system = System(json.loads(system_path.read_text()), log)
model.compile(system, Llm.Execution.from_json(exe))
model.run(system)
result = Runner.get_simulator_res_json(model)
result['experiment_contract'] = {
    'case': case, 'requested_case': a.case, 'parallelism': {'tp': a.tp, 'pp': a.pp, 'dp': dp, 'cp': a.cp, 'etp': etp, 'ep': ep, 'edp': edp}, 'model_path': str(model_path),
    'application': app, 'execution': exe,
    'architecture_approximation': bool(app.get('architecture_approximation')),
    'optimization': {
        'canonical': a.optimization_strategy,
        'calculon_activation_recompute': strategy,
        'megatron_recompute_flags': {
            'none': [],
            'attention-only': ['--recompute-granularity', 'selective',
                               '--recompute-modules', 'core_attn'],
            'full': ['--recompute-granularity', 'full', '--recompute-method',
                     'uniform', '--recompute-num-layers', '1'],
        }[a.optimization_strategy],
    },
    'training_hyperparameters': {
        'lr': a.lr, 'min_lr': a.min_lr,
        'warmup_fraction': a.warmup_fraction,
        'weight_decay': a.weight_decay, 'clip_grad': a.clip_grad,
        'precision': a.precision,
        'optimizer_sharding': a.use_distributed_optimizer,
        'use_precision_aware_optimizer': a.use_precision_aware_optimizer,
        'main_grads_dtype': a.main_grads_dtype,
        'main_params_dtype': a.main_params_dtype,
        'exp_avg_dtype': a.exp_avg_dtype,
        'exp_avg_sq_dtype': a.exp_avg_sq_dtype,
        'grad_reduce_in_bf16': a.grad_reduce_in_bf16,
        'optimizer_offload': a.optimizer_cpu_offload,
        'optimizer_offload_fraction': a.optimizer_offload_fraction,
        'use_torch_optimizer_for_cpu_offload':
            a.use_torch_optimizer_for_cpu_offload,
        'overlap_cpu_optimizer_d2h_h2d':
            a.overlap_cpu_optimizer_d2h_h2d,
        'pin_cpu_grads': a.pin_cpu_grads,
        'pin_cpu_params': a.pin_cpu_params,
    },
    'megatron_flags': {
        'gradient_accumulation_fusion': False,
        'transformer_impl': 'transformer_engine',
        'use_distributed_optimizer': a.use_distributed_optimizer,
        'use_precision_aware_optimizer': a.use_precision_aware_optimizer,
        'optimizer_cpu_offload': a.optimizer_cpu_offload,
        'sequence_parallel': a.sequence_parallel,
        'moe_kernel_path': ('grouped-gemm' if is_moe else None),
    },
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['summary'], indent=2))
