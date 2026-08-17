#!/usr/bin/env python3
"""Phase5: distributed DeepSeek-V3 MoE prediction-vs-measurement closure.

The default path is production-shaped: FP8 grouped experts, BF16 router/vector
work, 256 experts, top-k=8 and one shared expert.  ``dense-debug`` retains the
old low-memory BF16 test but is explicitly excluded from the MoE accuracy gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from calculon.llm.layers import (ElementWise, Linear, RouterPermutation,
                                 RouterSigmoid, RouterTopK)
from calculon.system import System
from calculon.llm.llm import Llm
from phase2_dsv3_op_catalog import default_exe_dict
from phase5_moe_runtime import (DeepSeekGroupedMoeRuntime, DenseDebugRuntime,
                                EventDAG, UtilizationSampler,
                                measure_distributed, read_hcu_utilization)


def relerr(pred, meas):
    return 100.0 * (pred - meas) / meas if meas > 0 else None


def layer_time(layer, stage):
    return float(layer.compute_processing_time(stage))


def formal_deepseek_prediction(args, world):
    """Compile the same Calculon Llm graph used by the service/API."""
    app_cfg = json.loads(args.model.read_text())
    app_cfg['seq_size'] = int(args.tokens)
    app_cfg['num_experts'] = int(args.experts)
    app_cfg['moe_topk'] = int(args.topk)
    app_cfg['num_shared_experts'] = int(args.shared_experts)
    app = Llm.Application(app_cfg)
    syst = System(json.loads(args.system.read_text()))
    exe_cfg = default_exe_dict('float8', 'bfloat16', 1, world)
    exe_cfg['num_procs'] = world
    exe_cfg['activation_recompute'] = args.activation_recompute
    exe = Llm.Execution.from_json(exe_cfg)
    llm = Llm(app, logging.getLogger('phase5'))
    llm.compile(syst, exe)
    layers = list(llm._moe_layers)
    router = [x for x in layers if x.name.startswith('MlpBlock_Router')]
    expert = [x for x in layers if x.name.startswith('MlpBlock_MoE_')]
    if not router or not expert:
        raise RuntimeError('compiled Calculon graph has no router/expert layers')

    permutation = [x for x in router if isinstance(x, RouterPermutation)]
    router_base = [x for x in router if not isinstance(x, RouterPermutation)]
    router_fw = sum(layer_time(x, 'fw') for x in router_base)
    permutation_fw = sum(layer_time(x, 'fw') for x in permutation)
    expert_fw = sum(layer_time(x, 'fw') for x in expert)
    router_agrad = sum(layer_time(x, 'agrad') for x in router_base + permutation)
    router_wgrad = sum(layer_time(x, 'wgrad') for x in router_base + permutation)
    expert_agrad = sum(layer_time(x, 'agrad') for x in expert)
    expert_wgrad = sum(layer_time(x, 'wgrad') for x in expert)
    router_recompute = 0.0
    expert_recompute = 0.0
    if args.activation_recompute == 'full':
        router_recompute = sum(layer_time(x, 'fw') for x in router
                               if x.needs_recompute)
        expert_recompute = sum(layer_time(x, 'fw') for x in expert
                               if x.needs_recompute)
    router_bw = router_agrad + router_wgrad + router_recompute
    expert_bw = expert_agrad + expert_wgrad + expert_recompute
    return syst, {
        'router_fw_s': router_fw,
        'router_permutation_fw_s': permutation_fw,
        'router_bw_s': router_bw,
        'expert_fw_s': expert_fw,
        'expert_bw_s': expert_bw,
        'expert_recompute_s': expert_recompute,
        'expert_agrad_s': expert_agrad,
        'expert_wgrad_s': expert_wgrad,
        'formal_layer_count': len(layers),
        'formal_router_layers': [x.name for x in router],
        'formal_expert_layers': [x.name for x in expert],
    }


def formal_dense_prediction(args, world):
    cfg = json.loads(args.system.read_text())
    syst = System(cfg)
    syst.set_datatypes(args.dense_dtype, args.dense_dtype)
    router_layers = [
        Linear('MlpBlock_Router', syst, args.tokens, args.hidden, world),
        RouterTopK('MlpBlock_RouterTopK', syst, args.tokens, 1, world),
    ]
    expert_layers = [
        Linear('DenseDebug_Gate', syst, args.tokens, args.hidden,
               args.ffn_hidden),
        Linear('DenseDebug_Up', syst, args.tokens, args.hidden,
               args.ffn_hidden),
        ElementWise('DenseDebug_GateUp', syst,
                    args.tokens * args.ffn_hidden,
                    args.tokens * args.ffn_hidden),
        Linear('DenseDebug_Down', syst, args.tokens, args.ffn_hidden,
               args.hidden),
    ]
    router_fw = sum(layer_time(x, 'fw') for x in router_layers)
    expert_fw = sum(layer_time(x, 'fw') for x in expert_layers)
    expert_agrad = sum(layer_time(x, 'agrad') for x in expert_layers)
    expert_wgrad = sum(layer_time(x, 'wgrad') for x in expert_layers)
    return syst, {
        'router_fw_s': router_fw, 'router_permutation_fw_s': 0.0,
        'router_bw_s': sum(layer_time(x, 'agrad') + layer_time(x, 'wgrad')
                           for x in router_layers),
        'expert_fw_s': expert_fw,
        'expert_bw_s': expert_agrad + expert_wgrad,
        'expert_recompute_s': 0.0,
        'expert_agrad_s': expert_agrad,
        'expert_wgrad_s': expert_wgrad,
        'formal_layer_count': len(router_layers) + len(expert_layers),
        'formal_router_layers': [x.name for x in router_layers],
        'formal_expert_layers': [x.name for x in expert_layers],
    }


def make_dags(formal, dispatch_s, combine_s, topk, world, deepseek=True):
    active = float(topk) / world + (1.0 if deepseek else 0.0)
    shared_fraction = (1.0 / active) if deepseek and active > 0 else 0.0
    routed_fraction = 1.0 - shared_fraction if deepseek else 1.0
    pack = formal['router_permutation_fw_s'] * .5

    fw = EventDAG()
    fw.add('router', formal['router_fw_s'] + pack, 'compute')
    fw.add('ep_dispatch', dispatch_s, 'network', ('router',))
    if shared_fraction:
        fw.add('shared_expert', formal['expert_fw_s'] * shared_fraction,
               'compute', ('router',))
    fw.add('pack', pack, 'compute', ('ep_dispatch',))
    fw.add('routed_expert', formal['expert_fw_s'] * routed_fraction,
           'compute', ('pack',))
    fw.add('unpack', pack, 'compute', ('routed_expert',))
    fw.add('ep_combine', combine_s, 'network', ('unpack',))
    deps = ('ep_combine', 'shared_expert') if shared_fraction else ('ep_combine',)
    fw.add('forward_done', 0.0, 'barrier', deps)
    fw_total, fw_timeline = fw.schedule()

    bw = EventDAG()
    bw.add('ep_combine_bwd', combine_s, 'network')
    if shared_fraction:
        bw.add('shared_expert_bwd', formal['expert_bw_s'] * shared_fraction,
               'compute')
    bw.add('unpack_grad', pack, 'compute', ('ep_combine_bwd',))
    bw.add('routed_expert_bwd', formal['expert_bw_s'] * routed_fraction,
           'compute', ('unpack_grad',))
    bw.add('pack_grad', pack, 'compute', ('routed_expert_bwd',))
    bw.add('ep_dispatch_bwd', dispatch_s, 'network', ('pack_grad',))
    deps = ('ep_dispatch_bwd', 'shared_expert_bwd') if shared_fraction else (
        'ep_dispatch_bwd',)
    bw.add('router_bwd', formal['router_bw_s'], 'compute', deps)
    bw_total, bw_timeline = bw.schedule()

    serial_fw_s = (formal['router_fw_s'] + 2 * pack + dispatch_s + combine_s
                   + formal['expert_fw_s'])
    return {
        'forward_s': fw_total, 'backward_s': bw_total,
        'serial_forward_s': serial_fw_s,
        'forward_timeline': fw_timeline,
        'backward_timeline': bw_timeline,
    }


def run_flow_rank0(syst, net_index, world, formal, dispatch_bytes,
                   combine_bytes):
    net = syst.get_network(net_index)
    ep_bw = net.collective_bandwidth(
        'all_to_all', max(dispatch_bytes, combine_bytes))
    ep_latency = net.collective_latency('all_to_all', world)
    net.flow_network_init(
        net.flow_bandwidth('tp'), net.flow_bandwidth('cp'), ep_bw,
        net.flow_bandwidth('pp'), net.flow_bandwidth('dp'),
        net._topology, net.flow_latency('tp'), net.flow_latency('cp'),
        ep_latency, net.flow_latency('pp'), net.flow_latency('dp'))
    flow = net.total_flow_network_time(
        1, 1, 1, 0.0, 0.0, 1, 0, 0, 0, 0, 0, True,
        ep=world, fwd_mla_time=formal['router_fw_s'],
        fwd_ffn_time=formal['expert_fw_s'],
        bwd_mla_time=formal['router_bw_s'],
        bwd_ffn_time=formal['expert_bw_s'],
        fwd_ep_dispatch_size=dispatch_bytes,
        fwd_ep_combine_size=combine_bytes,
        bwd_ep_dispatch_size=dispatch_bytes,
        bwd_ep_combine_size=combine_bytes)
    global_s = float(flow[0])
    if max(dispatch_bytes, combine_bytes) > 0 and global_s <= 0:
        raise RuntimeError('LLMFlowSimulator returned zero for non-zero EP bytes')
    return {
        'global_s': global_s, 'total_comm_s': float(flow[12]),
        'batch_ep_fw_s': float(flow[19]),
        'batch_ep_bw_s': float(flow[20]),
        'batch_ep_s': float(flow[21]),
        'timeline_event_count': int(flow[13]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('deepseek-moe', 'dense-debug'),
                   default='deepseek-moe')
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--hidden', type=int, default=7168)
    p.add_argument('--ffn-hidden', type=int, default=2048)
    p.add_argument('--experts', type=int, default=256)
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--shared-experts', type=int, default=1)
    p.add_argument('--dense-dtype', choices=('float16', 'bfloat16'),
                   default='bfloat16')
    p.add_argument('--dtype', dest='dense_dtype',
                   choices=('float16', 'bfloat16'), help=argparse.SUPPRESS)
    p.add_argument('--activation-recompute', choices=('none', 'full'),
                   default='full')
    p.add_argument('--measure-wgrad', action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--max-cv-pct', type=float, default=5.0)
    p.add_argument('--max-preexisting-util-pct', type=float, default=5.0)
    p.add_argument('--gate-phase0-pct', type=float, default=15.0)
    p.add_argument('--gate-phase2-pct', type=float, default=20.0)
    p.add_argument('--gate-phase3-pct', type=float, default=20.0)
    p.add_argument('--gate-phase4-pct', type=float, default=20.0)
    p.add_argument('--gate-phase5-pct', type=float, default=20.0)
    p.add_argument('--gate-component-pct', type=float, help=argparse.SUPPRESS)
    p.add_argument('--gate-core-pct', type=float, help=argparse.SUPPRESS)
    p.add_argument('--system', type=Path,
                   default=ROOT/'systems'/'BW1100.json')
    p.add_argument('--model', type=Path,
                   default=ROOT/'models'/'deepseek-v3-671b.json')
    p.add_argument('--network-index', type=int, default=0)
    p.add_argument('--output', type=Path,
                   default=ROOT/'test'/'bw1100'/'phase5_distributed_moe_quick.json')
    p.add_argument('--csv', type=Path,
                   default=ROOT/'test'/'bw1100'/'phase5_distributed_moe_quick.csv')
    p.add_argument(
        '--strict-exit-code', action='store_true',
        help=('return exit code 1 when an accuracy or measurement-quality '
              'gate fails; by default a completed validation exits 0 and '
              'records the gate result in JSON/CSV'))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.gate_component_pct is not None:
        args.gate_phase2_pct = args.gate_phase3_pct = args.gate_phase4_pct = (
            args.gate_component_pct)
    if args.gate_core_pct is not None:
        args.gate_phase5_pct = args.gate_core_pct
    if args.dry_run:
        print(json.dumps({
            'mode': args.mode, 'world_size': '2-4',
            'matrix_dtype': 'float8' if args.mode == 'deepseek-moe'
                            else args.dense_dtype,
            'vector_dtype': 'bfloat16', 'experts': args.experts,
            'topk': args.topk, 'shared_experts': args.shared_experts,
            'prediction_path': 'Calculon Llm/System + LLMFlowSimulator C++',
            'quality_gate': {'max_cv_pct': args.max_cv_pct,
                             'max_preexisting_util_pct':
                                 args.max_preexisting_util_pct},
        }, indent=2))
        return

    if args.mode == 'deepseek-moe':
        if args.shared_experts != 1:
            p.error('the DeepSeek-V3 quick runtime currently requires '
                    '--shared-experts 1')
        if args.experts != 256 or args.topk != 8:
            p.error('production Phase5 requires --experts 256 --topk 8; '
                    'use dense-debug for non-production unit tests')

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    if not 2 <= world <= 4:
        raise SystemExit(f'Phase5 requires 2-4 ranks, got {world}')
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device('cuda', local_rank)
    physical = [int(v) for v in os.environ.get(
        'PHASE5_PHYSICAL_DEVICES', ','.join(map(str, range(world)))).split(',')]
    pre_util = read_hcu_utilization(physical) if rank == 0 else None
    holder = [pre_util]
    dist.broadcast_object_list(holder, src=0)
    pre_util = holder[0]
    preexisting_busy = any(v is not None and v > args.max_preexisting_util_pct
                           for v in pre_util.values())
    free, total = torch.cuda.mem_get_info(dev)
    memory_before = {'free_GiB': free/1024**3, 'total_GiB': total/1024**3}
    if args.mode == 'deepseek-moe':
        runtime = DeepSeekGroupedMoeRuntime(
            args.tokens, args.hidden, args.ffn_hidden, args.experts, args.topk,
            args.shared_experts, dev, args.measure_wgrad)
        syst, formal = formal_deepseek_prediction(args, world)
    else:
        dtype = {'float16': torch.float16,
                 'bfloat16': torch.bfloat16}[args.dense_dtype]
        runtime = DenseDebugRuntime(args.tokens, args.hidden, args.ffn_hidden,
                                    dev, dtype)
        syst, formal = formal_dense_prediction(args, world)

    dispatch_bytes, combine_bytes = runtime.communication_bytes()
    imbalance = runtime.imbalance()
    net = syst.get_network(args.network_index)
    dispatch_s = net.collective_time(
        'all_to_all', dispatch_bytes, world, bottleneck_bytes=dispatch_bytes)
    combine_s = net.collective_time(
        'all_to_all', combine_bytes, world, bottleneck_bytes=combine_bytes)
    dags = make_dags(formal, dispatch_s, combine_s, args.topk, world,
                     args.mode == 'deepseek-moe')

    sampler = UtilizationSampler(physical) if rank == 0 else None
    if sampler:
        sampler.__enter__()
    measurements = {}
    fns = [
        ('router', runtime.router),
        ('ep_dispatch', runtime.dispatch),
        ('expert_forward', runtime.expert_forward),
        ('expert_backward', runtime.expert_backward),
        ('router_backward', runtime.router_backward),
        ('forward_core', runtime.forward_core),
        ('backward_core', runtime.backward_core),
        ('train_core', runtime.train_core),
        ('overlap_serial', runtime.forward_core_serial),
        ('overlap_async', runtime.forward_core),
    ]
    if args.mode == 'deepseek-moe':
        fns[2:2] = [
            ('packing', runtime.pack),
            ('expert_recompute', runtime.expert_recompute),
            ('expert_agrad', runtime.expert_agrad),
            ('expert_wgrad', runtime.expert_wgrad),
        ]
    try:
        for name, fn in fns:
            measurements[name] = measure_distributed(
                fn, args.warmup, args.iters, dev)
    finally:
        if sampler:
            sampler.__exit__(None, None, None)

    flow_holder = [None]
    if rank == 0:
        try:
            flow_holder[0] = {'result': run_flow_rank0(
                syst, args.network_index, world, formal,
                dispatch_bytes, combine_bytes), 'error': None}
        except Exception as exc:
            flow_holder[0] = {'result': None, 'error': repr(exc)}
    dist.broadcast_object_list(flow_holder, src=0)
    flow_status = flow_holder[0]

    pred = {
        'router': (formal['router_fw_s'] +
                   .5 * formal['router_permutation_fw_s']) * 1e3,
        'ep_dispatch': dispatch_s * 1e3,
        'expert_forward': formal['expert_fw_s'] * 1e3,
        'expert_backward': formal['expert_bw_s'] * 1e3,
        'router_backward': formal['router_bw_s'] * 1e3,
        'forward_core': dags['forward_s'] * 1e3,
        'backward_core': dags['backward_s'] * 1e3,
        'overlap_serial': dags['serial_forward_s'] * 1e3,
        'overlap_async': dags['forward_s'] * 1e3,
    }
    if 'packing' in measurements:
        pred['packing'] = .5 * formal['router_permutation_fw_s'] * 1e3
        pred['expert_recompute'] = formal['expert_recompute_s'] * 1e3
        pred['expert_agrad'] = formal['expert_agrad_s'] * 1e3
        pred['expert_wgrad'] = formal['expert_wgrad_s'] * 1e3
    pred['train_core'] = (flow_status['result']['global_s'] * 1e3
                          if flow_status['result'] else math.nan)

    phase_map = {
        'router': ('phase2', args.gate_phase2_pct),
        'router_backward': ('phase2', args.gate_phase2_pct),
        'packing': ('phase2', args.gate_phase2_pct),
        'expert_forward': ('phase3', args.gate_phase3_pct),
        'expert_recompute': ('phase3', args.gate_phase3_pct),
        'expert_agrad': ('phase3', args.gate_phase3_pct),
        'expert_wgrad': ('phase3', args.gate_phase3_pct),
        'expert_backward': ('phase3', args.gate_phase3_pct),
        'ep_dispatch': ('phase4', args.gate_phase4_pct),
        'forward_core': ('phase5', args.gate_phase5_pct),
        'backward_core': ('phase5', args.gate_phase5_pct),
        'train_core': ('phase5', args.gate_phase5_pct),
        'overlap_serial': ('phase5-diagnostic', args.gate_phase5_pct),
        'overlap_async': ('phase5-diagnostic', args.gate_phase5_pct),
    }
    rows = []
    for name, stats in measurements.items():
        measured_ms = stats['measured_ms']
        predicted_ms = pred[name]
        error = relerr(predicted_ms, measured_ms)
        phase, limit = phase_map[name]
        gated = name not in ('overlap_serial', 'overlap_async')
        passed = (math.isfinite(predicted_ms) and abs(error) <= limit
                  and stats['max_rank_cv_pct'] <= args.max_cv_pct)
        rows.append({
            'component': name, 'layer': phase,
            'measured_ms': measured_ms, 'predicted_ms': predicted_ms,
            'error_pct': error, 'max_rank_cv_pct': stats['max_rank_cv_pct'],
            'p10_ms': stats['global_p10_ms'],
            'p90_ms': stats['global_p90_ms'],
            'gate_pct': limit if gated else '',
            'pass': passed if gated else '',
        })
    checks = {r['component']: bool(r['pass']) for r in rows
              if r['pass'] != ''}
    quality = {
        'preexisting_hcu_utilization_pct': pre_util,
        'preexisting_gpu_busy': preexisting_busy,
        'max_allowed_preexisting_util_pct': args.max_preexisting_util_pct,
        'max_allowed_cv_pct': args.max_cv_pct,
        'all_measurements_stable': all(
            x['max_rank_cv_pct'] <= args.max_cv_pct
            for x in measurements.values()),
        'utilization_during_test': sampler.summary() if sampler else None,
        'memory_before_allocation': memory_before,
        'per_rank_statistics': {k: v['per_rank']
                                for k, v in measurements.items()},
    }
    quality['valid'] = (not preexisting_busy and
                        quality['all_measurements_stable'])
    deepseek_complete = (args.mode == 'deepseek-moe' and args.measure_wgrad)
    result = {
        'schema': 'bw1100.phase5.distributed_moe_quick.v2',
        'mode': args.mode, 'world_size': world,
        'backend': dist.get_backend(), 'matrix_dtype': (
            'float8' if args.mode == 'deepseek-moe' else args.dense_dtype),
        'vector_dtype': 'bfloat16' if args.mode == 'deepseek-moe'
                        else args.dense_dtype,
        'shape': {
            'tokens_per_rank': args.tokens, 'hidden': args.hidden,
            'ffn_hidden': args.ffn_hidden, 'num_experts': args.experts,
            'topk': args.topk, 'shared_experts': args.shared_experts,
            'experts_per_rank': (args.experts // world
                                 if args.mode == 'deepseek-moe' else 1),
            'token_imbalance': imbalance,
        },
        'runtime_backend': runtime.backend,
        'prediction_source': {
            'calculon': formal,
            'llm_flow': flow_status,
            'event_dag': dags,
        },
        'network_model': {
            'tier': args.network_index,
            'physical_link_bandwidth_GBps': net._bw / 1e9,
            'physical_link_efficiency': net._eff,
            'physical_link_latency_us': net._latency * 1e6,
            'collective': 'all_to_all',
            'collective_bandwidth_GBps':
                net.collective_bandwidth(
                    'all_to_all', max(dispatch_bytes, combine_bytes)) / 1e9,
            'collective_latency_us':
                net.collective_latency('all_to_all', world) * 1e6,
            'participants': world, 'dispatch_bytes': dispatch_bytes,
            'combine_bytes': combine_bytes,
        },
        'rows': rows,
        'quality': quality,
        'layered_acceptance': {
            'phase0_kernel_curves_pct': args.gate_phase0_pct,
            'phase2_operator_pct': args.gate_phase2_pct,
            'phase3_grouped_fused_pct': args.gate_phase3_pct,
            'phase4_communication_pct': args.gate_phase4_pct,
            'phase5_distributed_core_pct': args.gate_phase5_pct,
        },
        'checks': checks,
        'deepseek_validation_complete': deepseek_complete,
        'pass': (args.mode == 'deepseek-moe' and deepseek_complete and
                 quality['valid'] and flow_status['error'] is None and
                 all(checks.values())),
        'scope': {
            'validated': ['router fw/bwd', 'FP8 grouped expert fw/agrad/wgrad',
                          'EP dispatch/combine', 'event-DAG composition',
                          'C++ LLMFlow end-to-end closure', 'overlap'],
            'debug_only': ['dense-debug'],
            'not_covered': ['optimizer', 'PP pipeline bubble',
                            'full-model extrapolation'],
        },
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + '.tmp')
        tmp.write_text(json.dumps(result, indent=2) + '\n')
        os.replace(tmp, args.output)
        with args.csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(json.dumps(result, indent=2), flush=True)
        if not result['pass']:
            failed = [row['component'] for row in rows
                      if row.get('pass') is False]
            print(
                'PHASE5_VALIDATION_NOT_ACCEPTED: measurement completed, but '
                f'quality/accuracy gates failed: {failed}. See quality and '
                'rows in the JSON result.',
                file=sys.stderr, flush=True)
    dist.barrier()
    dist.destroy_process_group()
    # A validation rejection is a valid experimental result, not a process
    # failure.  Keep strict CI behavior available explicitly without making
    # torchrun report the completed experiment as ChildFailedError.
    raise SystemExit(1 if args.strict_exit_code and not result['pass'] else 0)


if __name__ == '__main__':
    main()
