#!/usr/bin/env python3
"""Phase3 Step B — MoE weight split: compute track vs mem×257 track.

EP=1 → Calculon weight_multiplier≈257 (full expert residency) while
flop_multiplier≈9 (active compute). Single H20 cannot load 257 experts;
profiling MUST split:

  Compute track:  keep 1 (or topk) weight copy; time ≈ 9× expert GEMM
                  compare to Calculon flop_mult=9 pred_f / pred_max(compute)
  Mem track:      Calculon mem_time with weight_mult=257
                  vs measured / estimated active-expert HBM traffic only
                  → report gap; do NOT gate <20%

Example:
  python test/phase3_stepB_moe_weight_split.py --predict-only --seq-size 1024
  python test/phase3_stepB_moe_weight_split.py --seq-size 1024 \\
      --phase2-csv test/phase2_g4g5.csv --csv test/phase3_stepB.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TEST_DIR)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase2_dsv3_op_catalog import compile_dsv3  # noqa: E402
from phase3_dsv3_common import (  # noqa: E402
    banner,
    collect_layer_times,
    load_phase2_meas_index,
    mape,
    rel_err_pct,
    write_csv,
)


# MoE expert path layers (abstract Linear with flop_mult / weight_mult).
_EXPERT_KEYS = ('MlpBlock_MoE_Gate', 'MlpBlock_MoE_Up', 'MlpBlock_MoE_Down')
_ROUTER_KEY = 'MlpBlock_Router'


@dataclass
class SplitRow:
    seq_size: int
    name: str
    cls: str
    stage: str
    flop_mult: float
    weight_mult: float
    # Calculon as compiled (257 weights, 9 flops)
    pred_f_s: float
    pred_m_s: float
    pred_max_s: float
    bound: str
    # Synthetic: mem if only active_equiv weight copies resided
    pred_m_active_s: float
    pred_max_active_s: float
    activation_bytes: float
    shared_expert_weight_bytes: float
    routed_expert_distinct_weight_bytes: float
    workspace_router_bytes: float
    total_decomposed_bytes: float
    distinct_routed_experts: int
    grouped_meas_s: Optional[float]
    err_grouped_pct: Optional[float]
    grouped_comparable: bool
    grouped_backend: str
    # Ratio mem_257 / mem_active
    mem_inflate_x: float
    # Phase2 physical meas (9× expert) if available
    meas_compute_s: Optional[float]
    err_compute_pct: Optional[float]
    track: str
    notes: str


def _is_expert(name: str) -> bool:
    return any(k in name for k in _EXPERT_KEYS)


def _is_router(name: str) -> bool:
    return _ROUTER_KEY in name


def _active_equiv(app) -> float:
    return float(getattr(app, 'moe_topk', 8) + getattr(app, 'num_shared_experts', 1))


def synthesize_active_mem(
    pred_m_s: float, weight_mult: float, active_w: float,
) -> float:
    """Scale mem_time from weight_mult → active_w (activations unchanged).

    Conservative: scale total mem by active_w/weight_mult. Slightly
    underestimates activation share but matches Phase3 design intent
    (expose 257 vs active gap).
    """
    if weight_mult <= 0:
        return pred_m_s
    return pred_m_s * (active_w / weight_mult)


def decompose_moe_bytes(r, app, stage: str, distinct_routed: int):
    """Separate traffic from storage multipliers using Linear dimensions."""
    m, ci, co = r._dims
    elements = m * ci + m * co + ci * co * r.weight_mult
    bpe = r.bytes / elements if elements else 0.0
    activation = (m * ci + m * co) * bpe * r.flop_mult
    one_weight = ci * co * bpe
    shared_n = int(getattr(app, 'num_shared_experts', 1))
    shared_weight = shared_n * one_weight
    routed_weight = distinct_routed * one_weight
    # Router/workspace is reported separately rather than hidden in expert
    # weight traffic. It is filled by the router row at block level.
    return activation, shared_weight, routed_weight, 0.0


def run(llm, app, args, meas_idx, catalog_dims, grouped=None) -> List[SplitRow]:
    stages = ['fw'] if args.stage == 'fw' else (
        ['fw', 'agrad', 'wgrad'] if args.stage == 'all' else [args.stage])
    rows_lt = collect_layer_times(llm, 'moe', stages=stages)
    active_w = _active_equiv(app)
    out: List[SplitRow] = []
    grouped_meta = next((v[1] for v in (grouped or {}).values()
                         if v and len(v) > 1 and v[1]), {})
    requested_routed = int(grouped_meta.get(
        'routed_experts', getattr(app, 'num_experts', 256)))
    distinct_routed = min(requested_routed,
                          int(app.seq_size * getattr(app, 'moe_topk', 8)))
    router_workspace = sum(
        r.bytes for r in rows_lt if _is_router(r.name) and r.stage == 'fw')

    for r in rows_lt:
        if not (_is_expert(r.name) or _is_router(r.name)):
            continue
        if _is_router(r.name):
            track = 'router_skinny'
            # Router is not 257-weight; report compute only.
            pred_m_act = r.pred_m_s
            inflate = 1.0
            notes = 'Router: not subject to weight_mult=257; skinny GEMM residual'
        else:
            track = 'expert_moe'
            pred_m_act = synthesize_active_mem(
                r.pred_m_s, r.weight_mult, active_w)
            inflate = (r.pred_m_s / pred_m_act) if pred_m_act > 0 else float('nan')
            notes = (
                f'compute≡flop_mult={r.flop_mult}; '
                f'mem_calc≡weight_mult={r.weight_mult}; '
                f'mem_active_synth≡{active_w}'
            )

        dims = catalog_dims.get((r.name, r.stage))
        if dims and _is_expert(r.name):
            r._dims = dims
            act_b, shared_b, routed_b, work_b = decompose_moe_bytes(
                r, app, r.stage, distinct_routed)
            work_b = 0.0
        else:
            act_b, shared_b, routed_b = r.bytes, 0.0, 0.0
            work_b = router_workspace if _is_router(r.name) else 0.0

        pred_max_act = max(r.pred_f_s, pred_m_act)
        meas = meas_idx.get((r.name, r.stage))
        # Compute-track: active-resident roofline vs Phase2 physical×9.
        # (Do NOT use pred_max with weight_mult=257 — that is mem track.)
        if _is_expert(r.name):
            ref_pred = pred_max_act
        else:
            ref_pred = r.pred_max_s
        err = rel_err_pct(ref_pred, meas)

        out.append(SplitRow(
            seq_size=int(app.seq_size),
            name=r.name, cls=r.cls, stage=r.stage,
            flop_mult=r.flop_mult, weight_mult=r.weight_mult,
            pred_f_s=r.pred_f_s, pred_m_s=r.pred_m_s,
            pred_max_s=r.pred_max_s, bound=r.bound,
            pred_m_active_s=pred_m_act, pred_max_active_s=pred_max_act,
            activation_bytes=act_b,
            shared_expert_weight_bytes=shared_b,
            routed_expert_distinct_weight_bytes=routed_b,
            workspace_router_bytes=work_b,
            total_decomposed_bytes=act_b + shared_b + routed_b + work_b,
            distinct_routed_experts=distinct_routed,
            grouped_meas_s=(grouped.get(r.name, (None, None))[0]
                            if grouped and _is_expert(r.name)
                            and r.stage == 'fw' else None),
            err_grouped_pct=(rel_err_pct(
                r.pred_max_s, grouped.get(r.name, (None, None))[0])
                if grouped and _is_expert(r.name) and r.stage == 'fw'
                else None),
            grouped_comparable=bool(
                grouped and _is_expert(r.name) and r.stage == 'fw'
                and grouped.get(r.name, (None, None))[0] is not None
                and requested_routed == int(getattr(app, 'num_experts', 256))),
            grouped_backend=(
                grouped.get(r.name, (None, {}))[1].get('backend', '')
                if grouped and _is_expert(r.name) else ''),
            mem_inflate_x=inflate,
            meas_compute_s=meas, err_compute_pct=err,
            track=track, notes=notes,
        ))
    return out


def print_report(rows: Sequence[SplitRow], active_w: float) -> None:
    print('\n--- Step B summary (MoE weight split) ---')
    print(f'active_equiv (topk+shared) = {active_w}')
    print(f'{"track":14} {"name":28} {"wm":>5} {"fm":>4} '
          f'{"pred_f":>10} {"pred_m257":>10} {"pred_mAct":>10} '
          f'{"inflX":>6} {"meas":>10} {"err%":>7}')
    for r in rows:
        def fmt(x):
            return f'{x:10.4e}' if x is not None else f'{"—":>10}'

        err = f'{r.err_compute_pct:7.1f}' if r.err_compute_pct is not None else f'{"—":>7}'
        print(
            f'{r.track:14} {r.name[-28:]:28} {r.weight_mult:5.0f} {r.flop_mult:4.0f} '
            f'{fmt(r.pred_f_s)} {fmt(r.pred_m_s)} {fmt(r.pred_m_active_s)} '
            f'{r.mem_inflate_x:6.1f} {fmt(r.meas_compute_s)} {err}'
        )

    experts = [r for r in rows if r.track == 'expert_moe' and r.stage == 'fw']
    if experts:
        print('\nMoE byte decomposition (per forward expert projection):')
        for r in experts:
            gib = 1024.0 ** 3
            print(f'  {r.name}: activation={r.activation_bytes/gib:.3f} GiB; '
                  f'shared_weight={r.shared_expert_weight_bytes/gib:.3f} GiB; '
                  f'routed_distinct_weight={r.routed_expert_distinct_weight_bytes/gib:.3f} GiB; '
                  f'workspace/router={r.workspace_router_bytes/gib:.3f} GiB; '
                  f'distinct_routed={r.distinct_routed_experts}')
            if r.grouped_meas_s is not None:
                print(f'    grouped={r.grouped_meas_s*1e3:.4f} ms; '
                      f'pred_err={r.err_grouped_pct:+.1f}%; '
                      f'backend={r.grouped_backend}')
        infl = sum(r.mem_inflate_x for r in experts) / len(experts)
        print(f'\nMem inflate (257 / active≈{active_w}): ~{infl:.1f}×  '
              f'→ report-only, NOT a <20% gate')
        cerrs = [r.err_compute_pct for r in experts]
        m = mape(cerrs)
        if m is not None:
            gate = 'PASS' if m < 20.0 else 'REVIEW'
            print(f'Compute-track MAPE vs Phase2 physical: {m:.1f}%  → {gate} '
                  f'(gate <20%)')
        # Bound diagnosis
        for r in experts:
            if r.pred_m_s > r.pred_f_s:
                print(f'  WARN {r.name}: Calculon bound=memory due to '
                      f'weight_mult={r.weight_mult:.0f}; active-resident '
                      f'bound would be '
                      f'{"compute" if r.pred_f_s >= r.pred_m_active_s else "memory"}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=os.path.join(os.path.dirname(_ROOT), 'models/deepseek-v3-671b.json'))
    p.add_argument('--system', default=os.path.join(os.path.dirname(_ROOT), 'systems/BW1100.json'))
    p.add_argument('--matrix-dtype', default='float8')
    p.add_argument('--vector-dtype', default='bfloat16')
    p.add_argument('--seq-size', type=int, default=1024,
                   help='Must match Phase2 G4 CSV seq for compute-track err')
    p.add_argument('--microbatch-size', type=int, default=1)
    p.add_argument('--expert-par', type=int, default=1)
    p.add_argument('--stage', default='fw',
                   choices=['fw', 'agrad', 'wgrad', 'all'])
    p.add_argument('--predict-only', action='store_true',
                   help='Alias kept for CLI symmetry; Step B is CSV/predict driven')
    p.add_argument('--phase2-csv', nargs='*',
                   default=[os.path.join(_TEST_DIR, 'phase2_dsv3_microbench.csv')],
                   help='Prefer phase2_g4g5.csv for MoE physical meas')
    p.add_argument('--csv', default=os.path.join(_TEST_DIR, 'phase3_stepB.csv'))
    p.set_defaults(grouped_expert=True)
    p.add_argument('--grouped-expert', dest='grouped_expert',
                   action='store_true',
                   help='Run one-launch Triton grouped FP8 benchmark (default)')
    p.add_argument('--no-grouped-expert', dest='grouped_expert',
                   action='store_false',
                   help='Skip grouped benchmark; retain prediction/CSV only')
    p.add_argument('--grouped-routed-experts', type=int, default=None,
                   help='Debug/validation override; default=model num_experts')
    p.add_argument('--grouped-warmup', type=int, default=5)
    p.add_argument('--grouped-iters', type=int, default=20)
    p.add_argument('--grouped-min-ms', type=float, default=100.0)
    p.add_argument('--grouped-max-iters', type=int, default=100)
    p.add_argument('--device', default='auto',
                   help='HIP device index, or auto=most free VRAM (default)')
    p.add_argument('--update-json', nargs='?', const='__SYSTEM__', default=None,
                   help='Write comparable grouped timings into system JSON; '
                        'optional path, default=--system')
    p.add_argument('--auto-seq-regimes', action='store_true',
                   help='Delegate to the 3-regime seq calibrator '
                        '(tokens/expert 4,32,128)')
    args = p.parse_args()

    if args.auto_seq_regimes:
        import subprocess
        script = os.path.join(_TEST_DIR, 'calibrate_grouped_moe_seq_model.py')
        cmd = [sys.executable, script, '--system', args.system,
               '--device', str(args.device), '--warmup', str(args.grouped_warmup),
               '--iters', str(args.grouped_iters)]
        raise SystemExit(subprocess.call(cmd))

    if not args.predict_only and args.grouped_expert:
        import torch
        if args.device == 'auto':
            choices = []
            for dev in range(torch.cuda.device_count()):
                try:
                    free_b, total_b = torch.cuda.mem_get_info(dev)
                    choices.append((free_b, total_b, dev))
                except RuntimeError:
                    continue
            if not choices:
                raise SystemExit('No usable HIP device found')
            free_b, total_b, device = max(choices)
        else:
            device = int(args.device)
            free_b, total_b = torch.cuda.mem_get_info(device)
        # Full Gate/Up requires ~4.5 GiB after direct-FP8 allocation. Keep a
        # margin for Triton workspace and other allocations.
        if free_b < 6 * 1024 ** 3:
            raise SystemExit(
                f'HIP device {device} has only {free_b/1024**3:.2f} GiB free; '
                'grouped Step B requires at least 6 GiB free')
        torch.cuda.set_device(device)
        print(f'Using HIP device {device}: free={free_b/1024**3:.2f} GiB / '
              f'total={total_b/1024**3:.2f} GiB', flush=True)

    llm, app, syst, exe = compile_dsv3(
        args.model, args.system,
        matrix_dtype=args.matrix_dtype, vector_dtype=args.vector_dtype,
        seq_size=args.seq_size, microbatch_size=args.microbatch_size,
        expert_par=args.expert_par,
    )
    active_w = _active_equiv(app)
    banner('Phase3 Step B — MoE weight split',
           seq=app.seq_size, ep=exe.expert_par,
           num_experts=getattr(app, 'num_experts', None),
           active_equiv=active_w, predict_only=args.predict_only)

    meas_idx = load_phase2_meas_index(args.phase2_csv, expect_seq=args.seq_size)
    if args.phase2_csv:
        print(f'Loaded Phase2 meas keys: {len(meas_idx)}')

    from phase2_dsv3_op_catalog import build_catalog
    cat = build_catalog(llm, app, exe, syst, stages=('fw',), blocks=('moe',))
    catalog_dims = {(r.name, r.stage): (r.batch_seq, r.c_in, r.c_out)
                    for r in cat if r.cls == 'Linear' and r.c_in and r.c_out}
    grouped = {}
    if args.grouped_expert and not args.predict_only:
        if args.matrix_dtype != 'float8':
            raise SystemExit('grouped expert benchmark currently requires '
                             '--matrix-dtype float8')
        from phase3_fused_block import measure_grouped_expert
        routed_experts = (args.grouped_routed_experts
                          if args.grouped_routed_experts is not None
                          else int(getattr(app, 'num_experts', 256)))
        if not 1 <= routed_experts <= int(getattr(app, 'num_experts', 256)):
            raise SystemExit('--grouped-routed-experts must be in '
                             f'[1, {getattr(app, "num_experts", 256)}]')
        # Do not leave an old successful CSV at the requested output path if
        # the GPU process faults before the new measurements are complete.
        if os.path.exists(args.csv):
            previous = args.csv + '.previous'
            os.replace(args.csv, previous)
            print(f'Previous CSV moved to: {previous}', flush=True)
        for expert in (r for r in cat
                       if r.cls == 'Linear' and _is_expert(r.name)):
            print(f'Grouped expert START: {expert.name} '
                  f'M={app.seq_size} N={expert.c_out} K={expert.c_in} '
                  f'routed_experts={routed_experts}', flush=True)
            grouped[expert.name] = measure_grouped_expert(
                app.seq_size, expert.c_out, expert.c_in,
                routed_experts=routed_experts,
                topk=int(getattr(app, 'moe_topk', 8)),
                shared_experts=int(getattr(app, 'num_shared_experts', 1)),
                warmup=args.grouped_warmup, iters=args.grouped_iters,
                min_ms=args.grouped_min_ms,
                max_iters=args.grouped_max_iters)
            print('Grouped expert DONE:', expert.name, grouped[expert.name],
                  flush=True)
    rows = run(llm, app, args, meas_idx, catalog_dims, grouped)
    write_csv(args.csv, rows)
    print_report(rows, active_w)

    if args.update_json is not None:
        target = args.system if args.update_json == '__SYSTEM__' else args.update_json
        entries = {}
        for r in rows:
            if not r.grouped_comparable or r.grouped_meas_s is None:
                continue
            dims = catalog_dims.get((r.name, r.stage))
            if not dims:
                continue
            m, ci, co = dims
            entries[r.name] = {
                'shape': [int(m), int(ci), int(co)],
                'weight_multiplier': float(r.weight_mult),
                'flop_multiplier': float(r.flop_mult),
                'routed_experts': int(r.distinct_routed_experts),
                'latency_s': float(r.grouped_meas_s),
                'effective_bandwidth_Bps': (
                    float(r.total_decomposed_bytes) / float(r.grouped_meas_s)),
                'effective_flops_per_s': (
                    2.0 * float(m) * float(ci) * float(co)
                    * float(r.flop_mult) / float(r.grouped_meas_s)),
                'backend': r.grouped_backend,
            }
        if not entries:
            raise SystemExit('--update-json requested but no complete grouped '
                             'measurements are available')
        with open(target) as f:
            cfg = json.load(f)
        cfg.setdefault('grouped_moe_shape_latency_s', {})[
            args.matrix_dtype] = entries
        tmp = target + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(cfg, f, indent=2)
            f.write('\n')
        os.replace(tmp, target)
        print(f'Updated grouped MoE calibration: {target} '
              f'({len(entries)} projections)', flush=True)

    print('\nInterpretation:')
    print('  • T_compute (flop_mult=9 / physical×9): use for H4 block timing')
    print('  • T_mem(257): Calculon residency model for capacity / mem_time;')
    print('    do not compare directly to single-card active-expert traffic')


if __name__ == '__main__':
    main()
