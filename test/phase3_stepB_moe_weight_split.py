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


def run(llm, app, args, meas_idx) -> List[SplitRow]:
    stages = ['fw'] if args.stage == 'fw' else (
        ['fw', 'agrad', 'wgrad'] if args.stage == 'all' else [args.stage])
    rows_lt = collect_layer_times(llm, 'moe', stages=stages)
    active_w = _active_equiv(app)
    out: List[SplitRow] = []

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
            name=r.name, cls=r.cls, stage=r.stage,
            flop_mult=r.flop_mult, weight_mult=r.weight_mult,
            pred_f_s=r.pred_f_s, pred_m_s=r.pred_m_s,
            pred_max_s=r.pred_max_s, bound=r.bound,
            pred_m_active_s=pred_m_act, pred_max_active_s=pred_max_act,
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
    p.add_argument('--model', default=os.path.join(_ROOT, 'models/deepseek-v3-671b.json'))
    p.add_argument('--system', default=os.path.join(_ROOT, 'systems/H20.json'))
    p.add_argument('--matrix-dtype', default='float8')
    p.add_argument('--vector-dtype', default='bfloat16')
    p.add_argument('--seq-size', type=int, default=4096,
                   help='Must match Phase2 G4 CSV seq for compute-track err')
    p.add_argument('--microbatch-size', type=int, default=1)
    p.add_argument('--expert-par', type=int, default=1)
    p.add_argument('--stage', default='fw',
                   choices=['fw', 'agrad', 'wgrad', 'all'])
    p.add_argument('--predict-only', action='store_true',
                   help='Alias kept for CLI symmetry; Step B is CSV/predict driven')
    p.add_argument('--phase2-csv', nargs='*', default=[],
                   help='Prefer phase2_g4g5.csv for MoE physical meas')
    p.add_argument('--csv', default=os.path.join(_TEST_DIR, 'phase3_stepB.csv'))
    args = p.parse_args()

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

    rows = run(llm, app, args, meas_idx)
    write_csv(args.csv, rows)
    print_report(rows, active_w)

    print('\nInterpretation:')
    print('  • T_compute (flop_mult=9 / physical×9): use for H4 block timing')
    print('  • T_mem(257): Calculon residency model for capacity / mem_time;')
    print('    do not compare directly to single-card active-expert traffic')


if __name__ == '__main__':
    main()
