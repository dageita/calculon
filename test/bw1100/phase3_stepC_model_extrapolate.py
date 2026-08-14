#!/usr/bin/env python3
"""Phase3 Step C — Model-level compute extrapolation (no comm).

  T_pred ≈ n_dense · T_dense_block + n_moe · T_moe_block
         ≈ 3 · T_dense + 58 · T_moe   (DS-V3-671B defaults)

Sources for T_* (choose via --source):
  calc_block  — Calculon _block_*_time (feeds .so)
  sum_pred    — Σ per-op pred_max
  sum_iso     — Σ Phase2 isolated meas (from Step A CSV or phase2 CSVs)
  block_meas  — Step A block_meas (sequential proxy / fused)

Compares lines to bound compute error injected into LLMFlowSimulator
(excluding EP/TP/PP).

Example:
  # fw only (default)
  python test/phase3_stepC_model_extrapolate.py --predict-only --seq-size 4096 \\
      --stepA-csv test/phase3_stepA.csv --csv test/phase3_stepC.csv

  # Full training-step compute: fw + agrad + wgrad (needs Step A --stage all CSV)
  python test/phase3_stepC_model_extrapolate.py --predict-only --seq-size 4096 \\
      --stage all --stepA-csv test/phase3_stepA_all.csv \\
      --csv test/phase3_stepC_all.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TEST_DIR)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase2_dsv3_op_catalog import compile_dsv3  # noqa: E402
from phase3_dsv3_common import (  # noqa: E402
    STAGES,
    banner,
    collect_layer_times,
    homogeneous_block_stats,
    load_phase2_meas_index,
    model_layer_counts,
    rel_err_pct,
    sum_pred,
    write_csv,
)


@dataclass
class ExtrapolateRow:
    stage: str
    n_dense: int
    n_moe: int
    n_total: int
    T_dense_calc_s: float
    T_moe_calc_s: float
    T_dense_sumpred_s: float
    T_moe_sumpred_s: float
    T_dense_iso_s: Optional[float]
    T_moe_iso_s: Optional[float]
    T_dense_blkmeas_s: Optional[float]
    T_moe_blkmeas_s: Optional[float]
    model_calc_s: float
    model_calc_catalog_scope_s: float
    model_structural_residual_s: float
    model_sumpred_s: float
    model_iso_s: Optional[float]
    model_blkmeas_s: Optional[float]
    err_calc_vs_iso_pct: Optional[float]
    err_calc_vs_blkmeas_pct: Optional[float]
    notes: str


def _load_stepA(path: str) -> Dict[Tuple[str, str], dict]:
    out: Dict[Tuple[str, str], dict] = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, newline='') as f:
        for d in csv.DictReader(f):
            out[(d['block'], d['stage'])] = d
    return out


def _f(d: Optional[dict], key: str) -> Optional[float]:
    if not d:
        return None
    v = d.get(key)
    if v is None or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None


def run_stage(llm, app, stage: str, stepA: dict, meas_idx,
              args) -> ExtrapolateRow:
    n_dense, n_moe, n_total = model_layer_counts(app)

    # Fresh Calculon numbers
    dense_ops = collect_layer_times(llm, 'dense', stages=(stage,))
    moe_ops = collect_layer_times(llm, 'moe', stages=(stage,))
    sd = sum_pred(dense_ops, stage)
    sm = sum_pred(moe_ops, stage)
    bd = homogeneous_block_stats(llm, 'dense')
    bm = homogeneous_block_stats(llm, 'moe')
    key = {'fw': 'block_fw_s', 'agrad': 'block_agrad_s',
           'wgrad': 'block_wgrad_s'}[stage]
    Td_calc, Tm_calc = bd[key], bm[key]
    Td_sp, Tm_sp = sd['pred_max_s'], sm['pred_max_s']
    # Step A isolated measurements cover the Phase2 catalog, not structural
    # ElementWise/DropOut/comm placeholders. Keep both scopes explicit.
    Td_struct = _f(stepA.get(('dense', stage)), 'structural_pred_s') or 0.0
    Tm_struct = _f(stepA.get(('moe', stage)), 'structural_pred_s') or 0.0
    Td_catalog = Td_calc - Td_struct
    Tm_catalog = Tm_calc - Tm_struct

    # Prefer Step A CSV for iso / block_meas
    da = stepA.get(('dense', stage))
    ma = stepA.get(('moe', stage))
    Td_iso = _f(da, 'sum_iso_meas_s')
    Tm_iso = _f(ma, 'sum_iso_meas_s')
    Td_bm = _f(da, 'block_meas_s')
    Tm_bm = _f(ma, 'block_meas_s')

    notes = []
    def _true(d, key):
        return bool(d) and str(d.get(key, '')).strip().lower() in (
            '1', 'true', 'yes')

    # Complete status is authoritative. Old CSVs without this field fail
    # closed, preventing a partial MoE proxy from being extrapolated 58 times.
    if da and not _true(da, 'block_meas_complete'):
        Td_bm = None
        notes.append('dense_block_meas_incomplete')
    if ma and not _true(ma, 'block_meas_complete'):
        Tm_bm = None
        notes.append('moe_block_meas_incomplete')
    if da or ma:
        notes.append('stepA_csv')
    # Drop incomplete Σiso (Step A marks INCOMPLETE in notes).
    for label, d, val_name in (
        ('dense', da, 'Td_iso'), ('moe', ma, 'Tm_iso'),
    ):
        if d and 'INCOMPLETE' in (d.get('notes') or ''):
            if label == 'dense':
                Td_iso = None
            else:
                Tm_iso = None
            notes.append(f'{label}_iso_incomplete')
    # Fallback: rebuild iso from phase2 CSVs if missing
    if (Td_iso is None or Tm_iso is None) and meas_idx:
        from phase3_stepA_block_aggregate import _sum_iso_from_index
        if Td_iso is None:
            Td_iso, h, m = _sum_iso_from_index(dense_ops, meas_idx, stage)
            notes.append(f'dense_iso_csv hit={h} miss={m}')
            if m > 0:
                Td_iso = None
        if Tm_iso is None:
            Tm_iso, h, m = _sum_iso_from_index(moe_ops, meas_idx, stage)
            notes.append(f'moe_iso_csv hit={h} miss={m}')
            if m > 0:
                Tm_iso = None

    def scale(td, tm):
        if td is None or tm is None:
            return None
        return n_dense * td + n_moe * tm

    model_calc = scale(Td_calc, Tm_calc)
    model_calc_catalog = scale(Td_catalog, Tm_catalog)
    model_structural = scale(Td_struct, Tm_struct)
    model_sp = scale(Td_sp, Tm_sp)
    model_iso = scale(Td_iso, Tm_iso)
    model_bm = scale(Td_bm, Tm_bm)

    return ExtrapolateRow(
        stage=stage, n_dense=n_dense, n_moe=n_moe, n_total=n_total,
        T_dense_calc_s=Td_calc, T_moe_calc_s=Tm_calc,
        T_dense_sumpred_s=Td_sp, T_moe_sumpred_s=Tm_sp,
        T_dense_iso_s=Td_iso, T_moe_iso_s=Tm_iso,
        T_dense_blkmeas_s=Td_bm, T_moe_blkmeas_s=Tm_bm,
        model_calc_s=model_calc or 0.0,
        model_calc_catalog_scope_s=model_calc_catalog or 0.0,
        model_structural_residual_s=model_structural or 0.0,
        model_sumpred_s=model_sp or 0.0,
        model_iso_s=model_iso, model_blkmeas_s=model_bm,
        err_calc_vs_iso_pct=rel_err_pct(model_calc_catalog, model_iso),
        err_calc_vs_blkmeas_pct=rel_err_pct(model_calc_catalog, model_bm),
        notes='; '.join(notes) or 'predict_only',
    )


def print_report(rows: Sequence[ExtrapolateRow]) -> None:
    print('\n--- Step C summary (model compute, no comm) ---')
    for r in rows:
        print(f'\nstage={r.stage}  layers={r.n_dense}·dense + {r.n_moe}·moe '
              f'(total {r.n_total})')
        print(f'  T_dense_calc={r.T_dense_calc_s:.4e}  '
              f'T_moe_calc={r.T_moe_calc_s:.4e}')
        print(f'  model_calc     = {r.model_calc_s:.4e} s   '
              f'(← .so compute bound)')
        print(f'    catalog scope= {r.model_calc_catalog_scope_s:.4e} s; '
              f'structural residual={r.model_structural_residual_s:.4e} s')
        print(f'  model_sumpred  = {r.model_sumpred_s:.4e} s')
        if r.model_iso_s is not None:
            print(f'  model_Σiso     = {r.model_iso_s:.4e} s   '
                  f'err(calc↔iso)={r.err_calc_vs_iso_pct:.1f}%')
        else:
            print('  model_Σiso     = —  (pass --stepA-csv or --phase2-csv)')
        if r.model_blkmeas_s is not None:
            print(f'  model_blkmeas  = {r.model_blkmeas_s:.4e} s   '
                  f'err(calc↔blk)={r.err_calc_vs_blkmeas_pct:.1f}%')
        print(f'  notes: {r.notes}')

    if len(rows) > 1:
        # Full microbatch compute ≈ Σ_stages (n_d·T_d + n_m·T_m)
        calc = sum(r.model_calc_s for r in rows)
        iso_vals = [r.model_iso_s for r in rows if r.model_iso_s is not None]
        print('\n--- Full training-step compute (Σ fw+agrad+wgrad, no comm) ---')
        print(f'  model_calc_step = {calc:.4e} s   (← .so compute bound)')
        if len(iso_vals) == len(rows):
            iso = sum(iso_vals)
            err = 100.0 * (calc - iso) / iso if iso > 0 else float('nan')
            print(f'  model_Σiso_step = {iso:.4e} s   err(calc↔iso)={err:.1f}%')
        else:
            print('  model_Σiso_step = —  (incomplete iso for some stages)')
        print('  (one microbatch, all 61 layers; exclude optim / remat / comm)')

    print('\nThis is a compute-only upper/lower bound for LLMFlowSimulator;')
    print('EP/TP/PP / bubble are out of scope for Phase3.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=os.path.join(os.path.dirname(_ROOT), 'models/deepseek-v3-671b.json'))
    p.add_argument('--system', default=os.path.join(os.path.dirname(_ROOT), 'systems/BW1100.json'))
    p.add_argument('--matrix-dtype', default='float8')
    p.add_argument('--vector-dtype', default='bfloat16')
    p.add_argument('--seq-size', type=int, default=1024)
    p.add_argument('--microbatch-size', type=int, default=1)
    p.add_argument('--expert-par', type=int, default=1)
    p.add_argument('--stage', default='fw',
                   choices=['fw', 'agrad', 'wgrad', 'all'])
    p.add_argument('--predict-only', action='store_true')
    p.add_argument('--stepA-csv', default='',
                   help='phase3_stepA.csv from Step A')
    p.add_argument('--phase2-csv', nargs='*',
                   default=[os.path.join(_TEST_DIR, 'phase2_dsv3_microbench.csv')])
    p.add_argument('--csv', default=os.path.join(_TEST_DIR, 'phase3_stepC.csv'))
    args = p.parse_args()

    stages = list(STAGES) if args.stage == 'all' else [args.stage]
    llm, app, syst, exe = compile_dsv3(
        args.model, args.system,
        matrix_dtype=args.matrix_dtype, vector_dtype=args.vector_dtype,
        seq_size=args.seq_size, microbatch_size=args.microbatch_size,
        expert_par=args.expert_par,
    )
    n_dense, n_moe, n_total = model_layer_counts(app)
    banner('Phase3 Step C — model extrapolate',
           seq=app.seq_size,
           formula=f'{n_dense}·dense + {n_moe}·moe',
           stages=','.join(stages))

    stepA = _load_stepA(args.stepA_csv)
    meas_idx = load_phase2_meas_index(args.phase2_csv, expect_seq=args.seq_size)
    rows = [run_stage(llm, app, st, stepA, meas_idx, args) for st in stages]
    write_csv(args.csv, rows)
    print_report(rows)


if __name__ == '__main__':
    main()
