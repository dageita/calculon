#!/usr/bin/env python3
"""Phase3 Step A — Dense / MoE block aggregate (H4).

Four-line comparison per block template (dense / moe), stages fw|agrad|wgrad:

  1. Σ calculon op time     — sum of per-layer max(flops_time, mem_time)
  2. Σ isolated op meas     — Phase2 CSV ingest and/or live re-measure
  3. Block meas             — sequential_proxy (=Σiso) or fused MLA+FFN
  4. _block_*_time          — Calculon homogeneous block stats (→ .so)

Gates (per stage, not averaged):
  |Σpred − Σiso| small (H2 carry-over)
  |block_meas − Σiso| = fusion gap (H4; only meaningful with --block-mode fused)
  |block_meas − _block_*| = delivery error  (gate <20% per stage)

Example:
  # Predict-only (no CUDA)
  python test/phase3_stepA_block_aggregate.py --predict-only \\
      --seq-size 4096 --csv test/phase3_stepA.csv

  # Live sequential proxy
  python test/phase3_stepA_block_aggregate.py --seq-size 4096 \\
      --block-mode sequential --csv test/phase3_stepA_meas.csv

  # H4 fused MLA+FFN (training-path style)
  python test/phase3_stepA_block_aggregate.py --seq-size 4096 \\
      --block-mode fused --stage fw --csv test/phase3_stepA_fused.csv
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

from phase2_dsv3_op_catalog import build_catalog, compile_dsv3  # noqa: E402
from phase3_dsv3_common import (  # noqa: E402
    STAGES,
    banner,
    collect_layer_times,
    homogeneous_block_stats,
    load_phase2_meas_index,
    mape,
    rel_err_pct,
    sum_pred,
    write_csv,
)


@dataclass
class AggregateRow:
    block: str
    stage: str
    n_ops: int
    n_charged: int
    sum_pred_f_s: float
    sum_pred_m_s: float
    sum_pred_max_s: float
    block_calc_s: float
    block_calc_flops_s: float
    block_calc_mem_s: float
    structural_pred_s: float
    block_calc_catalog_scope_s: float
    sum_iso_meas_s: Optional[float]
    block_meas_s: Optional[float]
    iso_complete: bool
    block_meas_complete: bool
    comparable: bool
    block_mode: str
    err_sumpred_vs_iso_pct: Optional[float]
    err_block_vs_iso_pct: Optional[float]
    err_block_vs_calc_pct: Optional[float]
    fusion_gap_pct: Optional[float]
    notes: str


def _sum_iso_from_index(
    layer_rows, meas_idx: Dict[Tuple[str, str], float], stage: str,
) -> Tuple[Optional[float], int, int]:
    total = 0.0
    hit = miss = 0
    for r in layer_rows:
        if r.stage != stage:
            continue
        if not r.charges_compute:
            continue
        # Skip fused-zero ops (SiLU/Softmax) — they contribute 0 meas.
        if r.flops <= 0 and r.pred_max_s <= 0:
            continue
        key = (r.name, stage)
        if key in meas_idx:
            total += meas_idx[key]
            hit += 1
        else:
            miss += 1
    if hit == 0:
        return None, hit, miss
    return total, hit, miss


def _load_grouped_index(paths, expect_seq: int):
    out = {}
    for path in paths or []:
        if not os.path.exists(path):
            continue
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                if row.get('seq_size') and int(row['seq_size']) != expect_seq:
                    continue
                if str(row.get('grouped_comparable', '')).lower() != 'true':
                    continue
                try:
                    value = float(row['grouped_meas_s'])
                except (KeyError, TypeError, ValueError):
                    continue
                out[(row['name'], row['stage'])] = value
    return out


def _live_measure_sum(
    llm, app, exe, syst, block: str, stage: str, args,
) -> Tuple[Optional[float], str]:
    """Re-measure charged ops via Phase2 microbench; return Σ meas."""
    from phase2_dsv3_op_microbench import measure_row, resolve_g2_kernels

    cat = build_catalog(
        llm, app, exe, syst, stages=(stage,), blocks=(block,))
    # Only rows that charge compute time.
    charged = [
        r for r in cat
        if not (r.flops <= 0 and r.pred_max_s <= 0)
        and r.cls not in ('Fork', 'TPComm', 'DropOut', 'ElementWise')
    ]
    g2 = resolve_g2_kernels(args.g2_kernel, args.matrix_dtype)
    total = 0.0
    n_ok = 0
    skips = []
    active_equiv = float(
        getattr(app, 'moe_topk', 8) + getattr(app, 'num_shared_experts', 1))
    for r in charged:
        # Prefer G4 physical track for MoE experts.
        brows = measure_row(
            r, args.matrix_dtype, app.seq_size, app.hidden, app.attn_heads,
            active_equiv, args.warmup, args.iters, args.min_ms, args.max_iters,
            measure=True, syst=syst, vector_dtype=args.vector_dtype,
            g2_kernels=g2,
        )
        picked = None
        for br in brows:
            if br.meas_s is None:
                continue
            if br.track == 'physical':
                picked = br.meas_s
                break
            if picked is None and br.track in ('abstract', 'assert'):
                if br.meas_s is not None:
                    picked = br.meas_s
        if picked is None:
            # assert / fused / skipped
            note = brows[0].notes if brows else ''
            sk = brows[0].skipped if brows else 'no_meas'
            if sk or 'PASS' in (note or '') or 'fused' in (brows[0].kernel if brows else ''):
                continue
            skips.append(f'{r.name}:{sk or "no_meas"}')
            continue
        total += float(picked)
        n_ok += 1
    note = f'live_ops={n_ok}'
    if skips:
        note += f' skips={len(skips)}'
    if n_ok == 0:
        return None, note + ' FAIL_no_meas'
    return total, note


def run_block(llm, app, exe, syst, block: str, stage: str, args,
              meas_idx) -> AggregateRow:
    layers = collect_layer_times(llm, block, stages=(stage,))
    all_layers = collect_layer_times(
        llm, block, stages=(stage,), include_structural=True)
    structural_pred = sum(
        r.pred_max_s for r in all_layers if r.scope == 'structural')
    s = sum_pred(layers, stage)
    bstats = homogeneous_block_stats(llm, block)
    key = {
        'fw': 'block_fw_s',
        'agrad': 'block_agrad_s',
        'wgrad': 'block_wgrad_s',
    }[stage]
    block_calc = bstats[key]
    block_calc_catalog = max(0.0, block_calc - structural_pred)

    sum_iso = None
    iso_complete = False
    notes = []
    if meas_idx:
        sum_iso, hit, miss = _sum_iso_from_index(layers, meas_idx, stage)
        iso_complete = (miss == 0 and hit > 0)
        notes.append(f'phase2_csv hit={hit} miss={miss}')
        if miss > 0:
            notes.append('Σiso INCOMPLETE (missing ops, often G2@wrong seq)')
            sum_iso = None

    block_meas = None
    block_meas_complete = False
    block_mode = args.block_mode
    if not args.predict_only:
        # BW1100 Phase2 owns backend-matched isolated measurements.  Reuse its
        # CSV rather than the H20-only live measure_row/_scaled_mm API.
        live = sum_iso
        notes.append('BW1100 Σiso from backend-matched Phase2 CSV')
        iso_complete = (live is not None and iso_complete)
        if args.block_mode == 'sequential':
            block_meas = live if iso_complete else None
            block_meas_complete = block_meas is not None
            block_mode = 'sequential_proxy'
            notes.append('block_meas=Σiso (sequential proxy; not H4 fused)')
        elif args.block_mode == 'fused':
            from phase3_fused_block import measure_fused_block
            cat = build_catalog(
                llm, app, exe, syst, stages=(stage,), blocks=(block,))
            active_equiv = float(
                getattr(app, 'moe_topk', 8)
                + getattr(app, 'num_shared_experts', 1))
            fused_t, fused_note = measure_fused_block(
                app, block, stage,
                microbatch=args.microbatch_size,
                warmup=args.warmup, iters=args.iters,
                min_ms=args.min_ms, max_iters=args.max_iters,
                fused_impl=args.fused_impl,
                catalog_rows=cat,
                matrix_dtype=args.matrix_dtype,
                active_equiv=active_equiv,
            )
            block_meas = fused_t
            block_meas_complete = fused_t is not None
            block_mode = f'fused_{args.fused_impl}'
            notes.append(fused_note)
            if fused_t is None:
                notes.append('fused FAILED — fusion gate N/A')
    elif args.block_mode == 'sequential' and sum_iso is not None and iso_complete:
        block_meas = sum_iso
        block_meas_complete = True
        block_mode = 'sequential_proxy_from_csv'
        notes.append('block_meas=Σiso from Phase2 CSV (not H4 fused)')
    elif args.block_mode == 'fused':
        notes.append('fused requires GPU measure (omit --predict-only)')
    elif sum_iso is not None and not iso_complete:
        notes.append('block_meas withheld until Σiso complete')

    err_sp_iso = rel_err_pct(s['pred_max_s'], sum_iso)
    # Calculon delivery always vs Σiso when available (kernel-aligned H2 path).
    err_deliv_iso = rel_err_pct(block_calc_catalog, sum_iso)
    # vs block_meas: sequential(=iso) or fused training-path.
    err_deliv_blk = rel_err_pct(block_calc, block_meas) if block_meas is not None else None
    # Prefer iso for the gated "deliv" column; fall back to block_meas.
    err_blk_calc = err_deliv_iso if err_deliv_iso is not None else err_deliv_blk
    err_blk_iso = None
    if block_meas is not None and sum_iso is not None and sum_iso > 0:
        # (block_meas − Σiso)/Σiso — negative if fused slower
        err_blk_iso = 100.0 * (block_meas - sum_iso) / sum_iso
    fusion_gap = None
    if block_meas is not None and sum_iso is not None and sum_iso > 0:
        # positive ⇒ fused faster than isolated sum
        fusion_gap = 100.0 * (sum_iso - block_meas) / sum_iso
    if (block_mode.startswith('fused') and err_deliv_blk is not None
            and err_deliv_iso is not None):
        notes.append(f'deliv_vs_fused={err_deliv_blk:+.1f}% (not gated)')

    return AggregateRow(
        block=block, stage=stage,
        n_ops=int(s['n_ops']), n_charged=int(s['n_charged']),
        sum_pred_f_s=s['pred_f_s'], sum_pred_m_s=s['pred_m_s'],
        sum_pred_max_s=s['pred_max_s'],
        block_calc_s=block_calc,
        block_calc_flops_s=bstats['block_fw_flops_s'] if stage == 'fw' else float('nan'),
        block_calc_mem_s=bstats['block_fw_mem_s'] if stage == 'fw' else float('nan'),
        structural_pred_s=structural_pred,
        block_calc_catalog_scope_s=block_calc_catalog,
        sum_iso_meas_s=sum_iso, block_meas_s=block_meas,
        iso_complete=iso_complete,
        block_meas_complete=block_meas_complete,
        comparable=bool(iso_complete or block_meas_complete),
        block_mode=block_mode,
        err_sumpred_vs_iso_pct=err_sp_iso,
        err_block_vs_iso_pct=err_blk_iso,
        err_block_vs_calc_pct=err_blk_calc,
        fusion_gap_pct=fusion_gap,
        notes='; '.join(notes),
    )


def print_report(rows: Sequence[AggregateRow], gate_pct: float = 20.0) -> None:
    print('\n--- Step A summary ---')
    hdr = (f'{"block":6} {"stage":5} {"Σpred":>10} {"_block":>10} '
           f'{"Σiso":>10} {"blk_m":>10} {"sp↔iso":>8} {"deliv":>8} {"fuse%":>7}')
    print(hdr)
    for r in rows:
        def fmt(x, w=10):
            return f'{x:{w}.4e}' if x is not None and x == x else f'{"—":>{w}}'

        def fp(x, w=8):
            return f'{x:{w}.1f}' if x is not None else f'{"—":>{w}}'

        print(
            f'{r.block:6} {r.stage:5} {fmt(r.sum_pred_max_s)} '
            f'{fmt(r.block_calc_s)} {fmt(r.sum_iso_meas_s)} '
            f'{fmt(r.block_meas_s)} {fp(r.err_sumpred_vs_iso_pct)} '
            f'{fp(r.err_block_vs_calc_pct)} {fp(r.fusion_gap_pct, 7)}'
        )
        if r.notes:
            print(f'         notes: {r.notes}')
        print(f'         scope: catalog={r.block_calc_catalog_scope_s:.4e}s; '
              f'structural residual={r.structural_pred_s:.4e}s')

    # Per-stage gates (do NOT average fw/agrad/wgrad into one MAPE).
    print(f'\n--- Delivery gates per stage (|_block vs Σiso|, <{gate_pct:.0f}%) ---')
    stages_seen = []
    for r in rows:
        if r.stage not in stages_seen:
            stages_seen.append(r.stage)
    any_gate = False
    for stage in stages_seen:
        sub = [r for r in rows if r.stage == stage]
        usable = []
        for r in sub:
            if r.err_block_vs_calc_pct is None:
                continue
            if r.sum_iso_meas_s is None or not r.iso_complete:
                continue
            usable.append(r)
        errs = [r.err_block_vs_calc_pct for r in usable]
        m = mape(errs)
        if m is None:
            print(f'  {stage:5}  N/A  (need complete Σiso)')
            continue
        any_gate = True
        gate = 'PASS' if m < gate_pct else 'REVIEW'
        detail = ', '.join(
            f'{r.block}:{r.err_block_vs_calc_pct:+.1f}%' for r in usable)
        print(f'  {stage:5}  MAPE={m:.1f}%  → {gate}  [{detail}]')

        fused_rows = [
            r for r in sub
            if r.fusion_gap_pct is not None
            and r.block_mode.startswith('fused')
            and r.block_meas_s is not None
        ]
        if fused_rows:
            fg = mape([r.fusion_gap_pct for r in fused_rows])
            signed = ', '.join(
                f'{r.block}:{r.fusion_gap_pct:+.1f}%' for r in fused_rows)
            print(f'         H4 fusion_gap (Σiso−fused)/Σiso: '
                  f'|MAPE|={fg:.1f}%  [{signed}]')
            print('         (positive ⇒ fused faster; not averaged into '
                  'delivery gate; |gap|>25% ⇒ review fusion model)')
    if not any_gate:
        print('  (no stage had complete delivery metrics)')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=os.path.join(os.path.dirname(_ROOT), 'models/deepseek-v3-671b.json'))
    p.add_argument('--system', default=os.path.join(os.path.dirname(_ROOT), 'systems/BW1100.json'))
    p.add_argument('--matrix-dtype', default='float8')
    p.add_argument('--vector-dtype', default='bfloat16')
    p.add_argument('--seq-size', type=int, default=1024,
                   help='Must match Phase2 CSV seq when ingesting Σiso')
    p.add_argument('--microbatch-size', type=int, default=1)
    p.add_argument('--expert-par', type=int, default=1)
    p.add_argument('--blocks', nargs='+', default=['dense', 'moe'])
    p.add_argument('--stage', default='fw',
                   choices=['fw', 'agrad', 'wgrad', 'all'])
    p.add_argument('--predict-only', action='store_true')
    p.add_argument('--phase2-csv', nargs='*',
                   default=[os.path.join(_TEST_DIR, 'phase2_dsv3_microbench.csv')],
                   help='Phase2 microbench CSVs for Σ isolated meas')
    p.add_argument('--stepB-csv', nargs='*',
                   default=[os.path.join(_TEST_DIR, 'phase3_stepB.csv')],
                   help='Step B CSV containing comparable grouped expert rows')
    p.add_argument('--block-mode', choices=['sequential', 'fused'],
                   default='sequential',
                   help='sequential=Σiso proxy; fused=H4 block measure')
    p.add_argument(
        '--fused-impl', default='kernel_chain',
        choices=['kernel_chain'],
        help='kernel_chain: same Phase2 kernels in one timed region (fair H4); '
             'BW1100 HIP chain: Triton FP8 + BF16 BMM + RMSNorm')
    p.add_argument('--g2-kernel', default='bf16',
                   choices=['auto', 'fp8', 'bf16', 'both'])
    p.add_argument('--gate-pct', type=float, default=20.0,
                   help='Per-stage delivery MAPE gate (default 20)')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--min-ms', type=float, default=200.0)
    p.add_argument('--max-iters', type=int, default=500)
    p.add_argument('--csv', default=os.path.join(_TEST_DIR, 'phase3_stepA.csv'))
    args = p.parse_args()

    stages = list(STAGES) if args.stage == 'all' else [args.stage]
    llm, app, syst, exe = compile_dsv3(
        args.model, args.system,
        matrix_dtype=args.matrix_dtype, vector_dtype=args.vector_dtype,
        seq_size=args.seq_size, microbatch_size=args.microbatch_size,
        expert_par=args.expert_par,
    )
    banner('Phase3 Step A — block aggregate (H4)',
           seq=app.seq_size, mbs=exe.microbatch_size,
           blocks=','.join(args.blocks), stages=','.join(stages),
           predict_only=args.predict_only, block_mode=args.block_mode,
           fused_impl=args.fused_impl if args.block_mode == 'fused' else 'n/a',
           bmm_dtype=syst.get_bmm_dtype(),
           bmm_scale_attn=syst.get_bmm_time_scale('attn_score'))

    meas_idx = load_phase2_meas_index(args.phase2_csv, expect_seq=args.seq_size)
    grouped_idx = _load_grouped_index(args.stepB_csv, args.seq_size)
    # Grouped expert rows supersede Phase2's deliberately non-comparable serial
    # expert diagnostics and make the MoE isolated sum complete.
    meas_idx.update(grouped_idx)
    if args.phase2_csv:
        print(f'Loaded Phase2 meas keys: {len(meas_idx)}')
    if grouped_idx:
        print(f'Loaded grouped expert keys: {len(grouped_idx)}')

    rows: List[AggregateRow] = []
    for block in args.blocks:
        for stage in stages:
            rows.append(run_block(
                llm, app, exe, syst, block, stage, args, meas_idx))

    write_csv(args.csv, rows)
    print_report(rows, gate_pct=args.gate_pct)

    # Per-op detail dump next to aggregate CSV.
    detail = args.csv.replace('.csv', '_ops.csv')
    if detail == args.csv:
        detail = args.csv + '_ops.csv'
    op_rows = []
    for block in args.blocks:
        op_rows.extend(collect_layer_times(
            llm, block, stages=stages, include_structural=True))
    write_csv(detail, op_rows)


if __name__ == '__main__':
    main()
