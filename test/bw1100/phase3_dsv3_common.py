#!/usr/bin/env python3
"""Shared helpers for Phase3 DS-V3 block / MoE / model-extrapolation scripts."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TEST_DIR)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase2_dsv3_op_catalog import (  # noqa: E402
    SKIP_CLASSES,
    compile_dsv3,
)

STAGES = ('fw', 'agrad', 'wgrad')


@dataclass
class LayerTime:
    block: str
    name: str
    cls: str
    stage: str
    flops: float
    bytes: float
    pred_f_s: float
    pred_m_s: float
    pred_max_s: float
    bound: str
    weight_mult: float
    flop_mult: float
    charges_compute: bool
    scope: str


def rel_err_pct(pred: float, meas: Optional[float]) -> Optional[float]:
    if meas is None or meas <= 0:
        return None
    return 100.0 * (pred - meas) / meas


def mape(errs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [abs(e) for e in errs if e is not None and not math.isnan(e)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def write_csv(path: str, rows: Sequence[Any]) -> None:
    if not rows:
        print('No rows to write:', path)
        return
    fields = list(asdict(rows[0]).keys())
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    out_dir = os.path.dirname(os.path.abspath(path)) or '.'
    fd, tmp = tempfile.mkstemp(prefix='.phase3-', suffix='.csv.tmp',
                               dir=out_dir, text=True)
    try:
        with os.fdopen(fd, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    print(f'Wrote CSV: {path}  ({len(rows)} rows)')


def layer_stage_times(layer, stage: str) -> Tuple[float, float, float, float, float]:
    if stage == 'fw':
        flops = float(layer.get_fw_flops())
        nbytes = float(layer.get_fw_mem_accessed())
    elif stage == 'agrad':
        flops = float(layer.get_agrad_flops())
        nbytes = float(layer.get_agrad_mem_accessed())
    elif stage == 'wgrad':
        flops = float(layer.get_wgrad_flops())
        nbytes = float(layer.get_wgrad_mem_accessed())
    else:
        raise ValueError(stage)
    pred_f = float(layer.compute_flops_time(stage))
    pred_m = float(layer.compute_mem_time(stage))
    pred_max = float(layer.sys.get_processing_time(pred_f, pred_m))
    return flops, nbytes, pred_f, pred_m, pred_max


def iter_template_layers(llm, block: str):
    if block == 'dense':
        layers = llm._dense_layers
    elif block == 'moe':
        layers = llm._moe_layers
    else:
        raise ValueError(block)
    if layers is None:
        raise SystemExit(f'No {block} template (is model MoE/MLA?)')
    return layers


def collect_layer_times(
    llm, block: str, stages: Sequence[str] = ('fw',),
    include_structural: bool = False,
) -> List[LayerTime]:
    rows: List[LayerTime] = []
    for layer in iter_template_layers(llm, block):
        cls = layer.__class__.__name__
        structural = cls in SKIP_CLASSES
        if structural and not include_structural:
            continue
        wm = float(getattr(layer, 'weight_multiplier', 1.0) or 1.0)
        fm = float(getattr(layer, 'flop_multiplier', 1.0) or 1.0)
        for stage in stages:
            flops, nbytes, pf, pm, pmax = layer_stage_times(layer, stage)
            if stage == 'wgrad' and flops == 0 and getattr(layer, 'weight_space', 0) == 0:
                continue
            charges = not (flops <= 0 and nbytes <= 0)
            rows.append(LayerTime(
                block=block, name=layer.name, cls=cls, stage=stage,
                flops=flops, bytes=nbytes, pred_f_s=pf, pred_m_s=pm,
                pred_max_s=pmax,
                bound='compute' if pf >= pm else 'memory',
                weight_mult=wm, flop_mult=fm, charges_compute=charges,
                scope='structural' if structural else 'catalog',
            ))
    return rows


def sum_pred(rows: Sequence[LayerTime], stage: str = 'fw') -> Dict[str, float]:
    sub = [r for r in rows if r.stage == stage]
    return {
        'pred_f_s': sum(r.pred_f_s for r in sub),
        'pred_m_s': sum(r.pred_m_s for r in sub),
        'pred_max_s': sum(r.pred_max_s for r in sub),
        'flops': sum(r.flops for r in sub),
        'bytes': sum(r.bytes for r in sub),
        'n_ops': float(len(sub)),
        'n_charged': float(sum(1 for r in sub if r.charges_compute)),
    }


def homogeneous_block_stats(llm, block: str) -> Dict[str, float]:
    """Run Calculon homogeneous block accumulation; return key times."""
    saved = llm._llm_block
    try:
        llm._llm_block = list(iter_template_layers(llm, block))
        llm._compute_block_stats_homogeneous()
        return {
            'block_fw_s': float(llm._block_fw_time),
            'block_fw_flops_s': float(llm._block_fw_flops_time),
            'block_fw_mem_s': float(llm._block_fw_mem_time),
            'block_agrad_s': float(llm._block_agrad_time),
            'block_wgrad_s': float(llm._block_wgrad_time),
            'block_weight_bytes': float(llm._block_weight_space),
        }
    finally:
        llm._llm_block = saved


def load_phase2_meas_index(
    paths: Sequence[str],
    expect_seq: Optional[int] = None,
) -> Dict[Tuple[str, str], float]:
    """Map (name, stage) -> meas_s from Phase2 microbench CSVs (physical rows).

    If expect_seq is set, warn when Linear/RMSNorm rows look like a different
    sequence length (column ``m`` ≈ batch_seq for those kernels).
    """
    out: Dict[Tuple[str, str], float] = {}
    seq_hits: Dict[int, int] = {}
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            bw1100_schema = 'measured_us' in (reader.fieldnames or [])
            for d in reader:
                if bw1100_schema:
                    # Serial expert diagnostics are not measurements of the
                    # grouped MoE op and must not enter a block Σiso.
                    if str(d.get('comparable', 'True')).strip().lower() in ('false', '0', 'no'):
                        continue
                    raw_meas = d.get('measured_us')
                    scale = 1e-6
                else:
                    if d.get('track') and d.get('track') != 'physical':
                        continue
                    raw_meas = d.get('meas_s')
                    scale = 1.0
                if not raw_meas:
                    continue
                try:
                    meas = float(raw_meas) * scale
                except ValueError:
                    continue
                name = d.get('name') or ''
                stage = d.get('stage') or 'fw'
                # Prefer dtype-matched / non-fp8-loop rows when duplicates exist.
                key = (name, stage)
                kern = d.get('kernel') or ''
                if key in out and 'fp8' in kern and 'bmm_fp8' in kern:
                    continue
                try:
                    m = int(float(d.get('m') or 0))
                except ValueError:
                    m = 0
                # Heuristic: GEMM/RMSNorm/BMM use m≈seq (mbs=1).
                if m in (1024, 2048, 4096, 8192):
                    seq_hits[m] = seq_hits.get(m, 0) + 1
                    if expect_seq is not None and m != int(expect_seq):
                        # Skip mismatched-seq rows (e.g. G2 S=1024 vs S=4096).
                        continue
                out[key] = meas
    if expect_seq is not None and seq_hits:
        skipped = sum(c for s, c in seq_hits.items() if s != int(expect_seq))
        kept = seq_hits.get(int(expect_seq), 0)
        if skipped:
            print(
                f'WARNING: skipped {skipped} Phase2 rows with m≠seq={expect_seq} '
                f'(kept {kept}). Re-measure those groups at --seq-size {expect_seq} '
                f'for a complete Σiso (common: G2 was S=1024).'
            )
    return out


def model_layer_counts(app) -> Tuple[int, int, int]:
    """Return (n_dense, n_moe, n_total)."""
    n_dense = int(getattr(app, 'first_k_dense', 0) or 0)
    n_total = int(app.num_blocks)
    n_moe = int(getattr(app, 'num_moe_blocks', max(0, n_total - n_dense)))
    return n_dense, n_moe, n_total


def banner(title: str, **kwargs) -> None:
    print(f'=== {title} ===')
    for k, v in kwargs.items():
        print(f'  {k}={v}')
