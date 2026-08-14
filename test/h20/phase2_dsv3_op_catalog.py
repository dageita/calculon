#!/usr/bin/env python3
"""Phase2: export DeepSeek-V3 Calculon single-op catalog (H2 / Exp-2).

Compiles deepseek-v3-671b on H20 (single GPU, no comm) and dumps per-layer
flops / bytes / roofline times for dense + MoE block templates.

Does not measure CUDA kernels — use phase2_dsv3_op_microbench.py for that.

Example:
  python test/phase2_dsv3_op_catalog.py
  python test/phase2_dsv3_op_catalog.py --seq-size 1024 \\
      --csv test/phase2_dsv3_catalog.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TEST_DIR)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from calculon.system import System  # noqa: E402
from calculon.llm.llm import Llm  # noqa: E402
from calculon.llm.layers import (  # noqa: E402
    BatchMatMul,
    GeLU,
    LayerNorm,
    Linear,
    RMSNorm,
    SoftMax,
    SiLU,
)

# Structural / comm layers — not H2 microbench targets.
SKIP_CLASSES = frozenset({
    'Fork', 'TPComm', 'DropOut', 'ElementWise',
})

STAGES: Tuple[str, ...] = ('fw', 'agrad', 'wgrad')

# name substring → Phase2 group (first match wins; order matters).
_GROUP_RULES: List[Tuple[str, str]] = [
    ('_MLA_WUK', 'G6'),
    ('_MLA_WUV', 'G6'),
    ('_MLA_QNorm', 'G5'),
    ('_MLA_KVNorm', 'G5'),
    ('_MLA_QAbsorb', 'G2'),
    ('_MLA_ScoreKV', 'G2'),
    ('_MLA_ScorePE', 'G2'),
    ('_MLA_AttnKV', 'G2'),
    ('_MLA_VAbsorb', 'G2'),
    ('_MLA_', 'G1'),                 # remaining MLA Linears
    ('MlpBlock_MoE_', 'G4'),
    ('MlpBlock_Router', 'G4'),
    ('MlpBlock_Gate', 'G3'),
    ('MlpBlock_Up', 'G3'),
    ('MlpBlock_Down', 'G3'),
    ('MlpBlock_SiLU', 'G3'),
    ('LayerNorm', 'G5'),
    ('RMSNorm', 'G5'),
    ('SoftMax', 'G5'),
    ('SiLU', 'G5'),
    ('GeLU', 'G5'),
]


@dataclass
class CatalogRow:
    block: str                 # dense | moe
    group: str
    name: str
    cls: str
    engine: str                # matrix | vector
    stage: str
    flops: float
    bytes: float
    ai: float
    gflops: float
    pred_f_s: float
    pred_m_s: float
    pred_max_s: float
    bound: str
    # Recovered Linear / BMM shape hints (0 if unknown).
    batch_seq: int
    c_in: int
    c_out: int
    bmm_batch: int
    bmm_m: int
    bmm_n: int
    bmm_k: int
    weight_mult: float
    flop_mult: float
    act_size: int
    notes: str


def assign_group(name: str, cls: str) -> str:
    for needle, group in _GROUP_RULES:
        if needle in name:
            return group
    if cls in ('LayerNorm', 'RMSNorm', 'SoftMax', 'GeLU', 'SiLU'):
        return 'G5'
    if cls == 'Linear':
        return 'G1'
    if cls == 'BatchMatMul':
        return 'G2'
    return 'other'


def default_exe_dict(
    matrix_dtype: str,
    vector_dtype: str,
    microbatch_size: int = 1,
    expert_par: int = 1,
) -> Dict[str, Any]:
    return {
        'num_procs': 1,
        'tensor_par': 1,
        'pipeline_par': 1,
        'data_par': 1,
        'expert_par': expert_par,
        'context_par': 1,
        'tensor_par_net': 0,
        'pipeline_par_net': 0,
        'data_par_net': 0,
        'expert_par_net': 0,
        'context_par_net': 0,
        'batch_size': microbatch_size,
        'microbatch_size': microbatch_size,
        'datatype': matrix_dtype,
        'matrix_dtype': matrix_dtype,
        'vector_dtype': vector_dtype,
        'fused_activation': True,
        'attention_type': 'mla',
        'activation_recompute': 'none',
        'pipeline_interleaving': 1,
        'optimizer_sharding': False,
        'tensor_par_comm_type': 'ar',
        'tensor_par_overlap': 'none',
        'seq_par_ag_redo': False,
        'data_par_overlap': False,
        'weight_offload': False,
        'activations_offload': False,
        'optimizer_offload': False,
        'training': True,
    }


def compile_dsv3(
    model_path: str,
    system_path: str,
    matrix_dtype: str = 'float8',
    vector_dtype: str = 'bfloat16',
    seq_size: Optional[int] = None,
    microbatch_size: int = 1,
    expert_par: int = 1,
) -> Tuple[Llm, Llm.Application, System, Llm.Execution]:
    with open(model_path) as f:
        app_cfg = json.load(f)
    if seq_size is not None:
        app_cfg = dict(app_cfg)
        app_cfg['seq_size'] = int(seq_size)
    with open(system_path) as f:
        sys_cfg = json.load(f)

    app = Llm.Application(app_cfg)
    syst = System(sys_cfg)
    exe = Llm.Execution.from_json(default_exe_dict(
        matrix_dtype, vector_dtype, microbatch_size, expert_par))
    llm = Llm(app, logging.getLogger('phase2_catalog'))
    llm.compile(syst, exe)
    return llm, app, syst, exe


def _recover_linear_shape(
    layer: Linear, batch_seq: int,
) -> Tuple[int, int, int, float, float, str]:
    """Return (batch_seq, c_in, c_out, weight_mult, flop_mult, notes)."""
    notes = ''
    if batch_seq <= 0:
        return 0, 0, 0, 1.0, 1.0, 'no_batch_seq'
    c_in = int(round(layer.inputs_size / batch_seq))
    c_out = int(round(layer.output_size / batch_seq))
    if c_in <= 0 or c_out <= 0:
        return batch_seq, 0, 0, 1.0, 1.0, 'bad_linear_shape'
    base_w = float(c_in * c_out)
    wm = float(layer.weight_space) / base_w if base_w > 0 else 1.0
    base_flops = 2.0 * batch_seq * c_in * c_out
    fm = (float(layer.get_fw_flops()) / base_flops) if base_flops > 0 else 0.0
    if abs(fm) < 1e-12 and layer.get_fw_flops() == 0:
        notes = 'flop_mult=0 (absorb WUK/WUV)'
    return batch_seq, c_in, c_out, wm, fm, notes


def _recover_bmm_shape(
    layer: BatchMatMul, app: Llm.Application, exe: Llm.Execution,
) -> Tuple[int, int, int, int, str]:
    """Best-effort (batch, m, n, k) from DS-V3 absorb naming + sizes."""
    # BatchMatMul: inputs = batch*(m*n + n*k), output = batch*m*k,
    # fw_flops = batch*2*m*n*k
    name = layer.name
    mbs = exe.microbatch_size
    tp = exe.tensor_par
    heads_tp = app.attn_heads // tp
    S = app.seq_size
    batch = mbs * heads_tp

    # Known absorb shapes from _build_mla_attn_block.
    table = {
        'QAbsorb': (batch, S, app.qk_nope_head_dim, app.kv_lora_rank),
        'ScoreKV': (batch, S, app.kv_lora_rank, S),
        'ScorePE': (batch, S, app.qk_rope_head_dim, S),
        'AttnKV': (batch, S, S, app.kv_lora_rank),
        'VAbsorb': (batch, S, app.kv_lora_rank, app.v_head_dim),
        'Key_Query': (batch, S, app.qk_nope_head_dim + app.qk_rope_head_dim, S),
        'Attn': (batch, S, S, app.v_head_dim),
    }
    for key, shape in table.items():
        if key in name:
            b, m, n, k = shape
            expect = b * 2 * m * n * k
            if abs(expect - layer.get_fw_flops()) < 1.0:
                return b, m, n, k, ''
            return b, m, n, k, f'flops_mismatch expect={expect:.0f}'
    return 0, 0, 0, 0, 'unknown_bmm'


def _stage_flops_bytes(layer, stage: str) -> Tuple[float, float]:
    if stage == 'fw':
        return float(layer.get_fw_flops()), float(layer.get_fw_mem_accessed())
    if stage == 'agrad':
        return float(layer.get_agrad_flops()), float(layer.get_agrad_mem_accessed())
    if stage == 'wgrad':
        return float(layer.get_wgrad_flops()), float(layer.get_wgrad_mem_accessed())
    raise ValueError(stage)


def layer_to_rows(
    layer,
    block: str,
    app: Llm.Application,
    exe: Llm.Execution,
    syst: System,
    stages: Sequence[str],
) -> List[CatalogRow]:
    cls = layer.__class__.__name__
    if cls in SKIP_CLASSES:
        return []
    group = assign_group(layer.name, cls)
    engine = 'matrix' if layer.use_matrix_engine() else 'vector'
    batch_seq = int(getattr(exe, 'microbatch_size', 1) * app.seq_size)

    c_in = c_out = 0
    bmm_batch = bmm_m = bmm_n = bmm_k = 0
    wm = fm = 1.0
    act_size = 0
    notes = ''

    if isinstance(layer, Linear):
        batch_seq, c_in, c_out, wm, fm, notes = _recover_linear_shape(
            layer, batch_seq)
    elif isinstance(layer, BatchMatMul):
        bmm_batch, bmm_m, bmm_n, bmm_k, notes = _recover_bmm_shape(
            layer, app, exe)
    elif isinstance(layer, (LayerNorm, RMSNorm, SoftMax, GeLU, SiLU)):
        act_size = int(layer.inputs_size)
        if isinstance(layer, SoftMax) and getattr(layer, '_fused', False):
            notes = (notes + '; ' if notes else '') + (
                'KNOWN_GAP: fused into flash-attn (isolated softmax not charged)')


    rows: List[CatalogRow] = []
    for stage in stages:
        flops, nbytes = _stage_flops_bytes(layer, stage)
        # Skip empty wgrad on layers without weights.
        if stage == 'wgrad' and flops == 0 and getattr(layer, 'weight_space', 0) == 0:
            continue
        pred_f = float(layer.compute_flops_time(stage))
        pred_m = float(layer.compute_mem_time(stage))
        pred_max = syst.get_processing_time(pred_f, pred_m)
        ai = flops / nbytes if nbytes > 0 else float('inf')
        bound = 'compute' if pred_f >= pred_m else 'memory'
        if math.isinf(ai):
            ai_out = -1.0
        else:
            ai_out = ai
        rows.append(CatalogRow(
            block=block, group=group, name=layer.name, cls=cls, engine=engine,
            stage=stage, flops=flops, bytes=nbytes, ai=ai_out,
            gflops=flops / 1e9,
            pred_f_s=pred_f, pred_m_s=pred_m, pred_max_s=pred_max,
            bound=bound,
            batch_seq=batch_seq, c_in=c_in, c_out=c_out,
            bmm_batch=bmm_batch, bmm_m=bmm_m, bmm_n=bmm_n, bmm_k=bmm_k,
            weight_mult=wm, flop_mult=fm, act_size=act_size, notes=notes,
        ))
    return rows


def build_catalog(
    llm: Llm,
    app: Llm.Application,
    exe: Llm.Execution,
    syst: System,
    stages: Sequence[str] = STAGES,
    blocks: Sequence[str] = ('dense', 'moe'),
) -> List[CatalogRow]:
    rows: List[CatalogRow] = []
    mapping = []
    if 'dense' in blocks and llm._dense_layers is not None:
        mapping.append(('dense', llm._dense_layers))
    if 'moe' in blocks and llm._moe_layers is not None:
        mapping.append(('moe', llm._moe_layers))
    if not mapping:
        mapping.append(('block', list(llm._llm_block)))

    seen = set()
    for block, layers in mapping:
        for layer in layers:
            key = (block, layer.name)
            if key in seen:
                continue
            seen.add(key)
            rows.extend(layer_to_rows(layer, block, app, exe, syst, stages))
    return rows


def write_csv(path: str, rows: Sequence[CatalogRow]) -> None:
    if not rows:
        print('No rows to write.')
        return
    fieldnames = list(asdict(rows[0]).keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f'Wrote CSV: {path}  ({len(rows)} rows)')


def print_table(rows: Sequence[CatalogRow], stage: str = 'fw') -> None:
    sub = [r for r in rows if r.stage == stage]
    hdr = (f'{"blk":5s} {"grp":3s} {"name":36s} {"cls":12s} {"eng":3s} '
           f'{"GF":>9s} {"MB":>8s} {"t_us":>9s} {"bd":1s} {"fm":>5s}')
    print(hdr)
    print('-' * len(hdr))
    for r in sub:
        print(
            f'{r.block:5s} {r.group:3s} {r.name:36s} {r.cls:12s} {r.engine[:3]:3s} '
            f'{r.gflops:9.2f} {r.bytes/1e6:8.2f} {r.pred_max_s*1e6:9.1f} '
            f'{r.bound[0]:1s} {r.flop_mult:5.2f}'
            + (f'  {r.notes}' if r.notes else '')
        )


def summarize(rows: Sequence[CatalogRow], app: Llm.Application, exe: Llm.Execution) -> None:
    fw = [r for r in rows if r.stage == 'fw']
    print('\n=== Phase2 catalog summary ===')
    print(f'seq_size={app.seq_size}  mbs={exe.microbatch_size}  '
          f'batch_seq={exe.microbatch_size * app.seq_size}  '
          f'TP={exe.tensor_par} EP={exe.expert_par}')
    if app.is_moe:
        active = app.moe_topk / exe.expert_par + app.num_shared_experts
        stored = app.num_experts // exe.expert_par + app.num_shared_experts
        print(f'MoE active_equiv={active:.2f}  experts_stored/rank={stored}')
    print(f'FW ops: {len(fw)}')
    for g in ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'other'):
        sub = [r for r in fw if r.group == g]
        if not sub:
            continue
        t = sum(r.pred_max_s for r in sub) * 1e6
        print(f'  {g}: N={len(sub):2d}  Σpred_max={t:10.1f} us')
    g6 = [r for r in fw if r.group == 'G6']
    if g6:
        ok = all(r.flops == 0 for r in g6)
        print(f'G6 WUK/WUV fw_flops==0: {"PASS" if ok else "FAIL"}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=os.path.join(_ROOT, 'models/deepseek-v3-671b.json'))
    p.add_argument('--system', default=os.path.join(_ROOT, 'systems/H20.json'))
    p.add_argument('--matrix-dtype', default='float8')
    p.add_argument('--vector-dtype', default='bfloat16')
    p.add_argument('--seq-size', type=int, default=None,
                   help='Override model seq_size (smaller helps G2 memory).')
    p.add_argument('--microbatch-size', type=int, default=1)
    p.add_argument('--expert-par', type=int, default=1)
    p.add_argument('--blocks', nargs='+', default=['dense', 'moe'],
                   choices=['dense', 'moe'])
    p.add_argument('--stages', nargs='+', default=['fw'],
                   choices=list(STAGES))
    p.add_argument('--csv', default=os.path.join(_TEST_DIR, 'phase2_dsv3_catalog.csv'))
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    llm, app, syst, exe = compile_dsv3(
        args.model, args.system,
        matrix_dtype=args.matrix_dtype,
        vector_dtype=args.vector_dtype,
        seq_size=args.seq_size,
        microbatch_size=args.microbatch_size,
        expert_par=args.expert_par,
    )
    print(f'System: {args.system}')
    print(f'Model:  {args.model}')
    print(f'matrix_dtype={args.matrix_dtype}  vector_dtype={args.vector_dtype}  '
          f'processing_mode={getattr(syst, "proc_mode", "?")}')

    rows = build_catalog(
        llm, app, exe, syst, stages=args.stages, blocks=args.blocks)
    if not args.quiet:
        print_table(rows, stage=args.stages[0])
    summarize(rows, app, exe)
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == '__main__':
    main()
