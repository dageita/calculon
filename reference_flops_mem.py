#!/usr/bin/env python3
"""Gold standard from DeepSeek-V3/inference/model.py — not from HF config alone.

Verifies calculon against the official inference implementation:
  https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

Covered (model.py semantics):
  - MLA fused weights (wq_a/wq_b, wkv_a/wkv_b, wo) + q_norm/kv_norm
  - Default attn_impl = \"absorb\" (naive optional via --attn-impl)
  - SwiGLU MLP / MoE (n_dense_layers prefix; moe_layer_freq must be 1)
  - Untied LM head (vocab * hidden)
  - Shared experts = one MLP with inter = n_shared * moe_inter_dim
  - Gate bias when dim == 7168
  - MTP: NOT in model.py; excluded by default (--include-mtp for HF estimate)

Usage:
  python3 reference_flops_mem.py
  python3 reference_flops_mem.py --attn-impl absorb   # default, matches model.py
  python3 reference_flops_mem.py --attn-impl naive
  python3 reference_flops_mem.py --include-mtp        # HF-only extras, not in model.py
  python3 reference_flops_mem.py --ep 8 --tp 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Path to official inference model (documentation / provenance only).
MODEL_PY = os.path.normpath(
  os.path.join(ROOT, '..', 'DeepSeek-V3', 'inference', 'model.py'))


# ---------------------------------------------------------------------------
# Gold formulas mirroring DeepSeek-V3/inference/model.py
# ---------------------------------------------------------------------------

def mla_weight_params_model_py(h, n_h, q_lora, kv_lora, qk_nope, qk_rope, v_dim):
  """MLA params as packed in model.py MLA.__init__ (plus RMSNorm scales).

  wq_a:  (q_lora, h)
  wq_b:  (n_h * qk, q_lora)          qk = nope + rope
  wkv_a: (kv_lora + rope, h)
  wkv_b: (n_h * (nope + v), kv_lora)
  wo:    (h, n_h * v)
  q_norm / kv_norm: scale-only
  """
  qk = qk_nope + qk_rope
  wq_a = h * q_lora
  wq_b = q_lora * (n_h * qk)
  wkv_a = h * (kv_lora + qk_rope)
  wkv_b = kv_lora * (n_h * (qk_nope + v_dim))
  wo = h * (n_h * v_dim)
  norms = q_lora + kv_lora  # q_norm + kv_norm
  return wq_a + wq_b + wkv_a + wkv_b + wo + norms


def mla_proj_flops_per_token(h, n_h, q_lora, kv_lora, qk_nope, qk_rope, v_dim,
                             attn_impl='absorb'):
  """Projection GEMM FLOPs per token (factor 2). Absorb does not run full wkv_b."""
  qk = qk_nope + qk_rope
  q_flops = 2 * (h * q_lora + q_lora * n_h * qk)       # wq_a + wq_b
  kv_a = 2 * h * (kv_lora + qk_rope)                    # wkv_a
  wo = 2 * h * n_h * v_dim
  if attn_impl == 'naive':
    wkv_b = 2 * kv_lora * n_h * (qk_nope + v_dim)
    return q_flops + kv_a + wkv_b + wo
  # absorb: wkv_b applied via q_absorb / v_absorb einsums (counted in scores)
  return q_flops + kv_a + wo


def mla_score_flops_per_token(n_h, kv_lora, qk_nope, qk_rope, v_dim, seq,
                              attn_impl='absorb'):
  """Attention-core FLOPs per token attending to `seq` context positions.

  Matches model.py forward() with T=1 decode (or per-token average of prefill
  upper-bounded by full seq — same as calculon's BatchMatMul /seq reporting).
  """
  if attn_impl == 'naive':
    qk = qk_nope + qk_rope
    # einsum QK + AttnV
    return 2 * n_h * seq * qk + 2 * n_h * seq * v_dim
  # absorb (default in model.py):
  #   q_nope @ wkv_b[:,:nope]     : 2 * H * nope * kv_lora
  #   q_latent · kv_cache          : 2 * H * kv_lora * seq
  #   q_pe · pe_cache              : 2 * H * rope * seq
  #   scores · kv_cache            : 2 * H * seq * kv_lora
  #   out @ wkv_b[:,-v:]           : 2 * H * kv_lora * v
  q_absorb = 2 * n_h * qk_nope * kv_lora
  scores = 2 * n_h * seq * (kv_lora + qk_rope) + 2 * n_h * seq * kv_lora
  v_absorb = 2 * n_h * kv_lora * v_dim
  return q_absorb + scores + v_absorb


def kv_cache_bytes_per_token(n_h, kv_lora, qk_nope, qk_rope, v_dim,
                             bytes_per_elem, attn_impl='absorb'):
  """KV cache footprint per cached token (model.py buffers)."""
  if attn_impl == 'naive':
    qk = qk_nope + qk_rope
    return (n_h * qk + n_h * v_dim) * bytes_per_elem  # k_cache + v_cache
  return (kv_lora + qk_rope) * bytes_per_elem           # kv_cache + pe_cache


def swiglu_ffn_params(h, f):
  return 3 * h * f


def reference_model(cfg, ep=1, bytes_per_elem=2, attn_impl='absorb',
                    include_mtp=False):
  """Gold metrics aligned with DeepSeek-V3/inference/model.py."""
  if attn_impl not in ('absorb', 'naive'):
    raise ValueError(f'attn_impl must be absorb|naive, got {attn_impl}')

  h = cfg['hidden']
  n_h = cfg['attn_heads']
  d = cfg.get('attn_size') or 128
  layers = cfg['num_blocks']
  vocab = cfg.get('vocab_size') or 51200
  seq = cfg['seq_size']

  num_experts = cfg.get('num_experts') or 0
  topk = cfg.get('moe_topk') or 0
  shared = cfg.get('num_shared_experts') or 0
  moe_f = cfg.get('moe_feedforward') or 0
  # model.py: n_dense_layers — HF first_k_dense_replace maps here
  first_k = cfg.get('first_k_dense') or cfg.get('n_dense_layers') or 0
  freq = cfg.get('moe_layer_freq') or 1
  f_dense = cfg['feedforward']
  mtp_layers = cfg.get('num_nextn_predict_layers') or 0

  q_lora = cfg.get('q_lora_rank') or 0
  kv_lora = cfg.get('kv_lora_rank') or 0
  qk_nope = cfg.get('qk_nope_head_dim') or d
  qk_rope = cfg.get('qk_rope_head_dim') or 0
  v_dim = cfg.get('v_head_dim') or d
  use_mla = q_lora > 0 and kv_lora > 0

  # model.py has no moe_layer_freq — every layer after n_dense_layers is MoE
  if num_experts > 0 and freq != 1:
    raise ValueError(
      f'moe_layer_freq={freq} is not supported by DeepSeek-V3/inference/model.py '
      f'(uses n_dense_layers only, equivalent to freq=1). Refusing gold compare.')

  if use_mla:
    attn_w = mla_weight_params_model_py(
      h, n_h, q_lora, kv_lora, qk_nope, qk_rope, v_dim)
  else:
    attn_w = 4 * h * n_h * d

  # Block RMSNorm: attn_norm + ffn_norm (scale only)
  ln_w = 2 * h
  # Final transformer norm
  final_norm_w = h

  is_moe = num_experts > 0
  # model.py: layer_id < n_dense_layers → MLP else MoE
  dense_layers = first_k if is_moe else layers
  moe_layers = (layers - first_k) if is_moe else 0

  gate_bias_w = 0
  if is_moe:
    dense_ffn_w = swiglu_ffn_params(h, f_dense)
    expert_w = swiglu_ffn_params(h, moe_f)
    # Shared = one MLP(inter = shared * moe_f) — same params as shared * expert
    shared_w = swiglu_ffn_params(h, shared * moe_f)
    router_w = h * num_experts
    # model.py Gate.bias only when dim == 7168
    gate_bias_w = num_experts if h == 7168 else 0
    moe_ffn_w_total = num_experts * expert_w + shared_w + router_w + gate_bias_w
    moe_ffn_w_active = topk * expert_w + shared_w + router_w + gate_bias_w
    assert num_experts % ep == 0, 'num_experts must be divisible by ep'
    experts_local = num_experts // ep
    moe_ffn_w_local = (
      experts_local * expert_w + shared_w + router_w + gate_bias_w)
  else:
    dense_ffn_w = 2 * h * f_dense  # legacy GeLU path if ever used
    moe_ffn_w_total = moe_ffn_w_active = moe_ffn_w_local = 0
    expert_w = shared_w = 0

  embed_w = vocab * h
  lm_head_w = vocab * h  # untied ColumnParallelLinear in model.py

  # MTP: absent from inference/model.py (convert.py skips layers.61)
  mtp_w = 0
  mtp_note = 'excluded (not in inference/model.py)'
  if include_mtp and mtp_layers > 0:
    # Approximate one next-n block ≈ one MoE transformer block (tech-report ~14B)
    one_block = attn_w + ln_w + (moe_ffn_w_total if is_moe else dense_ffn_w)
    mtp_w = mtp_layers * one_block
    mtp_note = f'included estimate ({mtp_layers}× transformer block ≈ {mtp_w/1e9:.2f}B)'
  elif mtp_layers > 0:
    mtp_note = (
      f'config has num_nextn_predict_layers={mtp_layers} but gold excludes MTP '
      f'to match model.py; pass --include-mtp for HF training estimate')

  total_params = (
    dense_layers * (attn_w + ln_w + dense_ffn_w) +
    moe_layers * (attn_w + ln_w + moe_ffn_w_total) +
    embed_w + lm_head_w + final_norm_w + mtp_w
  )
  activated_params = (
    dense_layers * (attn_w + ln_w + dense_ffn_w) +
    moe_layers * (attn_w + ln_w + moe_ffn_w_active) +
    embed_w + lm_head_w + final_norm_w + mtp_w
  )

  # ---- Per-token forward FLOPs ----
  if use_mla:
    attn_proj = mla_proj_flops_per_token(
      h, n_h, q_lora, kv_lora, qk_nope, qk_rope, v_dim, attn_impl)
    attn_core = mla_score_flops_per_token(
      n_h, kv_lora, qk_nope, qk_rope, v_dim, seq, attn_impl)
  else:
    attn_proj = 2 * attn_w
    attn_core = 2 * n_h * seq * d + 2 * n_h * seq * d

  attn_flops = attn_proj + attn_core

  if is_moe:
    dense_ffn_flops = 2 * dense_ffn_w
    dense_act_flops = 9 * f_dense
    router_flops = 2 * h * num_experts
    moe_ffn_flops = 2 * (topk * expert_w + shared_w)
    moe_act_flops = 9 * moe_f * (topk + shared)
    moe_ffn_flops_rank = 2 * ((topk / ep) * expert_w + shared_w)
    moe_act_flops_rank = 9 * moe_f * (topk / ep + shared)

    flops_dense_layer = attn_flops + dense_ffn_flops + dense_act_flops
    flops_moe_layer = (
      attn_flops + router_flops + moe_ffn_flops + moe_act_flops)
    flops_moe_layer_rank = (
      attn_flops + router_flops + moe_ffn_flops_rank + moe_act_flops_rank)
    fw_flops_per_token = (
      dense_layers * flops_dense_layer + moe_layers * flops_moe_layer)
    fw_flops_per_token_rank = (
      dense_layers * flops_dense_layer + moe_layers * flops_moe_layer_rank)
  else:
    fw_flops_per_token = layers * (attn_flops + 2 * dense_ffn_w + 8 * f_dense)
    fw_flops_per_token_rank = fw_flops_per_token

  # LM head: inference model.py only scores last token → 2*h*vocab once
  # Training would be per-token; gold uses inference (last-token) by default.
  lm_head_flops_infer = 2 * h * vocab
  fw_flops_per_token_with_head = fw_flops_per_token  # body only; head separate
  fw_flops_per_token_rank_with_head = fw_flops_per_token_rank

  bytes_local = (
    dense_layers * (attn_w + ln_w + dense_ffn_w) +
    moe_layers * (attn_w + ln_w + moe_ffn_w_local) +
    embed_w + lm_head_w + final_norm_w + mtp_w
  ) * bytes_per_elem
  bytes_unsharded = (
    dense_layers * (attn_w + ln_w + dense_ffn_w) +
    moe_layers * (attn_w + ln_w + moe_ffn_w_total) +
    embed_w + lm_head_w + final_norm_w + mtp_w
  ) * bytes_per_elem

  kv_tok = kv_cache_bytes_per_token(
    n_h, kv_lora, qk_nope, qk_rope, v_dim, bytes_per_elem, attn_impl
  ) if use_mla else (2 * n_h * d * bytes_per_elem)

  return {
    'gold_source': MODEL_PY if os.path.exists(MODEL_PY) else 'model.py (path missing)',
    'attn_impl': attn_impl,
    'use_mla': use_mla,
    'is_moe': is_moe,
    'dense_layers': dense_layers,
    'moe_layers': moe_layers,
    'mtp_note': mtp_note,
    'mtp_w': mtp_w,
    'attn_w_per_layer': attn_w,
    'dense_ffn_w_per_layer': dense_ffn_w if dense_layers else 0,
    'moe_ffn_w_total_per_layer': moe_ffn_w_total if is_moe else 0,
    'moe_ffn_w_active_per_layer': moe_ffn_w_active if is_moe else 0,
    'moe_ffn_w_local_per_layer': moe_ffn_w_local if is_moe else 0,
    'expert_w': expert_w if is_moe else 0,
    'embed_w': embed_w,
    'lm_head_w': lm_head_w,
    'final_norm_w': final_norm_w,
    'gate_bias_w': gate_bias_w,
    'total_params': total_params,
    'activated_params': activated_params,
    'local_params': bytes_local / bytes_per_elem,
    'fw_flops_per_token': fw_flops_per_token_with_head,
    'fw_flops_per_token_rank': fw_flops_per_token_rank_with_head,
    'lm_head_flops_infer': lm_head_flops_infer,
    'weight_bytes_ep_sharded': bytes_local,
    'weight_bytes_unsharded': bytes_unsharded,
    'kv_cache_bytes_per_token': kv_tok,
    'kv_cache_bytes_seq': kv_tok * seq,
  }


# ---------------------------------------------------------------------------
# Calculon side
# ---------------------------------------------------------------------------

def calculon_metrics(cfg, ep=1, tp=1, bytes_per_elem=2, attn_impl='absorb',
                     include_mtp=False):
  from calculon.llm.llm import Llm
  from calculon import System
  import logging

  log = logging.getLogger('ref')
  log.setLevel(logging.ERROR)

  cfg = dict(cfg)
  cfg['mla_attn_impl'] = attn_impl
  cfg['include_mtp'] = include_mtp
  app = Llm.Application(cfg)

  with open(os.path.join(ROOT, 'systems', 'a100_80g.json')) as f:
    sys_cfg = json.load(f)
  for net in sys_cfg['networks']:
    net.setdefault('topology', 'fully connected')
  sys_cfg['mem1']['GiB'] = 1_000_000
  sys_cfg['networks'][0]['size'] = max(tp * ep, 8)

  syst = System(sys_cfg, log)
  attn = 'mla' if (cfg.get('q_lora_rank') or 0) > 0 else 'multihead'
  num_procs = tp * ep
  exe = Llm.Execution.from_json({
    'num_procs': num_procs,
    'tensor_par': tp,
    'pipeline_par': 1,
    'data_par': 1,
    'expert_par': ep,
    'context_par': 1,
    'tensor_par_net': 0,
    'pipeline_par_net': 0,
    'data_par_net': 0,
    'expert_par_net': 0,
    'context_par_net': 0,
    'batch_size': 1,
    'microbatch_size': 1,
    'datatype': 'float16' if bytes_per_elem == 2 else 'float32',
    'fused_activation': False,
    'attention_type': attn,
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
  })

  model = Llm(app, log)
  model.compile(syst, exe)
  model._compute_block_stats()

  def layer_weight_params(layers):
    return sum(layer.get_weight() for layer in layers) / bytes_per_elem

  def layer_fw_flops(layers):
    return sum(layer.get_fw_flops() for layer in layers)

  # Embed + LM head + final norm live outside the block graph.
  extras = app.vocab_size * app.hidden  # embed
  extras += app.vocab_size * app.hidden  # untied LM head
  extras += app.hidden                   # final norm
  if include_mtp and (cfg.get('num_nextn_predict_layers') or 0) > 0:
    extras += app.mtp_params()

  if getattr(model, '_dense_layers', None) is not None:
    nd, nm = app.first_k_dense, app.num_moe_blocks
    w_dense = layer_weight_params(model._dense_layers)
    w_moe = layer_weight_params(model._moe_layers)
    total_params = nd * w_dense + nm * w_moe + extras
    weight_bytes = total_params * bytes_per_elem
    seq = app.seq_size
    fw_flops_per_token = (
      (nd * layer_fw_flops(model._dense_layers) +
       nm * layer_fw_flops(model._moe_layers)) / seq)
  else:
    extras_legacy = (app.vocab_size + app.seq_size) * app.hidden
    w_block = layer_weight_params(model._llm_block)
    total_params = app.num_blocks * w_block + extras_legacy
    weight_bytes = total_params * bytes_per_elem
    fw_flops_per_token = (
      app.num_blocks * layer_fw_flops(model._llm_block) / app.seq_size)

  return {
    'app_total_params': app.num_parameters(),
    'app_activated_params': app.num_activated_parameters(),
    'layer_total_params': total_params,
    'layer_fw_flops_per_token': fw_flops_per_token,
    'layer_weight_bytes': weight_bytes,
    'block_weight_space': model._block_weight_space,
  }


def pct_err(ref, got):
  if ref == 0:
    return 0.0 if got == 0 else float('inf')
  return abs(got - ref) / ref * 100.0


def fmt_params(n):
  if abs(n) >= 1e12:
    return f'{n/1e12:.3f}T'
  if abs(n) >= 1e9:
    return f'{n/1e9:.3f}B'
  if abs(n) >= 1e6:
    return f'{n/1e6:.3f}M'
  if abs(n) >= 1e3:
    return f'{n/1e3:.3f}K'
  return f'{n:.3f}'


def fmt_flops(n):
  if abs(n) >= 1e12:
    return f'{n/1e12:.3f} TFLOPs'
  if abs(n) >= 1e9:
    return f'{n/1e9:.3f} GFLOPs'
  if abs(n) >= 1e6:
    return f'{n/1e6:.3f} MFLOPs'
  if abs(n) >= 1e3:
    return f'{n/1e3:.3f} KFLOPs'
  return f'{n:.3f} FLOPs'


def fmt_bytes(n):
  if abs(n) >= 1e12:
    return f'{n/1e12:.3f} TB'
  if abs(n) >= 1e9:
    return f'{n/1e9:.3f} GB'
  if abs(n) >= 1e6:
    return f'{n/1e6:.3f} MB'
  if abs(n) >= 1e3:
    return f'{n/1e3:.3f} KB'
  return f'{n:.3f} B'


def main():
  ap = argparse.ArgumentParser(
    description='Compare calculon vs DeepSeek-V3/inference/model.py gold')
  ap.add_argument('--config', default=os.path.join(ROOT, 'models', 'deepseek-v3-671b.json'))
  ap.add_argument('--ep', type=int, default=1)
  ap.add_argument('--tp', type=int, default=1)
  ap.add_argument('--bytes', type=int, default=2, dest='bytes_per_elem')
  ap.add_argument('--attn-impl', choices=['absorb', 'naive'], default='absorb',
                  help='MLA attention impl; default absorb matches model.py')
  ap.add_argument('--include-mtp', action='store_true',
                  help='Add HF MTP estimate (NOT in inference/model.py)')
  ap.add_argument('--skip-calculon', action='store_true')
  ap.add_argument('--tol', type=float, default=5.0)
  args = ap.parse_args()

  with open(args.config) as f:
    cfg = json.load(f)

  ref = reference_model(
    cfg, ep=args.ep, bytes_per_elem=args.bytes_per_elem,
    attn_impl=args.attn_impl, include_mtp=args.include_mtp)

  print('=' * 72)
  print(f'Gold source: DeepSeek-V3/inference/model.py')
  print(f'  resolved: {ref["gold_source"]}')
  print(f'Reference: {os.path.basename(args.config)}')
  print(f'  MLA={ref["use_mla"]}  MoE={ref["is_moe"]}  attn_impl={ref["attn_impl"]}')
  print(f'  dense_layers={ref["dense_layers"]}  moe_layers={ref["moe_layers"]}')
  print(f'  EP={args.ep}  TP={args.tp}  bytes/elem={args.bytes_per_elem}')
  print(f'  MTP: {ref["mtp_note"]}')
  print('-' * 72)
  print(f'  attn_w/layer:          {fmt_params(ref["attn_w_per_layer"])}')
  if ref['is_moe']:
    print(f'  dense_ffn_w/layer:     {fmt_params(ref["dense_ffn_w_per_layer"])}')
    print(f'  moe_ffn_w total/layer: {fmt_params(ref["moe_ffn_w_total_per_layer"])}')
    print(f'  moe_ffn_w active/layer:{fmt_params(ref["moe_ffn_w_active_per_layer"])}')
    print(f'  moe_ffn_w local/layer: {fmt_params(ref["moe_ffn_w_local_per_layer"])} (EP-sharded)')
    if ref['gate_bias_w']:
      print(f'  gate bias (dim=7168):  {ref["gate_bias_w"]}')
  print(f'  embed:                 {fmt_params(ref["embed_w"])}')
  print(f'  LM head (untied):      {fmt_params(ref["lm_head_w"])}')
  print(f'  final RMSNorm:         {fmt_params(ref["final_norm_w"])}')
  if ref['mtp_w']:
    print(f'  MTP (estimate):        {fmt_params(ref["mtp_w"])}')
  print(f'  TOTAL params:         {fmt_params(ref["total_params"])}  '
        f'({ref["total_params"]/1e9:.2f}B params)')
  print(f'  ACTIVATED params:      {fmt_params(ref["activated_params"])}  '
        f'({ref["activated_params"]/1e9:.2f}B params)')
  print(f'  FW FLOPs/token (global): {fmt_flops(ref["fw_flops_per_token"])}')
  print(f'  FW FLOPs/token (per-rank):{fmt_flops(ref["fw_flops_per_token_rank"])}')
  print(f'  LM head FLOPs (infer last-tok): {fmt_flops(ref["lm_head_flops_infer"])}')
  print(f'  Weight bytes (EP={args.ep}): {fmt_bytes(ref["weight_bytes_ep_sharded"])}')
  print(f'  Weight bytes (unshard):  {fmt_bytes(ref["weight_bytes_unsharded"])}')
  print(f'  KV cache / token ({ref["attn_impl"]}): {fmt_bytes(ref["kv_cache_bytes_per_token"])}')
  print(f'  KV cache × seq:          {fmt_bytes(ref["kv_cache_bytes_seq"])}')

  if args.skip_calculon:
    return 0

  if args.tp > 1:
    print('=' * 72)
    print('NOTE: TP>1 shards MLA up/output; full layer-graph check needs --tp 1')

  print('=' * 72)
  print('Calculon vs model.py gold')
  print('-' * 72)
  try:
    calc = calculon_metrics(
      cfg, ep=args.ep, tp=args.tp, bytes_per_elem=args.bytes_per_elem,
      attn_impl=args.attn_impl, include_mtp=args.include_mtp)
  except Exception as e:
    print(f'  CALCULON FAILED: {type(e).__name__}: {e}')
    return 2

  rows = [
    ('app.num_parameters()', ref['total_params'], calc['app_total_params'],
     fmt_params),
    ('app.num_activated_parameters()', ref['activated_params'],
     calc['app_activated_params'], fmt_params),
    ('layer-graph local params', ref['local_params'], calc['layer_total_params'],
     fmt_params),
    ('layer-graph FW FLOPs/token/rank', ref['fw_flops_per_token_rank'],
     calc['layer_fw_flops_per_token'], fmt_flops),
    ('layer-graph weight bytes (EP)', ref['weight_bytes_ep_sharded'],
     calc['layer_weight_bytes'], fmt_bytes),
  ]
  if args.tp > 1:
    rows = [
      ('app.num_parameters()', ref['total_params'], calc['app_total_params'],
       fmt_params),
      ('app.num_activated_parameters()', ref['activated_params'],
       calc['app_activated_params'], fmt_params),
    ]

  print(f'  {"metric":<36} {"reference":>16} {"calculon":>16} {"err%":>8}')
  worst = 0.0
  for name, r, g, fmt_fn in rows:
    err = pct_err(r, g)
    worst = max(worst, err)
    flag = ' OK' if err <= args.tol else ' FAIL'
    print(f'  {name:<36} {fmt_fn(r):>16} {fmt_fn(g):>16} {err:>7.2f}%{flag}')

  print('-' * 72)
  if worst <= args.tol:
    print(f'PASS: max error {worst:.2f}% <= {args.tol}%')
    return 0
  print(f'FAIL: max error {worst:.2f}% > {args.tol}%')
  return 1


if __name__ == '__main__':
  sys.exit(main())
