#!/usr/bin/env python3
"""Phase3 H4: fused block microbench.

Two implementations (``--fused-impl``):

  kernel_chain (default, fair H4)
      Run the *same* Phase2 physical kernels as Σiso back-to-back in **one**
      timed CUDA region (no sync between ops). fusion_gap isolates
      launch/scheduling fusion — not dtype mismatch.

  absorb (training-path style)
      Single DeepSeek MLA-absorb + SwiGLU forward (softmax in-region).
      Linears use FP8 ``_scaled_mm`` when available so GEMM dtype matches Σiso;
      BMM stays BF16.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from calibrate_h20_matrix_efficiency import _align, _gemm_fp8
from calibrate_h20_common import torch_storage_dtype


def _cuda_time(run: Callable[[], None], warmup: int, iters: int,
               min_ms: float, max_iters: int = 500) -> float:
    for _ in range(max(1, warmup)):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        run()
    end.record()
    torch.cuda.synchronize()
    probe_ms = start.elapsed_time(end) / 5.0
    n = max(iters, int(math.ceil(min_ms / max(probe_ms, 1e-6))))
    n = min(n, max_iters)
    start.record()
    for _ in range(n):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n / 1e3


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if hasattr(F, 'rms_norm'):
        return F.rms_norm(x, (x.shape[-1],), weight=w, eps=eps)
    rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x.float() * rms).to(x.dtype) * w


# ---------------------------------------------------------------------------
# kernel_chain: same ops as Σiso, one timed region
# ---------------------------------------------------------------------------

def _make_fp8_linear_run(m: int, n: int, k: int, repeats: int = 1):
    device = torch.device('cuda')
    storage = torch_storage_dtype('float8')
    m, n, k = _align(m), _align(n), _align(k)
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16).to(storage)
    b = torch.randn(n, k, device=device, dtype=torch.bfloat16).to(storage)
    scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
    scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    aux = torch.empty((), device=device, dtype=torch.float32)

    def run():
        for _ in range(repeats):
            _gemm_fp8(a, b, scale_a, scale_b, out_pair=(out, aux))

    return run


def _make_bmm_bf16_run(batch: int, m: int, n: int, k: int, repeats: int = 1):
    """BF16 bmm; repeats=2 for agrad (Calculon agrad_flops=2×fw)."""
    device = torch.device('cuda')
    m, n, k = _align(m), _align(n), _align(k)
    a0 = torch.randn(batch, m, n, device=device, dtype=torch.bfloat16)
    b0 = torch.randn(batch, n, k, device=device, dtype=torch.bfloat16)
    out = torch.empty(batch, m, k, device=device, dtype=torch.bfloat16)

    def run():
        for _ in range(repeats):
            torch.bmm(a0, b0, out=out)

    return run


def _make_rms_run(act_size: int, hidden: int):
    device = torch.device('cuda')
    tokens = max(1, act_size // max(hidden, 1))
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    w = torch.ones(hidden, device=device, dtype=torch.bfloat16)
    eps = 1e-6
    if hasattr(F, 'rms_norm'):
        def run():
            F.rms_norm(x, (hidden,), w, eps)
    else:
        def run():
            rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
            (x.float() * rms).to(x.dtype) * w
    return run


def build_kernel_chain_runs(
    catalog_rows, app, stage: str, matrix_dtype: str, active_equiv: float,
) -> Tuple[List[Callable[[], None]], str]:
    """Build zero-arg callables matching Phase2 physical tracks."""
    from phase2_dsv3_op_microbench import linear_gemm_mnk

    runs: List[Callable[[], None]] = []
    n_lin = n_bmm = n_norm = 0
    for r in catalog_rows:
        if r.flops <= 0 and r.pred_max_s <= 0:
            continue
        if r.cls == 'Linear' and r.c_in > 0 and r.c_out > 0:
            m_g, n_g, k_g = linear_gemm_mnk(stage, r.batch_seq, r.c_in, r.c_out)
            reps = 1
            if r.group == 'G4' and 'MoE_' in r.name and r.flop_mult > 1.01:
                reps = max(1, int(round(active_equiv)))
            # bench_linear uses (m, n=c_out, k=c_in) ≡ (m_g, n_g, k_g)
            runs.append(_make_fp8_linear_run(m_g, n_g, k_g, repeats=reps))
            n_lin += 1
        elif r.cls == 'BatchMatMul' and r.bmm_batch > 0:
            if stage == 'wgrad':
                continue
            reps = 2 if stage == 'agrad' else 1
            runs.append(_make_bmm_bf16_run(
                r.bmm_batch, r.bmm_m, r.bmm_n, r.bmm_k, repeats=reps))
            n_bmm += 1
        elif r.cls in ('LayerNorm', 'RMSNorm') and r.act_size > 0:
            hidden = app.hidden
            if 'QNorm' in r.name or 'KVNorm' in r.name:
                hidden = max(1, r.act_size // max(app.seq_size, 1))
            runs.append(_make_rms_run(r.act_size, hidden))
            n_norm += 1
    note = f'kernel_chain stage={stage} lin={n_lin} bmm={n_bmm} norm={n_norm}'
    return runs, note


def measure_kernel_chain(
    catalog_rows, app, stage: str, matrix_dtype: str, active_equiv: float,
    warmup: int, iters: int, min_ms: float, max_iters: int,
) -> Tuple[Optional[float], str]:
    if not torch.cuda.is_available():
        return None, 'no_cuda'
    runs, note = build_kernel_chain_runs(
        catalog_rows, app, stage, matrix_dtype, active_equiv)
    if not runs:
        return None, note + ' empty_chain'

    def run_all():
        for fn in runs:
            fn()

    try:
        lat = _cuda_time(run_all, warmup, iters, min_ms, max_iters)
        return lat, note + ' (one timed region, same kernels as Σiso)'
    except RuntimeError as e:
        return None, f'OOM_or_err:{e}'


# ---------------------------------------------------------------------------
# absorb: training-path MLA + FFN (FP8 Linear + BF16 BMM)
# ---------------------------------------------------------------------------

class _Fp8Weight:
    def __init__(self, out_f: int, in_f: int, device):
        storage = torch_storage_dtype('float8')
        self.w = torch.randn(out_f, in_f, device=device, dtype=torch.bfloat16).to(storage)
        self.scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
        self.out_f = out_f
        self.in_f = in_f

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        m, k = flat.shape
        # pad to alignment if needed
        m_a, n_a, k_a = _align(m), _align(self.out_f), _align(k)
        if (m, k) != (m_a, k_a) or self.out_f != n_a:
            # fallback bf16
            w_bf = self.w.to(torch.bfloat16)
            return F.linear(x, w_bf)
        a = flat.to(self.w.dtype)
        out = torch.empty(m_a, n_a, device=x.device, dtype=torch.bfloat16)
        aux = torch.empty((), device=x.device, dtype=torch.float32)
        # weight stored [N,K]; _gemm_fp8 expects b as [N,K] and does b.t()
        _gemm_fp8(a, self.w, self.scale_a, self.scale_b, out_pair=(out, aux))
        y = out[:m, :self.out_f]
        return y.view(*x.shape[:-1], self.out_f)


class _BlockWeights:
    def __init__(self, app, block: str, device: torch.device):
        H = int(app.hidden)
        heads = int(app.attn_heads)
        q_lora = int(app.q_lora_rank)
        kv_lora = int(app.kv_lora_rank)
        qk_nope = int(app.qk_nope_head_dim)
        qk_rope = int(app.qk_rope_head_dim)
        v_dim = int(app.v_head_dim)
        self.heads = heads
        self.kv_lora = kv_lora
        self.qk_nope = qk_nope
        self.qk_rope = qk_rope
        self.v_dim = v_dim
        self.softmax_scale = (qk_nope + qk_rope) ** -0.5
        dt = torch.bfloat16

        def p(*shape):
            return torch.randn(*shape, device=device, dtype=dt)

        self.attn_norm = p(H)
        self.mlp_norm = p(H)
        self.q_norm_w = p(q_lora)
        self.kv_norm_w = p(kv_lora)
        self.wdq = _Fp8Weight(q_lora, H, device)
        self.wuq = _Fp8Weight(heads * (qk_nope + qk_rope), q_lora, device)
        self.wdkv = _Fp8Weight(kv_lora + qk_rope, H, device)
        self.wuk = p(heads, qk_nope, kv_lora)
        self.wuv = p(heads, v_dim, kv_lora)
        self.wo = _Fp8Weight(H, heads * v_dim, device)
        self.block = block
        if block == 'dense':
            ff = int(app.feedforward)
            self.w_gate = _Fp8Weight(ff, H, device)
            self.w_up = _Fp8Weight(ff, H, device)
            self.w_down = _Fp8Weight(H, ff, device)
            self.active_equiv = 1
            self.w_router = None
        else:
            ff = int(app.moe_feedforward)
            n_exp = int(getattr(app, 'num_experts', 256) or 256)
            topk = int(getattr(app, 'moe_topk', 8) or 8)
            shared = int(getattr(app, 'num_shared_experts', 1) or 1)
            self.active_equiv = topk + shared
            self.w_router = _Fp8Weight(n_exp, H, device)
            self.w_gate = _Fp8Weight(ff, H, device)
            self.w_up = _Fp8Weight(ff, H, device)
            self.w_down = _Fp8Weight(H, ff, device)


def _swiglu_fp8(x, w_gate, w_up, w_down):
    return w_down(F.silu(w_gate(x)) * w_up(x))


def _mla_absorb_fw(x: torch.Tensor, w: _BlockWeights) -> torch.Tensor:
    B, S, _H = x.shape
    h = _rms_norm(x, w.attn_norm)
    q = w.wuq(_rms_norm(w.wdq(h), w.q_norm_w))
    q = q.view(B, S, w.heads, w.qk_nope + w.qk_rope)
    q_nope, q_pe = q.split([w.qk_nope, w.qk_rope], dim=-1)
    kv = w.wdkv(h)
    kv_a, k_pe = kv.split([w.kv_lora, w.qk_rope], dim=-1)
    kv_c = _rms_norm(kv_a, w.kv_norm_w)

    q_nope_b = q_nope.permute(2, 0, 1, 3).reshape(w.heads, B * S, w.qk_nope)
    q_abs = torch.bmm(q_nope_b, w.wuk).view(w.heads, B, S, w.kv_lora)
    q_abs = q_abs.permute(1, 2, 0, 3).contiguous()

    bh = B * w.heads
    q_kv = q_abs.permute(0, 2, 1, 3).reshape(bh, S, w.kv_lora)
    kv_t = (kv_c.unsqueeze(1).expand(-1, w.heads, -1, -1)
            .reshape(bh, S, w.kv_lora).transpose(1, 2))
    scores = torch.bmm(q_kv, kv_t)
    q_pe_b = q_pe.permute(0, 2, 1, 3).reshape(bh, S, w.qk_rope)
    pe_t = (k_pe.unsqueeze(1).expand(-1, w.heads, -1, -1)
            .reshape(bh, S, w.qk_rope).transpose(1, 2))
    scores = (scores + torch.bmm(q_pe_b, pe_t)) * w.softmax_scale
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
    ctx = torch.bmm(
        attn,
        kv_c.unsqueeze(1).expand(-1, w.heads, -1, -1).reshape(bh, S, w.kv_lora))
    ctx = ctx.view(B, w.heads, S, w.kv_lora).permute(0, 2, 1, 3)
    ctx_b = ctx.permute(2, 0, 1, 3).reshape(w.heads, B * S, w.kv_lora)
    v = torch.bmm(ctx_b, w.wuv.transpose(1, 2))
    v = v.view(w.heads, B, S, w.v_dim).permute(1, 2, 0, 3).reshape(B, S, -1)
    return w.wo(v)


def _ffn_fw(x: torch.Tensor, w: _BlockWeights) -> torch.Tensor:
    h = _rms_norm(x, w.mlp_norm)
    if w.block == 'dense':
        return _swiglu_fp8(h, w.w_gate, w.w_up, w.w_down)
    _ = w.w_router(h)
    out = torch.zeros_like(h)
    for _i in range(w.active_equiv):
        out = out + _swiglu_fp8(h, w.w_gate, w.w_up, w.w_down)
    return out / float(w.active_equiv)


def fused_block_forward(x: torch.Tensor, w: _BlockWeights) -> torch.Tensor:
    y = x + _mla_absorb_fw(x, w)
    y = y + _ffn_fw(y, w)
    return y


def measure_absorb(
    app, block: str, stage: str, microbatch: int,
    warmup: int, iters: int, min_ms: float, max_iters: int,
) -> Tuple[Optional[float], str]:
    if not torch.cuda.is_available():
        return None, 'no_cuda'
    if stage != 'fw':
        return None, 'absorb_impl_fw_only (use kernel_chain for agrad/wgrad)'
    device = torch.device('cuda')
    B, S, H = int(microbatch), int(app.seq_size), int(app.hidden)
    w = _BlockWeights(app, block, device)
    x = torch.randn(B, S, H, device=device, dtype=torch.bfloat16)

    def run():
        fused_block_forward(x, w)

    try:
        lat = _cuda_time(run, warmup, iters, min_ms, max_iters)
        kind = 'SwiGLU' if block == 'dense' else f'MoE×{w.active_equiv}'
        return lat, (
            f'absorb_fw MLA+{kind} linear=fp8,bmm=bf16,softmax_in_region '
            f'B={B} S={S}')
    except RuntimeError as e:
        return None, f'OOM_or_err:{e}'


def measure_fused_block(
    app,
    block: str,
    stage: str,
    microbatch: int = 1,
    warmup: int = 10,
    iters: int = 50,
    min_ms: float = 200.0,
    max_iters: int = 500,
    fused_impl: str = 'kernel_chain',
    catalog_rows: Optional[Sequence] = None,
    matrix_dtype: str = 'float8',
    active_equiv: float = 9.0,
) -> Tuple[Optional[float], str]:
    impl = (fused_impl or 'kernel_chain').strip().lower()
    if impl == 'absorb':
        return measure_absorb(
            app, block, stage, microbatch, warmup, iters, min_ms, max_iters)
    if impl in ('kernel_chain', 'chain', 'iso_chain'):
        if catalog_rows is None:
            return None, 'kernel_chain requires catalog_rows'
        return measure_kernel_chain(
            catalog_rows, app, stage, matrix_dtype, active_equiv,
            warmup, iters, min_ms, max_iters)
    return None, f'unknown_fused_impl:{fused_impl}'
