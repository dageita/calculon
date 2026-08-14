"""Standalone gfx938 FP8 GEMM used by BW1100 calibration.

This avoids torch._scaled_mm (unsupported by this DTK PyTorch build) and the
DTK 26.04 hipBLASLt FP8 solution set that only reaches the generic ~84T path.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_gemm(
    a_ptr, b_ptr, c_ptr, m, n, k,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, tl.cdiv(k, BLOCK_K)):
        k_mask = offs_k < k - start_k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < m) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < n), other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


def benchmark_fp8_gemm(m: int, n: int, k: int, warmup: int, iterations: int) -> float:
    """Return seconds per valid FP8 GEMM A[M,K] @ B[K,N]."""
    storage = torch.float8_e4m3fn
    a = torch.ones((m, k), device="cuda", dtype=torch.bfloat16).to(storage)
    # Allocate B as [N,K], then transpose to obtain a coalesced [K,N] view.
    b = torch.ones((n, k), device="cuda", dtype=torch.bfloat16).to(storage).t()
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    block_m = 64 if m < 128 else 128
    block_n = 64 if n < 128 else 128
    block_k = 32 if k <= 32 else 64 if k <= 64 else 128
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    def run() -> None:
        _fp8_gemm[grid](
            a, b, c, m, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=4, num_stages=1,
        )

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    value = float(c[0, 0])
    if abs(value - k) > max(1.0, 0.01 * k):
        raise RuntimeError(f"FP8 Triton correctness check failed: C[0,0]={value}, expected {k}")
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        run()
    stop.record(); stop.synchronize()
    return start.elapsed_time(stop) / 1000.0 / iterations
