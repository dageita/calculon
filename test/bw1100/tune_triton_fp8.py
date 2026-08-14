#!/usr/bin/env python3
"""Small gfx938 tile search for lmslim's FP8 Triton GEMM kernel."""
import itertools
import sys
import torch
import triton
import triton.language as tl
from lmslim.layers.gemm.fp8_utils import scaled_mm_kernel_fp8

m = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
n = int(sys.argv[2]) if len(sys.argv) > 2 else m
k = int(sys.argv[3]) if len(sys.argv) > 3 else m
a = torch.ones((m, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
b = torch.ones((n, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
sa = torch.ones((1,), device="cuda", dtype=torch.float32)
sb = torch.ones((1,), device="cuda", dtype=torch.float32)

def launch(bm, bn, bk, warps):
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    scaled_mm_kernel_fp8[grid](
        a, b, sa, sb, c, None, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        tl.float32, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
        BLOCK_SIZE_SCALE_A=1, BLOCK_SIZE_SCALE_B=1,
        num_warps=warps, num_stages=1,
    )

for bm, bn, bk, warps in itertools.product((128, 256), (128, 256), (64, 128), (4, 8, 16)):
    try:
        launch(bm, bn, bk, warps); torch.cuda.synchronize()
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True); start.record()
        for _ in range(3): launch(bm, bn, bk, warps)
        stop.record(); stop.synchronize(); latency = start.elapsed_time(stop) / 1000 / 3
        value = c[0, 0].item()
        print(f"BM={bm} BN={bn} BK={bk} W={warps}: {2*m*n*k/latency/1e12:.3f} TF, {latency*1e6:.1f} us, value={value}", flush=True)
    except Exception as exc:
        print(f"BM={bm} BN={bn} BK={bk} W={warps}: ERROR {type(exc).__name__}: {str(exc)[:120]}", flush=True)
