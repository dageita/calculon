#!/usr/bin/env python3
"""One-shot probe for the vendor lightop gfx938 W8A8 assembly path."""
import sys
import time

import torch
import lightop

m = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
n = int(sys.argv[2]) if len(sys.argv) > 2 else m
k = int(sys.argv[3]) if len(sys.argv) > 3 else m
mode = sys.argv[4] if len(sys.argv) > 4 else "dense"
raw_int8 = mode == "grouped-int8"
dtype = torch.int8 if raw_int8 else torch.float8_e4m3fnuz
a = torch.full((m, k), 0x38 if raw_int8 else 1.0, device="cuda", dtype=dtype)
b = torch.full((n, k), 0x38 if raw_int8 else 1.0, device="cuda", dtype=dtype)
scale_a = torch.ones((max(1, (m + 127) // 128), max(1, (k + 127) // 128)), device="cuda")
scale_b = torch.ones((max(1, (n + 127) // 128), max(1, (k + 127) // 128)), device="cuda")
print("inputs", a.shape, b.shape, a.dtype, scale_a.shape, scale_b.shape, flush=True)
try:
  if mode in ("grouped", "grouped-int8"):
    # Vendor masked grouped API follows its bundled BF16/W8A8 tests: A and D
    # are concatenated 2-D rows, while B has a leading group dimension.
    b = b.reshape(1, n, k)
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    a_scale = torch.ones(m, device="cuda", dtype=torch.float32)
    b_scale = torch.ones((1, n), device="cuda", dtype=torch.float32)
    masked_m = torch.tensor([m], device="cuda", dtype=torch.int32)
    def kernel():
      return lightop.m_grouped_fp8_gemm_nt_masked(
          (a, a_scale), (b, b_scale), d, masked_m, m)
  else:
    def kernel():
      return lightop.gemm_w8a8_asm(a, b, scale_a, scale_b, (128, 128), torch.bfloat16)
    for _ in range(2):
        out = kernel()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(10):
        out = kernel()
    stop.record(); stop.synchronize()
    latency = start.elapsed_time(stop) / 1000 / 10
    print("result", out.shape, out.dtype, out[0, 0].item(), latency, 2*m*n*k/latency/1e12, flush=True)
except Exception as exc:
    print(type(exc).__name__, exc, flush=True)
