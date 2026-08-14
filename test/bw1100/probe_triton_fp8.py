#!/usr/bin/env python3
import sys
import torch
from lmslim.layers.gemm.fp8_utils import triton_scaled_mm_fp8

m = int(sys.argv[1]); n = int(sys.argv[2]); k = int(sys.argv[3])
a = torch.ones((m, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
# Function expects B[K,N]. A transposed allocation provides column-contiguous B.
b = torch.ones((n, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
sa = torch.ones((1,), device="cuda", dtype=torch.float32)
sb = torch.ones((1,), device="cuda", dtype=torch.float32)
def run(): return triton_scaled_mm_fp8(a, b, sa, sb, torch.bfloat16)
for _ in range(2): out = run()
torch.cuda.synchronize(); start, stop = torch.cuda.Event(True), torch.cuda.Event(True); start.record()
for _ in range(10): out = run()
stop.record(); stop.synchronize(); latency = start.elapsed_time(stop)/1000/10
print(out.shape, out.dtype, out[0,0].item(), latency, 2*m*n*k/latency/1e12)
