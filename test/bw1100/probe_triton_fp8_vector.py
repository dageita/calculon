#!/usr/bin/env python3
"""Temporary launch-configuration probe for the BW1100 FP8 vector kernel."""
import torch
import triton

from triton_fp8_vector import _fp8_vector_fma


count = 64 << 20
storage = torch.float8_e4m3fn
x = torch.ones(count, device="cuda", dtype=torch.bfloat16).to(storage)
y = torch.ones(count, device="cuda", dtype=torch.bfloat16).to(storage)
output = torch.empty_like(x)

for block_size, num_warps in ((256, 4), (512, 4), (512, 8), (1024, 4), (1024, 8), (2048, 8)):
    grid = (triton.cdiv(count, block_size),)

    def run():
        _fp8_vector_fma[grid](x, y, output, count, BLOCK_SIZE=block_size,
                              num_warps=num_warps, num_stages=1)

    try:
        for _ in range(5):
            run()
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(20):
            run()
        stop.record(); stop.synchronize()
        latency = start.elapsed_time(stop) / 1000.0 / 20
        print(block_size, num_warps, latency * 1e6, count / latency / 1e9)
    except Exception as error:
        print(block_size, num_warps, "ERROR", error)
