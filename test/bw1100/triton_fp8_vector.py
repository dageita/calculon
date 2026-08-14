"""Standalone fused FP8 elementwise kernel for BW1100 (gfx938).

DTK 26.04 PyTorch does not expose FP8 elementwise arithmetic.  Expressing the
operation as ``x.to(bfloat16) * y.to(bfloat16) + x.to(bfloat16)`` launches
several conversion and arithmetic kernels and therefore does not calibrate an
FP8 vector operator.  This Triton kernel keeps FP8 at the memory boundary and
fuses conversion, arithmetic, and the FP8 store into one GPU launch.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_vector_fma(x_ptr, y_ptr, output_ptr, count, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < count
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float16)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float16)
    output = x * y + x
    # output_ptr is an E4M3 tensor, so tl.store performs the final FP8 cast.
    tl.store(output_ptr + offsets, output, mask=mask)


def benchmark_fp8_vector(count: int, warmup: int, iterations: int) -> float:
    """Return seconds per fused E4M3 ``x*y+x`` launch."""
    storage = torch.float8_e4m3fn
    x = torch.ones(count, device="cuda", dtype=torch.bfloat16).to(storage)
    y = torch.ones(count, device="cuda", dtype=torch.bfloat16).to(storage)
    output = torch.empty(count, device="cuda", dtype=storage)
    # A 1024-element/4-warp program is ~39% faster than the conservative
    # 256-element launch on the physical gfx938 device.
    block_size = 1024
    grid = (triton.cdiv(count, block_size),)

    def run() -> None:
        _fp8_vector_fma[grid](
            x, y, output, count,
            BLOCK_SIZE=block_size,
            num_warps=4,
            num_stages=1,
        )

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    # DTK PyTorch does not implement _local_scalar_dense_cuda for FP8.
    # Convert a one-element slice before moving it to the host instead.
    value = float(output[:1].float().cpu()[0])
    if value != 2.0:
        raise RuntimeError(f"FP8 Triton correctness check failed: output[0]={value}, expected 2.0")

    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        run()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / 1000.0 / iterations
