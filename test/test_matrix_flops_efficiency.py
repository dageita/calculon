import cupy as cp
import numpy as np
from time import perf_counter

def benchmark_gflops(matrix_size, dtype=cp.float16):
    # 初始化数据（使用FP16以匹配Tensor Core）
    a = cp.random.rand(matrix_size, matrix_size).astype(dtype)
    b = cp.random.rand(matrix_size, matrix_size).astype(dtype)
    
    # 预热GPU
    for _ in range(3):
        c = cp.matmul(a, b)
    
    # 正式测试
    start = perf_counter()
    iterations = 100
    for _ in range(iterations):
        c = cp.matmul(a, b)
    cp.cuda.Device().synchronize()
    elapsed = perf_counter() - start
    
    # 计算GFLOPS
    flops_per_matmul = 2 * matrix_size ** 3  # 2N^3 for matrix multiply
    total_flops = flops_per_matmul * iterations
    gflops = total_flops / (elapsed * 1e9)
    
    # A100 FP16峰值算力为312 TFLOPS
    peak_gflops = 119.5e3
    efficiency = gflops / peak_gflops
    
    return gflops, efficiency

# 测试不同矩阵大小（对应不同计算密度）
sizes = [4096, 2048, 1024, 512, 256, 64]  # 从高密度到低密度
results = []
for size in sizes:
    gflops, eff = benchmark_gflops(size)
    results.append([gflops, eff])

print("gflops_efficiency =", results)
