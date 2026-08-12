import cupy as cp
import numpy as np
import time
import pprint

def benchmark_matrix_multiplication(dtype, compute_densities):
    """
    优化后的矩阵乘法性能测试
    """
    results = []
    
    for density in compute_densities:
        # 计算矩阵尺寸
        M = int(round((density * 1e9 / 2) ** (1/3)))
        K = N = M
        
        # 调整尺寸为16的倍数
        M = max(16, (M // 16) * 16)
        N = max(16, (N // 16) * 16)
        K = max(16, (K // 16) * 16)
        
        # 计算实际运算量
        actual_density = 2 * M * N * K / 1e9
        
        # 减少迭代次数以避免内存不足
        iterations = max(1, min(100, int(100 * (16**3) / (M*N*K))))
        
        try:
            # 创建随机数生成器
            rng = cp.random.default_rng()
            
            # 预热
            for _ in range(3):
                A = rng.standard_normal((M, K), dtype=cp.float32).astype(dtype)
                B = rng.standard_normal((K, N), dtype=cp.float32).astype(dtype)
                C = cp.matmul(A, B)
                del A, B, C
                cp.cuda.Stream.null.synchronize()
            
            # 实际测试
            start = time.time()
            for _ in range(iterations):
                A = rng.standard_normal((M, K), dtype=cp.float32).astype(dtype)
                B = rng.standard_normal((K, N), dtype=cp.float32).astype(dtype)
                C = cp.matmul(A, B)
                del A, B, C
            cp.cuda.Stream.null.synchronize()
            elapsed = time.time() - start
            
            # 计算性能
            total_flops = 2 * M * N * K * iterations
            tflops = total_flops / elapsed / 1e12
            results.append((actual_density, tflops))
            
        except cp.cuda.memory.OutOfMemoryError:
            print(f"警告: matrix 内存不足，跳过 {actual_density:.1f} GFLOPS 的矩阵测试")
            continue
    
    return results

def benchmark_vector_operations(dtype, compute_densities):
    """
    优化后的向量操作性能测试
    """
    print(cp.get_default_memory_pool().used_bytes() / 1024**3, "GB used")
    results = []
    
    for density in compute_densities:
        N = int(density * 1e9)
        N = max(1024, N)
        actual_density = N / 1e9
        
        # 动态调整迭代次数 - 这里修正了括号不匹配的问题
        iterations = max(1, min(100, int(100 * (1e6 / N))))
        
        try:
            # 使用随机数生成器
            rng = cp.random.default_rng()
            
            # 预热
            for _ in range(3):
                A = rng.standard_normal(N, dtype=cp.float32).astype(dtype)
                B = rng.standard_normal(N, dtype=cp.float32).astype(dtype)
                C = A * B
                del A, B, C
                cp.cuda.Stream.null.synchronize()
            
            # 实际测试
            start = time.time()
            for _ in range(iterations):
                A = rng.standard_normal(N, dtype=cp.float32).astype(dtype)
                B = rng.standard_normal(N, dtype=cp.float32).astype(dtype)
                C = A * B
                del A, B, C
            cp.cuda.Stream.null.synchronize()
            elapsed = time.time() - start
            
            # 计算性能
            total_flops = N * iterations
            tflops = total_flops / elapsed / 1e12
            results.append((actual_density, tflops))
            
        except cp.cuda.memory.OutOfMemoryError:
            print(f"警告: vector 内存不足，跳过 {actual_density:.1f} GFLOPS 的向量测试")
            continue
    
    return results

def calculate_efficiency(results, peak_tflops):
    return [(density, tflops / peak_tflops) for density, tflops in results]

def main():
    # 调整测试范围以适应不同内存容量的GPU
    matrix_densities = [128, 64, 32, 16, 8, 4, 2, 1]  # GFLOPS
    vector_densities = [16, 8, 4, 2, 1, 0.5, 0.25]    # GFLOPS
    
    # 峰值性能设置
    matrix_peak_tflops = 120  # Tensor Core峰值(TFLOPS)
    vector_peak_tflops = 60   # CUDA Core峰值(TFLOPS)
    
    # 运行测试
    print("Running optimized matrix multiplication benchmarks...")
    matrix_results = benchmark_matrix_multiplication(cp.float16, matrix_densities)
    matrix_efficiency = calculate_efficiency(matrix_results, matrix_peak_tflops)
    
    print("Running optimized vector operation benchmarks...")
    vector_results = benchmark_vector_operations(cp.float16, vector_densities)
    vector_efficiency = calculate_efficiency(vector_results, vector_peak_tflops)
    
    # 添加边界情况
    matrix_efficiency.append((0, 0.1))
    vector_efficiency.append((0, 0.1))
    
    # 打印结果
    result = {
        "matrix": {
            "float16": {
                "tflops": matrix_peak_tflops,
                "gflops_efficiency": sorted(matrix_efficiency, key=lambda x: -x[0])
            }
        },
        "vector": {
            "float16": {
                "tflops": vector_peak_tflops,
                "gflops_efficiency": sorted(vector_efficiency, key=lambda x: -x[0])
            }
        }
    }
    
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result)

if __name__ == "__main__":
    # 初始化CuPy内存池
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    main()