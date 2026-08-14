// Native INT8 GEMM timing backend for BW1100 (gfx938).
// The operand/output types deliberately use the standard INT8 x INT8 -> INT32
// hipBLASLt contract.  This keeps the measurement on the integer matrix path,
// instead of measuring a PyTorch promotion/fallback kernel.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdio>
#include <cstdlib>
#include <limits>

#define HIP_CHECK(expr) do { hipError_t s = (expr); if (s != hipSuccess) { \
  std::fprintf(stderr, "HIP failure at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(s)); return 2; }} while (0)
#define LT_CHECK(expr) do { hipblasStatus_t s = (expr); if (s != HIPBLAS_STATUS_SUCCESS) { \
  std::fprintf(stderr, "hipBLASLt failure at %s:%d: %d\n", __FILE__, __LINE__, int(s)); return 3; }} while (0)

__global__ void fill_int8(signed char* data, size_t count) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) data[i] = 1;
  return;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s M N K warmup iterations\n", argv[0]);
    return 1;
  }
  const int m = std::atoi(argv[1]), n = std::atoi(argv[2]), k = std::atoi(argv[3]);
  const int warmup = std::atoi(argv[4]), iterations = std::atoi(argv[5]);
  if (m <= 0 || n <= 0 || k <= 0 || warmup < 0 || iterations <= 0) return 1;

  signed char *a = nullptr, *b = nullptr;
  int* d = nullptr;
  void* workspace = nullptr;
  HIP_CHECK(hipMalloc(&a, size_t(m) * k));
  HIP_CHECK(hipMalloc(&b, size_t(k) * n));
  HIP_CHECK(hipMalloc(&d, size_t(m) * n * sizeof(int)));
  fill_int8<<<(size_t(m) * k + 255) / 256, 256>>>(a, size_t(m) * k);
  fill_int8<<<(size_t(k) * n + 255) / 256, 256>>>(b, size_t(k) * n);
  HIP_CHECK(hipGetLastError());

  hipblasLtHandle_t handle;
  hipblasLtMatmulDesc_t op;
  hipblasLtMatrixLayout_t A, B, C, D;
  hipblasLtMatmulPreference_t pref;
  LT_CHECK(hipblasLtCreate(&handle));
  LT_CHECK(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32I, HIP_R_32I));
  LT_CHECK(hipblasLtMatrixLayoutCreate(&A, HIP_R_8I, m, k, m));
  LT_CHECK(hipblasLtMatrixLayoutCreate(&B, HIP_R_8I, k, n, k));
  LT_CHECK(hipblasLtMatrixLayoutCreate(&C, HIP_R_32I, m, n, m));
  LT_CHECK(hipblasLtMatrixLayoutCreate(&D, HIP_R_32I, m, n, m));
  LT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
  size_t workspace_bytes = 64ULL << 20;
  HIP_CHECK(hipMalloc(&workspace, workspace_bytes));
  LT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
  constexpr int max_algorithms = 64;
  hipblasLtMatmulHeuristicResult_t results[max_algorithms]{};
  int algo_count = 0;
  LT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle, op, A, B, C, D, pref, max_algorithms, results, &algo_count));
  if (!algo_count) {
    std::fprintf(stderr, "No INT8 hipBLASLt algorithm for %dx%dx%d\n", m, n, k);
    return 4;
  }
  const int alpha = 1, beta = 0;
  auto gemm = [&](const hipblasLtMatmulAlgo_t& algo) { return hipblasLtMatmul(handle, op, &alpha, a, A, b, B, &beta, d, C, d, D,
                                                                                &algo, workspace, workspace_bytes, 0); };
  int best = -1;
  float best_ms = std::numeric_limits<float>::infinity();
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
  for (int candidate = 0; candidate < algo_count; ++candidate) {
    if (results[candidate].state != HIPBLAS_STATUS_SUCCESS || results[candidate].workspaceSize > workspace_bytes) continue;
    if (gemm(results[candidate].algo) != HIPBLAS_STATUS_SUCCESS) continue;
    HIP_CHECK(hipEventRecord(start));
    for (int probe = 0; probe < 3; ++probe) if (gemm(results[candidate].algo) != HIPBLAS_STATUS_SUCCESS) break;
    HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
    float probe_ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&probe_ms, start, stop));
    if (probe_ms < best_ms) { best_ms = probe_ms; best = candidate; }
  }
  if (best < 0) {
    std::fprintf(stderr, "No runnable INT8 hipBLASLt algorithm for %dx%dx%d (candidates=%d)\n", m, n, k, algo_count);
    return 4;
  }
  std::fprintf(stderr, "INT8 algo: selected %d/%d (probe %.3f ms, waves %.2f)\n", best, algo_count, best_ms / 3.0f, results[best].wavesCount);
  auto run_best = [&]() { return gemm(results[best].algo); };
  for (int i = 0; i < warmup; ++i) LT_CHECK(run_best());
  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < iterations; ++i) LT_CHECK(run_best());
  HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
  const double latency_s = elapsed_ms / 1000.0 / iterations;
  std::printf("{\"latency_s\":%.9f,\"tflops\":%.6f}\n", latency_s, 2.0 * m * n * k / latency_s / 1e12);
}
