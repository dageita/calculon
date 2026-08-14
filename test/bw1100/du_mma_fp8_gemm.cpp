// BW1100 gfx938 FP8 GEMM through DTK's native HCU du_mma interface.
// Fast path: row-major A[M,K] * B[K,N] -> FP32 D[M,N], dimensions divisible
// by 256/128/32. One 512-thread block computes a 256x128 output tile.
#include <hip/hip_runtime.h>
#include <du_mma.h>

#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr) do { hipError_t s = (expr); if (s != hipSuccess) { \
  std::fprintf(stderr, "HIP failure at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(s)); return 2; }} while (0)

using namespace du::dumma;
using fp8 = __hip_fp8_e4m3;
using FragA = DUFragment<matrix_a, 16, 16, 32, fp8, row_major>;
using FragB = DUFragment<matrix_b, 16, 16, 32, fp8, col_major>;
using FragC = DUFragment<accumulator, 16, 16, 32, float>;

__global__ void fill_fp8(unsigned char* data, size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) data[i] = 0x38; // Standard E4M3 representation of 1.0.
  return;
}

__global__ __launch_bounds__(512) void fp8_gemm_256x128x32(
    const fp8* __restrict__ a, const fp8* __restrict__ b,
    float* __restrict__ d, int m, int n, int k) {
  __shared__ __align__(16) fp8 as[256][32];
  __shared__ __align__(16) fp8 bs[32][128];
  const int tid = threadIdx.x;
  const int wave = tid >> 6;
  const int wave_m = (wave >> 1) * 64;
  const int wave_n = (wave & 1) * 64;
  const int block_m = blockIdx.y * 256;
  const int block_n = blockIdx.x * 128;

  FragC acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
      du_fill_fragment(acc[i][j], 0.0f);

  for (int kk = 0; kk < k; kk += 32) {
    // Coalesced 16-byte moves feed the 12 KiB tile.
    const int a_row = tid >> 1;
    const int a_col = (tid & 1) << 4;
    *reinterpret_cast<uint4*>(&as[a_row][a_col]) =
        *reinterpret_cast<const uint4*>(a + (block_m + a_row) * k + kk + a_col);
    if (tid < 256) {
      const int b_row = tid >> 3;
      const int b_col = (tid & 7) << 4;
      *reinterpret_cast<uint4*>(&bs[b_row][b_col]) =
          *reinterpret_cast<const uint4*>(b + (kk + b_row) * n + block_n + b_col);
    }
    __syncthreads();

    FragA af[4];
    FragB bf[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      du_load_matrix_sync(af[i], &as[wave_m + 16 * i][0], 32);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      du_load_matrix_sync(bf[j], &bs[0][wave_n + 16 * j], 128);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        du_mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
      du_store_matrix_sync(
          d + (block_m + wave_m + 16 * i) * n + block_n + wave_n + 16 * j,
          acc[i][j], n, mem_row_major);
  return;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s M N K warmup iterations\n", argv[0]);
    return 1;
  }
  const int m = std::atoi(argv[1]), n = std::atoi(argv[2]), k = std::atoi(argv[3]);
  const int warmup = std::atoi(argv[4]), iterations = std::atoi(argv[5]);
  if (m <= 0 || n <= 0 || k <= 0 || warmup < 0 || iterations <= 0 ||
      m % 256 || n % 128 || k % 32) {
    std::fprintf(stderr, "du_mma fast path requires M divisible by 256, N by 128, K by 32\n");
    return 1;
  }
  unsigned char *a = nullptr, *b = nullptr;
  float* d = nullptr;
  HIP_CHECK(hipMalloc(&a, size_t(m) * k));
  HIP_CHECK(hipMalloc(&b, size_t(k) * n));
  HIP_CHECK(hipMalloc(&d, size_t(m) * n * sizeof(float)));
  fill_fp8<<<(size_t(m) * k + 255) / 256, 256>>>(a, size_t(m) * k);
  fill_fp8<<<(size_t(k) * n + 255) / 256, 256>>>(b, size_t(k) * n);
  HIP_CHECK(hipGetLastError());
  const dim3 grid(n / 128, m / 256);
  auto run = [&]() { fp8_gemm_256x128x32<<<grid, 512>>>(reinterpret_cast<fp8*>(a), reinterpret_cast<fp8*>(b), d, m, n, k); };
  for (int i = 0; i < warmup; ++i) run();
  HIP_CHECK(hipDeviceSynchronize());
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < iterations; ++i) run();
  HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
  const double latency_s = elapsed_ms / 1000.0 / iterations;
  float first = 0.0f;
  HIP_CHECK(hipMemcpy(&first, d, sizeof(first), hipMemcpyDeviceToHost));
  if (first != float(k)) {
    std::fprintf(stderr, "du_mma FP8 correctness failure: D[0]=%.1f expected=%d\n", first, k);
    return 5;
  }
  std::printf("{\"backend\":\"du_mma\",\"latency_s\":%.9f,\"tflops\":%.6f}\n",
              latency_s, 2.0 * m * n * k / latency_s / 1e12);
}
