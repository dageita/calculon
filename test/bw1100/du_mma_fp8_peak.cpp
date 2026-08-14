// Compute-only BW1100 FP8 HCU microbenchmark. This is diagnostic: unlike the
// GEMM backend it measures raw du_mma issue throughput without global memory.
#include <hip/hip_runtime.h>
#include <du_mma.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(expr) do { hipError_t s = (expr); if (s != hipSuccess) { \
  std::fprintf(stderr, "HIP failure: %s\n", hipGetErrorString(s)); return 2; }} while (0)
using namespace du::dumma;
using fp8 = __hip_fp8_e4m3;
using FragA = DUFragment<matrix_a, 16, 16, 32, fp8, row_major>;
using FragB = DUFragment<matrix_b, 16, 16, 32, fp8, col_major>;
using FragC = DUFragment<accumulator, 16, 16, 32, float>;

__global__ void peak_kernel(float* out, int repeats) {
  FragA a; FragB b; FragC c[8];
  du_fill_fragment(a, fp8(1.0f));
  du_fill_fragment(b, fp8(1.0f));
#pragma unroll
  for (int j = 0; j < 8; ++j) du_fill_fragment(c[j], 0.0f);
  for (int i = 0; i < repeats; ++i) {
#pragma unroll
    for (int j = 0; j < 8; ++j) du_mma_sync(c[j], a, b, c[j]);
  }
  if ((threadIdx.x & 63) == 0) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      out[(blockIdx.x * (blockDim.x / 64) + threadIdx.x / 64) * 8 + j] = c[j].x[0];
  }
  return;
}

int main(int argc, char** argv) {
  const int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int repeats = argc > 2 ? std::atoi(argv[2]) : 2048;
  float* out = nullptr; HIP_CHECK(hipMalloc(&out, size_t(blocks) * 4 * 8 * sizeof(float)));
  peak_kernel<<<blocks, 256>>>(out, 8); HIP_CHECK(hipDeviceSynchronize());
  hipEvent_t start, stop; HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start)); peak_kernel<<<blocks, 256>>>(out, repeats);
  HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  const double ops = double(blocks) * 4 * repeats * 8 * 2 * 16 * 16 * 32;
  std::printf("{\"blocks\":%d,\"repeats\":%d,\"ms\":%.4f,\"tflops\":%.3f}\n", blocks, repeats, ms, ops / (ms / 1000.0) / 1e12);
}
