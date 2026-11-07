// Prints the properties of each GPU that matter for writing kernels.
// Run this first: if it works, your driver, toolkit and build are all fine.
#include <cstdio>

#include "cuda_utils.cuh"

int main() {
  int driver = 0, runtime = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  std::printf("CUDA driver %d.%d, runtime %d.%d\n", driver / 1000, (driver % 1000) / 10,
              runtime / 1000, (runtime % 1000) / 10);

  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  for (int d = 0; d < count; ++d) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, d));

    // Memory clock (kHz) via the attribute API; the cudaDeviceProp field was
    // removed in CUDA 13. DDR-style memory moves data twice per clock.
    int mem_clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, d));
    double peak_gbs = 2.0 * mem_clock_khz * 1e3 * (p.memoryBusWidth / 8) / 1e9;

    std::printf("\nDevice %d: %s\n", d, p.name);
    std::printf("  Compute capability       %d.%d\n", p.major, p.minor);
    std::printf("  Streaming multiprocessors %d\n", p.multiProcessorCount);
    std::printf("  Global memory            %.1f GiB\n", p.totalGlobalMem / double(1 << 30));
    std::printf("  Memory bus width         %d bits\n", p.memoryBusWidth);
    std::printf("  Peak memory bandwidth    ~%.0f GB/s (theoretical)\n", peak_gbs);
    std::printf("  L2 cache                 %.0f MiB\n", p.l2CacheSize / double(1 << 20));
    std::printf("  Warp size                %d\n", p.warpSize);
    std::printf("  Max threads per block    %d\n", p.maxThreadsPerBlock);
    std::printf("  Max threads per SM       %d\n", p.maxThreadsPerMultiProcessor);
    std::printf("  Shared memory per block  %zu KiB (opt-in up to %zu KiB)\n",
                p.sharedMemPerBlock / 1024, p.sharedMemPerBlockOptin / 1024);
    std::printf("  Shared memory per SM     %zu KiB\n", p.sharedMemPerMultiprocessor / 1024);
    std::printf("  Registers per block      %d\n", p.regsPerBlock);
  }
  return 0;
}
