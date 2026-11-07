#include "kernels/elementwise.cuh"

#include "cuda_utils.cuh"
#include <cmath>

// -----------------------------------------------------------------------------
// The kernel: code that runs on the GPU.
//
// `__global__` marks a function the CPU can launch on the GPU. The launch
// creates a *grid* of *blocks*, each block made of *threads*. Every thread
// runs this same function; the only thing that differs between them is
// their position, read from built-in variables:
//
//   blockIdx.x   which block this thread is in      (0 .. gridDim.x - 1)
//   blockDim.x   threads per block                  (256 here)
//   threadIdx.x  this thread's position in its block (0 .. blockDim.x - 1)
//
// So `blockIdx.x * blockDim.x + threadIdx.x` gives every thread in the grid a
// unique global index, and we let thread i handle element i.
//
// No `out` overlaps between threads, so threads never need to coordinate.
// Most transformer kernels aren't this simple; softmax and LayerNorm need
// threads to cooperate on a reduction, which is what makes them interesting.
// -----------------------------------------------------------------------------
__global__ void add_kernel(float *out, const float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // The grid is rounded up to whole blocks, so the last block usually has
  // threads past the end of the array. Without this check they'd read and
  // write out of bounds, and it might even *appear* to work. Always guard.
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

// -----------------------------------------------------------------------------
// The launcher: ordinary C++ on the CPU that picks a launch configuration.
// Keeping launch details here means the rest of the code just calls a
// normal-looking function.
// -----------------------------------------------------------------------------
void add_forward(float *out, const float *a, const float *b, int n,
                 cudaStream_t stream, int block_size) {
  if (n <= 0)
    return;
  int grid_size = ceil_div(n, block_size);

  // <<<blocks, threads per block, dynamic shared memory bytes, stream>>>
  // This returns immediately; the GPU runs the kernel asynchronously.
  add_kernel<<<grid_size, block_size, 0, stream>>>(out, a, b, n);
  CUDA_CHECK_KERNEL();
}

__global__ void gelu_kernel(float *out, const float *in, int n) {
  constexpr float kSqrt2OverPi = 0.7978845608f;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    out[i] = 0.5f * x * (1 + tanhf(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
  }
}

void gelu_forward(float *out, const float *in, int n, cudaStream_t stream,
                  int block_size) {
  if (n <= 0)
    return;

  int grid_size = ceil_div(n, block_size);

  gelu_kernel<<<grid_size, block_size, 0, stream>>>(out, in, n);
  CUDA_CHECK_KERNEL();
}
