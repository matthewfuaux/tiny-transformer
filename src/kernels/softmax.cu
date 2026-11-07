#include "kernels/softmax.cuh"

#include "cuda_utils.cuh"
#include <cmath>

__global__ void softmax_kernel(float *out, const float *in, int cols) {
  __shared__ float smem[256];
  const float *row_in = in + blockIdx.x * cols;
  float *row_out = out + blockIdx.x * cols;

  float local_max = -INFINITY;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    local_max = fmaxf(local_max, row_in[j]);
  }

  smem[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  float row_max = smem[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    local_sum += expf(row_in[j] - row_max);
  }

  smem[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float row_sum = smem[0];
  __syncthreads();

  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    row_out[j] = expf(row_in[j] - row_max) / row_sum;
  }
}

void softmax_forward(float *out, const float *in, int rows, int cols,
                     cudaStream_t stream) {
  if (rows <= 0 || cols <= 0)
    return;

  softmax_kernel<<<rows, 256, 0, stream>>>(out, in, cols);
  CUDA_CHECK_KERNEL();
}
