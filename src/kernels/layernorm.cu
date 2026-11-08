#include "cuda_utils.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/reduce.cuh"

__global__ void layernorm_kernel(float *out, const float *in,
                                 const float *weight, const float *bias,
                                 int cols) {
  constexpr float kLayerNormEps = 1e-5f;
  __shared__ float smem[256];
  const float *row_in = in + blockIdx.x * cols;
  float *row_out = out + blockIdx.x * cols;

  float local_sum = 0.0f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    local_sum += row_in[j];
  }

  float row_mean = block_reduce_sum(local_sum, smem) / cols;

  float local_var = 0.0f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    float d = row_in[j] - row_mean;
    local_var += d * d;
  }

  float var = block_reduce_sum(local_var, smem) / cols;
  float rstd = rsqrtf(var + kLayerNormEps);

  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    row_out[j] = (row_in[j] - row_mean) * rstd * weight[j] + bias[j];
  }
}

void layernorm_forward(float *out, const float *in, const float *weight,
                       const float *bias, int rows, int cols,
                       cudaStream_t stream) {
  if (cols <= 0 || rows <= 0)
    return;
  layernorm_kernel<<<rows, 256, 0, stream>>>(out, in, weight, bias, cols);
  CUDA_CHECK_KERNEL();
}
