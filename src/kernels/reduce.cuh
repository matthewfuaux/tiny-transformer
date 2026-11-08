#pragma once

__device__ inline float block_reduce_max(float val, float *smem) {
  smem[threadIdx.x] = val;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  float result = smem[0];
  __syncthreads();
  return result;
}

__device__ inline float block_reduce_sum(float val, float *smem) {
  smem[threadIdx.x] = val;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float result = smem[0];
  __syncthreads();
  return result;
}
