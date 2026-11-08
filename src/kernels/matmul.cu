#include "cuda_utils.cuh"

__global__ void matmul_kernel(float *out, const float *A, const float *B, int M,
                              int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N)
    return;
  float sum = 0.0f;
  for (int k = 0; k < K; k++) {
    sum += A[row * K + k] * B[k * N + col];
  }
  out[row * N + col] = sum;
}

void matmul_forward(float *out, const float *A, const float *B, int M, int N,
                    int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0)
    return;

  dim3 block(16, 16);
  dim3 grid(ceil_div(N, 16), ceil_div(M, 16));
  matmul_kernel<<<grid, block, 0, stream>>>(out, A, B, M, N, K);
  CUDA_CHECK_KERNEL();
}
