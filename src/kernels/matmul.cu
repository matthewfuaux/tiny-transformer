#include "cuda_utils.cuh"

constexpr int TILE = 16;

__global__ void matmul_kernel(float *out, const float *A, const float *B, int M,
                              int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * TILE + tx;
  int row = blockIdx.y * TILE + ty;

  float sum = 0.0f;
  for (int t = 0; t < K; t += TILE) {
    int a_col = t + tx;
    int b_row = t + ty;
    As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
    __syncthreads();

    for (int k = 0; k < TILE; k++) {
      sum += As[ty][k] * Bs[k][tx];
    }
    // Wait before the next iteration overwrites the tiles.
    __syncthreads();
  }

  if (row < M && col < N)
    out[row * N + col] = sum;
}

void matmul_forward(float *out, const float *A, const float *B, int M, int N,
                    int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0)
    return;

  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  matmul_kernel<<<grid, block, 0, stream>>>(out, A, B, M, N, K);
  CUDA_CHECK_KERNEL();
}
