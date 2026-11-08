#include "cuda_utils.cuh"

__global__ void embedding_kernel(float *out, const int *tokens,
                                 const float *wte, const float *wpe, int B,
                                 int T, int C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < B * T * C) {
    int c = i % C;
    int bt = i / C;
    int t = bt % T;
    int id = tokens[bt];
    out[i] = wte[id * C + c] + wpe[t * C + c];
  }
}

void embedding_forward(float *out, const int *tokens, const float *wte,
                       const float *wpe, int B, int T, int C,
                       cudaStream_t stream) {
  int n = B * T * C;
  if (n <= 0)
    return;
  embedding_kernel<<<ceil_div(n, 256), 256, 0, stream>>>(out, tokens, wte, wpe,
                                                         B, T, C);
  CUDA_CHECK_KERNEL();
}
