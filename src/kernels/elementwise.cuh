#pragma once

#include <cuda_runtime.h>

void add_forward(float *out, const float *a, const float *b, int n,
                 cudaStream_t stream = 0, int block_size = 256);

void gelu_forward(float *out, const float *in, int n, cudaStream_t stream = 0,
                  int block_size = 256);
