#pragma once
#include <cuda_runtime.h>

void layernorm_forward(float *out, const float *in, const float *weight,
                       const float *bias, int rows, int cols,
                       cudaStream_t stream = 0);
