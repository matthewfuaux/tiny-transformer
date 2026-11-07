#pragma once

#include <cuda_runtime.h>

void softmax_forward(float *out, const float *in, int rows, int cols,
                     cudaStream_t stream = 0);
