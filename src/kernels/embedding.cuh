#pragma once

void embedding_forward(float *out, const int *tokens, const float *wte,
                       const float *wpe, int B, int T, int C,
                       cudaStream_t stream = 0);
