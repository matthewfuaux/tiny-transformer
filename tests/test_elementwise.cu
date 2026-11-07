#include <cmath>          // std::tanh, std::fabs, std::isnan
#include <cstdio>         // printf
#include <cuda_runtime.h> // cudaMalloc, cudaMemcpy, cudaFree
#include <random>
#include <vector>

#include "kernels/elementwise.cuh" // gelu_forward

// CPU reference: same formula as the kernel, in plain float maths.
static float gelu_cpu(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608f;
  return 0.5f * x *
         (1.0f + std::tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

bool test_gelu(int n) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  std::vector<float> in(n);
  std::vector<float> out(n);

  for (auto &x : in)
    x = dist(rng);

  float *d_in = nullptr;
  cudaMalloc(&d_in, n * sizeof(float));

  float *d_out = nullptr;
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_in, in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  gelu_forward(d_out, d_in, n);
  cudaError_t err =
      cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);

  if (err != cudaSuccess) {
    printf("[FAIL] gelu n=%d: CUDA error: %s\n", n, cudaGetErrorString(err));
    return false;
  }

  // Compare every element against the CPU reference.
  const float atol = 1e-5f, rtol = 1e-5f;
  float max_err = 0.0f;
  int bad = 0, first_bad = -1;
  for (int i = 0; i < n; ++i) {
    float expected = gelu_cpu(in[i]);
    float diff = std::fabs(expected - out[i]);
    if (diff > max_err || std::isnan(diff))
      max_err = diff;
    if (!(diff <= atol + rtol * std::fabs(expected))) { // NaN fails too
      if (first_bad < 0)
        first_bad = i;
      ++bad;
    }
  }

  if (bad > 0) {
    printf("[FAIL] gelu n=%d: %d mismatches, first at [%d]: in %g expected %g "
           "got %g\n",
           n, bad, first_bad, in[first_bad], gelu_cpu(in[first_bad]),
           out[first_bad]);
    return false;
  }
  printf("[ OK ] gelu n=%d (max err %.3g)\n", n, max_err);
  return true;
}

int main() {
  bool ok = true;
  for (int n : {1, 7, 255, 256, 257, 1000, 1 << 20, 10'000'019})
    ok &= test_gelu(n);

  printf("\n%s\n", ok ? "ALL PASSED" : "SOME TESTS FAILED");
  return ok ? 0 : 1;
}
