#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda_utils.cuh"
#include "kernels/elementwise.cuh"
#include "test_utils.h"

static float gelu_cpu(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608f;
  return 0.5f * x *
         (1.0f + std::tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

static bool test_add(int n) {
  auto a = random_vector(n, -1.0f, 1.0f, /*seed=*/1);
  auto b = random_vector(n, -1.0f, 1.0f, /*seed=*/2);
  std::vector<float> expected(n);
  for (int i = 0; i < n; ++i)
    expected[i] = a[i] + b[i];

  DeviceBuffer<float> d_a(a), d_b(b), d_out(n);
  add_forward(d_out.data(), d_a.data(), d_b.data(), n);

  char name[64];
  std::snprintf(name, sizeof(name), "add n=%d", n);
  return all_close(expected, d_out.to_host(), name, 0.0f,
                   0.0f); // float add is exact
}

static bool test_gelu(int n) {
  auto in = random_vector(n, -5.0f, 5.0f);
  std::vector<float> expected(n);
  for (int i = 0; i < n; ++i)
    expected[i] = gelu_cpu(in[i]);

  DeviceBuffer<float> d_in(in), d_out(n);
  gelu_forward(d_out.data(), d_in.data(), n);

  char name[64];
  std::snprintf(name, sizeof(name), "gelu n=%d", n);
  return all_close(expected, d_out.to_host(), name, 1e-5f, 1e-5f);
}

int main() {
  bool ok = true;

  for (int n : {1, 7, 255, 256, 257, 1000, 1 << 20, 10'000'019}) {
    ok &= test_add(n);
    ok &= test_gelu(n);
  }

  std::printf("\n%s\n", ok ? "ALL PASSED" : "SOME TESTS FAILED");
  return ok ? 0 : 1;
}
