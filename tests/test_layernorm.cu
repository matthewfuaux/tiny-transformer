#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda_utils.cuh"
#include "kernels/layernorm.cuh"
#include "test_utils.h"

static std::vector<float> layernorm_cpu(const std::vector<float> &in,
                                        const std::vector<float> &weight,
                                        const std::vector<float> &bias,
                                        int rows, int cols) {
  constexpr double kEps = 1e-5;
  std::vector<float> out(rows * cols);
  for (int r = 0; r < rows; r++) {
    const float *x = &in[r * cols];

    double sum = 0.0;
    for (int c = 0; c < cols; c++)
      sum += x[c];
    double mean = sum / cols;

    double sq = 0.0;
    for (int c = 0; c < cols; c++) {
      double d = x[c] - mean;
      sq += d * d;
    }
    double var = sq / cols;
    double rstd = 1.0 / std::sqrt(var + kEps);

    for (int c = 0; c < cols; c++)
      out[r * cols + c] = (float)((x[c] - mean) * rstd * weight[c] + bias[c]);
  }
  return out;
}

static std::vector<float> layernorm_gpu(const std::vector<float> &in,
                                        const std::vector<float> &weight,
                                        const std::vector<float> &bias,
                                        int rows, int cols) {
  DeviceBuffer<float> d_in(in), d_w(weight), d_b(bias), d_out(rows * cols);
  layernorm_forward(d_out.data(), d_in.data(), d_w.data(), d_b.data(), rows,
                    cols);
  return d_out.to_host();
}

static bool test_layernorm(int rows, int cols) {
  auto in = random_vector(rows * cols, -5.0f, 5.0f, /*seed=*/1);
  auto weight = random_vector(cols, 0.5f, 1.5f, /*seed=*/2);
  auto bias = random_vector(cols, -1.0f, 1.0f, /*seed=*/3);

  auto expected = layernorm_cpu(in, weight, bias, rows, cols);
  auto actual = layernorm_gpu(in, weight, bias, rows, cols);

  char name[64];
  std::snprintf(name, sizeof(name), "layernorm %dx%d", rows, cols);
  return all_close(expected, actual, name, 1e-5f, 1e-5f);
}

static bool test_layernorm_hand_checked() {
  std::vector<float> in = {3.0f, 4.0f, 5.0f};
  std::vector<float> weight = {1.0f, 1.0f, 1.0f};
  std::vector<float> bias = {0.0f, 0.0f, 0.0f};
  std::vector<float> expected = {-1.22473f, 0.0f, 1.22473f};

  auto actual = layernorm_gpu(in, weight, bias, 1, 3);
  return all_close(expected, actual, "layernorm [3,4,5] by hand", 1e-4f, 1e-4f);
}

static bool test_layernorm_constant_row() {
  int cols = 100;
  std::vector<float> in(cols, 7.0f);
  auto weight = random_vector(cols, 0.5f, 1.5f, 2);
  auto bias = random_vector(cols, -1.0f, 1.0f, 3);

  auto actual = layernorm_gpu(in, weight, bias, 1, cols);
  return all_close(bias, actual, "layernorm constant row", 1e-5f, 1e-5f);
}

struct Shape {
  int rows, cols;
};

int main() {
  bool ok = true;

  ok &= test_layernorm_hand_checked();
  ok &= test_layernorm_constant_row();

  for (Shape s : {Shape{1, 1}, Shape{1, 1000}, Shape{1000, 1}, Shape{64, 257},
                  Shape{4096, 256}}) {
    ok &= test_layernorm(s.rows, s.cols);
  }

  std::printf("\n%s\n", ok ? "ALL PASSED" : "SOME TESTS FAILED");
  return ok ? 0 : 1;
}
