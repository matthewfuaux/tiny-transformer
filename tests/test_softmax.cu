#include "cuda_utils.cuh"
#include "kernels/softmax.cuh"
#include "test_utils.h"
#include <cmath>
#include <cstdio>
#include <vector>

static std::vector<float> softmax_cpu(const std::vector<float> &in, int rows,
                                      int cols) {
  std::vector<float> out(rows * cols);
  for (int r = 0; r < rows; r++) {
    float max = -INFINITY;
    for (int c = 0; c < cols; c++) {
      max = fmaxf(max, in[r * cols + c]);
    }

    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
      sum += expf(in[r * cols + c] - max);
    }

    for (int c = 0; c < cols; c++) {
      out[r * cols + c] = expf(in[r * cols + c] - max) / sum;
    }
  }

  return out;
}

static bool test_softmax(int rows, int cols) {
  auto in = random_vector(rows * cols, -100.0f, 100.0f);
  auto expected = softmax_cpu(in, rows, cols);

  DeviceBuffer<float> d_in(in), d_out(rows * cols);
  softmax_forward(d_out.data(), d_in.data(), rows, cols);

  char name[64];
  std::snprintf(name, sizeof(name), "softmax %dx%d", rows, cols);
  return all_close(expected, d_out.to_host(), name, 1e-5f, 1e-5f);
}
struct Shape {
  int rows, cols;
};

int main() {
  bool ok = true;

  for (Shape s : {Shape{1, 1}, Shape{1, 1000}, Shape{1000, 1}, Shape{64, 257},
                  Shape{4096, 256}}) {
    ok &= test_softmax(s.rows, s.cols);
  }

  std::printf("\n%s\n", ok ? "ALL PASSED" : "SOME TESTS FAILED");
  return ok ? 0 : 1;
}
