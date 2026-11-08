#include <cstdio>
#include <random>
#include <vector>

#include "cuda_utils.cuh"
#include "kernels/embedding.cuh"
#include "test_utils.h"

static std::vector<float> embedding_cpu(const std::vector<int> &tokens,
                                        const std::vector<float> &wte,
                                        const std::vector<float> &wpe, int B,
                                        int T, int C) {
  std::vector<float> out(B * T * C);
  for (int b = 0; b < B; b++)
    for (int t = 0; t < T; t++) {
      int id = tokens[b * T + t];
      for (int c = 0; c < C; c++)
        out[(b * T + t) * C + c] = wte[id * C + c] + wpe[t * C + c];
    }
  return out;
}

static std::vector<float> embedding_gpu(const std::vector<int> &tokens,
                                        const std::vector<float> &wte,
                                        const std::vector<float> &wpe, int B,
                                        int T, int C) {
  DeviceBuffer<int> d_tokens(tokens);
  DeviceBuffer<float> d_wte(wte), d_wpe(wpe), d_out(B * T * C);
  embedding_forward(d_out.data(), d_tokens.data(), d_wte.data(), d_wpe.data(),
                    B, T, C);
  return d_out.to_host();
}

static std::vector<int> random_tokens(int n, int V, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, V - 1);
  std::vector<int> v(n);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

static bool test_embedding_hand_checked() {
  std::vector<int> tokens = {2, 0, 1, 1, 2, 0};
  std::vector<float> wte = {1, 2, 3, 4, 5, 6};
  std::vector<float> wpe = {10, 10, 20, 20, 30, 30};
  std::vector<float> expected = {15, 16, 21, 22, 33, 34,
                                 13, 14, 25, 26, 31, 32};

  auto actual = embedding_gpu(tokens, wte, wpe, 2, 3, 2);
  return all_close(expected, actual, "embedding hand-checked", 0.0f, 0.0f);
}

static bool test_embedding(int B, int T, int C, int V) {
  auto tokens = random_tokens(B * T, V, 1);
  auto wte = random_vector(V * C, -1.0f, 1.0f, 2);
  auto wpe = random_vector(T * C, -1.0f, 1.0f, 3);

  auto expected = embedding_cpu(tokens, wte, wpe, B, T, C);
  auto actual = embedding_gpu(tokens, wte, wpe, B, T, C);

  char name[64];
  std::snprintf(name, sizeof(name), "embedding B=%d T=%d C=%d V=%d", B, T, C,
                V);
  return all_close(expected, actual, name, 0.0f, 0.0f);
}

int main() {
  bool ok = true;

  ok &= test_embedding_hand_checked();
  ok &= test_embedding(1, 1, 1, 1);
  ok &= test_embedding(2, 3, 5, 7);
  ok &= test_embedding(4, 128, 129, 65);
  ok &= test_embedding(32, 128, 128, 65);

  std::printf("\n%s\n", ok ? "ALL PASSED" : "SOME TESTS FAILED");
  return ok ? 0 : 1;
}
