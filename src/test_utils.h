// Host-side helpers for kernel tests: random data and result comparison.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

// Fixed seed by default, so a failing test fails the same way every run.
inline std::vector<float> random_vector(size_t n, float lo = -1.0f, float hi = 1.0f,
                                        unsigned seed = 1234) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

// Element passes if |expected - actual| <= atol + rtol * |expected|.
// GPU and CPU results are rarely bit-identical (different instruction order,
// fused multiply-add, fast math functions), so we always compare with a
// tolerance. Picking the tolerance is part of designing each test.
inline bool all_close(const std::vector<float>& expected, const std::vector<float>& actual,
                      const char* name, float atol = 1e-5f, float rtol = 1e-5f) {
  if (expected.size() != actual.size()) {
    std::printf("[FAIL] %s: size mismatch (%zu vs %zu)\n", name, expected.size(),
                actual.size());
    return false;
  }
  float max_err = 0.0f;
  size_t bad = 0, first_bad = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    float err = std::fabs(expected[i] - actual[i]);
    if (err > max_err || std::isnan(actual[i])) max_err = std::isnan(actual[i]) ? INFINITY : err;
    if (!(err <= atol + rtol * std::fabs(expected[i]))) {  // also catches NaN
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  if (bad) {
    std::printf("[FAIL] %s: %zu/%zu mismatches, max err %.3g, first at [%zu]: "
                "expected %.7g got %.7g\n",
                name, bad, expected.size(), max_err, first_bad, expected[first_bad],
                actual[first_bad]);
    return false;
  }
  std::printf("[ OK ] %s (n=%zu, max err %.3g)\n", name, expected.size(), max_err);
  return true;
}
