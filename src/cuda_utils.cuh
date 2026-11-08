#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error: %s\n  in %s\n  at %s:%d\n",            \
                   cudaGetErrorString(err_), #call, __FILE__, __LINE__);       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#ifdef SYNC_AFTER_KERNEL
#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)
#else
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#endif

__host__ __device__ constexpr int ceil_div(int n, int b) {
  return (n + b - 1) / b;
}

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t n) : n_(n) {
    if (n_ > 0)
      CUDA_CHECK(cudaMalloc(&ptr_, n_ * sizeof(T)));
  }

  explicit DeviceBuffer(const std::vector<T> &host)
      : DeviceBuffer(host.size()) {
    copy_from_host(host);
  }

  ~DeviceBuffer() {
    if (ptr_)
      cudaFree(ptr_);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept : ptr_(other.ptr_), n_(other.n_) {
    other.ptr_ = nullptr;
    other.n_ = 0;
  }
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      if (ptr_)
        cudaFree(ptr_);
      ptr_ = other.ptr_;
      n_ = other.n_;
      other.ptr_ = nullptr;
      other.n_ = 0;
    }
    return *this;
  }

  T *data() { return ptr_; }
  const T *data() const { return ptr_; }
  size_t size() const { return n_; }
  size_t bytes() const { return n_ * sizeof(T); }

  void copy_from_host(const std::vector<T> &host) {
    check_size(host.size());
    CUDA_CHECK(cudaMemcpy(ptr_, host.data(), bytes(), cudaMemcpyHostToDevice));
  }

  std::vector<T> to_host() const {
    std::vector<T> host(n_);
    CUDA_CHECK(cudaMemcpy(host.data(), ptr_, bytes(), cudaMemcpyDeviceToHost));
    return host;
  }

  void zero() { CUDA_CHECK(cudaMemset(ptr_, 0, bytes())); }

private:
  void check_size(size_t n) const {
    if (n != n_) {
      std::fprintf(stderr, "DeviceBuffer size mismatch: buffer %zu, host %zu\n",
                   n_, n);
      std::exit(EXIT_FAILURE);
    }
  }

  T *ptr_ = nullptr;
  size_t n_ = 0;
};

class GpuTimer {
public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  GpuTimer(const GpuTimer &) = delete;
  GpuTimer &operator=(const GpuTimer &) = delete;

  void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start_, s)); }
  void stop(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(stop_, s)); }

  float elapsed_ms() {
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};
