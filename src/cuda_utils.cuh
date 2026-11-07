// Small helpers used everywhere: error checking, RAII device memory, timing.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

// -----------------------------------------------------------------------------
// Error checking
//
// Almost every CUDA API call returns a cudaError_t. Ignoring it is the #1
// source of "my program prints garbage and I don't know why". Wrap every call.
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error: %s\n  in %s\n  at %s:%d\n",            \
                   cudaGetErrorString(err_), #call, __FILE__, __LINE__);       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Kernel launches (kernel<<<...>>>) don't return an error code. Two kinds of
// error can happen:
//   1. Launch errors (bad grid size, too much shared memory): reported
//      immediately by cudaGetLastError().
//   2. Errors while the kernel runs (out-of-bounds access): reported only at
//      the next synchronizing call, because launches are asynchronous.
// Call this right after every launch. Build with -DSYNC_AFTER_KERNEL=ON to
// also catch kind 2 at the exact launch that caused it.
#ifdef SYNC_AFTER_KERNEL
#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)
#else
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#endif

// Integer division rounding up: how many blocks of size b cover n elements.
__host__ __device__ constexpr int ceil_div(int n, int b) {
  return (n + b - 1) / b;
}

// -----------------------------------------------------------------------------
// DeviceBuffer<T>: owns an array in GPU memory.
//
// RAII ("resource acquisition is initialization"): the constructor allocates,
// the destructor frees, so memory can't leak even on early returns. Copying is
// disabled (two owners would double-free); moving transfers ownership.
// -----------------------------------------------------------------------------
template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t n) : n_(n) {
    if (n_ > 0)
      CUDA_CHECK(cudaMalloc(&ptr_, n_ * sizeof(T)));
  }

  // Allocate and upload a host vector in one step.
  explicit DeviceBuffer(const std::vector<T> &host)
      : DeviceBuffer(host.size()) {
    copy_from_host(host);
  }

  ~DeviceBuffer() {
    if (ptr_)
      cudaFree(ptr_); // no CUDA_CHECK: destructors must not exit
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
    // cudaMemcpy waits for all previously launched kernels to finish first,
    // so this is also where asynchronous kernel errors tend to surface.
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

// -----------------------------------------------------------------------------
// GpuTimer: measures GPU time with CUDA events.
//
// Don't time kernels with a CPU clock alone: the launch returns immediately,
// before the GPU has done any work. Events are recorded into the GPU's work
// queue, so the time between them is the time the GPU actually spent.
// -----------------------------------------------------------------------------
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
