# tiny-transformer

A small GPT-style transformer written from scratch in C++/CUDA, trained on
Tiny Shakespeare. No PyTorch, no autograd: every kernel and every gradient is
written by hand and tested against a CPU reference.

**Current phase: 1 — scaffolding and first kernel.**

## Prerequisites

- An NVIDIA driver recent enough for your toolkit (any current 5090 driver is fine).
- **CUDA Toolkit 12.8 or newer.** The RTX 5090 is Blackwell (compute capability
  12.0, `sm_120`); older toolkits can't compile for it.
- CMake 3.24+ and a C++17 compiler.

Pick one environment:

| Environment     | Compiler                                  | Notes                                                                                                                              |
| --------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Native Linux    | g++                                       | Simplest.                                                                                                                          |
| WSL2 on Windows | g++                                       | Install the CUDA toolkit _for WSL_ inside Ubuntu; do **not** install a Linux GPU driver inside WSL — the Windows driver is shared. |
| Native Windows  | Visual Studio 2022 (Desktop C++ workload) | Install VS first, then the CUDA toolkit so it registers with VS. Build from the "x64 Native Tools" prompt.                         |

## Build and run

```bash
cmake -S . -B build
cmake --build build -j

./build/device_info                     # 1. sanity check: should list your 5090
./build/test_elementwise                # 2. first kernel test + benchmark
ctest --test-dir build --output-on-failure   # runs every test
```

On native Windows (multi-config generator), add `--config Release` to the build
command, `-C Release` to ctest, and the executables are in `build\Release\`.

Debug build that synchronizes after every kernel so crashes point at the
right launch:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DSYNC_AFTER_KERNEL=ON
cmake --build build-debug -j
```

### What you should see

- `device_info`: compute capability 12.0, ~170 SMs, ~32 GiB, ~1.8 TB/s peak bandwidth.
- `test_elementwise`: every line `[ OK ]`, then a bandwidth table. A simple add
  is purely memory-bound, so a good result is roughly 80–90% of the peak that
  `device_info` printed. Notice how little block size matters here — that
  changes once kernels use shared memory.

## Layout

```
CMakeLists.txt
src/
  cuda_utils.cuh          CUDA_CHECK, DeviceBuffer (RAII GPU memory), GpuTimer
  test_utils.h            random_vector, all_close (tolerance comparison)
  kernels/
    elementwise.cuh/.cu   add_forward  (each kernel = __global__ fn + C++ launcher)
tests/
  test_elementwise.cu     CPU reference vs GPU, odd sizes, in-place, benchmark
tools/
  device_info.cu          prints GPU properties
```

Every new kernel follows the same pattern: header declaring a plain C++
launcher, `.cu` with the kernel and launcher, a CPU reference in the test,
then `add_kernel_test(...)` in CMakeLists.txt.

## Concepts in this phase

Read `src/kernels/elementwise.cu` top to bottom; the comments carry the ideas.

1. **Host vs device memory.** The CPU and GPU have separate memory.
   `cudaMalloc` allocates on the GPU; `cudaMemcpy` moves data across.
   `DeviceBuffer` wraps both so you can't leak or forget.
2. **Grid → blocks → threads.** A launch `kernel<<<grid, block>>>` runs the
   same function on `grid × block` threads. Each thread computes its global
   index `blockIdx.x * blockDim.x + threadIdx.x`.
3. **Bounds checks.** The grid is rounded up, so the last block overhangs the
   array. `if (i < n)` is mandatory.
4. **Asynchrony.** Launches return immediately; errors inside a kernel surface
   later. Hence `CUDA_CHECK_KERNEL()` and the `SYNC_AFTER_KERNEL` option.
5. **Test against the CPU.** Every kernel gets an obviously-correct CPU twin
   and a test with awkward sizes.

## Exercises (do these before Phase 2)

**1. Break it on purpose.** Delete the `if (i < n)` in `add_kernel`, rebuild,
and run the test under NVIDIA's memory checker:

```bash
compute-sanitizer ./build/test_elementwise
```

The test may still pass — but the sanitizer reports the out-of-bounds writes.
That's why the sanitizer is part of the workflow. Put the check back.

**2. Write GELU yourself.** The transformer's MLP uses GELU (tanh approximation):

```
gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))
```

- Add `gelu_forward(float* out, const float* in, int n, ...)` to
  `elementwise.cuh/.cu`, following the shape of `add_forward`.
- In the test, write `gelu_cpu` and test the same odd sizes with inputs in
  `[-5, 5]`.
- Use `tanhf` on the GPU. Unlike add, results won't be bit-identical to the
  CPU, so use a tolerance like `atol = rtol = 1e-5`. Think about why.

## Roadmap

1. ~~Scaffolding, first kernel~~ ← you are here
2. Forward kernels: matmul (naive → tiled), softmax, LayerNorm, embedding
3. Assemble the forward pass; validate against PyTorch
4. Backward pass; gradient checks
5. Loss, AdamW, training loop, sampling on Tiny Shakespeare
6. Optimization: profiling, fused kernels, FlashAttention-style attention, BF16
