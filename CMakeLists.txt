cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

project(tiny_transformer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SYNC_AFTER_KERNEL "Synchronize after every kernel launch" OFF)

add_library(kernels STATIC
  src/kernels/elementwise.cu
  src/kernels/softmax.cu
  src/kernels/layernorm.cu
  src/kernels/embedding.cu
)
target_include_directories(kernels PUBLIC src)
# -lineinfo maps GPU instructions back to source lines in Nsight Compute.
target_compile_options(kernels PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
if(SYNC_AFTER_KERNEL)
  target_compile_definitions(kernels PUBLIC SYNC_AFTER_KERNEL)
endif()

add_executable(device_info tools/device_info.cu)
target_include_directories(device_info PRIVATE src)

enable_testing()

function(add_kernel_test name)
  add_executable(${name} tests/${name}.cu)
  target_link_libraries(${name} PRIVATE kernels)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

add_kernel_test(test_elementwise)
add_kernel_test(test_softmax)
add_kernel_test(test_layernorm)
add_kernel_test(test_embedding)
