cmake_minimum_required(VERSION 3.18)
project(kuiper_style_llm LANGUAGES CXX)
option(KLLM_USE_CUDA "Build CUDA backend" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(KLLM_USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86" CACHE STRING "CUDA architectures" FORCE)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()
add_library(kllm_core
  src/base/half.cpp
  src/base/tensor.cpp
  src/op/dispatch.cpp
  src/op/cpu/add.cpp
  src/op/cpu/attention.cpp
  src/op/cpu/embedding.cpp
  src/op/cpu/kv_cache.cpp
  src/op/cpu/matmul.cpp
  src/op/cpu/rms_norm.cpp
  src/op/cpu/rope.cpp
  src/op/cpu/silu.cpp
  src/model/weights.cpp
  src/model/llama.cpp
  src/sampler/sampler.cpp
)
target_include_directories(kllm_core PUBLIC include)
target_compile_definitions(kllm_core PUBLIC KLLM_USE_CUDA=$<BOOL:${KLLM_USE_CUDA}>)
if(MSVC)
  target_compile_options(kllm_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(kllm_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()
if(KLLM_USE_CUDA)
  target_sources(kllm_core PRIVATE
    src/op/cuda/add.cu
    src/op/cuda/attention.cu
    src/op/cuda/embedding.cu
    src/op/cuda/kv_cache.cu
    src/op/cuda/matmul.cu
    src/op/cuda/rms_norm.cu
    src/op/cuda/rope.cu
    src/op/cuda/silu.cu
  )
  target_include_directories(kllm_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  set_target_properties(kllm_core PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )
  target_link_libraries(kllm_core PUBLIC CUDA::cudart CUDA::cublas)
endif()
add_executable(kllm_demo demo/kllm_demo.cpp)
target_link_libraries(kllm_demo PRIVATE kllm_core)

# Legacy combined test
add_executable(kllm_tests tests/test_ops.cpp)
target_link_libraries(kllm_tests PRIVATE kllm_core)

# Individual operator tests
add_executable(test_half tests/test_half.cpp)
target_link_libraries(test_half PRIVATE kllm_core)

add_executable(test_embedding tests/test_embedding.cpp)
target_link_libraries(test_embedding PRIVATE kllm_core)

add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE kllm_core)

add_executable(test_add_bias tests/test_add_bias.cpp)
target_link_libraries(test_add_bias PRIVATE kllm_core)

add_executable(test_select_last_row tests/test_select_last_row.cpp)
target_link_libraries(test_select_last_row PRIVATE kllm_core)

add_executable(test_rms_norm tests/test_rms_norm.cpp)
target_link_libraries(test_rms_norm PRIVATE kllm_core)

add_executable(test_silu tests/test_silu.cpp)
target_link_libraries(test_silu PRIVATE kllm_core)

add_executable(test_rope tests/test_rope.cpp)
target_link_libraries(test_rope PRIVATE kllm_core)

add_executable(test_kv_cache tests/test_kv_cache.cpp)
target_link_libraries(test_kv_cache PRIVATE kllm_core)

add_executable(test_attention tests/test_attention.cpp)
target_link_libraries(test_attention PRIVATE kllm_core)

add_executable(test_sampler tests/test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE kllm_core)

enable_testing()
add_test(NAME kllm_tests COMMAND kllm_tests)
add_test(NAME test_half COMMAND test_half)
add_test(NAME test_embedding COMMAND test_embedding)
add_test(NAME test_matmul COMMAND test_matmul)
add_test(NAME test_add_bias COMMAND test_add_bias)
add_test(NAME test_select_last_row COMMAND test_select_last_row)
add_test(NAME test_rms_norm COMMAND test_rms_norm)
add_test(NAME test_silu COMMAND test_silu)
add_test(NAME test_rope COMMAND test_rope)
add_test(NAME test_kv_cache COMMAND test_kv_cache)
add_test(NAME test_attention COMMAND test_attention)
add_test(NAME test_sampler COMMAND test_sampler)