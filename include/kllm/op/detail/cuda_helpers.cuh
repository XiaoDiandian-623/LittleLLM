#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "kllm/base/tensor.h"
namespace kllm::op::detail {
inline void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}
inline void check_cublas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS call failed with status " + std::to_string(static_cast<int>(status)));
  }
}
inline std::uint32_t dtype_id(DType dtype) {
  return static_cast<std::uint32_t>(dtype);
}
inline cudaDataType_t cuda_dtype(DType dtype) {
  if (dtype == DType::F32) {
    return CUDA_R_32F;
  }
  if (dtype == DType::F16) {
    return CUDA_R_16F;
  }
  throw std::runtime_error("cuBLAS matmul only supports F32/F16 tensors");
}
__device__ inline float cuda_read(const void* data, std::uint32_t dtype, std::size_t idx) {
  if (dtype == 0) {
    return reinterpret_cast<const float*>(data)[idx];
  }
  if (dtype == 1) {
    return __half2float(reinterpret_cast<const __half*>(data)[idx]);
  }
  return static_cast<float>(reinterpret_cast<const int*>(data)[idx]);
}
__device__ inline void cuda_write(void* data, std::uint32_t dtype, std::size_t idx, float value) {
  if (dtype == 0) {
    reinterpret_cast<float*>(data)[idx] = value;
    return;
  }
  if (dtype == 1) {
    reinterpret_cast<__half*>(data)[idx] = __float2half(value);
    return;
  }
  reinterpret_cast<int*>(data)[idx] = static_cast<int>(value);
}
}  // namespace kllm::op::detail