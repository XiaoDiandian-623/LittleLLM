#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>

#include "kllm/base/half.h"
#include "kllm/base/tensor.h"

namespace kllm::op::detail {

inline void require_cpu(const Tensor& tensor) {
  if (tensor.device() != DeviceType::CPU) {
    throw std::runtime_error("expected a CPU tensor");
  }
}

inline float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

inline float read_cpu(const Tensor& tensor, std::size_t index) {
  if (tensor.dtype() == DType::F32) {
    return tensor.data_as<float>()[index];
  }
  if (tensor.dtype() == DType::F16) {
    return half_to_float(tensor.data_as<float16_t>()[index]);
  }
  if (tensor.dtype() == DType::I32) {
    return static_cast<float>(tensor.data_as<std::int32_t>()[index]);
  }
  throw std::runtime_error("unsupported dtype");
}

inline void write_cpu(Tensor& tensor, std::size_t index, float value) {
  if (tensor.dtype() == DType::F32) {
    tensor.data_as<float>()[index] = value;
    return;
  }
  if (tensor.dtype() == DType::F16) {
    tensor.data_as<float16_t>()[index] = float_to_half(value);
    return;
  }
  if (tensor.dtype() == DType::I32) {
    tensor.data_as<std::int32_t>()[index] = static_cast<std::int32_t>(value);
    return;
  }
  throw std::runtime_error("unsupported dtype");
}

}  // namespace kllm::op::detail

