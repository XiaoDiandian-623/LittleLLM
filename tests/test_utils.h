#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "kllm/base/device.h"
#include "kllm/base/tensor.h"

namespace test_utils {

constexpr float kTol = 1e-4f;

inline void fail(const std::string& message) {
  throw std::runtime_error(message);
}

inline void expect_close(float actual, float expected, float tol = kTol) {
  if (std::fabs(actual - expected) > tol) {
    fail("expected " + std::to_string(expected) + ", got " + std::to_string(actual));
  }
}

inline void expect_vec_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tol = kTol) {
  if (actual.size() != expected.size()) {
    fail("vector size mismatch: expected " + std::to_string(expected.size()) +
         ", got " + std::to_string(actual.size()));
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > tol) {
      fail("mismatch at index " + std::to_string(i) +
           ": expected " + std::to_string(expected[i]) +
           ", got " + std::to_string(actual[i]));
    }
  }
}

inline kllm::Tensor make_tensor(
    const std::vector<std::int64_t>& shape,
    const std::vector<float>& values,
    kllm::DeviceType device) {
  kllm::Tensor tensor(shape, kllm::DType::F32, device);
  tensor.copy_from_host(values.data(), values.size() * sizeof(float));
  return tensor;
}

}  // namespace test_utils
