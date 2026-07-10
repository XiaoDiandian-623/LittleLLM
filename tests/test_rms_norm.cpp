#include <cmath>
#include <iostream>
#include "kllm/op/ops.h"
#include "test_utils.h"

using namespace test_utils;

void test_rms_norm(kllm::DeviceType device) {
  auto x = make_tensor({1, 2}, {3, 4}, device);
  auto w = make_tensor({2}, {1, 1}, device);
  kllm::Tensor out({1, 2}, kllm::DType::F32, device);
  kllm::op::rms_norm(x, w, out, 0.0f);
  const float scale = 1.0f / std::sqrt((9.0f + 16.0f) / 2.0f);
  expect_vec_close(out.to_float_vector(), {3.0f * scale, 4.0f * scale});
}

int main() {
  try {
    test_rms_norm(kllm::DeviceType::CPU);
    std::cout << "[OK] CPU rms_norm test passed\n";

    if (kllm::cuda_available()) {
      test_rms_norm(kllm::DeviceType::CUDA);
      std::cout << "[OK] CUDA rms_norm test passed\n";
    } else {
      std::cout << "[SKIP] CUDA is not available\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
