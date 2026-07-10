#include <cmath>
#include <iostream>
#include "kllm/op/ops.h"
#include "test_utils.h"

using namespace test_utils;

void test_silu(kllm::DeviceType device) {
  auto gate = make_tensor({1, 2}, {0, 1}, device);
  auto up = make_tensor({1, 2}, {2, 2}, device);
  kllm::Tensor out({1, 2}, kllm::DType::F32, device);
  kllm::op::silu_mul(gate, up, out);
  const float silu1 = 1.0f / (1.0f + std::exp(-1.0f));
  expect_vec_close(out.to_float_vector(), {0.0f, 2.0f * silu1});
}

int main() {
  try {
    test_silu(kllm::DeviceType::CPU);
    std::cout << "[OK] CPU silu test passed\n";

    if (kllm::cuda_available()) {
      test_silu(kllm::DeviceType::CUDA);
      std::cout << "[OK] CUDA silu test passed\n";
    } else {
      std::cout << "[SKIP] CUDA is not available\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
