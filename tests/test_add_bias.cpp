#include <iostream>
#include "kllm/op/ops.h"
#include "test_utils.h"

using namespace test_utils;

void test_add_bias(kllm::DeviceType device) {
  auto data = make_tensor({2, 2}, {4, 5, 10, 11}, device);
  auto bias = make_tensor({2}, {1, -1}, device);
  kllm::op::add_bias_inplace(data, &bias);
  expect_vec_close(data.to_float_vector(), {5, 4, 11, 10});
}

int main() {
  try {
    test_add_bias(kllm::DeviceType::CPU);
    std::cout << "[OK] CPU add_bias test passed\n";

    if (kllm::cuda_available()) {
      test_add_bias(kllm::DeviceType::CUDA);
      std::cout << "[OK] CUDA add_bias test passed\n";
    } else {
      std::cout << "[SKIP] CUDA is not available\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
