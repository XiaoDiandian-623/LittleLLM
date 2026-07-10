#include <iostream>
#include "kllm/op/ops.h"
#include "test_utils.h"

using namespace test_utils;

void test_rope_position_zero(kllm::DeviceType device) {
  auto x = make_tensor({1, 1, 4}, {1, 2, 3, 4}, device);
  kllm::op::rope_inplace(x, 0, 10000.0f);
  expect_vec_close(x.to_float_vector(), {1, 2, 3, 4});
}

int main() {
  try {
    test_rope_position_zero(kllm::DeviceType::CPU);
    std::cout << "[OK] CPU rope test passed\n";

    if (kllm::cuda_available()) {
      test_rope_position_zero(kllm::DeviceType::CUDA);
      std::cout << "[OK] CUDA rope test passed\n";
    } else {
      std::cout << "[SKIP] CUDA is not available\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
