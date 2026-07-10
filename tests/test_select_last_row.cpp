#include <iostream>
#include "kllm/op/ops.h"
#include "test_utils.h"

using namespace test_utils;

void test_select_last_row(kllm::DeviceType device) {
  auto data = make_tensor({2, 2}, {5, 4, 11, 10}, device);
  kllm::Tensor last({1, 2}, kllm::DType::F32, device);
  kllm::op::select_last_row(data, last);
  expect_vec_close(last.to_float_vector(), {11, 10});
}

int main() {
  try {
    test_select_last_row(kllm::DeviceType::CPU);
    std::cout << "[OK] CPU select_last_row test passed\n";

    if (kllm::cuda_available()) {
      test_select_last_row(kllm::DeviceType::CUDA);
      std::cout << "[OK] CUDA select_last_row test passed\n";
    } else {
      std::cout << "[SKIP] CUDA is not available\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
