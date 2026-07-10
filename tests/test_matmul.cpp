#include <iostream>
#include "kllm/op/ops.h"
#include "test_utils.h"

using namespace test_utils;

void test_matmul(kllm::DeviceType device) {
  auto a = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6}, device);
  auto b = make_tensor({2, 3}, {1, 0, 1, 0, 1, 1}, device);
  kllm::Tensor out({2, 2}, kllm::DType::F32, device);
  kllm::op::matmul(a, b, out);
  expect_vec_close(out.to_float_vector(), {4, 5, 10, 11});
}

int main() {
  try {
    test_matmul(kllm::DeviceType::CPU);
    std::cout << "[OK] CPU matmul test passed\n";

    if (kllm::cuda_available()) {
      test_matmul(kllm::DeviceType::CUDA);
      std::cout << "[OK] CUDA matmul test passed\n";
    } else {
      std::cout << "[SKIP] CUDA is not available\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
