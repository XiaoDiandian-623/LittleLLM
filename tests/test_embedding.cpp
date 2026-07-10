#include <iostream>
#include "kllm/op/ops.h"
#include "test_utils.h"

using namespace test_utils;

void test_embedding(kllm::DeviceType device) {
  auto table = make_tensor({3, 2}, {1, 2, 3, 4, 5, 6}, device);
  kllm::Tensor out({2, 2}, kllm::DType::F32, device);
  kllm::op::embedding(table, {2, 0}, out);
  expect_vec_close(out.to_float_vector(), {5, 6, 1, 2});
}

int main() {
  try {
    test_embedding(kllm::DeviceType::CPU);
    std::cout << "[OK] CPU embedding test passed\n";

    if (kllm::cuda_available()) {
      test_embedding(kllm::DeviceType::CUDA);
      std::cout << "[OK] CUDA embedding test passed\n";
    } else {
      std::cout << "[SKIP] CUDA is not available\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
