#include <iostream>
#include "kllm/op/ops.h"
#include "test_utils.h"

using namespace test_utils;

void test_kv_cache(kllm::DeviceType device) {
  auto k = make_tensor({1, 1, 2}, {1, 0}, device);
  auto v = make_tensor({1, 1, 2}, {2, 3}, device);
  kllm::Tensor k_cache({2, 1, 2}, kllm::DType::F32, device);
  kllm::Tensor v_cache({2, 1, 2}, kllm::DType::F32, device);
  kllm::op::copy_kv_to_cache(k, v, k_cache, v_cache, 0);
  expect_vec_close(k_cache.to_float_vector(), {1, 0, 0, 0});
  expect_vec_close(v_cache.to_float_vector(), {2, 3, 0, 0});
}

int main() {
  try {
    test_kv_cache(kllm::DeviceType::CPU);
    std::cout << "[OK] CPU kv_cache test passed\n";

    if (kllm::cuda_available()) {
      test_kv_cache(kllm::DeviceType::CUDA);
      std::cout << "[OK] CUDA kv_cache test passed\n";
    } else {
      std::cout << "[SKIP] CUDA is not available\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
