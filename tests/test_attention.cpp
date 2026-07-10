#include <iostream>
#include "kllm/op/ops.h"
#include "test_utils.h"

using namespace test_utils;

void test_attention(kllm::DeviceType device) {
  auto k = make_tensor({1, 1, 2}, {1, 0}, device);
  auto v = make_tensor({1, 1, 2}, {2, 3}, device);
  kllm::Tensor k_cache({2, 1, 2}, kllm::DType::F32, device);
  kllm::Tensor v_cache({2, 1, 2}, kllm::DType::F32, device);
  kllm::op::copy_kv_to_cache(k, v, k_cache, v_cache, 0);

  auto q = make_tensor({1, 1, 2}, {1, 0}, device);
  kllm::Tensor out({1, 1, 2}, kllm::DType::F32, device);
  kllm::op::attention(q, k_cache, v_cache, out, 0, 1, 1, 1, 2);
  expect_vec_close(out.to_float_vector(), {2, 3});
}

int main() {
  try {
    test_attention(kllm::DeviceType::CPU);
    std::cout << "[OK] CPU attention test passed\n";

    if (kllm::cuda_available()) {
      test_attention(kllm::DeviceType::CUDA);
      std::cout << "[OK] CUDA attention test passed\n";
    } else {
      std::cout << "[SKIP] CUDA is not available\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
