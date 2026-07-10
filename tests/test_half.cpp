#include <iostream>
#include "kllm/base/half.h"
#include "test_utils.h"

using namespace test_utils;

void test_half_roundtrip() {
  const float value = 1.5f;
  const auto half = kllm::float_to_half(value);
  expect_close(kllm::half_to_float(half), value, 1e-3f);
}

int main() {
  try {
    test_half_roundtrip();
    std::cout << "[OK] Half type test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
