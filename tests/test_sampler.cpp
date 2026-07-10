#include <iostream>
#include "kllm/sampler/sampler.h"
#include "test_utils.h"

using namespace test_utils;

void test_sampler() {
  auto logits = make_tensor({1, 4}, {0, 1, 5, 2}, kllm::DeviceType::CPU);
  kllm::SamplingConfig cfg;
  cfg.greedy = true;
  kllm::Sampler sampler(cfg);
  const int token = sampler.sample(logits, {});
  if (token != 2) {
    fail("greedy sampler returned wrong token: expected 2, got " + std::to_string(token));
  }
}

int main() {
  try {
    test_sampler();
    std::cout << "[OK] Sampler test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
