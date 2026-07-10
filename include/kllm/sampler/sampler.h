#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "kllm/base/tensor.h"

namespace kllm {

struct SamplingConfig {
  int top_k = 40;
  float top_p = 0.9f;
  float temperature = 0.8f;
  float repetition_penalty = 1.1f;
  bool greedy = false;
  std::uint32_t seed = 1234;
};

class Sampler {
 public:
  explicit Sampler(SamplingConfig config);
  int sample(const Tensor& logits, const std::vector<int>& history);

 private:
  SamplingConfig config_;
  std::mt19937 rng_;
};

}  // namespace kllm

