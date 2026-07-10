#include "kllm/sampler/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace kllm {

Sampler::Sampler(SamplingConfig config) : config_(config), rng_(config.seed) {}

int Sampler::sample(const Tensor& logits_tensor, const std::vector<int>& history) {
  auto logits = logits_tensor.to_float_vector();
  if (logits.empty()) {
    throw std::runtime_error("empty logits tensor");
  }

  const int vocab = static_cast<int>(logits_tensor.shape().back());
  if (static_cast<int>(logits.size()) != vocab) {
    const int rows = static_cast<int>(logits.size()) / vocab;
    std::vector<float> last(logits.begin() + (rows - 1) * vocab, logits.end());
    logits = std::move(last);
  }

  if (config_.repetition_penalty != 1.0f) {
    std::unordered_set<int> seen(history.begin(), history.end());
    for (const int token : seen) {
      if (token < 0 || token >= vocab) {
        continue;
      }
      if (logits[token] < 0.0f) {
        logits[token] *= config_.repetition_penalty;
      } else {
        logits[token] /= config_.repetition_penalty;
      }
    }
  }

  if (config_.greedy || config_.temperature <= 0.0f) {
    return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
  }

  const float temperature = std::max(config_.temperature, 1e-5f);
  for (auto& value : logits) {
    value /= temperature;
  }

  std::vector<int> indices(vocab);
  std::iota(indices.begin(), indices.end(), 0);
  const int k = config_.top_k > 0 ? std::min(config_.top_k, vocab) : vocab;
  if (k < vocab) {
    std::nth_element(indices.begin(), indices.begin() + k, indices.end(), [&](int a, int b) {
      return logits[a] > logits[b];
    });
  }
  indices.resize(k);
  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    return logits[a] > logits[b];
  });

  float max_logit = -std::numeric_limits<float>::infinity();
  for (const int idx : indices) {
    max_logit = std::max(max_logit, logits[idx]);
  }

  std::vector<float> probs(indices.size());
  float denom = 0.0f;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    probs[i] = std::exp(logits[indices[i]] - max_logit);
    denom += probs[i];
  }
  for (auto& p : probs) {
    p /= denom;
  }

  if (config_.top_p > 0.0f && config_.top_p < 1.0f) {
    float cumulative = 0.0f;
    std::size_t keep = probs.size();
    for (std::size_t i = 0; i < probs.size(); ++i) {
      cumulative += probs[i];
      if (cumulative >= config_.top_p) {
        keep = i + 1;
        break;
      }
    }
    indices.resize(keep);
    probs.resize(keep);
    const float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    for (auto& p : probs) {
      p /= sum;
    }
  }

  std::discrete_distribution<int> distribution(probs.begin(), probs.end());
  return indices[distribution(rng_)];
}

}  // namespace kllm