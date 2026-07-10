#include "kllm/op/cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "kllm/op/detail/cpu_helpers.h"

namespace kllm::op {

void cpu_attention(
    const Tensor& q,
    const Tensor& k_cache,
    const Tensor& v_cache,
    Tensor& out,
    int past_len,
    int total_len,
    int num_heads,
    int num_kv_heads,
    int head_dim) {
  detail::require_cpu(q);
  detail::require_cpu(k_cache);
  detail::require_cpu(v_cache);
  detail::require_cpu(out);

  const int seq = static_cast<int>(q.shape().at(0));
  const int groups = num_heads / num_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> scores(total_len);

  for (int t = 0; t < seq; ++t) {
    const int key_limit = past_len + t + 1;
    for (int h = 0; h < num_heads; ++h) {
      const int kv_h = h / groups;
      float max_score = -std::numeric_limits<float>::infinity();
      for (int pos = 0; pos < key_limit; ++pos) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          const std::size_t q_idx = (static_cast<std::size_t>(t) * num_heads + h) * head_dim + d;
          const std::size_t k_idx = (static_cast<std::size_t>(pos) * num_kv_heads + kv_h) * head_dim + d;
          dot += detail::read_cpu(q, q_idx) * detail::read_cpu(k_cache, k_idx);
        }
        scores[pos] = dot * scale;
        max_score = std::max(max_score, scores[pos]);
      }

      float denom = 0.0f;
      for (int pos = 0; pos < key_limit; ++pos) {
        scores[pos] = std::exp(scores[pos] - max_score);
        denom += scores[pos];
      }

      for (int d = 0; d < head_dim; ++d) {
        float value = 0.0f;
        for (int pos = 0; pos < key_limit; ++pos) {
          const float prob = scores[pos] / denom;
          const std::size_t v_idx = (static_cast<std::size_t>(pos) * num_kv_heads + kv_h) * head_dim + d;
          value += prob * detail::read_cpu(v_cache, v_idx);
        }
        const std::size_t out_idx = (static_cast<std::size_t>(t) * num_heads + h) * head_dim + d;
        detail::write_cpu(out, out_idx, value);
      }
    }
  }
}

}  // namespace kllm::op

