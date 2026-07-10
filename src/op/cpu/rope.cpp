#include "kllm/op/cpu_ops.h"

#include <cmath>
#include <stdexcept>

#include "kllm/op/detail/cpu_helpers.h"

namespace kllm::op {

void cpu_rope_inplace(Tensor& q_or_k, int past_len, float theta) {
  detail::require_cpu(q_or_k);

  const int seq = static_cast<int>(q_or_k.shape().at(0));
  const int heads = static_cast<int>(q_or_k.shape().at(1));
  const int head_dim = static_cast<int>(q_or_k.shape().at(2));
  if (head_dim % 2 != 0) {
    throw std::runtime_error("rope requires an even head_dim");
  }

  for (int t = 0; t < seq; ++t) {
    const int position = past_len + t;
    for (int h = 0; h < heads; ++h) {
      for (int d = 0; d < head_dim / 2; ++d) {
        const float inv_freq = std::pow(theta, -2.0f * static_cast<float>(d) / head_dim);
        const float angle = static_cast<float>(position) * inv_freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const std::size_t base = (static_cast<std::size_t>(t) * heads + h) * head_dim;
        const float x1 = detail::read_cpu(q_or_k, base + d);
        const float x2 = detail::read_cpu(q_or_k, base + d + head_dim / 2);
        detail::write_cpu(q_or_k, base + d, x1 * c - x2 * s);
        detail::write_cpu(q_or_k, base + d + head_dim / 2, x2 * c + x1 * s);
      }
    }
  }
}

}  // namespace kllm::op

