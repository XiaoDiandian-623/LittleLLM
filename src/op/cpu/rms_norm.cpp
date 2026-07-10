#include "kllm/op/cpu_ops.h"

#include <cmath>

#include "kllm/op/detail/cpu_helpers.h"

namespace kllm::op {

void cpu_rms_norm(const Tensor& x, const Tensor& weight, Tensor& out, float eps) {
  detail::require_cpu(x);
  detail::require_cpu(weight);
  detail::require_cpu(out);

  const int rows = static_cast<int>(x.shape().at(0));
  const int hidden = static_cast<int>(x.shape().at(1));
  for (int r = 0; r < rows; ++r) {
    float sum_sq = 0.0f;
    for (int c = 0; c < hidden; ++c) {
      const float v = detail::read_cpu(x, static_cast<std::size_t>(r) * hidden + c);
      sum_sq += v * v;
    }
    const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden) + eps);
    for (int c = 0; c < hidden; ++c) {
      const float v = detail::read_cpu(x, static_cast<std::size_t>(r) * hidden + c);
      const float w = detail::read_cpu(weight, c);
      detail::write_cpu(out, static_cast<std::size_t>(r) * hidden + c, v * scale * w);
    }
  }
}

}  // namespace kllm::op

