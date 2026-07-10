#include "kllm/op/cpu_ops.h"

#include <stdexcept>

#include "kllm/op/detail/cpu_helpers.h"

namespace kllm::op {

void cpu_matmul(const Tensor& a, const Tensor& b, Tensor& out) {
  detail::require_cpu(a);
  detail::require_cpu(b);
  detail::require_cpu(out);

  const int m = static_cast<int>(a.shape().at(0));
  const int k = static_cast<int>(a.shape().at(1));
  const int n = static_cast<int>(b.shape().at(0));
  if (b.shape().at(1) != k || out.shape().at(0) != m || out.shape().at(1) != n) {
    throw std::runtime_error("matmul shape mismatch");
  }

  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int i = 0; i < k; ++i) {
        sum += detail::read_cpu(a, static_cast<std::size_t>(row) * k + i) *
               detail::read_cpu(b, static_cast<std::size_t>(col) * k + i);
      }
      detail::write_cpu(out, static_cast<std::size_t>(row) * n + col, sum);
    }
  }
}

}  // namespace kllm::op

