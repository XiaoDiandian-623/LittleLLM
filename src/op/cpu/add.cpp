#include "kllm/op/cpu_ops.h"

#include <stdexcept>

#include "kllm/op/detail/cpu_helpers.h"

namespace kllm::op {

void cpu_add_bias_inplace(Tensor& x, const Tensor* bias) {
  if (bias == nullptr) {
    return;
  }
  detail::require_cpu(x);
  detail::require_cpu(*bias);

  const int rows = static_cast<int>(x.shape().at(0));
  const int cols = static_cast<int>(x.shape().at(1));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const std::size_t idx = static_cast<std::size_t>(r) * cols + c;
      detail::write_cpu(x, idx, detail::read_cpu(x, idx) + detail::read_cpu(*bias, c));
    }
  }
}

void cpu_add_inplace(Tensor& x, const Tensor& residual) {
  detail::require_cpu(x);
  detail::require_cpu(residual);
  if (x.numel() != residual.numel()) {
    throw std::runtime_error("add shape mismatch");
  }
  for (std::size_t i = 0; i < x.numel(); ++i) {
    detail::write_cpu(x, i, detail::read_cpu(x, i) + detail::read_cpu(residual, i));
  }
}

void cpu_select_last_row(const Tensor& x, Tensor& out) {
  detail::require_cpu(x);
  detail::require_cpu(out);
  const int rows = static_cast<int>(x.shape().at(0));
  const int cols = static_cast<int>(x.shape().at(1));
  if (out.shape().size() != 2 || out.shape().at(0) != 1 || out.shape().at(1) != cols) {
    throw std::runtime_error("select_last_row output shape mismatch");
  }
  const std::size_t src_base = static_cast<std::size_t>(rows - 1) * cols;
  for (int c = 0; c < cols; ++c) {
    detail::write_cpu(out, c, detail::read_cpu(x, src_base + c));
  }
}

}  // namespace kllm::op

