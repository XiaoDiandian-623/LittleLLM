#include "kllm/op/cpu_ops.h"

#include <stdexcept>

#include "kllm/op/detail/cpu_helpers.h"

namespace kllm::op {

void cpu_silu_mul(const Tensor& gate, const Tensor& up, Tensor& out) {
  detail::require_cpu(gate);
  detail::require_cpu(up);
  detail::require_cpu(out);
  if (gate.numel() != up.numel() || gate.numel() != out.numel()) {
    throw std::runtime_error("silu_mul shape mismatch");
  }
  for (std::size_t i = 0; i < gate.numel(); ++i) {
    const float g = detail::read_cpu(gate, i);
    detail::write_cpu(out, i, g * detail::sigmoid(g) * detail::read_cpu(up, i));
  }
}

}  // namespace kllm::op

