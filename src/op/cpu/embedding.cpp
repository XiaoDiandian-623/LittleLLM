#include "kllm/op/cpu_ops.h"

#include <stdexcept>

#include "kllm/op/detail/cpu_helpers.h"

namespace kllm::op {

void cpu_embedding(const Tensor& table, const std::vector<int>& token_ids, Tensor& out) {
  detail::require_cpu(table);
  detail::require_cpu(out);

  const int vocab = static_cast<int>(table.shape().at(0));
  const int hidden = static_cast<int>(table.shape().at(1));
  if (out.shape().size() != 2 || out.shape()[0] != static_cast<std::int64_t>(token_ids.size()) ||
      out.shape()[1] != hidden) {
    throw std::runtime_error("embedding output shape mismatch");
  }

  for (std::size_t t = 0; t < token_ids.size(); ++t) {
    const int token = token_ids[t];
    if (token < 0 || token >= vocab) {
      throw std::runtime_error("token id out of vocabulary range");
    }
    for (int h = 0; h < hidden; ++h) {
      detail::write_cpu(out, t * hidden + h, detail::read_cpu(table, static_cast<std::size_t>(token) * hidden + h));
    }
  }
}

}  // namespace kllm::op

