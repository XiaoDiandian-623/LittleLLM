#include "kllm/op/cpu_ops.h"

#include <stdexcept>

#include "kllm/op/detail/cpu_helpers.h"

namespace kllm::op {

void cpu_copy_kv_to_cache(const Tensor& k, const Tensor& v, Tensor& k_cache, Tensor& v_cache, int past_len) {
  detail::require_cpu(k);
  detail::require_cpu(v);
  detail::require_cpu(k_cache);
  detail::require_cpu(v_cache);

  const int seq = static_cast<int>(k.shape().at(0));
  const int kv_heads = static_cast<int>(k.shape().at(1));
  const int head_dim = static_cast<int>(k.shape().at(2));
  const int max_seq = static_cast<int>(k_cache.shape().at(0));
  if (past_len + seq > max_seq) {
    throw std::runtime_error("KV cache capacity exceeded");
  }

  const std::size_t row_size = static_cast<std::size_t>(kv_heads) * head_dim;
  for (int t = 0; t < seq; ++t) {
    for (std::size_t i = 0; i < row_size; ++i) {
      const std::size_t src = static_cast<std::size_t>(t) * row_size + i;
      const std::size_t dst = static_cast<std::size_t>(past_len + t) * row_size + i;
      detail::write_cpu(k_cache, dst, detail::read_cpu(k, src));
      detail::write_cpu(v_cache, dst, detail::read_cpu(v, src));
    }
  }
}

}  // namespace kllm::op

