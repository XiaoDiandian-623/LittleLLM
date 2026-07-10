#include "kllm/op/ops.h"

#include <stdexcept>
#include <vector>

#include "kllm/op/cpu_ops.h"
#include "kllm/op/detail/cpu_helpers.h"

#if KLLM_USE_CUDA
#include "kllm/op/cuda_ops.h"
#endif

namespace kllm::op {

namespace {

[[noreturn]] void cuda_not_compiled() {
  throw std::runtime_error("CUDA backend was not compiled. Rebuild with -DKLLM_USE_CUDA=ON.");
}

}  // namespace

float read_scalar_as_float(const Tensor& tensor, std::size_t index) {
  if (index >= tensor.numel()) {
    throw std::runtime_error("scalar read index out of range");
  }
  if (tensor.device() == DeviceType::CPU) {
    return detail::read_cpu(tensor, index);
  }

  std::vector<float> host = tensor.to_float_vector();
  return host[index];
}

void write_scalar_from_float(Tensor& tensor, std::size_t index, float value) {
  if (index >= tensor.numel()) {
    throw std::runtime_error("scalar write index out of range");
  }
  detail::require_cpu(tensor);
  detail::write_cpu(tensor, index, value);
}

void embedding(const Tensor& table, const std::vector<int>& token_ids, Tensor& out) {
  if (table.device() == DeviceType::CUDA) {
#if KLLM_USE_CUDA
    cuda_embedding(table, token_ids, out);
    return;
#else
    cuda_not_compiled();
#endif
  }
  cpu_embedding(table, token_ids, out);
}

void rms_norm(const Tensor& x, const Tensor& weight, Tensor& out, float eps) {
  if (x.device() == DeviceType::CUDA) {
#if KLLM_USE_CUDA
    cuda_rms_norm(x, weight, out, eps);
    return;
#else
    cuda_not_compiled();
#endif
  }
  cpu_rms_norm(x, weight, out, eps);
}

void matmul(const Tensor& a, const Tensor& b, Tensor& out) {
  if (a.device() == DeviceType::CUDA) {
#if KLLM_USE_CUDA
    cuda_matmul(a, b, out);
    return;
#else
    cuda_not_compiled();
#endif
  }
  cpu_matmul(a, b, out);
}

void add_bias_inplace(Tensor& x, const Tensor* bias) {
  if (x.device() == DeviceType::CUDA) {
#if KLLM_USE_CUDA
    cuda_add_bias_inplace(x, bias);
    return;
#else
    cuda_not_compiled();
#endif
  }
  cpu_add_bias_inplace(x, bias);
}

void add_inplace(Tensor& x, const Tensor& residual) {
  if (x.device() == DeviceType::CUDA) {
#if KLLM_USE_CUDA
    cuda_add_inplace(x, residual);
    return;
#else
    cuda_not_compiled();
#endif
  }
  cpu_add_inplace(x, residual);
}

void select_last_row(const Tensor& x, Tensor& out) {
  if (x.device() == DeviceType::CUDA) {
#if KLLM_USE_CUDA
    cuda_select_last_row(x, out);
    return;
#else
    cuda_not_compiled();
#endif
  }
  cpu_select_last_row(x, out);
}

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out) {
  if (gate.device() == DeviceType::CUDA) {
#if KLLM_USE_CUDA
    cuda_silu_mul(gate, up, out);
    return;
#else
    cuda_not_compiled();
#endif
  }
  cpu_silu_mul(gate, up, out);
}

void rope_inplace(Tensor& q_or_k, int past_len, float theta) {
  if (q_or_k.device() == DeviceType::CUDA) {
#if KLLM_USE_CUDA
    cuda_rope_inplace(q_or_k, past_len, theta);
    return;
#else
    cuda_not_compiled();
#endif
  }
  cpu_rope_inplace(q_or_k, past_len, theta);
}

void copy_kv_to_cache(const Tensor& k, const Tensor& v, Tensor& k_cache, Tensor& v_cache, int past_len) {
  if (k.device() == DeviceType::CUDA) {
#if KLLM_USE_CUDA
    cuda_copy_kv_to_cache(k, v, k_cache, v_cache, past_len);
    return;
#else
    cuda_not_compiled();
#endif
  }
  cpu_copy_kv_to_cache(k, v, k_cache, v_cache, past_len);
}

void attention(
    const Tensor& q,
    const Tensor& k_cache,
    const Tensor& v_cache,
    Tensor& out,
    int past_len,
    int total_len,
    int num_heads,
    int num_kv_heads,
    int head_dim) {
  if (q.device() == DeviceType::CUDA) {
#if KLLM_USE_CUDA
    cuda_attention(q, k_cache, v_cache, out, past_len, total_len, num_heads, num_kv_heads, head_dim);
    return;
#else
    cuda_not_compiled();
#endif
  }
  cpu_attention(q, k_cache, v_cache, out, past_len, total_len, num_heads, num_kv_heads, head_dim);
}

}  // namespace kllm::op

