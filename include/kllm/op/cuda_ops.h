#pragma once

#include <vector>

#include "kllm/base/tensor.h"

namespace kllm::op {

void cuda_embedding(const Tensor& table, const std::vector<int>& token_ids, Tensor& out);
void cuda_rms_norm(const Tensor& x, const Tensor& weight, Tensor& out, float eps);
void cuda_matmul(const Tensor& a, const Tensor& b, Tensor& out);
void cuda_add_bias_inplace(Tensor& x, const Tensor* bias);
void cuda_add_inplace(Tensor& x, const Tensor& residual);
void cuda_select_last_row(const Tensor& x, Tensor& out);
void cuda_silu_mul(const Tensor& gate, const Tensor& up, Tensor& out);
void cuda_rope_inplace(Tensor& q_or_k, int past_len, float theta);
void cuda_copy_kv_to_cache(const Tensor& k, const Tensor& v, Tensor& k_cache, Tensor& v_cache, int past_len);
void cuda_attention(
    const Tensor& q,
    const Tensor& k_cache,
    const Tensor& v_cache,
    Tensor& out,
    int past_len,
    int total_len,
    int num_heads,
    int num_kv_heads,
    int head_dim);

}  // namespace kllm::op

