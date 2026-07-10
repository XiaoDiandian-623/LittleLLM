#pragma once

#include <cstdint>
#include <vector>

#include "kllm/base/tensor.h"

namespace kllm::op {

float read_scalar_as_float(const Tensor& tensor, std::size_t index);
void write_scalar_from_float(Tensor& tensor, std::size_t index, float value);

void embedding(const Tensor& table, const std::vector<int>& token_ids, Tensor& out);
void rms_norm(const Tensor& x, const Tensor& weight, Tensor& out, float eps);
void matmul(const Tensor& a, const Tensor& b, Tensor& out);
void add_bias_inplace(Tensor& x, const Tensor* bias);
void add_inplace(Tensor& x, const Tensor& residual);
void select_last_row(const Tensor& x, Tensor& out);
void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out);
void rope_inplace(Tensor& q_or_k, int past_len, float theta);
void copy_kv_to_cache(const Tensor& k, const Tensor& v, Tensor& k_cache, Tensor& v_cache, int past_len);
void attention(
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
