#pragma once

#include <vector>

#include "kllm/base/tensor.h"

namespace kllm::op {

void cpu_embedding(const Tensor& table, const std::vector<int>& token_ids, Tensor& out);
void cpu_rms_norm(const Tensor& x, const Tensor& weight, Tensor& out, float eps);
void cpu_matmul(const Tensor& a, const Tensor& b, Tensor& out);
void cpu_add_bias_inplace(Tensor& x, const Tensor* bias);
void cpu_add_inplace(Tensor& x, const Tensor& residual);
void cpu_select_last_row(const Tensor& x, Tensor& out);
void cpu_silu_mul(const Tensor& gate, const Tensor& up, Tensor& out);
void cpu_rope_inplace(Tensor& q_or_k, int past_len, float theta);
void cpu_copy_kv_to_cache(const Tensor& k, const Tensor& v, Tensor& k_cache, Tensor& v_cache, int past_len);
void cpu_attention(
    const Tensor& q,
    const Tensor& k_cache,
    const Tensor& v_cache,
    Tensor& out,
    int past_len,
    int total_len,
    int num_heads,
    int num_kv_heads,
    int head_dim);

namespace cpu {
void quantize_int8(const Tensor& input, Tensor& out_data, Tensor& out_scale);
void dequantize_int8(const Tensor& data, const Tensor& scale, Tensor& output);
void matmul_int8(const Tensor& a, const Tensor& b_quant, const Tensor& b_scale, Tensor& out);
}  // namespace cpu

}  // namespace kllm::op

