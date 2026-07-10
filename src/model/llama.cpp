#include "kllm/model/llama.h"

#include <sstream>
#include <stdexcept>

#include "kllm/op/ops.h"

namespace kllm {

namespace {

Tensor linear(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    DeviceType device,
    DType output_dtype) {
  const int rows = static_cast<int>(x.shape().at(0));
  const int out_dim = static_cast<int>(weight.shape().at(0));
  Tensor out({rows, out_dim}, output_dtype, device);
  op::matmul(x, weight, out);
  op::add_bias_inplace(out, bias);
  return out;
}

}  // namespace

LlamaRunner::LlamaRunner(ModelWeights weights, DeviceType device, int max_seq_len)
    : weights_(std::move(weights)), device_(device) {
  const auto& cfg = weights_.config();
  activation_dtype_ = device_ == DeviceType::CUDA ? DType::F16 : DType::F32;
  max_seq_len_ = max_seq_len > 0 ? max_seq_len : cfg.max_seq_len;
  if (max_seq_len_ <= 0) {
    throw std::runtime_error("max_seq_len must be positive");
  }
  if (cfg.num_heads % cfg.num_kv_heads != 0) {
    throw std::runtime_error("num_heads must be divisible by num_kv_heads");
  }

  k_cache_.reserve(cfg.num_layers);
  v_cache_.reserve(cfg.num_layers);
  for (int i = 0; i < cfg.num_layers; ++i) {
    k_cache_.emplace_back(
        std::vector<std::int64_t>{max_seq_len_, cfg.num_kv_heads, cfg.head_dim},
        activation_dtype_,
        device_);
    v_cache_.emplace_back(
        std::vector<std::int64_t>{max_seq_len_, cfg.num_kv_heads, cfg.head_dim},
        activation_dtype_,
        device_);
  }
}

std::string LlamaRunner::layer_name(int layer, const std::string& suffix) const {
  std::ostringstream oss;
  oss << "model.layers." << layer << "." << suffix;
  return oss.str();
}

const Tensor& LlamaRunner::tensor(const std::string& name) const {
  return weights_.get(name);
}

const Tensor* LlamaRunner::optional_tensor(const std::string& name) const {
  return weights_.find(name);
}

Tensor LlamaRunner::forward(const std::vector<int>& input_ids, int past_len) {
  if (input_ids.empty()) {
    throw std::runtime_error("forward input_ids is empty");
  }
  const auto& cfg = weights_.config();
  const int seq = static_cast<int>(input_ids.size());
  if (past_len < 0 || past_len + seq > max_seq_len_) {
    throw std::runtime_error("sequence length exceeds KV cache capacity");
  }

  Tensor x({seq, cfg.hidden_size}, activation_dtype_, device_);
  op::embedding(tensor("model.embed_tokens.weight"), input_ids, x);

  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    Tensor attn_norm({seq, cfg.hidden_size}, activation_dtype_, device_);
    op::rms_norm(
        x,
        tensor(layer_name(layer, "input_layernorm.weight")),
        attn_norm,
        cfg.rms_norm_eps);

    Tensor q = linear(
        attn_norm,
        tensor(layer_name(layer, "self_attn.q_proj.weight")),
        optional_tensor(layer_name(layer, "self_attn.q_proj.bias")),
        device_,
        activation_dtype_);
    Tensor k = linear(
        attn_norm,
        tensor(layer_name(layer, "self_attn.k_proj.weight")),
        optional_tensor(layer_name(layer, "self_attn.k_proj.bias")),
        device_,
        activation_dtype_);
    Tensor v = linear(
        attn_norm,
        tensor(layer_name(layer, "self_attn.v_proj.weight")),
        optional_tensor(layer_name(layer, "self_attn.v_proj.bias")),
        device_,
        activation_dtype_);

    q.reshape({seq, cfg.num_heads, cfg.head_dim});
    k.reshape({seq, cfg.num_kv_heads, cfg.head_dim});
    v.reshape({seq, cfg.num_kv_heads, cfg.head_dim});
    op::rope_inplace(q, past_len, cfg.rope_theta);
    op::rope_inplace(k, past_len, cfg.rope_theta);

    op::copy_kv_to_cache(k, v, k_cache_[layer], v_cache_[layer], past_len);

    Tensor attn_ctx({seq, cfg.num_heads, cfg.head_dim}, activation_dtype_, device_);
    op::attention(
        q,
        k_cache_[layer],
        v_cache_[layer],
        attn_ctx,
        past_len,
        past_len + seq,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim);
    attn_ctx.reshape({seq, cfg.num_heads * cfg.head_dim});

    Tensor attn_out = linear(
        attn_ctx,
        tensor(layer_name(layer, "self_attn.o_proj.weight")),
        optional_tensor(layer_name(layer, "self_attn.o_proj.bias")),
        device_,
        activation_dtype_);
    op::add_inplace(attn_out, x);
    x = std::move(attn_out);

    Tensor ffn_norm({seq, cfg.hidden_size}, activation_dtype_, device_);
    op::rms_norm(
        x,
        tensor(layer_name(layer, "post_attention_layernorm.weight")),
        ffn_norm,
        cfg.rms_norm_eps);

    Tensor gate = linear(
        ffn_norm,
        tensor(layer_name(layer, "mlp.gate_proj.weight")),
        optional_tensor(layer_name(layer, "mlp.gate_proj.bias")),
        device_,
        activation_dtype_);
    Tensor up = linear(
        ffn_norm,
        tensor(layer_name(layer, "mlp.up_proj.weight")),
        optional_tensor(layer_name(layer, "mlp.up_proj.bias")),
        device_,
        activation_dtype_);
    Tensor swiglu({seq, cfg.intermediate_size}, activation_dtype_, device_);
    op::silu_mul(gate, up, swiglu);

    Tensor ffn_out = linear(
        swiglu,
        tensor(layer_name(layer, "mlp.down_proj.weight")),
        optional_tensor(layer_name(layer, "mlp.down_proj.bias")),
        device_,
        activation_dtype_);
    op::add_inplace(ffn_out, x);
    x = std::move(ffn_out);
  }

  Tensor normed({seq, cfg.hidden_size}, activation_dtype_, device_);
  op::rms_norm(x, tensor("model.norm.weight"), normed, cfg.rms_norm_eps);
  Tensor last({1, cfg.hidden_size}, activation_dtype_, device_);
  op::select_last_row(normed, last);

  const Tensor* lm_head = optional_tensor("lm_head.weight");
  if (lm_head == nullptr && cfg.tie_word_embeddings) {
    lm_head = &tensor("model.embed_tokens.weight");
  }
  if (lm_head == nullptr) {
    throw std::runtime_error("missing lm_head.weight");
  }
  return linear(last, *lm_head, nullptr, device_, activation_dtype_);
}

void LlamaRunner::reset_cache() {
  const auto& cfg = weights_.config();
  k_cache_.clear();
  v_cache_.clear();
  for (int i = 0; i < cfg.num_layers; ++i) {
    k_cache_.emplace_back(
        std::vector<std::int64_t>{max_seq_len_, cfg.num_kv_heads, cfg.head_dim},
        activation_dtype_,
        device_);
    v_cache_.emplace_back(
        std::vector<std::int64_t>{max_seq_len_, cfg.num_kv_heads, cfg.head_dim},
        activation_dtype_,
        device_);
  }
}

}  // namespace kllm
