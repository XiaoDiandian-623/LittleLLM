#pragma once

#include <cstdint>
#include <string>

namespace kllm {

enum class ModelType : std::uint32_t {
  Llama = 0,
  Qwen2 = 1,
};

struct ModelConfig {
  ModelType model_type = ModelType::Llama;
  std::int32_t vocab_size = 0;
  std::int32_t hidden_size = 0;
  std::int32_t intermediate_size = 0;
  std::int32_t num_layers = 0;
  std::int32_t num_heads = 0;
  std::int32_t num_kv_heads = 0;
  std::int32_t head_dim = 0;
  std::int32_t max_seq_len = 2048;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 10000.0f;
  bool tie_word_embeddings = false;
  bool qkv_bias = false;
  bool o_bias = false;
  bool mlp_bias = false;
};

std::string model_type_name(ModelType type);

}  // namespace kllm

