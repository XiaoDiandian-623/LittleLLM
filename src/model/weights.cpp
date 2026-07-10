#include "kllm/model/weights.h"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace kllm {

namespace {

template <typename T>
T read_pod(std::ifstream& in) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) {
    throw std::runtime_error("unexpected end of model file");
  }
  return value;
}

void read_exact(std::ifstream& in, void* data, std::size_t bytes) {
  in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
  if (!in) {
    throw std::runtime_error("unexpected end of model file");
  }
}

bool read_bool(std::ifstream& in) {
  return read_pod<std::uint8_t>(in) != 0;
}

}  // namespace

std::string model_type_name(ModelType type) {
  switch (type) {
    case ModelType::Llama:
      return "llama";
    case ModelType::Qwen2:
      return "qwen2";
  }
  return "unknown";
}

ModelWeights ModelWeights::load(const std::string& path, DeviceType device) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open model file: " + path);
  }

  char magic[8] = {};
  read_exact(in, magic, sizeof(magic));
  if (std::string(magic, sizeof(magic)) != "KLLM0001") {
    throw std::runtime_error("invalid model file magic, expected KLLM0001");
  }

  const auto version = read_pod<std::uint32_t>(in);
  if (version != 1) {
    throw std::runtime_error("unsupported model file version");
  }

  ModelWeights result;
  result.config_.model_type = static_cast<ModelType>(read_pod<std::uint32_t>(in));
  result.config_.vocab_size = read_pod<std::int32_t>(in);
  result.config_.hidden_size = read_pod<std::int32_t>(in);
  result.config_.intermediate_size = read_pod<std::int32_t>(in);
  result.config_.num_layers = read_pod<std::int32_t>(in);
  result.config_.num_heads = read_pod<std::int32_t>(in);
  result.config_.num_kv_heads = read_pod<std::int32_t>(in);
  result.config_.head_dim = read_pod<std::int32_t>(in);
  result.config_.max_seq_len = read_pod<std::int32_t>(in);
  result.config_.rms_norm_eps = read_pod<float>(in);
  result.config_.rope_theta = read_pod<float>(in);
  result.config_.tie_word_embeddings = read_bool(in);
  result.config_.qkv_bias = read_bool(in);
  result.config_.o_bias = read_bool(in);
  result.config_.mlp_bias = read_bool(in);

  const auto tensor_count = read_pod<std::uint32_t>(in);
  for (std::uint32_t i = 0; i < tensor_count; ++i) {
    const auto name_len = read_pod<std::uint32_t>(in);
    std::string name(name_len, '\0');
    read_exact(in, &name[0], name_len);

    const auto dtype = static_cast<DType>(read_pod<std::uint32_t>(in));
    const auto ndim = read_pod<std::uint32_t>(in);
    std::vector<std::int64_t> shape(ndim);
    for (std::uint32_t d = 0; d < ndim; ++d) {
      shape[d] = read_pod<std::int64_t>(in);
    }
    const auto bytes = read_pod<std::uint64_t>(in);
    Tensor tensor(shape, dtype, device);
    if (bytes != tensor.nbytes()) {
      throw std::runtime_error("tensor byte size mismatch: " + name);
    }
    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(bytes));
    read_exact(in, buffer.data(), buffer.size());
    tensor.copy_from_host(buffer.data(), buffer.size());
    result.tensors_.emplace(std::move(name), std::move(tensor));
  }

  return result;
}

const Tensor& ModelWeights::get(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::runtime_error("missing tensor: " + name);
  }
  return it->second;
}

const Tensor* ModelWeights::find(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    return nullptr;
  }
  return &it->second;
}

}  // namespace kllm
