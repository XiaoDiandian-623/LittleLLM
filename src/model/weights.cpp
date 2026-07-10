#include "kllm/model/weights.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "kllm/base/half.h"

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

    // For int8 quantized tensors, the file contains: quantized_data + scale (4 bytes)
    std::size_t expected_bytes = 0;
    bool is_quantized = (dtype == DType::I8);

    if (is_quantized) {
      // Calculate size for shape
      std::size_t numel = 1;
      for (const auto& dim : shape) {
        numel *= static_cast<std::size_t>(dim);
      }
      expected_bytes = numel * sizeof(std::int8_t) + sizeof(float);
    } else {
      Tensor temp(shape, dtype, DeviceType::CPU);
      expected_bytes = temp.nbytes();
    }

    if (bytes != expected_bytes) {
      throw std::runtime_error("tensor byte size mismatch: " + name +
                               " (expected " + std::to_string(expected_bytes) +
                               ", got " + std::to_string(bytes) + ")");
    }

    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(bytes));
    read_exact(in, buffer.data(), buffer.size());

    // For int8, dequantize on load to FP16/FP32
    if (is_quantized) {
      // Determine target dtype based on device
      DType target_dtype = (device == DeviceType::CUDA) ? DType::F16 : DType::F32;
      Tensor tensor(shape, target_dtype, device);

      // Extract scale from the end of buffer
      float scale;
      std::memcpy(&scale, buffer.data() + buffer.size() - sizeof(float), sizeof(float));

      // Dequantize on CPU then transfer to target device
      std::size_t numel = 1;
      for (const auto& dim : shape) {
        numel *= static_cast<std::size_t>(dim);
      }

      std::vector<float> dequantized(numel);
      const std::int8_t* quantized = reinterpret_cast<const std::int8_t*>(buffer.data());

      for (std::size_t i = 0; i < numel; ++i) {
        dequantized[i] = static_cast<float>(quantized[i]) * scale;
      }

      // Convert to target dtype and copy to device
      if (target_dtype == DType::F16) {
        std::vector<float16_t> fp16_data(numel);
        for (std::size_t i = 0; i < numel; ++i) {
          fp16_data[i] = float_to_half(dequantized[i]);
        }
        tensor.copy_from_host(fp16_data.data(), fp16_data.size() * sizeof(float16_t));
      } else {
        tensor.copy_from_host(dequantized.data(), dequantized.size() * sizeof(float));
      }

      result.tensors_.emplace(std::move(name), std::move(tensor));
    } else {
      Tensor tensor(shape, dtype, device);
      tensor.copy_from_host(buffer.data(), buffer.size());
      result.tensors_.emplace(std::move(name), std::move(tensor));
    }
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
