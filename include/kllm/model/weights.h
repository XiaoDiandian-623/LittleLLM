#pragma once

#include <string>
#include <unordered_map>

#include "kllm/base/tensor.h"
#include "kllm/model/config.h"

namespace kllm {

struct NamedTensor {
  std::string name;
  Tensor tensor;
};

class ModelWeights {
 public:
  static ModelWeights load(const std::string& path, DeviceType device);

  const ModelConfig& config() const { return config_; }
  const Tensor& get(const std::string& name) const;
  const Tensor* find(const std::string& name) const;

 private:
  ModelConfig config_;
  std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace kllm

