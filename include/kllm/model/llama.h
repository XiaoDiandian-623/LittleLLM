#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kllm/base/tensor.h"
#include "kllm/model/weights.h"

namespace kllm {

class LlamaRunner {
 public:
  LlamaRunner(ModelWeights weights, DeviceType device, int max_seq_len);

  const ModelConfig& config() const { return weights_.config(); }
  DeviceType device() const { return device_; }

  Tensor forward(const std::vector<int>& input_ids, int past_len);
  void reset_cache();

 private:
  std::string layer_name(int layer, const std::string& suffix) const;
  const Tensor& tensor(const std::string& name) const;
  const Tensor* optional_tensor(const std::string& name) const;

  ModelWeights weights_;
  DeviceType device_;
  DType activation_dtype_ = DType::F32;
  int max_seq_len_;
  std::vector<Tensor> k_cache_;
  std::vector<Tensor> v_cache_;
};

}  // namespace kllm
