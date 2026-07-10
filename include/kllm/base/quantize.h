#pragma once

#include <cstdint>
#include "kllm/base/tensor.h"

namespace kllm {

struct QuantizedTensor {
  Tensor data;
  Tensor scale;
  Tensor zero_point;
};

void quantize_per_tensor(const Tensor& input, QuantizedTensor& output);
void quantize_per_channel(const Tensor& input, QuantizedTensor& output, int axis = 0);
void dequantize(const QuantizedTensor& input, Tensor& output);

}  // namespace kllm
