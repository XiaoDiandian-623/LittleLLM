#include "kllm/op/cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace kllm::op::cpu {

void quantize_int8(const Tensor& input, Tensor& out_data, Tensor& out_scale) {
  const auto* in = input.data_as<float>();
  auto* out = out_data.data_as<std::int8_t>();
  auto* scale = out_scale.data_as<float>();

  const std::size_t numel = input.numel();

  float max_val = 0.0f;
  for (std::size_t i = 0; i < numel; ++i) {
    max_val = std::max(max_val, std::abs(in[i]));
  }

  const float s = max_val / 127.0f;
  scale[0] = s;

  if (s == 0.0f) {
    std::fill(out, out + numel, 0);
    return;
  }

  const float inv_scale = 1.0f / s;
  for (std::size_t i = 0; i < numel; ++i) {
    float val = std::round(in[i] * inv_scale);
    val = std::max(-128.0f, std::min(127.0f, val));
    out[i] = static_cast<std::int8_t>(val);
  }
}

void dequantize_int8(const Tensor& data, const Tensor& scale, Tensor& output) {
  const auto* in = data.data_as<std::int8_t>();
  const auto* s = scale.data_as<float>();
  auto* out = output.data_as<float>();

  const std::size_t numel = data.numel();
  const float scale_val = s[0];

  for (std::size_t i = 0; i < numel; ++i) {
    out[i] = static_cast<float>(in[i]) * scale_val;
  }
}

void matmul_int8(const Tensor& a, const Tensor& b_quant, const Tensor& b_scale, Tensor& out) {
  const auto& a_shape = a.shape();
  const auto& b_shape = b_quant.shape();
  const auto& out_shape = out.shape();

  const std::int64_t M = a_shape[0];
  const std::int64_t K = a_shape[1];
  const std::int64_t N = b_shape[1];

  const auto* a_data = a.data_as<float>();
  const auto* b_data = b_quant.data_as<std::int8_t>();
  const auto* scale_data = b_scale.data_as<float>();
  auto* out_data = out.data_as<float>();

  std::fill(out_data, out_data + M * N, 0.0f);

  for (std::int64_t m = 0; m < M; ++m) {
    for (std::int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (std::int64_t k = 0; k < K; ++k) {
        const float a_val = a_data[m * K + k];
        const std::int8_t b_val = b_data[k * N + n];
        sum += a_val * static_cast<float>(b_val);
      }
      out_data[m * N + n] = sum * scale_data[n];
    }
  }
}

}  // namespace kllm::op::cpu
