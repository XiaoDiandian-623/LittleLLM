#pragma once

#include <cstdint>

namespace kllm {

using float16_t = std::uint16_t;

float half_to_float(float16_t value);
float16_t float_to_half(float value);

}  // namespace kllm

