#include "kllm/base/half.h"

#include <cstring>

namespace kllm {

float half_to_float(float16_t value) {
  const std::uint32_t sign = (value & 0x8000u) << 16u;
  std::uint32_t exponent = (value >> 10u) & 0x1fu;
  std::uint32_t mantissa = value & 0x03ffu;

  std::uint32_t out = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      out = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1u;
        --exponent;
      }
      mantissa &= 0x03ffu;
      exponent = exponent + (127u - 15u);
      out = sign | (exponent << 23u) | (mantissa << 13u);
    }
  } else if (exponent == 0x1fu) {
    out = sign | 0x7f800000u | (mantissa << 13u);
  } else {
    exponent = exponent + (127u - 15u);
    out = sign | (exponent << 23u) | (mantissa << 13u);
  }

  float result = 0.0f;
  std::memcpy(&result, &out, sizeof(result));
  return result;
}

float16_t float_to_half(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const std::uint32_t sign = (bits >> 16u) & 0x8000u;
  std::int32_t exponent = static_cast<std::int32_t>((bits >> 23u) & 0xffu) - 127 + 15;
  std::uint32_t mantissa = bits & 0x007fffffu;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<float16_t>(sign);
    }
    mantissa = (mantissa | 0x00800000u) >> static_cast<std::uint32_t>(1 - exponent);
    return static_cast<float16_t>(sign | ((mantissa + 0x00001000u) >> 13u));
  }

  if (exponent >= 31) {
    return static_cast<float16_t>(sign | 0x7c00u);
  }

  return static_cast<float16_t>(
      sign | (static_cast<std::uint32_t>(exponent) << 10u) |
      ((mantissa + 0x00001000u) >> 13u));
}

}  // namespace kllm

