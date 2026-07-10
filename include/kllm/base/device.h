#pragma once

#include <string>

namespace kllm {

enum class DeviceType {
  CPU = 0,
  CUDA = 1,
};

DeviceType parse_device(const std::string& value);
std::string to_string(DeviceType device);
bool cuda_available();

}  // namespace kllm

