#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kllm/base/device.h"

namespace kllm {

enum class DType : std::uint32_t {
  F32 = 0,
  F16 = 1,
  I32 = 2,
  I8 = 3,
};

std::size_t dtype_size(DType dtype);
std::string dtype_name(DType dtype);

class Tensor {
 public:
  Tensor() = default;
  Tensor(std::vector<std::int64_t> shape, DType dtype, DeviceType device);
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;
  ~Tensor();

  void reset();
  void allocate(std::vector<std::int64_t> shape, DType dtype, DeviceType device);
  void reshape(std::vector<std::int64_t> shape);

  void* data() { return data_; }
  const void* data() const { return data_; }

  template <typename T>
  T* data_as() {
    return static_cast<T*>(data_);
  }

  template <typename T>
  const T* data_as() const {
    return static_cast<const T*>(data_);
  }

  const std::vector<std::int64_t>& shape() const { return shape_; }
  DType dtype() const { return dtype_; }
  DeviceType device() const { return device_; }
  std::size_t numel() const { return numel_; }
  std::size_t nbytes() const { return numel_ * dtype_size(dtype_); }
  bool empty() const { return data_ == nullptr; }

  void copy_from_host(const void* source, std::size_t bytes);
  void copy_to_host(void* target, std::size_t bytes) const;
  std::vector<float> to_float_vector() const;

 private:
  void* data_ = nullptr;
  std::vector<std::int64_t> shape_;
  DType dtype_ = DType::F32;
  DeviceType device_ = DeviceType::CPU;
  std::size_t numel_ = 0;
};

}  // namespace kllm
