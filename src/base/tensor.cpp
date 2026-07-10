#include "kllm/base/tensor.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "kllm/base/half.h"

#if KLLM_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace kllm {

namespace {

std::size_t compute_numel(const std::vector<std::int64_t>& shape) {
  std::size_t n = 1;
  for (const auto dim : shape) {
    if (dim < 0) {
      throw std::runtime_error("negative tensor shape is invalid");
    }
    n *= static_cast<std::size_t>(dim);
  }
  return n;
}

#if KLLM_USE_CUDA
void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}
#endif

}  // namespace

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::F32:
      return sizeof(float);
    case DType::F16:
      return sizeof(float16_t);
    case DType::I32:
      return sizeof(std::int32_t);
    case DType::I8:
      return sizeof(std::int8_t);
  }
  throw std::runtime_error("unknown dtype");
}

std::string dtype_name(DType dtype) {
  switch (dtype) {
    case DType::F32:
      return "f32";
    case DType::F16:
      return "f16";
    case DType::I32:
      return "i32";
    case DType::I8:
      return "i8";
  }
  return "unknown";
}

DeviceType parse_device(const std::string& value) {
  if (value == "cpu") {
    return DeviceType::CPU;
  }
  if (value == "cuda") {
    return DeviceType::CUDA;
  }
  throw std::runtime_error("device must be cpu or cuda");
}

std::string to_string(DeviceType device) {
  return device == DeviceType::CUDA ? "cuda" : "cpu";
}

bool cuda_available() {
#if KLLM_USE_CUDA
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
  return false;
#endif
}

Tensor::Tensor(std::vector<std::int64_t> shape, DType dtype, DeviceType device) {
  allocate(std::move(shape), dtype, device);
}

Tensor::Tensor(Tensor&& other) noexcept {
  *this = std::move(other);
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    reset();
    data_ = other.data_;
    shape_ = std::move(other.shape_);
    dtype_ = other.dtype_;
    device_ = other.device_;
    numel_ = other.numel_;
    other.data_ = nullptr;
    other.numel_ = 0;
  }
  return *this;
}

Tensor::~Tensor() {
  reset();
}

void Tensor::reset() {
  if (data_ == nullptr) {
    return;
  }
  if (device_ == DeviceType::CPU) {
    std::free(data_);
  } else {
#if KLLM_USE_CUDA
    cudaFree(data_);
#endif
  }
  data_ = nullptr;
  shape_.clear();
  numel_ = 0;
}

void Tensor::allocate(std::vector<std::int64_t> shape, DType dtype, DeviceType device) {
  reset();
  shape_ = std::move(shape);
  dtype_ = dtype;
  device_ = device;
  numel_ = compute_numel(shape_);
  if (numel_ == 0) {
    return;
  }

  const std::size_t bytes = nbytes();
  if (device_ == DeviceType::CPU) {
    data_ = std::malloc(bytes);
    if (data_ == nullptr) {
      throw std::bad_alloc();
    }
    std::memset(data_, 0, bytes);
  } else {
#if KLLM_USE_CUDA
    check_cuda(cudaMalloc(&data_, bytes));
    check_cuda(cudaMemset(data_, 0, bytes));
#else
    throw std::runtime_error("CUDA backend was not compiled. Rebuild with -DKLLM_USE_CUDA=ON.");
#endif
  }
}

void Tensor::reshape(std::vector<std::int64_t> shape) {
  if (compute_numel(shape) != numel_) {
    throw std::runtime_error("reshape changes tensor numel");
  }
  shape_ = std::move(shape);
}

void Tensor::copy_from_host(const void* source, std::size_t bytes) {
  if (bytes > nbytes()) {
    throw std::runtime_error("copy_from_host exceeds tensor size");
  }
  if (device_ == DeviceType::CPU) {
    std::memcpy(data_, source, bytes);
  } else {
#if KLLM_USE_CUDA
    check_cuda(cudaMemcpy(data_, source, bytes, cudaMemcpyHostToDevice));
#else
    throw std::runtime_error("CUDA backend was not compiled. Rebuild with -DKLLM_USE_CUDA=ON.");
#endif
  }
}

void Tensor::copy_to_host(void* target, std::size_t bytes) const {
  if (bytes > nbytes()) {
    throw std::runtime_error("copy_to_host exceeds tensor size");
  }
  if (device_ == DeviceType::CPU) {
    std::memcpy(target, data_, bytes);
  } else {
#if KLLM_USE_CUDA
    check_cuda(cudaMemcpy(target, data_, bytes, cudaMemcpyDeviceToHost));
#else
    throw std::runtime_error("CUDA backend was not compiled. Rebuild with -DKLLM_USE_CUDA=ON.");
#endif
  }
}

std::vector<float> Tensor::to_float_vector() const {
  std::vector<float> result(numel_);
  if (dtype_ == DType::F32) {
    copy_to_host(result.data(), result.size() * sizeof(float));
    return result;
  }

  if (dtype_ == DType::F16) {
    std::vector<float16_t> tmp(numel_);
    copy_to_host(tmp.data(), tmp.size() * sizeof(float16_t));
    for (std::size_t i = 0; i < tmp.size(); ++i) {
      result[i] = half_to_float(tmp[i]);
    }
    return result;
  }

  if (dtype_ == DType::I32) {
    std::vector<std::int32_t> tmp(numel_);
    copy_to_host(tmp.data(), tmp.size() * sizeof(std::int32_t));
    for (std::size_t i = 0; i < tmp.size(); ++i) {
      result[i] = static_cast<float>(tmp[i]);
    }
    return result;
  }

  if (dtype_ == DType::I8) {
    std::vector<std::int8_t> tmp(numel_);
    copy_to_host(tmp.data(), tmp.size() * sizeof(std::int8_t));
    for (std::size_t i = 0; i < tmp.size(); ++i) {
      result[i] = static_cast<float>(tmp[i]);
    }
    return result;
  }

  throw std::runtime_error("unsupported dtype conversion");
}

}  // namespace kllm
