#include "kllm/op/cuda_ops.h"

#include <cuda_runtime.h>

#include "kllm/op/detail/cuda_helpers.cuh"

namespace kllm::op {

namespace {

__global__ void add_bias_kernel(
    void* x,
    std::uint32_t x_dtype,
    const void* bias,
    std::uint32_t bias_dtype,
    int rows,
    int cols) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  
  if (idx >= total) {
    return;
  }
  const int col = idx % cols;
  detail::cuda_write(x, x_dtype, idx, detail::cuda_read(x, x_dtype, idx) + detail::cuda_read(bias, bias_dtype, col));
}

__global__ void add_kernel(
    void* x,
    std::uint32_t x_dtype,
    const void* y,
    std::uint32_t y_dtype,
    int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  detail::cuda_write(x, x_dtype, idx, detail::cuda_read(x, x_dtype, idx) + detail::cuda_read(y, y_dtype, idx));
}

__global__ void select_last_row_kernel(
    const void* x,
    std::uint32_t x_dtype,
    void* out,
    std::uint32_t out_dtype,
    int rows,
    int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }
  const std::size_t src = static_cast<std::size_t>(rows - 1) * cols + col;
  detail::cuda_write(out, out_dtype, col, detail::cuda_read(x, x_dtype, src));
}

}  // namespace

void cuda_add_bias_inplace(Tensor& x, const Tensor* bias) {
  if (bias == nullptr) {
    return;
  }
  const int rows = static_cast<int>(x.shape().at(0));
  const int cols = static_cast<int>(x.shape().at(1));
  const int total = rows * cols;
  add_bias_kernel<<<(total + 255) / 256, 256>>>(
      x.data(), detail::dtype_id(x.dtype()), bias->data(), detail::dtype_id(bias->dtype()), rows, cols);
  detail::check_cuda(cudaGetLastError());
}

void cuda_add_inplace(Tensor& x, const Tensor& residual) {
  const int total = static_cast<int>(x.numel());
  add_kernel<<<(total + 255) / 256, 256>>>(
      x.data(), detail::dtype_id(x.dtype()), residual.data(), detail::dtype_id(residual.dtype()), total);
  detail::check_cuda(cudaGetLastError());
}

void cuda_select_last_row(const Tensor& x, Tensor& out) {
  const int rows = static_cast<int>(x.shape().at(0));
  const int cols = static_cast<int>(x.shape().at(1));
  select_last_row_kernel<<<(cols + 255) / 256, 256>>>(
      x.data(), detail::dtype_id(x.dtype()), out.data(), detail::dtype_id(out.dtype()), rows, cols);
  detail::check_cuda(cudaGetLastError());
}

}  // namespace kllm::op

