#include "kllm/op/cuda_ops.h"

#include <cuda_runtime.h>

#include "kllm/op/detail/cuda_helpers.cuh"

namespace kllm::op {

namespace {

__global__ void silu_mul_kernel(
    const void* gate,
    std::uint32_t gate_dtype,
    const void* up,
    std::uint32_t up_dtype,
    void* out,
    std::uint32_t out_dtype,
    int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const float g = detail::cuda_read(gate, gate_dtype, idx);
  const float silu = g / (1.0f + expf(-g));
  detail::cuda_write(out, out_dtype, idx, silu * detail::cuda_read(up, up_dtype, idx));
}

}  // namespace

void cuda_silu_mul(const Tensor& gate, const Tensor& up, Tensor& out) {
  const int total = static_cast<int>(gate.numel());
  silu_mul_kernel<<<(total + 255) / 256, 256>>>(
      gate.data(), detail::dtype_id(gate.dtype()), up.data(), detail::dtype_id(up.dtype()),
      out.data(), detail::dtype_id(out.dtype()), total);
  detail::check_cuda(cudaGetLastError());
}

}  // namespace kllm::op

