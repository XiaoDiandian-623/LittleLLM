#include "kllm/op/cuda_ops.h"

#include <cuda_runtime.h>

#include "kllm/op/detail/cuda_helpers.cuh"

namespace kllm::op {

namespace {

__global__ void copy_kv_kernel(
    const void* src,
    std::uint32_t src_dtype,
    void* dst,
    std::uint32_t dst_dtype,
    int seq,
    int row_size,
    int past_len) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = seq * row_size;
  if (idx >= total) {
    return;
  }
  const int t = idx / row_size;
  const int inner = idx % row_size;
  const std::size_t dst_idx = static_cast<std::size_t>(past_len + t) * row_size + inner;
  detail::cuda_write(dst, dst_dtype, dst_idx, detail::cuda_read(src, src_dtype, idx));
}

}  // namespace

void cuda_copy_kv_to_cache(const Tensor& k, const Tensor& v, Tensor& k_cache, Tensor& v_cache, int past_len) {
  const int seq = static_cast<int>(k.shape().at(0));
  const int row_size = static_cast<int>(k.shape().at(1) * k.shape().at(2));
  const int total = seq * row_size;
  copy_kv_kernel<<<(total + 255) / 256, 256>>>(
      k.data(), detail::dtype_id(k.dtype()), k_cache.data(), detail::dtype_id(k_cache.dtype()), seq, row_size, past_len);
  copy_kv_kernel<<<(total + 255) / 256, 256>>>(
      v.data(), detail::dtype_id(v.dtype()), v_cache.data(), detail::dtype_id(v_cache.dtype()), seq, row_size, past_len);
  detail::check_cuda(cudaGetLastError());
}

}  // namespace kllm::op

