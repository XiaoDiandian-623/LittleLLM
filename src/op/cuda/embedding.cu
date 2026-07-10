#include "kllm/op/cuda_ops.h"

#include <cuda_runtime.h>

#include "kllm/op/detail/cuda_helpers.cuh"

namespace kllm::op {

namespace {

// --- Device buffer cache for token_ids ---

class TokenIdBufferCache {
 public:
  ~TokenIdBufferCache() {
    if (buffer_) {
      cudaFree(buffer_);
    }
  }

  int* get_buffer(std::size_t required_size) {
    if (required_size > capacity_) {
      if (buffer_) {
        cudaFree(buffer_);
      }
      const std::size_t new_capacity = required_size * 2;  // 2x growth
      detail::check_cuda(cudaMalloc(&buffer_, new_capacity * sizeof(int)));
      capacity_ = new_capacity;
    }
    return buffer_;
  }

 private:
  int* buffer_ = nullptr;
  std::size_t capacity_ = 0;
};

TokenIdBufferCache& get_token_buffer_cache() {
  static thread_local TokenIdBufferCache cache;
  return cache;
}

__global__ void embedding_kernel(
    const void* table,
    std::uint32_t table_dtype,
    const int* token_ids,
    int seq,
    int hidden,
    void* out,
    std::uint32_t out_dtype) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = seq * hidden;
  if (idx >= total) {
    return;
  }
  const int token = token_ids[idx / hidden];
  const int h = idx % hidden;
  detail::cuda_write(out, out_dtype, idx,
      detail::cuda_read(table, table_dtype, static_cast<std::size_t>(token) * hidden + h));
}

// --- Optimized kernel for F16/F32 without type dispatch ---

template <typename T>
__global__ void embedding_kernel_typed(
    const T* __restrict__ table,
    const int* __restrict__ token_ids,
    T* __restrict__ out,
    int seq,
    int hidden) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = seq * hidden;
  if (idx >= total) {
    return;
  }
  const int token = __ldg(token_ids + idx / hidden);
  const int h = idx % hidden;
  out[idx] = __ldg(table + static_cast<std::size_t>(token) * hidden + h);
}

}  // namespace

void cuda_embedding(const Tensor& table, const std::vector<int>& token_ids, Tensor& out) {
  const int seq = static_cast<int>(token_ids.size());
  const int hidden = static_cast<int>(table.shape().at(1));
  const int total = seq * hidden;

  int* d_tokens = get_token_buffer_cache().get_buffer(token_ids.size());
  detail::check_cuda(cudaMemcpy(d_tokens, token_ids.data(),
      token_ids.size() * sizeof(int), cudaMemcpyHostToDevice));

  constexpr int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  if (table.dtype() == DType::F16 && out.dtype() == DType::F16) {
    embedding_kernel_typed<__half><<<blocks, threads>>>(
        table.data_as<__half>(), d_tokens, out.data_as<__half>(), seq, hidden);
  } else if (table.dtype() == DType::F32 && out.dtype() == DType::F32) {
    embedding_kernel_typed<float><<<blocks, threads>>>(
        table.data_as<float>(), d_tokens, out.data_as<float>(), seq, hidden);
  } else {
    embedding_kernel<<<blocks, threads>>>(
        table.data(), detail::dtype_id(table.dtype()), d_tokens,
        seq, hidden, out.data(), detail::dtype_id(out.dtype()));
  }

  detail::check_cuda(cudaGetLastError());
}

}  // namespace kllm::op
