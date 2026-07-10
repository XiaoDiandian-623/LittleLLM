#include "kllm/op/cuda_ops.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "kllm/op/detail/cuda_helpers.cuh"

namespace kllm::op {

namespace {

constexpr int WARP_SIZE = 32;

__device__ inline float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__device__ inline float warp_reduce_max(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template <typename T>
__global__ void attention_kernel_optimized(
    const T* __restrict__ q,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    T* __restrict__ out,
    int seq,
    int past_len,
    int num_heads,
    int num_kv_heads,
    int head_dim) {
  const int t = blockIdx.x;
  const int h = blockIdx.y;

  if (t >= seq || h >= num_heads) {
    return;
  }

  const int groups = num_heads / num_kv_heads;
  const int kv_h = h / groups;
  const int key_limit = past_len + t + 1;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;

  extern __shared__ float smem[];
  float* scores = smem;
  float* warp_maxs = smem + key_limit;

  const T* q_row = q + (static_cast<std::size_t>(t) * num_heads + h) * head_dim;

  // Step 1: Compute Q·K^T scores in parallel and find max
  float local_max = -3.402823466e+38F;

  for (int pos = tid; pos < key_limit; pos += blockDim.x) {
    const T* k_row = k_cache + (static_cast<std::size_t>(pos) * num_kv_heads + kv_h) * head_dim;

    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += static_cast<float>(q_row[d]) * static_cast<float>(k_row[d]);
    }
    dot *= scale;
    scores[pos] = dot;
    local_max = fmaxf(local_max, dot);
  }

  // Warp-level max reduction
  local_max = warp_reduce_max(local_max);

  if (lane_id == 0) {
    warp_maxs[warp_id] = local_max;
  }
  __syncthreads();

  // Block-level max reduction
  if (warp_id == 0 && lane_id < num_warps) {
    local_max = warp_maxs[lane_id];
  } else if (warp_id == 0) {
    local_max = -3.402823466e+38F;
  }

  if (warp_id == 0) {
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
      warp_maxs[0] = local_max;
    }
  }
  __syncthreads();

  const float global_max = warp_maxs[0];

  // Step 2: Compute exp(score - max) and sum in parallel
  float local_sum = 0.0f;
  for (int pos = tid; pos < key_limit; pos += blockDim.x) {
    float exp_score = expf(scores[pos] - global_max);
    scores[pos] = exp_score;
    local_sum += exp_score;
  }

  // Warp-level sum reduction
  local_sum = warp_reduce_sum(local_sum);

  if (lane_id == 0) {
    warp_maxs[warp_id] = local_sum;
  }
  __syncthreads();

  // Block-level sum reduction
  if (warp_id == 0 && lane_id < num_warps) {
    local_sum = warp_maxs[lane_id];
  } else if (warp_id == 0) {
    local_sum = 0.0f;
  }

  if (warp_id == 0) {
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
      warp_maxs[0] = local_sum;
    }
  }
  __syncthreads();

  const float sum_exp = warp_maxs[0];
  const float inv_sum = 1.0f / sum_exp;

  // Step 3: Compute weighted sum of V in parallel
  T* out_row = out + (static_cast<std::size_t>(t) * num_heads + h) * head_dim;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int pos = 0; pos < key_limit; ++pos) {
      const T* v_row = v_cache + (static_cast<std::size_t>(pos) * num_kv_heads + kv_h) * head_dim;
      acc += scores[pos] * static_cast<float>(v_row[d]);
    }
    out_row[d] = static_cast<T>(acc * inv_sum);
  }
}

// Fallback kernel for mixed dtypes
__global__ void attention_kernel(
    const void* q,
    std::uint32_t q_dtype,
    const void* k_cache,
    std::uint32_t k_dtype,
    const void* v_cache,
    std::uint32_t v_dtype,
    void* out,
    std::uint32_t out_dtype,
    int seq,
    int past_len,
    int total_len,
    int num_heads,
    int num_kv_heads,
    int head_dim) {
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  if (t >= seq || h >= num_heads || threadIdx.x != 0) {
    return;
  }

  const int groups = num_heads / num_kv_heads;
  const int kv_h = h / groups;
  const int key_limit = past_len + t + 1;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  for (int d = 0; d < head_dim; ++d) {
    float denom = 0.0f;
    float max_score = -3.402823466e+38F;

    for (int pos = 0; pos < key_limit; ++pos) {
      float dot = 0.0f;
      for (int inner = 0; inner < head_dim; ++inner) {
        const std::size_t q_idx = (static_cast<std::size_t>(t) * num_heads + h) * head_dim + inner;
        const std::size_t k_idx = (static_cast<std::size_t>(pos) * num_kv_heads + kv_h) * head_dim + inner;
        dot += detail::cuda_read(q, q_dtype, q_idx) * detail::cuda_read(k_cache, k_dtype, k_idx);
      }
      max_score = fmaxf(max_score, dot * scale);
    }

    float value = 0.0f;
    for (int pos = 0; pos < key_limit; ++pos) {
      float dot = 0.0f;
      for (int inner = 0; inner < head_dim; ++inner) {
        const std::size_t q_idx = (static_cast<std::size_t>(t) * num_heads + h) * head_dim + inner;
        const std::size_t k_idx = (static_cast<std::size_t>(pos) * num_kv_heads + kv_h) * head_dim + inner;
        dot += detail::cuda_read(q, q_dtype, q_idx) * detail::cuda_read(k_cache, k_dtype, k_idx);
      }
      const float score = expf(dot * scale - max_score);
      denom += score;
      const std::size_t v_idx = (static_cast<std::size_t>(pos) * num_kv_heads + kv_h) * head_dim + d;
      value += score * detail::cuda_read(v_cache, v_dtype, v_idx);
    }

    const std::size_t out_idx = (static_cast<std::size_t>(t) * num_heads + h) * head_dim + d;
    detail::cuda_write(out, out_dtype, out_idx, value / denom);
  }
  (void)total_len;
}

}  // namespace

void cuda_attention(
    const Tensor& q,
    const Tensor& k_cache,
    const Tensor& v_cache,
    Tensor& out,
    int past_len,
    int total_len,
    int num_heads,
    int num_kv_heads,
    int head_dim) {
  const int seq = static_cast<int>(q.shape().at(0));
  const dim3 grid(seq, num_heads);

  if (q.dtype() == DType::F16 && k_cache.dtype() == DType::F16 &&
      v_cache.dtype() == DType::F16 && out.dtype() == DType::F16) {
    const int key_limit_max = past_len + seq;
    constexpr int threads = 256;
    const int num_warps = threads / WARP_SIZE;
    const std::size_t smem_size = (key_limit_max + num_warps) * sizeof(float);

    attention_kernel_optimized<__half><<<grid, threads, smem_size>>>(
        q.data_as<__half>(),
        k_cache.data_as<__half>(),
        v_cache.data_as<__half>(),
        out.data_as<__half>(),
        seq, past_len, num_heads, num_kv_heads, head_dim);
  } else if (q.dtype() == DType::F32 && k_cache.dtype() == DType::F32 &&
             v_cache.dtype() == DType::F32 && out.dtype() == DType::F32) {
    const int key_limit_max = past_len + seq;
    constexpr int threads = 256;
    const int num_warps = threads / WARP_SIZE;
    const std::size_t smem_size = (key_limit_max + num_warps) * sizeof(float);

    attention_kernel_optimized<float><<<grid, threads, smem_size>>>(
        q.data_as<float>(),
        k_cache.data_as<float>(),
        v_cache.data_as<float>(),
        out.data_as<float>(),
        seq, past_len, num_heads, num_kv_heads, head_dim);
  } else {
    attention_kernel<<<grid, 1>>>(
        q.data(), detail::dtype_id(q.dtype()),
        k_cache.data(), detail::dtype_id(k_cache.dtype()),
        v_cache.data(), detail::dtype_id(v_cache.dtype()),
        out.data(), detail::dtype_id(out.dtype()),
        seq, past_len, total_len, num_heads, num_kv_heads, head_dim);
  }

  detail::check_cuda(cudaGetLastError());
}

}  // namespace kllm::op
