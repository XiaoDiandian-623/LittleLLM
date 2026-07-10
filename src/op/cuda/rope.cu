#include "kllm/op/cuda_ops.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <unordered_map>
#include <mutex>

#include "kllm/op/detail/cuda_helpers.cuh"

namespace kllm::op {

namespace {

// --- Cached device buffer for inv_freq ---

struct InvFreqKey {
  int head_dim;
  float theta;

  bool operator==(const InvFreqKey& other) const {
    return head_dim == other.head_dim && theta == other.theta;
  }
};

struct InvFreqKeyHash {
  std::size_t operator()(const InvFreqKey& k) const {
    return std::hash<int>()(k.head_dim) ^ (std::hash<float>()(k.theta) << 1);
  }
};

class InvFreqCache {
 public:
  ~InvFreqCache() {
    for (auto& pair : cache_) {
      if (pair.second) {
        cudaFree(pair.second);
      }
    }
  }

  float* get_or_create(int head_dim, float theta) {
    std::lock_guard<std::mutex> lock(mutex_);
    InvFreqKey key{head_dim, theta};

    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }

    const int half_dim = head_dim / 2;
    std::vector<float> inv_freq(half_dim);
    for (int d = 0; d < half_dim; ++d) {
      inv_freq[d] = std::pow(theta, -2.0f * static_cast<float>(d) / static_cast<float>(head_dim));
    }

    float* d_inv_freq = nullptr;
    const std::size_t bytes = half_dim * sizeof(float);
    detail::check_cuda(cudaMalloc(&d_inv_freq, bytes));
    detail::check_cuda(cudaMemcpy(d_inv_freq, inv_freq.data(), bytes, cudaMemcpyHostToDevice));

    cache_[key] = d_inv_freq;
    return d_inv_freq;
  }

 private:
  std::unordered_map<InvFreqKey, float*, InvFreqKeyHash> cache_;
  std::mutex mutex_;
};

InvFreqCache& get_inv_freq_cache() {
  static InvFreqCache cache;
  return cache;
}

// --- Generic kernel: templated by dtype, uses precomputed inv_freq ---

template <typename T>
__global__ void rope_kernel(
    T* __restrict__ x,
    const float* __restrict__ inv_freq,
    int seq,
    int heads,
    int head_dim,
    int past_len) {
  const int half_dim = head_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = seq * heads * half_dim;
  if (idx >= total) {
    return;
  }

  const int d = idx % half_dim;
  const int h = (idx / half_dim) % heads;
  const int t = idx / (half_dim * heads);
  const int position = past_len + t;

  const std::size_t row_off = (static_cast<std::size_t>(t) * heads + h) * head_dim;
  T* base = x + row_off;

  float x1 = static_cast<float>(base[d]);
  float x2 = static_cast<float>(base[d + half_dim]);

  float s, c;
  __sincosf(static_cast<float>(position) * inv_freq[d], &s, &c);

  base[d] = static_cast<T>(x1 * c - x2 * s);
  base[d + half_dim] = static_cast<T>(x2 * c + x1 * s);
}

// --- FP16 vectorized kernel: process 2 adjacent d's per thread via half2 ---

__global__ void rope_kernel_f16_vec2(
    __half* __restrict__ x,
    const float* __restrict__ inv_freq,
    int seq,
    int heads,
    int head_dim,
    int past_len) {
  const int half_dim = head_dim / 2;
  const int half_dim_vec = half_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = seq * heads * half_dim_vec;
  if (idx >= total) {
    return;
  }

  const int dv = idx % half_dim_vec;
  const int h = (idx / half_dim_vec) % heads;
  const int t = idx / (half_dim_vec * heads);
  const int d = dv * 2;
  const int position = past_len + t;

  const std::size_t row_off = (static_cast<std::size_t>(t) * heads + h) * head_dim;
  __half* base = x + row_off;

  half2 x1_vec = __ldg(reinterpret_cast<const half2*>(base + d));
  half2 x2_vec = __ldg(reinterpret_cast<const half2*>(base + d + half_dim));

  float2 x1f = __half22float2(x1_vec);
  float2 x2f = __half22float2(x2_vec);

  float s0, c0, s1, c1;
  __sincosf(static_cast<float>(position) * inv_freq[d], &s0, &c0);
  __sincosf(static_cast<float>(position) * inv_freq[d + 1], &s1, &c1);

  float r1_0 = x1f.x * c0 - x2f.x * s0;
  float r2_0 = x2f.x * c0 + x1f.x * s0;
  float r1_1 = x1f.y * c1 - x2f.y * s1;
  float r2_1 = x2f.y * c1 + x1f.y * s1;

  reinterpret_cast<half2*>(base + d)[0] = __float22half2_rn(make_float2(r1_0, r1_1));
  reinterpret_cast<half2*>(base + d + half_dim)[0] = __float22half2_rn(make_float2(r2_0, r2_1));
}

// --- Host launch functions ---

template <typename T>
void launch_rope(Tensor& q_or_k, int past_len, float theta, float* d_inv_freq) {
  const int seq = static_cast<int>(q_or_k.shape().at(0));
  const int heads = static_cast<int>(q_or_k.shape().at(1));
  const int head_dim = static_cast<int>(q_or_k.shape().at(2));
  const int half_dim = head_dim / 2;
  const int total = seq * heads * half_dim;

  constexpr int threads = 256;
  rope_kernel<T><<<(total + threads - 1) / threads, threads>>>(
      q_or_k.data_as<T>(), d_inv_freq, seq, heads, head_dim, past_len);
  detail::check_cuda(cudaGetLastError());
}

void launch_rope_f16_vec2(Tensor& q_or_k, int past_len, float theta, float* d_inv_freq) {
  const int seq = static_cast<int>(q_or_k.shape().at(0));
  const int heads = static_cast<int>(q_or_k.shape().at(1));
  const int head_dim = static_cast<int>(q_or_k.shape().at(2));
  const int half_dim = head_dim / 2;
  const int half_dim_vec = half_dim / 2;
  const int total = seq * heads * half_dim_vec;

  constexpr int threads = 256;
  rope_kernel_f16_vec2<<<(total + threads - 1) / threads, threads>>>(
      q_or_k.data_as<__half>(), d_inv_freq, seq, heads, head_dim, past_len);
  detail::check_cuda(cudaGetLastError());
}

}  // namespace

void cuda_rope_inplace(Tensor& q_or_k, int past_len, float theta) {
  const int head_dim = static_cast<int>(q_or_k.shape().at(2));
  const int half_dim = head_dim / 2;

  float* d_inv_freq = get_inv_freq_cache().get_or_create(head_dim, theta);

  if (q_or_k.dtype() == DType::F16 && half_dim % 2 == 0) {
    launch_rope_f16_vec2(q_or_k, past_len, theta, d_inv_freq);
  } else if (q_or_k.dtype() == DType::F16) {
    launch_rope<__half>(q_or_k, past_len, theta, d_inv_freq);
  } else {
    launch_rope<float>(q_or_k, past_len, theta, d_inv_freq);
  }
}

}  // namespace kllm::op
