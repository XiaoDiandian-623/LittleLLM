#include "kllm/op/cuda_ops.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kllm/op/detail/cuda_helpers.cuh"

namespace kllm::op {

namespace {

// --- warp-level reduction ---

__device__ inline float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// --- generic kernel (F32/F16 input, any weight type) ---

template <typename T, typename WT>
__global__ void rms_norm_kernel(
    const T* __restrict__ x,
    const WT* __restrict__ weight,
    T* __restrict__ out,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  const T* x_row = x + static_cast<std::size_t>(row) * hidden;
  T* out_row = out + static_cast<std::size_t>(row) * hidden;

  // Step 1: per-thread sum of squares
  float local = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float v = static_cast<float>(__ldg(x_row + i));
    local = fmaf(v, v, local);
  }

  // Step 2: warp reduction
  local = warp_reduce_sum(local);

  // Step 3: block reduction via shared memory (one scalar per warp)
  __shared__ float warp_sums[32];  // max 32 warps with 1024 threads
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = (blockDim.x + 31) / 32;

  if (lane_id == 0) {
    warp_sums[warp_id] = local;
  }
  __syncthreads();

  // Step 4: warp 0 reduces warp-level sums and computes rsqrt
  if (warp_id == 0) {
    float total = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
    total = warp_reduce_sum(total);
    if (lane_id == 0) {
      warp_sums[0] = rsqrtf(total / static_cast<float>(hidden) + eps);
    }
  }
  __syncthreads();

  const float scale = warp_sums[0];

  // Step 5: scale x by rsqrt and weight
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float v = static_cast<float>(__ldg(x_row + i)) * scale *
              static_cast<float>(__ldg(weight + i));
    out_row[i] = static_cast<T>(v);
  }
}

// --- FP16 specialization using half2 for 2x memory throughput ---

__global__ void rms_norm_kernel_f16_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  const __half* x_row = x + static_cast<std::size_t>(row) * hidden;
  __half* out_row = out + static_cast<std::size_t>(row) * hidden;

  // Step 1: sum of squares, processing two halfs at a time
  float local = 0.0f;
  const int vec_end = hidden & ~1;  // round down to even
  for (int i = threadIdx.x * 2; i < vec_end; i += blockDim.x * 2) {
    half2 v2 = __ldg(reinterpret_cast<const half2*>(x_row + i));
    float2 vf = __half22float2(v2);
    local = fmaf(vf.x, vf.x, local);
    local = fmaf(vf.y, vf.y, local);
  }
  // tail element if hidden is odd
  if (threadIdx.x == 0 && (hidden & 1)) {
    float v = __half2float(__ldg(x_row + hidden - 1));
    local = fmaf(v, v, local);
  }

  // Step 2: warp reduction
  local = warp_reduce_sum(local);

  // Step 3: block reduction
  __shared__ float warp_sums[32];
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = (blockDim.x + 31) / 32;

  if (lane_id == 0) {
    warp_sums[warp_id] = local;
  }
  __syncthreads();

  // Step 4: final reduction + rsqrt
  if (warp_id == 0) {
    float total = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
    total = warp_reduce_sum(total);
    if (lane_id == 0) {
      warp_sums[0] = rsqrtf(total / static_cast<float>(hidden) + eps);
    }
  }
  __syncthreads();

  const float scale = warp_sums[0];

  // Step 5: apply, two halfs at a time
  for (int i = threadIdx.x * 2; i < vec_end; i += blockDim.x * 2) {
    half2 x2 = __ldg(reinterpret_cast<const half2*>(x_row + i));
    half2 w2 = __ldg(reinterpret_cast<const half2*>(weight + i));
    float2 xf = __half22float2(x2);
    float2 wf = __half22float2(w2);
    xf.x = xf.x * scale * wf.x;
    xf.y = xf.y * scale * wf.y;
    reinterpret_cast<half2*>(out_row)[i / 2] = __float22half2_rn(xf);
  }
  // tail element
  if (threadIdx.x == 0 && (hidden & 1)) {
    int i = hidden - 1;
    float v = __half2float(__ldg(x_row + i)) * scale *
              __half2float(__ldg(weight + i));
    out_row[i] = __float2half(v);
  }
}

// --- host dispatch ---

template <typename T, typename WT>
void launch_rms_norm(
    const Tensor& x, const Tensor& weight, Tensor& out, float eps) {
  const int rows = static_cast<int>(x.shape().at(0));
  const int hidden = static_cast<int>(x.shape().at(1));
  constexpr int threads = 512;
  rms_norm_kernel<T, WT>
      <<<rows, threads>>>(
          x.data_as<T>(), weight.data_as<WT>(), out.data_as<T>(),
          hidden, eps);
  detail::check_cuda(cudaGetLastError());
}

void launch_rms_norm_f16_f16(
    const Tensor& x, const Tensor& weight, Tensor& out, float eps) {
  const int rows = static_cast<int>(x.shape().at(0));
  const int hidden = static_cast<int>(x.shape().at(1));
  constexpr int threads = 512;
  rms_norm_kernel_f16_f16
      <<<rows, threads>>>(
          x.data_as<__half>(), weight.data_as<__half>(),
          out.data_as<__half>(), hidden, eps);
  detail::check_cuda(cudaGetLastError());
}

}  // namespace

void cuda_rms_norm(const Tensor& x, const Tensor& weight, Tensor& out, float eps) {
  // Dispatch to the most specialized kernel based on dtype combination
  if (x.dtype() == DType::F16 && weight.dtype() == DType::F16) {
    launch_rms_norm_f16_f16(x, weight, out, eps);
  } else if (x.dtype() == DType::F16) {
    launch_rms_norm<__half, float>(x, weight, out, eps);
  } else if (weight.dtype() == DType::F16) {
    launch_rms_norm<float, __half>(x, weight, out, eps);
  } else {
    launch_rms_norm<float, float>(x, weight, out, eps);
  }
}

}  // namespace kllm::op
