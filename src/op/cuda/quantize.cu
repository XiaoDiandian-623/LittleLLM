#include "kllm/op/cuda_ops.h"
#include "kllm/op/detail/cuda_helpers.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

namespace kllm::op::cuda {

__global__ void find_max_abs_kernel(const float* input, float* partial_max, int n) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float local_max = 0.0f;
  if (idx < n) {
    local_max = fabsf(input[idx]);
  }

  sdata[tid] = local_max;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && tid + s < blockDim.x) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_max[blockIdx.x] = sdata[0];
  }
}

__global__ void quantize_kernel(const float* input, std::int8_t* output, float scale, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float val = input[idx] / scale;
  val = fmaxf(-128.0f, fminf(127.0f, roundf(val)));
  output[idx] = static_cast<std::int8_t>(val);
}

__global__ void dequantize_kernel(const std::int8_t* input, const float* scale, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  output[idx] = static_cast<float>(input[idx]) * scale[0];
}

__global__ void matmul_int8_kernel(
    const float* a,
    const std::int8_t* b,
    const float* scale,
    float* out,
    int M,
    int K,
    int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) return;

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    float a_val = a[row * K + k];
    std::int8_t b_val = b[k * N + col];
    sum += a_val * static_cast<float>(b_val);
  }

  out[row * N + col] = sum * scale[col];
}

void quantize_int8(const Tensor& input, Tensor& out_data, Tensor& out_scale) {
  const int n = static_cast<int>(input.numel());
  const auto* in = input.data_as<float>();
  auto* out = out_data.data_as<std::int8_t>();
  auto* scale = out_scale.data_as<float>();

  const int block_size = 256;
  const int num_blocks = (n + block_size - 1) / block_size;

  float* d_partial_max;
  cudaMalloc(&d_partial_max, num_blocks * sizeof(float));

  find_max_abs_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
      in, d_partial_max, n);

  std::vector<float> h_partial_max(num_blocks);
  cudaMemcpy(h_partial_max.data(), d_partial_max, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

  float max_val = *std::max_element(h_partial_max.begin(), h_partial_max.end());
  float scale_val = max_val / 127.0f;

  cudaMemcpy(scale, &scale_val, sizeof(float), cudaMemcpyHostToDevice);

  if (scale_val > 0.0f) {
    quantize_kernel<<<num_blocks, block_size>>>(in, out, scale_val, n);
  } else {
    cudaMemset(out, 0, n * sizeof(std::int8_t));
  }

  cudaFree(d_partial_max);
  check_last_cuda_error();
}

void dequantize_int8(const Tensor& data, const Tensor& scale, Tensor& output) {
  const int n = static_cast<int>(data.numel());
  const auto* in = data.data_as<std::int8_t>();
  const auto* s = scale.data_as<float>();
  auto* out = output.data_as<float>();

  const int block_size = 256;
  const int num_blocks = (n + block_size - 1) / block_size;

  dequantize_kernel<<<num_blocks, block_size>>>(in, s, out, n);
  check_last_cuda_error();
}

void matmul_int8(const Tensor& a, const Tensor& b_quant, const Tensor& b_scale, Tensor& out) {
  const auto& a_shape = a.shape();
  const auto& b_shape = b_quant.shape();

  const int M = static_cast<int>(a_shape[0]);
  const int K = static_cast<int>(a_shape[1]);
  const int N = static_cast<int>(b_shape[1]);

  const auto* a_data = a.data_as<float>();
  const auto* b_data = b_quant.data_as<std::int8_t>();
  const auto* scale_data = b_scale.data_as<float>();
  auto* out_data = out.data_as<float>();

  dim3 block_size(16, 16);
  dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);

  matmul_int8_kernel<<<grid_size, block_size>>>(a_data, b_data, scale_data, out_data, M, K, N);
  check_last_cuda_error();
}

}  // namespace kllm::op::cuda
