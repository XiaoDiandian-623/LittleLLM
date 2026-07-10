#include "kllm/op/cuda_ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kllm/op/detail/cuda_helpers.cuh"

namespace kllm::op {
namespace {

constexpr int TILE_SIZE = 32;
constexpr int BLOCK_SIZE = 16;

// Optimized tiled matmul kernel with shared memory
template<typename T>
__global__ void matmul_kernel_tiled(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out,
    int m,
    int n,
    int k) {
  __shared__ T s_a[TILE_SIZE][TILE_SIZE];
  __shared__ T s_b[TILE_SIZE][TILE_SIZE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  const int row = by * TILE_SIZE + ty;
  const int col = bx * TILE_SIZE + tx;

  float sum = 0.0f;

  const int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < num_tiles; ++t) {
    const int tile_start = t * TILE_SIZE;

    // Load tile from A into shared memory with boundary check
    if (row < m && (tile_start + tx) < k) {
      s_a[ty][tx] = a[row * k + tile_start + tx];
    } else {
      s_a[ty][tx] = T(0);
    }

    // Load tile from B into shared memory with boundary check
    if ((tile_start + ty) < k && col < n) {
      s_b[ty][tx] = b[col * k + tile_start + ty];
    } else {
      s_b[ty][tx] = T(0);
    }

    __syncthreads();

    // Compute partial dot product
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += static_cast<float>(s_a[ty][i]) * static_cast<float>(s_b[i][tx]);
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    out[row * n + col] = static_cast<T>(sum);
  }
}

// Fallback kernel for mixed types
__global__ void matmul_kernel_generic(
    const void* a,
    std::uint32_t a_dtype,
    const void* b,
    std::uint32_t b_dtype,
    void* out,
    std::uint32_t out_dtype,
    int m,
    int n,
    int k) {
  __shared__ float s_a[BLOCK_SIZE][BLOCK_SIZE + 1];
  __shared__ float s_b[BLOCK_SIZE][BLOCK_SIZE + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int col = blockIdx.x * BLOCK_SIZE + tx;
  const int row = blockIdx.y * BLOCK_SIZE + ty;

  float sum = 0.0f;
  const int num_tiles = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int t = 0; t < num_tiles; ++t) {
    const int tile_start = t * BLOCK_SIZE;

    if (row < m && (tile_start + tx) < k) {
      s_a[ty][tx] = detail::cuda_read(a, a_dtype, static_cast<std::size_t>(row) * k + tile_start + tx);
    } else {
      s_a[ty][tx] = 0.0f;
    }

    if ((tile_start + ty) < k && col < n) {
      s_b[ty][tx] = detail::cuda_read(b, b_dtype, static_cast<std::size_t>(col) * k + tile_start + ty);
    } else {
      s_b[ty][tx] = 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      sum += s_a[ty][i] * s_b[i][tx];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    detail::cuda_write(out, out_dtype, static_cast<std::size_t>(row) * n + col, sum);
  }
}
cublasHandle_t cublas_handle() {
  static thread_local cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    detail::check_cublas(cublasCreate(&handle));
    detail::check_cublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  }
  return handle;
}

bool cublas_matmul(const Tensor& a, const Tensor& b, Tensor& out, int m, int n, int k) {
  if ((a.dtype() != DType::F32 && a.dtype() != DType::F16) ||
      (b.dtype() != DType::F32 && b.dtype() != DType::F16) ||
      (out.dtype() != DType::F32 && out.dtype() != DType::F16)) {
    return false;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Row-major out[m, n] = a[m, k] * b[n, k]^T is computed as the
  // equivalent column-major out^T[n, m] = b[n, k] * a[m, k]^T.
  const cublasStatus_t status = cublasGemmEx(
      cublas_handle(),
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      b.data(),
      detail::cuda_dtype(b.dtype()),
      k,
      a.data(),
      detail::cuda_dtype(a.dtype()),
      k,
      &beta,
      out.data(),
      detail::cuda_dtype(out.dtype()),
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS;
}

template<typename T>
void launch_tiled_matmul(const Tensor& a, const Tensor& b, Tensor& out, int m, int n, int k) {
  const dim3 block(TILE_SIZE, TILE_SIZE);
  const dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

  matmul_kernel_tiled<T><<<grid, block>>>(
      static_cast<const T*>(a.data()),
      static_cast<const T*>(b.data()),
      static_cast<T*>(out.data()),
      m, n, k);

  detail::check_cuda(cudaGetLastError());
}

}  // namespace

void cuda_matmul(const Tensor& a, const Tensor& b, Tensor& out) {
  const int m = static_cast<int>(a.shape().at(0));
  const int k = static_cast<int>(a.shape().at(1));
  const int n = static_cast<int>(b.shape().at(0));

  // Try cuBLAS first for best performance
  if (cublas_matmul(a, b, out, m, n, k)) {
    return;
  }

  // Use optimized tiled kernel for homogeneous types
  if (a.dtype() == b.dtype() && a.dtype() == out.dtype()) {
    if (a.dtype() == DType::F32) {
      launch_tiled_matmul<float>(a, b, out, m, n, k);
      return;
    } else if (a.dtype() == DType::F16) {
      launch_tiled_matmul<__half>(a, b, out, m, n, k);
      return;
    }
  }

  // Fallback to generic kernel for mixed types
  const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  matmul_kernel_generic<<<grid, block>>>(
      a.data(), detail::dtype_id(a.dtype()),
      b.data(), detail::dtype_id(b.dtype()),
      out.data(), detail::dtype_id(out.dtype()),
      m, n, k);

  detail::check_cuda(cudaGetLastError());
}

}  // namespace kllm::op
