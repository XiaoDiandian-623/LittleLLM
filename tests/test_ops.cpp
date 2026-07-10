#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kllm/base/device.h"
#include "kllm/base/half.h"
#include "kllm/base/tensor.h"
#include "kllm/op/ops.h"
#include "kllm/sampler/sampler.h"

namespace {

constexpr float kTol = 1e-4f;

void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void expect_close(float actual, float expected, float tol = kTol) {
  if (std::fabs(actual - expected) > tol) {
    fail("expected " + std::to_string(expected) + ", got " + std::to_string(actual));
  }
}

void expect_vec_close(const std::vector<float>& actual, const std::vector<float>& expected, float tol = kTol) {
  if (actual.size() != expected.size()) {
    fail("vector size mismatch");
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    expect_close(actual[i], expected[i], tol);
  }
}

kllm::Tensor make_tensor(
    const std::vector<std::int64_t>& shape,
    const std::vector<float>& values,
    kllm::DeviceType device) {
  kllm::Tensor tensor(shape, kllm::DType::F32, device);
  tensor.copy_from_host(values.data(), values.size() * sizeof(float));
  return tensor;
}

void test_half_roundtrip() {
  const float value = 1.5f;
  const auto half = kllm::float_to_half(value);
  expect_close(kllm::half_to_float(half), value, 1e-3f);
}

void test_embedding(kllm::DeviceType device) {
  auto table = make_tensor({3, 2}, {1, 2, 3, 4, 5, 6}, device);
  kllm::Tensor out({2, 2}, kllm::DType::F32, device);
  kllm::op::embedding(table, {2, 0}, out);
  expect_vec_close(out.to_float_vector(), {5, 6, 1, 2});
}

void test_matmul_and_add(kllm::DeviceType device) {
  auto a = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6}, device);
  auto b = make_tensor({2, 3}, {1, 0, 1, 0, 1, 1}, device);
  kllm::Tensor out({2, 2}, kllm::DType::F32, device);
  kllm::op::matmul(a, b, out);
  expect_vec_close(out.to_float_vector(), {4, 5, 10, 11});

  auto bias = make_tensor({2}, {1, -1}, device);
  kllm::op::add_bias_inplace(out, &bias);
  expect_vec_close(out.to_float_vector(), {5, 4, 11, 10});

  kllm::Tensor last({1, 2}, kllm::DType::F32, device);
  kllm::op::select_last_row(out, last);
  expect_vec_close(last.to_float_vector(), {11, 10});
}

void test_rms_norm(kllm::DeviceType device) {
  auto x = make_tensor({1, 2}, {3, 4}, device);
  auto w = make_tensor({2}, {1, 1}, device);
  kllm::Tensor out({1, 2}, kllm::DType::F32, device);
  kllm::op::rms_norm(x, w, out, 0.0f);
  const float scale = 1.0f / std::sqrt((9.0f + 16.0f) / 2.0f);
  expect_vec_close(out.to_float_vector(), {3.0f * scale, 4.0f * scale});
}

void test_silu(kllm::DeviceType device) {
  auto gate = make_tensor({1, 2}, {0, 1}, device);
  auto up = make_tensor({1, 2}, {2, 2}, device);
  kllm::Tensor out({1, 2}, kllm::DType::F32, device);
  kllm::op::silu_mul(gate, up, out);
  const float silu1 = 1.0f / (1.0f + std::exp(-1.0f));
  expect_vec_close(out.to_float_vector(), {0.0f, 2.0f * silu1});
}

void test_rope_position_zero(kllm::DeviceType device) {
  auto x = make_tensor({1, 1, 4}, {1, 2, 3, 4}, device);
  kllm::op::rope_inplace(x, 0, 10000.0f);
  expect_vec_close(x.to_float_vector(), {1, 2, 3, 4});
}

void test_kv_cache_and_attention(kllm::DeviceType device) {
  auto k = make_tensor({1, 1, 2}, {1, 0}, device);
  auto v = make_tensor({1, 1, 2}, {2, 3}, device);
  kllm::Tensor k_cache({2, 1, 2}, kllm::DType::F32, device);
  kllm::Tensor v_cache({2, 1, 2}, kllm::DType::F32, device);
  kllm::op::copy_kv_to_cache(k, v, k_cache, v_cache, 0);
  expect_vec_close(k_cache.to_float_vector(), {1, 0, 0, 0});
  expect_vec_close(v_cache.to_float_vector(), {2, 3, 0, 0});

  auto q = make_tensor({1, 1, 2}, {1, 0}, device);
  kllm::Tensor out({1, 1, 2}, kllm::DType::F32, device);
  kllm::op::attention(q, k_cache, v_cache, out, 0, 1, 1, 1, 2);
  expect_vec_close(out.to_float_vector(), {2, 3});
}

void test_sampler() {
  auto logits = make_tensor({1, 4}, {0, 1, 5, 2}, kllm::DeviceType::CPU);
  kllm::SamplingConfig cfg;
  cfg.greedy = true;
  kllm::Sampler sampler(cfg);
  const int token = sampler.sample(logits, {});
  if (token != 2) {
    fail("greedy sampler returned wrong token");
  }
}

void run_op_tests(kllm::DeviceType device) {
  test_embedding(device);
  test_matmul_and_add(device);
  test_rms_norm(device);
  test_silu(device);
  test_rope_position_zero(device);
  test_kv_cache_and_attention(device);
}

}  // namespace

int main() {
  try {
    test_half_roundtrip();
    run_op_tests(kllm::DeviceType::CPU);
    test_sampler();
    std::cout << "[OK] CPU operator tests passed\n";

    if (kllm::cuda_available()) {
      run_op_tests(kllm::DeviceType::CUDA);
      std::cout << "[OK] CUDA operator tests passed\n";
    } else {
      std::cout << "[SKIP] CUDA is not available in this build/runtime\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}

