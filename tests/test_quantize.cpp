#include <iostream>
#include <cmath>
#include <vector>

#include "kllm/base/device.h"
#include "kllm/base/tensor.h"
#include "kllm/op/ops.h"

using namespace kllm;

bool close(float a, float b, float atol = 1e-3f) {
  return std::abs(a - b) <= atol;
}

void test_quantize_dequantize_cpu() {
  std::cout << "Testing int8 quantize/dequantize on CPU...\n";

  Tensor input({4, 4}, DType::F32, DeviceType::CPU);
  std::vector<float> input_data = {
    1.0f, 2.0f, 3.0f, 4.0f,
    5.0f, 6.0f, 7.0f, 8.0f,
    -1.0f, -2.0f, -3.0f, -4.0f,
    -5.0f, -6.0f, -7.0f, -8.0f
  };
  input.copy_from_host(input_data.data(), input_data.size() * sizeof(float));

  Tensor quant_data({4, 4}, DType::I8, DeviceType::CPU);
  Tensor scale({1}, DType::F32, DeviceType::CPU);

  op::quantize_int8(input, quant_data, scale);

  Tensor output({4, 4}, DType::F32, DeviceType::CPU);
  op::dequantize_int8(quant_data, scale, output);

  std::vector<float> output_data = output.to_float_vector();

  bool passed = true;
  for (size_t i = 0; i < input_data.size(); ++i) {
    if (!close(input_data[i], output_data[i], 0.1f)) {
      std::cout << "Mismatch at " << i << ": input=" << input_data[i]
                << " output=" << output_data[i] << "\n";
      passed = false;
    }
  }

  if (passed) {
    std::cout << "CPU quantize/dequantize test PASSED\n";
  } else {
    std::cout << "CPU quantize/dequantize test FAILED\n";
  }
}

void test_matmul_int8_cpu() {
  std::cout << "Testing int8 matmul on CPU...\n";

  Tensor a({2, 3}, DType::F32, DeviceType::CPU);
  std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  a.copy_from_host(a_data.data(), a_data.size() * sizeof(float));

  Tensor b({3, 2}, DType::F32, DeviceType::CPU);
  std::vector<float> b_data = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  b.copy_from_host(b_data.data(), b_data.size() * sizeof(float));

  Tensor b_quant({3, 2}, DType::I8, DeviceType::CPU);
  Tensor b_scale_temp({1}, DType::F32, DeviceType::CPU);
  op::quantize_int8(b, b_quant, b_scale_temp);

  Tensor b_scale({2}, DType::F32, DeviceType::CPU);
  float scale_val = b_scale_temp.to_float_vector()[0];
  std::vector<float> scale_data = {scale_val, scale_val};
  b_scale.copy_from_host(scale_data.data(), scale_data.size() * sizeof(float));

  Tensor out_int8({2, 2}, DType::F32, DeviceType::CPU);
  op::matmul_int8(a, b_quant, b_scale, out_int8);

  Tensor out_fp32({2, 2}, DType::F32, DeviceType::CPU);
  op::matmul(a, b, out_fp32);

  std::vector<float> result_int8 = out_int8.to_float_vector();
  std::vector<float> result_fp32 = out_fp32.to_float_vector();

  bool passed = true;
  for (size_t i = 0; i < result_fp32.size(); ++i) {
    float rel_error = std::abs(result_int8[i] - result_fp32[i]) / (std::abs(result_fp32[i]) + 1e-5f);
    if (rel_error > 0.05f) {
      std::cout << "Mismatch at " << i << ": int8=" << result_int8[i]
                << " fp32=" << result_fp32[i] << " rel_error=" << rel_error << "\n";
      passed = false;
    }
  }

  if (passed) {
    std::cout << "CPU int8 matmul test PASSED\n";
  } else {
    std::cout << "CPU int8 matmul test FAILED\n";
  }
}

#if KLLM_USE_CUDA
void test_quantize_dequantize_cuda() {
  std::cout << "Testing int8 quantize/dequantize on CUDA...\n";

  Tensor input({4, 4}, DType::F32, DeviceType::CUDA);
  std::vector<float> input_data = {
    1.0f, 2.0f, 3.0f, 4.0f,
    5.0f, 6.0f, 7.0f, 8.0f,
    -1.0f, -2.0f, -3.0f, -4.0f,
    -5.0f, -6.0f, -7.0f, -8.0f
  };
  input.copy_from_host(input_data.data(), input_data.size() * sizeof(float));

  Tensor quant_data({4, 4}, DType::I8, DeviceType::CUDA);
  Tensor scale({1}, DType::F32, DeviceType::CUDA);

  op::quantize_int8(input, quant_data, scale);

  Tensor output({4, 4}, DType::F32, DeviceType::CUDA);
  op::dequantize_int8(quant_data, scale, output);

  std::vector<float> output_data = output.to_float_vector();

  bool passed = true;
  for (size_t i = 0; i < input_data.size(); ++i) {
    if (!close(input_data[i], output_data[i], 0.1f)) {
      std::cout << "Mismatch at " << i << ": input=" << input_data[i]
                << " output=" << output_data[i] << "\n";
      passed = false;
    }
  }

  if (passed) {
    std::cout << "CUDA quantize/dequantize test PASSED\n";
  } else {
    std::cout << "CUDA quantize/dequantize test FAILED\n";
  }
}

void test_matmul_int8_cuda() {
  std::cout << "Testing int8 matmul on CUDA...\n";

  Tensor a({2, 3}, DType::F32, DeviceType::CUDA);
  std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  a.copy_from_host(a_data.data(), a_data.size() * sizeof(float));

  Tensor b({3, 2}, DType::F32, DeviceType::CUDA);
  std::vector<float> b_data = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  b.copy_from_host(b_data.data(), b_data.size() * sizeof(float));

  Tensor b_quant({3, 2}, DType::I8, DeviceType::CUDA);
  Tensor b_scale_temp({1}, DType::F32, DeviceType::CUDA);
  op::quantize_int8(b, b_quant, b_scale_temp);

  Tensor b_scale({2}, DType::F32, DeviceType::CUDA);
  std::vector<float> scale_temp = b_scale_temp.to_float_vector();
  float scale_val = scale_temp[0];
  std::vector<float> scale_data = {scale_val, scale_val};
  b_scale.copy_from_host(scale_data.data(), scale_data.size() * sizeof(float));

  Tensor out_int8({2, 2}, DType::F32, DeviceType::CUDA);
  op::matmul_int8(a, b_quant, b_scale, out_int8);

  Tensor out_fp32({2, 2}, DType::F32, DeviceType::CUDA);
  op::matmul(a, b, out_fp32);

  std::vector<float> result_int8 = out_int8.to_float_vector();
  std::vector<float> result_fp32 = out_fp32.to_float_vector();

  bool passed = true;
  for (size_t i = 0; i < result_fp32.size(); ++i) {
    float rel_error = std::abs(result_int8[i] - result_fp32[i]) / (std::abs(result_fp32[i]) + 1e-5f);
    if (rel_error > 0.05f) {
      std::cout << "Mismatch at " << i << ": int8=" << result_int8[i]
                << " fp32=" << result_fp32[i] << " rel_error=" << rel_error << "\n";
      passed = false;
    }
  }

  if (passed) {
    std::cout << "CUDA int8 matmul test PASSED\n";
  } else {
    std::cout << "CUDA int8 matmul test FAILED\n";
  }
}
#endif

int main() {
  std::cout << "Running int8 quantization tests...\n\n";

  test_quantize_dequantize_cpu();
  test_matmul_int8_cpu();

#if KLLM_USE_CUDA
  if (cuda_available()) {
    std::cout << "\n";
    test_quantize_dequantize_cuda();
    test_matmul_int8_cuda();
  } else {
    std::cout << "\nCUDA not available, skipping CUDA tests\n";
  }
#endif

  std::cout << "\nAll tests completed.\n";
  return 0;
}
