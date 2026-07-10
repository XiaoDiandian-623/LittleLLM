#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#if KLLM_USE_CUDA
#include <cuda_runtime.h>
#endif

#include "kllm/base/device.h"
#include "kllm/model/llama.h"
#include "kllm/model/weights.h"
#include "kllm/sampler/sampler.h"

namespace {

struct Args {
  std::string model_path;
  std::string device = "cpu";
  std::string token_csv;
  std::string eos_csv;
  int max_new_tokens = 0;
  int max_seq_len = 0;
  int top_k = 40;
  float top_p = 0.9f;
  float temperature = 0.8f;
  float repetition_penalty = 1.1f;
  bool greedy = false;
  std::uint32_t seed = 1234;
};

void usage() {
  std::cerr
      << "Usage: kllm_demo --model model.kllm --tokens 1,2,3 [options]\n"
      << "Options:\n"
      << "  --device cpu|cuda\n"
      << "  --max-new-tokens N    0 means generate until EOS or KV cache limit\n"
      << "  --max-seq-len N\n"
      << "  --temperature F\n"
      << "  --top-k N\n"
      << "  --top-p F\n"
      << "  --repetition-penalty F\n"
      << "  --greedy\n"
      << "  --seed N\n"
      << "  --eos 2,151645\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(name + " requires a value");
      }
      return argv[++i];
    };

    if (key == "--model") {
      args.model_path = need_value(key);
    } else if (key == "--device") {
      args.device = need_value(key);
    } else if (key == "--tokens") {
      args.token_csv = need_value(key);
    } else if (key == "--eos") {
      args.eos_csv = need_value(key);
    } else if (key == "--max-new-tokens") {
      args.max_new_tokens = std::stoi(need_value(key));
    } else if (key == "--max-seq-len") {
      args.max_seq_len = std::stoi(need_value(key));
    } else if (key == "--temperature") {
      args.temperature = std::stof(need_value(key));
    } else if (key == "--top-k") {
      args.top_k = std::stoi(need_value(key));
    } else if (key == "--top-p") {
      args.top_p = std::stof(need_value(key));
    } else if (key == "--repetition-penalty") {
      args.repetition_penalty = std::stof(need_value(key));
    } else if (key == "--seed") {
      args.seed = static_cast<std::uint32_t>(std::stoul(need_value(key)));
    } else if (key == "--greedy") {
      args.greedy = true;
    } else if (key == "--help" || key == "-h") {
      usage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (args.model_path.empty()) {
    throw std::runtime_error("--model is required");
  }
  if (args.token_csv.empty()) {
    throw std::runtime_error("--tokens is required");
  }
  return args;
}

std::vector<int> parse_csv_ints(const std::string& csv) {
  std::vector<int> result;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      continue;
    }
    result.push_back(std::stoi(item));
  }
  return result;
}

void print_ids(const std::string& label, const std::vector<int>& ids) {
  std::cout << label << ":";
  for (std::size_t i = 0; i < ids.size(); ++i) {
    std::cout << (i == 0 ? " " : ",") << ids[i];
  }
  std::cout << "\n";
}

void synchronize_device(kllm::DeviceType device) {
#if KLLM_USE_CUDA
  if (device == kllm::DeviceType::CUDA) {
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
  }
#else
  (void)device;
#endif
}

double elapsed_ms(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_perf(
    int prompt_tokens,
    int generated_tokens,
    double prefill_ms,
    double decode_ms,
    const std::string& finish_reason) {
  const double prefill_tps =
      prefill_ms > 0.0 ? static_cast<double>(prompt_tokens) / (prefill_ms / 1000.0) : 0.0;
  const double decode_tps =
      decode_ms > 0.0 ? static_cast<double>(generated_tokens) / (decode_ms / 1000.0) : 0.0;

  std::cerr << "perf.prefill_tokens: " << prompt_tokens << "\n";
  std::cerr << "perf.prefill_ms: " << prefill_ms << "\n";
  std::cerr << "perf.prefill_tokens_per_s: " << prefill_tps << "\n";
  std::cerr << "perf.decode_tokens: " << generated_tokens << "\n";
  std::cerr << "perf.decode_ms: " << decode_ms << "\n";
  std::cerr << "perf.decode_tokens_per_s: " << decode_tps << "\n";
  std::cerr << "perf.finish_reason: " << finish_reason << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const auto device = kllm::parse_device(args.device);
    if (device == kllm::DeviceType::CUDA && !kllm::cuda_available()) {
      throw std::runtime_error("CUDA requested but not available or not compiled");
    }

    auto weights = kllm::ModelWeights::load(args.model_path, device);
    kllm::LlamaRunner runner(std::move(weights), device, args.max_seq_len);

    kllm::SamplingConfig sampling;
    sampling.top_k = args.top_k;
    sampling.top_p = args.top_p;
    sampling.temperature = args.temperature;
    sampling.repetition_penalty = args.repetition_penalty;
    sampling.greedy = args.greedy;
    sampling.seed = args.seed;
    kllm::Sampler sampler(sampling);

    std::vector<int> history = parse_csv_ints(args.token_csv);
    const std::vector<int> eos_ids = parse_csv_ints(args.eos_csv);
    const std::unordered_set<int> eos(eos_ids.begin(), eos_ids.end());
    std::vector<int> new_tokens;

    const int prompt_tokens = static_cast<int>(history.size());
    const int effective_max_seq_len =
        args.max_seq_len > 0 ? args.max_seq_len : runner.config().max_seq_len;
    const int remaining_cache_tokens = effective_max_seq_len - prompt_tokens;
    if (remaining_cache_tokens <= 0) {
      throw std::runtime_error("prompt already reaches or exceeds max sequence length");
    }
    const int generation_limit =
        args.max_new_tokens > 0 ? args.max_new_tokens : remaining_cache_tokens;

    const auto prefill_start = std::chrono::steady_clock::now();
    kllm::Tensor logits = runner.forward(history, 0);
    synchronize_device(device);
    const auto prefill_end = std::chrono::steady_clock::now();

    int past_len = static_cast<int>(history.size());
    std::string finish_reason = "length";
    const auto decode_start = std::chrono::steady_clock::now();
    for (int step = 0; step < generation_limit; ++step) {
      const int next = sampler.sample(logits, history);
      if (eos.find(next) != eos.end()) {
        finish_reason = "eos";
        break;
      }
      new_tokens.push_back(next);
      history.push_back(next);
      logits = runner.forward(std::vector<int>{next}, past_len);
      synchronize_device(device);
      ++past_len;
    }
    if (finish_reason != "eos" && static_cast<int>(new_tokens.size()) >= remaining_cache_tokens) {
      finish_reason = "cache_limit";
    }
    const auto decode_end = std::chrono::steady_clock::now();

    print_ids("new_token_ids", new_tokens);
    print_ids("all_token_ids", history);
    print_perf(
        prompt_tokens,
        static_cast<int>(new_tokens.size()),
        elapsed_ms(prefill_start, prefill_end),
        elapsed_ms(decode_start, decode_end),
        finish_reason);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    usage();
    return 1;
  }
}
