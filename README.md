# KLLM

一个轻量级大模型推理框架。

KLLM 不使用 `llama.cpp`，也不调用 Transformers 的模型前向过程。Python 仅用于模型下载、权重导出和复用 Hugging Face tokenizer，实际推理过程完全在 C++ / CUDA / CMake 项目中完成。

当前版本定位为**可读、可扩展的教学与实验框架**，而不是高性能生产级推理引擎。项目已经支持 CPU / CUDA 双后端、FP16 权重存储、KV cache、自定义 Tensor 与算子系统，适合继续扩展 FlashAttention、量化 Linear、Paged KV Cache、HTTP Server 等模块。

---

## Features

* C++17 推理核心
* CMake 构建系统
* 兼容 CUDA 11.6 + GCC 8.x
* CPU / CUDA 双后端
* 自定义 Tensor、算子、KV cache、采样器
* 支持 Hugging Face Llama2 和 Qwen2 / Qwen2.5 系列 decoder-only 权重导出
* 支持 FP16 权重存储
* CPU 激活使用 FP32，便于调试
* CUDA 激活和 KV cache 使用 FP16
* CUDA matmul 已接入 cuBLAS，可使用 FP16 activation 走 Tensor Core
* attention、采样和多数逐元素算子仍为朴素实现，便于阅读和替换

---

## Project Structure

```text
include/kllm/base       # Device、Tensor、FP16 转换
include/kllm/op         # 算子接口
include/kllm/op/detail  # CPU / CUDA 算子内部 helper
include/kllm/model      # config、weights、Llama / Qwen2 runner
include/kllm/sampler    # top-k / top-p / greedy sampler

src/base                # CPU / CUDA 内存与 Tensor
src/op/dispatch.cpp     # 公共 op 接口到 CPU / CUDA 后端的分发
src/op/cpu              # CPU 算子，每个算子单独文件
src/op/cuda             # CUDA 算子，每个算子单独 .cu 文件
src/model               # .kllm 权重加载和 decoder forward
src/sampler             # 采样实现

demo/kllm_demo.cpp      # C++ token-id 推理 demo
tests/test_ops.cpp      # CPU / CUDA 算子单元测试

tools/export_hf.py      # HF 权重导出为 .kllm
tools/chat_tokenizer.py # tokenizer + C++ demo 文本推理包装
```

---

## Requirements

### System

Ubuntu / WSL2:

```bash
sudo apt update
sudo apt install -y build-essential cmake python3-venv python3-dev
```

### Python

创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

安装依赖：

```bash
pip install -U pip
pip install -r requirements.txt
```

如果需要使用 CUDA 11.6 版本的 PyTorch 下载或处理模型，可以单独安装：

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu116 torch==1.13.1+cu116
pip install -r requirements.txt
```

---

## Build

### CPU Build

```bash
bash scripts/build_cpu.sh
```

### CUDA Build

```bash
bash scripts/build_cuda.sh
```

也可以直接使用 CMake 构建 CUDA 版本：

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DKLLM_USE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="60;61;70;75;80;86"

cmake --build build-cuda -j"$(nproc)"
```

---

## Test

运行 CPU 单元测试：

```bash
bash scripts/test_cpu.sh
```

运行 CUDA 单元测试：

```bash
bash scripts/test_cuda.sh
```

也可以直接执行构建产物：

```bash
./build-cpu/kllm_tests
./build-cuda/kllm_tests
```

`kllm_tests` 会始终运行 CPU 算子测试。如果构建启用了 CUDA，且运行时能够检测到 GPU，则会额外运行 CUDA 算子测试。

---

## Download and Export Models

建议先使用 `Qwen2.5-0.5B-Instruct` 验证整体流程。

### Download Qwen2.5-0.5B-Instruct

```bash
python tools/download_model.py \
  --repo-id Qwen/Qwen2.5-0.5B-Instruct \
  --out-dir models/qwen2.5-0.5b-instruct
```

### Export to `.kllm`

```bash
python tools/export_hf.py \
  --model-dir models/qwen2.5-0.5b-instruct \
  --out models/qwen2.5-0.5b-instruct.kllm \
  --dtype f16
```

### Download Llama2-7B

Llama2-7B 需要 Hugging Face 授权。

```bash
export HF_TOKEN=你的token

python tools/download_model.py \
  --repo-id meta-llama/Llama-2-7b-chat-hf \
  --out-dir models/llama2-7b-chat \
  --token "$HF_TOKEN"
```

导出为 `.kllm`：

```bash
python tools/export_hf.py \
  --model-dir models/llama2-7b-chat \
  --out models/llama2-7b-chat.kllm \
  --dtype f16
```

---

## Run

### C++ Demo with Token IDs

`kllm_demo` 直接接收 token ids 作为输入：

```bash
./build-cpu/kllm_demo \
  --model models/qwen2.5-0.5b-instruct.kllm \
  --device cpu \
  --tokens 151644,872,198,108386,151645 \
  --max-new-tokens 16
```

---

### Text Inference with Tokenizer Wrapper

由于当前框架不内置完整 BPE / SentencePiece tokenizer，实际文本输入建议通过 `tools/chat_tokenizer.py` 包装脚本完成。

CUDA 推理：

```bash
python tools/chat_tokenizer.py \
  --model-dir models/qwen2.5-0.5b-instruct \
  --kllm-model models/qwen2.5-0.5b-instruct.kllm \
  --binary build-cuda/kllm_demo \
  --device cuda \
  --chat \
  --prompt "用三句话解释什么是大模型 KV cache。" \
  --max-new-tokens 128
```

CPU 推理：

```bash
python tools/chat_tokenizer.py \
  --model-dir models/qwen2.5-0.5b-instruct \
  --kllm-model models/qwen2.5-0.5b-instruct.kllm \
  --binary build-cpu/kllm_demo \
  --device cpu \
  --chat \
  --prompt "介绍一下本项目的模块划分。"
```

---

## `.kllm` File Format

`.kllm` 是本项目定义的简单二进制权重格式。

整体结构如下：

```text
magic: "KLLM0001"
version: uint32

model config:
  model_type
  vocab
  hidden
  layer/head 参数
  rope/rms 参数

tensor_count: uint32

repeat tensor_count:
  name_len: uint32
  name: utf-8 bytes
  dtype: uint32      # 0=f32, 1=f16, 2=i32
  ndim: uint32
  dims: int64[ndim]
  nbytes: uint64
  raw tensor bytes
```

权重名沿用 Hugging Face 模型中的命名，例如：

```text
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.q_proj.bias
model.layers.0.mlp.gate_proj.weight
model.norm.weight
lm_head.weight
```

---

## Supported Models

当前支持常见 Hugging Face 权重结构的 decoder-only 模型：

* Llama / Llama2
* Qwen2
* Qwen2.5

---

## Current Limitations

当前版本仍有以下限制：

* 只支持 Llama / Llama2 和 Qwen2 / Qwen2.5 的常见 Hugging Face 权重结构
* 不支持 GGUF
* 不支持 GPTQ
* 不支持 AWQ
* 不支持 bitsandbytes
* 不内置完整 BPE / SentencePiece tokenizer
* 文本输入由 `tools/chat_tokenizer.py` 负责
* CUDA matmul 已使用 cuBLAS，但 attention、top-k / top-p 采样和部分逐元素 kernel 仍是朴素实现
* CUDA 激活和 KV cache 使用 FP16
* CPU 路径仍使用 FP32，便于调试
* 暂不支持 batch 推理
* 暂不支持 streaming 输出
* 暂不支持 paged KV cache
* 暂不支持量化权重

---

## Roadmap

下一步可以继续完善以下模块：

* 使用 Qwen2.5-0.5B 跑通 CPU / CUDA 正确性验证
* 将 attention decode kernel 改成 block 级并行实现
* 引入 FlashAttention 风格 attention kernel
* 增加 weight-only int8 / int4 Linear
* 增加完整 C++ tokenizer
* 或接入 tokenizers C API
* 增加 streaming token 输出
* 增加 OpenAI 风格 HTTP server
* 增加 batch 推理
* 增加 Paged KV Cache
* 增加更多算子单元测试与端到端测试

---

## Design Goal

KLLM 的目标不是直接追求极致性能，而是提供一个结构清晰、模块边界明确、适合学习和二次开发的大模型推理框架。

它适合用于：

* 学习 LLM 推理框架的基本组成
* 理解 Tensor、算子、权重加载、KV cache、采样器之间的关系
* 实验 CPU / CUDA 双后端设计
* 验证自定义 CUDA kernel
* 替换 attention、linear、sampling 等关键模块
* 作为后续高性能推理框架的原型项目

---

## License

TODO
