轻量级大模型推理框架
它不使用 llama.cpp，也不调用 Transformers 的模型前向；Python 只用于下载模型、导出权重和复用 Hugging Face tokenizer。实际推理在 C++/CUDA/CMake 项目中完成。
目标:
C++17 推理核心，兼容 CUDA 11.6 + GCC 8.x。
CMake 构建。
CPU / CUDA 双后端。
自定义 Tensor、算子、KV cache、采样器。
支持 Hugging Face Llama2 和 Qwen2/Qwen2.5 系列 decoder-only 权重导出。
支持 FP16 权重存储；CPU 激活使用 FP32，CUDA 激活和 KV cache 使用 FP16。
当前是可读、可扩展的框架版本，不是高性能版本。CUDA matmul 已接入 cuBLAS，并使用 FP16 activation 以便走 Tensor Core；attention、采样和多数逐元素算子仍是朴素实现，适合继续替换为 FlashAttention、量化 Linear、Paged KV Cache 等模块。
目录结构
include/kllm/base      # Device、Tensor、FP16 转换
include/kllm/op        # 算子接口
include/kllm/op/detail # CPU/CUDA 算子内部 helper
include/kllm/model     # config、weights、Llama/Qwen2 runner
include/kllm/sampler   # top-k/top-p/greedy sampler
src/base               # CPU/CUDA 内存与 Tensor
src/op/dispatch.cpp    # 公共 op 接口到 CPU/CUDA 后端的分发
src/op/cpu             # CPU 算子，每个算子单独文件
src/op/cuda            # CUDA 算子，每个算子单独 .cu 文件
src/model              # .kllm 权重加载和 decoder forward
src/sampler            # 采样实现
demo/kllm_demo.cpp     # C++ token-id 推理 demo
tests/test_ops.cpp     # CPU/CUDA 算子单元测试
tools/export_hf.py     # HF 权重导出为 .kllm
tools/chat_tokenizer.py# tokenizer + C++ demo 文本推理包装
依赖
Ubuntu/WSL2:
sudo apt update
sudo apt install -y build-essential cmake python3-venv python3-dev
Python 工具依赖：
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
你的 CUDA 是 11.6，如需 CUDA 版 PyTorch 下载工具，可单独安装：
pip install --extra-index-url https://download.pytorch.org/whl/cu116 torch==1.13.1+cu116
pip install -r requirements.txt
构建
CPU:
bash scripts/build_cpu.sh
CUDA:
bash scripts/build_cuda.sh
运行单元测试：
bash scripts/test_cpu.sh
bash scripts/test_cuda.sh
也可以直接运行构建产物：
./build-cpu/kllm_tests
./build-cuda/kllm_tests
kllm_tests 会始终运行 CPU 算子测试；如果构建启用了 CUDA 且运行时能看到 GPU，会额外运行 CUDA 算子测试。
等价 CMake 命令：
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DKLLM_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;61;70;75;80;86"
cmake --build build-cuda -j"$(nproc)"
下载和导出模型
建议先用 Qwen2.5-0.5B 验证：
python tools/download_model.py \
  --repo-id Qwen/Qwen2.5-0.5B-Instruct \
  --out-dir models/qwen2.5-0.5b-instruct
导出为本框架使用的 .kllm：
python tools/export_hf.py \
  --model-dir models/qwen2.5-0.5b-instruct \
  --out models/qwen2.5-0.5b-instruct.kllm \
  --dtype f16
Llama2-7B 需要 Hugging Face 授权：
export HF_TOKEN=你的token
python tools/download_model.py \
  --repo-id meta-llama/Llama-2-7b-chat-hf \
  --out-dir models/llama2-7b-chat \
  --token "$HF_TOKEN"

python tools/export_hf.py \
  --model-dir models/llama2-7b-chat \
  --out models/llama2-7b-chat.kllm \
  --dtype f16
运行
C++ demo 直接输入 token ids：
./build-cpu/kllm_demo \
  --model models/qwen2.5-0.5b-instruct.kllm \
  --device cpu \
  --tokens 151644,872,198,108386,151645 \
  --max-new-tokens 16
实际文本建议通过 tokenizer 包装脚本：
python tools/chat_tokenizer.py \
  --model-dir models/qwen2.5-0.5b-instruct \
  --kllm-model models/qwen2.5-0.5b-instruct.kllm \
  --binary build-cuda/kllm_demo \
  --device cuda \
  --chat \
  --prompt "用三句话解释什么是大模型 KV cache。" \
  --max-new-tokens 128
CPU 运行：
python tools/chat_tokenizer.py \
  --model-dir models/qwen2.5-0.5b-instruct \
  --kllm-model models/qwen2.5-0.5b-instruct.kllm \
  --binary build-cpu/kllm_demo \
  --device cpu \
  --chat \
  --prompt "介绍一下本项目的模块划分。"
文件格式
.kllm 是本项目定义的简单二进制格式：
magic: "KLLM0001"
version: uint32
model config: model_type, vocab, hidden, layer/head 参数, rope/rms 参数
tensor_count: uint32
repeat tensor_count:
  name_len: uint32
  name: utf-8 bytes
  dtype: uint32  # 0=f32, 1=f16, 2=i32
  ndim: uint32
  dims: int64[ndim]
  nbytes: uint64
  raw tensor bytes
权重名沿用 HF，例如：
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.q_proj.bias
model.layers.0.mlp.gate_proj.weight
model.norm.weight
lm_head.weight
当前限制
只支持 Llama/Llama2 和 Qwen2/Qwen2.5 的常见 HF 权重结构。
不支持 GGUF、GPTQ、AWQ、bitsandbytes。
不内置完整 BPE/SentencePiece tokenizer；文本输入由 tools/chat_tokenizer.py 负责。
CUDA matmul 已使用 cuBLAS；attention、top-k/top-p 采样和部分逐元素 kernel 仍是朴素实现。
CUDA 激活和 KV cache 已使用 FP16；CPU 路径仍使用 FP32，便于调试。
还没有 batch 推理、streaming 输出、paged KV cache、量化权重。
下一步建议
用 Qwen2.5-0.5B 跑通 CPU/CUDA 正确性。
将 attention decode kernel 改成 block 级并行或 FlashAttention 风格实现。
加 weight-only int8/int4 Linear。
加完整 C++ tokenizer 或接入 tokenizers C API。
加 streaming token 输出和 OpenAI 风格 HTTP server。