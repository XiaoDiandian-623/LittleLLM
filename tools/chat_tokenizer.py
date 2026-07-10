#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description="Use HF tokenizer around the C++ kllm_demo.")
    parser.add_argument("--model-dir", required=True, help="HF model dir for tokenizer/config.")
    parser.add_argument("--kllm-model", required=True, help="Exported .kllm model file.")
    parser.add_argument("--binary", default=None, help="Path to kllm_demo. Default: build-cuda/build-cpu by --device.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--chat", action="store_true", help="Use tokenizer.apply_chat_template.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="0 means generate until EOS or the model/KV-cache limit.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def resolve_binary(binary: str, device: str) -> str:
    if binary:
        if not os.path.exists(binary):
            raise FileNotFoundError(
                f"kllm_demo binary not found: {binary}. "
                "Use --binary build-cuda/kllm_demo for CUDA or --binary build-cpu/kllm_demo for CPU."
            )
        return binary

    candidates = []
    if device == "cuda":
        candidates.extend(["build-cuda/kllm_demo", "build/kllm_demo"])
    else:
        candidates.extend(["build-cpu/kllm_demo", "build/kllm_demo"])

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "kllm_demo binary was not found. Build first with `bash scripts/build_cuda.sh` "
        "or pass --binary explicitly."
    )


def load_eos_ids(model_dir: str, tokenizer) -> List[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    path = os.path.join(model_dir, "config.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        eos = cfg.get("eos_token_id")
        if isinstance(eos, list):
            ids.extend(int(x) for x in eos)
        elif eos is not None:
            ids.append(int(eos))
    return sorted(set(ids))


def parse_output_ids(stdout: str, key: str) -> List[int]:
    prefix = key + ":"
    for line in stdout.splitlines():
        if line.startswith(prefix):
            payload = line[len(prefix):].strip()
            if not payload:
                return []
            return [int(x) for x in payload.split(",") if x]
    raise RuntimeError(f"{key} not found in kllm_demo output:\n{stdout}")


def main():
    from transformers import AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, use_fast=True)

    if args.chat:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)

    eos_ids = load_eos_ids(args.model_dir, tokenizer)
    binary = resolve_binary(args.binary, args.device)
    cmd = [
        binary,
        "--model", args.kllm_model,
        "--tokens", ",".join(str(x) for x in token_ids),
        "--device", args.device,
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--top-k", str(args.top_k),
        "--top-p", str(args.top_p),
        "--repetition-penalty", str(args.repetition_penalty),
        "--eos", ",".join(str(x) for x in eos_ids),
    ]
    if args.greedy:
        cmd.append("--greedy")

    result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.stderr:
        sys.stderr.write(result.stderr)
    new_ids = parse_output_ids(result.stdout, "new_token_ids")
    print(tokenizer.decode(new_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
