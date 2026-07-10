#!/usr/bin/env python3
import argparse
import json
import os
import struct
from glob import glob
from typing import Dict, Iterable, List, Tuple


DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_I32 = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Export a HF Llama/Qwen2 model to .kllm.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    parser.add_argument("--include-bin", action="store_true", help="Allow pytorch_model*.bin checkpoints.")
    return parser.parse_args()


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def model_type_id(model_type: str) -> int:
    if model_type == "llama":
        return 0
    if model_type == "qwen2":
        return 1
    raise ValueError(f"unsupported model_type={model_type!r}; supported: llama, qwen2")


def checkpoint_files(model_dir: str, include_bin: bool) -> List[str]:
    safe_index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(safe_index):
        with open(safe_index, "r", encoding="utf-8") as f:
            index = json.load(f)
        return [os.path.join(model_dir, name) for name in sorted(set(index["weight_map"].values()))]

    safetensors = sorted(glob(os.path.join(model_dir, "*.safetensors")))
    if safetensors:
        return safetensors

    if include_bin:
        bin_index = os.path.join(model_dir, "pytorch_model.bin.index.json")
        if os.path.exists(bin_index):
            with open(bin_index, "r", encoding="utf-8") as f:
                index = json.load(f)
            return [os.path.join(model_dir, name) for name in sorted(set(index["weight_map"].values()))]
        bins = sorted(glob(os.path.join(model_dir, "pytorch_model*.bin")))
        if bins:
            return bins

    raise FileNotFoundError("no safetensors checkpoint found; pass --include-bin to allow .bin files")


def iter_tensors(files: Iterable[str]):
    import torch
    from safetensors import safe_open

    for path in files:
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)
        else:
            shard = torch.load(path, map_location="cpu")
            for key, tensor in shard.items():
                yield key, tensor
            del shard


def convert_tensor(tensor, dtype: str) -> Tuple[int, Tuple[int, ...], bytes]:
    import torch

    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64):
        if dtype == "f16":
            tensor = tensor.to(torch.float16)
            return DTYPE_F16, tuple(tensor.shape), tensor.numpy().tobytes(order="C")
        tensor = tensor.to(torch.float32)
        return DTYPE_F32, tuple(tensor.shape), tensor.numpy().tobytes(order="C")

    if tensor.dtype in (torch.int32, torch.int64):
        tensor = tensor.to(torch.int32)
        return DTYPE_I32, tuple(tensor.shape), tensor.numpy().tobytes(order="C")

    raise TypeError(f"unsupported tensor dtype: {tensor.dtype}")


def write_header(out, cfg: dict, tensor_count: int):
    hidden = int(cfg["hidden_size"])
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))
    head_dim = int(cfg.get("head_dim", hidden // num_heads))
    model_type = cfg["model_type"]

    qkv_bias = bool(cfg.get("attention_bias", model_type == "qwen2"))
    o_bias = bool(cfg.get("attention_output_bias", False))
    mlp_bias = bool(cfg.get("mlp_bias", False))

    out.write(b"KLLM0001")
    out.write(struct.pack("<I", 1))
    out.write(struct.pack("<I", model_type_id(model_type)))
    out.write(struct.pack("<i", int(cfg["vocab_size"])))
    out.write(struct.pack("<i", hidden))
    out.write(struct.pack("<i", int(cfg["intermediate_size"])))
    out.write(struct.pack("<i", int(cfg["num_hidden_layers"])))
    out.write(struct.pack("<i", num_heads))
    out.write(struct.pack("<i", num_kv_heads))
    out.write(struct.pack("<i", head_dim))
    out.write(struct.pack("<i", int(cfg.get("max_position_embeddings", 2048))))
    out.write(struct.pack("<f", float(cfg.get("rms_norm_eps", 1e-6))))
    out.write(struct.pack("<f", float(cfg.get("rope_theta", 10000.0))))
    out.write(struct.pack("<B", int(bool(cfg.get("tie_word_embeddings", False)))))
    out.write(struct.pack("<B", int(qkv_bias)))
    out.write(struct.pack("<B", int(o_bias)))
    out.write(struct.pack("<B", int(mlp_bias)))
    out.write(struct.pack("<I", tensor_count))


def write_tensor(out, name: str, dtype_id: int, shape: Tuple[int, ...], data: bytes):
    encoded = name.encode("utf-8")
    out.write(struct.pack("<I", len(encoded)))
    out.write(encoded)
    out.write(struct.pack("<I", dtype_id))
    out.write(struct.pack("<I", len(shape)))
    for dim in shape:
        out.write(struct.pack("<q", int(dim)))
    out.write(struct.pack("<Q", len(data)))
    out.write(data)


def main():
    args = parse_args()
    cfg = load_config(args.model_dir)
    files = checkpoint_files(args.model_dir, args.include_bin)

    converted: Dict[str, Tuple[int, Tuple[int, ...], bytes]] = {}
    for name, tensor in iter_tensors(files):
        if name.endswith(".inv_freq"):
            continue
        converted[name] = convert_tensor(tensor, args.dtype)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as out:
        write_header(out, cfg, len(converted))
        for name in sorted(converted.keys()):
            write_tensor(out, name, *converted[name])

    size_gb = os.path.getsize(args.out) / 1024**3
    print(f"wrote {args.out} ({len(converted)} tensors, {size_gb:.2f} GiB)")


if __name__ == "__main__":
    main()

