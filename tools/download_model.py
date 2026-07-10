#!/usr/bin/env python3
import argparse


def main():
    parser = argparse.ArgumentParser(description="Download a HF model snapshot.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--include-bin", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer*",
        "*.model",
        "*.json",
        "*.safetensors",
        "model.safetensors.index.json",
    ]
    if args.include_bin:
        patterns.extend(["pytorch_model*.bin", "pytorch_model.bin.index.json"])

    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        token=args.token,
        local_dir=args.out_dir,
        allow_patterns=patterns,
    )
    print(path)


if __name__ == "__main__":
    main()

