#!/usr/bin/env python3


import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpt2_tokenizer import (
    bytes_to_unicode, load_encoder, decode_tokens, build_bpe_from_dir as build_bpe,
)

def load_tokens(path: Path):
    tokens = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tokens.append(int(line))
        except ValueError:
            raise ValueError(f"Invalid token id: {line}")
    return tokens

def main():
    parser = argparse.ArgumentParser(description="Decode GPT-2 token IDs to text.")
    parser.add_argument("--token-file", help="Token file (one ID per line)")
    parser.add_argument("--bpe-dir", default="./model/gpt2-124m",
                        help="Dir with vocab.json or encoder.json")
    parser.add_argument("--model-dir", help="Model directory for generation")
    parser.add_argument("--seq-len", type=int, default=0, help="Sequence length for generation")
    parser.add_argument("--prompt-text", help="Prompt text for generation")
    parser.add_argument("--prompt-file", help="Prompt text file for generation")
    parser.add_argument("--prompt-tokens", help="Prompt token file (one ID per line)")
    parser.add_argument("--gen-bin-quantized", default="./cmake-build-release/src/generate_tokens",
                        help="Path to quantized generator")
    parser.add_argument("--output-tokens", help="Where to write generated tokens")
    args = parser.parse_args()

    bpe_dir = Path(args.bpe_dir)
    if not bpe_dir.exists() and args.bpe_dir == "./model/gpt2-124m":
        fallback = Path("./model/gpt2/gpt2-124m")
        if fallback.exists():
            bpe_dir = fallback
    try:
        decoder = load_encoder(bpe_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    prompt_text = args.prompt_text
    if not prompt_text and args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text()

    prompt_tokens_path = None
    if prompt_text:
        try:
            encode = build_bpe(bpe_dir)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        tokens = encode(prompt_text)
        if not tokens:
            print("Prompt encoding produced no tokens", file=sys.stderr)
            return 1
        prompt_tokens_path = Path("./cache/prompt_tokens_decode.txt")
        prompt_tokens_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_tokens_path.write_text("\n".join(str(t) for t in tokens) + "\n")
    elif args.prompt_tokens:
        prompt_tokens_path = Path(args.prompt_tokens)

    if prompt_tokens_path and args.model_dir and args.seq_len > 0:
        gen_bin = args.gen_bin_quantized
        gen_bin_path = Path(gen_bin)
        if not gen_bin_path.exists():
            print(f"Generator not found: {gen_bin_path}", file=sys.stderr)
            return 1
        output_tokens = args.output_tokens or "./benchmark_tokens/generated_tokens.txt"
        cmd = [
            str(gen_bin_path),
            "--model-dir", args.model_dir,
            "--seq-len", str(args.seq_len),
            "--prompt-tokens", str(prompt_tokens_path),
            "--output", output_tokens,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        token_path = Path(output_tokens)
    else:
        if not args.token_file:
            print("Missing --token-file (or provide prompt + model-dir + seq-len)", file=sys.stderr)
            return 1
        token_path = Path(args.token_file)
        if not token_path.exists():
            print(f"Token file not found: {token_path}", file=sys.stderr)
            return 1

    try:
        tokens = load_tokens(token_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not tokens:
        print("No tokens to decode", file=sys.stderr)
        return 1

    print(decode_tokens(tokens, decoder))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())