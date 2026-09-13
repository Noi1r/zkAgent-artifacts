#!/usr/bin/env python3


import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpt2_tokenizer import build_bpe


def write_tokens(path: Path, tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(t) for t in tokens) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate shift-check tokens from prompt text.")
    parser.add_argument("--model-dir", required=True, help="Model directory (e.g. ./model/gpt2-12layer)")
    parser.add_argument("--bpe-dir", default="./model/gpt2-124m",
                        help="GPT-2 BPE dir with vocab.json and merges.txt")
    parser.add_argument("--prompt-text", help="Prompt text (ASCII recommended)")
    parser.add_argument("--prompt-file", help="Prompt text file")
    parser.add_argument("--seq-len", type=int, default=12, help="Total sequence length")
    parser.add_argument(
        "--output",
        default=None,
        help="Output prefix path; writes <prefix>_token.txt and <prefix>_decode.txt",
    )
    parser.add_argument("--prompt-tokens-out", default="./cache/prompt_tokens.txt",
                        help="Where to save encoded prompt tokens")
    parser.add_argument("--gen-bin", default="./cmake-build-release/src/generate_tokens",
                        help="Path to generate_tokens binary")
    parser.add_argument("--single-token-prompt", action="store_true",
                        help="Use only the first prompt token")
    parser.add_argument("--prompt-len", type=int, default=0,
                        help="Trim prompt tokens to this length before generation (0 = no trim)")
    parser.add_argument("--threads", type=int, default=8,
                        help="Threads for generate_tokens (default: 8)")
    args = parser.parse_args()

    prompt_text = args.prompt_text
    if not prompt_text and args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text()
    if not prompt_text:
        prompt_text = "The quick brown fox jumps over the lazy dog."

    def resolve_bpe_paths(base: Path):
        vocab = base / "vocab.json"
        merges = base / "merges.txt"
        if vocab.exists() and merges.exists():
            return base, vocab, merges
        vocab = base / "encoder.json"
        merges = base / "vocab.bpe"
        if vocab.exists() and merges.exists():
            return base, vocab, merges
        return None

    bpe_dir = Path(args.bpe_dir)
    fallback_dir = None
    if args.bpe_dir == "./model/gpt2-124m":
        fallback_dir = Path("./model/gpt2/gpt2-124m")

    resolved = resolve_bpe_paths(bpe_dir)
    if resolved is None and fallback_dir is not None:
        resolved = resolve_bpe_paths(fallback_dir)
    if resolved is None:
        print(f"Missing BPE files in {bpe_dir}", file=sys.stderr)
        return 1

    bpe_dir, vocab_path, merges_path = resolved

    encode = build_bpe(vocab_path, merges_path)
    prompt_tokens = encode(prompt_text)
    if not prompt_tokens:
        print("Prompt encoding produced no tokens", file=sys.stderr)
        return 1

    if args.prompt_len and args.prompt_len > 0:
        prompt_tokens = prompt_tokens[:args.prompt_len]

    if args.single_token_prompt and len(prompt_tokens) > 1:
        print("Using only the first prompt token.", file=sys.stderr)
        prompt_tokens = prompt_tokens[:1]

    prompt_tokens_path = Path(args.prompt_tokens_out)
    write_tokens(prompt_tokens_path, prompt_tokens)

    if args.output:
        output_prefix = Path(args.output)
    else:
        model_name = Path(args.model_dir).name
        output_prefix = Path("data") / f"{model_name}_inference"
    output_path = str(output_prefix) + "_token.txt"
    decode_path = str(output_prefix) + "_response.txt"
    prompt_path = str(output_prefix) + "_prompt.txt"
    input_tokens_path = str(output_prefix) + "_input.txt"
    Path(prompt_path).write_text(prompt_text + "\n")
    Path(input_tokens_path).write_text("\n".join(str(t) for t in prompt_tokens) + "\n")

    gen_bin = Path(args.gen_bin)
    if not gen_bin.exists():
        print(f"generate_tokens binary not found: {gen_bin}", file=sys.stderr)
        return 1

    cmd = [
        str(gen_bin),
        "--model-dir", args.model_dir,
        "--seq-len", str(args.seq_len),
        "--prompt-tokens", str(prompt_tokens_path),
        "--output", output_path,
        "--threads", str(args.threads),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stdout:
        filtered = []
        for line in result.stdout.splitlines():
            if line.startswith("Shift check (suffix only):"):
                continue
            if line.startswith("Max exp table index:"):
                continue
            if line.startswith("Max range-check input (abs):"):
                continue
            if line.startswith("Wrote ") and " tokens to " in line:
                continue
            filtered.append(line)
        if filtered:
            print("\n".join(filtered))
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        print(f"generate_tokens failed with exit code {result.returncode}", file=sys.stderr)
        return result.returncode

    prompt_len = len(prompt_tokens)
    if decode_path is not None:
        decode_script = Path("./scripts/decode_gpt2_tokens.py")
        if not decode_script.exists():
            print("decode_gpt2_tokens.py not found; skipping decode.", file=sys.stderr)
        else:
            decode_cmd = [
                str(decode_script),
                "--token-file", output_path,
                "--bpe-dir", str(bpe_dir),
            ]
            try:
                decoded = subprocess.run(decode_cmd, check=True, text=True, capture_output=True)
                decoded_text = decoded.stdout.strip()
                Path(decode_path).write_text(decoded_text + "\n")
            except subprocess.CalledProcessError as exc:
                print(exc.stdout, end="")
                print(exc.stderr, end="", file=sys.stderr)
                print("Decode failed.", file=sys.stderr)
    else:
        decoded_text = ""

    response_len = 0
    try:
        response_len = len(Path(output_path).read_text().splitlines())
    except Exception:
        response_len = 0

    print()
    print(f"Prompt file: {prompt_path}")
    print(f"Input tokens file: {input_tokens_path}")
    print(f"Response file: {decode_path}")
    print(f"Output tokens file: {output_path}")
    print()
    print(f"Prompt text ({prompt_len} tokens):")
    print(f">>> {prompt_text}")
    print(f"Response text ({response_len} tokens):")
    print(f">>> {decoded_text}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())