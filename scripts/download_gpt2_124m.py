#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np

def _require_deps():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise SystemExit(f"Missing dependency: {exc}\nInstall: pip install torch transformers")

def export_gpt2_124m(output_dir: Path, quant_bits: int = 16) -> None:
    _require_deps()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = "gpt2"
    print(f"Downloading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    config = model.config
    config_path = output_dir / "config.txt"
    config_path.write_text(
        "\n".join(
            [
                f"vocab_size={config.vocab_size}",
                f"n_embd={config.n_embd}",
                f"n_layer={config.n_layer}",
                f"n_head={config.n_head}",
                f"n_positions={config.n_positions}",
                f"head_dim={config.n_embd // config.n_head}",
                f"ffn_dim={config.n_embd * 4}",
            ]
        )
        + "\n"
    )

    scale = 2 ** quant_bits

    wte = model.transformer.wte.weight.detach().numpy()
    wte.astype(np.float32).tofile(output_dir / "wte.bin")

    wpe = model.transformer.wpe.weight.detach().numpy()
    wpe.astype(np.float32).tofile(output_dir / "wpe.bin")

    for layer_idx in range(config.n_layer):
        layer = model.transformer.h[layer_idx]
        layer_dir = output_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        c_attn_w = layer.attn.c_attn.weight.detach().numpy().T
        c_attn_b = layer.attn.c_attn.bias.detach().numpy()
        c_attn_w.astype(np.float32).tofile(layer_dir / "c_attn_w.bin")
        c_attn_b.astype(np.float32).tofile(layer_dir / "c_attn_b.bin")

        c_proj_w = layer.attn.c_proj.weight.detach().numpy().T
        c_proj_b = layer.attn.c_proj.bias.detach().numpy()
        c_proj_w.astype(np.float32).tofile(layer_dir / "c_proj_w.bin")
        c_proj_b.astype(np.float32).tofile(layer_dir / "c_proj_b.bin")

        c_fc_w = layer.mlp.c_fc.weight.detach().numpy().T
        c_fc_b = layer.mlp.c_fc.bias.detach().numpy()
        c_fc_w.astype(np.float32).tofile(layer_dir / "c_fc_w.bin")
        c_fc_b.astype(np.float32).tofile(layer_dir / "c_fc_b.bin")

        c_proj2_w = layer.mlp.c_proj.weight.detach().numpy().T
        c_proj2_b = layer.mlp.c_proj.bias.detach().numpy()
        c_proj2_w.astype(np.float32).tofile(layer_dir / "mlp_proj_w.bin")
        c_proj2_b.astype(np.float32).tofile(layer_dir / "mlp_proj_b.bin")

        ln_1_w = layer.ln_1.weight.detach().numpy()
        ln_1_b = layer.ln_1.bias.detach().numpy()
        ln_2_w = layer.ln_2.weight.detach().numpy()
        ln_2_b = layer.ln_2.bias.detach().numpy()
        ln_1_w.astype(np.float32).tofile(layer_dir / "ln_1_w.bin")
        ln_1_b.astype(np.float32).tofile(layer_dir / "ln_1_b.bin")
        ln_2_w.astype(np.float32).tofile(layer_dir / "ln_2_w.bin")
        ln_2_b.astype(np.float32).tofile(layer_dir / "ln_2_b.bin")

    ln_f_w = model.transformer.ln_f.weight.detach().numpy()
    ln_f_b = model.transformer.ln_f.bias.detach().numpy()
    ln_f_w.astype(np.float32).tofile(output_dir / "ln_f_w.bin")
    ln_f_b.astype(np.float32).tofile(output_dir / "ln_f_b.bin")

    print(f"Export complete -> {output_dir}")

def main() -> int:
    out_dir = Path("model/gpt2-124m")
    export_gpt2_124m(out_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())