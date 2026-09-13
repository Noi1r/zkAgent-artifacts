import argparse
import csv
import json
import math
import os
import sys
import warnings
from collections import defaultdict

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

MAX_LEN = 1024
DEFAULT_TOKENS = 50_000
BOUND_CALIBRATED = "calibrated"
BOUND_STRUCTURAL = "structural"
BOUND_MODEL_STATIC = "model_static"
_BOUND_SOURCES = {BOUND_CALIBRATED, BOUND_STRUCTURAL, BOUND_MODEL_STATIC}



def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    return None


def _vector_per_token(value):
    value = _first_tensor(value)
    if value is None or value.dim() < 2:
        return None
    value = value.detach().abs()
    if value.dim() == 2:
        return value
    return value.amax(dim=tuple(range(2, value.dim())))


def _attention_per_token(value):
    if value is None or not torch.is_tensor(value) or value.dim() != 4:
        return None
    return value.detach().abs().amax(dim=(1, 3))


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, position_embeddings):
    if position_embeddings is None:
        return q, k
    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _pre_hidden(args, kwargs):
    if args and torch.is_tensor(args[0]):
        return args[0]
    return kwargs.get("hidden_states")


def _causal_valid_mask(batch, heads, tq, tk, device):
    q_pos = torch.arange(tk - tq, tk, device=device).view(1, 1, tq, 1)
    k_pos = torch.arange(tk, device=device).view(1, 1, 1, tk)
    return (k_pos <= q_pos).expand(batch, heads, tq, tk)


def _apply_additive_mask(scores, valid, attention_mask):
    if attention_mask is None:
        return scores, valid
    mask = attention_mask[..., : scores.shape[-2], : scores.shape[-1]]
    if mask.dtype == torch.bool:
        valid = valid & mask.expand_as(valid)
        return scores, valid
    masked_scores = scores + mask
    cutoff = torch.finfo(mask.dtype).min / 2 if mask.dtype.is_floating_point else None
    mask_valid = torch.isfinite(mask)
    if cutoff is not None:
        mask_valid = mask_valid & (mask > cutoff)
    return masked_scores, valid & mask_valid.expand_as(valid)


def _record_attention_family(record, prefix, scores, valid, alpha, value):
    score_pt = torch.where(valid, scores.abs(), torch.zeros_like(scores)).amax(dim=(1, 3))
    record(f"{prefix}.attn_score", score_pt)

    neg_inf = torch.full([], float("-inf"), dtype=scores.dtype, device=scores.device)
    masked_scores = torch.where(valid, scores, neg_inf)
    shifted = masked_scores - torch.logsumexp(masked_scores, dim=-1, keepdim=True)
    shifted_pt = torch.where(valid, shifted.abs(), torch.zeros_like(shifted)).amax(dim=(1, 3))
    record(f"{prefix}.softmax_shifted_input", shifted_pt)

    if alpha is None:
        alpha = torch.softmax(masked_scores, dim=-1)
    record(f"{prefix}.softmax_output", _attention_per_token(alpha))

    alpha_v = torch.matmul(alpha.to(value.dtype), value)
    record(f"{prefix}.alpha_v", _attention_per_token(alpha_v))


def _attention_weights_from_output(output, tq, tk):
    if not isinstance(output, (tuple, list)):
        return None
    for item in output[1:]:
        if torch.is_tensor(item) and item.dim() == 4 and item.shape[-2:] == (tq, tk):
            return item
    return None


def install_semantic_checkpoints(model, record):
    specs = {}
    handles = []
    state = defaultdict(dict)

    def add_spec(name, kind, bound_source):
        if bound_source not in _BOUND_SOURCES:
            raise ValueError(f"invalid bound source: {bound_source}")
        old = specs.get(name)
        spec = {"kind": kind, "bound_source": bound_source}
        if old is not None and old != spec:
            raise ValueError(f"conflicting checkpoint specification for {name}")
        specs[name] = spec

    def emit(name, value, reducer=_vector_per_token):
        per_tok = value if (torch.is_tensor(value) and value.dim() == 2) else reducer(value)
        if per_tok is not None:
            record(name, per_tok.detach())

    def output_hook(module, name, kind, source, transform=None):
        add_spec(name, kind, source)

        def hook(_module, _args, output):
            value = transform(output) if transform else output
            emit(name, value)

        handles.append(module.register_forward_hook(hook))

    def pre_hook(module, name, kind, source, transform=None):
        add_spec(name, kind, source)

        def hook(_module, args, kwargs):
            value = transform(args, kwargs) if transform else _pre_hidden(args, kwargs)
            emit(name, value)

        handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    model_type = getattr(model.config, "model_type", "")
    if model_type == "gpt2":
        blocks = list(model.transformer.h)
        if not blocks:
            raise ValueError("GPT-2 model has no transformer blocks")
        pre_hook(blocks[0], "embedding", "embedding", BOUND_MODEL_STATIC)

        for i, block in enumerate(blocks):
            prefix = f"layer.{i}"
            output_hook(block.ln_1, f"{prefix}.attn_norm", "norm", BOUND_STRUCTURAL)
            output_hook(block.ln_2, f"{prefix}.mlp_norm", "norm", BOUND_STRUCTURAL)

            for suffix in ("q_proj", "k_proj", "v_proj"):
                add_spec(f"{prefix}.{suffix}", "qkv", BOUND_CALIBRATED)

            def qkv_hook(_module, _args, output, i=i, prefix=prefix, attn=block.attn):
                q, k, v = output.split(attn.split_size, dim=-1)
                state[i].update(q=q.detach(), k=k.detach(), v=v.detach())
                emit(f"{prefix}.q_proj", q)
                emit(f"{prefix}.k_proj", k)
                emit(f"{prefix}.v_proj", v)

            handles.append(block.attn.c_attn.register_forward_hook(qkv_hook))

            for suffix, kind, source in (
                ("attn_score", "attention", BOUND_CALIBRATED),
                ("softmax_shifted_input", "softmax", BOUND_STRUCTURAL),
                ("softmax_output", "softmax", BOUND_STRUCTURAL),
                ("alpha_v", "attention_output", BOUND_STRUCTURAL),
            ):
                add_spec(f"{prefix}.{suffix}", kind, source)

            def attn_pre(_module, args, kwargs, i=i):
                state[i]["attention_mask"] = kwargs.get("attention_mask")

            handles.append(block.attn.register_forward_pre_hook(attn_pre, with_kwargs=True))

            def attn_hook(_module, _args, output, i=i, prefix=prefix, attn=block.attn):
                local = state[i]
                q_raw, k_raw, v_raw = local["q"], local["k"], local["v"]
                shape = (*q_raw.shape[:-1], attn.num_heads, attn.head_dim)
                q = q_raw.view(shape).transpose(1, 2)
                k = k_raw.view(shape).transpose(1, 2)
                v = v_raw.view(shape).transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-1, -2))
                if attn.scale_attn_weights:
                    scores = scores / math.sqrt(attn.head_dim)
                if getattr(attn, "scale_attn_by_inverse_layer_idx", False):
                    scores = scores / float(attn.layer_idx + 1)
                batch, heads, tq, tk = scores.shape
                valid = _causal_valid_mask(batch, heads, tq, tk, scores.device)
                scores, valid = _apply_additive_mask(scores, valid, local.get("attention_mask"))
                alpha = _attention_weights_from_output(output, tq, tk)
                _record_attention_family(record, prefix, scores, valid, alpha, v)
                local.clear()

            handles.append(block.attn.register_forward_hook(attn_hook))

            output_hook(
                block.attn.c_proj,
                f"{prefix}.attn_projection",
                "attention_output",
                BOUND_CALIBRATED,
            )
            pre_hook(block.ln_2, f"{prefix}.residual_1", "residual", BOUND_CALIBRATED)
            output_hook(block.mlp.c_fc, f"{prefix}.mlp_up", "mlp", BOUND_CALIBRATED)
            output_hook(
                block.mlp.act,
                f"{prefix}.mlp_activation",
                "mlp",
                BOUND_CALIBRATED,
            )
            output_hook(block.mlp.c_proj, f"{prefix}.mlp_down", "mlp", BOUND_CALIBRATED)
            output_hook(block, f"{prefix}.residual_2", "residual", BOUND_CALIBRATED)

        output_hook(model.transformer.ln_f, "final_norm", "norm", BOUND_STRUCTURAL)
        output_hook(model.lm_head, "logits", "logits", BOUND_CALIBRATED)

    elif model_type == "llama":
        blocks = list(model.model.layers)
        if not blocks:
            raise ValueError("Llama model has no decoder layers")
        pre_hook(blocks[0], "embedding", "embedding", BOUND_MODEL_STATIC)

        for i, block in enumerate(blocks):
            prefix = f"layer.{i}"
            output_hook(
                block.input_layernorm,
                f"{prefix}.attn_norm",
                "norm",
                BOUND_STRUCTURAL,
            )
            output_hook(
                block.post_attention_layernorm,
                f"{prefix}.mlp_norm",
                "norm",
                BOUND_STRUCTURAL,
            )

            for suffix, module in (
                ("q_proj", block.self_attn.q_proj),
                ("k_proj", block.self_attn.k_proj),
                ("v_proj", block.self_attn.v_proj),
            ):
                add_spec(f"{prefix}.{suffix}", "qkv", BOUND_CALIBRATED)

                def projection_hook(_module, _args, output, i=i, suffix=suffix, prefix=prefix):
                    state[i][suffix[0]] = output.detach()
                    emit(f"{prefix}.{suffix}", output)

                handles.append(module.register_forward_hook(projection_hook))

            for suffix, kind, source in (
                ("attn_score", "attention", BOUND_CALIBRATED),
                ("softmax_shifted_input", "softmax", BOUND_STRUCTURAL),
                ("softmax_output", "softmax", BOUND_STRUCTURAL),
                ("alpha_v", "attention_output", BOUND_STRUCTURAL),
            ):
                add_spec(f"{prefix}.{suffix}", kind, source)

            def llama_attn_pre(_module, args, kwargs, i=i):
                state[i]["position_embeddings"] = kwargs.get("position_embeddings")
                state[i]["attention_mask"] = kwargs.get("attention_mask")

            handles.append(
                block.self_attn.register_forward_pre_hook(llama_attn_pre, with_kwargs=True)
            )

            def llama_attn_hook(
                _module, _args, output, i=i, prefix=prefix, attn=block.self_attn
            ):
                local = state[i]
                q_raw, k_raw, v_raw = local["q"], local["k"], local["v"]
                input_shape = q_raw.shape[:-1]
                q = q_raw.view(*input_shape, -1, attn.head_dim).transpose(1, 2)
                k = k_raw.view(*input_shape, -1, attn.head_dim).transpose(1, 2)
                v = v_raw.view(*input_shape, -1, attn.head_dim).transpose(1, 2)
                q, k = _apply_rope(q, k, local.get("position_embeddings"))
                k = _repeat_kv(k, attn.num_key_value_groups)
                v = _repeat_kv(v, attn.num_key_value_groups)
                scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scaling
                batch, heads, tq, tk = scores.shape
                valid = _causal_valid_mask(batch, heads, tq, tk, scores.device)
                scores, valid = _apply_additive_mask(scores, valid, local.get("attention_mask"))
                alpha = _attention_weights_from_output(output, tq, tk)
                _record_attention_family(record, prefix, scores, valid, alpha, v)
                local.clear()

            handles.append(block.self_attn.register_forward_hook(llama_attn_hook))

            output_hook(
                block.self_attn.o_proj,
                f"{prefix}.attn_projection",
                "attention_output",
                BOUND_CALIBRATED,
            )
            pre_hook(
                block.post_attention_layernorm,
                f"{prefix}.residual_1",
                "residual",
                BOUND_CALIBRATED,
            )
            output_hook(block.mlp.gate_proj, f"{prefix}.mlp_gate", "mlp", BOUND_CALIBRATED)
            output_hook(block.mlp.up_proj, f"{prefix}.mlp_up", "mlp", BOUND_CALIBRATED)
            output_hook(
                block.mlp.act_fn,
                f"{prefix}.mlp_activation",
                "mlp",
                BOUND_CALIBRATED,
            )
            pre_hook(
                block.mlp.down_proj,
                f"{prefix}.mlp_gated_product",
                "mlp",
                BOUND_CALIBRATED,
            )
            output_hook(block.mlp.down_proj, f"{prefix}.mlp_down", "mlp", BOUND_CALIBRATED)
            output_hook(block, f"{prefix}.residual_2", "residual", BOUND_CALIBRATED)

        output_hook(model.model.norm, "final_norm", "norm", BOUND_STRUCTURAL)
        output_hook(model.lm_head, "logits", "logits", BOUND_CALIBRATED)

    else:
        raise ValueError(
            f"unsupported model_type={model_type!r}; semantic checkpoint adapters "
            "currently support GPT-2 and Llama-family models"
        )

    return specs, handles


def _remove_handles(handles):
    for handle in handles:
        handle.remove()



@torch.no_grad()
def calibrate_semantic(model, ids, device):
    calib = {}

    def record(name, per_tok):
        value = per_tok.max().item()
        calib[name] = max(calib.get(name, 0.0), value)

    specs, handles = install_semantic_checkpoints(model, record)
    for name in specs:
        calib.setdefault(name, 0.0)

    try:
        total = ids.size(1)
        for begin in tqdm(range(0, total, MAX_LEN), desc="Calibrate"):
            batch = ids[:, begin:min(begin + MAX_LEN, total)].to(device)
            model(batch, use_cache=False)
    finally:
        _remove_handles(handles)
    return calib, specs



@torch.no_grad()
def probe_semantic(model, calib, specs, ids, device, union_sources):
    per_ckpt = {
        name: dict(test_max=0.0, n_over=0, n_total=0, ratio=0.0)
        for name in calib
    }
    state = {"selected": None, "all": None}

    def merge_union(which, over):
        current = state[which]
        if current is None:
            state[which] = over.clone()
        elif current.shape != over.shape:
            raise ValueError(
                f"checkpoint token-shape mismatch: {current.shape} != {over.shape}"
            )
        else:
            state[which] = current | over

    def record(name, per_tok):
        cmax = calib[name]
        over = per_tok > cmax
        maximum = per_tok.max().item()
        row = per_ckpt[name]
        row["test_max"] = max(row["test_max"], maximum)
        row["n_over"] += over.sum().item()
        row["n_total"] += over.numel()
        ratio = maximum / cmax if cmax > 0 else (math.inf if maximum > 0 else 1.0)
        row["ratio"] = max(row["ratio"], ratio)
        merge_union("all", over)
        if specs[name]["bound_source"] in union_sources:
            merge_union("selected", over)

    observed_specs, handles = install_semantic_checkpoints(model, record)
    if observed_specs != specs:
        _remove_handles(handles)
        raise ValueError("checkpoint schema changed between calibration and probe")

    n_selected = 0
    n_all = 0
    n_total = 0
    try:
        total = ids.size(1)
        for begin in tqdm(range(0, total, MAX_LEN), desc="Probe"):
            state["selected"] = None
            state["all"] = None
            batch = ids[:, begin:min(begin + MAX_LEN, total)].to(device)
            n_total += batch.numel()
            model(batch, use_cache=False)
            if state["selected"] is not None:
                n_selected += state["selected"].sum().item()
            if state["all"] is not None:
                n_all += state["all"].sum().item()
    finally:
        _remove_handles(handles)
    return per_ckpt, n_selected, n_all, n_total



def _wt2_test(tok, n, cache_dir=None):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok.encode(text, return_tensors="pt")
    return ids[:, :n] if ids.size(1) > n else ids


def _wt103_test(tok, n, cache_dir=None):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", cache_dir=cache_dir)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok.encode(text, return_tensors="pt")
    total = ids.size(1)
    if total >= 2 * n:
        return ids[:, n: 2 * n]
    if total > n:
        return ids[:, n:]
    return ids


def _wt2_train(tok, n, cache_dir=None):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok.encode(text, return_tensors="pt")
    return ids[:, :n] if ids.size(1) > n else ids


def _c4_validation(tok, n, cache_dir=None):
    ds = load_dataset("allenai/c4", "en", split="validation",
                      streaming=True, cache_dir=cache_dir)
    texts, approx = [], 0
    next_tokenization_check = int(n * 1.1)
    ids = None
    for ex in ds:
        texts.append(ex["text"])
        approx += len(ex["text"]) // 4
        if approx < next_tokenization_check:
            continue
        ids = tok.encode("\n\n".join(texts), return_tensors="pt")
        if ids.size(1) >= n:
            return ids[:, :n]
        shortfall = n - ids.size(1)
        next_tokenization_check = approx + max(1024, int(shortfall * 1.5))
    if ids is None:
        ids = tok.encode("\n\n".join(texts), return_tensors="pt")
    raise ValueError(f"C4 validation yielded only {ids.size(1)} tokens; requested {n}")


_DATASETS = {
    "wt2":        ("WikiText-2 test",       _wt2_test),
    "wt2-train":  ("WikiText-2 train",      _wt2_train),
    "wt103":      ("WikiText-103 test",     _wt103_test),
    "c4":         ("C4 validation",         _c4_validation),
}


def load_split(tok, key, n, cache_dir=None, allow_c4_fallback=False):
    label, fn = _DATASETS[key]
    print(f"Loading {label} ({n} tokens) ...")
    try:
        ids = fn(tok, n, cache_dir)
    except Exception as e:
        if key == "c4" and allow_c4_fallback:
            print(f"  C4 load failed ({type(e).__name__}: {e}); falling back to WikiText-103.")
            ids = _wt103_test(tok, n, cache_dir)
            label = "WikiText-103 test (C4 fallback)"
        else:
            raise
    if ids.size(1) != n:
        raise ValueError(f"{label} yielded {ids.size(1)} tokens; requested exactly {n}")
    print(f"  tokens loaded: {ids.size(1)}")
    return label, ids



def report(model_name, calib_label, test_label, specs, calib, per_ckpt,
           n_selected, n_all, n_total_tokens, union_sources, top_k=20):
    rows_by_kind = defaultdict(list)
    rows_by_source = defaultdict(list)
    for name, checkpoint in per_ckpt.items():
        if checkpoint["n_over"] == 0:
            continue
        kind = specs[name]["kind"]
        source = specs[name]["bound_source"]
        row = (
            name, source, calib[name], checkpoint["test_max"],
            checkpoint["ratio"], checkpoint["n_over"], checkpoint["n_total"],
        )
        rows_by_kind[kind].append(row)
        rows_by_source[source].append(row)

    print(f"\n══ {model_name}  |  calib={calib_label}  test={test_label} ══")
    kind_counts = defaultdict(int)
    source_counts = defaultdict(int)
    for spec in specs.values():
        kind_counts[spec["kind"]] += 1
        source_counts[spec["bound_source"]] += 1
    print(f"\n  Checkpoints registered: {len(calib)}")
    print("    by kind   : " + ", ".join(
        f"{kind}={count}" for kind, count in sorted(kind_counts.items())
    ))
    print("    by source : " + ", ".join(
        f"{source}={count}" for source, count in sorted(source_counts.items())
    ))

    summary = {}
    for kind in sorted(rows_by_kind):
        rows = rows_by_kind[kind]
        n_checkpoints_over = len(rows)
        max_ratio = max(row[4] for row in rows)
        total_n_over = sum(row[5] for row in rows)
        total_n = sum(row[6] for row in rows)
        rate = 100.0 * total_n_over / max(total_n, 1)
        summary[kind] = {
            "checkpoints_over": n_checkpoints_over,
            "max_ratio": max_ratio,
            "checkpoint_token_rate_pct": rate,
        }
        print(f"\n  [{kind}] checkpoints with overflow: {n_checkpoints_over}  "
              f"max ratio={max_ratio:.3f}x  checkpoint-token rate={rate:.4f}%")

        rows.sort(key=lambda row: row[4], reverse=True)
        header = (
            f"    {'name':<44s} {'source':<12s} {'calib_max':>10s}  "
            f"{'test_max':>10s}  {'ratio':>8s}  {'over %':>8s}"
        )
        print(header)
        print("    " + "-" * (len(header) - 4))
        for name, source, cmax, tmax, ratio, n_over, n_total in rows[:top_k]:
            percent = 100.0 * n_over / max(n_total, 1)
            print(f"    {name[:44]:<44s} {source:<12s} {cmax:10.3f}  "
                  f"{tmax:10.3f}  {ratio:7.3f}x  {percent:7.4f}%")
        if len(rows) > top_k:
            print(f"    ({len(rows) - top_k} more checkpoints omitted)")

    source_summary = {}
    for source, rows in sorted(rows_by_source.items()):
        source_summary[source] = {
            "checkpoints_over": len(rows),
            "max_ratio": max(row[4] for row in rows),
        }

    selected_rate = 100.0 * n_selected / max(n_total_tokens, 1)
    all_rate = 100.0 * n_all / max(n_total_tokens, 1)
    source_label = ",".join(sorted(union_sources))
    print("\n  Per-token calibration-bound exceedance:")
    print(f"    primary union sources      : {source_label}")
    print(f"    test tokens scanned        : {n_total_tokens}")
    print(f"    selected-union tokens      : {n_selected}")
    print(f"    selected-union rate        : {selected_rate:.4f}%")
    print(f"    all-checkpoint tokens      : {n_all}")
    print(f"    all-checkpoint rate        : {all_rate:.4f}%")

    checkpoint_stats = {}
    for name, checkpoint in per_ckpt.items():
        checkpoint_stats[name] = {
            **specs[name],
            "calib_max": calib[name],
            "test_max": checkpoint["test_max"],
            "max_ratio": checkpoint["ratio"],
            "n_overflow_tokens": checkpoint["n_over"],
            "n_total_tokens": checkpoint["n_total"],
            "per_token_pct": (
                100.0 * checkpoint["n_over"] / checkpoint["n_total"]
                if checkpoint["n_total"] else 0.0
            ),
        }

    return {
        "model": model_name,
        "calib_dataset": calib_label,
        "test_dataset": test_label,
        "union_bound_sources": sorted(union_sources),
        "n_checkpoints": len(specs),
        "n_total_tokens": n_total_tokens,
        "n_overflow_tokens": n_selected,
        "per_token_pct": selected_rate,
        "n_all_checkpoint_overflow_tokens": n_all,
        "all_checkpoint_pct": all_rate,
        "per_kind": summary,
        "per_source": source_summary,
        "checkpoints": checkpoint_stats,
    }



def run_one(model_name, args, tok=None, model=None):
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    print(f"\n{'#' * 70}\n# Model: {model_name}\n{'#' * 70}")
    print(f"Device: {device}")

    if args.dtype == "auto":
        use_fp16 = "gpt2" not in model_name.lower() and device != "cpu"
        dtype = torch.float16 if use_fp16 else torch.float32
    else:
        dtype = {"fp32": torch.float32, "fp16": torch.float16,
                 "bf16": torch.bfloat16}[args.dtype]
    print(f"Dtype: {dtype}")

    print(f"Loading: {model_name}")
    mk = dict(
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    )
    if device == "cuda":
        mk["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **mk).eval()
    model.config.use_cache = False
    if device == "cpu":
        model = model.to("cpu")
    tok = AutoTokenizer.from_pretrained(
        model_name, local_files_only=args.local_files_only
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    calib_label, calib_ids = load_split(tok, args.calib_dataset,
                                        args.calib_tokens, args.cache_dir,
                                        args.allow_c4_fallback)
    print(f"\n── Calibrating ({calib_label}) ──")
    calib, specs = calibrate_semantic(model, calib_ids, device)
    print(f"  semantic checkpoints registered: {len(specs)}")
    n_zero = sum(1 for v in calib.values() if v == 0)
    if n_zero:
        print(f"  warn: {n_zero}/{len(calib)} checkpoints had max=0 "
              f"(unreached during calibration)")

    rows = []
    for ts_key in args.test_datasets:
        test_label, test_ids = load_split(tok, ts_key, args.test_tokens,
                                          args.cache_dir, args.allow_c4_fallback)
        print(f"\n── Probing ({test_label}) ──")
        per_ckpt, n_selected, n_all, n_total = probe_semantic(
            model, calib, specs, test_ids, device, args.union_sources)
        rows.append(report(
            model_name, calib_label, test_label, specs, calib, per_ckpt,
            n_selected, n_all, n_total, args.union_sources, args.top_k,
        ))

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(
        description='Measure calibration-bound exceedances at LLM checkpoints.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--models", default="gpt2",
                    help="Comma-separated HF model IDs (run sequentially).")
    ap.add_argument("--calib-dataset", default="wt2-train",
                    choices=list(_DATASETS.keys()))
    ap.add_argument("--test-datasets", default="wt2,wt103,c4",
                    help="Comma-separated test sets (any of "
                         f"{','.join(_DATASETS.keys())}).")
    ap.add_argument("--calib-tokens", type=int, default=DEFAULT_TOKENS)
    ap.add_argument("--test-tokens", type=int, default=DEFAULT_TOKENS)
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from the local HF cache (datasets may still download).",
    )
    ap.add_argument(
        "--allow-c4-fallback",
        action="store_true",
        help="Allow C4 load failures to fall back to WT103 (disabled for paper runs).",
    )
    ap.add_argument("--top-k", type=int, default=10,
                    help="Show worst-K overflow checkpoints per kind.")
    ap.add_argument(
        "--union-bound-sources",
        default=BOUND_CALIBRATED,
        help="Comma-separated sources included in the primary per-token union "
             "(calibrated,structural,model_static).",
    )
    ap.add_argument("--json-out", default=None,
                    help="Optional path to dump structured results as JSON.")
    ap.add_argument("--csv-out", default=None,
                    help="Optional CSV summary path.")
    args = ap.parse_args()

    args.test_datasets = [s.strip() for s in args.test_datasets.split(",") if s.strip()]
    for ts in args.test_datasets:
        if ts not in _DATASETS:
            raise ValueError(f"Unknown test dataset: {ts}")
    args.union_sources = {
        source.strip()
        for source in args.union_bound_sources.split(",")
        if source.strip()
    }
    unknown_sources = args.union_sources - _BOUND_SOURCES
    if unknown_sources or not args.union_sources:
        raise ValueError(
            f"Invalid --union-bound-sources: {sorted(unknown_sources)}; "
            f"choose from {sorted(_BOUND_SOURCES)}"
        )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_rows = []
    failed = []
    for mn in models:
        try:
            all_rows.extend(run_one(mn, args))
        except Exception as e:
            print(f"\n!! {mn} failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed.append((mn, f"{type(e).__name__}: {e}"))

    print("\n" + "=" * 78)
    print("  AGGREGATE SUMMARY (per-token overflow rate, union over checkpoints)")
    print("=" * 78)
    print(f"  {'Model':<40s} {'Test set':<28s} {'overflow %':>12s}")
    print("  " + "-" * 80)
    for r in all_rows:
        print(f"  {r['model']:<40s} {r['test_dataset']:<28s} "
              f"{r['per_token_pct']:11.4f}%")
    if failed:
        print("\n  Failed models:")
        for mn, err in failed:
            print(f"    {mn}: {err}")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\nJSON saved: {args.json_out}")

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "calib_dataset", "test_dataset",
                        "union_bound_sources", "n_checkpoints",
                        "n_total_tokens", "n_overflow_tokens",
                        "per_token_pct", "n_all_checkpoint_overflow_tokens",
                        "all_checkpoint_pct"])
            for r in all_rows:
                w.writerow([r["model"], r["calib_dataset"], r["test_dataset"],
                            ",".join(r["union_bound_sources"]), r["n_checkpoints"],
                            r["n_total_tokens"], r["n_overflow_tokens"],
                            f"{r['per_token_pct']:.6f}",
                            r["n_all_checkpoint_overflow_tokens"],
                            f"{r['all_checkpoint_pct']:.6f}"])
        print(f"CSV saved: {args.csv_out}")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nFATAL: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
