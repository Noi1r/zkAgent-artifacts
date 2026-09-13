#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D


transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=r".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=r".*Token indices sequence length.*")

MAX_LEN = 1024
STRIDE = 512
LAYERNORM_BOUND_MODES = ("parameter", "dimension")
ATTENTION_VALUE_MODES = ("structural", "dimension")
PAPER_CONFIGURATIONS = {
    "ours_full": {
        "layernorm_bound_mode": "parameter",
        "scheme": "weight_dep",
        "score_mode": "wide",
        "attention_value_mode": "structural",
        "residual_mode": "delayed",
    },
    "naive_dim": {
        "layernorm_bound_mode": "dimension",
        "scheme": "worst_dim",
        "score_mode": "requant",
        "attention_value_mode": "dimension",
        "residual_mode": "requant",
    },
}
MAX_EXACT_FLOAT_INTEGER = 1 << 53


class QuantizationInvariantError(RuntimeError):
    pass


@dataclass
class QTensor:

    q: torch.Tensor
    scale: float
    bound: int

    def real(self, dtype=torch.float64):
        return self.q.to(dtype) * self.scale


@dataclass
class QuantizedParameter:
    q: torch.Tensor
    scale: float
    bound: int


@dataclass
class RunDiagnostics:
    score_nonzero_inputs: int = 0
    score_collapsed_to_zero: int = 0
    score_max_abs_error: float = 0.0
    softmax_entries: int = 0
    softmax_zero_entries: int = 0
    softmax_rows: int = 0
    softmax_mass_deficit_sum: float = 0.0
    softmax_mass_deficit_max: float = 0.0
    layernorm_rows: int = 0
    layernorm_zero_variance_rows: int = 0
    layernorm_min_positive_variance: float = math.inf
    max_residual_actual: float = 0.0
    max_residual_bound: int = 0
    max_residual_bits: int = 0
    max_logit_actual: float = 0.0
    max_logit_bound: int = 0
    max_logit_bits: int = 0
    max_linear_utilization: float = 0.0
    stage_maxima: Dict[str, float] = field(default_factory=dict)

    def observe_stage(self, name: str, q: torch.Tensor, bound: int):
        actual = _max_abs(q)
        self.stage_maxima[name] = max(self.stage_maxima.get(name, 0.0), actual)
        if bound > 0:
            self.max_linear_utilization = max(
                self.max_linear_utilization, actual / float(bound)
            )

    def observe_residual(self, q: torch.Tensor, bound: int):
        actual = _max_abs(q)
        self.max_residual_actual = max(self.max_residual_actual, actual)
        self.max_residual_bound = max(self.max_residual_bound, int(bound))
        self.max_residual_bits = max(
            self.max_residual_bits, signed_bits_for_magnitude(bound)
        )

    def observe_logits(self, q: torch.Tensor, bound: int):
        actual = _max_abs(q)
        self.max_logit_actual = max(self.max_logit_actual, actual)
        self.max_logit_bound = max(self.max_logit_bound, int(bound))
        self.max_logit_bits = max(
            self.max_logit_bits, signed_bits_for_magnitude(bound)
        )

    def as_row(self):
        collapse_pct = (
            100.0 * self.score_collapsed_to_zero / self.score_nonzero_inputs
            if self.score_nonzero_inputs
            else 0.0
        )
        softmax_zero_pct = (
            100.0 * self.softmax_zero_entries / self.softmax_entries
            if self.softmax_entries
            else 0.0
        )
        mass_deficit_mean = (
            self.softmax_mass_deficit_sum / self.softmax_rows
            if self.softmax_rows
            else 0.0
        )
        min_variance = (
            self.layernorm_min_positive_variance
            if math.isfinite(self.layernorm_min_positive_variance)
            else 0.0
        )
        return {
            "score_zero_collapse_pct": f"{collapse_pct:.9f}",
            "score_max_abs_error": f"{self.score_max_abs_error:.12g}",
            "softmax_zero_pct": f"{softmax_zero_pct:.9f}",
            "softmax_mass_deficit_mean": f"{mass_deficit_mean:.9f}",
            "softmax_mass_deficit_max": f"{self.softmax_mass_deficit_max:.0f}",
            "layernorm_zero_variance_rows": self.layernorm_zero_variance_rows,
            "layernorm_min_positive_variance": f"{min_variance:.12g}",
            "max_residual_actual": f"{self.max_residual_actual:.0f}",
            "max_residual_bound": self.max_residual_bound,
            "max_residual_bits": self.max_residual_bits,
            "max_logit_actual": f"{self.max_logit_actual:.0f}",
            "max_logit_bound": self.max_logit_bound,
            "max_logit_bits": self.max_logit_bits,
            "max_linear_utilization": f"{self.max_linear_utilization:.12g}",
        }


def signed_bits_for_magnitude(bound):
    bound = int(math.ceil(bound))
    return 1 if bound <= 0 else bound.bit_length() + 1


def _max_abs(tensor):
    return tensor.detach().abs().max().item() if tensor.numel() else 0.0


def _check_integer_values(q, name):
    if not torch.isfinite(q).all():
        raise QuantizationInvariantError(f"{name}: non-finite integer tensor")
    if q.is_floating_point() and q.numel():
        error = (q - torch.round(q)).detach().abs().max().item()
        if error != 0:
            raise QuantizationInvariantError(
                f"{name}: tensor is not integer-valued (max fractional error {error})"
            )


def _check_bound(q, bound, name):
    _check_integer_values(q, name)
    actual = _max_abs(q)
    if actual > float(bound):
        raise QuantizationInvariantError(
            f"{name}: |q|={actual:.0f} exceeds declared bound {bound}"
        )


def _check_exact_float_storage(q, name):
    actual = _max_abs(q)
    if actual > MAX_EXACT_FLOAT_INTEGER:
        raise QuantizationInvariantError(
            f"{name}: integer magnitude {actual:.0f} exceeds float64 exact range 2^53"
        )


def _safe_ceil(value):
    return math.ceil(float(value))


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantize_parameter(x, R, name):

    x64 = x.detach().to(torch.float64)
    max_abs = _max_abs(x64)
    if max_abs == 0:
        q = torch.zeros_like(x64, dtype=torch.int32)
        return QuantizedParameter(q, 1.0, R)
    scale = max_abs / R
    q64 = torch.round(x64 / scale)
    _check_bound(q64, R, name)
    if _max_abs(q64) > torch.iinfo(torch.int32).max:
        raise QuantizationInvariantError(f"{name}: parameter does not fit int32")
    return QuantizedParameter(q64.to(torch.int32), scale, R)


def extract_affine(module):

    if isinstance(module, Conv1D):
        weight = module.weight.detach().t().contiguous()
    else:
        weight = module.weight.detach()
    weight = weight.to(torch.float64)
    bias = None if module.bias is None else module.bias.detach().to(torch.float64)
    return weight, bias



class FusedEmbedding:

    def __init__(self, token_weight, position_weight, R):
        self.R = R
        e = token_weight.detach().to(torch.float64)
        p = position_weight.detach().to(torch.float64)

        e_max = e.max(dim=0).values
        e_min = e.min(dim=0).values
        p_max = p.max(dim=0).values
        p_min = p.min(dim=0).values
        max_sum = e_max + p_max
        min_sum = e_min + p_min
        real_bound = torch.maximum(max_sum.abs(), min_sum.abs()).max().item()

        if real_bound == 0:
            self.scale = 1.0
            self.token_q = torch.zeros_like(e, dtype=torch.int32)
            self.position_q = torch.zeros_like(p, dtype=torch.int32)
            self.bound = 0
            return

        self.scale = real_bound / R
        token_q64 = torch.round(e / self.scale)
        position_q64 = torch.round(p / self.scale)
        if max(_max_abs(token_q64), _max_abs(position_q64)) > torch.iinfo(torch.int32).max:
            raise QuantizationInvariantError("fused embeddings do not fit int32")
        self.token_q = token_q64.to(torch.int32)
        self.position_q = position_q64.to(torch.int32)

        tq = self.token_q.to(torch.int64)
        pq = self.position_q.to(torch.int64)
        quant_max_sum = tq.max(dim=0).values + pq.max(dim=0).values
        quant_min_sum = tq.min(dim=0).values + pq.min(dim=0).values
        self.bound = int(
            torch.maximum(quant_max_sum.abs(), quant_min_sum.abs()).max().item()
        )

    def lookup(self, token_ids):
        batch, length = token_ids.shape
        if length > self.position_q.shape[0]:
            raise ValueError(
                f"sequence length {length} exceeds positional table "
                f"{self.position_q.shape[0]}"
            )
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0)
        token = self.token_q[token_ids].to(torch.int64)
        position = self.position_q[positions].to(torch.int64)
        q = (token + position).to(torch.float64)
        _check_bound(q, self.bound, "embedding_sum")
        _check_exact_float_storage(q, "embedding_sum")
        return QTensor(q, self.scale, self.bound)


class QuantLinear:

    def __init__(self, weight, bias, R, name):
        if weight.ndim != 2:
            raise ValueError(f"{name}: expected rank-2 weight")
        self.R = R
        self.name = name
        self.in_dim = weight.shape[1]
        self.out_dim = weight.shape[0]
        self.weight = quantize_parameter(weight, R, f"{name}.weight")
        self.weight_norm = self.weight.q.to(torch.float64) / R
        self.weight_l1 = self.weight.q.abs().to(torch.int64).sum(dim=1)
        self.bias_real = None if bias is None else bias.detach().to(torch.float64)
        self._bound_cache: Dict[
            Tuple[str, str],
            Tuple[int, Optional[torch.Tensor], float],
        ] = {}

    @classmethod
    def from_module(cls, module, R, name):
        weight, bias = extract_affine(module)
        return cls(weight, bias, R, name)

    def _bias_integer(self, accumulator_scale):
        if self.bias_real is None:
            return None
        if not math.isfinite(accumulator_scale) or accumulator_scale <= 0:
            raise QuantizationInvariantError(
                f"{self.name}: invalid accumulator scale {accumulator_scale}"
            )
        bq = torch.round(self.bias_real / accumulator_scale)
        _check_integer_values(bq, f"{self.name}.bias")
        _check_exact_float_storage(bq, f"{self.name}.bias")
        if _max_abs(bq) > torch.iinfo(torch.int64).max:
            raise QuantizationInvariantError(f"{self.name}: bias does not fit int64")
        return bq

    def bound_and_bias(self, x_scale, scheme):
        key = (scheme, float(x_scale).hex())
        if key in self._bound_cache:
            return self._bound_cache[key]

        accumulator_scale = x_scale * self.weight.scale
        bq = self._bias_integer(accumulator_scale)
        bias_i64 = (
            torch.zeros(self.out_dim, dtype=torch.int64, device=self.weight.q.device)
            if bq is None
            else bq.to(torch.int64)
        )

        if scheme == "weight_dep":
            per_output = self.weight_l1 * self.R + bias_i64.abs()
            bound = int(per_output.max().item())
        elif scheme == "worst_dim":
            bound = self.in_dim * self.R * self.R + int(bias_i64.abs().max().item())
        else:
            raise ValueError(f"unknown quantization scheme: {scheme}")

        bound = max(bound, 1)
        result = (bound, bq, accumulator_scale)
        self._bound_cache[key] = result
        return result

    def _accumulator_normalized(self, x, bq):
        if x.bound > self.R:
            raise QuantizationInvariantError(
                f"{self.name}: linear input bound {x.bound} exceeds R={self.R}"
            )
        x_norm = x.q.to(torch.float64) / self.R
        bias_norm = (
            None
            if bq is None
            else bq.to(torch.float64) / float(self.R * self.R)
        )
        return F.linear(x_norm, self.weight_norm, bias_norm)

    def __call__(self, x, scheme, diagnostics=None):
        bound, bq, accumulator_scale = self.bound_and_bias(x.scale, scheme)
        accumulator_norm = self._accumulator_normalized(x, bq)
        bound_norm = bound / float(self.R * self.R)
        q_out = torch.round(self.R * accumulator_norm / bound_norm)
        _check_bound(q_out, self.R, self.name)
        output_scale = accumulator_scale * bound / self.R
        if diagnostics is not None:
            diagnostics.observe_stage(self.name, q_out, self.R)
        return QTensor(q_out, output_scale, self.R)

    def wide_affine(self, x, scheme, diagnostics=None):

        bound, bq, accumulator_scale = self.bound_and_bias(x.scale, scheme)
        accumulator_norm = self._accumulator_normalized(x, bq)
        q_wide = torch.round(accumulator_norm * float(self.R * self.R))
        _check_bound(q_wide, bound, f"{self.name}.wide")
        _check_exact_float_storage(q_wide, f"{self.name}.wide")
        if diagnostics is not None:
            diagnostics.observe_stage(f"{self.name}.wide", q_wide, bound)
        return QTensor(q_wide, accumulator_scale, bound)

    def wide_output(self, x, diagnostics=None):

        bound, bq, accumulator_scale = self.bound_and_bias(
            x.scale, "weight_dep"
        )
        accumulator_norm = self._accumulator_normalized(x, bq)
        q_wide = torch.round(accumulator_norm * float(self.R * self.R))
        _check_bound(q_wide, bound, self.name)
        _check_exact_float_storage(q_wide, self.name)
        if diagnostics is not None:
            diagnostics.observe_logits(q_wide, bound)
        return QTensor(q_wide, accumulator_scale, bound)

class EpsilonFreeLayerNorm:

    def __init__(self, width, R, name):
        if width <= 1:
            raise ValueError(f"{name}: LayerNorm width must exceed one")
        self.width = width
        self.R = R
        self.name = name
        self.output_scale = math.sqrt(width - 1) / R
        self.coordinate_bound = math.sqrt(width - 1)

    def __call__(self, x, diagnostics=None):
        q = x.q.to(torch.float64)
        mean = q.mean(dim=-1, keepdim=True)
        centered = q - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        nonzero = variance > 0

        inv_std = torch.zeros_like(variance)
        inv_std[nonzero] = torch.rsqrt(variance[nonzero])
        normalized = centered * inv_std

        max_abs = _max_abs(normalized)
        numerical_slack = 1e-10 * max(1.0, self.coordinate_bound)
        if max_abs > self.coordinate_bound + numerical_slack:
            raise QuantizationInvariantError(
                f"{self.name}: normalized coordinate {max_abs} exceeds "
                f"sqrt(d-1)={self.coordinate_bound}"
            )
        normalized = torch.clamp(
            normalized, -self.coordinate_bound, self.coordinate_bound
        )
        q_out = torch.round(normalized / self.output_scale)
        _check_bound(q_out, self.R, self.name)

        if diagnostics is not None:
            flat_variance = variance.reshape(-1)
            diagnostics.layernorm_rows += flat_variance.numel()
            zero_count = int((flat_variance == 0).sum().item())
            diagnostics.layernorm_zero_variance_rows += zero_count
            positive = flat_variance[flat_variance > 0]
            if positive.numel():
                min_real_variance = (
                    positive.min().item() * x.scale * x.scale
                )
                diagnostics.layernorm_min_positive_variance = min(
                    diagnostics.layernorm_min_positive_variance,
                    min_real_variance,
                )
            diagnostics.observe_stage(self.name, q_out, self.R)

        return QTensor(q_out, self.output_scale, self.R)


class QuantLayerNormAffine:

    def __init__(self, layernorm, input_scale, R, name):
        width = int(layernorm.normalized_shape[-1])
        device = layernorm.weight.device if layernorm.weight is not None else None
        gamma_real = (
            layernorm.weight.detach().to(torch.float64)
            if layernorm.weight is not None
            else torch.ones(width, dtype=torch.float64, device=device)
        )
        beta_real = (
            layernorm.bias.detach().to(torch.float64)
            if layernorm.bias is not None
            else torch.zeros(width, dtype=torch.float64, device=device)
        )
        if gamma_real.numel() != width or beta_real.numel() != width:
            raise ValueError(f"{name}: invalid LayerNorm affine shape")
        if not math.isfinite(input_scale) or input_scale <= 0:
            raise QuantizationInvariantError(
                f"{name}: invalid normalized-input scale {input_scale}"
            )

        self.R = R
        self.name = name
        self.width = width
        self.input_scale = float(input_scale)
        self.gamma = quantize_parameter(gamma_real, R, f"{name}.gamma")
        self.accumulator_scale = self.input_scale * self.gamma.scale
        if not math.isfinite(self.accumulator_scale) or self.accumulator_scale <= 0:
            raise QuantizationInvariantError(
                f"{name}: invalid affine accumulator scale "
                f"{self.accumulator_scale}"
            )

        beta_q = torch.round(beta_real / self.accumulator_scale)
        _check_integer_values(beta_q, f"{name}.beta")
        _check_exact_float_storage(beta_q, f"{name}.beta")
        if _max_abs(beta_q) > torch.iinfo(torch.int64).max:
            raise QuantizationInvariantError(f"{name}: beta does not fit int64")
        self.beta_q = beta_q

        per_coordinate = (
            self.gamma.q.abs().to(torch.int64) * R
            + self.beta_q.abs().to(torch.int64)
        )
        self.parameter_bound = max(int(per_coordinate.max().item()), 1)
        self.dimension_bound = max(
            R * R + int(self.beta_q.abs().max().item()),
            1,
        )
        if self.parameter_bound > self.dimension_bound:
            raise QuantizationInvariantError(
                f"{name}: parameter bound {self.parameter_bound} exceeds "
                f"dimension bound {self.dimension_bound}"
            )
        for mode, bound in (
            ("parameter", self.parameter_bound),
            ("dimension", self.dimension_bound),
        ):
            if bound > MAX_EXACT_FLOAT_INTEGER:
                raise QuantizationInvariantError(
                    f"{name}: {mode} affine bound {bound} exceeds "
                    "float64 exact range 2^53"
                )
        self.output_scales = {
            "parameter": self.accumulator_scale * self.parameter_bound / R,
            "dimension": self.accumulator_scale * self.dimension_bound / R,
        }
        self.accumulator_bound = self.parameter_bound
        self.output_scale = self.output_scales["parameter"]

    def __call__(self, x, diagnostics=None, bound_mode="parameter"):
        if bound_mode not in LAYERNORM_BOUND_MODES:
            raise ValueError(f"unknown LayerNorm bound mode: {bound_mode}")
        if x.q.shape[-1] != self.width:
            raise ValueError(
                f"{self.name}: input width {x.q.shape[-1]} != {self.width}"
            )
        if x.bound > self.R:
            raise QuantizationInvariantError(
                f"{self.name}: normalized input bound {x.bound} exceeds "
                f"R={self.R}"
            )
        if not math.isclose(x.scale, self.input_scale, rel_tol=0.0, abs_tol=0.0):
            raise QuantizationInvariantError(
                f"{self.name}: normalized input scale {x.scale} != "
                f"{self.input_scale}"
            )

        accumulator = (
            x.q.to(torch.float64) * self.gamma.q.to(torch.float64)
            + self.beta_q
        )
        bound = (
            self.parameter_bound
            if bound_mode == "parameter"
            else self.dimension_bound
        )
        _check_bound(accumulator, bound, f"{self.name}.acc.{bound_mode}")
        _check_exact_float_storage(accumulator, f"{self.name}.acc")
        q_out = torch.round(
            self.R * accumulator / float(bound)
        )
        stage_name = f"{self.name}.{bound_mode}"
        _check_bound(q_out, self.R, stage_name)
        if diagnostics is not None:
            diagnostics.observe_stage(stage_name, q_out, self.R)
        return QTensor(q_out, self.output_scales[bound_mode], self.R)


def align_residual(x, y, name, diagnostics=None):

    if x.scale <= 0 or y.scale <= 0:
        raise QuantizationInvariantError(f"{name}: non-positive residual scale")
    output_scale = min(x.scale, y.scale)
    x_ratio = x.scale / output_scale
    y_ratio = y.scale / output_scale
    x_aligned = torch.round(x.q.to(torch.float64) * x_ratio)
    y_aligned = torch.round(y.q.to(torch.float64) * y_ratio)
    q_out = x_aligned + y_aligned

    output_bound = (
        _safe_ceil(x.bound * x_ratio)
        + _safe_ceil(y.bound * y_ratio)
    )
    _check_bound(q_out, output_bound, name)
    _check_exact_float_storage(q_out, name)
    if diagnostics is not None:
        diagnostics.observe_residual(q_out, output_bound)
    return QTensor(q_out, output_scale, output_bound)


class QuantGPT2:

    def __init__(
        self,
        model,
        Q,
        max_sequence_length=MAX_LEN,
    ):
        if getattr(model.config, "model_type", None) != "gpt2":
            raise ValueError("the PPL experiment currently supports GPT-2 only")
        if Q <= 0 or Q > 22:
            raise ValueError("Q must be in [1,22] for the float64/int32 simulator")
        self.model = model
        self.Q = Q
        self.R = 1 << Q
        self.max_sequence_length = max_sequence_length
        self.embedding = FusedEmbedding(
            model.transformer.wte.weight,
            model.transformer.wpe.weight,
            self.R,
        )

        self.blocks = []
        for layer_idx, block in enumerate(model.transformer.h):
            qkv_weight, qkv_bias = extract_affine(block.attn.c_attn)
            split = block.attn.split_size
            q_weight, k_weight, v_weight = qkv_weight.split(split, dim=0)
            if qkv_bias is None:
                q_bias = k_bias = v_bias = None
            else:
                q_bias, k_bias, v_bias = qkv_bias.split(split, dim=0)

            up_weight, up_bias = extract_affine(block.mlp.c_fc)

            ln1 = EpsilonFreeLayerNorm(
                block.ln_1.normalized_shape[-1],
                self.R,
                f"layer.{layer_idx}.ln1.normalized",
            )
            ln2 = EpsilonFreeLayerNorm(
                block.ln_2.normalized_shape[-1],
                self.R,
                f"layer.{layer_idx}.ln2.normalized",
            )

            ln1_affine = QuantLayerNormAffine(
                block.ln_1,
                ln1.output_scale,
                self.R,
                f"layer.{layer_idx}.ln1.affine",
            )
            ln2_affine = QuantLayerNormAffine(
                block.ln_2,
                ln2.output_scale,
                self.R,
                f"layer.{layer_idx}.ln2.affine",
            )

            self.blocks.append({
                "module": block,
                "ln1": ln1,
                "ln1_affine": ln1_affine,
                "q": QuantLinear(q_weight, q_bias, self.R, f"layer.{layer_idx}.q"),
                "k": QuantLinear(k_weight, k_bias, self.R, f"layer.{layer_idx}.k"),
                "v": QuantLinear(v_weight, v_bias, self.R, f"layer.{layer_idx}.v"),
                "attn_proj": QuantLinear.from_module(
                    block.attn.c_proj, self.R, f"layer.{layer_idx}.attn_proj"
                ),
                "ln2": ln2,
                "ln2_affine": ln2_affine,
                "ffn_up": QuantLinear(
                    up_weight, up_bias, self.R, f"layer.{layer_idx}.ffn_up"
                ),
                "ffn_down": QuantLinear.from_module(
                    block.mlp.c_proj, self.R, f"layer.{layer_idx}.ffn_down"
                ),
            })

        head_weight, head_bias = extract_affine(model.lm_head)
        self.final_ln = EpsilonFreeLayerNorm(
            model.transformer.ln_f.normalized_shape[-1],
            self.R,
            "final_ln.normalized",
        )
        self.final_ln_affine = QuantLayerNormAffine(
            model.transformer.ln_f,
            self.final_ln.output_scale,
            self.R,
            "final_ln.affine",
        )
        self.lm_head = QuantLinear(
            head_weight, head_bias, self.R, "lm_head"
        )

    def _attention_scores(
        self, q, k, head_dim, layer_idx, module, score_mode, diagnostics
    ):
        raw = torch.matmul(
            q.q.to(torch.float64),
            k.q.to(torch.float64).transpose(-1, -2),
        )
        raw_bound = head_dim * self.R * self.R
        _check_bound(raw, raw_bound, f"layer.{layer_idx}.score_wide")

        scale = q.scale * k.scale / math.sqrt(head_dim)
        if getattr(module, "scale_attn_by_inverse_layer_idx", False):
            scale /= layer_idx + 1

        if score_mode == "wide":
            return QTensor(raw, scale, raw_bound)
        if score_mode != "requant":
            raise ValueError(f"unknown score mode: {score_mode}")

        q_score = torch.round(raw / (head_dim * self.R))
        score_scale = scale * head_dim * self.R
        _check_bound(q_score, self.R, f"layer.{layer_idx}.score_requant")

        if diagnostics is not None:
            length = raw.shape[-1]
            causal = torch.tril(
                torch.ones(
                    length, length, dtype=torch.bool, device=raw.device
                )
            ).view(1, 1, length, length).expand_as(raw)
            nonzero = (raw != 0) & causal
            diagnostics.score_nonzero_inputs += int(nonzero.sum().item())
            diagnostics.score_collapsed_to_zero += int(
                ((q_score == 0) & nonzero).sum().item()
            )
            if causal.any():
                error = (
                    q_score * score_scale - raw * scale
                ).abs()[causal]
                diagnostics.score_max_abs_error = max(
                    diagnostics.score_max_abs_error,
                    error.max().item() if error.numel() else 0.0,
                )
        return QTensor(q_score, score_scale, self.R)

    def _softmax_floor(self, scores, diagnostics):
        length = scores.q.shape[-1]
        causal = torch.tril(
            torch.ones(
                length, length, dtype=torch.bool, device=scores.q.device
            )
        ).view(1, 1, length, length)
        score_real = scores.real(torch.float64)
        masked = score_real.masked_fill(~causal, float("-inf"))
        zeta = torch.logsumexp(masked, dim=-1, keepdim=True)
        probabilities = torch.exp(masked - zeta)
        probabilities = torch.where(
            causal, torch.clamp(probabilities, 0.0, 1.0), torch.zeros_like(probabilities)
        )
        alpha_q = torch.floor(probabilities * self.R)
        _check_bound(alpha_q, self.R, "softmax_floor")

        row_mass = alpha_q.sum(dim=-1)
        if (row_mass > self.R).any():
            excess = (row_mass - self.R).max().item()
            raise QuantizationInvariantError(
                f"softmax floor row mass exceeds R by {excess}"
            )

        if diagnostics is not None:
            expanded_causal = causal.expand_as(alpha_q)
            causal_values = alpha_q[expanded_causal]
            diagnostics.softmax_entries += causal_values.numel()
            diagnostics.softmax_zero_entries += int(
                (causal_values == 0).sum().item()
            )
            deficit = self.R - row_mass
            diagnostics.softmax_rows += deficit.numel()
            diagnostics.softmax_mass_deficit_sum += deficit.sum().item()
            diagnostics.softmax_mass_deficit_max = max(
                diagnostics.softmax_mass_deficit_max, deficit.max().item()
            )

        return QTensor(alpha_q, 1.0 / self.R, self.R)

    def _attention_value(
        self,
        alpha,
        value,
        layer_idx,
        diagnostics,
        attention_value_mode="structural",
    ):
        if attention_value_mode not in ATTENTION_VALUE_MODES:
            raise ValueError(
                f"unknown attention-value mode: {attention_value_mode}"
            )
        accumulator = torch.matmul(
            alpha.q.to(torch.float64), value.q.to(torch.float64)
        )
        if attention_value_mode == "structural":
            accumulator_bound = self.R * self.R
        else:
            accumulator_bound = (
                self.max_sequence_length * self.R * self.R
            )
        _check_bound(
            accumulator,
            accumulator_bound,
            f"layer.{layer_idx}.alpha_v_acc.{attention_value_mode}",
        )
        output = torch.round(
            self.R * accumulator / float(accumulator_bound)
        )
        output_scale = (
            alpha.scale * value.scale * accumulator_bound / self.R
        )
        stage_name = f"layer.{layer_idx}.alpha_v.{attention_value_mode}"
        _check_bound(output, self.R, stage_name)
        if diagnostics is not None:
            diagnostics.observe_stage(stage_name, output, self.R)
        return QTensor(output, output_scale, self.R)

    def _gelu(self, x, activation, name, diagnostics):
        real_input = x.real(torch.float64)
        real_output = activation(real_input)
        tolerance = 1e-10 * max(1.0, _max_abs(real_input))
        magnitude_excess = (
            real_output.abs() - real_input.abs()
        ).detach().max().item()
        if magnitude_excess > tolerance:
            raise QuantizationInvariantError(
                f"{name}: GELU violates |g(x)| <= |x| by {magnitude_excess}"
            )
        q_output = torch.round(real_output / x.scale)
        _check_bound(q_output, self.R, name)
        if diagnostics is not None:
            diagnostics.observe_stage(name, q_output, self.R)
        return QTensor(q_output, x.scale, self.R)

    @torch.no_grad()
    def __call__(self, input_ids, configuration,
                 diagnostics: Optional[RunDiagnostics] = None):
        if configuration not in PAPER_CONFIGURATIONS:
            raise ValueError(f"unknown paper configuration: {configuration}")
        config = PAPER_CONFIGURATIONS[configuration]
        scheme = config["scheme"]
        score_mode = config["score_mode"]
        residual_mode = config["residual_mode"]
        attention_value_mode = config["attention_value_mode"]
        layernorm_bound_mode = config["layernorm_bound_mode"]

        batch, length = input_ids.shape
        if length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {length} exceeds {self.max_sequence_length}"
            )

        hidden = self.embedding.lookup(input_ids)

        for layer_idx, state in enumerate(self.blocks):
            block = state["module"]
            residual = hidden

            normalized = state["ln1"](hidden, diagnostics)
            normalized = state["ln1_affine"](
                normalized,
                diagnostics,
                bound_mode=layernorm_bound_mode,
            )
            q_proj = state["q"](normalized, scheme, diagnostics)
            k_proj = state["k"](normalized, scheme, diagnostics)
            v_proj = state["v"](normalized, scheme, diagnostics)

            shape = (
                batch,
                length,
                block.attn.num_heads,
                block.attn.head_dim,
            )
            q_head = QTensor(
                q_proj.q.view(shape).transpose(1, 2),
                q_proj.scale,
                q_proj.bound,
            )
            k_head = QTensor(
                k_proj.q.view(shape).transpose(1, 2),
                k_proj.scale,
                k_proj.bound,
            )
            v_head = QTensor(
                v_proj.q.view(shape).transpose(1, 2),
                v_proj.scale,
                v_proj.bound,
            )

            scores = self._attention_scores(
                q_head,
                k_head,
                block.attn.head_dim,
                layer_idx,
                block.attn,
                score_mode,
                diagnostics,
            )
            alpha = self._softmax_floor(scores, diagnostics)
            attention = self._attention_value(
                alpha,
                v_head,
                layer_idx,
                diagnostics,
                attention_value_mode=attention_value_mode,
            )
            attention_q = (
                attention.q.transpose(1, 2)
                .contiguous()
                .view(batch, length, -1)
            )
            attention = QTensor(
                attention_q, attention.scale, attention.bound
            )

            if residual_mode == "delayed":
                projected = state["attn_proj"].wide_affine(
                    attention, scheme, diagnostics
                )
            else:
                projected = state["attn_proj"](
                    attention, scheme, diagnostics
                )
            hidden = align_residual(
                residual,
                projected,
                f"layer.{layer_idx}.residual1",
                diagnostics,
            )

            residual = hidden
            normalized = state["ln2"](hidden, diagnostics)
            normalized = state["ln2_affine"](
                normalized,
                diagnostics,
                bound_mode=layernorm_bound_mode,
            )
            up = state["ffn_up"](normalized, scheme, diagnostics)
            activated = self._gelu(
                up,
                block.mlp.act,
                f"layer.{layer_idx}.gelu",
                diagnostics,
            )
            if residual_mode == "delayed":
                down = state["ffn_down"].wide_affine(
                    activated, scheme, diagnostics
                )
            else:
                down = state["ffn_down"](
                    activated, scheme, diagnostics
                )
            hidden = align_residual(
                residual,
                down,
                f"layer.{layer_idx}.residual2",
                diagnostics,
            )

        normalized = self.final_ln(hidden, diagnostics)
        normalized = self.final_ln_affine(
            normalized,
            diagnostics,
            bound_mode=layernorm_bound_mode,
        )
        logits = self.lm_head.wide_output(normalized, diagnostics)
        return logits.real(torch.float64)


@torch.no_grad()
def eval_ppl(forward, input_ids, device, desc="PPL"):

    total = input_ids.size(1)
    if total < 2:
        raise ValueError("PPL requires at least two tokens")

    total_nll = 0.0
    total_predicted = 0
    prev_end = 0

    for begin in tqdm(range(0, total, STRIDE), desc=f"  {desc}", leave=False):
        end = min(begin + MAX_LEN, total)
        target_length = end - prev_end
        ids = input_ids[:, begin:end].to(device)
        labels = ids.clone()
        labels[:, :-target_length] = -100

        logits = forward(ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        predicted = int((shift_labels != -100).sum().item())
        total_nll += loss_sum.item()
        total_predicted += predicted
        prev_end = end
        if end == total:
            break

    if total_predicted == 0:
        raise QuantizationInvariantError("PPL evaluation predicted zero tokens")
    return math.exp(total_nll / total_predicted)


def _join_text(dataset):
    return "\n\n".join(text for text in dataset["text"] if text.strip())


def load_wikitext2(tokenizer, cache_dir=None):
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        cache_dir=cache_dir,
    )
    return tokenizer.encode(_join_text(dataset), return_tensors="pt")


def load_wikitext103(tokenizer, target_tokens, cache_dir=None):
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="test",
        cache_dir=cache_dir,
    )
    ids = tokenizer.encode(_join_text(dataset), return_tensors="pt")
    return ids[:, :target_tokens] if ids.size(1) > target_tokens else ids


def load_c4(
    tokenizer,
    target_tokens,
    cache_dir=None,
    prefix_jsonl=None,
):
    if prefix_jsonl is not None:
        texts = []
        with open(prefix_jsonl, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(
                    record.get("text"), str
                ):
                    raise ValueError(
                        f"{prefix_jsonl}:{line_number}: expected JSON object "
                        "with string field 'text'"
                    )
                texts.append(record["text"])
        ids = tokenizer.encode("\n\n".join(texts), return_tensors="pt")
        if ids.size(1) < target_tokens:
            raise RuntimeError(
                f"C4 prefix {prefix_jsonl} produced only {ids.size(1)} "
                f"tokens; need {target_tokens}"
            )
        return ids[:, :target_tokens]

    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True,
        cache_dir=cache_dir,
    )
    texts, approximate_tokens = [], 0
    next_tokenization_check = target_tokens * 1.1
    for example in dataset:
        texts.append(example["text"])
        approximate_tokens += len(example["text"]) // 4
        if approximate_tokens < next_tokenization_check:
            continue
        ids = tokenizer.encode("\n\n".join(texts), return_tensors="pt")
        if ids.size(1) >= target_tokens:
            return ids[:, :target_tokens]
        next_tokenization_check += target_tokens * 0.1
    raise RuntimeError(
        f"C4 stream ended before producing {target_tokens} tokens"
    )


def load_eval_sets(
    tokenizer,
    dataset_arg,
    subset_tokens,
    cache_dir=None,
    c4_prefix_jsonl=None,
):
    wanted = {dataset_arg}
    if dataset_arg == "all":
        wanted = {"wt2", "wt103", "c4"}
    elif "+" in dataset_arg:
        wanted = set(dataset_arg.split("+"))

    output = {}
    if "wt2" in wanted:
        output["WikiText-2"] = load_wikitext2(tokenizer, cache_dir)
    if "wt103" in wanted:
        output["WikiText-103"] = load_wikitext103(
            tokenizer, subset_tokens, cache_dir
        )
    if "c4" in wanted:
        output["C4"] = load_c4(
            tokenizer,
            subset_tokens,
            cache_dir,
            prefix_jsonl=c4_prefix_jsonl,
        )

    for name, ids in output.items():
        print(f"  {name}: {ids.size(1)} tokens")
    return output


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Evaluate GPT-2 perplexity across fixed-point quantization bit widths.')
    parser.add_argument("--model", default="gpt2")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=[
            "wt2",
            "wt103",
            "c4",
            "all",
            "wt2+wt103",
            "wt2+c4",
            "wt2+wt103+c4",
        ],
    )
    parser.add_argument("--subset-tokens", type=int, default=200_000)
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=None,
        help="Optional deterministic prefix limit for smoke tests.",
    )
    parser.add_argument("--Q-list", default="18,20,22")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--c4-prefix-jsonl",
        default=None,
        help=(
            "Optional deterministic C4 validation-prefix JSONL. Each line "
            "must contain a string field named 'text'."
        ),
    )
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--metadata-out", default=None)
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval().to(device)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_sets = load_eval_sets(
        tokenizer,
        args.dataset,
        args.subset_tokens,
        args.cache_dir,
        c4_prefix_jsonl=args.c4_prefix_jsonl,
    )
    if args.max_eval_tokens is not None:
        eval_sets = {
            name: ids[:, : args.max_eval_tokens]
            for name, ids in eval_sets.items()
        }

    baselines = {}
    fp_forward = lambda ids: model(ids, use_cache=False).logits
    for dataset_name, ids in eval_sets.items():
        baselines[dataset_name] = eval_ppl(
            fp_forward, ids, device, f"FP32/{dataset_name}"
        )
        print(f"FP32 {dataset_name}: {baselines[dataset_name]:.6f}")

    q_values = [int(item) for item in args.Q_list.split(",") if item.strip()]
    configurations = [
        {"configuration": name, **config}
        for name, config in PAPER_CONFIGURATIONS.items()
    ]

    rows = []
    for Q in q_values:
        quantized_model = QuantGPT2(
            model,
            Q,
            MAX_LEN,
        )
        print(
            f"Q={Q} LayerNorm=explicit: fused embedding "
            f"scale={quantized_model.embedding.scale:.8g}, "
            f"bound={quantized_model.embedding.bound}"
        )
        for config in configurations:
            configuration = config["configuration"]
            layernorm_bound_mode = config["layernorm_bound_mode"]
            scheme = config["scheme"]
            score_mode = config["score_mode"]
            attention_value_mode = config["attention_value_mode"]
            residual_mode = config["residual_mode"]
            for dataset_name, ids in eval_sets.items():
                diagnostics = RunDiagnostics()
                forward = lambda batch, c=configuration, d=diagnostics: quantized_model(
                    batch, c, d
                )
                label = (
                    f"{configuration}/ln=explicit/"
                    f"lnbound={layernorm_bound_mode}/{scheme}/"
                    f"{score_mode}/{attention_value_mode}/"
                    f"{residual_mode}/Q={Q}/{dataset_name}"
                )
                ppl = eval_ppl(forward, ids, device, label)
                row = {
                    "dataset": dataset_name,
                    "Q": Q,
                    "configuration": configuration,
                    "layernorm_affine_mode": "explicit",
                    "layernorm_bound_mode": layernorm_bound_mode,
                    "scheme": scheme,
                    "score_mode": score_mode,
                    "attention_value_mode": attention_value_mode,
                    "residual_mode": residual_mode,
                    "ppl": f"{ppl:.9f}",
                    "fp_baseline": f"{baselines[dataset_name]:.9f}",
                    "delta_ppl": f"{ppl - baselines[dataset_name]:.9f}",
                    "model": args.model,
                    "execution": "gpt2-fixed-point-ppl",
                }
                row.update(diagnostics.as_row())
                rows.append(row)
                print(
                    f"Q={Q} config={configuration} "
                    f"ln=explicit "
                    f"lnbound={layernorm_bound_mode} {scheme} {score_mode} "
                    f"alphaV={attention_value_mode} {residual_mode} "
                    f"{dataset_name}: PPL={ppl:.6f}, residual_bits="
                    f"{diagnostics.max_residual_bits}, "
                    f"logit_bits={diagnostics.max_logit_bits}"
                )

        del quantized_model
        if device == "cuda":
            torch.cuda.empty_cache()

    if args.csv_out:
        write_csv(args.csv_out, rows)
        print(f"CSV written to: {args.csv_out}")

    if args.metadata_out:
        os.makedirs(os.path.dirname(args.metadata_out) or ".", exist_ok=True)
        with open(args.metadata_out, "w") as handle:
            json.dump(
                {
                    "model": args.model,
                    "datasets": {
                        name: ids.size(1) for name, ids in eval_sets.items()
                    },
                    "Q_list": q_values,
                    "max_eval_tokens": args.max_eval_tokens,
                    "c4_prefix_jsonl": args.c4_prefix_jsonl,
                    "c4_prefix_sha256": (
                        _sha256_file(args.c4_prefix_jsonl)
                        if args.c4_prefix_jsonl
                        else None
                    ),
                    "experiment_presets": list(PAPER_CONFIGURATIONS),
                    "resolved_configurations": configurations,
                    "schemes": sorted({c["scheme"] for c in configurations}),
                    "score_modes": sorted(
                        {c["score_mode"] for c in configurations}
                    ),
                    "attention_value_modes": sorted(
                        {c["attention_value_mode"] for c in configurations}
                    ),
                    "residual_modes": sorted(
                        {c["residual_mode"] for c in configurations}
                    ),
                    "layernorm_bound_modes": sorted(
                        {c["layernorm_bound_mode"] for c in configurations}
                    ),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                },
                handle,
                indent=2,
            )
        print(f"Metadata written to: {args.metadata_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"FATAL: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
