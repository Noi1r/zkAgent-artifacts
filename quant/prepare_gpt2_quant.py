#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


FORMAT = "zkagent_quant_plan"
ATTENTION_MIDDLE_BITS = 11
ATTENTION_LOW_EXP_TARGET = 7.952219286828541e-10 * float(1 << 23)


class QuantPlanError(RuntimeError):
    pass


def _read_config(path: Path) -> Dict[str, int]:
    if not path.is_file():
        raise QuantPlanError(f"missing model config: {path}")
    result: Dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise QuantPlanError(f"invalid config line: {raw!r}")
        key, value = line.split("=", 1)
        result[key.strip()] = int(value.strip())
    required = ("vocab_size", "n_embd", "n_layer", "n_head", "ffn_dim")
    missing = [key for key in required if key not in result]
    if missing:
        raise QuantPlanError(f"config is missing keys: {', '.join(missing)}")
    return result


def _read_f32(path: Path, count: Optional[int] = None) -> np.ndarray:
    if not path.is_file():
        raise QuantPlanError(f"missing parameter file: {path}")
    values = np.fromfile(path, dtype="<f4")
    if count is not None and values.size != count:
        raise QuantPlanError(
            f"{path}: expected {count} float32 values, found {values.size}"
        )
    if not np.isfinite(values).all():
        raise QuantPlanError(f"{path}: non-finite parameter")
    return values.astype(np.float64)


def _round_ties_even(values: np.ndarray) -> np.ndarray:

    rounded = np.rint(values)
    if not np.isfinite(rounded).all():
        raise QuantPlanError("non-finite quantized parameter")
    return rounded


@dataclass
class QuantizedParameter:
    q: np.ndarray
    scale: float


def _quantize_parameter(values: np.ndarray, R: int, name: str) -> QuantizedParameter:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    if maximum == 0.0:
        return QuantizedParameter(np.zeros(values.shape, dtype=np.int32), 1.0)
    scale = maximum / float(R)
    q64 = _round_ties_even(values / scale)
    actual = float(np.max(np.abs(q64))) if q64.size else 0.0
    if actual > R:
        raise QuantPlanError(f"{name}: |q|={actual:.0f} exceeds R={R}")
    if actual > np.iinfo(np.int32).max:
        raise QuantPlanError(f"{name}: quantized parameter does not fit int32")
    return QuantizedParameter(q64.astype(np.int32), scale)


def _ceil_log2_ratio(numerator: int, denominator: int) -> int:

    if numerator <= 0 or denominator <= 0:
        raise QuantPlanError("ceil_log2_ratio requires positive integers")
    if numerator <= denominator:
        return 0
    n = 0
    value = denominator
    while value < numerator:
        value <<= 1
        n += 1
    return n


@dataclass(frozen=True)
class Dyadic:
    c1: int
    c2: int
    n: int
    range_bits: int


def _linear_under_approx(B: int, R: int, Q: int) -> Dyadic:

    if B <= 0:
        raise QuantPlanError("linear bound must be positive")
    if B >= R:
        n = _ceil_log2_ratio(B, R)
        exponent = n + Q - 1
        if exponent < 0 or exponent >= 62:
            raise QuantPlanError(f"C2 exponent {exponent} does not fit int64")
        c2 = 1 << exponent
    else:
        n = 0
        c2 = R
    c1 = (R * c2) // B
    if c1 <= 0:
        raise QuantPlanError("under-approximation produced C1=0")
    if c1 >= (1 << 62) or c2 >= (1 << 62):
        raise QuantPlanError("dyadic coefficient does not fit signed int64")
    if c1 * B > R * c2:
        raise QuantPlanError("dyadic multiplier is not a lower approximation")
    range_bits = (2 * c2 - 1).bit_length()
    return Dyadic(c1=c1, c2=c2, n=n, range_bits=range_bits)


def _ratio_under_approx(source_scale: float, target_scale: float, Q: int) -> Dyadic:

    if not (math.isfinite(source_scale) and source_scale > 0.0):
        raise QuantPlanError(f"invalid source scale {source_scale}")
    if not (math.isfinite(target_scale) and target_scale > 0.0):
        raise QuantPlanError(f"invalid target scale {target_scale}")
    ratio = source_scale / target_scale
    numerator, denominator = ratio.as_integer_ratio()
    c2 = 1 << Q
    c1 = (numerator * c2) // denominator
    if c1 <= 0:
        raise QuantPlanError("scale alignment produced C1=0")
    if c1 >= (1 << 62):
        raise QuantPlanError("scale-alignment C1 does not fit signed int64")
    if c1 * denominator > numerator * c2:
        raise QuantPlanError("scale alignment is not a lower approximation")
    return Dyadic(c1=c1, c2=c2, n=0, range_bits=Q + 1)


def _round_bound(bound: int, dyadic: Dyadic) -> int:

    if bound < 0:
        raise QuantPlanError("negative magnitude bound")
    return (dyadic.c1 * bound + dyadic.c2 // 2) // dyadic.c2


def _nonlinear_dyadic(value: float, precision_bits: int) -> Tuple[int, int, float]:
    if not (math.isfinite(value) and value > 0.0) or not 1 <= precision_bits <= 30:
        raise QuantPlanError("invalid nonlinear scale or coefficient precision")
    exponent = math.frexp(value)[1] - precision_bits
    numerator, denominator = value.as_integer_ratio()
    if exponent < 0:
        numerator <<= -exponent
    else:
        denominator <<= exponent
    coefficient, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and coefficient & 1):
        coefficient += 1
    while coefficient % 2 == 0:
        coefficient //= 2
        exponent += 1
    try:
        represented = math.ldexp(float(coefficient), exponent)
    except OverflowError as exc:
        raise QuantPlanError("nonlinear represented scale overflows binary64") from exc
    return exponent, coefficient, represented


def _check_gate_coefficients(*coefficients: int) -> None:
    if any(abs(c) >= (1 << 126) for c in coefficients):
        raise QuantPlanError("nonlinear gate coefficient exceeds 126 bits")


def _check_field_bound(bound: int) -> None:
    if bound >= (1 << 253):
        raise QuantPlanError("nonlinear worst-case intermediate may wrap BN254 Fr")


def _layernorm_intermediate_bound(width, input_bound, e1, c1, e2, c2, gamma, beta):
    m = max(1, -e1, -e2)
    if min(e1, e2) < -125 or max(e1, e2) > 125 or max(m, e1 + m, e2 + m) >= 126:
        raise QuantPlanError("LayerNorm gate shift exceeds supported integer width")
    a, b, denominator = c1 << (e1 + m), c2 << (e2 + m), 1 << m
    _check_gate_coefficients(a, b, denominator)
    centered = 2 * width * input_bound
    variance = 1 + width * centered**2
    sigma = math.isqrt(variance) + 1
    numerator = a * int(np.max(np.abs(gamma.astype(object)), initial=0)) * centered
    numerator += b * int(np.max(np.abs(beta.astype(object)), initial=0)) * sigma
    _check_field_bound(2 * numerator + denominator * sigma + 2)


def _exclusive_bound_bits(upper_bound_exclusive: int) -> int:
    if upper_bound_exclusive <= 1:
        raise QuantPlanError("exclusive range bound must exceed one")
    return int(upper_bound_exclusive - 1).bit_length()


def _layernorm_range_bits(
    width: int, input_bound: int, term1_e: int, term2_e: int
) -> Tuple[int, int, int]:

    if width <= 1 or input_bound <= 0:
        raise QuantPlanError("LayerNorm range planning requires positive dimensions and bound")
    centered_bound = 2 * width * input_bound
    variance_bound = 1 + width * centered_bound * centered_bound
    if variance_bound >= (1 << 253):
        raise QuantPlanError("LayerNorm worst-case variance may wrap the BN254 field")
    variance_bits = variance_bound.bit_length()
    sigma_max = math.isqrt(variance_bound)
    if sigma_max * sigma_max < variance_bound:
        sigma_max += 1
    sqrt_residual_upper = 2 * sigma_max
    m = max(1, -term1_e, -term2_e)
    rounding_residual_upper = (1 << m) * sigma_max + 2
    if rounding_residual_upper > (1 << 126):
        raise QuantPlanError("LayerNorm rounding residual exceeds supported range")
    return (
        _exclusive_bound_bits(sqrt_residual_upper),
        _exclusive_bound_bits(rounding_residual_upper),
        variance_bits,
    )


def _gelu_range_bits(input_bound: int, scale_e: int, scale_c: int) -> Tuple[int, int, int]:
    ea, ca = -8, 48
    eb, cb = -8, 217
    ec, cc = -8, 252
    ed, cd = -8, 615
    ex, cx = scale_e, scale_c
    ey, cy = scale_e, scale_c
    m1 = max(0, -ec, -eb - ex, -ea - 2 * ex, ex - ey)
    m2 = max(0, -ed, -ex)
    shifts = (m1, ea + 2 * ex + m1, eb + ex + m1, ec + m1,
              ey - ex + m1, ed + m2, ex + m2)
    if min(shifts) < 0 or max(shifts) >= 126:
        raise QuantPlanError("GELU gate shift exceeds supported integer width")
    c1 = cx << m1
    c2 = (1 << (ea + 2 * ex + m1)) * cx * ca * cx * cx
    c3 = (1 << (eb + ex + m1)) * cx * cb * cx
    c4 = cx * cc * (1 << (ec + m1))
    c5 = (1 << (ey - ex + m1)) * cy
    c6 = cd * (1 << (ed + m2))
    c7 = cx * (1 << (ex + m2))

    _check_gate_coefficients(c1, c2, c3, c4, 2*c5+1, 2*c6+1, 2*c7)
    numerator = 2*c1*input_bound + c2*input_bound**3 + c3*input_bound**2 + c4*input_bound
    _check_field_bound(2*numerator + 2*c5 + 2)
    d1_max = input_bound + 1
    d2_max = max(c6 + 1, c7 * input_bound - c6, 1)
    rounding_max = 2 * c5 + 1
    if max(d1_max, d2_max, rounding_max) >= (1 << 126):
        raise QuantPlanError("GELU residual exceeds supported range")
    return d1_max.bit_length(), d2_max.bit_length(), rounding_max.bit_length()


class ArtifactWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.values: Dict[str, str] = {}
        self.tensor_hashes: Dict[str, str] = {}

    def put(self, key: str, value) -> None:
        if key in self.values:
            raise QuantPlanError(f"duplicate manifest key: {key}")
        if isinstance(value, float):
            if not math.isfinite(value):
                raise QuantPlanError(f"{key}: non-finite manifest value")
            encoded = value.hex()
        elif isinstance(value, bool):
            encoded = "1" if value else "0"
        else:
            encoded = str(value)
        if "\n" in encoded or "=" in key:
            raise QuantPlanError(f"invalid manifest entry {key}")
        self.values[key] = encoded

    def tensor(self, key: str, relative_path: str, values: np.ndarray, dtype: str) -> None:
        path = self.output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.asarray(values, dtype=dtype)
        array.tofile(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.tensor_hashes[relative_path] = digest
        self.put(f"{key}.file", relative_path)
        self.put(f"{key}.count", int(array.size))
        self.put(f"{key}.sha256", digest)

    def finish(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.output_dir / "quant_plan.txt"
        lines = [f"{key}={self.values[key]}" for key in sorted(self.values)]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        digest_lines = [
            f"{hashlib.sha256(manifest.read_bytes()).hexdigest()}  quant_plan.txt"
        ]
        digest_lines.extend(
            f"{self.tensor_hashes[path]}  {path}" for path in sorted(self.tensor_hashes)
        )
        (self.output_dir / "SHA256SUMS").write_text(
            "\n".join(digest_lines) + "\n", encoding="utf-8"
        )
        return manifest


@dataclass
class LinearResult:
    name: str
    input_scale: float
    input_bound: int
    weight_scale: float
    accumulator_scale: float
    accumulator_bound: int
    output_scale: float
    output_bound: int
    dyadic: Dyadic
    mode: str


class Planner:
    def __init__(self, model_dir: Path, output_dir: Path, Q: int):
        if Q <= 0 or Q > 22:
            raise QuantPlanError("Q must be in [1,22]")
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.config = _read_config(model_dir / "config.txt")
        self.Q = Q
        self.R = 1 << Q
        self.vocab = self.config["vocab_size"]
        self.hidden = self.config["n_embd"]
        self.layers = self.config["n_layer"]
        self.heads = self.config["n_head"]
        self.ffn = self.config["ffn_dim"]
        if self.hidden % self.heads:
            raise QuantPlanError("n_embd must be divisible by n_head")
        self.head_dim = self.hidden // self.heads
        self.writer = ArtifactWriter(output_dir)
        self.positions = 0

    @staticmethod
    def _split_bits(total: int, parts: int, cap: int) -> Tuple[int, ...]:
        if total < 0 or parts <= 0 or total > parts * cap:
            raise QuantPlanError("cannot split attention digit width")
        quotient, remainder = divmod(total, parts)
        result = tuple(quotient + (1 if i < remainder else 0)
                       for i in range(parts))
        if any(bits > cap for bits in result):
            raise QuantPlanError("attention digit exceeds lookup width cap")
        return result

    def _write_attention(
        self,
        block: int,
        q: LinearResult,
        k: LinearResult,
    ) -> int:
        if self.positions <= 0:
            raise QuantPlanError("attention planning requires position count")
        theta = self.R
        score_scale = (
            q.output_scale * k.output_scale / math.sqrt(float(self.head_dim))
        )
        if not (math.isfinite(score_scale) and score_scale > 0.0):
            raise QuantPlanError(f"block {block}: invalid attention score scale")
        score_bound = self.head_dim * self.R * self.R
        shift_extra = int(math.ceil(math.log(float(self.positions)) / score_scale)) + 1
        shift_upper_bound = 2 * score_bound + shift_extra
        if shift_upper_bound >= (1 << 62):
            raise QuantPlanError(f"block {block}: attention shift bound does not fit int64")
        total_bits = max(int(shift_upper_bound).bit_length(), ATTENTION_MIDDLE_BITS)
        desired_low = int(math.floor(math.log2(
            ATTENTION_LOW_EXP_TARGET / score_scale)))
        min_low = max(0, total_bits - ATTENTION_MIDDLE_BITS - 20)
        max_low = min(33, max(0, total_bits - ATTENTION_MIDDLE_BITS))
        low_bits = max(min_low, min(max_low, desired_low))
        high_bits = max(0, total_bits - low_bits - ATTENTION_MIDDLE_BITS)
        low_parts = self._split_bits(low_bits, 3, 11)
        high_parts = self._split_bits(high_bits, 2, 10)
        digit_bits = low_parts + (ATTENTION_MIDDLE_BITS,) + high_parts
        radix_bits = sum(digit_bits)
        if shift_upper_bound >= (1 << radix_bits):
            raise QuantPlanError(f"block {block}: attention digit domain is too small")

        low_boundary = 1 << low_bits
        high_boundary = 1 << (low_bits + ATTENTION_MIDDLE_BITS)
        epsilons = []
        for active_count in range(1, self.positions + 1):
            shift_and_low = theta * (
                math.exp(score_scale * (float(low_boundary - 1) + 0.5)) - 1.0
            )
            table_rounding = active_count / 2.0
            high_tail = (
                active_count * theta *
                math.exp(-score_scale * (float(high_boundary) - 0.5))
            )
            epsilon = int(math.ceil(shift_and_low + table_rounding + high_tail))
            if epsilon < 0 or epsilon >= theta:
                raise QuantPlanError(
                    f"block {block}: attention epsilon {epsilon} is outside [0,theta)"
                )
            epsilons.append(epsilon)

        epsilon_max = max(epsilons)
        mass_bound = theta + epsilon_max
        accumulator_bound = self.R * mass_bound
        output_bound = (
            accumulator_bound + theta - 1
        ) // theta
        prefix = f"attention.block.{block}"
        self.writer.put(f"{prefix}.score_scale", score_scale)
        self.writer.put(f"{prefix}.score_bound", score_bound)
        self.writer.put(f"{prefix}.score_bits", score_bound.bit_length())
        self.writer.put(f"{prefix}.theta", theta)
        self.writer.put(f"{prefix}.shift_upper_bound", shift_upper_bound)
        self.writer.put(f"{prefix}.shift_bits", shift_upper_bound.bit_length())
        self.writer.put(f"{prefix}.least_segments", 3)
        self.writer.put(f"{prefix}.nonlinear_segment", 3)
        self.writer.put(f"{prefix}.most_zero_segments", 2)
        self.writer.put(f"{prefix}.mass_bits", mass_bound.bit_length())
        self.writer.put(f"{prefix}.accumulator_bound", accumulator_bound)
        self.writer.put(
            f"{prefix}.accumulator_bits", accumulator_bound.bit_length()
        )
        self.writer.put(f"{prefix}.rounding_bits", self.Q + 1)
        self.writer.put(f"{prefix}.output_bound", output_bound)
        self.writer.put(f"{prefix}.output_bits", output_bound.bit_length())
        for i, bits in enumerate(digit_bits):
            self.writer.put(f"{prefix}.digit{i}_bits", bits)
        self.writer.tensor(
            f"{prefix}.epsilon",
            f"tensors/attention_block_{block}_epsilon_i64.bin",
            np.asarray(epsilons, dtype=np.int64),
            "<i8",
        )
        return output_bound

    def _write_linear(
        self,
        name: str,
        weight: np.ndarray,
        bias: Optional[np.ndarray],
        input_scale: float,
        input_bound: int,
        mode: str,
    ) -> LinearResult:
        if mode not in ("immediate", "wide"):
            raise QuantPlanError(f"{name}: invalid linear mode {mode}")
        if weight.ndim != 2:
            raise QuantPlanError(f"{name}: weight must be [out,in]")
        out_dim, in_dim = weight.shape
        if bias is not None and bias.shape != (out_dim,):
            raise QuantPlanError(f"{name}: bias shape mismatch")

        quantized = _quantize_parameter(weight, self.R, f"{name}.weight")
        accumulator_scale = input_scale * quantized.scale
        if not (math.isfinite(accumulator_scale) and accumulator_scale > 0.0):
            raise QuantPlanError(f"{name}: invalid accumulator scale")
        if bias is None:
            bias_q = np.zeros(out_dim, dtype=np.int64)
        else:
            bias_rounded = _round_ties_even(bias / accumulator_scale)
            if np.max(np.abs(bias_rounded), initial=0.0) > np.iinfo(np.int64).max:
                raise QuantPlanError(f"{name}: bias does not fit int64")
            bias_q = bias_rounded.astype(np.int64)

        l1 = np.abs(quantized.q.astype(np.int64)).sum(axis=1, dtype=np.int64)
        per_output = l1.astype(object) * int(input_bound) + np.abs(bias_q).astype(object)
        bound = max(int(max(per_output, default=1)), 1)
        if bound >= (1 << 62):
            raise QuantPlanError(f"{name}: accumulator bound does not fit int64")
        output_scale = accumulator_scale * float(bound) / float(self.R)
        dyadic = _linear_under_approx(bound, self.R, self.Q)

        prefix = f"linear.{name}"
        self.writer.put(f"{prefix}.in_dim", in_dim)
        self.writer.put(f"{prefix}.out_dim", out_dim)
        self.writer.put(f"{prefix}.input_scale", input_scale)
        self.writer.put(f"{prefix}.input_bound", input_bound)
        self.writer.put(f"{prefix}.weight_scale", quantized.scale)
        self.writer.put(f"{prefix}.accumulator_scale", accumulator_scale)
        self.writer.put(f"{prefix}.accumulator_bound", bound)
        self.writer.put(f"{prefix}.output_scale", output_scale)
        self.writer.put(f"{prefix}.output_bound", self.R)
        self.writer.put(f"{prefix}.mode", mode)
        self.writer.put(f"{prefix}.c1", dyadic.c1)
        self.writer.put(f"{prefix}.c2", dyadic.c2)
        self.writer.put(f"{prefix}.n", dyadic.n)
        self.writer.put(f"{prefix}.range_bits", dyadic.range_bits)
        file_prefix = name.replace(".", "/")
        self.writer.tensor(
            f"{prefix}.weight", f"tensors/{file_prefix}_weight_i32.bin",
            quantized.q, "<i4",
        )
        self.writer.tensor(
            f"{prefix}.bias", f"tensors/{file_prefix}_bias_i64.bin",
            bias_q, "<i8",
        )
        return LinearResult(
            name=name,
            input_scale=input_scale,
            input_bound=input_bound,
            weight_scale=quantized.scale,
            accumulator_scale=accumulator_scale,
            accumulator_bound=bound,
            output_scale=output_scale,
            output_bound=self.R,
            dyadic=dyadic,
            mode=mode,
        )

    def _write_layernorm(
        self, name: str, gamma: np.ndarray, beta: np.ndarray,
        input_bound: int,
    ) -> Tuple[float, int]:
        if gamma.shape != (self.hidden,) or beta.shape != (self.hidden,):
            raise QuantPlanError(f"{name}: LayerNorm shape mismatch")
        normalized_scale = math.sqrt(self.hidden - 1.0) / float(self.R)
        gamma_qp = _quantize_parameter(gamma, self.R, f"{name}.gamma")
        beta_scale = normalized_scale * gamma_qp.scale
        if not (math.isfinite(beta_scale) and beta_scale > 0.0):
            raise QuantPlanError(f"{name}: invalid beta accumulator scale")
        beta_q_f = _round_ties_even(beta / beta_scale)
        if np.max(np.abs(beta_q_f), initial=0.0) > np.iinfo(np.int64).max:
            raise QuantPlanError(f"{name}: beta does not fit int64")
        beta_q = beta_q_f.astype(np.int64)
        coordinate_bounds = (
            np.abs(gamma_qp.q.astype(np.int64)).astype(object) * self.R
            + np.abs(beta_q).astype(object)
        )
        accumulator_bound = max(int(max(coordinate_bounds, default=1)), 1)
        output_scale = beta_scale * float(accumulator_bound) / float(self.R)

        term1 = gamma_qp.scale * math.sqrt(float(self.hidden)) / output_scale
        term2 = beta_scale / output_scale
        e1, c1, represented1 = _nonlinear_dyadic(term1, self.Q)
        e2, c2, represented2 = _nonlinear_dyadic(term2, self.Q)
        _layernorm_intermediate_bound(
            self.hidden, input_bound, e1, c1, e2, c2, gamma_qp.q, beta_q
        )
        sigma_range_bits, rounding_range_bits, variance_bits = _layernorm_range_bits(
            self.hidden, input_bound, e1, e2
        )
        q_sigma_range_bits, q_rounding_range_bits, q_variance_bits = _layernorm_range_bits(
            self.hidden, self.R, e1, e2
        )

        prefix = f"ln.{name}"
        self.writer.put(f"{prefix}.width", self.hidden)
        self.writer.put(f"{prefix}.input_bound", input_bound)
        self.writer.put(f"{prefix}.normalized_scale", normalized_scale)
        self.writer.put(f"{prefix}.gamma_scale", gamma_qp.scale)
        self.writer.put(f"{prefix}.beta_scale", beta_scale)
        self.writer.put(f"{prefix}.accumulator_bound", accumulator_bound)
        self.writer.put(f"{prefix}.output_scale", output_scale)
        self.writer.put(f"{prefix}.output_bound", self.R)
        self.writer.put(f"{prefix}.term1_c", c1)
        self.writer.put(f"{prefix}.term1_e", e1)
        self.writer.put(f"{prefix}.term1_represented", represented1)
        self.writer.put(f"{prefix}.term2_c", c2)
        self.writer.put(f"{prefix}.term2_e", e2)
        self.writer.put(f"{prefix}.term2_represented", represented2)
        self.writer.put(f"{prefix}.sigma_range_bits", sigma_range_bits)
        self.writer.put(f"{prefix}.rounding_range_bits", rounding_range_bits)
        self.writer.put(f"{prefix}.variance_bits", variance_bits)
        self.writer.put(f"{prefix}.q_input_sigma_range_bits", q_sigma_range_bits)
        self.writer.put(f"{prefix}.q_input_rounding_range_bits", q_rounding_range_bits)
        self.writer.put(f"{prefix}.q_input_variance_bits", q_variance_bits)
        file_prefix = name.replace(".", "/")
        self.writer.tensor(
            f"{prefix}.gamma", f"tensors/{file_prefix}_gamma_i32.bin",
            gamma_qp.q, "<i4",
        )
        self.writer.tensor(
            f"{prefix}.beta", f"tensors/{file_prefix}_beta_i64.bin",
            beta_q, "<i8",
        )
        return output_scale, self.R

    def _write_residual(
        self,
        name: str,
        left_scale: float,
        left_bound: int,
        right_scale: float,
        right_bound: int,
    ) -> Tuple[float, int]:
        output_scale = min(left_scale, right_scale)
        left = _ratio_under_approx(left_scale, output_scale, self.Q)
        right = _ratio_under_approx(right_scale, output_scale, self.Q)
        left_out_bound = _round_bound(left_bound, left)
        right_out_bound = _round_bound(right_bound, right)
        output_bound = left_out_bound + right_out_bound
        if output_bound >= (1 << 62):
            raise QuantPlanError(f"{name}: residual bound does not fit int64")

        prefix = f"residual.{name}"
        self.writer.put(f"{prefix}.left_scale", left_scale)
        self.writer.put(f"{prefix}.left_bound", left_bound)
        self.writer.put(f"{prefix}.right_scale", right_scale)
        self.writer.put(f"{prefix}.right_bound", right_bound)
        self.writer.put(f"{prefix}.output_scale", output_scale)
        self.writer.put(f"{prefix}.output_bound", output_bound)
        for side, value in (("left", left), ("right", right)):
            self.writer.put(f"{prefix}.{side}_c1", value.c1)
            self.writer.put(f"{prefix}.{side}_c2", value.c2)
            self.writer.put(f"{prefix}.{side}_range_bits", value.range_bits)
        return output_scale, output_bound

    def _embedding(self) -> Tuple[float, int, np.ndarray]:
        wte = _read_f32(self.model_dir / "wte.bin", self.vocab * self.hidden).reshape(
            self.vocab, self.hidden
        )
        wpe_values = _read_f32(self.model_dir / "wpe.bin")
        if wpe_values.size % self.hidden:
            raise QuantPlanError("wpe.bin size is not divisible by n_embd")
        positions = wpe_values.size // self.hidden
        self.positions = int(positions)
        wpe = wpe_values.reshape(positions, self.hidden)
        real_max = wte.max(axis=0) + wpe.max(axis=0)
        real_min = wte.min(axis=0) + wpe.min(axis=0)
        bound_real = float(np.max(np.maximum(np.abs(real_max), np.abs(real_min))))
        scale = 1.0 if bound_real == 0.0 else bound_real / float(self.R)
        token_q = _round_ties_even(wte / scale).astype(np.int32)
        position_q = _round_ties_even(wpe / scale).astype(np.int32)
        quant_max = token_q.astype(np.int64).max(axis=0) + position_q.astype(np.int64).max(axis=0)
        quant_min = token_q.astype(np.int64).min(axis=0) + position_q.astype(np.int64).min(axis=0)
        bound = int(np.max(np.maximum(np.abs(quant_max), np.abs(quant_min))))

        self.writer.put("embedding.positions", positions)
        self.writer.put("embedding.scale", scale)
        self.writer.put("embedding.bound", bound)
        self.writer.tensor(
            "embedding.token", "tensors/embedding_token_i32.bin", token_q, "<i4"
        )
        self.writer.tensor(
            "embedding.position", "tensors/embedding_position_i32.bin",
            position_q, "<i4",
        )
        return scale, bound, wte

    def run(self) -> Path:
        w = self.writer
        w.put("format", FORMAT)
        w.put("nonlinear.coefficient_bits", self.Q)
        w.put("nonlinear.coefficient_rounding", "nearest_even")
        w.put("rounding.nonlinear", "nearest_away")
        w.put("q_bits", self.Q)
        w.put("R", self.R)
        w.put("rounding.parameter", "ties_to_even")
        w.put("rounding.activation", "floor_plus_half")
        w.put("dyadic.mode", "under_approx_floor")
        w.put("scheme.linear", "column_l1")
        w.put("scheme.score", "wide")
        w.put("scheme.attention_softmax", "zkllm_digits")
        w.put("scheme.attention_value", "division_free_fixed_theta")
        w.put("scheme.residual", "delayed")
        w.put("scheme.lm_head", "wide")
        w.put("model.vocab_size", self.vocab)
        w.put("model.positions", self.config.get("n_positions", self.config.get("block_size", 0)))
        w.put("model.hidden", self.hidden)
        w.put("model.layers", self.layers)
        w.put("model.heads", self.heads)
        w.put("model.head_dim", self.head_dim)
        w.put("model.ffn", self.ffn)

        residual_scale, residual_bound, real_wte = self._embedding()

        for block in range(self.layers):
            layer_dir = self.model_dir / f"layer_{block}"
            ln1_gamma = _read_f32(layer_dir / "ln_1_w.bin", self.hidden)
            ln1_beta = _read_f32(layer_dir / "ln_1_b.bin", self.hidden)
            ln1_scale, ln1_bound = self._write_layernorm(
                f"block.{block}.ln1", ln1_gamma, ln1_beta, residual_bound
            )

            qkv_w = _read_f32(
                layer_dir / "c_attn_w.bin", 3 * self.hidden * self.hidden
            ).reshape(3 * self.hidden, self.hidden)
            qkv_b = _read_f32(layer_dir / "c_attn_b.bin", 3 * self.hidden)
            q_w, k_w, v_w = np.split(qkv_w, 3, axis=0)
            q_b, k_b, v_b = np.split(qkv_b, 3)
            q = self._write_linear(
                f"block.{block}.q", q_w, q_b, ln1_scale, ln1_bound, "immediate"
            )
            k = self._write_linear(
                f"block.{block}.k", k_w, k_b, ln1_scale, ln1_bound, "immediate"
            )
            v = self._write_linear(
                f"block.{block}.v", v_w, v_b, ln1_scale, ln1_bound, "immediate"
            )
            w.put(f"attention.block.{block}.q_scale", q.output_scale)
            w.put(f"attention.block.{block}.k_scale", k.output_scale)
            w.put(f"attention.block.{block}.v_scale", v.output_scale)
            w.put(f"attention.block.{block}.value_output_scale", v.output_scale)
            attention_output_bound = self._write_attention(block, q, k)
            w.put(
                f"attention.block.{block}.value_output_bound",
                attention_output_bound,
            )

            proj_w = _read_f32(
                layer_dir / "c_proj_w.bin", self.hidden * self.hidden
            ).reshape(self.hidden, self.hidden)
            proj_b = _read_f32(layer_dir / "c_proj_b.bin", self.hidden)
            projection = self._write_linear(
                f"block.{block}.attn_proj", proj_w, proj_b,
                v.output_scale, attention_output_bound, "wide",
            )
            residual_scale, residual_bound = self._write_residual(
                f"block.{block}.attn",
                residual_scale,
                residual_bound,
                projection.accumulator_scale,
                projection.accumulator_bound,
            )

            ln2_gamma = _read_f32(layer_dir / "ln_2_w.bin", self.hidden)
            ln2_beta = _read_f32(layer_dir / "ln_2_b.bin", self.hidden)
            ln2_scale, ln2_bound = self._write_layernorm(
                f"block.{block}.ln2", ln2_gamma, ln2_beta, residual_bound
            )

            up_w = _read_f32(
                layer_dir / "c_fc_w.bin", self.ffn * self.hidden
            ).reshape(self.ffn, self.hidden)
            up_b = _read_f32(layer_dir / "c_fc_b.bin", self.ffn)
            up = self._write_linear(
                f"block.{block}.ffn_up", up_w, up_b,
                ln2_scale, ln2_bound, "immediate",
            )
            gelu_e, gelu_c, gelu_repr = _nonlinear_dyadic(up.output_scale, self.Q)
            w.put(f"gelu.block.{block}.input_scale", up.output_scale)
            w.put(f"gelu.block.{block}.scale_e", gelu_e)
            w.put(f"gelu.block.{block}.scale_c", gelu_c)
            w.put(f"gelu.block.{block}.represented_scale", gelu_repr)
            w.put(f"gelu.block.{block}.output_scale", up.output_scale)
            w.put(f"gelu.block.{block}.output_bound", self.R)
            gelu_d1_bits, gelu_d2_bits, gelu_rounding_bits = _gelu_range_bits(
                self.R, gelu_e, gelu_c
            )
            w.put(f"gelu.block.{block}.d1_range_bits", gelu_d1_bits)
            w.put(f"gelu.block.{block}.d2_range_bits", gelu_d2_bits)
            w.put(f"gelu.block.{block}.rounding_range_bits", gelu_rounding_bits)

            down_w = _read_f32(
                layer_dir / "mlp_proj_w.bin", self.hidden * self.ffn
            ).reshape(self.hidden, self.ffn)
            down_b = _read_f32(layer_dir / "mlp_proj_b.bin", self.hidden)
            down = self._write_linear(
                f"block.{block}.ffn_down", down_w, down_b,
                up.output_scale, self.R, "wide",
            )
            residual_scale, residual_bound = self._write_residual(
                f"block.{block}.ffn",
                residual_scale,
                residual_bound,
                down.accumulator_scale,
                down.accumulator_bound,
            )

        final_gamma = _read_f32(self.model_dir / "ln_f_w.bin", self.hidden)
        final_beta = _read_f32(self.model_dir / "ln_f_b.bin", self.hidden)
        final_scale, final_bound = self._write_layernorm(
            "final", final_gamma, final_beta, residual_bound
        )
        head = self._write_linear(
            "lm_head", real_wte, None, final_scale, final_bound, "wide"
        )
        w.put("lm_head.logit_scale", head.accumulator_scale)
        w.put("lm_head.logit_bound", head.accumulator_bound)

        return w.finish()


def build_plan(model_dir: Path, output_dir: Optional[Path] = None, Q: int = 20) -> Path:
    model_dir = Path(model_dir)
    output_dir = Path(output_dir) if output_dir is not None else model_dir / "quantized"
    return Planner(model_dir, output_dir, Q).run()


def main() -> int:
    parser = argparse.ArgumentParser(description='Build the input-independent GPT-2 quantization plan.')
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--Q", type=int, default=20)
    args = parser.parse_args()
    manifest = build_plan(args.model_dir, args.output_dir, args.Q)
    print(f"quantization plan written to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
