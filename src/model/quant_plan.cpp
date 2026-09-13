#include "model/quant_plan.hpp"
#include "model/nonlinear_quant.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <iterator>
#include <sstream>
#include <type_traits>
#include <cctype>

namespace
{

    std::string joinPath(const std::string &left, const std::string &right)
    {
        if (left.empty())
            return right;
        if (left.back() == '/')
            return left + right;
        return left + "/" + right;
    }

    bool parseSigned(const std::string &text, long long &value)
    {
        if (text.empty())
            return false;
        char *end = nullptr;
        errno = 0;
        long long parsed = std::strtoll(text.c_str(), &end, 10);
        if (errno != 0 || end == text.c_str() || *end != '\0')
            return false;
        value = parsed;
        return true;
    }

    bool parseFloat(const std::string &text, double &value)
    {
        if (text.empty())
            return false;
        char *end = nullptr;
        errno = 0;
        double parsed = std::strtod(text.c_str(), &end);
        if (errno != 0 || end == text.c_str() || *end != '\0' || !std::isfinite(parsed))
        {
            return false;
        }
        value = parsed;
        return true;
    }

}

bool GPT2QuantPlan::fail(const std::string &message, std::string *error) const
{
    if (error)
        *error = message;
    return false;
}

bool GPT2QuantPlan::parseManifest(const std::string &path, std::string *error)
{
    std::ifstream input(path);
    if (!input)
        return fail("cannot open quantization manifest: " + path, error);
    values_.clear();
    std::string line;
    int line_no = 0;
    while (std::getline(input, line))
    {
        ++line_no;
        if (line.empty() || line[0] == '#')
            continue;
        const std::size_t split = line.find('=');
        if (split == std::string::npos || split == 0)
        {
            return fail("invalid quantization manifest line " + std::to_string(line_no), error);
        }
        const std::string key = line.substr(0, split);
        const std::string value = line.substr(split + 1);
        if (!values_.emplace(key, value).second)
        {
            return fail("duplicate quantization manifest key: " + key, error);
        }
    }
    return true;
}

bool GPT2QuantPlan::getString(const std::string &key, std::string &out,
                              std::string *error) const
{
    auto it = values_.find(key);
    if (it == values_.end())
        return fail("missing quantization key: " + key, error);
    out = it->second;
    return true;
}

bool GPT2QuantPlan::getInt(const std::string &key, int &out, std::string *error) const
{
    ll value = 0;
    if (!getI64(key, value, error))
        return false;
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    {
        return fail("quantization integer out of int range: " + key, error);
    }
    out = static_cast<int>(value);
    return true;
}

bool GPT2QuantPlan::getI64(const std::string &key, ll &out, std::string *error) const
{
    std::string text;
    if (!getString(key, text, error))
        return false;
    long long value = 0;
    if (!parseSigned(text, value))
        return fail("invalid int64 quantization value: " + key, error);
    out = static_cast<ll>(value);
    return true;
}

bool GPT2QuantPlan::getDouble(const std::string &key, double &out,
                              std::string *error) const
{
    std::string text;
    if (!getString(key, text, error))
        return false;
    if (!parseFloat(text, out))
        return fail("invalid floating quantization value: " + key, error);
    return true;
}

template <typename T>
bool GPT2QuantPlan::loadTensor(const std::string &key, std::vector<T> &out,
                               std::string *error) const
{
    static_assert(std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value,
                  "quant tensors use int32 or int64");
    std::string relative;
    ll count_i64 = 0;
    if (!getString(key + ".file", relative, error) ||
        !getI64(key + ".count", count_i64, error))
    {
        return false;
    }
    if (count_i64 < 0 || static_cast<unsigned long long>(count_i64) >
                             std::numeric_limits<std::size_t>::max() / sizeof(T))
    {
        return fail("invalid tensor count: " + key, error);
    }
    const std::size_t count = static_cast<std::size_t>(count_i64);
    const std::string path = joinPath(artifact_dir_, relative);
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return fail("cannot open quantization tensor: " + path, error);
    out.resize(count);
    if (count)
    {
        input.read(reinterpret_cast<char *>(out.data()), count * sizeof(T));
        if (!input)
            return fail("short read from quantization tensor: " + path, error);
    }
    char trailing = 0;
    if (input.read(&trailing, 1))
    {
        return fail("quantization tensor has trailing bytes: " + path, error);
    }
    return true;
}

bool GPT2QuantPlan::loadLinear(const std::string &name, QuantLinearPlan &out,
                               std::string *error)
{
    out = QuantLinearPlan{};
    out.name = name;
    const std::string p = "linear." + name;
    if (!getString(p + ".mode", out.mode, error) ||
        !getInt(p + ".in_dim", out.in_dim, error) ||
        !getInt(p + ".out_dim", out.out_dim, error) ||
        !getDouble(p + ".input_scale", out.input_scale, error) ||
        !getI64(p + ".input_bound", out.input_bound, error) ||
        !getDouble(p + ".weight_scale", out.weight_scale, error) ||
        !getDouble(p + ".accumulator_scale", out.accumulator_scale, error) ||
        !getI64(p + ".accumulator_bound", out.accumulator_bound, error) ||
        !getDouble(p + ".output_scale", out.output_scale, error) ||
        !getI64(p + ".output_bound", out.output_bound, error) ||
        !getI64(p + ".c1", out.requant.c1, error) ||
        !getI64(p + ".c2", out.requant.c2, error) ||
        !getInt(p + ".n", out.requant.n, error) ||
        !getInt(p + ".range_bits", out.requant.range_bits, error) ||
        !loadTensor(p + ".weight", out.weight, error) ||
        !loadTensor(p + ".bias", out.bias, error))
    {
        return false;
    }
    if (out.mode != "immediate" && out.mode != "wide")
    {
        return fail(name + ": invalid linear mode", error);
    }
    const std::size_t expected_weight =
        static_cast<std::size_t>(out.in_dim) * static_cast<std::size_t>(out.out_dim);
    if (out.weight.size() != expected_weight || out.bias.size() != static_cast<std::size_t>(out.out_dim))
    {
        return fail(name + ": quantized linear tensor shape mismatch", error);
    }
    if (out.requant.c1 <= 0 || out.requant.c2 <= 0 || out.accumulator_bound <= 0 ||
        out.output_bound != R)
    {
        return fail(name + ": invalid linear bound or dyadic coefficient", error);
    }
    const __int128 lhs = static_cast<__int128>(out.requant.c1) * out.accumulator_bound;
    const __int128 rhs = static_cast<__int128>(R) * out.requant.c2;
    if (lhs > rhs)
        return fail(name + ": C1/C2 is not an under-approximation", error);
    return true;
}

bool GPT2QuantPlan::loadLayerNorm(const std::string &name, QuantLayerNormPlan &out,
                                  std::string *error)
{
    out = QuantLayerNormPlan{};
    out.name = name;
    const std::string p = "ln." + name;
    if (!getInt(p + ".width", out.width, error) ||
        !getI64(p + ".input_bound", out.input_bound, error) ||
        !getDouble(p + ".normalized_scale", out.normalized_scale, error) ||
        !getDouble(p + ".gamma_scale", out.gamma_scale, error) ||
        !getDouble(p + ".beta_scale", out.beta_scale, error) ||
        !getI64(p + ".accumulator_bound", out.accumulator_bound, error) ||
        !getDouble(p + ".output_scale", out.output_scale, error) ||
        !getI64(p + ".output_bound", out.output_bound, error) ||
        !getInt(p + ".term1_c", out.term1_c, error) ||
        !getInt(p + ".term1_e", out.term1_e, error) ||
        !getInt(p + ".term2_c", out.term2_c, error) ||
        !getInt(p + ".term2_e", out.term2_e, error) ||
        !getInt(p + ".sigma_range_bits", out.sigma_range_bits, error) ||
        !getInt(p + ".rounding_range_bits", out.rounding_range_bits, error) ||
        !getInt(p + ".variance_bits", out.variance_bits, error) ||
        !getInt(p + ".q_input_sigma_range_bits",
                out.q_input_sigma_range_bits, error) ||
        !getInt(p + ".q_input_rounding_range_bits",
                out.q_input_rounding_range_bits, error) ||
        !getInt(p + ".q_input_variance_bits",
                out.q_input_variance_bits, error) ||
        !loadTensor(p + ".gamma", out.gamma, error) ||
        !loadTensor(p + ".beta", out.beta, error))
    {
        return false;
    }
    if (out.width <= 1 || out.gamma.size() != static_cast<std::size_t>(out.width) ||
        out.beta.size() != static_cast<std::size_t>(out.width) || out.output_bound != R ||
        out.term1_c <= 0 || out.term2_c <= 0 || out.input_bound <= 0 ||
        out.sigma_range_bits <= 0 || out.sigma_range_bits > 126 ||
        out.rounding_range_bits <= 0 || out.rounding_range_bits > 126 ||
        out.variance_bits <= 0 || out.variance_bits > 253 ||
        out.q_input_sigma_range_bits <= 0 ||
        out.q_input_sigma_range_bits > 126 ||
        out.q_input_rounding_range_bits <= 0 ||
        out.q_input_rounding_range_bits > 126 ||
        out.q_input_variance_bits <= 0 ||
        out.q_input_variance_bits > 126)
    {
        return fail(name + ": invalid LayerNorm plan", error);
    }
    if (out.term1_c >= (1 << q_bits) || out.term2_c >= (1 << q_bits))
        return fail(name + ": nonlinear coefficient exceeds declared precision", error);
    try
    {
        (void)zkagent_integer::layernorm_coefficients(out.term1_c, out.term1_e,
                                                      out.term2_c, out.term2_e);
    }
    catch (const std::exception &exception)
    {
        return fail(name + ": " + exception.what(), error);
    }
    return true;
}

bool GPT2QuantPlan::loadResidual(const std::string &name, QuantResidualPlan &out,
                                 std::string *error)
{
    out = QuantResidualPlan{};
    out.name = name;
    const std::string p = "residual." + name;
    if (!getDouble(p + ".left_scale", out.left_scale, error) ||
        !getI64(p + ".left_bound", out.left_bound, error) ||
        !getDouble(p + ".right_scale", out.right_scale, error) ||
        !getI64(p + ".right_bound", out.right_bound, error) ||
        !getDouble(p + ".output_scale", out.output_scale, error) ||
        !getI64(p + ".output_bound", out.output_bound, error) ||
        !getI64(p + ".left_c1", out.left.c1, error) ||
        !getI64(p + ".left_c2", out.left.c2, error) ||
        !getInt(p + ".left_range_bits", out.left.range_bits, error) ||
        !getI64(p + ".right_c1", out.right.c1, error) ||
        !getI64(p + ".right_c2", out.right.c2, error) ||
        !getInt(p + ".right_range_bits", out.right.range_bits, error))
    {
        return false;
    }
    if (out.left.c1 <= 0 || out.left.c2 <= 0 || out.right.c1 <= 0 ||
        out.right.c2 <= 0 || out.left_bound < 0 || out.right_bound < 0 ||
        out.output_bound <= 0)
    {
        return fail(name + ": invalid residual plan", error);
    }
    return true;
}

bool GPT2QuantPlan::loadAttention(const std::string &name,
                                  QuantAttentionPlan &out,
                                  std::string *error)
{
    out = QuantAttentionPlan{};
    const std::string p = "attention." + name;
    if (!getDouble(p + ".score_scale", out.score_scale, error) ||
        !getI64(p + ".score_bound", out.score_bound, error) ||
        !getInt(p + ".score_bits", out.score_bits, error) ||
        !getI64(p + ".theta", out.theta, error) ||
        !getI64(p + ".shift_upper_bound", out.shift_upper_bound, error) ||
        !getInt(p + ".shift_bits", out.shift_bits, error) ||
        !getInt(p + ".least_segments", out.least_segments, error) ||
        !getInt(p + ".nonlinear_segment", out.nonlinear_segment, error) ||
        !getInt(p + ".most_zero_segments", out.most_zero_segments, error) ||
        !getInt(p + ".mass_bits", out.mass_bits, error) ||
        !getI64(p + ".accumulator_bound", out.accumulator_bound, error) ||
        !getInt(p + ".accumulator_bits", out.accumulator_bits, error) ||
        !getInt(p + ".rounding_bits", out.rounding_bits, error) ||
        !getI64(p + ".output_bound", out.output_bound, error) ||
        !getInt(p + ".output_bits", out.output_bits, error) ||
        !loadTensor(p + ".epsilon", out.epsilon_by_count, error))
    {
        return false;
    }
    for (int i = 0; i < 6; ++i)
    {
        if (!getInt(p + ".digit" + std::to_string(i) + "_bits",
                    out.digit_bits[static_cast<std::size_t>(i)], error))
        {
            return false;
        }
    }
    if (!(out.score_scale > 0.0) || out.score_bound <= 0 || out.theta != R ||
        out.shift_upper_bound <= 0 || out.output_bound < R ||
        out.score_bits <= 0 || out.shift_bits <= 0 || out.mass_bits <= 0 ||
        out.accumulator_bound <= 0 || out.accumulator_bits <= 0 ||
        out.rounding_bits != q_bits + 1 || out.output_bits <= 0 ||
        out.least_segments != 3 || out.nonlinear_segment != 3 ||
        out.most_zero_segments != 2 ||
        out.epsilon_by_count.size() != static_cast<std::size_t>(positions))
    {
        return fail(name + ": invalid attention plan", error);
    }
    int total_bits = 0;
    for (int bits : out.digit_bits)
    {
        if (bits < 0 || bits > 16)
        {
            return fail(name + ": invalid attention digit width", error);
        }
        total_bits += bits;
    }
    if (total_bits <= 0 || total_bits >= 63 ||
        out.shift_upper_bound >= (static_cast<ll>(1) << total_bits))
    {
        return fail(name + ": attention digit domain is too small", error);
    }
    ll epsilon_max = 0;
    for (int64_t epsilon : out.epsilon_by_count)
    {
        if (epsilon < 0 || static_cast<ll>(epsilon) >= out.theta)
        {
            return fail(name + ": invalid attention row-mass tolerance", error);
        }
        epsilon_max = std::max(epsilon_max, static_cast<ll>(epsilon));
    }
    const ll expected_accumulator = R * (out.theta + epsilon_max);
    const ll expected_output =
        (expected_accumulator + out.theta - 1) / out.theta;
    if (out.accumulator_bound != expected_accumulator || out.output_bound != expected_output)
    {
        return fail(name + ": inconsistent attention output bound", error);
    }
    return true;
}

bool GPT2QuantPlan::load(const std::string &model_dir, std::string *error)
{
    is_loaded_ = false;
    artifact_dir_ = joinPath(model_dir, "quantized");
    if (!parseManifest(joinPath(artifact_dir_, "quant_plan.txt"), error))
        return false;

    std::string format;
    std::string coefficient_mode;
    std::string activation_rounding;
    std::string attention_softmax;
    if (!getString("format", format, error) || format != "zkagent_quant_plan")
    {
        return fail("unsupported quantization manifest format", error);
    }
    if (!getString("dyadic.mode", coefficient_mode, error) ||
        coefficient_mode != "under_approx_floor")
    {
        return fail("quantization plan must use under_approx_floor", error);
    }
    if (!getString("rounding.activation", activation_rounding, error) ||
        activation_rounding != "floor_plus_half")
    {
        return fail("quantization plan must use floor_plus_half activation rounding", error);
    }
    if (!getString("scheme.attention_softmax", attention_softmax, error) ||
        attention_softmax != "zkllm_digits")
    {
        return fail("quantization plan must use zkllm_digits attention", error);
    }
    if (!getInt("q_bits", q_bits, error) || !getI64("R", R, error) ||
        !getInt("model.vocab_size", vocab_size, error) ||
        !getInt("embedding.positions", positions, error) ||
        !getInt("model.hidden", hidden, error) ||
        !getInt("model.layers", layer_count, error) ||
        !getInt("model.heads", heads, error) ||
        !getInt("model.head_dim", head_dim, error) ||
        !getInt("model.ffn", ffn, error) ||
        !getDouble("embedding.scale", embedding_scale, error) ||
        !getI64("embedding.bound", embedding_bound, error) ||
        !loadTensor("embedding.token", token_embedding, error) ||
        !loadTensor("embedding.position", position_embedding, error))
    {
        return false;
    }
    if (q_bits <= 0 || q_bits > 22 || R != (ll(1) << q_bits))
    {
        return fail("invalid Q/R in quantization plan", error);
    }
    int nonlinear_bits = 0;
    std::string nonlinear_coefficient_rounding, nonlinear_rounding;
    if (!getInt("nonlinear.coefficient_bits", nonlinear_bits, error) || nonlinear_bits != q_bits ||
        !getString("nonlinear.coefficient_rounding", nonlinear_coefficient_rounding, error) ||
        nonlinear_coefficient_rounding != "nearest_even" ||
        !getString("rounding.nonlinear", nonlinear_rounding, error) ||
        nonlinear_rounding != "nearest_away")
    {
        return fail("unsupported nonlinear quantization codec", error);
    }
    if (token_embedding.size() != static_cast<std::size_t>(vocab_size) * hidden ||
        position_embedding.size() != static_cast<std::size_t>(positions) * hidden)
    {
        return fail("embedding tensor shape mismatch", error);
    }

    blocks.resize(layer_count);
    for (int i = 0; i < layer_count; ++i)
    {
        const std::string base = "block." + std::to_string(i);
        QuantBlockPlan &block = blocks[i];
        if (!loadLayerNorm(base + ".ln1", block.ln1, error) ||
            !loadLinear(base + ".q", block.q, error) ||
            !loadLinear(base + ".k", block.k, error) ||
            !loadLinear(base + ".v", block.v, error) ||
            !loadAttention(base, block.attention, error) ||
            !loadLinear(base + ".attn_proj", block.attn_proj, error) ||
            !loadResidual(base + ".attn", block.attn_residual, error) ||
            !loadLayerNorm(base + ".ln2", block.ln2, error) ||
            !loadLinear(base + ".ffn_up", block.ffn_up, error) ||
            !getInt("gelu." + base + ".scale_c", block.gelu_scale_c, error) ||
            !getInt("gelu." + base + ".scale_e", block.gelu_scale_e, error) ||
            !getDouble("gelu." + base + ".input_scale", block.gelu_input_scale, error) ||
            !getDouble("gelu." + base + ".output_scale", block.gelu_output_scale, error) ||
            !getInt("gelu." + base + ".d1_range_bits",
                    block.gelu_d1_range_bits, error) ||
            !getInt("gelu." + base + ".d2_range_bits",
                    block.gelu_d2_range_bits, error) ||
            !getInt("gelu." + base + ".rounding_range_bits",
                    block.gelu_rounding_range_bits, error) ||
            !loadLinear(base + ".ffn_down", block.ffn_down, error) ||
            !loadResidual(base + ".ffn", block.ffn_residual, error))
        {
            return false;
        }
        if (block.gelu_d1_range_bits <= 0 || block.gelu_d1_range_bits > 126 ||
            block.gelu_d2_range_bits <= 0 || block.gelu_d2_range_bits > 126 ||
            block.gelu_rounding_range_bits <= 0 ||
            block.gelu_rounding_range_bits > 126)
        {
            return fail(base + ": invalid GELU range widths", error);
        }
        if (block.gelu_scale_c >= (1 << q_bits))
            return fail(base + ": nonlinear coefficient exceeds declared precision", error);
        try
        {
            (void)zkagent_integer::gelu_coefficients(block.gelu_scale_c, block.gelu_scale_e);
        }
        catch (const std::exception &exception)
        {
            return fail(base + ": " + exception.what(), error);
        }
    }
    if (!loadLayerNorm("final", final_ln, error) ||
        !loadLinear("lm_head", lm_head, error) ||
        !getDouble("lm_head.logit_scale", logit_scale, error) ||
        !getI64("lm_head.logit_bound", logit_bound, error))
    {
        return false;
    }
    is_loaded_ = true;
    return true;
}

bool GPT2QuantPlan::validateDimensions(int expected_layers, int expected_heads,
                                       int expected_head_dim, int expected_hidden,
                                       int expected_ffn, int expected_vocab,
                                       std::string *error) const
{
    if (!is_loaded_)
        return fail("quantization plan is not loaded", error);
    if (layer_count != expected_layers || heads != expected_heads ||
        head_dim != expected_head_dim || hidden != expected_hidden ||
        ffn != expected_ffn || vocab_size != expected_vocab)
    {
        std::ostringstream message;
        message << "quantization plan/model dimension mismatch: plan=(L=" << layer_count
                << ",H=" << heads << ",Hd=" << head_dim << ",d=" << hidden
                << ",ffn=" << ffn << ",vocab=" << vocab_size << ") expected=(L="
                << expected_layers << ",H=" << expected_heads << ",Hd="
                << expected_head_dim << ",d=" << expected_hidden << ",ffn="
                << expected_ffn << ",vocab=" << expected_vocab << ")";
        return fail(message.str(), error);
    }
    return true;
}

int GPT2QuantPlan::maxRangeBits() const
{
    int result = 0;
    auto include_linear = [&](const QuantLinearPlan &linear)
    {
        result = std::max(result, linear.requant.range_bits);
    };
    for (const auto &block : blocks)
    {
        include_linear(block.q);
        include_linear(block.k);
        include_linear(block.v);
        include_linear(block.attn_proj);
        include_linear(block.ffn_up);
        include_linear(block.ffn_down);
        result = std::max(result, block.ln1.sigma_range_bits);
        result = std::max(result, block.ln1.rounding_range_bits);
        result = std::max(result, block.ln2.sigma_range_bits);
        result = std::max(result, block.ln2.rounding_range_bits);
        result = std::max(result, block.gelu_d1_range_bits);
        result = std::max(result, block.gelu_d2_range_bits);
        result = std::max(result, block.gelu_rounding_range_bits);
        result = std::max(result, block.attention.mass_bits);
        result = std::max(result, block.attention.rounding_bits);
        result = std::max(result, block.attn_residual.left.range_bits);
        result = std::max(result, block.attn_residual.right.range_bits);
        result = std::max(result, block.ffn_residual.left.range_bits);
        result = std::max(result, block.ffn_residual.right.range_bits);
    }
    result = std::max(result, final_ln.sigma_range_bits);
    result = std::max(result, final_ln.rounding_range_bits);
    result = std::max(result, lm_head.requant.range_bits);
    return result;
}

template bool GPT2QuantPlan::loadTensor<int32_t>(
    const std::string &, std::vector<int32_t> &, std::string *) const;
template bool GPT2QuantPlan::loadTensor<int64_t>(
    const std::string &, std::vector<int64_t> &, std::string *) const;
