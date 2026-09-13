

#undef NDEBUG
#include "model/gpt2.hpp"
#include "common/integer_arithmetic.hpp"
#include "pcs/hyrax.hpp"
#include "model/nonlinear_quant.hpp"
#include "model/linear_witness.hpp"
#include "common/field_integer.hpp"
#include "lookup/lookup_witness.hpp"
#include "model/attention_softmax.hpp"
#include "common/timer.hpp"
#include <iostream>
#include <memory>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <climits>
#include <chrono>
#include <stdexcept>
#include <fstream>
#include <limits>
#include <omp.h>
#include <exception>

static ll ll_abs(ll x) { return (x < 0) ? -x : x; }

namespace
{

    constexpr char CircuitCacheType[] = "zkagent.gpt2.circuit-cache\n";

    void validate_nonnegative_bits(ll value, unsigned bits, const char *label)
    {
        if (bits == 0 || bits > 126)
        {
            throw std::invalid_argument(std::string(label) +
                                        " has an invalid declared range width");
        }
        const ll limit = static_cast<ll>(1) << bits;
        if (value < 0 || value >= limit)
        {
            throw std::runtime_error(std::string(label) +
                                     " is outside its declared non-negative range");
        }
    }

    void validate_nonnegative_bits_pair(
        ll first, ll second, unsigned bits, const char *label)
    {
        validate_nonnegative_bits(first, bits, label);
        validate_nonnegative_bits(second, bits, label);
    }
}

namespace
{
    struct AttentionAdviceLayout
    {
        int zeta_base;
        std::array<int, 6> digit_base;
        int middle_base, high0_base, high1_base, probability_base;
        int mass_lower_base, mass_upper_base, y_base, round_residual_base;

        AttentionAdviceLayout(int aux_start, int rows, int scores, int output_values)
            : zeta_base(aux_start + 10)
        {
            digit_base[0] = zeta_base + rows;
            for (int segment = 1; segment < 6; ++segment)
                digit_base[segment] = digit_base[segment - 1] + scores;
            middle_base = digit_base[5] + scores;
            high0_base = middle_base + scores;
            high1_base = high0_base + scores;
            probability_base = high1_base + scores;
            mass_lower_base = probability_base + scores;
            mass_upper_base = mass_lower_base + rows;
            y_base = mass_upper_base + rows;
            round_residual_base = y_base + output_values;
        }

        void writeRow(vector<F> &values, int row, const AttentionSoftmaxRowWitness &w) const
        {
            values[zeta_base + row] = zkagent_integer::fr_from_i128(w.zeta);
            values[mass_lower_base + row] = zkagent_integer::fr_from_i128(w.lower_residual);
            values[mass_upper_base + row] = zkagent_integer::fr_from_i128(w.upper_residual);
        }

        void writeScore(vector<F> &values, int index, int column, const AttentionSoftmaxRowWitness &w) const
        {
            for (int segment = 0; segment < 6; ++segment)
                values[digit_base[segment] + index] = zkagent_integer::fr_from_i128(w.digits[segment][column]);
            values[middle_base + index] = zkagent_integer::fr_from_i128(w.middle_outputs[column]);
            values[high0_base + index] = zkagent_integer::fr_from_i128(w.high_zero_flags[0][column]);
            values[high1_base + index] = zkagent_integer::fr_from_i128(w.high_zero_flags[1][column]);
            values[probability_base + index] = zkagent_integer::fr_from_i128(w.probabilities[column]);
        }

        void writeOutput(vector<F> &values, int index, int compact, ll weighted,
                         const QuantAttentionPlan &plan) const
        {
            if (weighted < -plan.accumulator_bound || weighted > plan.accumulator_bound)
                throw std::runtime_error("attention accumulator exceeds offline bound");
            const ll y = zkagent_integer::floor_div_i128(2 * weighted + plan.theta, 2 * plan.theta);
            if (ll_abs(y) > plan.output_bound)
                throw std::runtime_error("attention output exceeds offline bound");
            const ll residual = 2 * weighted - 2 * plan.theta * y + plan.theta;
            validate_nonnegative_bits(
                residual, static_cast<unsigned>(plan.rounding_bits),
                "attention fixed-theta rounding residual");
            values[y_base + index] = zkagent_integer::fr_from_i128(y);
            values[round_residual_base + compact] = zkagent_integer::fr_from_i128(residual);
        }
    };
}

const GPT2CircuitProfileSpec &gpt2CircuitProfileSpec(
    GPT2CircuitProfile profile)
{
    static const GPT2CircuitProfileSpec CompleteInference{
        "complete-inference",
        false,
        true,
        true,
        true,
        true,
    };
    static const GPT2CircuitProfileSpec ZkGPTArtifactTransformer{
        "zkgpt-artifact-transformer",
        true,
        false,
        false,
        false,
        true,
    };
    switch (profile)
    {
    case GPT2CircuitProfile::CompleteInference:
        return CompleteInference;
    case GPT2CircuitProfile::ZkGPTArtifactTransformer:
        return ZkGPTArtifactTransformer;
    }
    throw std::invalid_argument("unknown GPT-2 circuit profile");
}

using std::cerr;
using std::cout;
using std::endl;
using std::max;
using std::min;
using zkagent_integer::fr_from_i128;

using zkagent_integer::fr_to_i128;

static std::string i128_to_string(ll v)
{
    if (v == 0)
        return "0";
    bool neg = v < 0;
    if (neg)
        v = -v;
    std::string s;
    while (v > 0)
    {
        int digit = (int)(v % 10);
        s.push_back((char)('0' + digit));
        v /= 10;
    }
    if (neg)
        s.push_back('-');
    std::reverse(s.begin(), s.end());
    return s;
}

GPT2::GPT2(KZG_SRS &uni_srs_, MVKZG_SRS &mv_srs_, int depth, int headnum_, int headdim_, int attn_dim_, int linear_dim_,
           int seq_len, int vocab_size_, int threads_)
    : uni_srs(uni_srs_), mv_srs(mv_srs_), layer_num(depth), headnum(headnum_), headdim(headdim_),
      attn_dim(attn_dim_), linear_dim(linear_dim_), len(seq_len),
      vocab_size(vocab_size_), threads(threads_)
{

    if (depth <= 0 || depth > model_limits::MaxTransformerLayers)
    {
        throw std::invalid_argument("GPT-2 layer count must be in [1, " +
                                    std::to_string(model_limits::MaxTransformerLayers) + "]");
    }
    if (seq_len > 1024)
    {
        cerr << "Warning: seq_len " << seq_len << " exceeds max position embedding size 1024" << endl;
    }
    total_in_size = 0;
    q_offset = 0;
    channel_in = 0;
    channel_out = 0;

    for (int *&matrix : mat_values)
    {
        matrix = nullptr;
    }

    embedding_matrix = new float[(long long)vocab_size * attn_dim];
    position_embedding = nullptr;
    quantized_embedding_matrix = nullptr;

    save_intermediates = false;
    intermediates_dir = "";

    gens = nullptr;
    gens_size = 0;
    gens_base.clear();

    quant_scale = 1.0;

    int qkv_dim = 3 * attn_dim;
    for (int i = 0; i < depth; i++)
    {
        full_conn.emplace_back(qkv_dim, attn_dim);
        full_conn.emplace_back(attn_dim, attn_dim);
        full_conn.emplace_back(linear_dim, attn_dim);
        full_conn.emplace_back(attn_dim, linear_dim);
    }

    SIZE = 1 + 23 * depth + 3 + 2;
}

GPT2::~GPT2()
{
    for (int *&matrix : mat_values)
    {
        if (matrix)
        {
            delete[] matrix;
            matrix = nullptr;
        }
    }

    if (embedding_matrix)
    {
        delete[] embedding_matrix;
        embedding_matrix = nullptr;
    }
    if (position_embedding)
    {
        delete[] position_embedding;
        position_embedding = nullptr;
    }
    if (quantized_embedding_matrix)
    {
        delete[] quantized_embedding_matrix;
        quantized_embedding_matrix = nullptr;
    }
    if (owned_embedded_input)
    {
        delete[] owned_embedded_input;
        owned_embedded_input = nullptr;
    }

    if (gens)
    {
        delete[] gens;
        gens = nullptr;
    }
}

void GPT2::setInput(const ll *data)
{
    input_data = data;
}

void GPT2::setSeqLen(int new_seq_len)
{
    len = new_seq_len;
    prepared_weights_owner_ = nullptr;
    prepared_attention_ready_ = false;
}

void GPT2::setCircuitProfile(GPT2CircuitProfile profile)
{
    circuit_profile_ = profile;
    prepared_weights_owner_ = nullptr;
    prepared_fc_witness_.clear();
    prepared_attention_ready_ = false;
    lm_head_output_offset = 0;
    embed_matrix_circuit_offset = 0;
}

std::size_t GPT2::ensureAttentionLookupGroup(const std::string &key, int input_bits,
                                             bool mapped, const std::vector<ll> &table_outputs)
{
    for (std::size_t i = 0; i < attention_lookup_groups.size(); ++i)
    {
        const auto &group = attention_lookup_groups[i];
        if (group.key != key)
            continue;
        if (group.input_bits != input_bits || group.mapped != mapped || group.table_outputs != table_outputs)
            throw std::runtime_error("inconsistent attention lookup group");
        return i;
    }
    AttentionLookupGroup group;
    group.key = key;
    group.input_bits = input_bits;
    group.mapped = mapped;
    group.table_outputs = table_outputs;
    attention_lookup_groups.push_back(std::move(group));
    return attention_lookup_groups.size() - 1;
}

void GPT2::recordAttentionLookup(const std::string &key, int input_bits, bool mapped,
                                 int input_offset, int output_offset, const std::vector<ll> &table_outputs)
{
    auto &group = attention_lookup_groups[ensureAttentionLookupGroup(key, input_bits, mapped, table_outputs)];
    group.input_offsets.push_back(input_offset);
    if (mapped)
        group.output_offsets.push_back(output_offset);
}

void GPT2::recordPowerOfTwoRange(
    const std::string &id, std::size_t offset, std::size_t count,
    unsigned bit_width, std::size_t stride)
{
    range_query_specs.push_back(RangeQuerySpec::powerOfTwo(
        id, {InputSpan(offset, count, stride)}, bit_width));
}

const QuantLinearPlan &GPT2::linearPlanForFc(int fc_id, int group) const
{
    if (!quant_plan.loaded())
    {
        throw std::runtime_error("quantization plan is not loaded");
    }
    if (fc_id < 0 || fc_id >= (int)full_conn.size())
    {
        throw std::out_of_range("FC id outside quantization plan");
    }
    const QuantBlockPlan &block = quant_plan.blocks.at(fc_id / 4);
    switch (fc_id % 4)
    {
    case 0:
        if (group == 0)
            return block.q;
        if (group == 1)
            return block.k;
        if (group == 2)
            return block.v;
        throw std::out_of_range("QKV quantization group outside [0,3)");
    case 1:
        return block.attn_proj;
    case 2:
        return block.ffn_up;
    case 3:
        return block.ffn_down;
    }
    throw std::logic_error("unreachable FC quantization mapping");
}

const QuantLayerNormPlan &GPT2::layerNormPlan(int ln_id) const
{
    if (!quant_plan.loaded())
    {
        throw std::runtime_error("quantization plan is not loaded");
    }
    if (ln_id == layer_num * 2)
        return quant_plan.final_ln;
    if (ln_id < 0 || ln_id >= layer_num * 2)
    {
        throw std::out_of_range("LayerNorm id outside quantization plan");
    }
    const QuantBlockPlan &block = quant_plan.blocks.at(ln_id / 2);
    return (ln_id & 1) ? block.ln2 : block.ln1;
}

void GPT2::initQuantPolicy(prover &pr)
{
    (void)pr;
    fc_round_requants.clear();
    for (int i = 0; i < (int)full_conn.size(); ++i)
    {
        FCRoundRequant entry;
        entry.group_count = (i % 4 == 0) ? 3 : 1;
        entry.group_width = (i % 4 == 0) ? attn_dim : full_conn[i].channel_out;
        for (int group = 0; group < entry.group_count; ++group)
        {
            const QuantDyadic &source = linearPlanForFc(i, group).requant;
            entry.group[group] = {source.c1, source.c2, source.n, source.range_bits};
        }
        fc_round_requants.push_back(entry);
    }
}

InputCommitmentLayout GPT2::inputCommitmentLayout() const
{
    if (!quant_plan.loaded() || layer_num <= 0 || len <= 0)
        throw std::logic_error("fixed input layout requires the public model and shape");
    const auto padded = [](std::size_t value)
    {
        std::size_t n = 1;
        while (n < value)
        {
            if (n >= (std::size_t(1) << 30))
                throw std::overflow_error("input shape too large");
            n <<= 1;
        }
        return n;
    };
    const auto &profile = circuitProfileSpec();
    const std::size_t t = padded(len), h = padded(attn_dim), f = padded(linear_dim), qkv = padded(3 * attn_dim);
    const std::size_t matrices = layer_num * (h * qkv + h * h + h * f + f * h);
    const std::size_t parameters = matrices + layer_num * (qkv + h + f + h) +
                                   (2 * layer_num + (profile.include_final_layer_norm ? 1 : 0)) * 2 * h;
    const auto ln = [t, h]()
    { return 10 + 3 * t * h + 5 * t; };
    std::size_t advice = 0;
    for (int block = 0; block < layer_num; ++block)
    {
        advice += 2 * ln();
        const std::size_t out[] = {qkv, h, f, h};
        for (int stage = 0; stage < 4; ++stage)
        {
            if (profile.force_immediate_linear_requant || linearPlanForFc(block * 4 + stage).mode == "immediate")
                advice += 20 + 2 * t * out[stage];
        }
        if (profile.include_skip_connections)
            advice += 2 * (20 + 5 * t * h);
        const std::size_t score_count = headnum * t * (t + 1) / 2;
        advice += 10 + 3 * headnum * t + 10 * score_count + t * h + headnum * t * headdim;
        advice += 10 + 3 * t * f + 4 * t * linear_dim;
    }
    if (profile.include_final_layer_norm)
        advice += ln();
    const std::size_t head_size = profile.include_lm_head ? padded(vocab_size) * h : 0;
    const std::size_t head_output = profile.include_lm_head ? 20 + t * vocab_size : 0;
    std::size_t domain = padded(t * h + parameters + advice + head_size + head_output);
    for (;;)
    {
        int bits = 0;
        for (std::size_t n = domain; n > 1; n >>= 1)
            ++bits;
        const std::size_t columns = std::size_t(1) << (bits - bits / 2);
        const auto align = [columns](std::size_t n)
        { return (n + columns - 1) / columns * columns; };
        const std::size_t begin = align(t * h), end = align(begin + parameters);
        InputCommitmentLayout result;
        result.variables = bits;
        result.fixed_rows.push_back({begin / columns, (end - begin) / columns});
        std::size_t count = end + advice;
        if (profile.include_lm_head)
        {
            const std::size_t head_begin = align(count), head_end = align(head_begin + head_size);
            result.fixed_rows.push_back({head_begin / columns, (head_end - head_begin) / columns});
            count = head_end + head_output;
        }
        if (count > domain)
        {
            domain = padded(count);
            continue;
        }
        result.value_count = count;
        result.validate();
        return result;
    }
}

void GPT2::initParam(prover &pr)
{
    total_in_size = 0;

    int real_len = len;
    const auto input_layout = inputCommitmentLayout();
    i64 pos = input_layout.fixed_rows.front().first * input_layout.columns();

    for (int i = 0; i < full_conn.size(); ++i)
    {
        auto &fc = full_conn[i];

        pr.fc_real_input_row[i] = real_len;
        len = pr.fc_input_row[i] = 1 << ceilPow2BitLength(len);
        pr.fc_real_input_col[i] = fc.channel_in;
        pr.fc_real_row[i] = fc.channel_in;
        pr.fc_input_col[i] = 1 << ceilPow2BitLength(fc.channel_in);

        fc.channel_in = pr.fc_row[i] = 1 << ceilPow2BitLength(fc.channel_in);
        pr.fc_real_col[i] = fc.channel_out;
        fc.channel_out = pr.fc_col[i] = 1 << ceilPow2BitLength(fc.channel_out);

        fc.weight_start_id = pos;
        pr.fc_start_id[i] = fc.weight_start_id;
        u32 para_size = pr.fc_row[i] * pr.fc_col[i];
        pos += para_size;
    }
    total_in_size = pos;
}

void GPT2::prepareWeights(prover &pr)
{
    cout << "Preparing weights for circuit..." << endl;

    for (int &bias_offset : pr.fc_bias_id)
    {
        bias_offset = -1;
    }

    for (int i = 0; i < full_conn.size(); ++i)
    {
        auto &fc = full_conn[i];
        prepareFCWeights(fc.weight_start_id, pr.fc_real_row[i], pr.fc_real_col[i], i,
                         fc.channel_in, fc.channel_out);
    }

    for (int i = 0; i < full_conn.size(); ++i)
    {
        prepareFCBias(pr, pr.fc_real_col[i], i, pr.fc_col[i]);
    }

    int ch_in = full_conn[0].channel_in;
    for (int block = 0; block < layer_num; block++)
    {
        prepareLayerNormWeights(block * 2, ch_in);
        prepareLayerNormWeights(block * 2 + 1, ch_in);
    }
    if (!full_conn.empty() && circuitProfileSpec().include_final_layer_norm)
    {
        int final_ln_id = layer_num * 2;
        int final_ch_in = full_conn.back().channel_out;
        prepareFinalLayerNormWeights(final_ln_id, final_ch_in);
    }

    const auto input_layout = inputCommitmentLayout();
    const auto &model_rows = input_layout.fixed_rows.front();
    const std::size_t aligned_end = (model_rows.first + model_rows.count) * input_layout.columns();
    if (val[0].size() > aligned_end)
        throw std::logic_error("fixed input layout underestimated model parameters");
    val[0].resize(aligned_end, Fr(0));
    total_in_size = aligned_end;
    const auto kernel_start = std::chrono::steady_clock::now();
    prepareLinearWitnessKernels(pr, false);
    fresh_fc_timing_.kernel_preparation_seconds += std::chrono::duration<double>(
                                                       std::chrono::steady_clock::now() - kernel_start)
                                                       .count();
    cout << "Weights prepared. Total input size: " << total_in_size << endl;
}

void GPT2::prepareFCWeights(i64 first_fc_id, int real_r, int real_c, int id, int ch_in, int ch_out)
{

    if (id < 0 || id >= model_limits::FconnCapacity)
    {
        throw std::out_of_range("FC layer id exceeds model metadata capacity");
    }

    auto val_0 = val[0].begin() + first_fc_id;
    if (mat_values[id])
        delete[] mat_values[id];
    mat_values[id] = new int[(long long)ch_out * ch_in];

    const int group_count = (id % 4 == 0) ? 3 : 1;
    for (int group = 0; group < group_count; ++group)
    {
        const QuantLinearPlan &plan = linearPlanForFc(id, group);
        if (plan.in_dim != real_r ||
            plan.out_dim != ((id % 4 == 0) ? attn_dim : real_c))
        {
            throw std::runtime_error("quantized FC tensor shape does not match circuit");
        }
    }

    const int thread_n = std::max(1, threads);
#pragma omp parallel for num_threads(thread_n) schedule(static)
    for (i64 co = 0; co < ch_out; ++co)
    {
        const int group = (id % 4 == 0) ? std::min<int>(co / attn_dim, 2) : 0;
        const auto &plan = linearPlanForFc(id, group);
        const i64 local_co = (id % 4 == 0) ? co % attn_dim : co;
        for (i64 ci = 0; ci < ch_in; ++ci)
        {
            const int q = (co < real_c && ci < real_r)
                              ? plan.weight[local_co * real_r + ci]
                              : 0;
            mat_values[id][co * ch_in + ci] = q;
            val_0[co * ch_in + ci] = q;
        }
    }
}

void GPT2::prepareFCBias(prover &pr, int real_c, int id, int ch_out)
{
    int bias_offset = (int)val[0].size();
    val[0].resize(bias_offset + ch_out);
    total_in_size += ch_out;
    for (int co = 0; co < ch_out; ++co)
    {
        ll q = 0;
        if (co < real_c)
        {
            const int group = (id % 4 == 0) ? co / attn_dim : 0;
            const int local_co = (id % 4 == 0) ? co % attn_dim : co;
            const QuantLinearPlan &plan = linearPlanForFc(id, group);
            q = static_cast<ll>(plan.bias.at(local_co));
        }
        val[0][bias_offset + co] = q;
    }
    pr.fc_bias_id[id] = bias_offset;
}

void GPT2::prepareLayerNormWeights(int ln_id, int ch_in)
{
    int orgsize = val[0].size();
    val[0].resize(orgsize + 2 * ch_in);
    total_in_size += 2 * ch_in;

    layer_norm_w_q_start[ln_id] = orgsize;
    layer_norm_b_q_start[ln_id] = orgsize + ch_in;

    const QuantLayerNormPlan &plan = layerNormPlan(ln_id);
    if (plan.width != attn_dim)
    {
        throw std::runtime_error("quantized LayerNorm width does not match circuit");
    }
    layer_norm_c1[ln_id] = plan.term1_c;
    layer_norm_e1[ln_id] = plan.term1_e;
    layer_norm_c2[ln_id] = plan.term2_c;
    layer_norm_e2[ln_id] = plan.term2_e;
    for (int i = 0; i < ch_in; ++i)
    {
        val[0][layer_norm_w_q_start[ln_id] + i] =
            (i < attn_dim) ? static_cast<ll>(plan.gamma[i]) : 0;
        val[0][layer_norm_b_q_start[ln_id] + i] =
            (i < attn_dim) ? static_cast<ll>(plan.beta[i]) : 0;
    }
}

void GPT2::prepareFinalLayerNormWeights(int ln_id, int ch_in)
{
    int orgsize = val[0].size();
    val[0].resize(orgsize + 2 * ch_in);
    total_in_size += 2 * ch_in;

    layer_norm_w_q_start[ln_id] = orgsize;
    layer_norm_b_q_start[ln_id] = orgsize + ch_in;

    const QuantLayerNormPlan &plan = layerNormPlan(ln_id);
    layer_norm_c1[ln_id] = plan.term1_c;
    layer_norm_e1[ln_id] = plan.term1_e;
    layer_norm_c2[ln_id] = plan.term2_c;
    layer_norm_e2[ln_id] = plan.term2_e;
    for (int i = 0; i < ch_in; ++i)
    {
        val[0][layer_norm_w_q_start[ln_id] + i] =
            (i < attn_dim) ? static_cast<ll>(plan.gamma[i]) : 0;
        val[0][layer_norm_b_q_start[ln_id] + i] =
            (i < attn_dim) ? static_cast<ll>(plan.beta[i]) : 0;
    }
}

void GPT2::inputLayer(layer &circuit, int ch_in)
{
    initLayer(circuit, total_in_size, layerType::INPUT);
    calcInputLayer(circuit, ch_in);
}

void GPT2::calcInputLayer(layer &circuit, int ch_in)
{
    val[0].resize(circuit.size);
    assert(val[0].size() == total_in_size);

    auto val_0 = val[0].begin();
    int hidden = attn_dim;

    int input_stride = ch_in;

    if (input_data)
    {

        for (i64 i = 0; i < len; i++)
        {
            for (i64 j = 0; j < hidden; j++)
            {
                val[0][i * input_stride + j] = F(input_data[i * hidden + j]);
            }

            for (i64 j = hidden; j < input_stride; j++)
                val[0][i * input_stride + j] = 0;
        }
    }
    else
    {

        for (i64 i = 0; i < len; i++)
        {
            for (i64 j = 0; j < hidden; j++)
            {
                ll s = rand() % 2000 - 1000;
                val[0][i * input_stride + j] = F(s);
            }
            for (i64 j = hidden; j < input_stride; j++)
                val[0][i * input_stride + j] = 0;
        }
    }

    if (build_gates)
    {

        val_0 = val[0].begin() + len * input_stride;
        for (; val_0 < val[0].begin() + circuit.size; ++val_0)
            val_0->clear();
    }
}

GPT2::GateWitnessSchedule GPT2::GateWitnessSchedule::build(const layer &circuit)
{
    GateWitnessSchedule result;
    auto group = [&](const auto &gates, std::vector<u32> &offsets,
                     std::vector<u32> &indices)
    {
        if (gates.size() > std::numeric_limits<u32>::max())
            throw std::length_error("witness gate index exceeds uint32");
        offsets.assign(static_cast<std::size_t>(circuit.size) + 1, 0);
        for (const auto &gate : gates)
            if (gate.g < circuit.size)
                ++offsets[static_cast<std::size_t>(gate.g) + 1];
        for (std::size_t i = 1; i < offsets.size(); ++i)
            offsets[i] += offsets[i - 1];
        indices.resize(offsets.back());
        auto cursor = offsets;
        for (std::size_t i = 0; i < gates.size(); ++i)
            if (gates[i].g < circuit.size)
                indices[cursor[gates[i].g]++] = static_cast<u32>(i);
    };
    group(circuit.uni_gates, result.unary_offsets, result.unary_indices);
    group(circuit.bin_gates, result.binary_offsets, result.binary_indices);
    return result;
}

void GPT2::prepareGateWitnessSchedules(const prover &pr)
{
    if (prepared_gate_owner_ == &pr.C && prepared_gate_schedules_.size() == pr.C.circuit.size())
        return;
    std::vector<GateWitnessSchedule> schedules(pr.C.circuit.size());
    for (std::size_t i = 1; i < schedules.size(); ++i)
        if (pr.C.circuit[i].ty != layerType::FCONN)
            schedules[i] = GateWitnessSchedule::build(pr.C.circuit[i]);
    prepared_gate_schedules_ = std::move(schedules);
    prepared_gate_owner_ = &pr.C;
}

void GPT2::calcNormalLayer(const layer &circuit, i64 layer_id,
                           const GateWitnessSchedule *schedule)
{
    val[layer_id].resize(circuit.size);
    GateWitnessSchedule local;
    if (!schedule)
    {
        local = GateWitnessSchedule::build(circuit);
        schedule = &local;
    }
    if (schedule->unary_offsets.size() != static_cast<std::size_t>(circuit.size) + 1 ||
        schedule->binary_offsets.size() != static_cast<std::size_t>(circuit.size) + 1)
        throw std::runtime_error("prepared gate witness shape changed");
    auto map_u0 = [&](u32 u) -> u32
    {
        return !circuit.ori_id_u.empty() && u < circuit.ori_id_u.size() ? circuit.ori_id_u[u] : u;
    };
    auto map_v0 = [&](u32 v) -> u32
    {
        return !circuit.ori_id_v.empty() && v < circuit.ori_id_v.size() ? circuit.ori_id_v[v] : v;
    };
    auto add_scaled = [](F &sum, const F &value, ll scalar)
    {
        if (scalar == 1)
            sum += value;
        else if (scalar == -1)
            sum -= value;
        else if (scalar != 0)
            sum += value * fr_from_i128(scalar);
    };

    const int thread_n = threads > 0 ? threads : 1;

#pragma omp parallel for num_threads(thread_n) schedule(dynamic, 64)
    for (i64 g = 0; g < (i64)circuit.size; g++)
    {
        val[layer_id][g].clear();

        for (u32 at = schedule->unary_offsets[g]; at < schedule->unary_offsets[g + 1]; ++at)
        {
            const auto &gate = circuit.uni_gates[schedule->unary_indices[at]];
            u32 u = gate.u;
            if (gate.lu == 0)
                u = map_u0(u);
            add_scaled(val[layer_id][g], val[gate.lu].at(u), gate.sc);
        }

        for (u32 at = schedule->binary_offsets[g]; at < schedule->binary_offsets[g + 1]; ++at)
        {
            const auto &gate = circuit.bin_gates[schedule->binary_indices[at]];
            int bin_lu = gate.getLayerIdU(layer_id);
            int bin_lv = gate.getLayerIdV(layer_id);
            u32 u = gate.u;
            u32 v = gate.v;
            if (bin_lu == 0)
                u = map_u0(u);
            if (bin_lv == 0)
                v = map_v0(v);
            add_scaled(val[layer_id][g], val[bin_lu].at(u) * val[bin_lv][v], gate.sc);
        }
    }
}

void GPT2::computeLayerNormWitness(int ln_id, int input_offset, int aux_start,
                                    int real_width, int padded_width, int real_rows,
                                    const bool *sparsity_map)
{
    const int c1 = layer_norm_c1[ln_id], e1 = layer_norm_e1[ln_id];
    const int c2 = layer_norm_c2[ln_id], e2 = layer_norm_e2[ln_id];
    const auto coefficients = zkagent_integer::layernorm_coefficients(c1, e1, c2, e2);
    const i64 block_len = static_cast<i64>(len) * padded_width;
    const int gamma_offset = layer_norm_w_q_start[ln_id];
    const int beta_offset = layer_norm_b_q_start[ln_id];
    const QuantLayerNormPlan &plan = layerNormPlan(ln_id);
    const unsigned sigma_range_bits = static_cast<unsigned>(
        circuitProfileSpec().force_immediate_linear_requant
            ? plan.q_input_sigma_range_bits
            : plan.sigma_range_bits);
    const unsigned rounding_range_bits = static_cast<unsigned>(
        circuitProfileSpec().force_immediate_linear_requant
            ? plan.q_input_rounding_range_bits
            : plan.rounding_range_bits);
    const unsigned variance_bits = static_cast<unsigned>(
        circuitProfileSpec().force_immediate_linear_requant
            ? plan.q_input_variance_bits
            : plan.variance_bits);
    val[0][aux_start] = 1;

    vector<ll> gamma(real_width), beta(real_width);
    for (int col = 0; col < real_width; ++col)
    {
        gamma[col] = fr_to_i128(val[0][gamma_offset + col]);
        beta[col] = fr_to_i128(val[0][beta_offset + col]);
    }
    vector<ll> row(padded_width, 0);
    vector<ll> centered(real_width, 0);
    for (int position = 0; position < real_rows; ++position)
    {
        if (sparsity_map && !sparsity_map[position * padded_width])
            continue;
        ll sum = 0;
        for (int col = 0; col < padded_width; ++col)
        {
            row[col] = fr_to_i128(val[0][input_offset +
                                         static_cast<i64>(position) * padded_width + col]);
            sum += row[col];
        }
        for (int col = 0; col < real_width; ++col)
        {
            centered[col] = row[col] * real_width - sum;
        }
        const auto variance = zkagent_integer::layernorm_variance(
            centered.data(), real_width, variance_bits);
        const ll sigma = variance.sigma;
        const ll sigma_upper = variance.upper;
        const ll sigma_lower = variance.lower;
        validate_nonnegative_bits_pair(
            sigma_upper, sigma_lower, sigma_range_bits,
            "LayerNorm sigma residual");
        const int sum_offset = aux_start + 10 + block_len + position;
        const int variance_offset = aux_start + 10 + block_len + len + position;
        const int sigma_offset = aux_start + 10 + block_len + 2 * len + position;
        const int sigma_upper_offset =
            aux_start + 10 + block_len + 3 * len + position;
        const int sigma_lower_offset =
            aux_start + 10 + block_len + 4 * len + position;
        val[0][sum_offset] = fr_from_i128(sum);
        val[0][variance_offset] = zkagent_integer::fr_from_bigint(variance.value);
        val[0][sigma_offset] = fr_from_i128(sigma);
        val[0][sigma_upper_offset] = fr_from_i128(sigma_upper);
        val[0][sigma_lower_offset] = fr_from_i128(sigma_lower);

        for (int col = 0; col < real_width; ++col)
        {
            const auto rounded = zkagent_integer::layernorm_round(
                coefficients, gamma[col], centered[col], beta[col], sigma);
            const ll q = rounded.value;
            if (ll_abs(q) > plan.output_bound)
            {
                throw std::runtime_error("cached LayerNorm output exceeds plan bound");
            }
            const int g = position * padded_width + col;
            const int compact = position * real_width + col;
            const int q_offset = aux_start + 10 + g;
            const int term1_offset =
                aux_start + 10 + block_len + 5 * len + compact;
            const int term2_offset = term1_offset + block_len;
            val[0][q_offset] = q;
            const ll term1 = rounded.lower, term2 = rounded.upper;
            validate_nonnegative_bits_pair(
                term1, term2, rounding_range_bits,
                "LayerNorm output rounding residual");
            val[0][term1_offset] = fr_from_i128(term1);
            val[0][term2_offset] = fr_from_i128(term2);
        }
    }
}

void GPT2::computeRoundWitness(int fc_id, const vector<F> &accumulator,
                               int q_output_offset, int out_width)
{
    if (q_output_offset < 20)
        throw std::runtime_error("invalid cached round offset");
    const int orgsize = q_output_offset - 20;
    const i64 block_len = static_cast<i64>(len) * out_width;
    val[0][orgsize] = 1;
    const FCRoundRequant &groups = fc_round_requants[fc_id];
    for (i64 g = 0; g < block_len; ++g)
    {
        const int col = static_cast<int>(g % out_width);
        const int group = (groups.group_count == 1)
                              ? 0
                              : std::min(col / groups.group_width,
                                         groups.group_count - 1);
        const RoundRequant &requant = groups.group[group];
        const ll a = fr_to_i128(accumulator[g]);
        if (ll_abs(a) > linearPlanForFc(fc_id, group).accumulator_bound)
        {
            throw std::runtime_error("cached affine accumulator exceeds plan bound");
        }
        const ll q = zkagent_integer::round_dyadic(a, requant);
        if (ll_abs(q) > linearPlanForFc(fc_id, group).output_bound)
            throw std::runtime_error("requantized linear output exceeds offline bound");
        const ll s = 2 * requant.c1 * a - 2 * requant.c2 * q + requant.c2;
        assert(s >= 0 && s < 2 * requant.c2);
        val[0][q_output_offset + g] = q;
        val[0][q_output_offset + block_len + g] = fr_from_i128(s);
    }
}

void GPT2::computeResidualWitness(const vector<F> &accumulator,
                                  int residual_offset, int output_offset,
                                  int width, const QuantResidualPlan &plan)
{
    const i64 block_len = static_cast<i64>(len) * width;
    const int left_q = output_offset - 4 * block_len;
    const int right_q = output_offset - 3 * block_len;
    const int left_s = output_offset - 2 * block_len;
    const int right_s = output_offset - block_len;
    const int orgsize = left_q - 20;
    if (orgsize < 0)
        throw std::runtime_error("invalid cached residual layout");
    val[0][orgsize] = 1;
    const RoundRequant left{plan.left.c1, plan.left.c2, plan.left.n,
                            plan.left.range_bits};
    const RoundRequant right{plan.right.c1, plan.right.c2, plan.right.n,
                             plan.right.range_bits};
    for (i64 g = 0; g < block_len; ++g)
    {
        const ll x = fr_to_i128(val[0][residual_offset + g]);
        const ll a = fr_to_i128(accumulator[g]);
        if (ll_abs(x) > plan.left_bound || ll_abs(a) > plan.right_bound)
        {
            throw std::runtime_error("cached residual operand exceeds plan bound");
        }
        const ll qx = zkagent_integer::round_dyadic(x, left);
        const ll qa = zkagent_integer::round_dyadic(a, right);
        const ll sx = 2 * left.c1 * x - 2 * left.c2 * qx + left.c2;
        const ll sa = 2 * right.c1 * a - 2 * right.c2 * qa + right.c2;
        assert(sx >= 0 && sx < 2 * left.c2);
        assert(sa >= 0 && sa < 2 * right.c2);
        const ll sum = qx + qa;
        if (ll_abs(sum) > plan.output_bound)
        {
            throw std::runtime_error("cached residual output exceeds plan bound");
        }
        val[0][left_q + g] = qx;
        val[0][right_q + g] = qa;
        val[0][left_s + g] = fr_from_i128(sx);
        val[0][right_s + g] = fr_from_i128(sa);
        val[0][output_offset + g] = sum;
    }
}

zkagent_integer::GeluCoefficients GPT2::computeGeluWitness(
    int input_offset, int aux_start, int real_width, int padded_width,
    const QuantBlockPlan &block)
{
    const auto coefficients = zkagent_integer::gelu_coefficients(
        block.gelu_scale_c, block.gelu_scale_e);
    const i64 block_len = static_cast<i64>(len) * padded_width;
    const int compact_len = len * real_width;
    const int y_offset = aux_start + 10;
    const int abs_offset = y_offset + block_len;
    const int selector_offset = abs_offset + block_len;
    const int d1_offset = selector_offset + block_len;
    const int d2_offset = d1_offset + compact_len;
    const int term1_advice_offset = d2_offset + compact_len;
    const int term2_advice_offset = term1_advice_offset + compact_len;
    val[0][aux_start] = 1;
    for (i64 g = 0; g < block_len; ++g)
    {
        if (g % padded_width >= real_width)
            continue;
        const ll qx = fr_to_i128(val[0][input_offset + g]);
        const auto result = zkagent_integer::gelu_round(coefficients, qx);
        const int compact = (g / padded_width) * real_width + g % padded_width;
        val[0][abs_offset + g] = fr_from_i128(result.magnitude);
        val[0][selector_offset + g] = fr_from_i128(result.selector);
        val[0][d1_offset + compact] = result.d1;
        val[0][d2_offset + compact] = result.d2;
        val[0][y_offset + g] = fr_from_i128(result.rounded.value);
        const ll term1 = result.rounded.lower, term2 = result.rounded.upper;
        validate_nonnegative_bits_pair(
            term1, term2,
            static_cast<unsigned>(block.gelu_rounding_range_bits),
            "cached GELU output rounding residual");
        val[0][term1_advice_offset + compact] = fr_from_i128(term1);
        val[0][term2_advice_offset + compact] = fr_from_i128(term2);
    }
    return coefficients;
}

void GPT2::ln_checker_layer1(layer &circuit, i64 &layer_id, int ln_id, int ey, int cy, int real_cn_in, int ch_in, bool *sparsity_map)
{
    (void)ey;
    (void)cy;
    i64 block_len = len * ch_in;
    const int compact_len = len * real_cn_in;
    const QuantLayerNormPlan &range_plan = layerNormPlan(ln_id);
    const bool q_bounded_input =
        circuitProfileSpec().force_immediate_linear_requant;
    const unsigned rounding_range_bits = static_cast<unsigned>(
        q_bounded_input ? range_plan.q_input_rounding_range_bits
                        : range_plan.rounding_range_bits);
    const unsigned sigma_range_bits = static_cast<unsigned>(
        q_bounded_input ? range_plan.q_input_sigma_range_bits
                        : range_plan.sigma_range_bits);
    int output_size = block_len + 3 * len;
    if (build_gates)
    {
        initLayer(circuit, output_size, layerType::LAYER_NORM_1);
        circuit.need_phase2 = true;
        circuit.zero_start_id = block_len + 2 * len;
    }

    int orgsize = val[0].size();
    val[0].resize(orgsize + 10 + block_len * 3 + 5 * len);
    for (int i = orgsize; i < val[0].size(); i++)
        val[0][i] = 0;
    ln_aux_start = orgsize;
    val[0][orgsize] = 1;
    total_in_size += 10 + block_len * 3 + 5 * len;

    computeLayerNormWitness(ln_id, q_offset, orgsize, real_cn_in, ch_in, len, sparsity_map);

    const int t1_base = orgsize + 10 + block_len + len * 5;
    const int t2_base = t1_base + block_len;
    recordPowerOfTwoRange("layernorm/" + std::to_string(ln_id) + "/round/lower",
                          t1_base, compact_len, rounding_range_bits);
    recordPowerOfTwoRange("layernorm/" + std::to_string(ln_id) + "/round/upper",
                          t2_base, compact_len, rounding_range_bits);
    recordPowerOfTwoRange("layernorm/" + std::to_string(ln_id) + "/sqrt/lower",
                          orgsize + 10 + block_len + len * 3,
                          len, sigma_range_bits);
    recordPowerOfTwoRange("layernorm/" + std::to_string(ln_id) + "/sqrt/upper",
                          orgsize + 10 + block_len + len * 4,
                          len, sigma_range_bits);

    if (build_gates)
    {

        for (i64 i = 0; i < len; i++)
        {
            if (sparsity_map && !sparsity_map[i * ch_in])
                continue;

            int sum_off = orgsize + 10 + block_len + i;
            int sig_off = orgsize + 10 + block_len + len * 2 + i;
            int b_off = orgsize + 10 + block_len + len + i;

            for (i64 co = 0; co < ch_in; ++co)
            {
                int g = matIdx(i, co, ch_in);
                if (sparsity_map && !sparsity_map[g])
                    continue;

                circuit.uni_gates.emplace_back(g, q_offset + g, 0, real_cn_in);
                circuit.uni_gates.emplace_back(g, sum_off, 0, -1);
                circuit.uni_gates.emplace_back(block_len + len * 2 + i, q_offset + g, 0, 1);
            }

            circuit.bin_gates.emplace_back(block_len + i, sig_off, sig_off, 1, 0);
            circuit.uni_gates.emplace_back(block_len + i, sig_off, 0, 1);
            circuit.uni_gates.emplace_back(block_len + i, b_off, 0, -1);

            circuit.bin_gates.emplace_back(block_len + len + i, sig_off, sig_off, -1, 0);
            circuit.uni_gates.emplace_back(block_len + len + i, sig_off, 0, 1);
            circuit.uni_gates.emplace_back(block_len + len + i, b_off, 0, 1);
            circuit.uni_gates.emplace_back(block_len + len + i, orgsize, 0, -1);
            circuit.uni_gates.emplace_back(block_len + len * 2 + i, sum_off, 0, -1);
        }
    }

    calcNormalLayer(circuit, layer_id);
    layer_id++;
}

void GPT2::ln_checker_layer2(layer &circuit, i64 &layer_id, int ln_id, int ey, int cy, int ch_in, bool *sparsity_map)
{
    (void)ey;
    (void)cy;
    int qw_off = layer_norm_w_q_start[ln_id];
    int qb_off = layer_norm_b_q_start[ln_id];
    int c1 = layer_norm_c1[ln_id], e1 = layer_norm_e1[ln_id];
    int c2 = layer_norm_c2[ln_id], e2 = layer_norm_e2[ln_id];
    const auto coefficients = zkagent_integer::layernorm_coefficients(c1, e1, c2, e2);

    i64 block_len = len * ch_in;
    int output_size = 2 * block_len + 3 * len;
    if (build_gates)
    {
        initLayer(circuit, output_size, layerType::LAYER_NORM_2);
        circuit.need_phase2 = true;
        circuit.zero_start_id = 2 * block_len;
    }

    int orgsize = ln_aux_start;

    if (build_gates)
    {
        for (i64 i = 0; i < len; i++)
        {
            int sig_off = orgsize + 10 + block_len + len * 2 + i;
            int b_off = orgsize + 10 + block_len + len + i;

            for (i64 co = 0; co < ch_in; ++co)
            {
                int g = matIdx(i, co, ch_in);
                int y_off = orgsize + 10 + g;
                if (sparsity_map && !sparsity_map[g])
                    continue;

                circuit.bin_gates.emplace_back(g, g, qw_off + co, -coefficients.a, 2);
                circuit.bin_gates.emplace_back(g, sig_off, qb_off + co, -coefficients.b, 0);
                circuit.bin_gates.emplace_back(g, sig_off, y_off, coefficients.denominator, 0);
                circuit.uni_gates.emplace_back(g, orgsize, 0, 1);
                circuit.uni_gates.emplace_back(g, sig_off, 0, (coefficients.denominator / 2));

                circuit.bin_gates.emplace_back(g + block_len, g, qw_off + co, coefficients.a, 2);
                circuit.bin_gates.emplace_back(g + block_len, sig_off, qb_off + co, coefficients.b, 0);
                circuit.bin_gates.emplace_back(g + block_len, sig_off, y_off, -(coefficients.denominator), 0);
                circuit.uni_gates.emplace_back(g + block_len, orgsize, 0, 1);
                circuit.uni_gates.emplace_back(g + block_len, sig_off, 0, (coefficients.denominator / 2));

                circuit.bin_gates.emplace_back(2 * block_len + 2 * len + i, g, g, 1, 1);
            }

            if (sparsity_map && !sparsity_map[i * ch_in])
                continue;

            int sigma_upper_off = orgsize + 10 + block_len + len * 3 + i;
            int sigma_lower_off = orgsize + 10 + block_len + len * 4 + i;
            circuit.uni_gates.emplace_back(2 * block_len + i,
                                           block_len + i, layer_id - 1, 1);
            circuit.uni_gates.emplace_back(2 * block_len + i,
                                           sigma_upper_off, 0, -1);
            circuit.uni_gates.emplace_back(2 * block_len + len + i,
                                           block_len + len + i, layer_id - 1, 1);
            circuit.uni_gates.emplace_back(2 * block_len + len + i,
                                           sigma_lower_off, 0, -1);

            circuit.uni_gates.emplace_back(2 * block_len + 2 * len + i, orgsize, 0, 1);
            circuit.uni_gates.emplace_back(2 * block_len + 2 * len + i, b_off, 0, -1);
        }
    }

    calcNormalLayer(circuit, layer_id);

    for (i64 i = 0; i < block_len; i++)
    {
        if (sparsity_map && !sparsity_map[i])
            continue;
        if (val[layer_id][i].isNegative())
        {
            cerr << "ERROR: ln_checker_layer2 val[" << layer_id << "][" << i << "] is negative" << endl;
        }
        if (val[layer_id][block_len + i].isNegative())
        {
            cerr << "ERROR: ln_checker_layer2 val[" << layer_id << "][" << (block_len + i) << "] is negative" << endl;
        }
    }
    int nonzero_count = 0;
    for (i64 i = 0; i < len; i++)
    {
        Fr &sigma_upper_check = val[layer_id][2 * block_len + i];
        Fr &sigma_lower_check = val[layer_id][2 * block_len + len + i];
        Fr &var_check = val[layer_id][2 * block_len + 2 * len + i];
        int b_off_dbg = orgsize + 10 + block_len + len + i;
        if (!var_check.isZero())
        {
            if (nonzero_count < 3)
            {
                ll B_val = fr_to_i128(val[0][b_off_dbg]);
                cerr << "ERROR: ln_checker_layer2 layer_id=" << layer_id
                     << " pos=" << i << " b_off=" << b_off_dbg
                     << " B=" << (long long)B_val
                     << " orgsize=" << orgsize << " block_len=" << block_len
                     << " var_check=" << var_check << endl;
                nonzero_count++;
            }
        }
        if ((!sigma_upper_check.isZero() || !sigma_lower_check.isZero()) &&
            nonzero_count < 3)
        {
            cerr << "ERROR: ln_checker_layer2 layer_id=" << layer_id
                 << " sigma bound check at pos " << i
                 << " upper=" << sigma_upper_check
                 << " lower=" << sigma_lower_check << endl;
            nonzero_count++;
        }
    }

    layer_id++;
}

void GPT2::ln_checker_layer3(layer &circuit, i64 &layer_id, int ln_id, int ey, int cy, int real_cn_in, int ch_in, bool *sparsity_map)
{
    i64 block_len = len * ch_in;
    int compact_len = len * real_cn_in;
    int output_size = 2 * compact_len;
    if (build_gates)
    {
        initLayer(circuit, output_size, layerType::LAYER_NORM_3);
        circuit.need_phase2 = false;
        circuit.zero_start_id = 0;
    }

    int orgsize = ln_aux_start;

    if (build_gates)
    {
        for (i64 i = 0; i < len; i++)
        {
            for (i64 co = 0; co < real_cn_in; ++co)
            {
                int g = matIdx(i, co, ch_in);
                int p = matIdx(i, co, real_cn_in);
                int t1_off = orgsize + 10 + block_len + len * 5 + p;
                int t2_off = t1_off + block_len;

                circuit.uni_gates.emplace_back(p, g, layer_id - 1, 1);
                circuit.uni_gates.emplace_back(p, t1_off, 0, -1);
                circuit.uni_gates.emplace_back(p + compact_len,
                                               g + block_len, layer_id - 1, 1);
                circuit.uni_gates.emplace_back(p + compact_len, t2_off, 0, -1);
            }
        }
    }

    calcNormalLayer(circuit, layer_id);
    for (int p = 0; p < output_size; ++p)
    {
        assert(val[layer_id][p].isZero());
    }
    layer_id++;

    q_offset = ln_aux_start + 10;
}

void GPT2::fullyConnLayer(layer &circuit, i64 &layer_id, int fc_id,
                          int x_offset, int x_layer, int ch_in, int ch_out)
{
    if (fc_id < 0 || static_cast<std::size_t>(fc_id) >= prepared_fc_witness_.size() ||
        x_layer != 0 || x_offset < 0 ||
        static_cast<std::size_t>(x_offset) > val[x_layer].size())
    {
        throw std::runtime_error("fresh FCONN witness has no prepared model/input span");
    }
    const auto &kernel = prepared_fc_witness_[fc_id];
    if (kernel.inputStride() != static_cast<std::size_t>(ch_in) ||
        kernel.outputStride() != static_cast<std::size_t>(ch_out))
    {
        throw std::runtime_error("fresh FCONN witness shape differs from prepared model");
    }
    const i64 size = static_cast<i64>(ch_out) * len;
    if (build_gates)
    {
        initLayer(circuit, size, layerType::FCONN);
        circuit.need_phase2 = true;
        const std::size_t input_count = static_cast<std::size_t>(len) * ch_in;
        if (input_count == 0 ||
            input_count > val[0].size() - static_cast<std::size_t>(x_offset) ||
            input_count > std::numeric_limits<u32>::max() ||
            input_count - 1 > std::numeric_limits<u32>::max() -
                                  static_cast<std::size_t>(x_offset))
            throw std::runtime_error("FCONN input tensor exceeds committed input storage");
        circuit.size_u[0] = static_cast<u32>(input_count);
        circuit.ori_id_u.resize(input_count);
        for (std::size_t i = 0; i < input_count; ++i)
            circuit.ori_id_u[i] = static_cast<u32>(x_offset + i);
    }
    if (circuit.size != static_cast<std::size_t>(size))
        throw std::runtime_error("fresh FCONN circuit output shape changed");

    const auto start = std::chrono::steady_clock::now();
    val[layer_id] = kernel.evaluate(val[x_layer].data() + x_offset,
                                    val[x_layer].size() - x_offset, len, threads);
    fresh_fc_timing_.witness_seconds += std::chrono::duration<double>(
                                            std::chrono::steady_clock::now() - start)
                                            .count();
    ++fresh_fc_timing_.evaluated_layers;
    ++layer_id;
}

void GPT2::roundLayer(layer &circuit, i64 &layer_id, int fc_id, int ch_out, bool *sparsity_map)
{
    assert(fc_id >= 0 && fc_id < (int)fc_round_requants.size());
    const FCRoundRequant &fc_requant = fc_round_requants[fc_id];
    i64 block_len = len * ch_out;

    i64 size = block_len;
    if (build_gates)
    {
        initLayer(circuit, size, layerType::RELU);

        circuit.need_phase2 = true;
        circuit.zero_start_id = 0;
    }

    int orgsize = val[0].size();
    val[0].resize(orgsize + 20 + block_len * 2);
    q_offset = orgsize + 20;
    total_in_size += 20 + block_len * 2;
    val[0][orgsize] = 1;

    computeRoundWitness(fc_id, val[layer_id - 1], q_offset, ch_out);

    if (build_gates)
    {
        for (i64 g = 0; g < block_len; ++g)
        {
            if (sparsity_map && !sparsity_map[g])
                continue;

            int p = g;
            int q = g + orgsize + 20;
            int one_idx = orgsize;
            int s = q + block_len;
            const int co = static_cast<int>(g % ch_out);
            int group = (fc_requant.group_count == 1)
                            ? 0
                            : std::min(co / fc_requant.group_width,
                                       fc_requant.group_count - 1);
            const RoundRequant &requant = fc_requant.group[group];
            if (requant.c1 > std::numeric_limits<ll>::max() / 2 ||
                requant.c2 > std::numeric_limits<ll>::max() / 2)
            {
                throw std::overflow_error("FC dyadic gate coefficient does not fit int64");
            }

            circuit.bin_gates.emplace_back(g, p, one_idx, 2 * requant.c1, 2);
            circuit.uni_gates.emplace_back(g, q, 0, -2 * requant.c2);
            circuit.uni_gates.emplace_back(g, one_idx, 0, requant.c2);
            circuit.uni_gates.emplace_back(g, s, 0, -1);
        }
    }

    const std::size_t residual_base =
        static_cast<std::size_t>(q_offset) + static_cast<std::size_t>(block_len);
    for (int group = 0; group < fc_requant.group_count; ++group)
    {
        std::vector<InputSpan> spans;
        if (fc_requant.group_count == 1)
        {
            spans.push_back(InputSpan(residual_base,
                                      static_cast<std::size_t>(block_len)));
        }
        else
        {
            const int column_start = group * fc_requant.group_width;
            const int column_count = group + 1 == fc_requant.group_count
                                         ? ch_out - column_start
                                         : fc_requant.group_width;
            for (int row = 0; row < len; ++row)
            {
                spans.push_back(InputSpan(
                    residual_base + static_cast<std::size_t>(row) * ch_out + column_start,
                    static_cast<std::size_t>(column_count)));
            }
        }
        range_query_specs.push_back(RangeQuerySpec::powerOfTwo(
            "linear/" + std::to_string(fc_id) + "/round/" +
                std::to_string(group),
            std::move(spans),
            static_cast<unsigned>(fc_requant.group[group].range_bits)));
    }

    calcNormalLayer(circuit, layer_id);
    for (i64 g = 0; g < block_len; ++g)
    {
        assert(val[layer_id][g].isZero());
    }
    layer_id++;
}

void GPT2::delayedResidualAddLayers(prover &pr, i64 &layer_id, int ch_out,
                                    int residual_offset, const QuantResidualPlan &plan)
{
    const i64 block_len = static_cast<i64>(len) * ch_out;
    const RoundRequant left{plan.left.c1, plan.left.c2, plan.left.n,
                            plan.left.range_bits};
    const RoundRequant right{plan.right.c1, plan.right.c2, plan.right.n,
                             plan.right.range_bits};
    if (left.c1 > std::numeric_limits<ll>::max() / 2 ||
        left.c2 > std::numeric_limits<ll>::max() / 2 ||
        right.c1 > std::numeric_limits<ll>::max() / 2 ||
        right.c2 > std::numeric_limits<ll>::max() / 2)
    {
        throw std::overflow_error("residual-alignment coefficient does not fit a gate scalar");
    }

    const int orgsize = static_cast<int>(val[0].size());
    const int one_idx = orgsize;
    const int left_q_offset = orgsize + 20;
    const int right_q_offset = left_q_offset + block_len;
    const int left_s_offset = right_q_offset + block_len;
    const int right_s_offset = left_s_offset + block_len;
    const int output_offset = right_s_offset + block_len;
    val[0].resize(orgsize + 20 + 5 * block_len);
    total_in_size += 20 + 5 * block_len;
    val[0][one_idx] = 1;

    computeResidualWitness(val[layer_id - 1], residual_offset, output_offset, ch_out, plan);

    layer &check = pr.C.circuit[layer_id];
    if (build_gates)
    {
        initLayer(check, 2 * block_len, layerType::RELU);
        check.need_phase2 = false;
        check.zero_start_id = 0;
        for (i64 g = 0; g < block_len; ++g)
        {
            check.uni_gates.emplace_back(g, residual_offset + g, 0, 2 * left.c1);
            check.uni_gates.emplace_back(g, left_q_offset + g, 0, -2 * left.c2);
            check.uni_gates.emplace_back(g, one_idx, 0, left.c2);
            check.uni_gates.emplace_back(g, left_s_offset + g, 0, -1);

            const i64 out = block_len + g;
            check.uni_gates.emplace_back(out, g, layer_id - 1, 2 * right.c1);
            check.uni_gates.emplace_back(out, right_q_offset + g, 0, -2 * right.c2);
            check.uni_gates.emplace_back(out, one_idx, 0, right.c2);
            check.uni_gates.emplace_back(out, right_s_offset + g, 0, -1);
        }
    }
    calcNormalLayer(check, layer_id);
    for (i64 g = 0; g < 2 * block_len; ++g)
        assert(val[layer_id][g].isZero());
    layer_id++;

    recordPowerOfTwoRange("residual/left/" + std::to_string(layer_id),
                          left_s_offset, static_cast<std::size_t>(block_len),
                          static_cast<unsigned>(left.range_bits));
    recordPowerOfTwoRange("residual/right/" + std::to_string(layer_id),
                          right_s_offset, static_cast<std::size_t>(block_len),
                          static_cast<unsigned>(right.range_bits));

    layer &add = pr.C.circuit[layer_id];
    if (build_gates)
    {
        initLayer(add, block_len, layerType::RELU);
        add.need_phase2 = false;
        add.zero_start_id = 0;
        for (i64 g = 0; g < block_len; ++g)
        {
            add.uni_gates.emplace_back(g, left_q_offset + g, 0, 1);
            add.uni_gates.emplace_back(g, right_q_offset + g, 0, 1);
            add.uni_gates.emplace_back(g, output_offset + g, 0, -1);
        }
    }
    calcNormalLayer(add, layer_id);
    for (i64 g = 0; g < block_len; ++g)
    {
        if (!val[layer_id][g].isZero())
        {
            throw std::runtime_error("residual addition witness mismatch");
        }
    }
    q_offset = output_offset;
    layer_id++;
}

void GPT2::multi_head_matrix_QK(layer &circuit, i64 &layer_id, int ch_out)
{
    const int HEAD = headnum;
    const int HSIZE = headdim;
    int output_size = HEAD * len * (len + 1) / 2;

    if (build_gates)
    {
        initLayer(circuit, output_size, layerType::MHA_QK);
        circuit.need_phase2 = true;
    }

    if (build_gates)
    {
        for (int head = 0; head < HEAD; head++)
        {
            int T = 0;
            for (int i = 0; i < len; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    int targ_gate = head * len * (len + 1) / 2 + T;
                    for (int k = 0; k < HSIZE; k++)
                    {
                        int col_i = head * HSIZE + k;
                        int col_j = HEAD * HSIZE + head * HSIZE + k;
                        int idi = i * ch_out + col_i;
                        int idj = j * ch_out + col_j;
                        int gate_i = q_offset + idi;
                        int gate_j = q_offset + idj;
                        circuit.bin_gates.emplace_back(targ_gate, gate_i, gate_j, 1, 0);
                    }
                    ++T;
                }
            }
        }
    }

    calcNormalLayer(circuit, layer_id);
    layer_id++;
}

void GPT2::softmax_layer_1(layer &circuit, i64 &layer_id, int ch_in,
                           int ch_out, int block_idx)
{
    const int HEAD = headnum;
    const int HSIZE = headdim;
    const int row_count = HEAD * len;
    const int score_count = HEAD * len * (len + 1) / 2;
    const int attention_count = HEAD * len * HSIZE;
    const QuantAttentionPlan &plan = quant_plan.blocks.at(block_idx).attention;

    const int orgsize = static_cast<int>(val[0].size());
    const AttentionAdviceLayout advice(orgsize, row_count, score_count, len * ch_in);
    const int input_size = advice.round_residual_base + attention_count - orgsize;

    val[0].resize(static_cast<size_t>(orgsize) + input_size, Fr(0));
    val[0][orgsize] = Fr(1);
    softmax_aux_start = orgsize;
    total_in_size += input_size;

    const PreparedAttentionSoftmax prepared_softmax(plan);
    const auto &middle_table = prepared_softmax.middleTable();

    const int output_size = attention_count + 2 * score_count + 2 * row_count;
    if (build_gates)
    {
        initLayer(circuit, output_size, layerType::SOFTMAX_1);
        circuit.need_phase2 = true;
        circuit.zero_start_id = attention_count + score_count;
    }

    const int tri = len * (len + 1) / 2;
    for (int head = 0; head < HEAD; ++head)
    {
        for (int row = 0; row < len; ++row)
        {
            const int row_index = head * len + row;
            std::vector<ll> scores(static_cast<size_t>(row + 1));
            for (int col = 0; col <= row; ++col)
            {
                const int score_index = head * tri + row * (row + 1) / 2 + col;
                scores[static_cast<size_t>(col)] = fr_to_i128(
                    val[layer_id - 1][score_index]);
            }
            AttentionSoftmaxRowWitness witness = prepared_softmax.build(scores);
            if (!witness.success)
            {
                throw std::runtime_error(
                    "attention Softmax row witness failed: " + witness.message);
            }
            advice.writeRow(val[0], row_index, witness);

            for (int col = 0; col <= row; ++col)
            {
                const int score_index = head * tri + row * (row + 1) / 2 + col;
                advice.writeScore(val[0], score_index, col, witness);
                for (int segment = 0; segment < 6; ++segment)
                {
                    const int offset = advice.digit_base[segment] + score_index;
                    recordAttentionLookup(
                        "attention/digit/" + std::to_string(
                                                 plan.digit_bits[static_cast<size_t>(segment)]),
                        plan.digit_bits[static_cast<size_t>(segment)], false,
                        offset, -1, {});
                }
                recordAttentionLookup(
                    "attention/middle/block/" + std::to_string(block_idx),
                    plan.digit_bits[3], true, advice.digit_base[3] + score_index,
                    advice.middle_base + score_index, middle_table);
                for (int high = 0; high < 2; ++high)
                {
                    const int segment = 4 + high;
                    const int bits = plan.digit_bits[static_cast<size_t>(segment)];
                    std::vector<ll> zero_table(static_cast<size_t>(1) << bits, 0);
                    zero_table[0] = 1;
                    recordAttentionLookup(
                        "attention/zero/" + std::to_string(bits), bits, true,
                        advice.digit_base[segment] + score_index,
                        (high == 0 ? advice.high0_base : advice.high1_base) + score_index,
                        zero_table);
                }
            }

            for (int d = 0; d < HSIZE; ++d)
            {
                ll weighted = 0;
                for (int col = 0; col <= row; ++col)
                {
                    const int score_index = head * tri + row * (row + 1) / 2 + col;
                    const int value_col = 2 * HEAD * HSIZE + head * HSIZE + d;
                    weighted += fr_to_i128(val[0][q_offset + col * ch_out + value_col]) * fr_to_i128(val[0][advice.probability_base + score_index]);
                }
                const int y_index = row * ch_in + head * HSIZE + d;
                const int compact = (head * len + row) * HSIZE + d;
                advice.writeOutput(val[0], y_index, compact, weighted, plan);
            }
        }
    }

    recordPowerOfTwoRange(
        "attention/" + std::to_string(block_idx) + "/mass/lower",
        advice.mass_lower_base, row_count, static_cast<unsigned>(plan.mass_bits));
    recordPowerOfTwoRange(
        "attention/" + std::to_string(block_idx) + "/mass/upper",
        advice.mass_upper_base, row_count, static_cast<unsigned>(plan.mass_bits));
    recordPowerOfTwoRange(
        "attention/" + std::to_string(block_idx) + "/output/round",
        advice.round_residual_base, attention_count,
        static_cast<unsigned>(plan.rounding_bits));

    if (build_gates)
    {
        for (int head = 0; head < HEAD; ++head)
        {
            for (int row = 0; row < len; ++row)
            {
                const int row_index = head * len + row;
                const ll epsilon = static_cast<ll>(
                    plan.epsilon_by_count[static_cast<size_t>(row)]);
                const int lower_check = attention_count + 2 * score_count + row_index;
                const int upper_check = lower_check + row_count;
                circuit.uni_gates.emplace_back(
                    lower_check, orgsize, 0, -(plan.theta - epsilon));
                circuit.uni_gates.emplace_back(
                    lower_check, advice.mass_lower_base + row_index, 0, -1);
                circuit.uni_gates.emplace_back(
                    upper_check, orgsize, 0, plan.theta + epsilon);
                circuit.uni_gates.emplace_back(
                    upper_check, advice.mass_upper_base + row_index, 0, -1);

                for (int col = 0; col <= row; ++col)
                {
                    const int score_index = head * tri + row * (row + 1) / 2 + col;
                    const int partial = attention_count + score_index;
                    const int reconstruction = attention_count + score_count + score_index;
                    circuit.bin_gates.emplace_back(
                        partial, advice.middle_base + score_index,
                        advice.high0_base + score_index, 1, 0);
                    circuit.uni_gates.emplace_back(
                        reconstruction, score_index, layer_id - 1, 1);
                    circuit.uni_gates.emplace_back(
                        reconstruction, advice.zeta_base + row_index, 0, -1);
                    for (int segment = 0; segment < 6; ++segment)
                    {
                        circuit.uni_gates.emplace_back(
                            reconstruction, advice.digit_base[segment] + score_index, 0,
                            attention_digit_weight(plan, segment));
                    }
                    circuit.uni_gates.emplace_back(
                        lower_check, advice.probability_base + score_index, 0, 1);
                    circuit.uni_gates.emplace_back(
                        upper_check, advice.probability_base + score_index, 0, -1);
                }

                for (int d = 0; d < HSIZE; ++d)
                {
                    const int output = (head * len + row) * HSIZE + d;
                    for (int col = 0; col <= row; ++col)
                    {
                        const int score_index = head * tri + row * (row + 1) / 2 + col;
                        const int value_col = 2 * HEAD * HSIZE + head * HSIZE + d;
                        circuit.bin_gates.emplace_back(
                            output, q_offset + col * ch_out + value_col,
                            advice.probability_base + score_index, 1, 0);
                    }
                }
            }
        }
    }

    calcNormalLayer(circuit, layer_id);
    for (int output = attention_count + score_count;
         output < output_size; ++output)
    {
        assert(val[layer_id][output].isZero());
    }
    layer_id++;
}

void GPT2::softmax_layer_2(layer &circuit, i64 &layer_id, int ch_in,
                           int block_idx)
{
    const int HEAD = headnum;
    const int HSIZE = headdim;
    const int row_count = HEAD * len;
    const int score_count = HEAD * len * (len + 1) / 2;
    const int attention_count = HEAD * len * HSIZE;
    const QuantAttentionPlan &plan = quant_plan.blocks.at(block_idx).attention;
    const int orgsize = softmax_aux_start;
    const AttentionAdviceLayout advice(orgsize, row_count, score_count, len * ch_in);

    const int output_size = score_count + attention_count;
    if (build_gates)
    {
        initLayer(circuit, output_size, layerType::SOFTMAX_2);
        circuit.need_phase2 = true;
        circuit.zero_start_id = score_count;
        for (int score = 0; score < score_count; ++score)
        {
            circuit.bin_gates.emplace_back(
                score, attention_count + score,
                advice.high1_base + score, 1, 2);
        }
        for (int output = 0; output < attention_count; ++output)
        {
            const int head = output / (len * HSIZE);
            const int remainder = output % (len * HSIZE);
            const int row = remainder / HSIZE;
            const int d = remainder % HSIZE;
            const int y_index = row * ch_in + head * HSIZE + d;
            const int check = score_count + output;
            circuit.uni_gates.emplace_back(check, output, layer_id - 1, 2);
            circuit.uni_gates.emplace_back(
                check, advice.y_base + y_index, 0, -2 * plan.theta);
            circuit.uni_gates.emplace_back(check, orgsize, 0, plan.theta);
            circuit.uni_gates.emplace_back(
                check, advice.round_residual_base + output, 0, -1);
        }
    }
    calcNormalLayer(circuit, layer_id);
    for (int output = score_count; output < output_size; ++output)
    {
        assert(val[layer_id][output].isZero());
    }
    layer_id++;
}

void GPT2::softmax_layer_3(layer &circuit, i64 &layer_id, int block_idx)
{
    (void)block_idx;
    const int HEAD = headnum;
    const int row_count = HEAD * len;
    const int score_count = HEAD * len * (len + 1) / 2;
    const int orgsize = softmax_aux_start;
    const int base = orgsize + 10;
    const int probability_base = base + row_count + 9 * score_count;
    const int y_base = probability_base + score_count + 2 * row_count;

    if (build_gates)
    {
        initLayer(circuit, score_count, layerType::SOFTMAX_3);
        circuit.need_phase2 = false;
        circuit.zero_start_id = 0;
        for (int score = 0; score < score_count; ++score)
        {
            circuit.uni_gates.emplace_back(score, score, layer_id - 1, 1);
            circuit.uni_gates.emplace_back(
                score, probability_base + score, 0, -1);
        }
    }
    calcNormalLayer(circuit, layer_id);
    for (int score = 0; score < score_count; ++score)
    {
        assert(val[layer_id][score].isZero());
    }
    layer_id++;
    q_offset = y_base;
}

void GPT2::gelu_checker_layer1(layer &circuit, i64 &layer_id, int block_idx,
                               int real_cn_out, int ch_out)
{
    const QuantBlockPlan &gelu_plan = quant_plan.blocks.at(block_idx);
    i64 block_len = len * ch_out;
    int compact_len = len * real_cn_out;
    int output_size = 6 * block_len;
    if (build_gates)
    {
        initLayer(circuit, output_size, layerType::GELU_1);
        circuit.need_phase2 = true;
        circuit.zero_start_id = block_len * 2;
    }

    int orgsize = val[0].size();
    val[0].resize(orgsize + 10 + block_len * 3 + compact_len * 4);
    for (int i = orgsize; i < val[0].size(); i++)
        val[0][i] = 0;
    gelu_aux_start = orgsize;
    total_in_size += 10 + compact_len * 4 + block_len * 3;

    const auto coefficients = computeGeluWitness(q_offset, orgsize, real_cn_out, ch_out, gelu_plan);
    const ll C6 = coefficients.c6, C7 = coefficients.c7;
    val[0][orgsize] = 1;

    const int d1_base = orgsize + 10 + block_len * 3;
    const int d2_base = d1_base + compact_len;
    const int term1_base = d2_base + compact_len;
    const int term2_base = term1_base + compact_len;
    recordPowerOfTwoRange("gelu/" + std::to_string(block_idx) + "/d1",
                          d1_base, compact_len,
                          gelu_plan.gelu_d1_range_bits);
    recordPowerOfTwoRange("gelu/" + std::to_string(block_idx) + "/d2",
                          d2_base, compact_len,
                          gelu_plan.gelu_d2_range_bits);
    recordPowerOfTwoRange("gelu/" + std::to_string(block_idx) + "/round/lower",
                          term1_base, compact_len,
                          gelu_plan.gelu_rounding_range_bits);
    recordPowerOfTwoRange("gelu/" + std::to_string(block_idx) + "/round/upper",
                          term2_base, compact_len,
                          gelu_plan.gelu_rounding_range_bits);

    if (build_gates)
    {
        for (i64 g = 0; g < block_len; ++g)
        {
            if (g % ch_out >= real_cn_out)
                continue;

            int gp = g / ch_out * real_cn_out + g % ch_out;
            int abs_off = orgsize + 10 + block_len + g;
            int t_off = orgsize + 10 + block_len * 2 + g;
            int d1_off = orgsize + 10 + block_len * 3 + gp;
            int d2_off = orgsize + 10 + block_len * 3 + len * real_cn_out + gp;
            int c1 = orgsize;
            int q_off = g + q_offset;

            int abs_mult_t_off = g;
            int q_square_off = g + block_len;
            int dt1_off = g + block_len * 2;
            int dt2_off = g + block_len * 3;
            int t2_off = g + block_len * 4;
            int abs_check_off = g + block_len * 5;

            circuit.bin_gates.emplace_back(abs_mult_t_off, abs_off, t_off, 1, 0);
            circuit.bin_gates.emplace_back(q_square_off, q_off, q_off, 1, 0);

            circuit.uni_gates.emplace_back(dt1_off, abs_off, 0, 1);
            circuit.uni_gates.emplace_back(dt1_off, c1, 0, 1);
            circuit.uni_gates.emplace_back(dt1_off, d1_off, 0, -1);

            circuit.uni_gates.emplace_back(dt2_off, c1, 0, -C6);
            circuit.uni_gates.emplace_back(dt2_off, t_off, 0, 2ll * C6 + 1);
            circuit.uni_gates.emplace_back(dt2_off, abs_off, 0, C7);
            circuit.bin_gates.emplace_back(dt2_off, abs_off, t_off, -2ll * C7, 0);
            circuit.uni_gates.emplace_back(dt2_off, d2_off, 0, -1);

            circuit.bin_gates.emplace_back(t2_off, t_off, t_off, 1, 0);
            circuit.uni_gates.emplace_back(t2_off, t_off, 0, -1);

            circuit.bin_gates.emplace_back(abs_check_off, q_off, q_off, 1, 0);
            circuit.bin_gates.emplace_back(abs_check_off, abs_off, abs_off, -1, 0);
        }
    }

    calcNormalLayer(circuit, layer_id);

    layer_id++;
}

void GPT2::gelu_checker_layer2(layer &circuit, i64 &layer_id, int block_idx,
                               int real_cn_out, int ch_out)
{
    const auto &plan = quant_plan.blocks.at(block_idx);
    const auto c = zkagent_integer::gelu_coefficients(plan.gelu_scale_c, plan.gelu_scale_e);
    const ll C1 = c.c1, C2 = c.c2, C3 = c.c3, C4 = c.c4, C5 = c.c5;

    i64 block_len = len * ch_out;
    int output_size = 2 * block_len;
    if (build_gates)
    {
        initLayer(circuit, output_size, layerType::GELU_2);
        circuit.need_phase2 = true;
    }

    int orgsize = gelu_aux_start;

    for (i64 g = 0; g < block_len; ++g)
    {
        if (g % ch_out >= real_cn_out)
            continue;

        int y_off = orgsize + 10 + g;
        int abs_off = orgsize + 10 + block_len + g;
        int t_off = orgsize + 10 + block_len * 2 + g;
        int c1 = orgsize;
        int q_off = g + q_offset;

        int term1_off = g;
        int term2_off = g + block_len;
        int abs_mult_t_off = g;
        int q_square_off = g + block_len;

        if (build_gates)
        {
            circuit.uni_gates.emplace_back(term1_off, c1, 0, C5 + 1);
            circuit.uni_gates.emplace_back(term1_off, y_off, 0, C5 * 2);
            circuit.uni_gates.emplace_back(term1_off, abs_off, 0, -C1);
            circuit.uni_gates.emplace_back(term1_off, q_off, 0, -C1);
            circuit.uni_gates.emplace_back(term1_off, abs_mult_t_off, layer_id - 1, C4);
            circuit.bin_gates.emplace_back(term1_off, abs_mult_t_off, q_square_off, C2, 1);
            circuit.bin_gates.emplace_back(term1_off, q_square_off, t_off, -C3, 2);

            circuit.uni_gates.emplace_back(term2_off, c1, 0, C5 + 1);
            circuit.uni_gates.emplace_back(term2_off, y_off, 0, -C5 * 2);
            circuit.uni_gates.emplace_back(term2_off, abs_off, 0, C1);
            circuit.uni_gates.emplace_back(term2_off, q_off, 0, C1);
            circuit.uni_gates.emplace_back(term2_off, abs_mult_t_off, layer_id - 1, -C4);
            circuit.bin_gates.emplace_back(term2_off, abs_mult_t_off, q_square_off, -C2, 1);
            circuit.bin_gates.emplace_back(term2_off, q_square_off, t_off, C3, 2);
        }
    }

    calcNormalLayer(circuit, layer_id);
    layer_id++;
}

void GPT2::gelu_checker_layer3(layer &circuit, i64 &layer_id, int real_cn_out, int ch_out)
{
    i64 block_len = len * ch_out;
    int compact_len = len * real_cn_out;
    int output_size = 2 * compact_len;
    if (build_gates)
    {
        initLayer(circuit, output_size, layerType::GELU_3);
        circuit.need_phase2 = false;
        circuit.zero_start_id = 0;
    }

    int orgsize = gelu_aux_start;

    if (build_gates)
    {
        for (i64 g = 0; g < block_len; ++g)
        {
            if (g % ch_out >= real_cn_out)
                continue;

            int gp = g / ch_out * real_cn_out + g % ch_out;
            int term1_advice_off =
                orgsize + 10 + block_len * 3 + 2 * compact_len + gp;
            int term2_advice_off = term1_advice_off + compact_len;

            circuit.uni_gates.emplace_back(gp, g, layer_id - 1, 1);
            circuit.uni_gates.emplace_back(gp, term1_advice_off, 0, -1);
            circuit.uni_gates.emplace_back(
                gp + compact_len, g + block_len, layer_id - 1, 1);
            circuit.uni_gates.emplace_back(
                gp + compact_len, term2_advice_off, 0, -1);
        }
    }

    calcNormalLayer(circuit, layer_id);
    for (int p = 0; p < output_size; ++p)
    {
        assert(val[layer_id][p].isZero());
    }
    layer_id++;
    q_offset = gelu_aux_start + 10;
}

namespace
{
    int merge_bucket(layerType t)
    {
        switch (t)
        {
        case layerType::MHA_QK:
        case layerType::GELU_1:
        case layerType::LAYER_NORM_1:
            return 1;
        case layerType::SOFTMAX_1:
        case layerType::GELU_2:
        case layerType::LAYER_NORM_2:
            return 2;
        case layerType::SOFTMAX_2:
        case layerType::GELU_3:
        case layerType::LAYER_NORM_3:
            return 3;
        case layerType::SOFTMAX_3:
            return 4;
        default:
            return -1;
        }
    }

    bool layer_depends_on_prev(const layer &cur)
    {
        for (const auto &g : cur.uni_gates)
        {
            if (g.lu != 0)
                return true;
        }
        for (const auto &g : cur.bin_gates)
        {
            if (g.l != 0)
                return true;
        }
        return false;
    }

    bool layer_has_invalid_dep(const layer &cur, int layer_idx)
    {
        for (const auto &g : cur.uni_gates)
        {
            if (g.lu != 0 && g.lu != static_cast<u32>(layer_idx - 1))
                return true;
        }
        for (const auto &g : cur.bin_gates)
        {
            if (g.l != 0 && g.l != 1 && g.l != 2)
                return true;
        }
        return false;
    }

    void prepare_gpt2_gate_intervals(layeredCircuit &circuit)
    {
        for (int i = 1; i < circuit.size; ++i)
        {
            layer &current = circuit.circuit[i];
            if (merge_bucket(current.ty) < 0)
                continue;
            prepareParallelGateIntervals(current);
        }
    }
}

void GPT2::merge_layer(prover &pr, i64 layer_id)
{
    if (layer_id <= 1)
        return;

    vector<int> bucket_of(layer_id, -1);
    vector<char> mergeable(layer_id, 0);
    for (int i = 1; i < layer_id; i++)
    {
        bucket_of[i] = merge_bucket(pr.C.circuit[i].ty);
    }

    for (int i = 1; i < layer_id; i++)
    {
        int bucket = bucket_of[i];
        if (bucket < 0)
            continue;
        const layer &cur = pr.C.circuit[i];
        if (layer_has_invalid_dep(cur, i))
            continue;
        bool needs_prev = layer_depends_on_prev(cur);
        if (needs_prev)
        {
            int prev = i - 1;
            if (prev <= 0)
                continue;
            if (bucket_of[prev] != bucket - 1)
                continue;
        }
        mergeable[i] = 1;
    }

    for (int i = 1; i < layer_id; i++)
    {
        if (!mergeable[i])
            continue;
        if (!layer_depends_on_prev(pr.C.circuit[i]))
            continue;
        if (!mergeable[i - 1])
            mergeable[i] = 0;
    }

    for (int i = 1; i < layer_id; i++)
    {
        if (mergeable[i])
            continue;
        const layer &cur = pr.C.circuit[i];
        bool needs_prev = false;
        for (const auto &g : cur.uni_gates)
        {
            if (g.lu != 0 && g.lu == static_cast<u32>(i - 1))
            {
                needs_prev = true;
                break;
            }
        }
        if (!needs_prev)
        {
            for (const auto &g : cur.bin_gates)
            {
                if (g.l != 0)
                {
                    needs_prev = true;
                    break;
                }
            }
        }
        if (needs_prev && mergeable[i - 1])
        {
            cerr << "merge_layer: unsafe dependency at layer " << i << ", skip merge" << endl;
            return;
        }
    }

    vector<vector<int>> bucket_layers(5);
    vector<int> active_len(layer_id, 0);
    vector<int> active_offset(layer_id, 0);
    vector<int> zero_offset(layer_id, 0);
    int bucket_sizes[5] = {0, 0, 0, 0, 0};
    int bucket_active[5] = {0, 0, 0, 0, 0};
    int bucket_zero[5] = {0, 0, 0, 0, 0};
    for (int i = 1; i < layer_id; i++)
    {
        if (!mergeable[i])
            continue;
        int bucket = bucket_of[i];
        const layer &cur = pr.C.circuit[i];
        int size = cur.size;
        int active = cur.zero_start_id;
        if (active < 0)
            active = 0;
        if (active > size)
            active = size;
        int zero = size - active;
        active_len[i] = active;
        active_offset[i] = bucket_active[bucket];
        zero_offset[i] = bucket_zero[bucket];
        bucket_active[bucket] += active_len[i];
        bucket_zero[bucket] += zero;
        bucket_sizes[bucket] += size;
        bucket_layers[bucket].push_back(i);
    }

    bool any_merge = false;
    for (int b = 1; b <= 4; b++)
    {
        if (!bucket_layers[b].empty())
        {
            any_merge = true;
            break;
        }
    }
    if (!any_merge)
        return;

    auto map_index = [&](int layer_idx, int idx) -> int
    {
        int bucket = bucket_of[layer_idx];
        if (bucket <= 0)
            return idx;
        int active = active_len[layer_idx];
        if (idx < active)
            return active_offset[layer_idx] + idx;
        return bucket_active[bucket] + zero_offset[layer_idx] + (idx - active);
    };

    vector<layer> new_circuit;
    vector<vector<F>> new_val;
    new_circuit.reserve(layer_id);
    new_val.reserve(layer_id);

    new_circuit.push_back(pr.C.circuit[0]);
    new_val.push_back(pr.val[0]);

    vector<int> new_index(layer_id, -1);
    new_index[0] = 0;

    for (int i = 1; i < layer_id; i++)
    {
        if (mergeable[i])
            continue;
        layer copied = pr.C.circuit[i];
        for (auto &g : copied.uni_gates)
        {
            if (g.lu != 0)
            {
                int mapped = (g.lu < new_index.size()) ? new_index[g.lu] : -1;
                if (mapped < 0)
                {
                    cerr << "merge_layer: unexpected uni_gate dependency at layer " << i << endl;
                    return;
                }
                g.lu = mapped;
            }
        }
        new_index[i] = new_circuit.size();
        new_circuit.push_back(std::move(copied));
        new_val.push_back(pr.val[i]);
    }

    vector<int> bucket_new_index(5, -1);
    for (int b = 1; b <= 4; b++)
    {
        if (bucket_layers[b].empty())
            continue;

        bucket_new_index[b] = new_circuit.size();
        layer merged;
        merged.ty = pr.C.circuit[bucket_layers[b][0]].ty;
        merged.need_phase2 = false;
        initLayer(merged, bucket_sizes[b], merged.ty);
        merged.zero_start_id = bucket_active[b];

        vector<F> merged_val(bucket_sizes[b]);

        for (int idx : bucket_layers[b])
        {
            const layer &src = pr.C.circuit[idx];
            if (src.need_phase2)
                merged.need_phase2 = true;
            const auto &src_val = pr.val[idx];
            for (int j = 0; j < src.size; ++j)
            {
                int mapped = map_index(idx, j);
                merged_val[mapped] = src_val[j];
            }

            for (auto g : src.uni_gates)
            {
                g.g = map_index(idx, g.g);
                if (g.lu != 0)
                {
                    int prev = static_cast<int>(g.lu);
                    int prev_bucket = bucket_of[prev];
                    if (prev_bucket <= 0 || bucket_new_index[prev_bucket] < 0)
                    {
                        cerr << "merge_layer: invalid uni_gate prev mapping at layer " << idx << endl;
                        return;
                    }
                    g.u = map_index(prev, g.u);
                    g.lu = bucket_new_index[prev_bucket];
                }
                merged.uni_gates.push_back(g);
            }

            for (auto g : src.bin_gates)
            {
                g.g = map_index(idx, g.g);
                if (g.l != 0)
                {
                    int prev = idx - 1;
                    int prev_bucket = bucket_of[prev];
                    if (prev_bucket <= 0 || bucket_new_index[prev_bucket] < 0)
                    {
                        cerr << "merge_layer: invalid bin_gate prev mapping at layer " << idx << endl;
                        return;
                    }
                    g.u = map_index(prev, g.u);
                    if (g.l & 1)
                        g.v = map_index(prev, g.v);
                }
                merged.bin_gates.push_back(g);
            }
        }

        merged.bin_gates_v = merged.bin_gates;
        new_circuit.push_back(std::move(merged));
        new_val.push_back(std::move(merged_val));
    }

    pr.C.circuit.swap(new_circuit);
    pr.val.swap(new_val);
    pr.C.size = pr.C.circuit.size();
}

void GPT2::prepareCircuitData(prover &pr)
{
    fresh_fc_timing_ = FreshFcTiming{};
    pr.prepared_model_values = false;
    prepared_gate_schedules_.clear();
    prepared_gate_owner_ = nullptr;
    prepared_attention_ready_ = false;
    if (!quant_plan.loaded())
    {
        throw std::runtime_error(
            "GPT-2 circuit requires model_dir/quantized/quant_plan.txt");
    }
    initParam(pr);

    pr.C.init(220, SIZE);
    pr.val.resize(SIZE);
    val = pr.val.begin();

    i64 layer_id = 0;
    inputLayer(pr.C.circuit[layer_id], full_conn[0].channel_in);

    prepareWeights(pr);
}

void GPT2::buildCircuit(prover &pr, bool merge)
{
    pr.prepared_model_values = false;
    prepared_gate_schedules_.clear();
    prepared_gate_owner_ = nullptr;
    prepared_attention_ready_ = false;
    cout << (build_gates ? "Building circuit structure..." : "Recomputing circuit witness from cache...")
         << endl;
    const GPT2CircuitProfileSpec &profile = circuitProfileSpec();

    fc_round_requants.clear();
    fc_round_offsets.clear();
    fc_input_scales.clear();
    block_attn_residual_offsets.clear();
    block_ffn_residual_offsets.clear();
    range_query_specs.clear();
    attention_lookup_groups.clear();
    i64 layer_id = 1;

    initQuantPolicy(pr);

    int block_residual_offset = 0;
    int ffn_residual_offset = 0;

    for (int i = 0; i < full_conn.size(); ++i)
    {
        auto &fc = full_conn[i];
        int ch_in = fc.channel_in;
        int ch_out = fc.channel_out;

        if (i == 0)
        {
            q_offset = 0;
        }

        if (i % 4 == 0)
        {
            block_residual_offset = q_offset;
        }

        bool *sparsity = new bool[pr.fc_input_row[i] * pr.fc_input_col[i]];
        memset(sparsity, 0, sizeof(bool) * pr.fc_input_row[i] * pr.fc_input_col[i]);
        for (int j = 0; j < pr.fc_real_input_row[i]; j++)
        {
            for (int k = 0; k < pr.fc_real_input_col[i]; k++)
            {
                sparsity[j * pr.fc_input_col[i] + k] = true;
            }
        }

        int block_idx = i / 4;

        if (i % 4 == 0 || i % 4 == 2)
        {

            bool is_ln2 = (i % 4 == 2);
            int ln_id = block_idx * 2 + (is_ln2 ? 1 : 0);

            ln_checker_layer1(pr.C.circuit[layer_id], layer_id, ln_id, 0, 1,
                              pr.fc_real_input_col[i], ch_in, sparsity);

            ln_checker_layer2(pr.C.circuit[layer_id], layer_id, ln_id, 0, 1, ch_in, sparsity);
            ln_checker_layer3(pr.C.circuit[layer_id], layer_id, ln_id, 0, 1,
                              pr.fc_real_input_col[i], ch_in, sparsity);
        }
        fc_input_scales.push_back({0, 0, linearPlanForFc(i).input_scale});

        pr.fc_input_id[i] = q_offset;
        fullyConnLayer(pr.C.circuit[layer_id], layer_id, i, q_offset, 0,
                       ch_in, ch_out);

        const QuantLinearPlan &linear_plan = linearPlanForFc(i);
        if (linear_plan.mode == "immediate" ||
            profile.force_immediate_linear_requant)
        {
            roundLayer(pr.C.circuit[layer_id], layer_id, i, ch_out);
            fc_round_offsets.push_back(q_offset);
        }
        else
        {
            fc_round_offsets.push_back(-1);
        }

        if (profile.include_skip_connections && i % 4 == 1)
        {
            delayedResidualAddLayers(pr, layer_id, ch_out, block_residual_offset,
                                     quant_plan.blocks[block_idx].attn_residual);

            ffn_residual_offset = q_offset;
            block_attn_residual_offsets.push_back(q_offset);
        }

        if (profile.include_skip_connections && i % 4 == 3)
        {
            delayedResidualAddLayers(pr, layer_id, ch_out, ffn_residual_offset,
                                     quant_plan.blocks[block_idx].ffn_residual);
            block_ffn_residual_offsets.push_back(q_offset);
        }

        if (i % 4 == 0)
        {

            multi_head_matrix_QK(pr.C.circuit[layer_id], layer_id, ch_out);

            softmax_layer_1(
                pr.C.circuit[layer_id], layer_id, ch_in, ch_out, block_idx);
            softmax_layer_2(
                pr.C.circuit[layer_id], layer_id, ch_in, block_idx);
            softmax_layer_3(
                pr.C.circuit[layer_id], layer_id, block_idx);
        }

        if (i % 4 == 2)
        {
            gelu_checker_layer1(pr.C.circuit[layer_id], layer_id, block_idx,
                                pr.fc_real_col[i], ch_out);
            gelu_checker_layer2(pr.C.circuit[layer_id], layer_id, block_idx,
                                pr.fc_real_col[i], ch_out);
            gelu_checker_layer3(pr.C.circuit[layer_id], layer_id, pr.fc_real_col[i], ch_out);
        }

        delete[] sparsity;
    }

    if (!full_conn.empty() && profile.include_final_layer_norm)
    {
        int last_fc = (int)full_conn.size() - 1;
        int ln_ch_in = pr.fc_col[last_fc];
        int ln_real_ch_in = pr.fc_real_col[last_fc];
        int ln_real_rows = pr.fc_real_input_row[last_fc];
        int ln_rows = pr.fc_input_row[last_fc];

        bool *ln_sparsity = new bool[ln_rows * ln_ch_in];
        memset(ln_sparsity, 0, sizeof(bool) * ln_rows * ln_ch_in);
        for (int row = 0; row < ln_real_rows; row++)
        {
            for (int col = 0; col < ln_real_ch_in; col++)
            {
                ln_sparsity[row * ln_ch_in + col] = true;
            }
        }

        int ln_id = layer_num * 2;
        ln_checker_layer1(pr.C.circuit[layer_id], layer_id, ln_id, 0, 1,
                          ln_real_ch_in, ln_ch_in, ln_sparsity);
        ln_checker_layer2(pr.C.circuit[layer_id], layer_id, ln_id, 0, 1, ln_ch_in, ln_sparsity);
        ln_checker_layer3(pr.C.circuit[layer_id], layer_id, ln_id, 0, 1,
                          ln_real_ch_in, ln_ch_in, ln_sparsity);

        delete[] ln_sparsity;
    }

    channel_in = full_conn.back().channel_in;
    channel_out = full_conn.back().channel_out;

    if (profile.include_lm_head)
    {
        buildLMHeadCircuit(pr, layer_id);
    }
    else
    {
        pr.lm_fc_id = -1;
        lm_head_output_offset = 0;
        embed_matrix_circuit_offset = 0;
    }

    if (build_gates)
    {
        pr.C.circuit.resize(static_cast<size_t>(layer_id));
        pr.val.resize(static_cast<size_t>(layer_id));
        pr.C.size = static_cast<int>(layer_id);
        val = pr.val.begin();
    }

    if (build_gates && merge)
    {
        merge_layer(pr, layer_id);
        val = pr.val.begin();
    }

    const auto input_layout = inputCommitmentLayout();
    if (static_cast<std::size_t>(total_in_size) != input_layout.value_count)
        throw std::logic_error("input layout allocation count drifted from circuit construction");
    if (build_gates)
    {
        initLayer(pr.C.circuit[0], total_in_size, layerType::INPUT);
    }

    if (val[0].size() < (size_t)total_in_size)
    {
        size_t old_size = val[0].size();
        val[0].resize(total_in_size);

        for (size_t i = old_size; i < (size_t)total_in_size; i++)
        {
            val[0][i] = 0;
        }
    }

    if (build_gates)
    {
        pr.C.initSubset();
    }
    pr.mat_val = mat_values;

    std::string profile_error;
    if (!validateCircuitProfile(pr, &profile_error))
    {
        throw std::runtime_error("circuit profile validation failed: " +
                                 profile_error);
    }

    if (build_gates && merge)
        prepare_gpt2_gate_intervals(pr.C);
}

void GPT2::buildLMHeadCircuit(prover &pr, i64 &layer_id)
{
    if (!quant_plan.loaded())
    {
        throw std::runtime_error("LM head requires an offline quantization plan");
    }
    int transformer_output_offset = q_offset;

    int lm_fc_id = (int)full_conn.size();
    if (lm_fc_id >= model_limits::FconnCapacity)
    {
        throw std::out_of_range("LM-head FC id exceeds model metadata capacity");
    }
    pr.lm_fc_id = lm_fc_id;

    int padded_len = len;
    int padded_ch_in = channel_out;
    int padded_vocab = 1 << ceilPow2BitLength(vocab_size);

    if (g_debug)
    {
        cerr << "[LM Head] lm_fc_id=" << lm_fc_id
             << ", padded_len=" << padded_len
             << ", padded_ch_in=" << padded_ch_in
             << ", padded_vocab=" << padded_vocab
             << ", attn_dim=" << attn_dim
             << ", vocab_size=" << vocab_size
             << ", channel_in=" << channel_in
             << ", channel_out=" << channel_out
             << ", q_offset=" << q_offset
             << ", transformer_output_offset=" << transformer_output_offset << endl;
    }

    pr.fc_real_input_row[lm_fc_id] = len;
    pr.fc_real_input_col[lm_fc_id] = attn_dim;
    pr.fc_real_row[lm_fc_id] = attn_dim;
    pr.fc_real_col[lm_fc_id] = vocab_size;

    pr.fc_input_row[lm_fc_id] = padded_len;
    pr.fc_input_col[lm_fc_id] = padded_ch_in;
    pr.fc_row[lm_fc_id] = padded_ch_in;
    pr.fc_col[lm_fc_id] = padded_vocab;
    pr.fc_input_id[lm_fc_id] = transformer_output_offset;
    pr.fc_bias_id[lm_fc_id] = -1;

    const auto input_layout = inputCommitmentLayout();
    const auto &head_rows = input_layout.fixed_rows.at(1);
    const std::size_t head_begin = head_rows.first * input_layout.columns();
    const std::size_t head_end = (head_rows.first + head_rows.count) * input_layout.columns();
    if (val[0].size() > head_begin)
        throw std::logic_error("LM-head input layout overlap");
    val[0].resize(head_begin, Fr(0));
    total_in_size = head_begin;
    embed_matrix_circuit_offset = head_begin;
    i64 embed_size = (i64)padded_vocab * padded_ch_in;
    val[0].resize(embed_matrix_circuit_offset + embed_size);
    total_in_size += embed_size;

    if (mat_values[lm_fc_id])
        delete[] mat_values[lm_fc_id];
    mat_values[lm_fc_id] = new int[embed_size];
    const int thread_n = std::max(1, threads);
#pragma omp parallel for num_threads(thread_n) schedule(static)
    for (int v = 0; v < padded_vocab; ++v)
    {
        const i64 row = static_cast<i64>(v) * padded_ch_in;
        for (int j = 0; j < padded_ch_in; ++j)
        {
            const int q = v < vocab_size && j < attn_dim
                              ? quant_plan.lm_head.weight[static_cast<i64>(v) * attn_dim + j]
                              : 0;
            mat_values[lm_fc_id][row + j] = q;
            val[0][embed_matrix_circuit_offset + row + j] = q;
        }
    }
    pr.fc_start_id[lm_fc_id] = embed_matrix_circuit_offset;

    val[0].resize(head_end, Fr(0));
    total_in_size = head_end;

    const auto kernel_start = std::chrono::steady_clock::now();
    prepareLinearWitnessKernels(pr);
    fresh_fc_timing_.kernel_preparation_seconds += std::chrono::duration<double>(
                                                       std::chrono::steady_clock::now() - kernel_start)
                                                       .count();
    fullyConnLayer(pr.C.circuit[layer_id], layer_id, lm_fc_id,
                   transformer_output_offset, 0, padded_ch_in, padded_vocab);
    lmHeadWideOutputLayer(pr.C.circuit[layer_id], layer_id);
}

void GPT2::create(prover &pr, bool merge)
{
    prepareCircuitData(pr);
    buildCircuit(pr, merge);
}

bool GPT2::validateCircuitProfile(const prover &pr, std::string *error) const
{
    auto fail = [&](const std::string &message)
    {
        if (error)
            *error = message;
        return false;
    };
    if (pr.C.circuit.empty() || pr.C.circuit[0].ty != layerType::INPUT)
    {
        return fail("circuit is missing its input layer");
    }
    const layer &input_layer = pr.C.circuit[0];
    if (!input_layer.uni_gates.empty() ||
        !input_layer.bin_gates.empty() ||
        !input_layer.bin_gates_v.empty())
    {
        return fail("input layer must not contain arithmetic gates");
    }
    const GPT2CircuitProfileSpec &profile = circuitProfileSpec();
    const int transformer_fc_count = static_cast<int>(full_conn.size());
    int fconn_count = 0;
    for (const layer &circuit : pr.C.circuit)
    {
        if (circuit.ty == layerType::FCONN)
            ++fconn_count;
    }

    if (fc_round_offsets.size() != full_conn.size())
    {
        return fail("one round-offset record is required per Transformer affine layer");
    }
    if (profile.force_immediate_linear_requant)
    {
        for (int offset : fc_round_offsets)
        {
            if (offset < 0)
            {
                return fail("artifact Transformer profile requires immediate requantization after every affine layer");
            }
        }
    }

    if (profile.include_skip_connections)
    {
        if (block_attn_residual_offsets.size() != static_cast<size_t>(layer_num) ||
            block_ffn_residual_offsets.size() != static_cast<size_t>(layer_num))
        {
            return fail("complete profile is missing a residual output for a Transformer block");
        }
    }
    else if (!block_attn_residual_offsets.empty() ||
             !block_ffn_residual_offsets.empty())
    {
        return fail("artifact Transformer profile must not contain skip-connection outputs");
    }

    const int expected_fconn = transformer_fc_count +
                               (profile.include_lm_head ? 1 : 0);
    if (fconn_count != expected_fconn)
    {
        return fail("unexpected FCONN count for the selected circuit profile");
    }
    if (profile.include_lm_head)
    {
        if (pr.lm_fc_id != transformer_fc_count ||
            embed_matrix_circuit_offset <= 0 || lm_head_output_offset <= 0)
        {
            return fail("complete profile is missing the LM-head circuit metadata");
        }
    }
    else if (pr.lm_fc_id != -1 || embed_matrix_circuit_offset != 0 ||
             lm_head_output_offset != 0)
    {
        return fail("artifact Transformer profile must not contain an LM head");
    }

    if (profile.name == nullptr || profile.name[0] == '\0')
    {
        return fail("circuit profile has no stable name");
    }
    if (profile.include_external_lookups)
    {
        if (range_query_specs.empty() || attention_lookup_groups.empty())
        {
            return fail("lookup-enabled profile is missing range or mapped lookup metadata");
        }
        try
        {
            for (const RangeQuerySpec &spec : range_query_specs)
            {
                spec.validateInputDomain(pr.C.circuit[0].size);
            }
        }
        catch (const std::exception &error)
        {
            return fail(std::string("invalid range-query layout: ") + error.what());
        }
    }
    return true;
}

bool GPT2::saveCircuitCache(const char *path, prover &pr) const
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        return false;

    if (!pr.C.saveToFile((string(path) + ".circuit").c_str()))
        return false;

    f.write(CircuitCacheType, sizeof(CircuitCacheType) - 1);
    f.write((char *)&len, sizeof(len));
    f.write((char *)&q_offset, sizeof(q_offset));
    f.write((char *)&total_in_size, sizeof(total_in_size));
    f.write((char *)&lm_head_output_offset, sizeof(lm_head_output_offset));
    f.write((char *)&embed_matrix_circuit_offset, sizeof(embed_matrix_circuit_offset));
    f.write((char *)&channel_in, sizeof(channel_in));
    f.write((char *)&channel_out, sizeof(channel_out));
    const int profile_id = static_cast<int>(circuit_profile_);
    f.write((const char *)&profile_id, sizeof(profile_id));
    auto writeVec = [&](const auto &v)
    {
        size_t n = v.size();
        f.write((char *)&n, sizeof(n));
        if (n)
            f.write((char *)v.data(), n * sizeof(v[0]));
    };
    writeVec(fc_round_requants);
    writeVec(fc_round_offsets);
    writeVec(fc_input_scales);
    writeVec(block_attn_residual_offsets);
    writeVec(block_ffn_residual_offsets);

    const std::uint64_t range_spec_count = range_query_specs.size();
    f.write((const char *)&range_spec_count, sizeof(range_spec_count));
    for (const RangeQuerySpec &spec : range_query_specs)
    {
        const std::uint64_t id_size = spec.id().size();
        const std::uint32_t bit_width = spec.bitWidth();
        const std::uint64_t span_count = spec.spans().size();
        f.write((const char *)&id_size, sizeof(id_size));
        if (id_size)
            f.write(spec.id().data(), id_size);
        f.write((const char *)&bit_width, sizeof(bit_width));
        f.write((const char *)&span_count, sizeof(span_count));
        for (const InputSpan &span : spec.spans())
        {
            const std::uint64_t offset = span.offset;
            const std::uint64_t count = span.count;
            const std::uint64_t stride = span.stride;
            f.write((const char *)&offset, sizeof(offset));
            f.write((const char *)&count, sizeof(count));
            f.write((const char *)&stride, sizeof(stride));
        }
    }

    f.write((char *)pr.fc_row, sizeof(pr.fc_row));
    f.write((char *)pr.fc_col, sizeof(pr.fc_col));
    f.write((char *)pr.fc_start_id, sizeof(pr.fc_start_id));
    f.write((char *)pr.fc_input_row, sizeof(pr.fc_input_row));
    f.write((char *)pr.fc_input_col, sizeof(pr.fc_input_col));
    f.write((char *)pr.fc_input_id, sizeof(pr.fc_input_id));
    f.write((char *)pr.fc_real_row, sizeof(pr.fc_real_row));
    f.write((char *)pr.fc_real_col, sizeof(pr.fc_real_col));
    f.write((char *)pr.fc_real_input_row, sizeof(pr.fc_real_input_row));
    f.write((char *)pr.fc_real_input_col, sizeof(pr.fc_real_input_col));
    f.write((char *)pr.fc_bias_id, sizeof(pr.fc_bias_id));

    f.write((char *)layer_norm_w_q_start, sizeof(layer_norm_w_q_start));
    f.write((char *)layer_norm_b_q_start, sizeof(layer_norm_b_q_start));

    return f.good();
}

bool GPT2::loadCircuitCache(const char *path, prover &pr)
{
    pr.prepared_model_values = false;
    prepared_gate_schedules_.clear();
    prepared_gate_owner_ = nullptr;
    prepared_attention_ready_ = false;
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;

    std::array<char, sizeof(CircuitCacheType) - 1> cache_type{};
    f.read(cache_type.data(), cache_type.size());
    if (!f || std::string(cache_type.data(), cache_type.size()) != CircuitCacheType)
    {
        cerr << "Circuit cache has an incompatible format; rebuild required" << endl;
        return false;
    }
    int cached_len = 0;
    f.read((char *)&cached_len, sizeof(cached_len));
    int expected_len = 1 << ceilPow2BitLength(len);
    if (cached_len != expected_len)
    {
        cerr << "Cache seq_len mismatch: cache=" << cached_len
             << ", expected=" << expected_len << ". Rebuild circuit instead." << endl;
        return false;
    }

    int cached_q_offset = 0;
    i64 cached_total_in_size = 0;
    int cached_lm_head_output_offset = 0;
    int cached_embed_matrix_circuit_offset = 0;
    int cached_channel_in = 0;
    int cached_channel_out = 0;
    f.read((char *)&cached_q_offset, sizeof(cached_q_offset));
    f.read((char *)&cached_total_in_size, sizeof(cached_total_in_size));
    f.read((char *)&cached_lm_head_output_offset, sizeof(cached_lm_head_output_offset));
    f.read((char *)&cached_embed_matrix_circuit_offset,
           sizeof(cached_embed_matrix_circuit_offset));
    f.read((char *)&cached_channel_in, sizeof(cached_channel_in));
    f.read((char *)&cached_channel_out, sizeof(cached_channel_out));
    int cached_profile_id = -1;
    f.read((char *)&cached_profile_id, sizeof(cached_profile_id));
    if (cached_profile_id != static_cast<int>(circuit_profile_))
    {
        cerr << "Cache circuit profile mismatch: cache=" << cached_profile_id
             << ", expected=" << static_cast<int>(circuit_profile_)
             << ". Rebuild circuit instead." << endl;
        return false;
    }
    if (cached_total_in_size < 0 || static_cast<std::size_t>(cached_total_in_size) != inputCommitmentLayout().value_count)
    {
        cerr << "Cache input row layout mismatch; rebuild required" << endl;
        return false;
    }
    if (!pr.C.loadFromFile((string(path) + ".circuit").c_str()))
        return false;
    prepare_gpt2_gate_intervals(pr.C);
    len = cached_len;
    q_offset = cached_q_offset;
    total_in_size = cached_total_in_size;
    lm_head_output_offset = cached_lm_head_output_offset;
    embed_matrix_circuit_offset = cached_embed_matrix_circuit_offset;
    channel_in = cached_channel_in;
    channel_out = cached_channel_out;

    auto readVec = [&](auto &v, std::size_t expected)
    {
        size_t n = 0;
        f.read((char *)&n, sizeof(n));
        if (!f || n != expected)
            return false;
        v.resize(n);
        if (n)
            f.read((char *)v.data(), n * sizeof(v[0]));
        return static_cast<bool>(f);
    };
    const std::size_t residual_count = circuitProfileSpec().include_skip_connections
                                           ? static_cast<std::size_t>(layer_num) : 0;
    if (!readVec(fc_round_requants, full_conn.size()) ||
        !readVec(fc_round_offsets, full_conn.size()) ||
        !readVec(fc_input_scales, full_conn.size()) ||
        !readVec(block_attn_residual_offsets, residual_count) ||
        !readVec(block_ffn_residual_offsets, residual_count))
        return false;

    std::uint64_t range_spec_count = 0;
    f.read((char *)&range_spec_count, sizeof(range_spec_count));
    if (range_spec_count > 100000)
    {
        cerr << "Cache range-query specification count is malformed" << endl;
        return false;
    }
    range_query_specs.clear();
    range_query_specs.reserve(static_cast<std::size_t>(range_spec_count));
    for (std::uint64_t i = 0; i < range_spec_count; ++i)
    {
        std::uint64_t id_size = 0;
        std::uint32_t bit_width = 0;
        std::uint64_t span_count = 0;
        f.read((char *)&id_size, sizeof(id_size));
        if (id_size == 0 || id_size > 4096)
        {
            cerr << "Cache range-query id is malformed" << endl;
            return false;
        }
        std::string id(static_cast<std::size_t>(id_size), '\0');
        f.read(&id[0], static_cast<std::streamsize>(id_size));
        f.read((char *)&bit_width, sizeof(bit_width));
        f.read((char *)&span_count, sizeof(span_count));
        if (span_count == 0 || span_count > 1000000)
        {
            cerr << "Cache range-query span metadata is malformed" << endl;
            return false;
        }
        std::vector<InputSpan> spans;
        spans.reserve(static_cast<std::size_t>(span_count));
        for (std::uint64_t span_index = 0; span_index < span_count; ++span_index)
        {
            std::uint64_t offset = 0, count = 0, stride = 0;
            f.read((char *)&offset, sizeof(offset));
            f.read((char *)&count, sizeof(count));
            f.read((char *)&stride, sizeof(stride));
            spans.push_back(InputSpan(
                static_cast<std::size_t>(offset),
                static_cast<std::size_t>(count),
                static_cast<std::size_t>(stride)));
        }
        try
        {
            RangeQuerySpec spec = RangeQuerySpec::powerOfTwo(
                std::move(id), std::move(spans), bit_width);
            range_query_specs.push_back(std::move(spec));
        }
        catch (const std::exception &error)
        {
            cerr << "Cache range-query specification is invalid: "
                 << error.what() << endl;
            return false;
        }
    }

    f.read((char *)pr.fc_row, sizeof(pr.fc_row));
    f.read((char *)pr.fc_col, sizeof(pr.fc_col));
    f.read((char *)pr.fc_start_id, sizeof(pr.fc_start_id));
    f.read((char *)pr.fc_input_row, sizeof(pr.fc_input_row));
    f.read((char *)pr.fc_input_col, sizeof(pr.fc_input_col));
    f.read((char *)pr.fc_input_id, sizeof(pr.fc_input_id));
    f.read((char *)pr.fc_real_row, sizeof(pr.fc_real_row));
    f.read((char *)pr.fc_real_col, sizeof(pr.fc_real_col));
    f.read((char *)pr.fc_real_input_row, sizeof(pr.fc_real_input_row));
    f.read((char *)pr.fc_real_input_col, sizeof(pr.fc_real_input_col));
    f.read((char *)pr.fc_bias_id, sizeof(pr.fc_bias_id));

    f.read((char *)layer_norm_w_q_start, sizeof(layer_norm_w_q_start));
    f.read((char *)layer_norm_b_q_start, sizeof(layer_norm_b_q_start));

    pr.mat_val = mat_values;

    return f.good() && f.peek() == std::char_traits<char>::eof();
}

void GPT2::recomputeAuxValues(prover &pr)
{
    if (full_conn.empty())
        return;
    const GPT2CircuitProfileSpec &profile = circuitProfileSpec();
    if (fc_round_offsets.size() < full_conn.size() ||
        fc_round_requants.size() < full_conn.size() ||
        (profile.include_skip_connections &&
         (block_attn_residual_offsets.size() < static_cast<size_t>(layer_num) ||
          block_ffn_residual_offsets.size() < static_cast<size_t>(layer_num))))
    {
        throw std::runtime_error("cached quantization metadata is incomplete");
    }

    std::vector<int> fconn_layers;
    fconn_layers.reserve(full_conn.size() +
                         (profile.include_lm_head ? 1 : 0));
    for (int layer_id = 1; layer_id < pr.C.size; ++layer_id)
    {
        if (pr.C.circuit[layer_id].ty == layerType::FCONN)
            fconn_layers.push_back(layer_id);
    }
    const int expected_fconn = static_cast<int>(full_conn.size()) +
                               (profile.include_lm_head ? 1 : 0);
    if (static_cast<int>(fconn_layers.size()) != expected_fconn)
    {
        throw std::runtime_error(
            "cached circuit FCONN count does not match model before witness recomputation");
    }

    auto move_fc_accumulator = [&](int fc_id, vector<F> &&accumulator)
    {
        if (fc_id < 0 || fc_id >= expected_fconn)
        {
            throw std::runtime_error("cached affine layer id is outside FCONN mapping");
        }
        const int layer_id = fconn_layers[fc_id];
        const std::size_t expected_size =
            static_cast<std::size_t>(len) *
            static_cast<std::size_t>(pr.fc_col[fc_id]);
        if (accumulator.size() != expected_size ||
            pr.C.circuit[layer_id].size != expected_size)
        {
            throw std::runtime_error(
                "cached FCONN shape does not match affine accumulator");
        }
        pr.val[layer_id] = std::move(accumulator);
    };

    auto compute_fc = [&](int fc_id, int input_offset)
    {
        if (fc_id < 0 || static_cast<size_t>(fc_id) >= prepared_fc_witness_.size())
            throw std::runtime_error("unprepared cached FC witness kernel");
        const int in_width = pr.fc_input_col[fc_id];
        const int out_width = pr.fc_col[fc_id];
        if (len <= 0 || in_width <= 0 || out_width <= 0 ||
            pr.fc_row[fc_id] != in_width || input_offset < 0 ||
            static_cast<size_t>(input_offset) > val[0].size())
        {
            throw std::runtime_error("invalid cached FC witness layout");
        }
        const auto &kernel = prepared_fc_witness_[fc_id];
        if (kernel.inputStride() != static_cast<size_t>(in_width) ||
            kernel.outputStride() != static_cast<size_t>(out_width))
            throw std::runtime_error("prepared FC witness shape changed");
        return kernel.evaluate(val[0].data() + input_offset,
                               val[0].size() - input_offset, len, threads);
    };

    auto compute_attention_softmax = [&](int qkv_offset, int aux_start,
                                         int input_width, int qkv_width,
                                         int block_idx)
    {
        const int heads = headnum;
        const int head_width = headdim;
        const int row_count = heads * len;
        const int score_count = heads * len * (len + 1) / 2;
        const int attention_count = heads * len * head_width;
        const int tri = len * (len + 1) / 2;
        const QuantAttentionPlan &plan = quant_plan.blocks.at(block_idx).attention;
        const AttentionAdviceLayout advice(aux_start, row_count, score_count, len * input_width);
        val[0][aux_start] = Fr(1);

        const auto &tables = prepared_attention_.at(block_idx);
        const auto &bounds = prepared_qkv_bounds_.at(block_idx);
        const int real_width = 3 * heads * head_width;
        std::vector<std::int64_t> qkv(static_cast<std::size_t>(len) * real_width);
        std::vector<unsigned char> invalid(std::max(1, threads), 0);
#pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            unsigned char failed = 0;
#pragma omp for schedule(static)
            for (std::size_t i = 0; i < qkv.size(); ++i)
            {
                const std::size_t row = i / real_width, col = i % real_width;
                if (!zkagent_integer::decode_bounded_i64(val[0][qkv_offset + row * qkv_width + col],
                                                         bounds[col / (heads * head_width)], qkv[i]))
                    failed = 1;
            }
            invalid[tid] = failed;
        }
        if (std::find(invalid.begin(), invalid.end(), 1) != invalid.end())
            throw std::runtime_error("attention QKV exceeds its prepared bound");
        std::vector<std::exception_ptr> errors(row_count);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
        for (int row_index = 0; row_index < row_count; ++row_index)
        {
            try
            {
                const int head = row_index / len, row = row_index % len;
                std::vector<ll> scores(static_cast<size_t>(row + 1));
                for (int col = 0; col <= row; ++col)
                {
                    ll score = 0;
                    for (int d = 0; d < head_width; ++d)
                    {
                        const int q_col = head * head_width + d;
                        const int k_col = heads * head_width + head * head_width + d;
                        score += static_cast<ll>(qkv[static_cast<std::size_t>(row) * real_width + q_col]) * qkv[static_cast<std::size_t>(col) * real_width + k_col];
                    }
                    scores[static_cast<size_t>(col)] = score;
                }
                AttentionSoftmaxRowWitness witness = tables.build(scores);
                if (!witness.success)
                {
                    throw std::runtime_error(
                        "cached attention Softmax witness failed: " + witness.message);
                }
                advice.writeRow(val[0], row_index, witness);

                for (int col = 0; col <= row; ++col)
                {
                    const int score_index = head * tri + row * (row + 1) / 2 + col;
                    advice.writeScore(val[0], score_index, col, witness);
                }

                for (int d = 0; d < head_width; ++d)
                {
                    ll weighted = 0;
                    for (int col = 0; col <= row; ++col)
                    {
                        const int score_index = head * tri + row * (row + 1) / 2 + col;
                        const int value_col = 2 * heads * head_width + head * head_width + d;
                        weighted += static_cast<ll>(qkv[static_cast<std::size_t>(col) * real_width + value_col]) * witness.probabilities[static_cast<std::size_t>(col)];
                    }
                    const int y_index = row * input_width + head * head_width + d;
                    const int compact = (head * len + row) * head_width + d;
                    advice.writeOutput(val[0], y_index, compact, weighted, plan);
                }
            }
            catch (...)
            {
                errors[row_index] = std::current_exception();
            }
        }
        for (const auto &error : errors)
            if (error)
                std::rethrow_exception(error);
        (void)attention_count;
    };

    int current_offset = 0;
    int block_residual = 0;
    int ffn_residual = 0;
    for (int fc_id = 0; fc_id < static_cast<int>(full_conn.size()); ++fc_id)
    {
        const int block = fc_id / 4;
        const int in_width = pr.fc_input_col[fc_id];
        const int out_width = pr.fc_col[fc_id];
        const int real_rows = pr.fc_real_input_row[fc_id];
        if (fc_id % 4 == 0)
            block_residual = current_offset;
        if (fc_id % 4 == 0 || fc_id % 4 == 2)
        {
            const int ln_id = block * 2 + ((fc_id % 4 == 2) ? 1 : 0);
            const int output = pr.fc_input_id[fc_id];
            computeLayerNormWitness(ln_id, current_offset, output - 10,
                               pr.fc_real_input_col[fc_id], in_width, real_rows);
            current_offset = output;
        }
        vector<F> accumulator = compute_fc(fc_id, current_offset);
        const QuantLinearPlan &linear = linearPlanForFc(fc_id);
        if (linear.mode == "immediate" ||
            profile.force_immediate_linear_requant)
        {
            computeRoundWitness(fc_id, accumulator, fc_round_offsets[fc_id], out_width);
            current_offset = fc_round_offsets[fc_id];
        }
        if (fc_id % 4 == 0)
        {
            const int output = pr.fc_input_id[fc_id + 1];
            const int score_count = headnum * len * (len + 1) / 2;
            const int row_count = headnum * len;
            const int aux = output -
                            (10 + 3 * row_count + 10 * score_count);
            compute_attention_softmax(
                current_offset, aux, in_width, out_width, block);
            current_offset = output;
        }
        else if (fc_id % 4 == 1 && profile.include_skip_connections)
        {
            const int output = block_attn_residual_offsets[block];
            computeResidualWitness(accumulator, block_residual, output, out_width,
                                     quant_plan.blocks[block].attn_residual);
            current_offset = output;
            ffn_residual = output;
        }
        else if (fc_id % 4 == 2)
        {
            const int output = pr.fc_input_id[fc_id + 1];
            computeGeluWitness(current_offset, output - 10, pr.fc_real_col[fc_id], out_width,
                         quant_plan.blocks[block]);
            current_offset = output;
        }
        else if (fc_id % 4 == 3 && profile.include_skip_connections)
        {
            const int output = block_ffn_residual_offsets[block];
            computeResidualWitness(accumulator, ffn_residual, output, out_width,
                                     quant_plan.blocks[block].ffn_residual);
            current_offset = output;
        }

        move_fc_accumulator(fc_id, std::move(accumulator));
    }

    if (!profile.include_final_layer_norm && !profile.include_lm_head)
    {
        return;
    }

    const int lm_fc_id = static_cast<int>(full_conn.size());
    const int final_ln_id = layer_num * 2;
    const int final_ln_output = pr.fc_input_id[lm_fc_id];
    const int last_fc = lm_fc_id - 1;
    computeLayerNormWitness(final_ln_id, current_offset, final_ln_output - 10,
                       pr.fc_real_col[last_fc], pr.fc_col[last_fc],
                       pr.fc_real_input_row[last_fc]);
    current_offset = final_ln_output;

    vector<F> lm_accumulator = compute_fc(lm_fc_id, current_offset);
    for (int row = 0; row < len; ++row)
    {
        for (int token = 0; token < vocab_size; ++token)
        {
            const F &accumulator =
                lm_accumulator[static_cast<i64>(row) * pr.fc_col[lm_fc_id] + token];
            const ll logit = fr_to_i128(accumulator);
            if (ll_abs(logit) > quant_plan.logit_bound)
            {
                throw std::runtime_error("cached LM-head output exceeds plan bound");
            }
            val[0][lm_head_output_offset + static_cast<i64>(row) * vocab_size + token] = logit;
        }
    }
    move_fc_accumulator(lm_fc_id, std::move(lm_accumulator));
}

void GPT2::recomputeWitness(prover &pr)
{
    cout << "Recomputing witness values from cached circuit..." << endl;
    if (pr.val.size() != static_cast<size_t>(pr.C.size))
    {
        throw std::runtime_error("prepareWeightsForCache must run before recomputeWitness");
    }
    val = pr.val.begin();
    if (!prepared_attention_ready_)
        throw std::runtime_error("cached attention metadata is not prepared");

    recomputeAuxValues(pr);

    int fc_id = 0;
    const int total_fc = static_cast<int>(full_conn.size()) +
                         (circuitProfileSpec().include_lm_head ? 1 : 0);
    for (int layer_id = 1; layer_id < pr.C.size; ++layer_id)
    {
        layer &circuit = pr.C.circuit[layer_id];
        if (circuit.ty != layerType::FCONN)
        {
            calcNormalLayer(circuit, layer_id,
                            &prepared_gate_schedules_.at(static_cast<size_t>(layer_id)));
            continue;
        }
        if (fc_id >= total_fc)
        {
            throw std::runtime_error("cached circuit contains too many FCONN layers");
        }
        const std::size_t expected_size =
            static_cast<std::size_t>(len) *
            static_cast<std::size_t>(pr.fc_col[fc_id]);
        if (circuit.size != expected_size || val[layer_id].size() != expected_size)
        {
            throw std::runtime_error(
                "cached FCONN witness was not materialized with the expected shape");
        }
        ++fc_id;
    }
    if (fc_id != total_fc)
    {
        throw std::runtime_error("cached circuit FCONN count does not match model");
    }
    cout << "Witness recomputation complete (" << fc_id << " FCONN layers)" << endl;
}

void GPT2::prepareWeightsForCache(prover &pr)
{
    if (!quant_plan.loaded())
    {
        throw std::runtime_error("cached witness requires an offline quantization plan");
    }
    full_conn.clear();
    const int qkv_dim = 3 * attn_dim;
    for (int block = 0; block < layer_num; ++block)
    {
        full_conn.emplace_back(qkv_dim, attn_dim);
        full_conn.emplace_back(attn_dim, attn_dim);
        full_conn.emplace_back(linear_dim, attn_dim);
        full_conn.emplace_back(attn_dim, linear_dim);
    }
    for (int i = 0; i < static_cast<int>(full_conn.size()); ++i)
    {
        full_conn[i].weight_start_id = pr.fc_start_id[i];
        full_conn[i].channel_in = pr.fc_row[i];
        full_conn[i].channel_out = pr.fc_col[i];
    }
    const GPT2CircuitProfileSpec &profile = circuitProfileSpec();
    const int lm_fc_id = static_cast<int>(full_conn.size());
    pr.lm_fc_id = profile.include_lm_head ? lm_fc_id : -1;

    const bool reuse_weights = pr.prepared_model_values && prepared_weights_owner_ == &pr;
    pr.prepared_model_values = false;
    if (!reuse_weights)
    {
        prepared_attention_ready_ = false;
        pr.val.clear();
        pr.val.resize(pr.C.size);
    }
    pr.val[0].resize(total_in_size);
    val = pr.val.begin();
    const auto layout = inputCommitmentLayout();
    const int prepare_threads = std::max(1, threads);
#pragma omp parallel for num_threads(prepare_threads) schedule(static)
    for (std::size_t row = 0; row < layout.rows(); ++row)
    {
        if (reuse_weights && !layout.isRuntimeRow(row))
            continue;
        const std::size_t begin = row * layout.columns();
        const std::size_t end = std::min(begin + layout.columns(), val[0].size());
        for (std::size_t i = begin; i < end; ++i)
            val[0][i].clear();
    }

    const int input_stride = pr.fc_input_col[0];
    if (!input_data)
        throw std::runtime_error("cached witness has no embedded input");
    for (int row = 0; row < len; ++row)
    {
        for (int col = 0; col < attn_dim; ++col)
        {
            val[0][static_cast<i64>(row) * input_stride + col] =
                input_data[static_cast<i64>(row) * attn_dim + col];
        }
    }

    if (!reuse_weights)
    {
        for (int i = 0; i < static_cast<int>(full_conn.size()); ++i)
        {
            prepareFCWeights(pr.fc_start_id[i], pr.fc_real_row[i], pr.fc_real_col[i], i,
                             pr.fc_row[i], pr.fc_col[i]);
            const int bias_offset = pr.fc_bias_id[i];
            if (bias_offset < 0)
                continue;
            for (int co = 0; co < pr.fc_col[i]; ++co)
            {
                ll q = 0;
                if (co < pr.fc_real_col[i])
                {
                    const int group = (i % 4 == 0) ? co / attn_dim : 0;
                    const int local_co = (i % 4 == 0) ? co % attn_dim : co;
                    q = linearPlanForFc(i, group).bias.at(local_co);
                }
                val[0][bias_offset + co] = q;
            }
        }

        const int ln_count = layer_num * 2 +
                             (profile.include_final_layer_norm ? 1 : 0);
        for (int ln_id = 0; ln_id < ln_count; ++ln_id)
        {
            const QuantLayerNormPlan &plan = layerNormPlan(ln_id);
            layer_norm_c1[ln_id] = plan.term1_c;
            layer_norm_e1[ln_id] = plan.term1_e;
            layer_norm_c2[ln_id] = plan.term2_c;
            layer_norm_e2[ln_id] = plan.term2_e;
            const int padded = (ln_id == layer_num * 2)
                                   ? pr.fc_col[static_cast<int>(full_conn.size()) - 1]
                                   : pr.fc_input_col[(ln_id / 2) * 4];
            for (int col = 0; col < padded; ++col)
            {
                val[0][layer_norm_w_q_start[ln_id] + col] =
                    (col < attn_dim) ? static_cast<ll>(plan.gamma[col]) : 0;
                val[0][layer_norm_b_q_start[ln_id] + col] =
                    (col < attn_dim) ? static_cast<ll>(plan.beta[col]) : 0;
            }
        }

        if (profile.include_lm_head)
        {
            const int padded_ch_in = channel_out;
            const int padded_vocab = 1 << ceilPow2BitLength(vocab_size);
            const i64 lm_size = static_cast<i64>(padded_vocab) * padded_ch_in;
            if (mat_values[lm_fc_id])
                delete[] mat_values[lm_fc_id];
            mat_values[lm_fc_id] = new int[lm_size];
            memset(mat_values[lm_fc_id], 0, sizeof(int) * lm_size);
#pragma omp parallel for num_threads(prepare_threads) schedule(static)
            for (int token = 0; token < vocab_size; ++token)
            {
                const i64 row = static_cast<i64>(token) * padded_ch_in;
                for (int col = 0; col < attn_dim; ++col)
                {
                    const int q = quant_plan.lm_head.weight[static_cast<i64>(token) * attn_dim + col];
                    mat_values[lm_fc_id][row + col] = q;
                    val[0][embed_matrix_circuit_offset + row + col] = q;
                }
            }
        }
    }
    initQuantPolicy(pr);
    pr.mat_val = mat_values;
    prepareLinearWitnessKernels(pr);
    prepareGateWitnessSchedules(pr);
    prepareCachedAttention(pr);
    pr.prepared_model_values = true;
    prepared_weights_owner_ = &pr;
}

void GPT2::prepareCachedAttention(const prover &pr)
{
    if (prepared_attention_ready_)
        return;
    prepared_attention_.clear();
    prepared_qkv_bounds_.clear();
    attention_lookup_groups.clear();
    const int row_count = headnum * len, score_count = headnum * len * (len + 1) / 2;
    const unsigned __int128 capacity = (static_cast<unsigned __int128>(1) << 124) - 1;
    for (int block = 0; block < layer_num; ++block)
    {
        const auto &source = quant_plan.blocks.at(block);
        const auto &plan = source.attention;
        if (len > quant_plan.positions || plan.theta <= 0 || static_cast<unsigned __int128>(plan.theta) > capacity)
            throw std::invalid_argument("prepared attention shape/theta exceeds integer capacity");
        std::array<std::int64_t, 3> bounds{};
        const QuantLinearPlan *linear[] = {&source.q, &source.k, &source.v};
        for (int i = 0; i < 3; ++i)
        {
            const ll bound = (circuitProfileSpec().force_immediate_linear_requant || linear[i]->mode == "immediate")
                                 ? linear[i]->output_bound
                                 : linear[i]->accumulator_bound;
            if (bound < 0 || bound > std::numeric_limits<std::int64_t>::max())
                throw std::invalid_argument("prepared attention QKV exceeds int64 input capacity");
            bounds[i] = static_cast<std::int64_t>(bound);
        }
        if (bounds[0] && static_cast<unsigned __int128>(bounds[1]) > capacity / bounds[0] / headdim)
            throw std::invalid_argument("prepared attention QK prefix exceeds int128 capacity");
        if (bounds[2] && static_cast<unsigned __int128>(plan.theta) > capacity / bounds[2] / quant_plan.positions)
            throw std::invalid_argument("prepared attention PV prefix exceeds int128 capacity");
        prepared_qkv_bounds_.push_back(bounds);
        prepared_attention_.emplace_back(plan);
        const auto &tables = prepared_attention_.back();
        const int output = pr.fc_input_id[block * 4 + 1];
        const int aux = output - (10 + 3 * row_count + 10 * score_count);
        const int digit0 = aux + 10 + row_count;
        const int middle = digit0 + 6 * score_count;
        struct Column
        {
            std::size_t group;
            int input, output;
            std::size_t occurrence = 0;
        };
        std::vector<Column> columns;
        for (int segment = 0; segment < 6; ++segment)
        {
            const int bits = plan.digit_bits[segment];
            columns.push_back({ensureAttentionLookupGroup("attention/digit/" + std::to_string(bits), bits, false, {}),
                               digit0 + segment * score_count, -1, 0});
        }
        columns.push_back({ensureAttentionLookupGroup("attention/middle/block/" + std::to_string(block),
                                                      plan.digit_bits[3], true, tables.middleTable()),
                           digit0 + 3 * score_count, middle, 0});
        for (int high = 0; high < 2; ++high)
        {
            const int bits = plan.digit_bits[4 + high];
            columns.push_back({ensureAttentionLookupGroup("attention/zero/" + std::to_string(bits), bits, true, tables.zeroTable(high)),
                               digit0 + (4 + high) * score_count, middle + (1 + high) * score_count, 0});
        }
        std::vector<std::size_t> counts(attention_lookup_groups.size(), 0), bases(counts.size());
        for (auto &column : columns)
            column.occurrence = counts[column.group]++;
        for (std::size_t i = 0; i < counts.size(); ++i)
            if (counts[i])
            {
                auto &group = attention_lookup_groups[i];
                bases[i] = group.input_offsets.size();
                group.input_offsets.resize(bases[i] + counts[i] * score_count);
                if (group.mapped)
                    group.output_offsets.resize(group.input_offsets.size());
            }
#pragma omp parallel for num_threads(threads) schedule(static)
        for (int score = 0; score < score_count; ++score)
            for (const auto &column : columns)
            {
                auto &group = attention_lookup_groups[column.group];
                const auto at = bases[column.group] + score * counts[column.group] + column.occurrence;
                group.input_offsets[at] = column.input + score;
                if (group.mapped)
                    group.output_offsets[at] = column.output + score;
            }
    }
    prepared_attention_ready_ = true;
}

void GPT2::prepareLinearWitnessKernels(const prover &pr, bool include_lm_head)
{
    const int count = static_cast<int>(full_conn.size()) +
                      (include_lm_head && circuitProfileSpec().include_lm_head ? 1 : 0);
    const std::size_t ready = prepared_fc_witness_.size();
    if (ready >= static_cast<std::size_t>(count))
        return;
    std::vector<zkagent_integer::PreparedLinearWitness> prepared;
    prepared.reserve(static_cast<std::size_t>(count) - ready);
    for (int fc_id = static_cast<int>(ready); fc_id < count; ++fc_id)
    {
        const bool head = fc_id == static_cast<int>(full_conn.size());
        const int groups = !head && fc_id % 4 == 0 ? 3 : 1;
        ll input_bound = 0, accumulator_bound = 0;
        for (int group = 0; group < groups; ++group)
        {
            const auto &plan = head ? quant_plan.lm_head : linearPlanForFc(fc_id, group);
            input_bound = group == 0 ? plan.input_bound : std::min(input_bound, plan.input_bound);
            accumulator_bound = std::max(accumulator_bound, plan.accumulator_bound);
        }
        const int bias_offset = pr.fc_bias_id[fc_id];
        if (pr.fc_input_col[fc_id] <= 0 || pr.fc_col[fc_id] <= 0 ||
            pr.fc_row[fc_id] != pr.fc_input_col[fc_id] ||
            pr.fc_real_input_col[fc_id] <= 0 || pr.fc_real_col[fc_id] <= 0 ||
            (bias_offset >= 0 && static_cast<size_t>(bias_offset) > pr.val[0].size()))
            throw std::runtime_error("invalid FC model preparation layout");
        zkagent_integer::LinearWitnessModelView model;
        model.weights = mat_values[fc_id];
        model.weight_size = static_cast<size_t>(pr.fc_input_col[fc_id]) * pr.fc_col[fc_id];
        model.bias = bias_offset < 0 ? nullptr : pr.val[0].data() + bias_offset;
        model.bias_size = bias_offset < 0 ? 0 : pr.val[0].size() - bias_offset;
        model.input_stride = pr.fc_input_col[fc_id];
        model.output_stride = pr.fc_col[fc_id];
        model.real_inputs = pr.fc_real_input_col[fc_id];
        model.real_outputs = pr.fc_real_col[fc_id];
        model.input_bound = input_bound;
        model.accumulator_bound = accumulator_bound;
        prepared.push_back(zkagent_integer::PreparedLinearWitness::prepare(model, threads));
    }
    prepared_fc_witness_.reserve(count);
    for (auto &kernel : prepared)
        prepared_fc_witness_.push_back(std::move(kernel));
}

InputCommitmentPreparation GPT2::prepareInputCommitment(
    prover &pr, const std::string &key_file, const std::string &srs_cache, bool replace)
{
    const auto start = std::chrono::steady_clock::now();
    InputCommitmentPreparation result;
    const auto layout = inputCommitmentLayout();
    if (!replace && std::ifstream(key_file, std::ios::binary).good())
    {
        result.key = InputCommitmentKey::loadFromFile(key_file, layout);
        result.loaded = true;
    }
    else
    {
        if (pr.val.empty() || pr.val[0].empty())
            prepareWeightsForCache(pr);
        const int column_bits = layout.variables - layout.variables / 2;
        if (mv_srs.pk.srs_g1.size() < layout.columns())
            mv_srs.setup_cached(column_bits, srs_cache);
        const auto commit_start = std::chrono::steady_clock::now();
        result.key = ::prepareInputCommitmentKey(layout,
                                                 pr.val[0].data(), pr.val[0].size(), mv_srs.pk, mv_srs.vk, threads);
        result.fixed_commit_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - commit_start).count();
        result.key.saveToFile(key_file);
    }
    result.preparation_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

void GPT2::commitCircuit(prover &pr, const InputCommitmentKey &key)
{
    cout << "Committing circuit input..." << endl;

    const int commit_threads = threads > 0 ? threads : 1;
    int logn = pr.C.circuit[0].bit_length;
    u64 n_sqrt = 1ULL << (logn - (logn >> 1));

    if (!gens || gens_size <= 0)
    {
        throw std::runtime_error("Pedersen generators not initialized; call ensureGenerators() first");
    }
    if ((int)n_sqrt > gens_size)
    {
        cerr << "Error: Pedersen generators too small (need " << n_sqrt
             << ", have " << gens_size << ")" << endl;
        throw std::runtime_error("Pedersen generators too small for circuit commitment");
    }

    pr.gens.resize(n_sqrt);
    for (u64 i = 0; i < n_sqrt; i++)
    {
        pr.gens[i] = gens[i];
    }
    pr.gens.push_back(gens_base);

    timer elapsed;
    elapsed.start();
    if (!(key.layout() == inputCommitmentLayout()) || !key.matchesSrs(mv_srs.vk))
        throw std::invalid_argument("fixed input commitment key does not match this model/circuit");
    pr.commitInput(pr.gens, commit_threads, &key);
    elapsed.stop();
    pr.proof_size += 1ULL << (pr.cc.l / 2);
    cout << "Dynamic circuit input commit time: " << elapsed.elapse_sec() << "s" << endl;
}

ll *GPT2::getRangeProofInputs(int &out_count)
{

    std::size_t total = 0;
    for (const RangeQuerySpec &spec : range_query_specs)
    {
        total += spec.valueCount();
    }

    if (total == 0)
    {
        out_count = 0;
        return nullptr;
    }

    if (total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("range-proof query count exceeds the legacy API");
    }
    ll *deltas = new ll[total];
    std::size_t idx = 0;

    for (const RangeQuerySpec &spec : range_query_specs)
    {
        for (const InputSpan &span : spec.spans())
        {
            for (std::size_t i = 0; i < span.count; ++i)
            {
                deltas[idx++] = fr_to_i128(
                    val[0][span.offset + i * span.stride]);
            }
        }
    }

    out_count = static_cast<int>(total);
    cout << "Collected " << total << " delta values for range proof" << endl;
    return deltas;
}

LookupWitness GPT2::buildAttentionLookupWitness(const prover &pr) const
{
    if (attention_lookup_groups.empty())
    {
        throw std::runtime_error("attention lookup layout has no query groups");
    }
    if (pr.C.circuit.empty() || pr.val.empty())
    {
        throw std::runtime_error("attention lookup layout requires a built circuit witness");
    }
    const std::size_t input_domain_size =
        std::size_t(1) << pr.C.circuit[0].bit_length;
    LookupLayoutBuilder builder("attention-layout", input_domain_size);
    std::vector<std::vector<ll>> group_queries;
    std::vector<std::vector<ll>> group_query_outputs;

    for (const AttentionLookupGroup &source : attention_lookup_groups)
    {
        const int K = 1 << source.input_bits;
        if (source.input_offsets.empty() ||
            (source.mapped &&
             source.output_offsets.size() != source.input_offsets.size()))
        {
            throw std::runtime_error("malformed attention lookup group");
        }
        std::vector<ll> table(static_cast<std::size_t>(K));
        std::vector<ll> table_outputs;
        if (source.mapped)
            table_outputs.resize(static_cast<std::size_t>(K));
        for (int i = 0; i < K; ++i)
        {
            table[static_cast<std::size_t>(i)] = static_cast<ll>(i);
            if (source.mapped)
            {
                table_outputs[static_cast<std::size_t>(i)] =
                    source.table_outputs.at(static_cast<std::size_t>(i));
            }
        }
        const std::string table_id = source.mapped
                                         ? "random-tuple/" + source.key
                                         : "identity/" + std::to_string(source.input_bits);
        const std::size_t group_index = builder.addGroup(source.mapped
                                                             ? LookupGroupSpec::randomTuple(
                                                                   source.key, table_id, table, table_outputs,
                                                                   source.input_offsets.size(), table[0])
                                                             : LookupGroupSpec(
                                                                   source.key, table_id, table,
                                                                   source.input_offsets.size(), table[0]));

        LookupAffineBinding binding;
        binding.id = source.key;
        binding.group_index = group_index;
        binding.query_base = 0;
        binding.input_coefficient = 1;
        binding.output_coefficient = 0;
        binding.constant = 0;
        binding.random_tuple_compression = source.mapped;
        binding.input_offsets.reserve(source.input_offsets.size());
        if (source.mapped)
        {
            binding.output_offsets.reserve(source.output_offsets.size());
        }
        std::vector<ll> queries;
        std::vector<ll> query_outputs;
        queries.reserve(1ULL << ceilPow2BitLength(
                            static_cast<u32>(source.input_offsets.size())));
        if (source.mapped)
            query_outputs.reserve(queries.capacity());
        for (std::size_t query = 0; query < source.input_offsets.size(); ++query)
        {
            const int input_offset = source.input_offsets[query];
            const int output_offset = source.mapped
                                          ? source.output_offsets[query]
                                          : -1;
            if (input_offset < 0 ||
                static_cast<std::size_t>(input_offset) >= pr.val[0].size() ||
                (source.mapped && (output_offset < 0 ||
                                   static_cast<std::size_t>(output_offset) >= pr.val[0].size())))
            {
                throw std::out_of_range("attention lookup offset is outside the GKR input layer");
            }
            const ll input = fr_to_i128(pr.val[0][input_offset]);
            const ll output = source.mapped
                                  ? fr_to_i128(pr.val[0][output_offset])
                                  : 0;
            if (input < 0 || input >= K)
            {
                throw std::out_of_range("attention lookup input is outside its public table");
            }
            if (source.mapped &&
                table_outputs[static_cast<std::size_t>(input)] != output)
            {
                throw std::runtime_error("attention mapped lookup witness is inconsistent");
            }
            queries.push_back(input);
            if (source.mapped)
                query_outputs.push_back(output);
            binding.input_offsets.push_back(
                static_cast<std::size_t>(input_offset));
            if (source.mapped)
            {
                binding.output_offsets.push_back(
                    static_cast<std::size_t>(output_offset));
            }
        }
        builder.addAffineBinding(std::move(binding));
        const std::size_t padded_count =
            std::size_t(1) << ceilPow2BitLength(static_cast<u32>(queries.size()));
        queries.resize(padded_count, table[0]);
        group_queries.push_back(std::move(queries));
        if (source.mapped)
        {
            query_outputs.resize(padded_count, table_outputs[0]);
            group_query_outputs.push_back(std::move(query_outputs));
        }
        else
        {
            group_query_outputs.emplace_back();
        }
    }

    return {std::move(builder).finalize(), std::move(group_queries),
            std::move(group_query_outputs)};
}

LookupWitness GPT2::buildTransformerLookupWitness(
    const prover &pr, unsigned full_limb_bits) const
{
    if (pr.C.circuit.empty())
    {
        throw std::runtime_error("Transformer lookup layout requires a built circuit");
    }
    const std::size_t input_domain_size =
        std::size_t(1) << pr.C.circuit[0].bit_length;
    LookupWitness ranges = buildRangeLookupWitness(
        range_query_specs, pr.val[0], input_domain_size, full_limb_bits, threads);
    LookupWitness attention = buildAttentionLookupWitness(pr);
    std::vector<LookupWitness> components;
    components.reserve(2);
    components.push_back(std::move(ranges));
    components.push_back(std::move(attention));
    return mergeLookupWitnesses("transformer-lookup-layout", std::move(components));
}

void GPT2::loadTokensFromFile(const char *path)
{
    ifstream file(path);
    if (!file.is_open())
    {
        cerr << "Failed to open token file: " << path << endl;
        return;
    }

    input_tokens.clear();
    int token;
    while (file >> token && (int)input_tokens.size() < len)
    {
        if (token >= 0 && token < vocab_size)
        {
            input_tokens.push_back(token);
        }
        else
        {
            cerr << "Warning: token " << token << " out of range, skipping" << endl;
        }
    }

    cout << "Loaded " << input_tokens.size() << " tokens from " << path << endl;
}

void GPT2::setTokens(const vector<int> &tokens)
{
    input_tokens.clear();

    for (size_t i = 0; i < tokens.size() && (int)input_tokens.size() < len; i++)
    {
        if (tokens[i] >= 0 && tokens[i] < vocab_size)
        {
            input_tokens.push_back(tokens[i]);
        }
        else
        {
            cerr << "Warning: token " << tokens[i] << " out of range [0, " << vocab_size << "), skipping" << endl;
        }
    }

    std::mt19937 rng(456);
    std::uniform_int_distribution<int> dist(0, vocab_size - 1);
    while ((int)input_tokens.size() < len)
    {
        input_tokens.push_back(dist(rng));
    }
}

void GPT2::setTokensNoPad(const vector<int> &tokens)
{
    input_tokens.clear();

    for (size_t i = 0; i < tokens.size() && (int)input_tokens.size() < len; i++)
    {
        if (tokens[i] >= 0 && tokens[i] < vocab_size)
        {
            input_tokens.push_back(tokens[i]);
        }
        else
        {
            cerr << "Warning: token " << tokens[i] << " out of range [0, " << vocab_size << "), skipping" << endl;
        }
    }
}

void GPT2::initRandomEmbedding()
{
    cout << "Initializing random embedding matrix (" << vocab_size << " x " << attn_dim << ")..." << endl;
    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (long long i = 0; i < (long long)vocab_size * attn_dim; i++)
    {
        embedding_matrix[i] = dist(rng);
    }
    cout << "Embedding matrix initialized as raw float" << endl;
}

void GPT2::loadEmbedding(const char *path)
{
    cout << "Loading embedding from: " << path << endl;

    FILE *f = fopen(path, "rb");
    if (!f)
    {
        cerr << "Failed to open embedding file: " << path << ", using random embedding" << endl;
        initRandomEmbedding();
        return;
    }

    size_t expected = (size_t)vocab_size * attn_dim;
    size_t read_count = fread(embedding_matrix, sizeof(float), expected, f);
    fclose(f);

    if (read_count != expected)
    {
        cerr << "Read " << read_count << " elements, expected " << expected << endl;
        initRandomEmbedding();
        return;
    }

    cout << "Loaded embedding matrix (" << vocab_size << " x " << attn_dim
         << ") as raw float (will be quantized in prepareCircuitData)" << endl;
}

void GPT2::loadPositionEmbedding(const char *path)
{
    cout << "Loading position embedding from: " << path << endl;

    FILE *f = fopen(path, "rb");
    if (!f)
    {
        cerr << "Failed to open position embedding file: " << path << endl;
        return;
    }

    if (!position_embedding)
    {
        position_embedding = new float[1024 * attn_dim];
    }

    size_t max_elements = 1024 * attn_dim;
    size_t read_count = fread(position_embedding, sizeof(float), max_elements, f);
    fclose(f);

    cout << "Loaded position embedding (" << read_count / attn_dim << " x " << attn_dim
         << ") as raw float" << endl;
}

void GPT2::loadTransformerWeights(const char *model_dir)
{
    prepared_fc_witness_.clear();
    prepared_weights_owner_ = nullptr;
    prepared_attention_.clear();
    prepared_qkv_bounds_.clear();
    prepared_attention_ready_ = false;
    cout << "Loading offline quantization plan from: " << model_dir << "/quantized" << endl;

    std::string error;
    if (!quant_plan.load(model_dir, &error))
    {
        throw std::runtime_error(
            error + ". Run quant/prepare_gpt2_quant.py --model-dir " +
            std::string(model_dir));
    }
    if (!quant_plan.validateDimensions(layer_num, headnum, headdim, attn_dim,
                                       linear_dim, vocab_size, &error))
    {
        throw std::runtime_error(error);
    }
    quant_scale = quant_plan.embedding_scale;

    cout << "Quantization plan loaded successfully (Q=" << quant_plan.q_bits
         << ", max registered range bits=" << quant_plan.maxRangeBits() << ")" << endl;
}

void GPT2::initGenerators()
{
    if (gens)
        return;

    gens_size = max(attn_dim, 1 << 12);
    gens = new G1[gens_size];

    if ((int)mv_srs.pk.srs_g1.size() < gens_size)
    {
        cerr << "Error: MVKZG SRS too small (need " << gens_size
             << ", have " << mv_srs.pk.srs_g1.size() << ")" << endl;
        throw std::runtime_error("MVKZG SRS too small for Pedersen generators");
    }

    gens_base = hyrax::load_generators(gens, mv_srs.pk.srs_g1, gens_size);
    cout << "Initialized Pedersen generators from KZG SRS (size=" << gens_size << ")" << endl;
}

void GPT2::ensureGenerators(const prover &pr)
{
    if (pr.C.circuit.empty())
    {
        throw std::runtime_error("Circuit not initialized; cannot size generators");
    }

    int logn = pr.C.circuit[0].bit_length;
    u64 n_sqrt = 1ULL << (logn - (logn >> 1));

    int col_domain = 1;
    while (col_domain < attn_dim)
    {
        col_domain <<= 1;
    }

    int required = (int)n_sqrt;
    if (required < col_domain)
    {
        required = col_domain;
    }
    if (required < (1 << 12))
    {
        required = 1 << 12;
    }

    if (gens && gens_size >= required)
    {
        return;
    }
    if ((int)mv_srs.pk.srs_g1.size() < required)
    {
        cerr << "Error: MVKZG SRS too small (need " << required
             << ", have " << mv_srs.pk.srs_g1.size() << ")" << endl;
        throw std::runtime_error("MVKZG SRS too small for generator sizing");
    }

    if (gens)
    {
        delete[] gens;
    }
    gens_size = required;
    gens = new G1[gens_size];
    gens_base = hyrax::load_generators(gens, mv_srs.pk.srs_g1, gens_size);
    cout << "Initialized Pedersen generators from KZG SRS (size=" << gens_size << ")" << endl;
}

void GPT2::setupZkEmbed()
{
    if (!zkembed)
    {
        cout << "Setting up zkEmbed prover..." << endl;

        if (!quantized_embedding_matrix)
        {
            long long total_size = (long long)vocab_size * attn_dim;
            quantized_embedding_matrix = new ll[total_size];
            if (quant_plan.loaded())
            {
                if (quant_plan.token_embedding.size() != static_cast<size_t>(total_size))
                {
                    throw std::runtime_error("quantized token-embedding shape mismatch");
                }
                for (long long i = 0; i < total_size; i++)
                {
                    quantized_embedding_matrix[i] = quant_plan.token_embedding[i];
                }
            }
            else
            {
                for (long long i = 0; i < total_size; i++)
                {
                    quantized_embedding_matrix[i] = (ll)round(embedding_matrix[i] / quant_scale);
                }
            }
        }

        zkembed.reset(new ZkEmbed(
            uni_srs.pk, uni_srs.vk, mv_srs.pk, mv_srs.vk,
            vocab_size, attn_dim, threads));
        zkembed->preprocess(quantized_embedding_matrix);

        cout << "zkEmbed prover ready (paper protocol)" << endl;
    }
}

void GPT2::computeEmbedding()
{
    cout << "Computing embedding lookup..." << endl;

    if (owned_embedded_input)
    {
        delete[] owned_embedded_input;
        owned_embedded_input = nullptr;
    }

    int n = input_tokens.size();
    int padded_len = 1 << ceilPow2BitLength(len);
    if (padded_len != len)
    {
        cout << "Padding sequence length for embedding: " << len << " -> " << padded_len << endl;
    }
    owned_embedded_input = new ll[padded_len * attn_dim];

    if (quant_plan.loaded() && padded_len > quant_plan.positions)
    {
        throw std::runtime_error("padded sequence length exceeds quantized positional table");
    }
    cout << "Embedding quantization: offline scale=" << std::setprecision(17)
         << quant_scale << endl;
    for (int i = 0; i < n; i++)
    {
        int tid = input_tokens[i];
        for (int j = 0; j < attn_dim; j++)
        {
            if (quant_plan.loaded())
            {
                const ll token = quant_plan.token_embedding[(long long)tid * attn_dim + j];
                const ll position = quant_plan.position_embedding[(long long)i * attn_dim + j];
                const ll combined = token + position;
                if (ll_abs(combined) > quant_plan.embedding_bound)
                {
                    throw std::runtime_error("embedding output exceeds offline bound");
                }
                owned_embedded_input[i * attn_dim + j] = combined;
            }
            else
            {
                float tok_emb = embedding_matrix[(long long)tid * attn_dim + j];
                float pos_emb = position_embedding ? position_embedding[i * attn_dim + j] : 0.0f;
                float raw_val = tok_emb + pos_emb;
                owned_embedded_input[i * attn_dim + j] = (ll)round(raw_val / quant_scale);
            }
        }
    }

    for (int i = n; i < padded_len; i++)
    {
        for (int j = 0; j < attn_dim; j++)
        {
            owned_embedded_input[i * attn_dim + j] = quant_plan.loaded()
                                                         ? static_cast<ll>(quant_plan.position_embedding[(long long)i * attn_dim + j])
                                                         : (position_embedding
                                                                ? static_cast<ll>(round(position_embedding[(long long)i * attn_dim + j] / quant_scale))
                                                                : 0);
        }
    }

    setInput(owned_embedded_input);
    cout << "Embedded " << n << " tokens (quantized)" << endl;

    saveLayerOutput("embedding", owned_embedded_input, n, attn_dim);
}

std::vector<Fr> GPT2::prepareZkEmbedWitness() const
{
    if (!zkembed || !quantized_embedding_matrix || input_tokens.empty())
    {
        throw std::logic_error("prepare zkEmbed table/keys and tokens before its witness");
    }
    std::vector<ll> outputs(input_tokens.size() * static_cast<std::size_t>(attn_dim));
    for (std::size_t row = 0; row < input_tokens.size(); ++row)
    {
        for (int col = 0; col < attn_dim; ++col)
        {
            outputs[row * attn_dim + col] =
                quantized_embedding_matrix[static_cast<std::size_t>(input_tokens[row]) * attn_dim + col];
        }
    }
    return zkembed->prepareWitness(input_tokens.size(), outputs);
}

void GPT2::setSaveIntermediates(const char *output_dir)
{
    save_intermediates = true;
    intermediates_dir = output_dir;

    string cmd = "mkdir -p " + intermediates_dir;
    system(cmd.c_str());

    cout << "Intermediate outputs will be saved to: " << intermediates_dir << endl;
}

void GPT2::saveLayerOutput(const char *name, const ll *data, int rows, int cols)
{
    if (!save_intermediates)
        return;

    string filename = intermediates_dir + "/" + name + ".bin";
    FILE *f = fopen(filename.c_str(), "wb");
    if (!f)
    {
        cerr << "Warning: Could not save " << filename << endl;
        return;
    }

    size_t count = (size_t)rows * cols;
    int64_t *temp = new int64_t[count];
    for (size_t i = 0; i < count; i++)
    {
        temp[i] = (int64_t)data[i];
    }
    size_t bytes_written = fwrite(temp, sizeof(int64_t), count, f);
    delete[] temp;
    fclose(f);

    cout << "  Saved: " << name << " [" << rows << " x " << cols << "]"
         << " (" << bytes_written << " int64 values)" << endl;
}

void GPT2::dumpBlockOutputs(const char *output_dir)
{
    if (!save_intermediates || !output_dir)
    {
        return;
    }
    if (block_ffn_residual_offsets.size() < (size_t)layer_num)
    {
        cerr << "Warning: block_ffn_residual_offsets size " << block_ffn_residual_offsets.size()
             << " < layer_num " << layer_num << endl;
    }

    int rows = len;
    int real_cols = attn_dim;
    int padded_cols = full_conn.empty() ? attn_dim : (int)full_conn[1].channel_out;
    {
        string meta_path = intermediates_dir + "/meta.json";
        std::ofstream meta(meta_path);
        if (meta.is_open())
        {
            meta << "{\n";
            meta << "  \"rows\": " << rows << ",\n";
            meta << "  \"cols\": " << real_cols << ",\n";
            meta << "  \"padded_cols\": " << padded_cols << ",\n";
            meta << "  \"blocks\": " << layer_num << ",\n";
            meta << "  \"scale_source\": \"" << quant_plan.artifactDir()
                 << "/quant_plan.txt\"\n";
            meta << "}\n";
        }
    }

    vector<ll> buf((size_t)rows * real_cols);
    for (int block = 0; block < layer_num; block++)
    {
        if (block >= (int)block_ffn_residual_offsets.size())
        {
            break;
        }
        int offset = block_ffn_residual_offsets[block];
        for (int i = 0; i < rows; i++)
        {
            i64 base = (i64)offset + (i64)i * padded_cols;
            for (int j = 0; j < real_cols; j++)
            {
                buf[(size_t)i * real_cols + j] = fr_to_i128(val[0][base + j]);
            }
        }
        string name = "block_" + std::to_string(block) + "_out";
        saveLayerOutput(name.c_str(), buf.data(), rows, real_cols);
    }
}

void GPT2::dumpBlockStages(prover &pr, const char *output_dir)
{
    if (!save_intermediates || !output_dir)
    {
        return;
    }
    if (full_conn.size() < (size_t)layer_num * 4)
    {
        cerr << "Warning: full_conn size " << full_conn.size()
             << " < expected " << (layer_num * 4) << endl;
    }

    int rows = len;
    int n = attn_dim;
    int ffn = linear_dim;

    auto write_meta = [&]()
    {
        string meta_path = intermediates_dir + "/meta.json";
        std::ofstream meta(meta_path);
        if (!meta.is_open())
        {
            return;
        }
        meta << "{\n";
        meta << "  \"rows\": " << rows << ",\n";
        meta << "  \"attn_dim\": " << n << ",\n";
        meta << "  \"linear_dim\": " << ffn << ",\n";
        meta << "  \"blocks\": " << layer_num << ",\n";
        meta << "  \"quant_plan\": \"" << quant_plan.artifactDir()
             << "/quant_plan.txt\"\n";
        meta << "}\n";
    };

    write_meta();

    auto dump_stage = [&](int block, const string &stage, int offset, int real_cols, int padded_cols)
    {
        vector<ll> buf((size_t)rows * real_cols);
        for (int i = 0; i < rows; i++)
        {
            i64 base = (i64)offset + (i64)i * padded_cols;
            for (int j = 0; j < real_cols; j++)
            {
                buf[(size_t)i * real_cols + j] = fr_to_i128(val[0][base + j]);
            }
        }
        string name = "block_" + std::to_string(block) + "_" + stage;
        saveLayerOutput(name.c_str(), buf.data(), rows, real_cols);
    };

    for (int block = 0; block < layer_num; block++)
    {
        int fc0 = block * 4;
        int fc1 = fc0 + 1;
        int fc2 = fc0 + 2;
        int fc3 = fc0 + 3;

        if (fc3 >= (int)fc_round_offsets.size())
        {
            break;
        }
        int ln1_off = pr.fc_input_id[fc0];
        int qkv_off = fc_round_offsets[fc0];
        int softmax_off = pr.fc_input_id[fc1];
        int attn_proj_off = fc_round_offsets[fc1];
        int attn_resid_off = (block < (int)block_attn_residual_offsets.size()) ? block_attn_residual_offsets[block] : -1;
        int ln2_off = pr.fc_input_id[fc2];
        int ffn_up_off = fc_round_offsets[fc2];
        int gelu_off = pr.fc_input_id[fc3];
        int ffn_down_off = fc_round_offsets[fc3];
        int ffn_resid_off = (block < (int)block_ffn_residual_offsets.size()) ? block_ffn_residual_offsets[block] : -1;

        dump_stage(block, "ln1", ln1_off, n, pr.fc_input_col[fc0]);
        dump_stage(block, "qkv", qkv_off, 3 * n, pr.fc_col[fc0]);
        dump_stage(block, "softmax", softmax_off, n, pr.fc_input_col[fc1]);
        if (attn_proj_off >= 0)
        {
            dump_stage(block, "attn_proj", attn_proj_off, n, pr.fc_col[fc1]);
        }
        if (attn_resid_off >= 0)
        {
            dump_stage(block, "attn_resid", attn_resid_off, n, pr.fc_col[fc1]);
        }
        dump_stage(block, "ln2", ln2_off, n, pr.fc_input_col[fc2]);
        dump_stage(block, "ffn_up", ffn_up_off, ffn, pr.fc_col[fc2]);
        dump_stage(block, "gelu", gelu_off, ffn, pr.fc_input_col[fc3]);
        if (ffn_down_off >= 0)
        {
            dump_stage(block, "ffn_down", ffn_down_off, n, pr.fc_col[fc3]);
        }
        if (ffn_resid_off >= 0)
        {
            dump_stage(block, "ffn_resid", ffn_resid_off, n, pr.fc_col[fc3]);
        }
    }
}

void GPT2::lmHeadWideOutputLayer(layer &circuit, i64 &layer_id)
{
    int padded_vocab = 1 << ceilPow2BitLength(vocab_size);
    i64 block_len = (i64)len * vocab_size;

    if (build_gates)
    {
        initLayer(circuit, block_len, layerType::RELU);
        circuit.need_phase2 = false;
        circuit.zero_start_id = 0;
    }

    int orgsize = val[0].size();
    val[0].resize(orgsize + 20 + block_len);
    lm_head_output_offset = orgsize + 20;
    total_in_size += 20 + block_len;

    for (i64 g = 0; g < block_len; ++g)
    {
        int row = g / vocab_size;
        int col = g % vocab_size;
        i64 padded_idx = (i64)row * padded_vocab + col;
        ll p = fr_to_i128(val[layer_id - 1][padded_idx]);
        if (ll_abs(p) > quant_plan.logit_bound)
        {
            throw std::runtime_error("wide LM-head logit exceeds offline bound");
        }
        val[0][lm_head_output_offset + g] = p;
    }

    if (g_debug)
    {
        cerr << "lmHeadWideOutputLayer: block_len=" << block_len
             << ", bound=" << i128_to_string(quant_plan.logit_bound) << endl;
        cerr << "  val[0].size() after resize: " << val[0].size() << endl;
    }

    if (build_gates)
    {
        for (i64 g = 0; g < block_len; ++g)
        {
            int row = g / vocab_size;
            int col = g % vocab_size;
            i64 p = (i64)row * padded_vocab + col;
            int q = lm_head_output_offset + g;
            circuit.uni_gates.emplace_back(g, p, layer_id - 1, 1);
            circuit.uni_gates.emplace_back(g, q, 0, -1);
        }
    }

    calcNormalLayer(circuit, layer_id);
    for (i64 g = 0; g < block_len; ++g)
        assert(val[layer_id][g].isZero());
    layer_id++;
}
