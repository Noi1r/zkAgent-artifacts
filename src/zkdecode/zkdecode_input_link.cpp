#include "zkdecode/zkdecode_input_link.hpp"
#include "zkdecode/zkdecode_exp_digits.hpp"
#include "zkdecode/zkdecode_verifier.hpp"
#include "common/bit_utils.hpp"
#include "common/field_integer.hpp"
#include <cmath>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace zkdecode_input_detail
{
    using zkagent_integer::fr_from_i128;

    void append_context(const ZkDecodeStatement &statement, const ZkDecodeVerifierKey &key,
                        const ZkDecodeInputContext &context, FiatShamirTranscript &transcript)
    {
        const auto &spec = context.logits.spec();
        spec.validate();
        const auto &p = statement.parameters;
        if (!(p == key.parameters()) || !statement.logit_commitments.empty() ||
            spec.rows != std::size_t(p.token_count) || spec.columns != std::size_t(p.vocab_size) ||
            statement.output_tokens.size() != spec.rows ||
            context.input_commitments.size() != (std::size_t(1) << (spec.input_bits / 2)) ||
            !std::isfinite(context.logit_scale) || context.logit_scale <= 0)
            throw std::invalid_argument("incompatible decode input/view statement");
        if (p.kind == ZkDecodeProofKind::TopK && context.logit_scale != DecodeExpDigitsConfig{}.logit_scale)
            throw std::invalid_argument("TopK exp configuration does not match the input logit scale");
        transcript.append_u64(spec.input_bits);
        transcript.append_u64(spec.offset);
        transcript.append_u64(spec.rows);
        transcript.append_u64(spec.columns);
        transcript.append_u64(spec.row_stride);
        std::uint64_t scale_bits;
        static_assert(sizeof(scale_bits) == sizeof(context.logit_scale), "IEEE double required");
        std::memcpy(&scale_bits, &context.logit_scale, sizeof(scale_bits));
        transcript.append_u64(scale_bits);
        transcript.append_u64(static_cast<std::uint64_t>(p.kind));
        transcript.append_u64(static_cast<std::uint64_t>(p.lookup_backend));
        transcript.append_u64(p.k);
        transcript.append_fr(fr_from_i128(p.bound));
        transcript.append_fr(fr_from_i128(p.theta));
        transcript.append_fr(fr_from_i128(p.epsilon_sm));
        for (int output : statement.output_tokens)
            transcript.append_u64(output);
        transcript.append_u64(statement.sample_values.size());
        for (ll value : statement.sample_values)
            transcript.append_fr(fr_from_i128(value));
        transcript.append_g1_vector(context.input_commitments);
    }
    void bind_logit_claim(InputReductionLink &deferred_input_claim, const ZkDecodeInputContext &context,
                     const DecodeLogitEvaluation &evaluation, FiatShamirTranscript &transcript)
    {
        deferred_input_claim = InputReductionLink{};
        deferred_input_claim.input_domain_size = context.logits.spec().inputDomain();
        const Fr weight = transcript.nonzero_challenge();
        deferred_input_claim.addTensorEvaluation(context.logits, evaluation.point, evaluation.evaluation, weight);
    }
}

std::vector<std::vector<ll>> prepareZkDecodeInputWitness(
    const std::vector<Fr> &input, const InputTensorViewSpec &spec, ll bound)
{
    spec.validate();
    const auto end = spec.offset + (spec.rows - 1) * spec.row_stride + spec.columns;
    if (input.size() > spec.inputDomain() || input.size() < end || bound <= 0 || bound >= (ll(1) << 125))
        throw std::invalid_argument("invalid decode input witness dimensions/bound");
    std::vector<std::vector<ll>> rows(spec.rows, std::vector<ll>(spec.columns));
    for (std::size_t t = 0; t < spec.rows; ++t)
        for (std::size_t j = 0; j < spec.columns; ++j)
        {
            const Fr &field = input[spec.offset + t * spec.row_stride + j];
            ll value;
            if (zkagent_integer::decode_nonnegative_i128(field, bound - 1, value))
                rows[t][j] = value;
            else if (zkagent_integer::decode_nonnegative_i128(-field, bound, value))
                rows[t][j] = -value;
            else
                throw std::out_of_range("GKR logit is outside the declared decode bound");
        }
    return rows;
}

std::size_t zkDecodeRequiredGenerators(const ZkDecodeParameters &p)
{
    if (p.token_count <= 0 || p.vocab_size <= 0)
        throw std::invalid_argument("empty decode instance");
    const int n = zkagent_integer::ceil_log2(p.token_count) + zkagent_integer::ceil_log2(p.vocab_size);
    if (n > 30)
        throw std::invalid_argument("decode domain exceeds native PCS capacity");
    const int table_bits = p.kind == ZkDecodeProofKind::TopK ? 11 : 10;
    const int columns_bits = p.lookup_backend == LookupBackend::Shout
                                 ? (n + table_bits + 1) / 2
                                 : (std::max(n, table_bits) + 1) / 2;
    return std::size_t(1) << columns_bits;
}
