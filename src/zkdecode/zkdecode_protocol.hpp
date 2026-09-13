#ifndef ZKAGENT_ZKDECODE_PROTOCOL_HPP
#define ZKAGENT_ZKDECODE_PROTOCOL_HPP

// Shared protocol relations and transcript framing for the two roles.
#include "zkdecode/zkdecode_arithmetic.hpp"
#include "zkdecode/zkdecode_output.hpp"
#include "common/bit_utils.hpp"
#include "common/mle.hpp"
#include "common/fiat_shamir.hpp"
#include "common/field_integer.hpp"
#include "zkdecode/zkdecode_exp_digits.hpp"

#include <cstdint>
#include <stdexcept>

namespace zkdecode_detail {

// The two valid execution modes keep transcript ownership and the deferred
// logit opening together. Standalone components use separate local streams.
class ExecutionContext {
public:
    static ExecutionContext standalone() { return {}; }
    static ExecutionContext linked(FiatShamirTranscript &transcript,
                                   DecodeLogitEvaluation &deferred_logit_claim) {
        return ExecutionContext(transcript, deferred_logit_claim);
    }
    bool isInputLinked() const { return mode_ == Mode::InputLinked; }
    FiatShamirTranscript &transcript(FiatShamirTranscript &local) const {
        return isInputLinked() ? *shared_transcript_ : local;
    }
    FiatShamirTranscript &sharedTranscript() const { return *shared_transcript_; }
    DecodeLogitEvaluation *deferredLogitClaim() const { return deferred_logit_claim_; }

private:
    enum class Mode { Standalone, InputLinked };
    ExecutionContext() = default;
    ExecutionContext(FiatShamirTranscript &transcript, DecodeLogitEvaluation &claim)
        : mode_(Mode::InputLinked), shared_transcript_(&transcript), deferred_logit_claim_(&claim) {}
    Mode mode_ = Mode::Standalone;
    FiatShamirTranscript *shared_transcript_ = nullptr;
    DecodeLogitEvaluation *deferred_logit_claim_ = nullptr;
};

namespace arithmetic {

    constexpr int LocalRelationCount = 7;
    using zkagent_integer::next_power_of_two;
    using zkagent_integer::log2_power_of_two;
    using zkagent_integer::fr_from_i128;

    inline int column_index(DecodeTopKLinkColumn column)
    {
        return static_cast<int>(column);
    }

    inline void append_statement(FiatShamirTranscript &transcript,
                          const DecodeTopKLinkProof &proof,
                          int token_count, int vocab_size, int k)
    {
        if (proof.defer_logit_opening)
            transcript.append_u64(1);
        transcript.append_u64(token_count);
        transcript.append_u64(vocab_size);
        transcript.append_u64(k);
        transcript.append_u64(proof.token_domain);
        transcript.append_u64(proof.vocab_domain);
        transcript.append_u64(proof.num_vars);
        for (int column = 0; column < DecodeTopKLinkColumns; ++column)
        {
            transcript.append_u64(column);
            transcript.append_g1_vector(proof.column_commitments[column]);
        }
        transcript.append_g1_vector(proof.shift_commitments);
        transcript.append_g1_vector(proof.separator_commitments);
    }

    inline void append_tuple_table_context(FiatShamirTranscript &transcript)
    {
        const DecodeExpDigitsConfig config;
        const auto &tables = decode_exp_digits_public_tables(config);
        transcript.append_fr(fr_from_i128(config.expected_theta));
        const int middle = config.nonlinear_segment;
        for (int segment = middle; segment <= middle + config.most_zero_segments; ++segment)
        {
            const int base = config.bases[static_cast<std::size_t>(segment)];
            transcript.append_u64(static_cast<std::uint64_t>(base));
            const auto &outputs = tables.outputs[static_cast<std::size_t>(segment)];
            std::vector<Fr> field_outputs(outputs.size());
            for (std::size_t input = 0; input < outputs.size(); ++input)
                field_outputs[input] = fr_from_i128(outputs[input]);
            transcript.append_fr_vector(field_outputs);
        }
    }

    struct LinkChallenges
    {
        std::array<Fr, LocalRelationCount> relation;
        Fr cardinality;
        std::vector<Fr> zero_point;
        std::vector<Fr> token_point;
    };

    inline LinkChallenges derive_link_challenges(FiatShamirTranscript &transcript,
                                          int num_vars,
                                          int token_vars)
    {
        LinkChallenges result;
        for (int i = 0; i < LocalRelationCount; ++i)
        {
            result.relation[i] = transcript.nonzero_challenge();
        }
        result.cardinality = transcript.nonzero_challenge();
        result.zero_point.resize(num_vars);
        for (int i = 0; i < num_vars; ++i)
        {
            result.zero_point[i] = transcript.challenge();
        }
        result.token_point.resize(token_vars);
        for (int i = 0; i < token_vars; ++i)
        {
            result.token_point[i] = transcript.challenge();
        }
        return result;
    }

    inline Fr local_relation(
        const std::array<Fr, DecodeTopKLinkColumns> &value,
        const std::array<Fr, LocalRelationCount> &coefficient,
        const Fr &active_mask)
    {
        const Fr &logit = value[column_index(DecodeTopKLinkColumn::Logit)];
        const Fr &separator = value[column_index(DecodeTopKLinkColumn::Separator)];
        const Fr &shift = value[column_index(DecodeTopKLinkColumn::Shift)];
        const Fr &delta = value[column_index(DecodeTopKLinkColumn::Delta)];
        const Fr &selector = value[column_index(DecodeTopKLinkColumn::Selector)];
        const Fr &candidate_in = value[column_index(DecodeTopKLinkColumn::CandidateIn)];
        const Fr &candidate_out = value[column_index(DecodeTopKLinkColumn::CandidateOut)];
        const Fr &middle = value[column_index(DecodeTopKLinkColumn::MiddleOutput)];
        const Fr &high0 = value[column_index(DecodeTopKLinkColumn::HighZero0)];
        const Fr &high1 = value[column_index(DecodeTopKLinkColumn::HighZero1)];
        const Fr &probability = value[column_index(DecodeTopKLinkColumn::Probability)];

        Fr magnitude = Fr(0);
        Fr radix = Fr(1);
        for (int digit = 0; digit < 6; ++digit)
        {
            magnitude += radix * value[column_index(DecodeTopKLinkColumn::Digit0) + digit];
            radix *= Fr(DecodeExpDigitBases[digit]);
        }

        const Fr delta_relation = delta - logit + separator;
        const Fr boolean_relation = selector * (selector - Fr(1));
        const Fr candidate_in_relation = candidate_in - selector * delta;
        const Fr candidate_out_relation =
            candidate_out - (Fr(1) - selector) * (-delta);
        const Fr magnitude_relation =
            magnitude - selector * (shift - delta);
        const Fr probability_relation =
            probability - selector * middle * high0 * high1;

        return coefficient[0] * delta_relation + coefficient[1] * boolean_relation + coefficient[2] * candidate_in_relation + coefficient[3] * candidate_out_relation + coefficient[4] * magnitude_relation + coefficient[5] * probability_relation + coefficient[6] * selector * (Fr(1) - active_mask);
    }

    inline Fr token_prefix_mle(int count, const std::vector<Fr> &point)
    {
        if (count < 0)
            throw std::invalid_argument("public prefix exceeds its domain");
        return mle::prefix_sum(static_cast<std::size_t>(count), point);
    }

    inline int greedy_column_index(DecodeGreedyLinkColumn column)
    {
        return static_cast<int>(column);
    }

    inline void append_greedy_statement(
        FiatShamirTranscript &transcript,
        const DecodeGreedyLinkProof &proof,
        int token_count,
        int vocab_size)
    {
        if (proof.defer_logit_opening)
            transcript.append_u64(1);
        transcript.append_u64(token_count);
        transcript.append_u64(vocab_size);
        transcript.append_u64(proof.token_domain);
        transcript.append_u64(proof.vocab_domain);
        transcript.append_u64(proof.num_vars);
        for (int column = 0; column < DecodeGreedyLinkColumns; ++column)
        {
            transcript.append_u64(column);
            for (const G1 &commitment : proof.column_commitments[column])
            {
                transcript.append_g1(commitment);
            }
        }
        for (const G1 &commitment : proof.lambda_commitments)
            transcript.append_g1(commitment);
    }

} // namespace arithmetic
} // namespace zkdecode_detail

namespace zkdecode_detail
{
namespace output
{


    inline void append_statement(FiatShamirTranscript &transcript,
                          const DecodeSelectorProof &proof,
                          const std::vector<int> &output_tokens)
    {
        transcript.append_u64(proof.token_count);
        transcript.append_u64(proof.vocab_size);
        transcript.append_u64(proof.token_domain);
        transcript.append_u64(proof.vocab_domain);
        for (int token : output_tokens)
            transcript.append_u64(token);
        transcript.append_u64(proof.probability_commitments.size());
        for (const G1 &commitment : proof.probability_commitments)
        {
            transcript.append_g1(commitment);
        }
        for (int t = 0; t < proof.token_count; ++t)
        {
            transcript.append_fr(proof.total_mass_claims[t]);
            transcript.append_fr(proof.prefix_claims[t]);
            transcript.append_fr(proof.next_prefix_claims[t]);
        }
    }

    inline void append_statement(
        FiatShamirTranscript &transcript,
        const DecodeBatchPointZeroProof &proof,
        const std::vector<int> &output_tokens)
    {
        transcript.append_u64(proof.token_count);
        transcript.append_u64(proof.vocab_size);
        transcript.append_u64(proof.token_domain);
        transcript.append_u64(proof.vocab_domain);
        transcript.append_u64(proof.num_vars);
        for (int token : output_tokens)
        {
            transcript.append_u64(token);
        }
        transcript.append_g1_vector(proof.value_commitments);
    }

    struct SelectorChallenges
    {
        Fr eta_prefix;
        Fr eta_next;
        std::vector<Fr> token_batch;
    };

    inline SelectorChallenges derive_batching_challenges(
        FiatShamirTranscript &transcript, int token_count)
    {
        SelectorChallenges result;
        result.eta_prefix = transcript.nonzero_challenge();
        result.eta_next = transcript.nonzero_challenge();
        result.token_batch.resize(token_count);
        for (int t = 0; t < token_count; ++t)
        {
            result.token_batch[t] = transcript.nonzero_challenge();
        }
        return result;
    }

    inline std::vector<Fr> derive_token_coefficients(
        FiatShamirTranscript &transcript, int token_count)
    {
        std::vector<Fr> coefficients(static_cast<std::size_t>(token_count));
        for (int token = 0; token < token_count; ++token)
        {
            coefficients[static_cast<std::size_t>(token)] =
                transcript.challenge_excluding(std::vector<Fr>{Fr(0)});
        }
        return coefficients;
    }

    inline std::size_t boolean_point_index(
        int token, int output_token, int vocab_domain)
    {
        return static_cast<std::size_t>(token) * static_cast<std::size_t>(vocab_domain) + static_cast<std::size_t>(output_token);
    }

} // namespace output
} // namespace zkdecode_detail

#endif
