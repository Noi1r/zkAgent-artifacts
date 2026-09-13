#ifndef ZKAGENT_ZKDECODE_ARITHMETIC_HPP
#define ZKAGENT_ZKDECODE_ARITHMETIC_HPP

#include "pcs/hyrax.hpp"

#include <array>
#include <vector>
#include <stdexcept>

struct MVKZG_ProverKey;
struct MVKZG_VerifierKey;
class FiatShamirTranscript;

struct DecodeLogitEvaluation {
    std::vector<mcl::bn::Fr> point;
    mcl::bn::Fr evaluation = mcl::bn::Fr(0);
};

using mcl::bn::Fr;
using mcl::bn::G1;

constexpr int DecodeTopKLinkColumns = 17;
constexpr int DecodeTopKLinkDegree = 5;

enum class DecodeTopKLinkColumn : int {
    Logit = 0,
    Separator,
    Shift,
    Delta,
    Selector,
    CandidateIn,
    CandidateOut,
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    MiddleOutput,
    HighZero0,
    HighZero1,
    Probability,
};

struct DecodeTopKLinkWitnessView {
    int token_count = 0;
    int vocab_size = 0;
    int k = 0;
    std::vector<const ll*> logits;
    std::vector<const ll*> deltas;
    std::vector<const int*> selectors;
    std::vector<const ll*> candidate_in;
    std::vector<const ll*> candidate_out;
    std::array<std::vector<const ll*>, 6> digits;
    std::vector<const ll*> middle_outputs;
    std::vector<const ll*> high_zero_0;
    std::vector<const ll*> high_zero_1;
    std::vector<const ll*> probabilities;
    std::vector<ll> separators;
    std::vector<ll> shifts;
};

struct DecodeTopKLinkRound {
    std::array<Fr, DecodeTopKLinkDegree + 1> evaluations;

    Fr evaluate(const Fr& point) const;
};

struct DecodeTopKLinkProof {
    bool defer_logit_opening = false;
    int token_count = 0;
    int vocab_size = 0;
    int k = 0;
    int token_domain = 0;
    int vocab_domain = 0;
    int num_vars = 0;
    std::array<std::vector<G1>, DecodeTopKLinkColumns> column_commitments;
    std::vector<G1> shift_commitments;
    std::vector<G1> separator_commitments;
    Fr tuple_compression_challenge = Fr(0);
    std::vector<DecodeTopKLinkRound> sumcheck_rounds;
    std::array<Fr, DecodeTopKLinkColumns> final_evaluations;
    Fr shift_evaluation = Fr(0);
    Fr separator_evaluation = Fr(0);
    hyrax::OpeningProof combined_opening;
    hyrax::OpeningProof shift_opening;
};

struct DecodeTopKLinkResult {
    bool success = false;
    double prover_time = 0.0;
    DecodeTopKLinkProof proof;
};

inline Fr decode_topk_tuple_challenge(
    const DecodeTopKLinkProof &proof,
    int token_count, int vocab_size, int k)
{
    if (token_count <= 0 || vocab_size <= 0 || k <= 0 ||
        proof.token_count != token_count ||
        proof.vocab_size != vocab_size || proof.k != k)
    {
        throw std::invalid_argument(
            "decode tuple compression received inconsistent public dimensions");
    }
    if (proof.tuple_compression_challenge.isZero())
    {
        throw std::invalid_argument(
            "decode tuple compression challenge was not transcript-derived");
    }
    return proof.tuple_compression_challenge;
}

bool verify_decode_topk_link_proof(
    const DecodeTopKLinkProof& proof,
    int token_count,
    int vocab_size,
    int k,
    const MVKZG_VerifierKey& vk);

bool verify_decode_topk_link_proof(
    const DecodeTopKLinkProof& proof,
    int token_count,
    int vocab_size,
    int k,
    const MVKZG_VerifierKey& vk,
    FiatShamirTranscript& transcript);

inline const std::vector<G1> &decode_topk_probability_commitment(
    const DecodeTopKLinkProof &proof)
{
    return proof.column_commitments[static_cast<int>(DecodeTopKLinkColumn::Probability)];
}

inline const std::vector<G1> &decode_topk_column_commitment(
    const DecodeTopKLinkProof &proof,
    DecodeTopKLinkColumn column)
{
    return proof.column_commitments[static_cast<int>(column)];
}

enum class DecodeGreedyLinkColumn : int {
    Logit = 0,
    Lambda,
    NegativeDelta,
};

constexpr int DecodeGreedyLinkColumns = 3;

struct DecodeGreedyLinkWitnessView {
    int token_count = 0;
    int vocab_size = 0;
    std::vector<const ll*> logits;
    std::vector<const ll*> negative_deltas;
    std::vector<ll> lambdas;
};

struct DecodeGreedyLinkProof {
    bool defer_logit_opening = false;
    int token_count = 0;
    int vocab_size = 0;
    int token_domain = 0;
    int vocab_domain = 0;
    int num_vars = 0;
    std::array<std::vector<G1>, DecodeGreedyLinkColumns>
        column_commitments;
    std::array<Fr, DecodeGreedyLinkColumns> final_evaluations;
    std::vector<G1> lambda_commitments;
    Fr lambda_evaluation = Fr(0);
    hyrax::OpeningProof lambda_opening;
    hyrax::OpeningProof combined_opening;
};

struct DecodeGreedyLinkResult {
    bool success = false;
    double prover_time = 0.0;
    DecodeGreedyLinkProof proof;
};

bool verify_decode_greedy_link_proof(
    const DecodeGreedyLinkProof& proof,
    int token_count,
    int vocab_size,
    const MVKZG_VerifierKey& vk);

bool verify_decode_greedy_link_proof(
    const DecodeGreedyLinkProof& proof,
    int token_count,
    int vocab_size,
    const MVKZG_VerifierKey& vk,
    FiatShamirTranscript& transcript);

inline const std::vector<G1> &decode_greedy_negative_delta_commitment(
    const DecodeGreedyLinkProof &proof)
{
    return proof.column_commitments[static_cast<int>(DecodeGreedyLinkColumn::NegativeDelta)];
}

DecodeGreedyLinkResult prove_decode_greedy_link(
    const DecodeGreedyLinkWitnessView&, const MVKZG_ProverKey&,
    const MVKZG_VerifierKey&, int threads, FiatShamirTranscript&,
    DecodeLogitEvaluation* deferred_logit_claim = nullptr);
DecodeTopKLinkResult prove_decode_topk_link_view(
    const DecodeTopKLinkWitnessView&, const MVKZG_ProverKey&,
    const MVKZG_VerifierKey&, int threads, FiatShamirTranscript&,
    DecodeLogitEvaluation* deferred_logit_claim = nullptr);
bool reduce_decode_greedy_link_proof(
    const DecodeGreedyLinkProof&, int tokens, int vocab,
    const MVKZG_VerifierKey&, FiatShamirTranscript&, DecodeLogitEvaluation& deferred_logit_claim);
bool reduce_decode_topk_link_proof(
    const DecodeTopKLinkProof&, int tokens, int vocab, int k,
    const MVKZG_VerifierKey&, FiatShamirTranscript&, DecodeLogitEvaluation& deferred_logit_claim);

inline Fr DecodeTopKLinkRound::evaluate(const Fr &point) const
{
    // Initialize on first evaluation, after the caller has initialized the field.
    static const std::array<Fr, DecodeTopKLinkDegree + 1> inverse_denominators = []
    {
        std::array<Fr, DecodeTopKLinkDegree + 1> inverses;
        for (int j = 0; j <= DecodeTopKLinkDegree; ++j)
        {
            Fr denominator = Fr(1);
            for (int m = 0; m <= DecodeTopKLinkDegree; ++m)
                if (m != j)
                    denominator *= Fr(j - m);
            Fr::inv(inverses[static_cast<std::size_t>(j)], denominator);
        }
        return inverses;
    }();
    Fr result = Fr(0);
    for (int j = 0; j <= DecodeTopKLinkDegree; ++j)
    {
        Fr numerator = Fr(1);
        for (int m = 0; m <= DecodeTopKLinkDegree; ++m)
        {
            if (m == j)
                continue;
            numerator *= point - Fr(m);
        }
        result += evaluations[static_cast<std::size_t>(j)] * numerator * inverse_denominators[static_cast<std::size_t>(j)];
    }
    return result;
}

#endif
