#ifndef ZKAGENT_ZKDECODE_OUTPUT_HPP
#define ZKAGENT_ZKDECODE_OUTPUT_HPP

#include "pcs/hyrax.hpp"

#include <array>
#include <vector>

struct MVKZG_ProverKey;
struct MVKZG_VerifierKey;
class FiatShamirTranscript;

using mcl::bn::Fr;
using mcl::bn::G1;

struct DecodeSelectorRoundPoly {
    Fr coeffs[3] = {Fr(0), Fr(0), Fr(0)};

    Fr evaluate(const Fr &x) const;
};

struct DecodeSelectorProof {
    int token_count = 0;
    int vocab_size = 0;
    int token_domain = 0;
    int vocab_domain = 0;
    int num_vars = 0;

    std::vector<G1> probability_commitments;
    std::vector<Fr> total_mass_claims;
    std::vector<Fr> prefix_claims;
    std::vector<Fr> next_prefix_claims;
    std::vector<DecodeSelectorRoundPoly> sumcheck_rounds;
    Fr probability_evaluation = Fr(0);
    hyrax::OpeningProof probability_opening;
};

struct DecodeSelectorResult {
    bool success = false;
    double prover_time = 0.0;
    DecodeSelectorProof proof;
};

struct DecodeSamplingValues {
    ll total_mass = 0;
    ll prefix = 0;
    ll next_prefix = 0;
};

bool check_decode_sampling_claims(
    const DecodeSelectorProof& proof,
    const std::vector<G1>& expected_probability_commitments,
    int expected_vocab_size,
    const std::vector<ll>& sample_values,
    ll theta, ll epsilon,
    std::vector<DecodeSamplingValues>* values = nullptr);

bool verify_decode_sampling_proof(
    const DecodeSelectorProof& proof,
    const std::vector<int>& output_tokens,
    const std::vector<G1>& expected_probability_commitments,
    int expected_vocab_size,
    const std::vector<ll>& sample_values,
    ll theta, ll epsilon,
    const MVKZG_VerifierKey& vk);

bool verify_decode_selector_proof(
    const DecodeSelectorProof& proof,
    const std::vector<int>& output_tokens,
    const MVKZG_VerifierKey& vk);

bool verify_decode_selector_proof(
    const DecodeSelectorProof& proof,
    const std::vector<int>& output_tokens,
    const MVKZG_VerifierKey& vk,
    FiatShamirTranscript& transcript);

DecodeSelectorResult prove_decode_selector(
    const std::vector<std::vector<ll>>& probability_mass, const std::vector<int>& output_tokens,
    const MVKZG_ProverKey& pk, const MVKZG_VerifierKey& vk, int threads,
    FiatShamirTranscript& transcript, const std::vector<G1>* shared_probability_commitments = nullptr);

struct DecodeBatchPointRound
{
    std::array<Fr, 3> evaluations = {Fr(0), Fr(0), Fr(0)};

    Fr evaluate(const Fr &point) const;
};

struct DecodeBatchPointZeroProof
{
    int token_count = 0;
    int vocab_size = 0;
    int token_domain = 0;
    int vocab_domain = 0;
    int num_vars = 0;

    std::vector<G1> value_commitments;
    std::vector<DecodeBatchPointRound> sumcheck_rounds;
    Fr final_evaluation = Fr(0);
    hyrax::OpeningProof opening;
};

struct DecodeBatchPointZeroResult
{
    bool success = false;
    double prover_time = 0.0;
    DecodeBatchPointZeroProof proof;
};

bool verify_decode_batch_point_zero_proof(
    const DecodeBatchPointZeroProof &proof,
    const std::vector<int> &output_tokens,
    const MVKZG_VerifierKey &vk);

bool verify_decode_batch_point_zero_proof(
    const DecodeBatchPointZeroProof &proof,
    const std::vector<int> &output_tokens,
    const MVKZG_VerifierKey &vk,
    FiatShamirTranscript &transcript);

DecodeBatchPointZeroResult prove_decode_batch_point_zero(
    const std::vector<std::vector<ll>> &values, const std::vector<int> &output_tokens,
    const MVKZG_ProverKey &pk, const MVKZG_VerifierKey &vk, int threads,
    FiatShamirTranscript &transcript, const std::vector<G1> *shared_value_commitments = nullptr);

inline Fr DecodeSelectorRoundPoly::evaluate(const Fr &x) const
{
    return (coeffs[2] * x + coeffs[1]) * x + coeffs[0];
}

inline Fr DecodeBatchPointRound::evaluate(const Fr &point) const
{
    const Fr &y0 = evaluations[0];
    const Fr &y1 = evaluations[1];
    const Fr &y2 = evaluations[2];
    // Initialized on first use, after the field has been initialized.
    static const Fr inverse_two = Fr(1) / Fr(2);
    const Fr c2 = (y2 - Fr(2) * y1 + y0) * inverse_two;
    return (c2 * point + y1 - y0 - c2) * point + y0;
}

#endif
