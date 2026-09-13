#include "zkdecode/zkdecode_output.hpp"
#include "zkdecode/zkdecode_protocol.hpp"
#include "common/mle.hpp"
#include "common/bit_utils.hpp"
#include "common/field_integer.hpp"
#include "pcs/srs.hpp"

#include <exception>
#include <utility>

using namespace zkdecode_detail::output;

namespace
{
    using zkagent_integer::next_power_of_two;
    using zkagent_integer::log2_power_of_two;

    Fr batched_claim(const DecodeSelectorProof &proof, const SelectorChallenges &ch)
    {
        Fr result = Fr(0);
        for (int t = 0; t < proof.token_count; ++t)
        {
            Fr inner, tmp;
            Fr::mul(inner, ch.eta_prefix, proof.prefix_claims[t]);
            Fr::mul(tmp, ch.eta_next, proof.next_prefix_claims[t]);
            Fr::add(inner, inner, tmp);
            Fr::add(inner, inner, proof.total_mass_claims[t]);
            Fr::mul(inner, inner, ch.token_batch[t]);
            Fr::add(result, result, inner);
        }
        return result;
    }

    Fr weight_mle_at_point(const DecodeSelectorProof &proof,
                           const std::vector<int> &output_tokens,
                           const SelectorChallenges &ch,
                           const std::vector<Fr> &point)
    {
        const int vocab_vars = log2_power_of_two(proof.vocab_domain);
        std::vector<Fr> index_point(point.begin(), point.begin() + vocab_vars);
        std::vector<Fr> token_point(point.begin() + vocab_vars, point.end());
        Fr result = Fr(0);
        for (int t = 0; t < proof.token_count; ++t)
        {
            const Fr token_eq = mle::equality_at(t, token_point);
            const auto prefix_at_token = mle::prefix_and_equality(output_tokens[t], index_point);
            const Fr &prefix = prefix_at_token.prefix;
            const Fr next_prefix = prefix + prefix_at_token.equality;
            Fr weight, tmp;
            Fr::mul(weight, ch.eta_prefix, prefix);
            Fr::mul(tmp, ch.eta_next, next_prefix);
            Fr::add(weight, weight, tmp);
            Fr::add(weight, weight, Fr(1));
            Fr::mul(weight, weight, ch.token_batch[t]);
            Fr::mul(weight, weight, token_eq);
            Fr::add(result, result, weight);
        }
        return result;
    }

    Fr public_weight_at_point(
        const std::vector<Fr> &point,
        const std::vector<Fr> &token_coefficients,
        const std::vector<int> &output_tokens,
        int vocab_domain)
    {
        Fr result = Fr(0);
        for (int token = 0; token < static_cast<int>(output_tokens.size()); ++token)
        {
            std::size_t index = boolean_point_index(
                token, output_tokens[static_cast<std::size_t>(token)],
                vocab_domain);
            result += token_coefficients[static_cast<std::size_t>(token)] *
                      mle::equality_at(index, point);
        }
        return result;
    }

}

bool check_decode_sampling_claims(
    const DecodeSelectorProof &proof,
    const std::vector<G1> &expected_probability_commitments,
    int expected_vocab_size,
    const std::vector<ll> &sample_values,
    ll theta, ll epsilon,
    std::vector<DecodeSamplingValues> *values)
{
    if (values)
        values->clear();
    constexpr ll limit = static_cast<ll>(1) << 126;
    if (proof.token_count <= 0 ||
        proof.vocab_size <= 0 ||
        proof.vocab_size != expected_vocab_size ||
        theta <= 0 ||
        theta > limit / proof.vocab_size ||
        epsilon < 0 ||
        epsilon > limit - theta ||
        expected_probability_commitments.empty() ||
        expected_probability_commitments != proof.probability_commitments ||
        sample_values.size() != static_cast<size_t>(proof.token_count) ||
        proof.total_mass_claims.size() != sample_values.size() ||
        proof.prefix_claims.size() != sample_values.size() ||
        proof.next_prefix_claims.size() != sample_values.size())
        return false;

    const ll max_mass = theta * proof.vocab_size;
    const auto decode = [max_mass](const Fr &value, ll &result)
    {
        return zkagent_integer::decode_nonnegative_i128(value, max_mass, result);
    };
    std::vector<DecodeSamplingValues> checked(sample_values.size());
    for (size_t t = 0; t < sample_values.size(); ++t)
    {
        auto &row = checked[t];
        if (!decode(proof.total_mass_claims[t], row.total_mass) ||
            !decode(proof.prefix_claims[t], row.prefix) ||
            !decode(proof.next_prefix_claims[t], row.next_prefix) ||
            row.total_mass <= 0 ||
            row.total_mass < theta - epsilon ||
            row.total_mass > theta + epsilon ||
            row.prefix > row.next_prefix ||
            row.next_prefix > row.total_mass ||
            sample_values[t] < 0 ||
            sample_values[t] >= row.total_mass ||
            row.prefix > sample_values[t] ||
            sample_values[t] >= row.next_prefix)
        {
            return false;
        }
    }
    if (values)
        *values = std::move(checked);
    return true;
}

bool verify_decode_sampling_proof(
    const DecodeSelectorProof &proof,
    const std::vector<int> &output_tokens,
    const std::vector<G1> &expected_probability_commitments,
    int expected_vocab_size,
    const std::vector<ll> &sample_values,
    ll theta, ll epsilon,
    const MVKZG_VerifierKey &vk)
{
    return check_decode_sampling_claims(proof, expected_probability_commitments,
                                        expected_vocab_size, sample_values, theta, epsilon) &&
           verify_decode_selector_proof(proof, output_tokens, vk);
}

bool verify_decode_selector_proof(
    const DecodeSelectorProof &proof,
    const std::vector<int> &output_tokens,
    const MVKZG_VerifierKey &vk)
{
    FiatShamirTranscript transcript;
    return verify_decode_selector_proof(
        proof, output_tokens, vk, transcript);
}

bool verify_decode_selector_proof(
    const DecodeSelectorProof &proof,
    const std::vector<int> &output_tokens,
    const MVKZG_VerifierKey &vk,
    FiatShamirTranscript &transcript)
{
    if (proof.token_count <= 0 ||
        proof.vocab_size <= 0 ||
        static_cast<int>(output_tokens.size()) != proof.token_count ||
        proof.token_domain != next_power_of_two(proof.token_count) ||
        proof.vocab_domain != next_power_of_two(proof.vocab_size) ||
        proof.num_vars != log2_power_of_two(proof.token_domain) + log2_power_of_two(proof.vocab_domain) ||
        proof.total_mass_claims.size() != static_cast<std::size_t>(proof.token_count) ||
        proof.prefix_claims.size() != static_cast<std::size_t>(proof.token_count) ||
        proof.next_prefix_claims.size() != static_cast<std::size_t>(proof.token_count) ||
        proof.sumcheck_rounds.size() != static_cast<std::size_t>(proof.num_vars))
    {
        return false;
    }
    for (int token : output_tokens)
    {
        if (token < 0 || token >= proof.vocab_size)
            return false;
    }

    const int half = proof.num_vars / 2;
    const int row_count = 1 << half;
    if (proof.probability_commitments.size() != static_cast<std::size_t>(row_count))
    {
        return false;
    }
    bool ok = true;
    append_statement(transcript, proof, output_tokens);
    const SelectorChallenges ch = derive_batching_challenges(
        transcript, proof.token_count);
    Fr claim = batched_claim(proof, ch);
    std::vector<Fr> point;
    point.reserve(proof.num_vars);
    for (int round = 0; round < proof.num_vars && ok; ++round)
    {
        const DecodeSelectorRoundPoly &poly = proof.sumcheck_rounds[round];
        Fr at_one;
        Fr::add(at_one, poly.coeffs[0], poly.coeffs[1]);
        Fr::add(at_one, at_one, poly.coeffs[2]);
        if (poly.coeffs[0] + at_one != claim)
        {
            ok = false;
            break;
        }
        transcript.append_fr(poly.coeffs[0]);
        transcript.append_fr(poly.coeffs[1]);
        transcript.append_fr(poly.coeffs[2]);
        const Fr r = transcript.challenge();
        point.push_back(r);
        claim = poly.evaluate(r);
    }
    if (ok)
    {
        const Fr weight_eval = weight_mle_at_point(proof, output_tokens, ch, point);
        Fr expected;
        Fr::mul(expected, weight_eval, proof.probability_evaluation);
        ok = expected == claim;
    }
    if (ok)
    {
        ok = hyrax::verify(
            proof.probability_opening, point.data(), proof.probability_evaluation,
            proof.probability_commitments.data(),
            proof.num_vars, vk);
    }
    return ok;
}

bool verify_decode_batch_point_zero_proof(
    const DecodeBatchPointZeroProof &proof,
    const std::vector<int> &output_tokens,
    const MVKZG_VerifierKey &vk)
{
    FiatShamirTranscript transcript;
    return verify_decode_batch_point_zero_proof(
        proof, output_tokens, vk, transcript);
}

bool verify_decode_batch_point_zero_proof(
    const DecodeBatchPointZeroProof &proof,
    const std::vector<int> &output_tokens,
    const MVKZG_VerifierKey &vk,
    FiatShamirTranscript &transcript)
{
    if (proof.token_count <= 0 ||
        proof.vocab_size <= 0 ||
        output_tokens.size() != static_cast<std::size_t>(proof.token_count) ||
        proof.token_domain != next_power_of_two(proof.token_count) ||
        proof.vocab_domain != next_power_of_two(proof.vocab_size) ||
        proof.num_vars != log2_power_of_two(proof.token_domain) + log2_power_of_two(proof.vocab_domain) ||
        proof.sumcheck_rounds.size() != static_cast<std::size_t>(proof.num_vars))
    {
        return false;
    }
    for (int output : output_tokens)
    {
        if (output < 0 || output >= proof.vocab_size)
            return false;
    }
    const int row_count = 1 << (proof.num_vars / 2);
    if (proof.value_commitments.size() != static_cast<std::size_t>(row_count))
    {
        return false;
    }

    try
    {
        append_statement(transcript, proof, output_tokens);
        const std::vector<Fr> coefficients = derive_token_coefficients(
            transcript, proof.token_count);

        Fr claim = Fr(0);
        std::vector<Fr> point;
        point.reserve(static_cast<std::size_t>(proof.num_vars));
        for (int round = 0; round < proof.num_vars; ++round)
        {
            const DecodeBatchPointRound &message =
                proof.sumcheck_rounds[static_cast<std::size_t>(round)];
            if (message.evaluations[0] + message.evaluations[1] != claim)
            {
                return false;
            }
            transcript.append_fr(message.evaluations[0]);
            transcript.append_fr(message.evaluations[1]);
            transcript.append_fr(message.evaluations[2]);
            const Fr challenge = transcript.challenge_not_one();
            point.push_back(challenge);
            claim = message.evaluate(challenge);
        }

        transcript.append_fr(proof.final_evaluation);
        const Fr public_weight = public_weight_at_point(
            point, coefficients, output_tokens, proof.vocab_domain);
        if (claim != public_weight * proof.final_evaluation)
            return false;

        return hyrax::verify(
            proof.opening, point.data(), proof.final_evaluation,
            proof.value_commitments.data(),
            proof.num_vars, vk);
    }
    catch (const std::exception &)
    {
        return false;
    }
}
