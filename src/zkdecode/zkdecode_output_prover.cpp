#include "zkdecode/zkdecode_output.hpp"
#include "zkdecode/zkdecode_protocol.hpp"
#include "common/bit_utils.hpp"
#include "common/field_integer.hpp"
#include "pcs/srs.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>
#include <omp.h>

using namespace zkdecode_detail::output;

namespace
{
    using Clock = std::chrono::steady_clock;
    using zkagent_integer::fr_from_i128;
    using zkagent_integer::next_power_of_two;
    using zkagent_integer::log2_power_of_two;
}

DecodeSelectorResult prove_decode_selector(
    const std::vector<std::vector<ll>> &probability_mass,
    const std::vector<int> &output_tokens,
    const MVKZG_ProverKey &pk,
    const MVKZG_VerifierKey &vk,
    int threads,
    FiatShamirTranscript &prover_transcript,
    const std::vector<G1> *shared_probability_commitments)
{
    DecodeSelectorResult result;
    if (probability_mass.empty() || probability_mass.size() != output_tokens.size())
    {
        return result;
    }
    const int T = static_cast<int>(probability_mass.size());
    const int N = static_cast<int>(probability_mass[0].size());
    if (N <= 0)
        return result;
    for (int t = 0; t < T; ++t)
    {
        if (static_cast<int>(probability_mass[t].size()) != N || output_tokens[t] < 0 || output_tokens[t] >= N)
        {
            return result;
        }
        for (ll value : probability_mass[t])
        {
            if (value < 0)
                return result;
        }
    }

    DecodeSelectorProof &proof = result.proof;
    proof.token_count = T;
    proof.vocab_size = N;
    proof.token_domain = next_power_of_two(T);
    proof.vocab_domain = next_power_of_two(N);
    proof.num_vars = log2_power_of_two(proof.token_domain) + log2_power_of_two(proof.vocab_domain);
    const int domain = proof.token_domain * proof.vocab_domain;

    std::vector<Fr> p(domain, Fr(0));
    proof.total_mass_claims.resize(T, Fr(0));
    proof.prefix_claims.resize(T, Fr(0));
    proof.next_prefix_claims.resize(T, Fr(0));
    for (int t = 0; t < T; ++t)
    {
        for (int i = 0; i < N; ++i)
        {
            const Fr value = fr_from_i128(probability_mass[t][i]);
            p[t * proof.vocab_domain + i] = value;
            Fr::add(proof.total_mass_claims[t], proof.total_mass_claims[t], value);
            if (i < output_tokens[t])
            {
                Fr::add(proof.prefix_claims[t], proof.prefix_claims[t], value);
            }
            if (i <= output_tokens[t])
            {
                Fr::add(proof.next_prefix_claims[t], proof.next_prefix_claims[t], value);
            }
        }
    }

    const int half = proof.num_vars / 2;
    const int row_count = 1 << half;
    const int col_count = 1 << (proof.num_vars - half);
    if (pk.srs_g1.size() < static_cast<std::size_t>(col_count) ||
        vk.vars < proof.num_vars - half)
    {
        return result;
    }
    std::vector<G1> generators;
    if (shared_probability_commitments == nullptr)
    {
        generators.resize(col_count);
        hyrax::load_generators(generators.data(), pk.srs_g1, col_count);
    }

    const Clock::time_point prove_start = Clock::now();
    if (shared_probability_commitments != nullptr)
    {
        if (shared_probability_commitments->size() != static_cast<std::size_t>(row_count))
        {
            return result;
        }
        proof.probability_commitments = *shared_probability_commitments;
    }
    else
    {
        G1 *commitments = hyrax::commit(
            p.data(), generators.data(), proof.num_vars, std::max(threads, 1));
        proof.probability_commitments.assign(commitments, commitments + row_count);
        delete[] commitments;
    }

    append_statement(prover_transcript, proof, output_tokens);
    const SelectorChallenges ch = derive_batching_challenges(
        prover_transcript, T);
    std::vector<Fr> weight(domain, Fr(0));
    for (int t = 0; t < T; ++t)
    {
        for (int i = 0; i < proof.vocab_domain; ++i)
        {
            Fr value = Fr(1);
            if (i < output_tokens[t])
                Fr::add(value, value, ch.eta_prefix);
            if (i <= output_tokens[t])
                Fr::add(value, value, ch.eta_next);
            Fr::mul(value, value, ch.token_batch[t]);
            weight[t * proof.vocab_domain + i] = value;
        }
    }

    std::vector<Fr> challenges;
    challenges.reserve(proof.num_vars);
    std::vector<Fr> p_work = std::move(p);
    std::vector<Fr> opening_values;
    if (half == 0)
        opening_values = p_work;
    int current_size = domain;
    for (int round = 0; round < proof.num_vars; ++round)
    {
        DecodeSelectorRoundPoly round_poly;
        const int pairs = current_size / 2;
        for (int pair = 0; pair < pairs; ++pair)
        {
            const Fr w0 = weight[2 * pair];
            const Fr w1 = weight[2 * pair + 1];
            const Fr p0 = p_work[2 * pair];
            const Fr p1 = p_work[2 * pair + 1];
            Fr dw, dp, term;
            Fr::sub(dw, w1, w0);
            Fr::sub(dp, p1, p0);
            Fr::mul(term, w0, p0);
            Fr::add(round_poly.coeffs[0], round_poly.coeffs[0], term);
            Fr::mul(term, w0, dp);
            Fr cross;
            Fr::mul(cross, p0, dw);
            Fr::add(term, term, cross);
            Fr::add(round_poly.coeffs[1], round_poly.coeffs[1], term);
            Fr::mul(term, dw, dp);
            Fr::add(round_poly.coeffs[2], round_poly.coeffs[2], term);
        }
        proof.sumcheck_rounds.push_back(round_poly);
        prover_transcript.append_fr(round_poly.coeffs[0]);
        prover_transcript.append_fr(round_poly.coeffs[1]);
        prover_transcript.append_fr(round_poly.coeffs[2]);
        const Fr r = prover_transcript.challenge();
        challenges.push_back(r);
        for (int pair = 0; pair < pairs; ++pair)
        {
            Fr dw, dp;
            Fr::sub(dw, weight[2 * pair + 1], weight[2 * pair]);
            Fr::sub(dp, p_work[2 * pair + 1], p_work[2 * pair]);
            Fr::mul(dw, dw, r);
            Fr::mul(dp, dp, r);
            Fr::add(weight[pair], weight[2 * pair], dw);
            Fr::add(p_work[pair], p_work[2 * pair], dp);
        }
        current_size = pairs;
        // Hyrax rows use the low bits, already folded by these sumcheck rounds.
        if (round + 1 == half)
            opening_values.assign(p_work.begin(), p_work.begin() + pairs);
    }
    proof.probability_evaluation = p_work[0];
    try
    {
        proof.probability_opening = hyrax::open_folded(
            opening_values, challenges.data(), proof.num_vars, pk);
    }
    catch (const std::runtime_error &)
    {
        return result;
    }
    result.prover_time = std::chrono::duration<double>(Clock::now() - prove_start).count();

    result.success = true;
    return result;
}

DecodeBatchPointZeroResult prove_decode_batch_point_zero(
    const std::vector<std::vector<ll>> &values,
    const std::vector<int> &output_tokens,
    const MVKZG_ProverKey &pk,
    const MVKZG_VerifierKey &vk,
    int threads,
    FiatShamirTranscript &prover_transcript,
    const std::vector<G1> *shared_value_commitments)
{
    DecodeBatchPointZeroResult result;
    if (values.empty() || values.size() != output_tokens.size())
        return result;
    const int T = static_cast<int>(values.size());
    const int N = static_cast<int>(values[0].size());
    if (N <= 0)
        return result;
    for (int token = 0; token < T; ++token)
    {
        if (values[static_cast<std::size_t>(token)].size() != static_cast<std::size_t>(N) ||
            output_tokens[static_cast<std::size_t>(token)] < 0 ||
            output_tokens[static_cast<std::size_t>(token)] >= N)
        {
            return result;
        }
    }

    DecodeBatchPointZeroProof &proof = result.proof;
    proof.token_count = T;
    proof.vocab_size = N;
    proof.token_domain = next_power_of_two(T);
    proof.vocab_domain = next_power_of_two(N);
    proof.num_vars = log2_power_of_two(proof.token_domain) + log2_power_of_two(proof.vocab_domain);
    const std::size_t domain = static_cast<std::size_t>(proof.token_domain) * static_cast<std::size_t>(proof.vocab_domain);

    std::vector<Fr> work(domain, Fr(0));
    for (int token = 0; token < T; ++token)
    {
        for (int index = 0; index < N; ++index)
        {
            work[static_cast<std::size_t>(token) * proof.vocab_domain + index] =
                fr_from_i128(values[static_cast<std::size_t>(token)]
                                 [static_cast<std::size_t>(index)]);
        }
    }

    const int row_vars = proof.num_vars / 2;
    const int row_count = 1 << row_vars;
    const int col_count = 1 << (proof.num_vars - row_vars);
    if (pk.srs_g1.size() < static_cast<std::size_t>(col_count) ||
        vk.vars < proof.num_vars - proof.num_vars / 2)
    {
        return result;
    }
    std::vector<G1> generators;
    if (shared_value_commitments == nullptr)
    {
        generators.resize(static_cast<std::size_t>(col_count));
        hyrax::load_generators(generators.data(), pk.srs_g1, col_count);
    }

    const Clock::time_point prover_start = Clock::now();
    if (shared_value_commitments != nullptr)
    {
        if (shared_value_commitments->size() != static_cast<std::size_t>(row_count))
        {
            return result;
        }
        proof.value_commitments = *shared_value_commitments;
    }
    else
    {
        G1 *commitments = hyrax::commit(
            work.data(), generators.data(), proof.num_vars,
            std::max(threads, 1));
        proof.value_commitments.assign(commitments, commitments + row_count);
        delete[] commitments;
    }

    append_statement(prover_transcript, proof, output_tokens);
    std::vector<Fr> coefficients = derive_token_coefficients(prover_transcript, T);
    std::vector<std::size_t> remaining_indices(static_cast<std::size_t>(T));
    for (int token = 0; token < T; ++token)
    {
        remaining_indices[static_cast<std::size_t>(token)] =
            boolean_point_index(
                token, output_tokens[static_cast<std::size_t>(token)],
                proof.vocab_domain);
    }

    Fr claim = Fr(0);
    std::vector<Fr> point;
    point.reserve(static_cast<std::size_t>(proof.num_vars));
    std::vector<Fr> opening_values;
    if (row_vars == 0)
        opening_values = work;
    std::size_t active = domain;
    for (int round = 0; round < proof.num_vars; ++round)
    {
        DecodeBatchPointRound message;
        for (int token = 0; token < T; ++token)
        {
            const std::size_t t = static_cast<std::size_t>(token);
            const bool bit = (remaining_indices[t] & 1u) != 0;
            const std::size_t pair = remaining_indices[t] >> 1;
            const Fr f0 = work[2 * pair];
            const Fr f1 = work[2 * pair + 1];
            for (int sample = 0; sample <= 2; ++sample)
            {
                const Fr x(sample);
                const Fr equality = bit ? x : Fr(1) - x;
                const Fr f_value = f0 + x * (f1 - f0);
                message.evaluations[static_cast<std::size_t>(sample)] +=
                    coefficients[t] * equality * f_value;
            }
        }
        if (message.evaluations[0] + message.evaluations[1] != claim)
        {
            return result;
        }
        proof.sumcheck_rounds.push_back(message);
        prover_transcript.append_fr(message.evaluations[0]);
        prover_transcript.append_fr(message.evaluations[1]);
        prover_transcript.append_fr(message.evaluations[2]);
        const Fr challenge = prover_transcript.challenge_not_one();
        point.push_back(challenge);
        claim = message.evaluate(challenge);

        for (int token = 0; token < T; ++token)
        {
            const std::size_t t = static_cast<std::size_t>(token);
            coefficients[t] *= (remaining_indices[t] & 1u)
                                   ? challenge
                                   : Fr(1) - challenge;
            remaining_indices[t] >>= 1;
        }

        const std::size_t pairs = active / 2;
        std::vector<Fr> folded(pairs);
#pragma omp parallel for num_threads(threads) if (pairs >= 4096)
        for (long long pair = 0; pair < static_cast<long long>(pairs); ++pair)
        {
            const std::size_t p = static_cast<std::size_t>(pair);
            folded[p] = work[2 * p] + challenge * (work[2 * p + 1] - work[2 * p]);
        }
        work.swap(folded);
        active = pairs;
        // Preserve the PCS row reduction before the remaining rounds consume it.
        if (round + 1 == row_vars)
            opening_values = work;
    }

    proof.final_evaluation = work[0];
    prover_transcript.append_fr(proof.final_evaluation);
    Fr final_weight = Fr(0);
    for (const Fr &coefficient : coefficients)
        final_weight += coefficient;
    if (claim != final_weight * proof.final_evaluation)
        return result;
    try
    {
        proof.opening = hyrax::open_folded(
            opening_values, point.data(), proof.num_vars, pk);
    }
    catch (const std::runtime_error &)
    {
        return result;
    }
    result.prover_time = std::chrono::duration<double>(
                             Clock::now() - prover_start)
                             .count();

    result.success = true;
    return result;
}
