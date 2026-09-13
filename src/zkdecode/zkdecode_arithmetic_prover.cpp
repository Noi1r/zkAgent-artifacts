#include "zkdecode/zkdecode_arithmetic.hpp"
#include "zkdecode/zkdecode_protocol.hpp"
#include "pcs/srs.hpp"

#include <algorithm>
#include <chrono>
#include <omp.h>

using namespace zkdecode_detail::arithmetic;

namespace {
    using Clock = std::chrono::steady_clock;
    using mle::equality_weights;
    bool valid_witness_view(const DecodeTopKLinkWitnessView &witness)
    {
        if (witness.token_count <= 0 || witness.vocab_size <= 0 ||
            witness.k <= 0 || witness.k > witness.vocab_size)
            return false;
        const auto tokens = static_cast<std::size_t>(witness.token_count);
        const bool row_counts_match =
            witness.logits.size() == tokens && witness.deltas.size() == tokens &&
            witness.selectors.size() == tokens && witness.middle_outputs.size() == tokens &&
            witness.high_zero_0.size() == tokens && witness.high_zero_1.size() == tokens &&
            witness.probabilities.size() == tokens && witness.separators.size() == tokens &&
            witness.shifts.size() == tokens;
        const bool candidate_columns_match =
            witness.candidate_in.empty() == witness.candidate_out.empty() &&
            (witness.candidate_in.empty() ||
             (witness.candidate_in.size() == tokens && witness.candidate_out.size() == tokens));
        if (!row_counts_match || !candidate_columns_match)
            return false;
        for (const auto &digit : witness.digits)
        {
            if (digit.size() != static_cast<size_t>(witness.token_count))
            {
                return false;
            }
        }
        for (int token = 0; token < witness.token_count; ++token)
        {
            const size_t t = static_cast<size_t>(token);
            if (witness.logits[t] == nullptr ||
                witness.deltas[t] == nullptr ||
                witness.selectors[t] == nullptr ||
                witness.middle_outputs[t] == nullptr ||
                witness.high_zero_0[t] == nullptr ||
                witness.high_zero_1[t] == nullptr ||
                witness.probabilities[t] == nullptr ||
                (!witness.candidate_in.empty() &&
                (witness.candidate_in[t] == nullptr ||
                witness.candidate_out[t] == nullptr)))
            {
                return false;
            }
            for (const auto &digit : witness.digits)
            {
                if (digit[t] == nullptr)
                    return false;
            }
        }
        return true;
    }

    Fr witness_value_at(
        const DecodeTopKLinkWitnessView &witness,
        DecodeTopKLinkColumn column,
        int vocab_domain,
        std::size_t padded_index)
    {
        const int token = static_cast<int>(padded_index / vocab_domain);
        const int index = static_cast<int>(padded_index % vocab_domain);
        if (token >= witness.token_count || index >= witness.vocab_size)
        {
            if (column == DecodeTopKLinkColumn::MiddleOutput)
            {
                return fr_from_i128(ZkDecodeProbabilityScale);
            }
            if (column == DecodeTopKLinkColumn::HighZero0 || column == DecodeTopKLinkColumn::HighZero1)
            {
                return Fr(1);
            }
            return Fr(0);
        }
        const std::size_t t = static_cast<std::size_t>(token);
        const std::size_t i = static_cast<std::size_t>(index);
        switch (column)
        {
        case DecodeTopKLinkColumn::Logit:
            return fr_from_i128(witness.logits[t][i]);
        case DecodeTopKLinkColumn::Separator:
            return fr_from_i128(witness.separators[t]);
        case DecodeTopKLinkColumn::Shift:
            return fr_from_i128(witness.shifts[t]);
        case DecodeTopKLinkColumn::Delta:
            return fr_from_i128(witness.deltas[t][i]);
        case DecodeTopKLinkColumn::Selector:
            return fr_from_i128(witness.selectors[t][i]);
        case DecodeTopKLinkColumn::CandidateIn:
            return witness.candidate_in.empty()
                       ? fr_from_i128(witness.selectors[t][i]
                                        ? witness.deltas[t][i]
                                        : 0)
                       : fr_from_i128(witness.candidate_in[t][i]);
        case DecodeTopKLinkColumn::CandidateOut:
            return witness.candidate_out.empty()
                       ? fr_from_i128(witness.selectors[t][i]
                                        ? 0
                                        : -witness.deltas[t][i])
                       : fr_from_i128(witness.candidate_out[t][i]);
        case DecodeTopKLinkColumn::Digit0:
        case DecodeTopKLinkColumn::Digit1:
        case DecodeTopKLinkColumn::Digit2:
        case DecodeTopKLinkColumn::Digit3:
        case DecodeTopKLinkColumn::Digit4:
        case DecodeTopKLinkColumn::Digit5:
            return fr_from_i128(witness.digits[static_cast<std::size_t>(column_index(column) - column_index(DecodeTopKLinkColumn::Digit0))][t][i]);
        case DecodeTopKLinkColumn::MiddleOutput:
            return fr_from_i128(witness.middle_outputs[t][i]);
        case DecodeTopKLinkColumn::HighZero0:
            return fr_from_i128(witness.high_zero_0[t][i]);
        case DecodeTopKLinkColumn::HighZero1:
            return fr_from_i128(witness.high_zero_1[t][i]);
        case DecodeTopKLinkColumn::Probability:
            return fr_from_i128(witness.probabilities[t][i]);
        }
        throw std::logic_error("unknown top-k link column");
    }

    std::vector<Fr> materialize_column(
        const DecodeTopKLinkWitnessView &witness,
        DecodeTopKLinkColumn column,
        int token_domain,
        int vocab_domain)
    {
        const std::size_t domain = static_cast<std::size_t>(token_domain) * vocab_domain;
        std::vector<Fr> values(domain);
        for (std::size_t index = 0; index < domain; ++index)
        {
            values[index] = witness_value_at(
                witness, column, vocab_domain, index);
        }
        return values;
    }

    bool valid_greedy_view(const DecodeGreedyLinkWitnessView &witness)
    {
        if (witness.token_count <= 0 ||
            witness.vocab_size <= 0 ||
            witness.logits.size() != static_cast<size_t>(witness.token_count) ||
            witness.negative_deltas.size() != static_cast<size_t>(witness.token_count) ||
            witness.lambdas.size() != static_cast<size_t>(witness.token_count))
        {
            return false;
        }
        for (int token = 0; token < witness.token_count; ++token)
        {
            if (witness.logits[static_cast<size_t>(token)] == nullptr || witness.negative_deltas[static_cast<size_t>(token)] == nullptr)
            {
                return false;
            }
        }
        return true;
    }

    Fr greedy_value_at(
        const DecodeGreedyLinkWitnessView &witness,
        DecodeGreedyLinkColumn column,
        int vocab_domain,
        size_t padded_index)
    {
        const int token = static_cast<int>(padded_index / vocab_domain);
        const int index = static_cast<int>(padded_index % vocab_domain);
        if (token >= witness.token_count || index >= witness.vocab_size)
        {
            return Fr(0);
        }
        const size_t t = static_cast<size_t>(token);
        const size_t i = static_cast<size_t>(index);
        switch (column)
        {
        case DecodeGreedyLinkColumn::Logit:
            return fr_from_i128(witness.logits[t][i]);
        case DecodeGreedyLinkColumn::Lambda:
            return fr_from_i128(witness.lambdas[t]);
        case DecodeGreedyLinkColumn::NegativeDelta:
            return fr_from_i128(witness.negative_deltas[t][i]);
        }
        throw std::logic_error("unknown greedy link column");
    }

    std::vector<Fr> materialize_greedy_column(
        const DecodeGreedyLinkWitnessView &witness,
        DecodeGreedyLinkColumn column,
        int token_domain,
        int vocab_domain)
    {
        const size_t domain = static_cast<size_t>(token_domain) * vocab_domain;
        std::vector<Fr> values(domain);
        for (size_t index = 0; index < domain; ++index)
        {
            values[index] = greedy_value_at(
                witness, column, vocab_domain, index);
        }
        return values;
    }

} // namespace

DecodeTopKLinkResult prove_decode_topk_link_view(
    const DecodeTopKLinkWitnessView &witness,
    const MVKZG_ProverKey &pk,
    const MVKZG_VerifierKey &vk,
    int threads,
    FiatShamirTranscript &prover_transcript,
    DecodeLogitEvaluation *deferred_logit_claim)
{
    DecodeTopKLinkResult result;
    if (!valid_witness_view(witness))
        return result;

    DecodeTopKLinkProof &proof = result.proof;
    proof.defer_logit_opening = deferred_logit_claim != nullptr;
    proof.token_count = witness.token_count;
    proof.vocab_size = witness.vocab_size;
    proof.k = witness.k;
    proof.token_domain = next_power_of_two(witness.token_count);
    proof.vocab_domain = next_power_of_two(witness.vocab_size);
    proof.num_vars = log2_power_of_two(proof.token_domain) + log2_power_of_two(proof.vocab_domain);
    const int domain = proof.token_domain * proof.vocab_domain;
    const int row_count = 1 << (proof.num_vars / 2);
    const int col_count = 1 << (proof.num_vars - proof.num_vars / 2);
    const int token_vars = log2_power_of_two(proof.token_domain);
    const int shift_row_count = 1 << (token_vars / 2);
    const int shift_col_count = 1 << (token_vars - token_vars / 2);

    // Per-token shift and separator bind their broadcasts across the vocabulary.
    std::array<std::vector<Fr>, 2> shift_columns;
    for (auto &column : shift_columns)
    {
        column.assign(static_cast<size_t>(proof.token_domain), Fr(0));
    }
    for (int token = 0; token < witness.token_count; ++token)
    {
        shift_columns[0][static_cast<size_t>(token)] =
            fr_from_i128(witness.shifts[static_cast<size_t>(token)]);
        shift_columns[1][static_cast<size_t>(token)] =
            fr_from_i128(witness.separators[static_cast<size_t>(token)]);
    }
    const int required_columns = std::max(col_count, shift_col_count);
    if (pk.srs_g1.size() < static_cast<std::size_t>(required_columns) ||
        vk.vars < std::max(proof.num_vars - proof.num_vars / 2,
                           token_vars - token_vars / 2))
    {
        return result;
    }
    std::vector<G1> generators(col_count);
    hyrax::load_generators(
        generators.data(), pk.srs_g1, col_count);
    std::vector<G1> shift_generators(shift_col_count);
    hyrax::load_generators(
        shift_generators.data(), pk.srs_g1, shift_col_count);

    const Clock::time_point prover_start = Clock::now();
    for (int column = proof.defer_logit_opening ? 1 : 0; column < DecodeTopKLinkColumns; ++column)
    {
        std::vector<Fr> values = materialize_column(
            witness, static_cast<DecodeTopKLinkColumn>(column),
            proof.token_domain, proof.vocab_domain);
        G1 *commitment = hyrax::commit(
            values.data(), generators.data(), proof.num_vars,
            std::max(threads, 1));
        proof.column_commitments[column].assign(
            commitment, commitment + row_count);
        delete[] commitment;
    }
    for (int column = 0; column < 2; ++column)
    {
        G1 *commitment = hyrax::commit(
            shift_columns[column].data(), shift_generators.data(), token_vars,
            std::max(threads, 1));
        auto &target = column == 0 ? proof.shift_commitments : proof.separator_commitments;
        target.assign(commitment, commitment + shift_row_count);
        delete[] commitment;
    }

    append_statement(prover_transcript, proof, witness.token_count,
                     witness.vocab_size, witness.k);
    append_tuple_table_context(prover_transcript);
    proof.tuple_compression_challenge =
        prover_transcript.nonzero_challenge();
    const LinkChallenges challenges = derive_link_challenges(
        prover_transcript, proof.num_vars, token_vars);

    std::vector<Fr> zero_weight = equality_weights(challenges.zero_point);
    const std::vector<Fr> token_weight = equality_weights(challenges.token_point);
    std::vector<Fr> cardinality_work = token_weight;
    const int vocab_vars = log2_power_of_two(proof.vocab_domain);

    Fr claim = challenges.cardinality * Fr(witness.k) * token_prefix_mle(witness.token_count, challenges.token_point);
    std::array<std::vector<Fr>, DecodeTopKLinkColumns> work;
    std::vector<Fr> active_mask_work;
    const auto active_cell = [&](std::size_t index)
    {
        return Fr(index / proof.vocab_domain < std::size_t(witness.token_count) &&
                          index % proof.vocab_domain < std::size_t(witness.vocab_size)
                      ? 1
                      : 0);
    };
    if (proof.num_vars == 0)
    {
        for (int column = 0; column < DecodeTopKLinkColumns; ++column)
            work[column] = {witness_value_at(witness, static_cast<DecodeTopKLinkColumn>(column), proof.vocab_domain, 0)};
    }
    int active = domain;
    std::vector<Fr> sumcheck_point;
    sumcheck_point.reserve(proof.num_vars);
    for (int round = 0; round < proof.num_vars; ++round)
    {
        const int pairs = active / 2;
        DecodeTopKLinkRound message;
        for (Fr &value : message.evaluations)
            value = Fr(0);
        const int thread_count = std::max(1, threads);
        std::vector<DecodeTopKLinkRound> local_messages(
            static_cast<size_t>(thread_count));
        for (auto &local : local_messages)
        {
            for (Fr &value : local.evaluations)
                value = Fr(0);
        }
#pragma omp parallel num_threads(thread_count) if (pairs >= 4096)
        {
            const int tid = omp_get_thread_num();
            DecodeTopKLinkRound &local = local_messages[static_cast<size_t>(tid)];
#pragma omp for schedule(static)
            for (int pair = 0; pair < pairs; ++pair)
            {
                const int left = 2 * pair;
                const int right = left + 1;
                std::array<Fr, DecodeTopKLinkColumns> left_values;
                std::array<Fr, DecodeTopKLinkColumns> differences;
                for (int column = 0; column < DecodeTopKLinkColumns; ++column)
                {
                    left_values[column] = round == 0
                                              ? witness_value_at(
                                                    witness, static_cast<DecodeTopKLinkColumn>(column),
                                                    proof.vocab_domain, static_cast<size_t>(left))
                                              : work[column][left];
                    const Fr right_value = round == 0
                                               ? witness_value_at(
                                                     witness, static_cast<DecodeTopKLinkColumn>(column),
                                                     proof.vocab_domain, static_cast<size_t>(right))
                                               : work[column][right];
                    differences[column] = right_value - left_values[column];
                }
                for (int sample = 0; sample <= DecodeTopKLinkDegree; ++sample)
                {
                    const Fr x(sample);
                    std::array<Fr, DecodeTopKLinkColumns> value;
                    for (int column = 0; column < DecodeTopKLinkColumns; ++column)
                    {
                        value[column] = left_values[column] + differences[column] * x;
                    }
                    const Fr eq = zero_weight[left] + (zero_weight[right] - zero_weight[left]) * x;
                    Fr card;
                    if (round < vocab_vars)
                    {
                        const int row_width = proof.vocab_domain >> round;
                        card = token_weight[static_cast<size_t>(left / row_width)];
                    }
                    else
                    {
                        card = cardinality_work[left] + (cardinality_work[right] - cardinality_work[left]) * x;
                    }
                    local.evaluations[static_cast<std::size_t>(sample)] +=
                        eq * local_relation(value, challenges.relation,
                                            (round == 0 ? active_cell(left) : active_mask_work[left]) +
                                                x * ((round == 0 ? active_cell(right) : active_mask_work[right]) -
                                                     (round == 0 ? active_cell(left) : active_mask_work[left]))) +
                        challenges.cardinality * card * value[column_index(DecodeTopKLinkColumn::Selector)];
                }
            }
        }
        for (const auto &local : local_messages)
        {
            for (int sample = 0; sample <= DecodeTopKLinkDegree; ++sample)
            {
                message.evaluations[static_cast<size_t>(sample)] +=
                    local.evaluations[static_cast<size_t>(sample)];
            }
        }

        if (message.evaluations[0] + message.evaluations[1] != claim)
        {
            return result;
        }
        proof.sumcheck_rounds.push_back(message);
        for (int sample = 0; sample <= DecodeTopKLinkDegree; ++sample)
        {
            prover_transcript.append_fr(message.evaluations[static_cast<std::size_t>(sample)]);
        }
        const Fr challenge = prover_transcript.challenge_not_one();
        sumcheck_point.push_back(challenge);
        claim = message.evaluate(challenge);

        if (round == 0)
        {
            active_mask_work.resize(pairs);
            for (int pair = 0; pair < pairs; ++pair)
                active_mask_work[pair] = active_cell(2 * pair) + challenge *
                                                                     (active_cell(2 * pair + 1) - active_cell(2 * pair));
            for (int column = 0; column < DecodeTopKLinkColumns; ++column)
            {
                work[column].resize(static_cast<size_t>(pairs));
#pragma omp parallel for num_threads(thread_count) if (pairs >= 4096)
                for (int pair = 0; pair < pairs; ++pair)
                {
                    const size_t left = static_cast<size_t>(2 * pair);
                    const size_t right = left + 1;
                    const Fr left_value = witness_value_at(
                        witness, static_cast<DecodeTopKLinkColumn>(column),
                        proof.vocab_domain, left);
                    const Fr right_value = witness_value_at(
                        witness, static_cast<DecodeTopKLinkColumn>(column),
                        proof.vocab_domain, right);
                    work[column][static_cast<size_t>(pair)] = left_value + (right_value - left_value) * challenge;
                }
            }
        }
        else
        {
            for (int pair = 0; pair < pairs; ++pair)
            {
                active_mask_work[pair] = active_mask_work[2 * pair] + challenge *
                                                                          (active_mask_work[2 * pair + 1] - active_mask_work[2 * pair]);
                const int left = 2 * pair;
                const int right = left + 1;
                for (int column = 0; column < DecodeTopKLinkColumns; ++column)
                {
                    work[column][pair] = work[column][left] + (work[column][right] - work[column][left]) * challenge;
                }
            }
            for (auto &column : work)
                column.resize(pairs);
        }
        for (int pair = 0; pair < pairs; ++pair)
        {
            const int left = 2 * pair;
            const int right = left + 1;
            zero_weight[pair] = zero_weight[left] + (zero_weight[right] - zero_weight[left]) * challenge;
        }
        if (round >= vocab_vars)
        {
            for (int pair = 0; pair < pairs; ++pair)
            {
                const int left = 2 * pair;
                const int right = left + 1;
                cardinality_work[pair] = cardinality_work[left] + (cardinality_work[right] - cardinality_work[left]) * challenge;
            }
            cardinality_work.resize(pairs);
        }
        zero_weight.resize(pairs);
        active = pairs;
    }

    for (int column = 0; column < DecodeTopKLinkColumns; ++column)
    {
        proof.final_evaluations[column] = work[column][0];
        prover_transcript.append_fr(proof.final_evaluations[column]);
    }
    std::vector<Fr> index_point(
        sumcheck_point.begin(), sumcheck_point.begin() + vocab_vars);
    std::vector<Fr> row_point(
        sumcheck_point.begin() + vocab_vars, sumcheck_point.end());
    proof.shift_evaluation = mle::evaluate(shift_columns[0], row_point);
    prover_transcript.append_fr(proof.shift_evaluation);
    proof.separator_evaluation = mle::evaluate(shift_columns[1], row_point);
    prover_transcript.append_fr(proof.separator_evaluation);
    const Fr final_relation = local_relation(
        proof.final_evaluations, challenges.relation,
        token_prefix_mle(witness.token_count, row_point) * token_prefix_mle(witness.vocab_size, index_point));
    const Fr expected_final = zero_weight[0] * final_relation + challenges.cardinality * cardinality_work[0] * proof.final_evaluations[column_index(DecodeTopKLinkColumn::Selector)];
    if (claim != expected_final)
        return result;
    if (proof.final_evaluations[column_index(DecodeTopKLinkColumn::Shift)] != proof.shift_evaluation * token_prefix_mle(witness.vocab_size, index_point))
    {
        return result;
    }
    if (proof.final_evaluations[column_index(DecodeTopKLinkColumn::Separator)] !=
        proof.separator_evaluation * token_prefix_mle(witness.vocab_size, index_point))
        return result;

    std::array<Fr, DecodeTopKLinkColumns> opening_weights;
    std::vector<Fr> combined(domain, Fr(0));
    for (int column = proof.defer_logit_opening ? 1 : 0; column < DecodeTopKLinkColumns; ++column)
    {
        opening_weights[column] = prover_transcript.nonzero_challenge();
    }
#pragma omp parallel for num_threads(threads) if (domain >= 4096)
    for (int index = 0; index < domain; ++index)
    {
        Fr value = Fr(0);
        for (int column = proof.defer_logit_opening ? 1 : 0; column < DecodeTopKLinkColumns; ++column)
        {
            value += opening_weights[column] * witness_value_at(
                                                   witness, static_cast<DecodeTopKLinkColumn>(column),
                                                   proof.vocab_domain, static_cast<size_t>(index));
        }
        combined[static_cast<size_t>(index)] = value;
    }
    proof.combined_opening = hyrax::open(
        combined.data(), sumcheck_point.data(), proof.num_vars, pk);

    std::array<Fr, 2> shift_opening_weights;
    std::vector<Fr> combined_shift(static_cast<size_t>(proof.token_domain), Fr(0));
    for (int column = 0; column < 2; ++column)
    {
        shift_opening_weights[column] = prover_transcript.nonzero_challenge();
        for (int token = 0; token < proof.token_domain; ++token)
        {
            combined_shift[static_cast<size_t>(token)] +=
                shift_opening_weights[column] * shift_columns[column][static_cast<size_t>(token)];
        }
    }
    proof.shift_opening = hyrax::open(
        combined_shift.data(), row_point.data(), token_vars, pk);
    result.prover_time = std::chrono::duration<double>(
                             Clock::now() - prover_start)
                             .count();

    if (deferred_logit_claim)
    {
        deferred_logit_claim->point = sumcheck_point;
        deferred_logit_claim->evaluation = proof.final_evaluations[0];
    }
    result.success = true;
    return result;
}

DecodeGreedyLinkResult prove_decode_greedy_link(
    const DecodeGreedyLinkWitnessView &witness,
    const MVKZG_ProverKey &pk,
    const MVKZG_VerifierKey &vk,
    int threads,
    FiatShamirTranscript &prover_transcript,
    DecodeLogitEvaluation *deferred_logit_claim)
{
    DecodeGreedyLinkResult result;
    if (!valid_greedy_view(witness))
        return result;

    DecodeGreedyLinkProof &proof = result.proof;
    proof.defer_logit_opening = deferred_logit_claim != nullptr;
    proof.token_count = witness.token_count;
    proof.vocab_size = witness.vocab_size;
    proof.token_domain = next_power_of_two(witness.token_count);
    proof.vocab_domain = next_power_of_two(witness.vocab_size);
    proof.num_vars = log2_power_of_two(proof.token_domain) + log2_power_of_two(proof.vocab_domain);
    const int domain = proof.token_domain * proof.vocab_domain;
    const int row_count = 1 << (proof.num_vars / 2);
    const int col_count = 1 << (proof.num_vars - proof.num_vars / 2);
    if (pk.srs_g1.size() < static_cast<std::size_t>(col_count) ||
        vk.vars < proof.num_vars - proof.num_vars / 2)
    {
        return result;
    }
    std::vector<G1> generators(static_cast<size_t>(col_count));
    hyrax::load_generators(
        generators.data(), pk.srs_g1, col_count);

    const Clock::time_point prover_start = Clock::now();
    std::array<std::vector<Fr>, DecodeGreedyLinkColumns> columns;
    for (int column = 0; column < DecodeGreedyLinkColumns; ++column)
    {
        columns[column] = materialize_greedy_column(
            witness, static_cast<DecodeGreedyLinkColumn>(column),
            proof.token_domain, proof.vocab_domain);
    }
    for (int column = proof.defer_logit_opening ? 1 : 0; column < DecodeGreedyLinkColumns; ++column)
    {
        G1 *commitment = hyrax::commit(
            columns[column].data(), generators.data(), proof.num_vars,
            std::max(threads, 1));
        proof.column_commitments[column].assign(
            commitment, commitment + row_count);
        delete[] commitment;
    }

    const int token_vars = log2_power_of_two(proof.token_domain);
    std::vector<Fr> lambda_values(proof.token_domain, Fr(0));
    for (int token = 0; token < witness.token_count; ++token)
        lambda_values[token] = fr_from_i128(witness.lambdas[token]);
    G1 *lambda_commitment = hyrax::commit(lambda_values.data(), generators.data(), token_vars,
                                          std::max(threads, 1));
    proof.lambda_commitments.assign(lambda_commitment, lambda_commitment + (1 << (token_vars / 2)));
    delete[] lambda_commitment;
    append_greedy_statement(
        prover_transcript, proof, witness.token_count, witness.vocab_size);
    std::vector<Fr> point(static_cast<size_t>(proof.num_vars));
    for (int bit = 0; bit < proof.num_vars; ++bit)
    {
        point[static_cast<size_t>(bit)] = prover_transcript.challenge_not_one();
    }
    for (int column = 0; column < DecodeGreedyLinkColumns; ++column)
    {
        proof.final_evaluations[column] = mle::evaluate(
            columns[column], point);
        prover_transcript.append_fr(proof.final_evaluations[column]);
    }
    const int vocab_vars = log2_power_of_two(proof.vocab_domain);
    const std::vector<Fr> row_point(point.begin() + vocab_vars, point.end());
    proof.lambda_evaluation = mle::evaluate(lambda_values, row_point);
    prover_transcript.append_fr(proof.lambda_evaluation);
    if (proof.final_evaluations[greedy_column_index(DecodeGreedyLinkColumn::NegativeDelta)] - proof.final_evaluations[greedy_column_index(DecodeGreedyLinkColumn::Lambda)] + proof.final_evaluations[greedy_column_index(DecodeGreedyLinkColumn::Logit)] != Fr(0))
    {
        return result;
    }

    std::array<Fr, DecodeGreedyLinkColumns> weights;
    std::vector<Fr> combined(static_cast<size_t>(domain), Fr(0));
    for (int column = proof.defer_logit_opening ? 1 : 0; column < DecodeGreedyLinkColumns; ++column)
    {
        weights[column] = prover_transcript.nonzero_challenge();
    }
#pragma omp parallel for num_threads(threads) if (domain >= 4096)
    for (int index = 0; index < domain; ++index)
    {
        Fr value = Fr(0);
        for (int column = proof.defer_logit_opening ? 1 : 0; column < DecodeGreedyLinkColumns; ++column)
        {
            value += weights[column] * columns[column][static_cast<size_t>(index)];
        }
        combined[static_cast<size_t>(index)] = value;
    }
    proof.combined_opening = hyrax::open(
        combined.data(), point.data(), proof.num_vars, pk);
    proof.lambda_opening = hyrax::open(lambda_values.data(), row_point.data(), token_vars, pk);
    result.prover_time = std::chrono::duration<double>(
                             Clock::now() - prover_start)
                             .count();

    if (deferred_logit_claim)
    {
        deferred_logit_claim->point = point;
        deferred_logit_claim->evaluation = proof.final_evaluations[0];
    }
    result.success = true;
    return result;
}
