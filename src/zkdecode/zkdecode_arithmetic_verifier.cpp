#include "zkdecode/zkdecode_arithmetic.hpp"
#include "zkdecode/zkdecode_protocol.hpp"
#include "pcs/srs.hpp"

using namespace zkdecode_detail::arithmetic;

namespace {

bool verify_decode_topk_link_impl(
    const DecodeTopKLinkProof &proof,
    int token_count,
    int vocab_size,
    int k,
    const MVKZG_VerifierKey &vk,
    FiatShamirTranscript &transcript, DecodeLogitEvaluation *deferred_logit_claim)
{
    if (proof.defer_logit_opening != (deferred_logit_claim != nullptr))
        return false;
    if (token_count <= 0 ||
        vocab_size <= 0 ||
        k <= 0 ||
        k > vocab_size ||
        proof.token_count != token_count ||
        proof.vocab_size != vocab_size ||
        proof.k != k ||
        proof.token_domain != next_power_of_two(token_count) ||
        proof.vocab_domain != next_power_of_two(vocab_size) ||
        proof.num_vars != log2_power_of_two(proof.token_domain) + log2_power_of_two(proof.vocab_domain) ||
        proof.sumcheck_rounds.size() != static_cast<std::size_t>(proof.num_vars))
    {
        return false;
    }

    const int row_count = 1 << (proof.num_vars / 2);
    for (int column = 0; column < DecodeTopKLinkColumns; ++column)
    {
        const std::size_t expected = proof.defer_logit_opening && column == 0 ? 0 : row_count;
        if (proof.column_commitments[column].size() != expected)
            return false;
    }
    const int token_vars = log2_power_of_two(proof.token_domain);
    const int shift_row_count = 1 << (token_vars / 2);
    if (proof.shift_commitments.size() != std::size_t(shift_row_count) ||
        proof.separator_commitments.size() != std::size_t(shift_row_count))
        return false;
    try
    {
        append_statement(transcript, proof, token_count, vocab_size, k);
        append_tuple_table_context(transcript);
        if (proof.tuple_compression_challenge.isZero() ||
            proof.tuple_compression_challenge != transcript.nonzero_challenge())
        {
            return false;
        }
        const LinkChallenges challenges = derive_link_challenges(
            transcript, proof.num_vars, token_vars);

        Fr claim = challenges.cardinality * Fr(k) * token_prefix_mle(token_count, challenges.token_point);
        std::vector<Fr> point;
        point.reserve(proof.num_vars);
        for (int round = 0; round < proof.num_vars; ++round)
        {
            const DecodeTopKLinkRound &message = proof.sumcheck_rounds[round];
            if (message.evaluations[0] + message.evaluations[1] != claim)
            {
                return false;
            }
            for (int value = 0; value <= DecodeTopKLinkDegree; ++value)
            {
                transcript.append_fr(message.evaluations[static_cast<std::size_t>(value)]);
            }
            const Fr challenge = transcript.challenge_not_one();
            point.push_back(challenge);
            claim = message.evaluate(challenge);
        }

        for (const Fr &evaluation : proof.final_evaluations)
        {
            transcript.append_fr(evaluation);
        }
        transcript.append_fr(proof.shift_evaluation);
        transcript.append_fr(proof.separator_evaluation);

        Fr zero_eval = Fr(1);
        for (int bit = 0; bit < proof.num_vars; ++bit)
        {
            zero_eval *= point[static_cast<std::size_t>(bit)] * challenges.zero_point[static_cast<std::size_t>(bit)] + (Fr(1) - point[static_cast<std::size_t>(bit)]) * (Fr(1) - challenges.zero_point[static_cast<std::size_t>(bit)]);
        }

        const int vocab_vars = log2_power_of_two(proof.vocab_domain);
        std::vector<Fr> index_point(
            point.begin(), point.begin() + vocab_vars);
        std::vector<Fr> row_point(
            point.begin() + vocab_vars, point.end());
        Fr token_eval = Fr(1);
        for (int bit = 0; bit < token_vars; ++bit)
        {
            const Fr &coordinate = point[static_cast<std::size_t>(vocab_vars + bit)];
            const Fr &target = challenges.token_point[static_cast<std::size_t>(bit)];
            token_eval *= coordinate * target + (Fr(1) - coordinate) * (Fr(1) - target);
        }

        const Fr final_relation = local_relation(
            proof.final_evaluations, challenges.relation,
            token_prefix_mle(token_count, row_point) * token_prefix_mle(vocab_size, index_point));
        const Fr selector_eval = proof.final_evaluations[column_index(DecodeTopKLinkColumn::Selector)];
        if (claim != zero_eval * final_relation + challenges.cardinality * token_eval * selector_eval)
        {
            return false;
        }
        const Fr vocab_active_eval = token_prefix_mle(
            vocab_size, index_point);
        if (proof.final_evaluations[column_index(DecodeTopKLinkColumn::Shift)] != proof.shift_evaluation * vocab_active_eval)
        {
            return false;
        }
        if (proof.final_evaluations[column_index(DecodeTopKLinkColumn::Separator)] !=
            proof.separator_evaluation * vocab_active_eval)
            return false;

        std::array<Fr, DecodeTopKLinkColumns> opening_weights;
        Fr combined_evaluation = Fr(0);
        std::vector<G1> combined_commitments(row_count);
        for (int column = proof.defer_logit_opening ? 1 : 0; column < DecodeTopKLinkColumns; ++column)
        {
            opening_weights[column] = transcript.nonzero_challenge();
            combined_evaluation += opening_weights[column] * proof.final_evaluations[column];
        }
        for (int row = 0; row < row_count; ++row)
        {
            std::array<G1, DecodeTopKLinkColumns> bases;
            for (int column = proof.defer_logit_opening ? 1 : 0; column < DecodeTopKLinkColumns; ++column)
            {
                bases[column] = proof.column_commitments[column][row];
            }
            G1::mulVec(combined_commitments[row], bases.data() + (proof.defer_logit_opening ? 1 : 0),
                       opening_weights.data() + (proof.defer_logit_opening ? 1 : 0),
                       DecodeTopKLinkColumns - (proof.defer_logit_opening ? 1 : 0));
        }

        hyrax::OpeningBatchVerifier openings;
        if (!openings.append(
                proof.combined_opening, point.data(), combined_evaluation,
                combined_commitments.data(), proof.num_vars))
        {
            return false;
        }

        std::array<Fr, 2> shift_weights;
        Fr combined_shift_evaluation = Fr(0);
        std::vector<G1> combined_shift_commitments(shift_row_count);
        for (int column = 0; column < 2; ++column)
        {
            shift_weights[column] = transcript.nonzero_challenge();
            combined_shift_evaluation += shift_weights[column] * (column == 0 ? proof.shift_evaluation : proof.separator_evaluation);
        }
        for (int row = 0; row < shift_row_count; ++row)
        {
            std::array<G1, 2> bases = {
                proof.shift_commitments[row],
                proof.separator_commitments[row],
            };
            G1::mulVec(combined_shift_commitments[row], bases.data(),
                       shift_weights.data(), 2);
        }
        if (deferred_logit_claim)
        {
            deferred_logit_claim->point = point;
            deferred_logit_claim->evaluation = proof.final_evaluations[0];
        }
        return openings.append(
                   proof.shift_opening, row_point.data(), combined_shift_evaluation,
                   combined_shift_commitments.data(), token_vars) &&
               openings.verify(vk);
    }
    catch (const std::exception &)
    {
        return false;
    }
}

} // namespace

bool verify_decode_topk_link_proof(
    const DecodeTopKLinkProof &proof,
    int token_count,
    int vocab_size,
    int k,
    const MVKZG_VerifierKey &vk)
{
    FiatShamirTranscript transcript;
    return verify_decode_topk_link_proof(
        proof, token_count, vocab_size, k, vk, transcript);
}

bool verify_decode_topk_link_proof(const DecodeTopKLinkProof &proof,
                                   int token_count, int vocab_size, int k, const MVKZG_VerifierKey &vk, FiatShamirTranscript &transcript)
{
    return verify_decode_topk_link_impl(proof, token_count, vocab_size, k, vk, transcript, nullptr);
}

bool reduce_decode_topk_link_proof(const DecodeTopKLinkProof &proof,
                                   int token_count, int vocab_size, int k, const MVKZG_VerifierKey &vk, FiatShamirTranscript &transcript, DecodeLogitEvaluation &deferred_logit_claim)
{
    deferred_logit_claim = DecodeLogitEvaluation{};
    return verify_decode_topk_link_impl(proof, token_count, vocab_size, k, vk, transcript, &deferred_logit_claim);
}

namespace {

bool verify_decode_greedy_link_impl(
    const DecodeGreedyLinkProof &proof,
    int token_count,
    int vocab_size,
    const MVKZG_VerifierKey &vk,
    FiatShamirTranscript &transcript, DecodeLogitEvaluation *deferred_logit_claim)
{
    if (proof.defer_logit_opening != (deferred_logit_claim != nullptr))
        return false;
    if (token_count <= 0 ||
        vocab_size <= 0 ||
        proof.token_count != token_count ||
        proof.vocab_size != vocab_size ||
        proof.token_domain != next_power_of_two(token_count) ||
        proof.vocab_domain != next_power_of_two(vocab_size) ||
        proof.num_vars != log2_power_of_two(proof.token_domain) + log2_power_of_two(proof.vocab_domain))
    {
        return false;
    }
    const int row_count = 1 << (proof.num_vars / 2);
    for (int column = 0; column < DecodeGreedyLinkColumns; ++column)
    {
        const std::size_t expected = proof.defer_logit_opening && column == 0 ? 0 : row_count;
        if (proof.column_commitments[column].size() != expected)
            return false;
    }
    const int token_vars = log2_power_of_two(proof.token_domain);
    if (proof.lambda_commitments.size() != (std::size_t(1) << (token_vars / 2)))
        return false;
    try
    {
        append_greedy_statement(
            transcript, proof, token_count, vocab_size);
        std::vector<Fr> point(static_cast<size_t>(proof.num_vars));
        for (int bit = 0; bit < proof.num_vars; ++bit)
        {
            point[static_cast<size_t>(bit)] = transcript.challenge_not_one();
        }
        for (const Fr &evaluation : proof.final_evaluations)
        {
            transcript.append_fr(evaluation);
        }
        const Fr &logit = proof.final_evaluations[greedy_column_index(DecodeGreedyLinkColumn::Logit)];
        const Fr &lambda = proof.final_evaluations[greedy_column_index(DecodeGreedyLinkColumn::Lambda)];
        const Fr &negative_delta = proof.final_evaluations[greedy_column_index(DecodeGreedyLinkColumn::NegativeDelta)];
        const int vocab_vars = log2_power_of_two(proof.vocab_domain);
        const std::vector<Fr> vocab_point(point.begin(), point.begin() + vocab_vars);
        const std::vector<Fr> row_point(point.begin() + vocab_vars, point.end());
        transcript.append_fr(proof.lambda_evaluation);
        if (negative_delta - lambda + logit != Fr(0) ||
            lambda != proof.lambda_evaluation * token_prefix_mle(vocab_size, vocab_point))
            return false;

        std::array<Fr, DecodeGreedyLinkColumns> weights;
        Fr combined_evaluation = Fr(0);
        std::vector<G1> combined_commitments(static_cast<size_t>(row_count));
        for (int column = proof.defer_logit_opening ? 1 : 0; column < DecodeGreedyLinkColumns; ++column)
        {
            weights[column] = transcript.nonzero_challenge();
            combined_evaluation += weights[column] * proof.final_evaluations[column];
        }
        for (int row = 0; row < row_count; ++row)
        {
            std::array<G1, DecodeGreedyLinkColumns> bases;
            for (int column = proof.defer_logit_opening ? 1 : 0; column < DecodeGreedyLinkColumns; ++column)
            {
                bases[column] = proof.column_commitments[column][row];
            }
            G1::mulVec(combined_commitments[static_cast<size_t>(row)],
                       bases.data() + (proof.defer_logit_opening ? 1 : 0),
                       weights.data() + (proof.defer_logit_opening ? 1 : 0),
                       DecodeGreedyLinkColumns - (proof.defer_logit_opening ? 1 : 0));
        }
        if (deferred_logit_claim)
        {
            deferred_logit_claim->point = point;
            deferred_logit_claim->evaluation = proof.final_evaluations[0];
        }
        hyrax::OpeningBatchVerifier openings;
        return openings.append(proof.combined_opening, point.data(), combined_evaluation,
                               combined_commitments.data(), proof.num_vars) &&
               openings.append(proof.lambda_opening, row_point.data(), proof.lambda_evaluation,
                               proof.lambda_commitments.data(), token_vars) &&
               openings.verify(vk);
    }
    catch (const std::exception &)
    {
        return false;
    }
}

} // namespace

bool verify_decode_greedy_link_proof(
    const DecodeGreedyLinkProof &proof,
    int token_count,
    int vocab_size,
    const MVKZG_VerifierKey &vk)
{
    FiatShamirTranscript transcript;
    return verify_decode_greedy_link_proof(
        proof, token_count, vocab_size, vk, transcript);
}

bool verify_decode_greedy_link_proof(const DecodeGreedyLinkProof &proof,
                                     int token_count, int vocab_size, const MVKZG_VerifierKey &vk, FiatShamirTranscript &transcript)
{
    return verify_decode_greedy_link_impl(proof, token_count, vocab_size, vk, transcript, nullptr);
}

bool reduce_decode_greedy_link_proof(const DecodeGreedyLinkProof &proof,
                                     int token_count, int vocab_size, const MVKZG_VerifierKey &vk, FiatShamirTranscript &transcript, DecodeLogitEvaluation &deferred_logit_claim)
{
    deferred_logit_claim = DecodeLogitEvaluation{};
    return verify_decode_greedy_link_impl(proof, token_count, vocab_size, vk, transcript, &deferred_logit_claim);
}
