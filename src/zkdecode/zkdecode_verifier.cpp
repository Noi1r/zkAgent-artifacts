#include "zkdecode/zkdecode_verifier.hpp"
#include "zkdecode/zkdecode_protocol.hpp"
#include "zkdecode/zkdecode_input_link.hpp"

#include "common/field_integer.hpp"
#include "lookup/shout.hpp"
#include "pcs/srs.hpp"
#include "zkdecode/zkdecode_exp_digits.hpp"
#include "zkdecode/zkdecode_lookup.hpp"

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <stdexcept>

namespace
{
    using Clock = std::chrono::steady_clock;
    using zkagent_integer::fr_from_i128;

    int padded(int n)
    {
        if (n <= 0 || n > (1 << 29))
            throw std::invalid_argument("invalid decode dimension");
        return zkagent_integer::next_power_of_two(n);
    }

    int variables(int n)
    {
        return zkagent_integer::log2_power_of_two(n);
    }

    void validate(const ZkDecodeParameters &p)
    {
        const std::size_t cells = std::size_t(padded(p.token_count)) * padded(p.vocab_size);
        if (cells > (std::size_t(1) << 29) ||
            p.bound <= 0 || p.bound >= (ll(1) << 125) ||
            (p.lookup_backend != LookupBackend::Shout && p.lookup_backend != LookupBackend::LogUpGkr))
            throw std::invalid_argument("invalid decode parameters");
        if (p.kind == ZkDecodeProofKind::Greedy)
        {
            if (p.k != 0 || p.theta != 0 || p.epsilon_sm != 0)
                throw std::invalid_argument("greedy parameters contain sampling fields");
        }
        else if (p.kind == ZkDecodeProofKind::TopK)
        {
            const DecodeExpDigitsConfig config;
            if (p.k <= 0 || p.k > p.vocab_size || p.k > config.max_active_count ||
                p.bound > config.max_abs_logit || p.theta != config.expected_theta ||
                p.epsilon_sm < 0 || p.epsilon_sm >= (ll(1) << 120))
                throw std::invalid_argument("unsupported TopK scale, bound or parameters");
        }
        else
            throw std::invalid_argument("missing decode proof kind");
    }

}

ZkDecodeVerifierKey prepareZkDecodeVerifierKey(
    const ZkDecodeParameters &p, const MVKZG_ProverKey &pk,
    const MVKZG_VerifierKey &vk, int threads)
{
    validate(p);
    if (threads <= 0)
        throw std::invalid_argument("invalid decode thread count");
    ZkDecodeVerifierKey result;
    result.parameters_ = p;
    result.threads_ = threads;
    std::unique_ptr<LogupGkrProver> logup;
    std::size_t table_generators = 0;
    std::map<std::vector<ll>, std::shared_ptr<const ZkDecodeVerifierKey::PreparedTable>>
        prepared_tables;
    if (p.lookup_backend == LookupBackend::LogUpGkr)
        logup.reset(new LogupGkrProver(pk, vk));
    const auto prepare_table = [&](const std::vector<ll> &table)
        -> std::shared_ptr<const ZkDecodeVerifierKey::PreparedTable>
    {
        const auto found = prepared_tables.find(table);
        if (found != prepared_tables.end())
            return found->second;
        auto prepared = std::make_shared<ZkDecodeVerifierKey::PreparedTable>();
        prepared->values.reserve(table.size());
        for (ll value : table)
            prepared->values.push_back(fr_from_i128(value));
        if (logup)
        {
            const auto needed = std::size_t(1)
                                << ((variables(static_cast<int>(table.size())) + 1) / 2);
            if (needed > table_generators)
            {
                logup->init(needed);
                table_generators = needed;
            }
            prepared->logup = logup->logup_gkr_commit_public_table(
                table.data(), static_cast<int>(table.size()), threads,
                LogupGkrMode::Values);
        }
        const auto immutable =
            std::shared_ptr<const ZkDecodeVerifierKey::PreparedTable>(prepared);
        prepared_tables.emplace(table, immutable);
        return immutable;
    };
    result.lookup_plan_ = zkdecode_detail::lookupPlan(p);
    for (const auto &batch : result.lookup_plan_)
    {
        for (const auto &spec : zkdecode_detail::lookupGroups(batch))
        {
            ZkDecodeVerifierKey::Group group;
            group.spec = spec;
            group.table = prepare_table(zkdecode_detail::lookupTableInputs(spec));
            if (const auto *outputs = zkdecode_detail::lookupTableOutputs(spec))
                group.tuple_table_outputs = prepare_table(*outputs);
            result.groups_.push_back(std::move(group));
        }
    }
    return result;
}

bool ZkDecodeVerifierKey::verifyImpl(
    const ZkDecodeStatement &statement, const ZkDecodeProof &proof,
    const MVKZG_VerifierKey &vk, ZkDecodeVerifyStats *output_stats,
    const zkdecode_detail::ExecutionContext &execution) const
{
    const auto start = output_stats ? Clock::now() : Clock::time_point{};
    const auto finish = [&](bool success, const char *message)
    {
        if (output_stats)
        {
            output_stats->success = success;
            output_stats->input_claim_pending = success && execution.isInputLinked();
            output_stats->message = message;
            output_stats->total_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        }
        return success;
    };
    try
    {
        const auto &p = statement.parameters;
        validate(p);
        if (!(p == parameters_) || !(p == proof.parameters) ||
            statement.output_tokens != proof.output_tokens || statement.sample_values != proof.sample_values ||
            statement.output_tokens.size() != std::size_t(p.token_count) ||
            proof.lookup_proofs.size() != groups_.size())
            return finish(false, "statement, key or proof schema mismatch");
        for (int token : statement.output_tokens)
            if (token < 0 || token >= p.vocab_size)
                return finish(false, "output token outside vocabulary");
        const int n = variables(padded(p.token_count)) + variables(padded(p.vocab_size));
        if (statement.logit_commitments.size() != (execution.isInputLinked() ? 0 : (std::size_t(1) << (n / 2))))
            return finish(false, "logit commitment has wrong dimensions");
        const bool greedy = p.kind == ZkDecodeProofKind::Greedy;
        if (proof.has_greedy_link != greedy || proof.has_greedy_zero != greedy ||
            proof.has_topk_link == greedy || proof.has_selector == greedy ||
            (greedy ? !statement.sample_values.empty()
                    : statement.sample_values.size() != std::size_t(p.token_count)))
            return finish(false, "missing or inconsistent mandatory proof components");
        bool arithmetic_valid;
        if (execution.isInputLinked())
        {
            arithmetic_valid = greedy
                                   ? reduce_decode_greedy_link_proof(proof.greedy_link, p.token_count, p.vocab_size, vk, execution.sharedTranscript(), *execution.deferredLogitClaim())
                                   : reduce_decode_topk_link_proof(proof.topk_link, p.token_count, p.vocab_size, p.k, vk, execution.sharedTranscript(), *execution.deferredLogitClaim());
        }
        else if (greedy)
        {
            arithmetic_valid = proof.greedy_link.column_commitments[0] == statement.logit_commitments &&
                               verify_decode_greedy_link_proof(proof.greedy_link, p.token_count, p.vocab_size, vk);
        }
        else
        {
            arithmetic_valid = decode_topk_column_commitment(proof.topk_link, DecodeTopKLinkColumn::Logit) == statement.logit_commitments &&
                               verify_decode_topk_link_proof(proof.topk_link, p.token_count, p.vocab_size, p.k, vk);
        }
        if (!arithmetic_valid)
            return finish(false, "native arithmetic or source commitment failed");
        const Fr tuple_compression_challenge = greedy
                                                   ? Fr(0)
                                                   : decode_topk_tuple_challenge(
                                                         proof.topk_link, p.token_count, p.vocab_size, p.k);

        hyrax::OpeningBatchVerifier opening_batch;
        std::vector<LookupReadClaim> claims(groups_.size());
        ShoutVerifier shout(vk);
        for (std::size_t i = 0; i < groups_.size(); ++i)
        {
            const auto &group = groups_[i];
            const auto &record = proof.lookup_proofs[i];
            const std::vector<Fr> *table = &group.table->values;
            const LogupGkrTableCommitment *logup_table = &group.table->logup;
            std::vector<Fr> tuple_table;
            LogupGkrTableCommitment tuple_logup_table;
            if (group.usesRandomTupleCompression())
            {
                if (tuple_compression_challenge.isZero() ||
                    group.tuple_table_outputs->values.size() !=
                        group.table->values.size())
                {
                    return finish(false, "invalid random tuple lookup metadata");
                }
                if (p.lookup_backend == LookupBackend::Shout)
                {
                    tuple_table = group.table->values;
                    for (std::size_t entry = 0; entry < tuple_table.size(); ++entry)
                    {
                        tuple_table[entry] += tuple_compression_challenge *
                                              group.tuple_table_outputs->values[entry];
                    }
                    table = &tuple_table;
                }
                else
                {
                    tuple_logup_table = group.table->logup;
                    if (tuple_logup_table.mode != LogupGkrMode::Values ||
                        group.tuple_table_outputs->logup.mode != LogupGkrMode::Values ||
                        tuple_logup_table.padded_n != group.tuple_table_outputs->logup.padded_n ||
                        tuple_logup_table.log_n != group.tuple_table_outputs->logup.log_n ||
                        tuple_logup_table.value_commitments.size() !=
                            group.tuple_table_outputs->logup.value_commitments.size())
                    {
                        return finish(false, "random tuple LogUp table shape mismatch");
                    }
                    for (std::size_t row = 0;
                         row < tuple_logup_table.value_commitments.size(); ++row)
                    {
                        tuple_logup_table.value_commitments[row] +=
                            group.tuple_table_outputs->logup.value_commitments[row] *
                            tuple_compression_challenge;
                    }
                    logup_table = &tuple_logup_table;
                }
            }
            if (record.group_index != group.spec.group_index || record.query_base != 0 ||
                record.actual_query_count != group.spec.actual_count || record.query_count != group.spec.query_count ||
                record.table_size != static_cast<int>(group.table->values.size()) ||
                (!execution.isInputLinked() && record.wire_bytes.empty()))
                return finish(false, "lookup schema mismatch");
            bool valid = false;
            if (p.lookup_backend == LookupBackend::Shout)
            {
                valid = execution.isInputLinked()
                            ? record.shout_messages && !record.logup_messages && shout.verify_in_memory(*record.shout_messages, *table, record.table_size, group.spec.query_count, &claims[i], false, &opening_batch)
                            : shout.verify_serialized(record.wire_bytes, *table,
                                                      record.table_size, group.spec.query_count, nullptr, &claims[i], false, &opening_batch);
            }
            else
            {
                valid = execution.isInputLinked()
                            ? record.logup_messages && !record.shout_messages &&
                                  record.logup_messages->m == group.spec.query_count && logup_gkr_verify_in_memory(*record.logup_messages, *logup_table, vk, &claims[i], &opening_batch)
                            : logup_gkr_verify_serialized(record.wire_bytes,
                                                          group.spec.query_count, *logup_table, vk, &claims[i], &opening_batch);
            }
            if (!valid || claims[i].point.size() != std::size_t(variables(group.spec.query_count)) ||
                claims[i].commitments.size() != (std::size_t(1) << (variables(group.spec.query_count) / 2)))
                return finish(false, "lookup proof failed");
        }
        if (!opening_batch.verify(vk))
            return finish(false, "lookup PCS batch failed");

        if (!zkdecode_detail::verifyLookupBindings(lookup_plan_, proof, claims, threads_))
            return finish(false, "lookup commitments do not match their arithmetic sources");

        if (greedy)
        {
            if (proof.greedy_zero.token_count != p.token_count || proof.greedy_zero.vocab_size != p.vocab_size ||
                proof.greedy_zero.value_commitments != decode_greedy_negative_delta_commitment(proof.greedy_link) ||
                !(execution.isInputLinked() ? verify_decode_batch_point_zero_proof(proof.greedy_zero, statement.output_tokens, vk, execution.sharedTranscript())
                      : verify_decode_batch_point_zero_proof(proof.greedy_zero, statement.output_tokens, vk)))
                return finish(false, "greedy output proof or commitment link failed");
        }
        else
        {
            const auto &probability = decode_topk_probability_commitment(proof.topk_link);
            const bool output_valid = execution.isInputLinked() ? verify_decode_selector_proof(proof.selector, statement.output_tokens, vk, execution.sharedTranscript()) &&
                                                check_decode_sampling_claims(proof.selector, probability, p.vocab_size,
                                                                             statement.sample_values, p.theta, p.epsilon_sm)
                                          : verify_decode_sampling_proof(proof.selector, statement.output_tokens, probability,
                                                                         p.vocab_size, statement.sample_values, p.theta, p.epsilon_sm, vk);
            if (!output_valid)
                return finish(false, "selector, probability commitment, mass or sampling failed");
        }
        return finish(true, execution.isInputLinked() ? "native checks passed; logit opening deferred to GKR" : "complete native decode proof verified");
    }
    catch (const std::exception &)
    {
        return finish(false, "malformed decode proof or parameters");
    }
}

bool verifyZkDecodeProof(const ZkDecodeStatement &statement, const ZkDecodeProof &proof,
                         const ZkDecodeVerifierKey &key, const MVKZG_VerifierKey &vk, ZkDecodeVerifyStats *stats)
{
    return key.verifyImpl(statement, proof, vk, stats, zkdecode_detail::ExecutionContext::standalone());
}
bool reduceZkDecodeProof(const ZkDecodeStatement &statement, const ZkDecodeProof &proof,
                         const ZkDecodeVerifierKey &key, const MVKZG_VerifierKey &vk,
                         FiatShamirTranscript &transcript, DecodeLogitEvaluation &deferred_logit_claim, ZkDecodeVerifyStats *stats)
{
    deferred_logit_claim = DecodeLogitEvaluation{};
    const bool ok = key.verifyImpl(statement, proof, vk, stats, zkdecode_detail::ExecutionContext::linked(transcript, deferred_logit_claim));
    if (!ok)
        deferred_logit_claim = DecodeLogitEvaluation{};
    return ok;
}

std::unique_ptr<InputReductionLink> verifyZkDecodeInputLink(
    const ZkDecodeStatement &statement, const ZkDecodeProof &proof, const ZkDecodeVerifierKey &key,
    const ZkDecodeInputContext &context, const MVKZG_VerifierKey &vk,
    FiatShamirTranscript &transcript, ZkDecodeVerifyStats *stats)
{
    if (stats)
        *stats = ZkDecodeVerifyStats{};
    try
    {
        zkdecode_input_detail::append_context(statement, key, context, transcript);
        DecodeLogitEvaluation evaluation;
        if (!reduceZkDecodeProof(statement, proof, key, vk, transcript, evaluation, stats))
            return nullptr;
        auto result = std::unique_ptr<InputReductionLink>(new InputReductionLink);
        zkdecode_input_detail::bind_logit_claim(*result, context, evaluation, transcript);
        return result;
    }
    catch (const std::exception &error)
    {
        if (stats)
        {
            stats->success = false;
            stats->input_claim_pending = false;
            stats->message = error.what();
        }
        return nullptr;
    }
}


bool checkDecodeShiftConsistency(const std::vector<int> &tokens, std::size_t prompt,
                                 const std::vector<int> &outputs)
{
    if (prompt == 0 || prompt > tokens.size() || outputs.size() != tokens.size() - prompt)
        return false;
    return std::equal(outputs.begin(), outputs.end(), tokens.begin() + prompt);
}

bool checkDecodeShiftConsistency(const std::vector<int> &tokens,
                                 const std::vector<std::size_t> &positions,
                                 const std::vector<int> &outputs)
{
    if (positions.empty() || outputs.size() != positions.size())
        return false;
    for (std::size_t row = 0; row < positions.size(); ++row)
    {
        const auto position = positions[row];
        if (position == 0 || position >= tokens.size() ||
            (row != 0 && position <= positions[row - 1]) || outputs[row] != tokens[position])
            return false;
    }
    return true;
}
