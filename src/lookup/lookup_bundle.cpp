#include "lookup/lookup_bundle.hpp"
#include "common/bit_utils.hpp"

#include "common/field_integer.hpp"

#include "lookup/logup_gkr.hpp"
#include "lookup/shout.hpp"
#include "pcs/srs.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <omp.h>

namespace
{

    struct I128Hash
    {
        std::size_t operator()(ll value) const noexcept
        {
            const unsigned __int128 bits =
                static_cast<unsigned __int128>(value);
            std::uint64_t low = static_cast<std::uint64_t>(bits);
            std::uint64_t high = static_cast<std::uint64_t>(bits >> 64);
            low ^= high + 0x9e3779b97f4a7c15ULL + (low << 6) + (low >> 2);
            low ^= low >> 30;
            low *= 0xbf58476d1ce4e5b9ULL;
            low ^= low >> 27;
            low *= 0x94d049bb133111ebULL;
            low ^= low >> 31;
            return static_cast<std::size_t>(low);
        }
    };

    using zkagent_integer::fr_from_i128;
    using zkagent_integer::log2_power_of_two;

    void append_tuple_compression_context(
        FiatShamirTranscript &transcript, std::size_t group_index, const std::vector<Fr> &table_inputs,
        const std::vector<Fr> &table_outputs)
    {
        transcript.append_u64(group_index);
        transcript.append_fr_vector(table_inputs);
        transcript.append_fr_vector(table_outputs);
    }

    std::vector<Fr> derive_tuple_compression_challenges(
        const std::vector<LookupGroupSpec> &groups,
        const std::vector<std::vector<ll>> &group_queries,
        const std::vector<std::vector<ll>> &group_query_outputs,
        const std::vector<Fr> &supplied_tuple_challenges,
        FiatShamirTranscript *prover_transcript,
        FiatShamirTranscript *verifier_transcript)
    {
        if (groups.empty() || group_queries.size() != groups.size() ||
            (!group_query_outputs.empty() &&
             group_query_outputs.size() != groups.size()) ||
            (!supplied_tuple_challenges.empty() &&
             supplied_tuple_challenges.size() != groups.size()))
        {
            throw std::invalid_argument(
                "lookup tuple materialization has inconsistent group metadata");
        }

        std::vector<Fr> result(groups.size(), Fr(0));

        for (std::size_t group_index = 0; group_index < groups.size(); ++group_index)
        {
            const LookupGroupSpec &group = groups[group_index];
            const std::vector<ll> &queries = group_queries[group_index];
            if (queries.size() != group.paddedCount())
            {
                throw std::invalid_argument(
                    "lookup witness query vector has wrong padded size");
            }

            Fr tuple_challenge(0);
            if (group.usesRandomTupleCompression())
            {
                if (group.tableOutputs().size() != group.tableValues().size() ||
                    group_query_outputs.empty() ||
                    group_query_outputs[group_index].size() != group.paddedCount())
                {
                    throw std::invalid_argument(
                        "random tuple lookup is missing its output values");
                }
                if (!supplied_tuple_challenges.empty())
                {
                    tuple_challenge = supplied_tuple_challenges[group_index];
                    if (tuple_challenge.isZero())
                    {
                        throw std::invalid_argument(
                            "random tuple lookup received a zero compression challenge");
                    }
                }
                else
                {
                    if (prover_transcript == nullptr || verifier_transcript == nullptr)
                    {
                        throw std::invalid_argument(
                            "random tuple lookup requires a commitment-bound transcript");
                    }
                    std::vector<Fr> table_inputs(group.tableValues().size());
                    std::vector<Fr> table_outputs(group.tableOutputs().size());
                    for (std::size_t i = 0; i < group.tableValues().size(); ++i)
                    {
                        table_inputs[i] = fr_from_i128(group.tableValues()[i]);
                        table_outputs[i] = fr_from_i128(group.tableOutputs()[i]);
                    }
                    append_tuple_compression_context(
                        *prover_transcript, group_index, table_inputs, table_outputs);
                    append_tuple_compression_context(
                        *verifier_transcript, group_index, table_inputs, table_outputs);
                    const Fr prover_challenge = prover_transcript->nonzero_challenge();
                    const Fr verifier_challenge = verifier_transcript->nonzero_challenge();
                    if (prover_challenge != verifier_challenge)
                    {
                        throw std::runtime_error(
                            "lookup tuple-compression transcripts diverged");
                    }
                    tuple_challenge = prover_challenge;
                }
                result[group_index] = tuple_challenge;
            }
            else if ((!group_query_outputs.empty() &&
                      !group_query_outputs[group_index].empty()) ||
                     (!supplied_tuple_challenges.empty() &&
                      !supplied_tuple_challenges[group_index].isZero()))
            {
                throw std::invalid_argument(
                    "scalar lookup group has tuple-only witness data");
            }
        }
        return result;
    }

    std::vector<G1> commit_random_tuple_queries(
        const std::vector<ll> &inputs,
        const std::vector<ll> &outputs,
        std::size_t query_base,
        std::size_t actual_count,
        std::size_t padded_count,
        ll padding_input,
        ll padding_output,
        const Fr &tuple_challenge,
        const MVKZG_ProverKey &pk,
        int threads)
    {
        if (tuple_challenge.isZero() || inputs.size() != outputs.size() ||
            query_base > inputs.size() || actual_count > inputs.size() - query_base ||
            actual_count == 0 || actual_count > padded_count)
        {
            throw std::invalid_argument(
                "cannot commit malformed random tuple query columns");
        }
        const int variables = log2_power_of_two(padded_count);
        const std::size_t rows = std::size_t(1) << (variables / 2);
        std::vector<ll> padded_inputs;
        std::vector<ll> padded_outputs;
        const ll *input_values = nullptr;
        const ll *output_values = nullptr;
        if (actual_count == padded_count)
        {
            input_values = inputs.data() + query_base;
            output_values = outputs.data() + query_base;
        }
        else
        {
            padded_inputs.assign(padded_count, padding_input);
            padded_outputs.assign(padded_count, padding_output);
            std::copy_n(
                inputs.begin() + static_cast<std::ptrdiff_t>(query_base),
                actual_count, padded_inputs.begin());
            std::copy_n(
                outputs.begin() + static_cast<std::ptrdiff_t>(query_base),
                actual_count, padded_outputs.begin());
            input_values = padded_inputs.data();
            output_values = padded_outputs.data();
        }

        hyrax::CommitmentPair commitments = hyrax::commit_pair(
            input_values, output_values, const_cast<G1 *>(pk.srs_g1.data()),
            variables, threads);
        std::vector<G1> result(rows);
#pragma omp parallel for num_threads(threads) schedule(static) if (rows >= 64)
        for (long long raw_row = 0;
             raw_row < static_cast<long long>(rows); ++raw_row)
        {
            const std::size_t row = static_cast<std::size_t>(raw_row);
            result[row] = commitments.first[row] + commitments.second[row] * tuple_challenge;
        }
        return result;
    }

    std::size_t next_power_of_two(std::size_t value)
    {
        if (value == 0)
            throw std::invalid_argument("lookup packing requires a nonempty group");
        return zkagent_integer::next_power_of_two(value, "lookup packing overflows size_t");
    }

    std::size_t highest_power_of_two_at_most(std::size_t value)
    {
        if (value == 0)
        {
            throw std::invalid_argument("lookup packing received zero remaining rows");
        }
        std::size_t result = 1;
        while (result <= value / 2)
            result <<= 1;
        return result;
    }

    void checked_add(std::uint64_t &target, std::size_t value,
                     const char *label)
    {
        if (value > std::numeric_limits<std::uint64_t>::max() - target)
        {
            throw std::overflow_error(label);
        }
        target += static_cast<std::uint64_t>(value);
    }

    void checked_add_cells(std::uint64_t &target, std::size_t table_size,
                           std::size_t query_size, const char *label)
    {
        if (table_size != 0 &&
            query_size > std::numeric_limits<std::uint64_t>::max() / table_size)
        {
            throw std::overflow_error(label);
        }
        const std::uint64_t cells = static_cast<std::uint64_t>(
            table_size * query_size);
        if (cells > std::numeric_limits<std::uint64_t>::max() - target)
        {
            throw std::overflow_error(label);
        }
        target += cells;
    }

    std::vector<LookupQueryChunkPlan> plan_group_chunks(
        std::size_t actual_count, std::size_t max_chunks)
    {
        const std::size_t legacy_padded = next_power_of_two(actual_count);
        std::vector<std::size_t> binary_parts;
        std::size_t remaining = actual_count;
        while (remaining != 0)
        {
            const std::size_t part = highest_power_of_two_at_most(remaining);
            binary_parts.push_back(part);
            remaining -= part;
        }

        std::vector<std::size_t> best_capacities{legacy_padded};
        std::size_t best_total = legacy_padded;
        const std::size_t candidate_limit = std::min(max_chunks,
                                                     binary_parts.size());
        for (std::size_t count = 2; count <= candidate_limit; ++count)
        {
            std::vector<std::size_t> capacities;
            capacities.reserve(count);
            std::size_t consumed = 0;
            for (std::size_t chunk = 0; chunk + 1 < count; ++chunk)
            {
                capacities.push_back(binary_parts[chunk]);
                consumed += binary_parts[chunk];
            }
            const std::size_t tail_actual = actual_count - consumed;
            capacities.push_back(next_power_of_two(tail_actual));
            std::size_t total = 0;
            for (std::size_t capacity : capacities)
            {
                if (capacity > std::numeric_limits<std::size_t>::max() - total)
                {
                    throw std::overflow_error("lookup chunk capacity sum overflows");
                }
                total += capacity;
            }
            if (total < best_total)
            {
                best_total = total;
                best_capacities = std::move(capacities);
            }
        }

        std::vector<LookupQueryChunkPlan> result;
        result.reserve(best_capacities.size());
        std::size_t query_base = 0;
        for (std::size_t chunk = 0; chunk < best_capacities.size(); ++chunk)
        {
            const std::size_t capacity = best_capacities[chunk];
            const std::size_t actual = chunk + 1 == best_capacities.size()
                                           ? actual_count - query_base
                                           : capacity;
            if (actual == 0 || actual > capacity)
            {
                throw std::logic_error("lookup chunk planner produced an invalid chunk");
            }
            result.push_back({query_base, actual, capacity});
            query_base += actual;
        }
        if (query_base != actual_count)
        {
            throw std::logic_error("lookup chunk planner did not cover every query");
        }
        return result;
    }

}

namespace
{

    LookupPackingPlan plan_lookup_bundle_packing(
        const std::vector<LookupGroupSpec> &groups,
        const std::vector<std::vector<ll>> &group_queries,
        std::size_t max_chunks_per_group)
    {
        if (max_chunks_per_group == 0)
        {
            throw std::invalid_argument(
                "lookup packing requires at least one chunk per group");
        }
        if (group_queries.size() != groups.size() || groups.empty())
        {
            throw std::invalid_argument("lookup witness/layout group count mismatch");
        }

        LookupPackingPlan result;
        result.max_chunks_per_group = max_chunks_per_group;
        result.groups.reserve(groups.size());
        for (std::size_t group_index = 0;
             group_index < groups.size(); ++group_index)
        {
            const LookupGroupSpec &group = groups[group_index];
            if (group_queries[group_index].size() != group.paddedCount())
            {
                throw std::invalid_argument(
                    "lookup witness query vector has the wrong padded size");
            }
            LookupGroupPackingPlan group_plan;
            group_plan.group_index = group_index;
            group_plan.table_size = group.tableValues().size();
            group_plan.actual_count = group.actualCount();
            group_plan.legacy_padded_count = group.paddedCount();
            group_plan.chunks = plan_group_chunks(
                group.actualCount(), max_chunks_per_group);
            result.proof_chunks += group_plan.chunks.size();
            checked_add(result.actual_query_slots, group.actualCount(),
                        "lookup actual query count overflows");
            checked_add(result.legacy_padded_query_slots, group.paddedCount(),
                        "lookup legacy padded query count overflows");
            checked_add_cells(result.legacy_weighted_cells,
                              group_plan.table_size, group.paddedCount(),
                              "lookup legacy weighted work overflows");
            for (const LookupQueryChunkPlan &chunk : group_plan.chunks)
            {
                checked_add(result.packed_query_slots, chunk.padded_count,
                            "lookup packed query count overflows");
                checked_add_cells(result.packed_weighted_cells,
                                  group_plan.table_size, chunk.padded_count,
                                  "lookup packed weighted work overflows");
            }
            result.groups.push_back(std::move(group_plan));
        }
        return result;
    }

    std::size_t lookup_bundle_required_generators(
        const std::vector<LookupGroupSpec> &groups,
        const std::vector<std::vector<ll>> &group_queries,
        LookupBackend backend,
        std::size_t max_chunks_per_group)
    {
        if (backend != LookupBackend::Shout &&
            backend != LookupBackend::LogUpGkr)
        {
            throw std::invalid_argument("unknown lookup backend");
        }
        const LookupPackingPlan packing = plan_lookup_bundle_packing(
            groups, group_queries, max_chunks_per_group);
        std::size_t required = 1;
        for (const LookupGroupPackingPlan &group_plan : packing.groups)
        {
            const std::size_t K = group_plan.table_size;
            const int log_k = log2_power_of_two(K);
            for (const LookupQueryChunkPlan &chunk : group_plan.chunks)
            {
                const int log_t = log2_power_of_two(chunk.padded_count);
                if (backend == LookupBackend::Shout)
                {
                    const int log_kt = log_k + log_t;
                    const std::size_t columns =
                        std::size_t(1) << (log_kt - log_kt / 2);
                    required = std::max(required, columns);
                }
                else
                {
                    const std::size_t query_columns =
                        std::size_t(1) << (log_t - log_t / 2);
                    const std::size_t table_columns =
                        std::size_t(1) << (log_k - log_k / 2);
                    required = std::max(required,
                                        std::max(query_columns, table_columns));
                }
            }
        }
        return required;
    }

}

LookupPackingPlan planLookupBundlePacking(
    const LookupWitness &witness, std::size_t max_chunks_per_group)
{
    return plan_lookup_bundle_packing(
        witness.layout.groups(), witness.group_queries,
        max_chunks_per_group);
}

LookupPackingPlan planLookupBundlePacking(
    const LookupMembershipWitness &witness,
    std::size_t max_chunks_per_group)
{
    return plan_lookup_bundle_packing(
        witness.groups, witness.group_queries, max_chunks_per_group);
}

std::size_t lookupBundleRequiredGenerators(
    const LookupWitness &witness, LookupBackend backend,
    std::size_t max_chunks_per_group)
{
    return lookup_bundle_required_generators(
        witness.layout.groups(), witness.group_queries, backend,
        max_chunks_per_group);
}

std::size_t lookupBundleRequiredGenerators(
    const LookupMembershipWitness &witness, LookupBackend backend,
    std::size_t max_chunks_per_group)
{
    return lookup_bundle_required_generators(
        witness.groups, witness.group_queries, backend,
        max_chunks_per_group);
}

namespace
{

    struct LookupExecution
    {
        const MVKZG_ProverKey &pk;
        const MVKZG_VerifierKey &vk;
        int threads;
        LookupBackend backend;
        bool defer_query_openings;
        LookupOpeningVerification opening_verification;
        FiatShamirTranscript *prover_transcript;
        FiatShamirTranscript *verifier_transcript;
        G1WireEncoding g1_encoding;
        bool verify_proofs;
        hyrax::OpeningBatchVerifier opening_batch;
        std::unique_ptr<LogupGkrProver> logup_prover;
        std::vector<G1> shout_generators;
    };

    struct LookupGroupView
    {
        const LookupGroupSpec &group;
        const std::vector<ll> &queries;
        const std::vector<ll> *outputs;
        Fr tuple_challenge;
        const std::vector<G1> *precommitted_query;
    };

    struct PreparedLookupTable
    {
        std::unordered_map<ll, int, I128Hash> address_by_input;
        std::vector<Fr> values;
        int padding_address = 0;
        int log_n = 0;
        LogupGkrTableCommitment commitment;
    };

    struct PreparedLookupQuery
    {
        std::vector<int> read_addresses;
        std::vector<Fr> shout_values;
        LogupGkrPreparedValueWitness logup_values;
        std::vector<G1> generated_commitment;
        const std::vector<G1> *precommitted_query = nullptr;
        double commitment_seconds = 0.0;

        const std::vector<G1> *commitments() const
        {
            return generated_commitment.empty() ? precommitted_query : &generated_commitment;
        }
    };

    PreparedLookupTable prepare_lookup_table(
        const LookupGroupView &view, LookupExecution &execution,
        LookupBundleProof &result)
    {
        PreparedLookupTable prepared;
        const auto &group = view.group;
        const Fr &tuple_challenge = view.tuple_challenge;
        const auto backend = execution.backend;
        const int threads = execution.threads;
        auto &logup_prover = execution.logup_prover;
        const int K = static_cast<int>(group.tableValues().size());
        const auto address_map_start = std::chrono::steady_clock::now();
        auto &address_by_input = prepared.address_by_input;
        address_by_input.reserve(group.tableValues().size() * 2);
        auto &table = prepared.values;
        table.resize(static_cast<std::size_t>(K));
        for (int address = 0; address < K; ++address)
        {
            const ll input = group.tableValues()[static_cast<std::size_t>(address)];
            const auto inserted = address_by_input.emplace(input, address);
            if (!inserted.second && group.usesRandomTupleCompression() &&
                group.tableOutputs()[static_cast<std::size_t>(inserted.first->second)] != group.tableOutputs()[static_cast<std::size_t>(address)])
            {
                throw std::invalid_argument(
                    "random tuple lookup table maps one input to multiple outputs");
            }
            table[static_cast<std::size_t>(address)] = fr_from_i128(input);
            if (group.usesRandomTupleCompression())
            {
                table[static_cast<std::size_t>(address)] += tuple_challenge * fr_from_i128(group.tableOutputs()[static_cast<std::size_t>(address)]);
            }
        }
        const auto padding = address_by_input.find(group.paddingValue());
        if (padding == address_by_input.end())
        {
            throw std::logic_error("lookup padding input is absent from its table");
        }
        prepared.padding_address = padding->second;
        result.table_preprocess_time += std::chrono::duration<double>(
                                            std::chrono::steady_clock::now() - address_map_start)
                                            .count();

        auto &table_commitment = prepared.commitment;
        if (backend == LookupBackend::LogUpGkr)
        {
            prepared.log_n = log2_power_of_two(
                static_cast<std::size_t>(K));
            const auto table_start = std::chrono::steady_clock::now();
            table_commitment =
                logup_prover->logup_gkr_commit_public_table(
                    group.tableValues().data(), K, threads,
                    LogupGkrMode::Values);
            if (group.usesRandomTupleCompression())
            {
                const LogupGkrTableCommitment output_commitment =
                    logup_prover->logup_gkr_commit_public_table(
                        group.tableOutputs().data(), K, threads,
                        LogupGkrMode::Values);
                if (table_commitment.mode != LogupGkrMode::Values ||
                    output_commitment.mode != LogupGkrMode::Values ||
                    table_commitment.padded_n != output_commitment.padded_n ||
                    table_commitment.log_n != output_commitment.log_n ||
                    table_commitment.value_commitments.size() != output_commitment.value_commitments.size())
                {
                    throw std::logic_error(
                        "random tuple LogUp table commitments have incompatible shapes");
                }
                for (std::size_t row = 0;
                     row < table_commitment.value_commitments.size(); ++row)
                {
                    table_commitment.value_commitments[row] +=
                        output_commitment.value_commitments[row] * tuple_challenge;
                }
            }
            result.table_preprocess_time += std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() - table_start)
                                                .count();
        }
        return prepared;
    }

    PreparedLookupQuery prepare_lookup_query(
        const LookupGroupView &view, const LookupQueryChunkPlan &chunk,
        const PreparedLookupTable &prepared_table, const LookupExecution &execution,
        LookupBundleProof &result)
    {
        PreparedLookupQuery prepared;
        prepared.precommitted_query = view.precommitted_query;
        const auto &group = view.group;
        const auto &source_queries = view.queries;
        const auto *source_outputs = view.outputs;
        const auto *precommitted_query = view.precommitted_query;
        const Fr &tuple_challenge = view.tuple_challenge;
        const auto &table = prepared_table.values;
        const auto &address_by_input = prepared_table.address_by_input;
        const int padding_address = prepared_table.padding_address;
        const int K = static_cast<int>(table.size());
        const int threads = execution.threads;
        const auto backend = execution.backend;
        const auto &pk = execution.pk;
        if (chunk.padded_count >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("lookup proof chunk exceeds int domain");
        }
        const auto witness_start = std::chrono::steady_clock::now();
        const int T = static_cast<int>(chunk.padded_count);
        auto &read_addresses = prepared.read_addresses;
        if (backend == LookupBackend::Shout)
            read_addresses.assign(chunk.padded_count, padding_address);
        auto &prepared_query_values = prepared.shout_values;
        prepared_query_values.assign(
            chunk.padded_count, table[static_cast<std::size_t>(padding_address)]);
        const int preparation_threads = std::max(
            1, std::min(threads, static_cast<int>(chunk.actual_count)));
        std::atomic<bool> membership_valid(true);
        std::vector<std::vector<ll>> local_multiplicities;
        if (backend == LookupBackend::LogUpGkr)
        {
            local_multiplicities.assign(
                static_cast<std::size_t>(preparation_threads),
                std::vector<ll>(static_cast<std::size_t>(K), 0));
        }
#pragma omp parallel num_threads(preparation_threads)
        {
            const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
            for (long long raw_i = 0;
                 raw_i < static_cast<long long>(chunk.actual_count);
                 ++raw_i)
            {
                const std::size_t i = static_cast<std::size_t>(raw_i);
                const ll input = source_queries[chunk.query_base + i];
                const auto found = address_by_input.find(input);
                if (found == address_by_input.end())
                {
                    membership_valid.store(false);
                    continue;
                }
                const int address = found->second;
                if (backend == LookupBackend::Shout)
                    read_addresses[i] = address;
                if (source_outputs != nullptr)
                {
                    const ll output = (*source_outputs)[chunk.query_base + i];
                    if (output != group.tableOutputs()[static_cast<std::size_t>(address)])
                    {
                        membership_valid.store(false);
                        continue;
                    }
                }
                prepared_query_values[i] = table[static_cast<std::size_t>(address)];
                if (backend == LookupBackend::LogUpGkr)
                {
                    local_multiplicities[static_cast<std::size_t>(tid)]
                                        [static_cast<std::size_t>(address)]++;
                }
            }
        }
        if (!membership_valid.load())
        {
            throw std::out_of_range(
                "lookup tuple is absent from its declared public table");
        }
        auto &prepared_logup_query = prepared.logup_values;
        if (backend == LookupBackend::LogUpGkr)
        {
            prepared_logup_query.padded_m = T;
            prepared_logup_query.log_m = log2_power_of_two(
                chunk.padded_count);
            prepared_logup_query.values =
                std::move(prepared_query_values);
            prepared_logup_query.multiplicities.assign(
                static_cast<std::size_t>(K), 0);
            for (const std::vector<ll> &local : local_multiplicities)
            {
                for (int address = 0; address < K; ++address)
                {
                    prepared_logup_query.multiplicities[static_cast<std::size_t>(address)] +=
                        local[static_cast<std::size_t>(address)];
                }
            }
            prepared_logup_query.multiplicities[static_cast<std::size_t>(padding_address)] +=
                static_cast<ll>(chunk.padded_count - chunk.actual_count);
        }
        result.witness_preparation_time += std::chrono::duration<double>(
                                               std::chrono::steady_clock::now() - witness_start)
                                               .count();

        auto &generated_query_commitment = prepared.generated_commitment;
        double &generated_query_commitment_time = prepared.commitment_seconds;
        if (precommitted_query == nullptr &&
            source_outputs != nullptr)
        {
            const auto commitment_start =
                std::chrono::steady_clock::now();
            generated_query_commitment = commit_random_tuple_queries(
                source_queries, *source_outputs,
                chunk.query_base, chunk.actual_count, chunk.padded_count,
                group.paddingValue(),
                group.tableOutputs()[static_cast<std::size_t>(
                    padding_address)],
                tuple_challenge, pk, preparation_threads);
            generated_query_commitment_time =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - commitment_start)
                    .count();
            result.query_commitment_time +=
                generated_query_commitment_time;
            result.prover_time += generated_query_commitment_time;
        }
        return prepared;
    }

    void record_prover_timing(
        LookupGroupProofRecord &record, LookupBundleProof &result,
        LookupGroupProverTiming timing, double generated_query_commitment_time)
    {
        result.query_commitment_time += timing.query_commit_seconds;
        result.auxiliary_commitment_time += timing.auxiliary_commit_seconds;
        result.prover_time += timing.total_seconds;
        result.relation_preparation_time += timing.relation_seconds;
        result.sumcheck_time += timing.sumcheck_seconds;
        result.opening_prover_time += timing.opening_seconds;
        timing.total_seconds += generated_query_commitment_time;
        timing.query_commit_seconds += generated_query_commitment_time;
        record.timing = timing;
    }

    void execute_shout_chunk(
        const PreparedLookupTable &prepared_table, const PreparedLookupQuery &query,
        LookupExecution &execution, LookupGroupProofRecord &record,
        LookupBundleProof &result)
    {
        const int K = record.table_size;
        const int T = record.query_count;
        const auto &pk = execution.pk;
        const auto &vk = execution.vk;
        const int threads = execution.threads;
        constexpr bool retain_wire_bytes = true;
        auto &opening_batch = execution.opening_batch;
        const auto *effective_query_commitment = query.commitments();
        const double generated_query_commitment_time = query.commitment_seconds;
        const auto &table = prepared_table.values;
        const auto &read_addresses = query.read_addresses;
        const auto &prepared_query_values = query.shout_values;
        auto &shout_generators = execution.shout_generators;
        ShoutProver prover(
            shout_generators.data(), pk);
        ShoutVerifier verifier(vk);
        ShoutProveTiming timing;
        const auto prove_start = std::chrono::steady_clock::now();
        ShoutProof proof = prover.prove_prepared_values(
            table, read_addresses, prepared_query_values,
            K, T, threads, &timing, execution.g1_encoding,
            execution.defer_query_openings, retain_wire_bytes, execution.prover_transcript,
            effective_query_commitment);
        const double total_prove = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - prove_start)
                                       .count();
        const double query_commitment_time =
            (timing.commit_read_values_ms + timing.finalize_read_values_ms) / 1000.0;
        const double auxiliary_commitment_time =
            (timing.commit_ra_ms + timing.finalize_ra_ms) / 1000.0;
        record_prover_timing(record, result, {
            total_prove, query_commitment_time, auxiliary_commitment_time,
            (timing.eq_ms + timing.read_poly_ms) / 1000.0,
            (timing.core_sumcheck_ms + timing.booleanity_sumcheck_ms) / 1000.0,
            (timing.open_read_value_ms + timing.open_ra_ms) / 1000.0},
            generated_query_commitment_time);
        if (proof.wire_bytes.empty())
        {
            delete[] proof.RA_comm;
            throw std::runtime_error(
                "SHOUT failed to construct a lookup proof");
        }
        if (execution.verify_proofs)
        {
            const auto verify_start = std::chrono::steady_clock::now();
            const bool batch_openings = execution.opening_verification ==
                                        LookupOpeningVerification::BatchMvKzg;
            const bool verified = verifier.verify_in_memory(
                proof, table, K, T, &record.read_claim,
                execution.defer_query_openings,
                batch_openings ? &opening_batch : nullptr,
                execution.verifier_transcript);
            result.verifier_time += std::chrono::duration<double>(
                                        std::chrono::steady_clock::now() - verify_start)
                                        .count();
            result.verified = result.verified && verified;
            if (!batch_openings)
            {
                const std::size_t claims = execution.defer_query_openings ? 2 : 3;
                result.mvkzg_opening_claims += claims;
                result.mvkzg_verification_batches += claims;
            }
        }
        else
        {
            record.read_claim.commitments = proof.read_value_commitments;
        }
        const auto materialize_start = std::chrono::steady_clock::now();
        record.wire_bytes = std::move(proof.wire_bytes);
        record.proof_size_bytes = shout_proof_size_bytes(proof);
        result.proof_materialization_time +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - materialize_start)
                .count();
        if (!execution.verify_proofs)
        {
            record.shout_messages = std::shared_ptr<const ShoutProof>(
                new ShoutProof(std::move(proof)), [](const ShoutProof *messages)
                {
                delete[] messages->RA_comm;
                delete messages; });
            proof.RA_comm = nullptr;
        }
        delete[] proof.RA_comm;
        proof.RA_comm = nullptr;
    }

    void execute_logup_chunk(
        const PreparedLookupTable &prepared_table, const PreparedLookupQuery &query,
        LookupExecution &execution, LookupGroupProofRecord &record,
        LookupBundleProof &result)
    {
        const int K = record.table_size;
        const int threads = execution.threads;
        constexpr bool retain_wire_bytes = true;
        auto &opening_batch = execution.opening_batch;
        const auto *effective_query_commitment = query.commitments();
        const double generated_query_commitment_time = query.commitment_seconds;
        auto &logup_prover = execution.logup_prover;
        const auto &prepared_logup_query = query.logup_values;
        const LogupGkrPreparedValueTable prepared_logup_table{
            K, prepared_table.log_n, &prepared_table.values};
        const auto &table_commitment = prepared_table.commitment;
        LogupGkrProveTiming timing;
        const auto prove_start = std::chrono::steady_clock::now();
        LogupGkrProof proof =
            logup_prover->logup_gkr_prove_prepared_values(
                prepared_logup_query, prepared_logup_table, threads,
                execution.g1_encoding, &timing, &table_commitment,
                execution.defer_query_openings, retain_wire_bytes, execution.prover_transcript,
                effective_query_commitment);
        const double total_prove = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - prove_start)
                                       .count();
        const double query_commitment_time =
            timing.query_commitment_ms / 1000.0;
        const double auxiliary_commitment_time =
            timing.multiplicity_commitment_ms / 1000.0;
        const double table_commitment_time =
            timing.table_commitment_ms / 1000.0;
        result.table_preprocess_time += table_commitment_time;
        record_prover_timing(record, result, {
            std::max(0.0, total_prove - table_commitment_time),
            query_commitment_time, auxiliary_commitment_time,
            timing.tree_build_ms / 1000.0,
            timing.sumcheck_ms / 1000.0, timing.opening_ms / 1000.0},
            generated_query_commitment_time);
        if (execution.verify_proofs)
        {
            const auto verify_start = std::chrono::steady_clock::now();
            const bool batch_openings = execution.opening_verification ==
                                        LookupOpeningVerification::BatchMvKzg;
            const bool verified = logup_prover->logup_gkr_verify_in_memory_with_table(
                proof, table_commitment,
                &record.read_claim, execution.defer_query_openings,
                execution.verifier_transcript,
                batch_openings ? &opening_batch : nullptr);
            result.verifier_time += std::chrono::duration<double>(
                                        std::chrono::steady_clock::now() - verify_start)
                                        .count();
            result.verified = result.verified && verified;
            if (!batch_openings)
            {
                const std::size_t claims = execution.defer_query_openings ? 2 : 3;
                result.mvkzg_opening_claims += claims;
                result.mvkzg_verification_batches += claims;
            }
        }
        else
        {
            record.read_claim.commitments = proof.f_combined_comm;
        }
        const auto materialize_start = std::chrono::steady_clock::now();
        record.wire_bytes = std::move(proof.wire_bytes);
        record.proof_size_bytes = logup_gkr_proof_size_bytes(proof);
        result.proof_materialization_time +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - materialize_start)
                .count();
        if (!execution.verify_proofs)
            record.logup_messages = std::make_shared<LogupGkrProof>(std::move(proof));
    }

    void finalize_lookup_bundle(
        const LookupExecution &execution, LookupBundleProof &result)
    {
        const auto &vk = execution.vk;
        const auto &opening_batch = execution.opening_batch;
        if (execution.verify_proofs && execution.opening_verification == LookupOpeningVerification::BatchMvKzg)
        {
            result.mvkzg_opening_claims = opening_batch.size();
            result.mvkzg_verification_batches =
                opening_batch.size() == 0 ? 0 : 1;
            std::vector<Fr> batch_weights;
            if (execution.prover_transcript != nullptr)
            {
                std::vector<Fr> verifier_weights;
                opening_batch.derive_weights(*execution.prover_transcript, batch_weights);
                opening_batch.derive_weights(*execution.verifier_transcript, verifier_weights);
                if (batch_weights != verifier_weights)
                {
                    result.verified = false;
                }
            }
            const bool openings_verified = batch_weights.empty()
                                               ? opening_batch.verify(
                                                     vk, &result.mvkzg_batch_verify_time)
                                               : opening_batch.verify_with_weights(
                                                     vk, batch_weights, &result.mvkzg_batch_verify_time);
            result.verifier_time += result.mvkzg_batch_verify_time;
            result.verified = result.verified && openings_verified;
        }
        result.unattributed_prover_time = std::max(
            0.0, result.prover_time - result.query_commitment_time -
                     result.auxiliary_commitment_time -
                     result.relation_preparation_time -
                     result.sumcheck_time - result.opening_prover_time);
    }

    LookupBundleProof prove_and_verify_lookup_bundle(
        const std::vector<LookupGroupSpec> &groups,
        const std::vector<std::vector<ll>> &group_queries,
        const std::vector<std::vector<ll>> &group_query_outputs,
        const std::vector<Fr> &supplied_tuple_challenges,
        const std::vector<std::vector<G1>> &group_query_commitments,
        const MVKZG_ProverKey &pk,
        const MVKZG_VerifierKey &vk,
        int threads, LookupBackend backend,
        bool defer_query_openings,
        std::size_t max_chunks_per_group,
        LookupOpeningVerification opening_verification,
        FiatShamirTranscript *prover_transcript,
        FiatShamirTranscript *verifier_transcript,
        G1WireEncoding g1_encoding, bool verify_proofs = true)
    {
        LookupBundleProof result;
        result.backend = backend;
        result.opening_verification = opening_verification;
        result.query_openings_deferred = defer_query_openings;
        result.packing = plan_lookup_bundle_packing(
            groups, group_queries, max_chunks_per_group);
        if (backend != LookupBackend::Shout &&
            backend != LookupBackend::LogUpGkr)
        {
            throw std::invalid_argument("unknown lookup backend");
        }
        if ((prover_transcript == nullptr) !=
            (verifier_transcript == nullptr))
        {
            throw std::invalid_argument(
                "lookup outer transcripts must be supplied as a prover/verifier pair");
        }
        if (!group_query_commitments.empty() &&
            group_query_commitments.size() != groups.size())
        {
            throw std::invalid_argument(
                "lookup precommitted query metadata has the wrong group count");
        }
        if (threads <= 0)
            throw std::invalid_argument("lookup proof threads must be positive");
        const auto materialization_start = std::chrono::steady_clock::now();
        result.tuple_compression_challenges =
            derive_tuple_compression_challenges(
                groups, group_queries, group_query_outputs,
                supplied_tuple_challenges, prover_transcript, verifier_transcript);
        result.witness_preparation_time += std::chrono::duration<double>(
                                               std::chrono::steady_clock::now() - materialization_start)
                                               .count();
        const std::size_t required_generators =
            lookup_bundle_required_generators(
                groups, group_queries, backend, max_chunks_per_group);
        if (pk.srs_g1.size() < required_generators)
        {
            throw std::invalid_argument("MVKZG SRS is too small for lookup bundle");
        }

        LookupExecution execution{
            pk, vk, threads, backend, defer_query_openings, opening_verification,
            prover_transcript, verifier_transcript, g1_encoding, verify_proofs};
        if (backend == LookupBackend::LogUpGkr)
        {
            execution.logup_prover.reset(new LogupGkrProver(pk, vk));
            execution.logup_prover->init(required_generators);
        }
        else
        {
            // MSM may normalize bases in place; own one mutable SRS copy per bundle.
            execution.shout_generators.assign(
                pk.srs_g1.begin(), pk.srs_g1.begin() + required_generators);
        }

        if (!verify_proofs && (prover_transcript || verifier_transcript ||
                               defer_query_openings))
            throw std::invalid_argument("prover-only lookup requires standalone FS and complete openings");
        result.verified = verify_proofs;
        result.groups.reserve(result.packing.proof_chunks);
        for (const LookupGroupPackingPlan &group_plan : result.packing.groups)
        {
            const std::size_t group_index = group_plan.group_index;
            const LookupGroupSpec &group = groups[group_index];
            const std::vector<ll> &source_queries = group_queries[group_index];
            const std::vector<ll> *source_outputs =
                group.usesRandomTupleCompression()
                    ? &group_query_outputs[group_index]
                    : nullptr;
            const Fr tuple_challenge =
                result.tuple_compression_challenges[group_index];
            const std::vector<G1> *precommitted_query = nullptr;
            if (!group_query_commitments.empty() &&
                !group_query_commitments[group_index].empty())
            {
                if (group_plan.chunks.size() != 1)
                {
                    throw std::invalid_argument(
                        "precommitted lookup query cannot be split across proof chunks");
                }
                precommitted_query = &group_query_commitments[group_index];
            }
            const LookupGroupView view{
                group, source_queries, source_outputs, tuple_challenge, precommitted_query};
            const PreparedLookupTable table = prepare_lookup_table(view, execution, result);
            for (const LookupQueryChunkPlan &chunk : group_plan.chunks)
            {
                const PreparedLookupQuery query =
                    prepare_lookup_query(view, chunk, table, execution, result);
                LookupGroupProofRecord record;
                record.group_index = group_index;
                record.query_base = chunk.query_base;
                record.actual_query_count = chunk.actual_count;
                record.table_size = static_cast<int>(table.values.size());
                record.query_count = static_cast<int>(chunk.padded_count);
                if (backend == LookupBackend::Shout)
                    execute_shout_chunk(table, query, execution, record, result);
                else
                    execute_logup_chunk(table, query, execution, record, result);

                const bool claim_complete = !record.read_claim.commitments.empty() &&
                    record.read_claim.point.size() ==
                        static_cast<std::size_t>(log2_power_of_two(chunk.padded_count));
                result.verified = result.verified && claim_complete;
                result.proof_size_bytes += record.proof_size_bytes;
                result.groups.push_back(std::move(record));
            }
        }
        finalize_lookup_bundle(execution, result);
        return result;
    }

}

LookupBundleProof proveAndVerifyLookupBundle(
    const LookupWitness &witness,
    const MVKZG_ProverKey &pk,
    const MVKZG_VerifierKey &vk,
    int threads, LookupBackend backend,
    bool defer_query_openings,
    std::size_t max_chunks_per_group,
    LookupOpeningVerification opening_verification,
    FiatShamirTranscript *prover_transcript,
    FiatShamirTranscript *verifier_transcript,
    G1WireEncoding g1_encoding)
{
    return prove_and_verify_lookup_bundle(
        witness.layout.groups(), witness.group_queries,
        witness.group_query_outputs, {}, {}, pk, vk, threads,
        backend, defer_query_openings, max_chunks_per_group,
        opening_verification, prover_transcript,
        verifier_transcript, g1_encoding);
}

LookupBundleProof proveAndVerifyLookupBundle(
    const LookupMembershipWitness &witness,
    const MVKZG_ProverKey &pk,
    const MVKZG_VerifierKey &vk,
    int threads, LookupBackend backend,
    std::size_t max_chunks_per_group,
    LookupOpeningVerification opening_verification,
    FiatShamirTranscript *prover_transcript,
    FiatShamirTranscript *verifier_transcript,
    G1WireEncoding g1_encoding)
{
    return prove_and_verify_lookup_bundle(
        witness.groups, witness.group_queries, witness.group_query_outputs,
        witness.tuple_compression_challenges, witness.group_query_commitments,
        pk, vk, threads, backend,
        false, max_chunks_per_group,
        opening_verification, prover_transcript,
        verifier_transcript, g1_encoding);
}

double LookupBundleProof::proverTimeWithPrecommittedQueries() const
{
    return std::max(0.0, prover_time - query_commitment_time);
}

LookupBundleProof proveLookupBundle(
    const LookupMembershipWitness &witness, const MVKZG_ProverKey &pk,
    const MVKZG_VerifierKey &vk, int threads, LookupBackend backend,
    std::size_t max_chunks_per_group, G1WireEncoding encoding)
{
    return prove_and_verify_lookup_bundle(
        witness.groups, witness.group_queries, witness.group_query_outputs,
        witness.tuple_compression_challenges, witness.group_query_commitments,
        pk, vk, threads, backend, false, max_chunks_per_group,
        LookupOpeningVerification::BatchMvKzg,
        nullptr, nullptr, encoding, false);
}
