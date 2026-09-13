#include "lookup/lookup_input_link.hpp"
#include "common/polynomial.h"
#include "common/bit_utils.hpp"
#include "common/mle.hpp"

#include "common/field_integer.hpp"
#include "common/fiat_shamir.hpp"
#include "lookup/lookup_common.hpp"
#include "pcs/hyrax.hpp"
#include "pcs/srs.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

namespace
{

    using Clock = std::chrono::steady_clock;

    struct QueryCoefficientBlock
    {
        std::size_t query_base = 0;
        std::size_t relation_base = 0;
        std::size_t row_count = 0;
        std::vector<Fr> relation_point;
        Fr scale = Fr(0);
    };

    struct LinkRelationPlan
    {
        std::size_t input_domain_size = 0;
        std::vector<InputAffineRelationClaim> affine_input_relations;
        std::vector<InputReconstructionRelationClaim>
            reconstruction_input_relations;
        std::vector<std::vector<QueryCoefficientBlock>> query_blocks;
        Fr constant = Fr(0);
        std::size_t relation_claims = 0;
    };

    constexpr std::size_t MinParallelItemsPerThread = 256;

    int normalized_threads(int requested)
    {
        return std::max(1, requested);
    }

    bool use_parallel(std::size_t items, int threads)
    {
        return threads > 1 &&
               items >= static_cast<std::size_t>(threads) * MinParallelItemsPerThread;
    }

    int log2_power_of_two(std::size_t value)
    {
        return zkagent_integer::log2_power_of_two(
            value, "lookup input link requires power-of-two domains");
    }

    using mle::equality_weights;

    void append_public_statement(
        FiatShamirTranscript &transcript,
        const LookupLayout &layout,
        const LookupBundleProof &lookup_proof)
    {
        if (lookup_proof.tuple_compression_challenges.size() != layout.groups().size())
            throw std::invalid_argument("lookup proof tuple-compression metadata has wrong size");
        transcript.append_u64(layout.inputDomainSize());
        transcript.append_u64(layout.groups().size());
        transcript.append_u64(layout.affineBindings().size());
        transcript.append_u64(layout.reconstructionBindings().size());
        transcript.append_u64(static_cast<std::uint64_t>(lookup_proof.backend));
        transcript.append_u64(lookup_proof.groups.size());

        for (std::size_t group_index = 0;
             group_index < layout.groups().size(); ++group_index)
        {
            const LookupGroupSpec &group =
                layout.groups()[group_index];
            transcript.append_u64(group.actualCount());
            transcript.append_u64(group.paddedCount());
            transcript.append_fr(
                zkagent_integer::fr_from_i128(group.paddingValue()));
            transcript.append_u64(group.usesRandomTupleCompression() ? 1 : 0);
            transcript.append_fr(lookup_proof.tuple_compression_challenges[group_index]);
        }

        for (std::size_t record_index = 0;
             record_index < lookup_proof.groups.size(); ++record_index)
        {
            const LookupGroupProofRecord &record =
                lookup_proof.groups[record_index];
            transcript.append_u64(record.group_index);
            transcript.append_u64(record.query_base);
            transcript.append_u64(record.actual_query_count);
            transcript.append_u64(static_cast<std::uint64_t>(
                                       record.query_count));
            transcript.append_g1_vector(record.read_claim.commitments,
                G1WireEncoding::Compressed);
            transcript.append_fr_vector(record.read_claim.point);
            transcript.append_fr(record.read_claim.evaluation);
        }

        for (std::size_t binding_index = 0;
             binding_index < layout.affineBindings().size();
             ++binding_index)
        {
            const LookupAffineBinding &binding =
                layout.affineBindings()[binding_index];
            transcript.append_u64(binding.group_index);
            transcript.append_u64(binding.query_base);
            transcript.append_u64(binding.input_offsets.size());
            transcript.append_fr(
                zkagent_integer::fr_from_i128(binding.input_coefficient));
            transcript.append_fr(
                zkagent_integer::fr_from_i128(binding.output_coefficient));
            transcript.append_fr(
                zkagent_integer::fr_from_i128(binding.constant));
            transcript.append_u64(binding.random_tuple_compression ? 1 : 0);
        }

        for (std::size_t binding_index = 0;
             binding_index < layout.reconstructionBindings().size();
             ++binding_index)
        {
            const LookupReconstructionBinding &binding =
                layout.reconstructionBindings()[binding_index];
            transcript.append_u64(binding.input_offsets.size());
            transcript.append_fr(
                zkagent_integer::fr_from_i128(binding.input_coefficient));
            transcript.append_fr(
                zkagent_integer::fr_from_i128(binding.constant));
            transcript.append_u64(binding.lookup_terms.size());
            for (std::size_t term = 0;
                 term < binding.lookup_terms.size(); ++term)
            {
                const LookupReconstructionTerm &lookup_term =
                    binding.lookup_terms[term];
                transcript.append_u64(lookup_term.group_index);
                transcript.append_u64(lookup_term.query_base);
                transcript.append_fr(
                    zkagent_integer::fr_from_i128(lookup_term.coefficient));
            }
        }
    }

    void validate_layout_and_proof(
        const LookupLayout &layout,
        const LookupBundleProof &lookup_proof)
    {
        if (!lookup_proof.verified || lookup_proof.packing.max_chunks_per_group == 0 || lookup_proof.packing.groups.size() != layout.groups().size() || lookup_proof.packing.proof_chunks != lookup_proof.groups.size())
        {
            throw std::invalid_argument(
                "lookup input link requires a verified complete lookup bundle");
        }
        if (!zkagent_integer::is_power_of_two(layout.inputDomainSize()))
        {
            throw std::invalid_argument(
                "lookup input link received a non-power-of-two input domain");
        }
        std::size_t record_index = 0;
        for (std::size_t group_index = 0;
             group_index < layout.groups().size(); ++group_index)
        {
            const LookupGroupSpec &group = layout.groups()[group_index];
            const LookupGroupPackingPlan &group_plan =
                lookup_proof.packing.groups[group_index];
            if (group_plan.group_index != group_index ||
                group_plan.actual_count != group.actualCount() ||
                group_plan.legacy_padded_count != group.paddedCount() ||
                group_plan.chunks.empty() ||
                group_plan.chunks.size() >
                    lookup_proof.packing.max_chunks_per_group)
            {
                throw std::invalid_argument(
                    "lookup input link group shape mismatch");
            }
            std::size_t expected_query_base = 0;
            for (const LookupQueryChunkPlan &chunk : group_plan.chunks)
            {
                if (chunk.query_base != expected_query_base ||
                    chunk.actual_count == 0 ||
                    !zkagent_integer::is_power_of_two(chunk.padded_count) ||
                    chunk.actual_count > chunk.padded_count ||
                    expected_query_base > group.actualCount() ||
                    chunk.actual_count >
                        group.actualCount() - expected_query_base)
                {
                    throw std::invalid_argument(
                        "lookup input link received a non-covering chunk plan");
                }
                if (record_index >= lookup_proof.groups.size())
                {
                    throw std::invalid_argument(
                        "lookup input link is missing a packed query claim");
                }
                const LookupGroupProofRecord &record =
                    lookup_proof.groups[record_index++];
                if (record.group_index != group_index ||
                    record.query_base != chunk.query_base ||
                    record.actual_query_count != chunk.actual_count ||
                    record.query_count != static_cast<int>(chunk.padded_count) ||
                    record.table_size !=
                        static_cast<int>(group.tableValues().size()))
                {
                    throw std::invalid_argument(
                        "lookup input link packed query shape mismatch");
                }
                const int variables =
                    log2_power_of_two(chunk.padded_count);
                const std::size_t expected_commitments =
                    std::size_t(1) << (variables / 2);
                if (record.read_claim.point.size() !=
                        static_cast<std::size_t>(variables) ||
                    record.read_claim.commitments.size() !=
                        expected_commitments)
                {
                    throw std::invalid_argument(
                        "lookup input link received an incomplete query claim");
                }
                expected_query_base += chunk.actual_count;
            }
            if (expected_query_base != group.actualCount())
            {
                throw std::invalid_argument(
                    "lookup input link chunk plan does not cover the group");
            }
        }
        if (record_index != lookup_proof.groups.size())
        {
            throw std::invalid_argument(
                "lookup input link received extra packed query claims");
        }
    }

    void validate_prover_inputs(
        const LookupWitness &witness,
        const LookupBundleProof &lookup_proof)
    {
        validate_layout_and_proof(witness.layout, lookup_proof);
        if (witness.group_queries.size() != witness.layout.groups().size() ||
            lookup_proof.tuple_compression_challenges.size() != witness.layout.groups().size() ||
            (!witness.group_query_outputs.empty() &&
             witness.group_query_outputs.size() != witness.layout.groups().size()))
        {
            throw std::invalid_argument(
                "lookup input link witness/layout group count mismatch");
        }
        for (std::size_t group = 0;
             group < witness.layout.groups().size(); ++group)
        {
            if (witness.group_queries[group].size() != witness.layout.groups()[group].paddedCount())
            {
                throw std::invalid_argument(
                    "lookup input link witness group has the wrong domain");
            }
            const bool tuple =
                witness.layout.groups()[group].usesRandomTupleCompression();
            if (tuple && (witness.group_query_outputs.empty() ||
                          witness.group_query_outputs[group].size() != witness.layout.groups()[group].paddedCount() ||
                          lookup_proof.tuple_compression_challenges[group].isZero()))
            {
                throw std::invalid_argument(
                    "random tuple lookup input link is missing witness data");
            }
            if (!tuple && !witness.group_query_outputs.empty() &&
                !witness.group_query_outputs[group].empty())
            {
                throw std::invalid_argument(
                    "scalar lookup input link has tuple output values");
            }
        }
    }

    std::vector<Fr> derive_relation_point(
        FiatShamirTranscript &transcript, std::size_t row_count)
    {
        const int variables = log2_power_of_two(
            zkagent_integer::next_power_of_two(row_count));
        std::vector<Fr> point(static_cast<std::size_t>(variables));
        for (int bit = 0; bit < variables; ++bit)
        {
            point[static_cast<std::size_t>(bit)] = transcript.challenge();
        }
        return point;
    }

    Fr prefix_equality_sum(const std::vector<Fr> &point, std::size_t limit)
    {
        if (limit == 0)
            throw std::invalid_argument("equality prefix length lies outside its Boolean domain");
        return mle::prefix_sum(limit, point);
    }

    void append_split_query_blocks(
        std::vector<std::vector<QueryCoefficientBlock>> &query_blocks,
        const LookupBundleProof &lookup_proof,
        std::size_t group_index,
        std::size_t relation_query_base,
        std::size_t row_count,
        const std::vector<Fr> &relation_point,
        const Fr &scale)
    {
        if (row_count == 0 ||
            relation_query_base >
                std::numeric_limits<std::size_t>::max() - row_count)
        {
            throw std::invalid_argument(
                "lookup link received an invalid relation query interval");
        }
        const std::size_t relation_end = relation_query_base + row_count;
        std::size_t covered = 0;
        for (std::size_t record_index = 0;
             record_index < lookup_proof.groups.size(); ++record_index)
        {
            const LookupGroupProofRecord &record =
                lookup_proof.groups[record_index];
            if (record.group_index != group_index)
                continue;
            const std::size_t record_end =
                record.query_base + record.actual_query_count;
            const std::size_t begin =
                std::max(relation_query_base, record.query_base);
            const std::size_t end = std::min(relation_end, record_end);
            if (begin >= end)
                continue;
            const std::size_t count = end - begin;
            query_blocks[record_index].push_back({begin - record.query_base,
                                                  begin - relation_query_base,
                                                  count,
                                                  relation_point,
                                                  scale});
            covered += count;
        }
        if (covered != row_count)
        {
            throw std::invalid_argument(
                "packed lookup claims do not cover a relation query interval");
        }
    }

    LinkRelationPlan build_relation_plan(
        const LookupLayout &layout,
        const LookupBundleProof &lookup_proof,
        FiatShamirTranscript &transcript)
    {
        LinkRelationPlan plan;
        plan.input_domain_size = layout.inputDomainSize();
        plan.query_blocks.resize(lookup_proof.groups.size());
        if (lookup_proof.tuple_compression_challenges.size() != layout.groups().size())
        {
            throw std::invalid_argument(
                "lookup tuple-compression challenge vector has wrong size");
        }

        for (std::size_t binding_index = 0;
             binding_index < layout.affineBindings().size();
             ++binding_index)
        {
            const LookupAffineBinding &binding =
                layout.affineBindings()[binding_index];
            const Fr rho = transcript.challenge_excluding(std::vector<Fr>{Fr(0)});
            const std::vector<Fr> relation_point = derive_relation_point(
                transcript, binding.input_offsets.size());
            const Fr input_coefficient =
                zkagent_integer::fr_from_i128(binding.input_coefficient);
            const Fr output_coefficient = binding.random_tuple_compression
                                              ? lookup_proof.tuple_compression_challenges[binding.group_index]
                                              : zkagent_integer::fr_from_i128(binding.output_coefficient);
            if (binding.random_tuple_compression &&
                output_coefficient.isZero())
            {
                throw std::invalid_argument(
                    "random tuple lookup has a zero compression challenge");
            }
            plan.affine_input_relations.push_back({binding_index, relation_point, rho,
                                                   input_coefficient, output_coefficient});
            plan.constant += rho * zkagent_integer::fr_from_i128(binding.constant) * prefix_equality_sum(relation_point, binding.input_offsets.size());
            append_split_query_blocks(
                plan.query_blocks, lookup_proof, binding.group_index,
                binding.query_base, binding.input_offsets.size(),
                relation_point, -rho);
            ++plan.relation_claims;
        }

        for (std::size_t binding_index = 0;
             binding_index < layout.reconstructionBindings().size();
             ++binding_index)
        {
            const LookupReconstructionBinding &binding =
                layout.reconstructionBindings()[binding_index];
            const Fr rho = transcript.challenge_excluding(std::vector<Fr>{Fr(0)});
            const std::vector<Fr> relation_point = derive_relation_point(
                transcript, binding.input_offsets.size());
            plan.reconstruction_input_relations.push_back({binding_index, relation_point, rho});
            plan.constant += rho * zkagent_integer::fr_from_i128(binding.constant) * prefix_equality_sum(relation_point, binding.input_offsets.size());
            for (const LookupReconstructionTerm &term :
                 binding.lookup_terms)
            {
                append_split_query_blocks(
                    plan.query_blocks, lookup_proof, term.group_index,
                    term.query_base, binding.input_offsets.size(),
                    relation_point,
                    rho * zkagent_integer::fr_from_i128(term.coefficient));
            }
            ++plan.relation_claims;
        }

        for (std::size_t record_index = 0;
             record_index < plan.query_blocks.size(); ++record_index)
        {
            if (plan.query_blocks[record_index].empty())
            {
                throw std::invalid_argument(
                    "packed lookup claim is not linked by a public relation");
            }
            const std::size_t domain = static_cast<std::size_t>(
                lookup_proof.groups[record_index].query_count);
            for (const QueryCoefficientBlock &block :
                 plan.query_blocks[record_index])
            {
                const std::size_t local_domain =
                    std::size_t(1) << block.relation_point.size();
                if (block.row_count == 0 ||
                    block.row_count > local_domain ||
                    block.relation_base > local_domain - block.row_count ||
                    block.row_count > domain ||
                    block.query_base > domain - block.row_count)
                {
                    throw std::out_of_range(
                        "lookup relation block lies outside its query polynomial");
                }
            }
        }
        return plan;
    }

    struct RoundEvaluations
    {
        Fr eval0 = Fr(0);
        Fr eval1 = Fr(0);
        Fr eval2 = Fr(0);
    };

    RoundEvaluations product_round(
        const std::vector<Fr> &coefficients,
        const std::vector<Fr> &values,
        std::size_t active,
        int threads)
    {
        const std::size_t pairs = active / 2;
        const int actual_threads = std::max(
            1, std::min<int>(threads, static_cast<int>(pairs)));
        std::vector<RoundEvaluations> local(
            static_cast<std::size_t>(actual_threads));
#pragma omp parallel num_threads(actual_threads) if (actual_threads > 1)
        {
            const int tid = omp_get_thread_num();
            RoundEvaluations sum;
#pragma omp for schedule(static)
            for (long long raw_pair = 0;
                 raw_pair < static_cast<long long>(pairs); ++raw_pair)
            {
                const std::size_t pair = static_cast<std::size_t>(raw_pair);
                const Fr &c0 = coefficients[2 * pair];
                const Fr &c1 = coefficients[2 * pair + 1];
                const Fr &v0 = values[2 * pair];
                const Fr &v1 = values[2 * pair + 1];
                sum.eval0 += c0 * v0;
                sum.eval1 += c1 * v1;
                sum.eval2 += (c1 + c1 - c0) * (v1 + v1 - v0);
            }
            local[static_cast<std::size_t>(tid)] = sum;
        }
        RoundEvaluations result;
        for (const RoundEvaluations &sum : local)
        {
            result.eval0 += sum.eval0;
            result.eval1 += sum.eval1;
            result.eval2 += sum.eval2;
        }
        return result;
    }

    void fold_with_reusable_scratch(
        std::vector<Fr> &values,
        std::vector<Fr> &scratch,
        std::size_t active,
        const Fr &challenge,
        int threads)
    {
        if (active == 0 || (active & 1U) != 0 ||
            active > values.size())
        {
            throw std::invalid_argument(
                "lookup link fold has inconsistent active storage");
        }
        const std::size_t pairs = active / 2;
        scratch.resize(pairs);
        const int actual_threads = std::max(
            1, std::min<int>(threads, static_cast<int>(pairs)));
#pragma omp parallel for schedule(static) num_threads(actual_threads) if (actual_threads > 1)
        for (long long raw_pair = 0;
             raw_pair < static_cast<long long>(pairs); ++raw_pair)
        {
            const std::size_t pair = static_cast<std::size_t>(raw_pair);
            const Fr left = values[2 * pair];
            scratch[pair] = left +
                            (values[2 * pair + 1] - left) * challenge;
        }
        values.swap(scratch);
    }

    struct FactoredEquality
    {
        std::vector<Fr> low, high;
        std::size_t low_bits = 0;
        std::size_t domain() const { return low.size() * high.size(); }
    };

    class EqualityCache
    {
    public:
        const FactoredEquality &get(const std::vector<Fr> &point, int threads,
                                    LookupInputLinkTiming *timing)
        {
            if (point.size() >= sizeof(std::size_t) * 8)
                throw std::invalid_argument("equality point exceeds the index domain");
            std::string key;
            key.reserve(point.size() * 32);
            for (const Fr &coordinate : point)
            {
                std::array<std::uint8_t, 32> bytes{};
                if (coordinate.serialize(bytes.data(), bytes.size()) != bytes.size())
                    throw std::runtime_error("cannot encode equality-cache coordinate");
                key.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
            }
            const auto found = tables_.find(key);
            if (found != tables_.end())
            {
                if (timing)
                    ++timing->equality_cache_hits;
                return found->second;
            }
            FactoredEquality result;
            result.low_bits = point.size() / 2;
            result.low = equality_weights(
                std::vector<Fr>(point.begin(), point.begin() + result.low_bits), threads);
            result.high = equality_weights(
                std::vector<Fr>(point.begin() + result.low_bits, point.end()), threads);
            if (timing)
            {
                timing->equality_tables += 2;
                timing->equality_values += result.low.size() + result.high.size();
            }
            return tables_.emplace(std::move(key), std::move(result)).first->second;
        }

    private:
        std::map<std::string, FactoredEquality> tables_;
    };

    void add_equality_slice(std::vector<Fr> &coefficients,
                            const FactoredEquality &weights, const Fr &scale,
                            std::size_t source_base, std::size_t destination_base,
                            std::size_t count, int threads)
    {
        if (count == 0 || count > weights.domain() ||
            source_base > weights.domain() - count || count > coefficients.size() ||
            destination_base > coefficients.size() - count)
            throw std::invalid_argument("equality slice lies outside its domains");
        const std::size_t low_size = weights.low.size();
        const std::size_t first = source_base >> weights.low_bits;
        const std::size_t last = (source_base + count - 1) >> weights.low_bits;
        const int workers = std::max(1, std::min<int>(threads, static_cast<int>(last - first + 1)));
#pragma omp parallel for schedule(static) num_threads(workers) if (use_parallel(count, workers))
        for (long long raw = static_cast<long long>(first);
             raw <= static_cast<long long>(last); ++raw)
        {
            const std::size_t high = static_cast<std::size_t>(raw);
            const std::size_t begin = std::max(source_base, high * low_size);
            const std::size_t end = std::min(source_base + count, (high + 1) * low_size);
            const Fr scaled_high = scale * weights.high[high];
            std::size_t destination = destination_base + begin - source_base;
            for (std::size_t source = begin; source < end; ++source, ++destination)
                coefficients[destination] += scaled_high * weights.low[source & (low_size - 1)];
        }
    }

    std::vector<Fr> materialize_query_coefficients(
        std::size_t domain,
        const std::vector<QueryCoefficientBlock> &blocks,
        int requested_threads,
        LookupInputLinkTiming *timing,
        EqualityCache &cache)
    {
        const int threads = normalized_threads(requested_threads);
        std::vector<Fr> coefficients(domain);
#pragma omp parallel for schedule(static) num_threads(threads) if (use_parallel(domain, threads))
        for (long long raw = 0; raw < static_cast<long long>(domain); ++raw)
        {
            coefficients[static_cast<std::size_t>(raw)].clear();
        }
        for (const QueryCoefficientBlock &block : blocks)
        {
            const auto &weights = cache.get(block.relation_point, threads, timing);
            if (block.row_count == 0 || block.row_count > weights.domain() ||
                block.relation_base > weights.domain() - block.row_count ||
                block.row_count > domain ||
                block.query_base > domain - block.row_count)
            {
                throw std::invalid_argument(
                    "cannot materialize a malformed query coefficient block");
            }
            add_equality_slice(coefficients, weights, block.scale,
                               block.relation_base, block.query_base, block.row_count, threads);
        }
        return coefficients;
    }

    Fr shifted_interval_identity_evaluation(
        const std::vector<Fr> &source_point,
        const std::vector<Fr> &destination_point,
        std::size_t source_base,
        std::size_t destination_base,
        std::size_t limit)
    {
        if (source_point.size() >= sizeof(std::size_t) * 8 ||
            destination_point.size() >= sizeof(std::size_t) * 8)
        {
            throw std::invalid_argument(
                "shifted identity evaluation has incompatible points");
        }
        const std::size_t source_domain =
            std::size_t(1) << source_point.size();
        const std::size_t destination_domain =
            std::size_t(1) << destination_point.size();
        if (limit == 0 || limit > source_domain ||
            source_base > source_domain - limit ||
            limit > destination_domain ||
            destination_base > destination_domain - limit)
        {
            throw std::invalid_argument(
                "shifted identity slice exceeds its Boolean domain");
        }
        const std::size_t comparison_offset = source_domain - limit;
        Fr state[2][2][2];
        Fr next[2][2][2];
        for (int source_carry = 0; source_carry < 2; ++source_carry)
        {
            for (int destination_carry = 0; destination_carry < 2;
                 ++destination_carry)
            {
                for (int compare_carry = 0; compare_carry < 2;
                     ++compare_carry)
                {
                    state[source_carry][destination_carry][compare_carry] =
                        Fr(0);
                }
            }
        }
        state[0][0][0] = Fr(1);

        const std::size_t bits =
            std::max(source_point.size(), destination_point.size());
        for (std::size_t bit = 0; bit < bits; ++bit)
        {
            for (int source_carry = 0; source_carry < 2; ++source_carry)
            {
                for (int destination_carry = 0; destination_carry < 2;
                     ++destination_carry)
                {
                    for (int compare_carry = 0; compare_carry < 2;
                         ++compare_carry)
                    {
                        next[source_carry][destination_carry][compare_carry] =
                            Fr(0);
                    }
                }
            }
            const unsigned source_base_bit = bit < source_point.size()
                                                 ? static_cast<unsigned>((source_base >> bit) & 1U)
                                                 : 0;
            const unsigned destination_base_bit = bit < destination_point.size()
                                                      ? static_cast<unsigned>((destination_base >> bit) & 1U)
                                                      : 0;
            const unsigned comparison_bit = bit < source_point.size()
                                                ? static_cast<unsigned>((comparison_offset >> bit) & 1U)
                                                : 0;
            const unsigned choices = bit < source_point.size() ? 2 : 1;
            for (unsigned source_carry = 0; source_carry < 2;
                 ++source_carry)
            {
                for (unsigned destination_carry = 0;
                     destination_carry < 2; ++destination_carry)
                {
                    for (unsigned compare_carry = 0; compare_carry < 2;
                         ++compare_carry)
                    {
                        for (unsigned index_bit = 0; index_bit < choices;
                             ++index_bit)
                        {
                            const unsigned source_sum =
                                source_base_bit + index_bit + source_carry;
                            const unsigned source_bit = source_sum & 1U;
                            const unsigned next_source_carry = source_sum >> 1U;
                            const unsigned destination_sum =
                                destination_base_bit + index_bit +
                                destination_carry;
                            const unsigned destination_bit =
                                destination_sum & 1U;
                            const unsigned next_destination_carry =
                                destination_sum >> 1U;
                            unsigned next_compare_carry = compare_carry;
                            Fr weight = state[source_carry][destination_carry]
                                             [compare_carry];
                            if (weight.isZero())
                                continue;
                            if (bit < source_point.size())
                            {
                                const unsigned compare_sum =
                                    comparison_bit + index_bit + compare_carry;
                                next_compare_carry = compare_sum >> 1U;
                                weight *= source_bit != 0
                                              ? source_point[bit]
                                              : Fr(1) - source_point[bit];
                            }
                            else if (source_bit != 0)
                            {
                                continue;
                            }
                            if (bit < destination_point.size())
                            {
                                weight *= destination_bit != 0
                                              ? destination_point[bit]
                                              : Fr(1) - destination_point[bit];
                            }
                            else if (destination_bit != 0)
                            {
                                continue;
                            }
                            next[next_source_carry][next_destination_carry]
                                [next_compare_carry] += weight;
                        }
                    }
                }
            }
            for (int source_carry = 0; source_carry < 2; ++source_carry)
            {
                for (int destination_carry = 0; destination_carry < 2;
                     ++destination_carry)
                {
                    for (int compare_carry = 0; compare_carry < 2;
                         ++compare_carry)
                    {
                        state[source_carry][destination_carry][compare_carry] =
                            next[source_carry][destination_carry][compare_carry];
                    }
                }
            }
        }
        return state[0][0][0];
    }

    Fr shifted_truncated_identity_evaluation(
        const std::vector<Fr> &source_point,
        const std::vector<Fr> &destination_point,
        std::size_t base,
        std::size_t limit)
    {
        return shifted_interval_identity_evaluation(
            source_point, destination_point, 0, base, limit);
    }

    Fr evaluate_query_coefficient(
        std::size_t domain,
        const std::vector<QueryCoefficientBlock> &blocks,
        const std::vector<Fr> &point)
    {
        if (domain != (std::size_t(1) << point.size()))
        {
            throw std::invalid_argument(
                "query coefficient evaluation point has the wrong dimension");
        }
        Fr result = Fr(0);
        for (const QueryCoefficientBlock &block : blocks)
        {
            const std::size_t local_bits = block.relation_point.size();
            const std::size_t local_domain = std::size_t(1) << local_bits;
            if (block.row_count == 0 ||
                block.row_count > local_domain ||
                block.relation_base > local_domain - block.row_count ||
                block.row_count > domain ||
                block.query_base > domain - block.row_count)
            {
                throw std::invalid_argument(
                    "query coefficient slice lies outside its group");
            }
            const Fr basis = shifted_interval_identity_evaluation(
                block.relation_point, point,
                block.relation_base, block.query_base, block.row_count);
            result += block.scale * basis;
        }
        return result;
    }

    std::vector<Fr> materialize_query_values(
        const LookupWitness &witness,
        const LookupBundleProof &lookup_proof,
        const LookupGroupProofRecord &record,
        int threads)
    {
        if (record.group_index >= witness.layout.groups().size() ||
            record.query_count <= 0)
        {
            throw std::invalid_argument(
                "cannot materialize an invalid packed lookup record");
        }
        const LookupGroupSpec &group =
            witness.layout.groups()[record.group_index];
        const std::vector<ll> &queries =
            witness.group_queries[record.group_index];
        const bool tuple = group.usesRandomTupleCompression();
        const Fr tuple_challenge = tuple
                                       ? lookup_proof.tuple_compression_challenges.at(record.group_index)
                                       : Fr(0);
        const std::vector<ll> *outputs = tuple
                                             ? &witness.group_query_outputs.at(record.group_index)
                                             : nullptr;
        ll padding_output = 0;
        if (tuple)
        {
            const auto padding = std::find(
                group.tableValues().begin(), group.tableValues().end(),
                group.paddingValue());
            if (padding == group.tableValues().end() || tuple_challenge.isZero())
            {
                throw std::invalid_argument(
                    "invalid random tuple lookup padding/challenge");
            }
            padding_output = group.tableOutputs().at(static_cast<std::size_t>(
                padding - group.tableValues().begin()));
        }
        if (record.query_base > group.actualCount() ||
            record.actual_query_count >
                group.actualCount() - record.query_base ||
            record.actual_query_count >
                static_cast<std::size_t>(record.query_count))
        {
            throw std::invalid_argument(
                "packed lookup record exceeds its source query group");
        }
        const std::size_t domain =
            static_cast<std::size_t>(record.query_count);
        std::vector<Fr> values(domain);
        const int actual_threads = std::max(
            1, std::min<int>(threads, record.query_count));
#pragma omp parallel for schedule(static) num_threads(actual_threads) if (actual_threads > 1)
        for (long long raw_index = 0;
             raw_index < static_cast<long long>(domain); ++raw_index)
        {
            const std::size_t index = static_cast<std::size_t>(raw_index);
            const ll value = index < record.actual_query_count
                                 ? queries[record.query_base + index]
                                 : group.paddingValue();
            values[index] = zkagent_integer::fr_from_i128(value);
            if (tuple)
            {
                const ll output = index < record.actual_query_count
                                      ? (*outputs)[record.query_base + index]
                                      : padding_output;
                values[index] +=
                    tuple_challenge * zkagent_integer::fr_from_i128(output);
            }
        }
        return values;
    }

    struct QueryBucket
    {
        std::size_t domain = 0;
        std::vector<std::size_t> records;
    };

    std::vector<QueryBucket> build_query_buckets(
        const LookupBundleProof &lookup_proof)
    {
        std::map<std::size_t, std::vector<std::size_t>> by_domain;
        for (std::size_t record = 0;
             record < lookup_proof.groups.size(); ++record)
        {
            const int query_count = lookup_proof.groups[record].query_count;
            if (query_count <= 0)
            {
                throw std::invalid_argument(
                    "lookup link record has an invalid query domain");
            }
            by_domain[static_cast<std::size_t>(query_count)].push_back(record);
        }
        std::vector<QueryBucket> buckets;
        buckets.reserve(by_domain.size());
        for (auto &entry : by_domain)
        {
            buckets.push_back({entry.first, std::move(entry.second)});
        }
        return buckets;
    }

    void add_lookup_claim_coefficient(
        std::vector<Fr> &coefficients,
        const LookupReadClaim &read_claim,
        const Fr &scale,
        int requested_threads,
        LookupInputLinkTiming *timing,
        EqualityCache &cache)
    {
        const int threads = normalized_threads(requested_threads);
        const auto &weights = cache.get(read_claim.point, threads, timing);
        if (weights.domain() != coefficients.size())
        {
            throw std::invalid_argument(
                "lookup claim point does not match its query domain");
        }
        add_equality_slice(coefficients, weights, scale, 0, 0, coefficients.size(), threads);
    }

    struct QueryProverState
    {
        std::size_t record = 0;
        Fr lookup_claim_challenge = Fr(0);
        std::vector<Fr> coefficients;
        std::vector<Fr> values;
        std::vector<Fr> fold_scratch;
        std::vector<Fr> row_folded;
    };

    void prove_query_bucket(
        const QueryBucket &bucket,
        const LinkRelationPlan &plan,
        const LookupWitness &witness,
        const LookupBundleProof &lookup_proof,
        const MVKZG_ProverKey &pk,
        int threads,
        FiatShamirProofWriter &writer,
        LookupInputLinkTiming *timing,
        EqualityCache &equality_cache)
    {
        const int variables = log2_power_of_two(bucket.domain);
        const int row_variables = variables / 2;
        const std::size_t columns = std::size_t(1) << (variables - row_variables);
        std::vector<QueryProverState> states;
        states.reserve(bucket.records.size());
        for (std::size_t record : bucket.records)
        {
            const Fr eta = writer.transcript().nonzero_challenge();
            QueryProverState state;
            state.record = record;
            state.lookup_claim_challenge = eta;
            auto stage_start = Clock::now();
            state.coefficients = materialize_query_coefficients(
                bucket.domain, plan.query_blocks[record], threads, timing, equality_cache);
            add_lookup_claim_coefficient(
                state.coefficients,
                lookup_proof.groups[record].read_claim, eta, threads, timing, equality_cache);
            if (timing)
                timing->coefficient_seconds += std::chrono::duration<double>(
                                                   Clock::now() - stage_start)
                                                   .count();
            stage_start = Clock::now();
            state.values = materialize_query_values(
                witness, lookup_proof, lookup_proof.groups[record], threads);
            if (timing)
                timing->query_value_seconds += std::chrono::duration<double>(
                                                   Clock::now() - stage_start)
                                                   .count();
            if (row_variables == 0)
                state.row_folded = state.values;
            states.push_back(std::move(state));
        }

        Fr claim = Fr(0);
        std::vector<Fr> point;
        point.reserve(static_cast<std::size_t>(variables));
        std::size_t active = bucket.domain;
        for (int round = 0; round < variables; ++round)
        {
            const auto round_start = Clock::now();
            RoundEvaluations message;
            for (const QueryProverState &state : states)
            {
                const RoundEvaluations contribution = product_round(
                    state.coefficients, state.values, active, threads);
                message.eval0 += contribution.eval0;
                message.eval1 += contribution.eval1;
                message.eval2 += contribution.eval2;
            }
            if (round == 0)
            {
                claim = message.eval0 + message.eval1;
                writer.write_fr(claim);
            }
            else if (claim != message.eval0 + message.eval1)
            {
                throw std::logic_error(
                    "batched query-claim prover produced an inconsistent sumcheck");
            }
            writer.write_fr(message.eval0);
            writer.write_fr(message.eval2);
            const Fr challenge = writer.challenge_not_one();
            point.push_back(challenge);
            claim = polynomial::evaluate_quadratic_samples(
                message.eval0, message.eval1, message.eval2, challenge);
            if (timing)
                timing->round_seconds += std::chrono::duration<double>(
                                             Clock::now() - round_start)
                                             .count();
            const auto fold_start = Clock::now();
            for (QueryProverState &state : states)
            {
                fold_with_reusable_scratch(
                    state.coefficients, state.fold_scratch,
                    active, challenge, threads);
                fold_with_reusable_scratch(
                    state.values, state.fold_scratch,
                    active, challenge, threads);
                if (round + 1 == row_variables)
                    state.row_folded = state.values;
            }
            if (timing)
                timing->fold_seconds += std::chrono::duration<double>(
                                            Clock::now() - fold_start)
                                            .count();
            active /= 2;
        }
        if (variables == 0)
        {
            for (const QueryProverState &state : states)
            {
                claim += state.coefficients[0] * state.values[0];
            }
            writer.write_fr(claim);
        }

        Fr final_claim = Fr(0);
        for (const QueryProverState &state : states)
        {
            Fr coefficient_evaluation = evaluate_query_coefficient(
                bucket.domain, plan.query_blocks[state.record], point);
            coefficient_evaluation += state.lookup_claim_challenge * shifted_truncated_identity_evaluation(
                                                                         lookup_proof.groups[state.record].read_claim.point,
                                                                         point, 0, bucket.domain);
            if (state.coefficients[0] != coefficient_evaluation)
            {
                throw std::logic_error(
                    "structured batched coefficient disagrees with folding");
            }
            const Fr evaluation = state.values[0];
            writer.write_fr(evaluation);
            final_claim += coefficient_evaluation * evaluation;
        }
        if (claim != final_claim)
        {
            throw std::logic_error(
                "batched query-claim final product is inconsistent");
        }

        const auto opening_start = Clock::now();
        std::vector<Fr> opening_challenges;
        opening_challenges.reserve(states.size());
        for (std::size_t i = 0; i < states.size(); ++i)
            opening_challenges.push_back(writer.transcript().nonzero_challenge());
        for (const auto &state : states)
            if (state.row_folded.size() != columns)
                throw std::logic_error("query sumcheck did not retain its PCS row fold");
        std::vector<Fr> aggregate(columns);
#pragma omp parallel for schedule(static) num_threads(threads) if (use_parallel(columns, threads))
        for (long long raw = 0;
             raw < static_cast<long long>(columns); ++raw)
        {
            const std::size_t index = static_cast<std::size_t>(raw);
            Fr value = Fr(0);
            for (std::size_t i = 0; i < states.size(); ++i)
            {
                value += opening_challenges[i] * states[i].row_folded[index];
            }
            aggregate[index] = value;
        }
        const hyrax::OpeningProof opening = hyrax::open_folded(
            aggregate, point.data(), variables, pk);
        writer.write_g1_vector(opening.eval_proof.q_comm);
        if (timing)
            timing->opening_seconds += std::chrono::duration<double>(
                                           Clock::now() - opening_start)
                                           .count();
    }

    Fr verify_query_bucket(
        const QueryBucket &bucket,
        const LinkRelationPlan &plan,
        const LookupBundleProof &lookup_proof,
        hyrax::OpeningBatchVerifier &opening_batch,
        FiatShamirProofReader &reader, int threads)
    {
        const int variables = log2_power_of_two(bucket.domain);
        std::vector<Fr> lookup_claim_challenges;
        lookup_claim_challenges.reserve(bucket.records.size());
        Fr lookup_claim_offset = Fr(0);
        for (std::size_t record : bucket.records)
        {
            const Fr eta = reader.transcript().nonzero_challenge();
            lookup_claim_challenges.push_back(eta);
            lookup_claim_offset +=
                eta * lookup_proof.groups[record].read_claim.evaluation;
        }

        Fr claim = reader.read_fr();
        const Fr initial_claim = claim;
        std::vector<Fr> point;
        point.reserve(static_cast<std::size_t>(variables));
        for (int round = 0; round < variables; ++round)
        {
            const Fr eval0 = reader.read_fr();
            const Fr eval2 = reader.read_fr();
            const Fr eval1 = claim - eval0;
            const Fr challenge = reader.challenge_not_one();
            point.push_back(challenge);
            claim = polynomial::evaluate_quadratic_samples(eval0, eval1, eval2, challenge);
        }

        Fr final_claim = Fr(0);
        std::vector<Fr> query_evaluations;
        query_evaluations.reserve(bucket.records.size());
        for (std::size_t i = 0; i < bucket.records.size(); ++i)
        {
            const std::size_t record = bucket.records[i];
            const Fr evaluation = reader.read_fr();
            query_evaluations.push_back(evaluation);
            Fr coefficient_evaluation = evaluate_query_coefficient(
                bucket.domain, plan.query_blocks[record], point);
            coefficient_evaluation += lookup_claim_challenges[i] * shifted_truncated_identity_evaluation(
                                                                       lookup_proof.groups[record].read_claim.point,
                                                                       point, 0, bucket.domain);
            final_claim += coefficient_evaluation * evaluation;
        }
        if (claim != final_claim)
        {
            throw std::invalid_argument(
                "batched query-claim sumcheck final check failed");
        }

        std::vector<Fr> opening_challenges;
        opening_challenges.reserve(bucket.records.size());
        Fr aggregate_evaluation = Fr(0);
        for (std::size_t i = 0; i < bucket.records.size(); ++i)
        {
            const Fr challenge = reader.transcript().nonzero_challenge();
            opening_challenges.push_back(challenge);
            aggregate_evaluation += challenge * query_evaluations[i];
        }
        std::vector<hyrax::RowCommitmentTerm> commitment_terms;
        commitment_terms.reserve(bucket.records.size());
        for (std::size_t i = 0; i < bucket.records.size(); ++i)
            commitment_terms.push_back({&lookup_proof.groups[bucket.records[i]].read_claim.commitments,
                                        opening_challenges[i]});
        hyrax::OpeningProof opening;
        opening.eval_proof.q_comm = reader.read_g1_vector(static_cast<std::size_t>(variables - variables / 2));
        if (!opening_batch.append_combined(
                opening, point.data(), aggregate_evaluation,
                commitment_terms, variables, threads))
        {
            throw std::invalid_argument(
                "failed to append batched query-claim opening");
        }
        return initial_claim - lookup_claim_offset;
    }

    InputReductionLink make_input_link(
        LinkRelationPlan plan,
        const LookupLayout &layout,
        const Fr &claimed_evaluation,
        std::size_t query_claims)
    {
        InputReductionLink link;
        link.input_domain_size = plan.input_domain_size;
        link.layout = &layout;
        link.affine_relations = std::move(plan.affine_input_relations);
        link.reconstruction_relations =
            std::move(plan.reconstruction_input_relations);
        link.claimed_evaluation = claimed_evaluation;
        link.query_claims = query_claims;
        link.relation_claims = plan.relation_claims;
        return link;
    }

}

LookupInputLinkProof proveLookupInputLink(
    const LookupWitness &witness,
    const LookupBundleProof &lookup_proof,
    const MVKZG_ProverKey &pk,
    int threads,
    double *prover_time,
    LookupInputLinkTiming *timing)
{
    FiatShamirTranscript transcript;
    return proveLookupInputLink(
        witness, lookup_proof, pk, threads, transcript, prover_time, timing);
}

LookupInputLinkProof proveLookupInputLink(
    const LookupWitness &witness,
    const LookupBundleProof &lookup_proof,
    const MVKZG_ProverKey &pk,
    int threads,
    FiatShamirTranscript &transcript,
    double *prover_time,
    LookupInputLinkTiming *timing)
{
    const auto start = Clock::now();
    if (timing)
        *timing = LookupInputLinkTiming{};
    validate_prover_inputs(witness, lookup_proof);
    if (threads <= 0)
    {
        throw std::invalid_argument(
            "lookup input link requires a positive thread count");
    }

    FiatShamirProofWriter writer(
        G1WireEncoding::Compressed, true, &transcript);
    append_public_statement(writer.transcript(), witness.layout, lookup_proof);
    LinkRelationPlan plan =
        build_relation_plan(
            witness.layout, lookup_proof, writer.transcript());
    const std::vector<QueryBucket> buckets =
        build_query_buckets(lookup_proof);
    EqualityCache equality_cache;
    if (timing)
        timing->statement_seconds = std::chrono::duration<double>(
                                        Clock::now() - start)
                                        .count();
    for (std::size_t bucket = 0; bucket < buckets.size(); ++bucket)
    {
        prove_query_bucket(
            buckets[bucket], plan, witness, lookup_proof,
            pk, threads, writer, timing, equality_cache);
    }

    LookupInputLinkProof proof;
    proof.compressed_payload_bytes = writer.compressed_payload_bytes();
    proof.wire_bytes = writer.take_bytes();
    if (prover_time != nullptr)
    {
        *prover_time = std::chrono::duration<double>(
                           Clock::now() - start)
                           .count();
    }
    return proof;
}

bool verifyLookupInputLink(
    const LookupLayout &layout,
    const LookupBundleProof &lookup_proof,
    const LookupInputLinkProof &proof,
    const MVKZG_VerifierKey &vk,
    int threads,
    InputReductionLink *input_link,
    double *verifier_time)
{
    FiatShamirTranscript transcript;
    return verifyLookupInputLink(
        layout, lookup_proof, proof, vk, threads, input_link,
        transcript, verifier_time);
}

bool verifyLookupInputLink(
    const LookupLayout &layout,
    const LookupBundleProof &lookup_proof,
    const LookupInputLinkProof &proof,
    const MVKZG_VerifierKey &vk,
    int threads,
    InputReductionLink *input_link,
    FiatShamirTranscript &transcript,
    double *verifier_time)
{
    const auto start = Clock::now();
    try
    {
        validate_layout_and_proof(layout, lookup_proof);
        if (threads <= 0)
        {
            throw std::invalid_argument(
                "lookup input link requires a positive thread count");
        }
        FiatShamirProofReader reader(proof.wire_bytes, &transcript);
        append_public_statement(reader.transcript(), layout, lookup_proof);
        LinkRelationPlan plan =
            build_relation_plan(
                layout, lookup_proof, reader.transcript());

        Fr query_claim_sum = Fr(0);
        const std::vector<QueryBucket> buckets =
            build_query_buckets(lookup_proof);
        hyrax::OpeningBatchVerifier opening_batch;
        for (std::size_t bucket = 0; bucket < buckets.size(); ++bucket)
        {
            query_claim_sum += verify_query_bucket(
                buckets[bucket], plan, lookup_proof,
                opening_batch, reader, threads);
        }
        if (opening_batch.size() != buckets.size() ||
            !opening_batch.verify(vk))
        {
            throw std::invalid_argument(
                "batched query-claim commitment openings failed");
        }
        if (!reader.at_end())
        {
            throw std::invalid_argument(
                "lookup input-link proof has trailing bytes");
        }
        if (input_link != nullptr)
        {
            const Fr input_claim = -query_claim_sum - plan.constant;
            *input_link = make_input_link(
                std::move(plan), layout, input_claim,
                lookup_proof.groups.size());
        }
        if (verifier_time != nullptr)
        {
            *verifier_time = std::chrono::duration<double>(
                                 Clock::now() - start)
                                 .count();
        }
        return true;
    }
    catch (const std::exception &error)
    {
        if (std::getenv("LOOKUP_INPUT_LINK_DEBUG") != nullptr)
        {
            std::cerr << "lookup input link verification failed: "
                      << error.what() << std::endl;
        }
        if (verifier_time != nullptr)
        {
            *verifier_time = std::chrono::duration<double>(
                                 Clock::now() - start)
                                 .count();
        }
        return false;
    }
}

LookupInputLinkResult proveAndVerifyLookupInputLink(
    const LookupWitness &witness,
    const LookupBundleProof &lookup_proof,
    const MVKZG_ProverKey &pk,
    const MVKZG_VerifierKey &vk,
    int threads,
    FiatShamirTranscript *prover_transcript,
    FiatShamirTranscript *verifier_transcript)
{
    if ((prover_transcript == nullptr) !=
        (verifier_transcript == nullptr))
    {
        throw std::invalid_argument(
            "lookup-link outer transcripts must be supplied as a pair");
    }
    LookupInputLinkResult result;
    if (prover_transcript != nullptr)
    {
        result.proof = proveLookupInputLink(
            witness, lookup_proof, pk, threads, *prover_transcript,
            &result.prover_time, &result.timing);
        result.verified = verifyLookupInputLink(
            witness.layout, lookup_proof, result.proof, vk, threads,
            &result.input_link, *verifier_transcript,
            &result.verifier_time);
    }
    else
    {
        result.proof = proveLookupInputLink(
            witness, lookup_proof, pk, threads, &result.prover_time, &result.timing);
        result.verified = verifyLookupInputLink(
            witness.layout, lookup_proof, result.proof, vk, threads,
            &result.input_link, &result.verifier_time);
    }
    result.proof_size_bytes = result.proof.compressed_payload_bytes;
    result.query_claims = lookup_proof.groups.size();
    result.opening_buckets = build_query_buckets(lookup_proof).size();
    result.opening_verification_batches =
        result.opening_buckets == 0 ? 0 : 1;
    result.relation_claims = witness.layout.affineBindings().size() + witness.layout.reconstructionBindings().size();
    return result;
}
