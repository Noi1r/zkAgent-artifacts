#include "zkdecode/zkdecode_lookup.hpp"
#include "common/field_integer.hpp"
#include "common/bit_utils.hpp"
#include "zkdecode/zkdecode_proof.hpp"

#include "pcs/hyrax.hpp"
#include "pcs/srs.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

    using zkdecode_detail::RangeLimbBits;

    enum class QueryCommitmentRole
    {
        InterfaceQuery,
        GeneratedQueries,
    };

    struct DecodeLookupRequest
    {
        std::string id;
        std::string table_id;
        std::vector<ll> table;
        std::vector<ll> table_outputs;
        std::vector<ll> queries;
        std::vector<ll> query_outputs;
        Fr tuple_compression_challenge = Fr(0);
        std::vector<G1> expected_commitments;
        double precommitted_query_time = 0.0;
    };

    std::vector<G1> combine_commitments(
        const std::vector<G1> &low,
        const std::vector<G1> &high,
        const Fr &high_weight,
        int threads)
    {
        if (low.size() != high.size())
        {
            throw std::invalid_argument("lookup commitment dimensions differ");
        }
        std::vector<G1> result(low.size());
        const int worker_count = std::max(threads, 1);
#pragma omp parallel for num_threads(worker_count) schedule(static) if (result.size() >= 64)
        for (long long raw_row = 0;
             raw_row < static_cast<long long>(result.size()); ++raw_row)
        {
            const std::size_t row = static_cast<std::size_t>(raw_row);
            result[row] = low[row] + high[row] * high_weight;
        }
        return result;
    }

    DecodeLookupRequest identity_request(
        std::string id,
        std::vector<ll> queries,
        int table_bits,
        std::vector<G1> expected_commitments = {},
        const std::vector<ll> *public_table = nullptr)
    {
        if (queries.empty() || table_bits <= 0 || table_bits > RangeLimbBits)
        {
            throw std::invalid_argument("invalid identity lookup request");
        }
        const int table_size = 1 << table_bits;
        for (ll query : queries)
        {
            if (query < 0 || query >= table_size)
            {
                throw std::out_of_range("identity lookup query is outside its table");
            }
        }
        DecodeLookupRequest request;
        request.id = std::move(id);
        request.table_id = "range/" + std::to_string(table_bits);
        if (public_table != nullptr)
        {
            if (public_table->size() != static_cast<std::size_t>(table_size))
                throw std::invalid_argument("identity lookup public table has the wrong size");
            request.table = *public_table;
        }
        else
        {
            request.table.resize(static_cast<std::size_t>(table_size));
            for (int value = 0; value < table_size; ++value)
                request.table[static_cast<std::size_t>(value)] = value;
        }
        request.queries = std::move(queries);
        request.expected_commitments = std::move(expected_commitments);
        return request;
    }

    DecodeLookupRequest mapped_request(
        std::string id,
        std::string table_id,
        std::vector<ll> inputs,
        std::vector<ll> outputs,
        const std::vector<ll> &public_inputs,
        const std::vector<ll> &public_outputs,
        int input_bits,
        const Fr &tuple_compression_challenge,
        int threads,
        const std::vector<G1> &expected_inputs,
        const std::vector<G1> &expected_outputs)
    {
        if (inputs.empty() ||
            inputs.size() != outputs.size() ||
            input_bits <= 0 ||
            input_bits >= 30 ||
            tuple_compression_challenge.isZero() ||
            public_inputs.size() != (static_cast<std::size_t>(1) << input_bits) ||
            public_outputs.size() != public_inputs.size())
        {
            throw std::invalid_argument("invalid mapped public-table lookup");
        }
        DecodeLookupRequest request;
        request.id = std::move(id);
        request.table_id = std::move(table_id);
        request.table = public_inputs;
        request.table_outputs = public_outputs;
        request.tuple_compression_challenge = tuple_compression_challenge;
        for (std::size_t row = 0; row < inputs.size(); ++row)
        {
            const ll input = inputs[row];
            if (input < 0 || input >= static_cast<ll>(public_outputs.size()) || outputs[row] != public_outputs[static_cast<std::size_t>(input)])
            {
                throw std::out_of_range(
                    "mapped lookup witness does not match its public table");
            }
        }
        request.queries = std::move(inputs);
        request.query_outputs = std::move(outputs);
        if (expected_inputs.empty() != expected_outputs.empty())
        {
            throw std::invalid_argument(
                "mapped lookup binding requires both committed columns");
        }
        if (!expected_inputs.empty())
        {
            const auto commitment_start = std::chrono::steady_clock::now();
            request.expected_commitments = combine_commitments(
                expected_inputs, expected_outputs,
                tuple_compression_challenge, threads);
            request.precommitted_query_time = std::chrono::duration<double>(
                                                  std::chrono::steady_clock::now() - commitment_start)
                                                  .count();
        }
        return request;
    }

    DecodeLookupResult prove_requests(
        std::vector<DecodeLookupRequest> requests,
        int threads,
        const MVKZG_ProverKey &pk,
        const MVKZG_VerifierKey &vk,
        LookupBackend backend,
        G1WireEncoding encoding,
        QueryCommitmentRole query_role,
        bool preserve_group_commitments = false)
    {
        if (requests.empty())
            return {};

        LookupMembershipWitness witness;
        witness.groups.reserve(requests.size());
        witness.group_queries.reserve(requests.size());
        witness.group_query_outputs.reserve(requests.size());
        witness.tuple_compression_challenges.reserve(requests.size());
        witness.group_query_commitments.reserve(requests.size());
        bool commitment_linked = preserve_group_commitments;
        for (std::size_t index = 0; index < requests.size(); ++index)
        {
            DecodeLookupRequest &request = requests[index];
            if (request.queries.empty() || request.table.empty())
            {
                throw std::invalid_argument("empty zkDecode lookup request");
            }
            if (query_role == QueryCommitmentRole::InterfaceQuery &&
                request.expected_commitments.empty())
            {
                throw std::invalid_argument("excluded query commitment must be supplied at the lookup interface");
            }
            commitment_linked = commitment_linked || !request.expected_commitments.empty();
            const bool tuple_compressed = !request.table_outputs.empty();
            const ll padding_input = request.table.front();
            const ll padding_output = tuple_compressed
                                          ? request.table_outputs.front()
                                          : 0;
            witness.groups.push_back(tuple_compressed
                                         ? LookupGroupSpec::randomTuple(
                                               std::move(request.id),
                                               std::move(request.table_id),
                                               std::move(request.table),
                                               std::move(request.table_outputs),
                                               request.queries.size(), padding_input)
                                         : LookupGroupSpec(
                                               std::move(request.id),
                                               std::move(request.table_id),
                                               std::move(request.table),
                                               request.queries.size(), padding_input));
            request.queries.resize(
                witness.groups.back().paddedCount(), padding_input);
            witness.group_queries.push_back(std::move(request.queries));
            if (tuple_compressed)
            {
                request.query_outputs.resize(
                    witness.groups.back().paddedCount(),
                    padding_output);
                witness.group_query_outputs.push_back(
                    std::move(request.query_outputs));
                witness.tuple_compression_challenges.push_back(
                    request.tuple_compression_challenge);
            }
            else
            {
                witness.group_query_outputs.emplace_back();
                witness.tuple_compression_challenges.emplace_back(Fr(0));
            }
            witness.group_query_commitments.push_back(
                request.expected_commitments);
        }

        LookupBundleProof bundle = proveLookupBundle(
            witness, pk, vk, threads, backend, commitment_linked ? 1 : 2, encoding);

        DecodeLookupResult result;
        const bool generated_queries = query_role == QueryCommitmentRole::GeneratedQueries;
        result.prover_time = generated_queries
                                 ? bundle.prover_time
                                 : bundle.proverTimeWithPrecommittedQueries();
        result.setup_time = bundle.table_preprocess_time + bundle.witness_preparation_time;
        if (!generated_queries)
        {
            result.setup_time += bundle.query_commitment_time;
            for (const DecodeLookupRequest &request : requests)
                result.setup_time += request.precommitted_query_time;
        }
        result.proof_records = std::move(bundle.groups);
        return result;
    }

}

namespace zkdecode_detail
{
    std::vector<LookupBatch> lookupPlan(const ZkDecodeParameters &p)
    {
        const auto cells = static_cast<std::size_t>(zkagent_integer::next_power_of_two(p.token_count)) *
                           zkagent_integer::next_power_of_two(p.vocab_size);
        const int candidate_bits = decodeRangeBits(2 * p.bound);
        if (p.kind == ZkDecodeProofKind::Greedy)
            return {{LookupBatchKind::GreedyRange, cells, candidate_bits}};
        return {
            {LookupBatchKind::CandidateIn, cells, candidate_bits},
            {LookupBatchKind::CandidateOut, cells, candidate_bits},
            {LookupBatchKind::ExpDigits, cells, 0},
        };
    }

    std::vector<LookupGroup> lookupGroups(const LookupBatch &batch)
    {
        if (batch.value_count == 0 || batch.value_count > (std::size_t(1) << 29))
            throw std::invalid_argument("invalid decode lookup domain");
        const int query_count = zkagent_integer::next_power_of_two(static_cast<int>(batch.value_count));
        std::vector<LookupGroup> groups;
        if (batch.kind == LookupBatchKind::ExpDigits)
        {
            for (int digit = 0; digit < static_cast<int>(DecodeExpDigitBases.size()); ++digit)
                groups.push_back({static_cast<std::size_t>(digit), batch.value_count, query_count,
                                  decode_exp_digit_bits(DecodeExpDigitBases[digit]), 0, digit});
        }
        else
        {
            const int bits = std::max(batch.range_bits, 1);
            if (bits >= 127)
                throw std::invalid_argument("decode range width is too large");
            for (int shift = 0; shift < bits; shift += RangeLimbBits)
                groups.push_back({groups.size(), batch.value_count, query_count,
                                  std::min(RangeLimbBits, bits - shift), shift, -1});
        }
        return groups;
    }

    const std::vector<ll> &lookupTableInputs(const LookupGroup &group)
    {
        if (group.digit >= 0)
            return decode_exp_digits_public_tables(DecodeExpDigitsConfig{}).inputs.at(group.digit);
        static const std::array<std::vector<ll>, RangeLimbBits + 1> ranges = []
        {
            std::array<std::vector<ll>, RangeLimbBits + 1> tables;
            for (int bits = 0; bits <= RangeLimbBits; ++bits)
            {
                auto &table = tables[bits];
                table.resize(std::size_t(1) << bits);
                for (std::size_t i = 0; i < table.size(); ++i)
                    table[i] = i;
            }
            return tables;
        }();
        return ranges.at(group.table_bits);
    }

    const std::vector<ll> *lookupTableOutputs(const LookupGroup &group)
    {
        if (group.digit < DecodeExpDigitsConfig{}.nonlinear_segment)
            return nullptr;
        return &decode_exp_digits_public_tables(DecodeExpDigitsConfig{}).outputs.at(group.digit);
    }

    const std::vector<G1> &rangeCommitment(const ZkDecodeProof &proof, LookupBatchKind source)
    {
        switch (source)
        {
        case LookupBatchKind::GreedyRange:
            return decode_greedy_negative_delta_commitment(proof.greedy_link);
        case LookupBatchKind::CandidateIn:
            return decode_topk_column_commitment(proof.topk_link, DecodeTopKLinkColumn::CandidateIn);
        case LookupBatchKind::CandidateOut:
            return decode_topk_column_commitment(proof.topk_link, DecodeTopKLinkColumn::CandidateOut);
        case LookupBatchKind::ExpDigits:
            throw std::invalid_argument("exp lookup has one commitment per digit");
        }
        throw std::invalid_argument("unknown decode lookup source");
    }

    bool verifyLookupBindings(const std::vector<LookupBatch> &plan, const ZkDecodeProof &proof,
                              const std::vector<LookupReadClaim> &claims, int threads)
    {
        std::size_t cursor = 0;
        for (const LookupBatch &batch : plan)
        {
            const auto groups = lookupGroups(batch);
            if (cursor + groups.size() > claims.size())
                return false;
            if (batch.kind != LookupBatchKind::ExpDigits)
            {
                const auto &expected = rangeCommitment(proof, batch.kind);
                std::vector<G1> reconstructed(expected.size());
                for (auto &point : reconstructed)
                    point.clear();
                for (const auto &group : groups)
                {
                    const auto &claim = claims[cursor++];
                    if (claim.commitments.size() != expected.size())
                        return false;
                    const Fr weight = zkagent_integer::fr_from_i128(ll(1) << group.bit_shift);
                    for (std::size_t row = 0; row < expected.size(); ++row)
                        reconstructed[row] += claim.commitments[row] * weight;
                }
                if (reconstructed != expected)
                    return false;
            }
            else
            {
                const auto column = [&](DecodeTopKLinkColumn c) -> const std::vector<G1> &
                { return decode_topk_column_commitment(proof.topk_link, c); };
                const Fr rho = decode_topk_tuple_challenge(proof.topk_link,
                    proof.parameters.token_count, proof.parameters.vocab_size, proof.parameters.k);
                for (const auto &group : groups)
                {
                    const auto &input = column(static_cast<DecodeTopKLinkColumn>(
                        static_cast<int>(DecodeTopKLinkColumn::Digit0) + group.digit));
                    const auto &claim = claims[cursor++];
                    if (!lookupTableOutputs(group))
                    {
                        if (claim.commitments != input)
                            return false;
                    }
                    else
                    {
                        const auto output = group.digit == 3 ? DecodeTopKLinkColumn::MiddleOutput
                                            : group.digit == 4 ? DecodeTopKLinkColumn::HighZero0
                                                               : DecodeTopKLinkColumn::HighZero1;
                        if (claim.commitments != combine_commitments(input, column(output), rho, threads))
                            return false;
                    }
                }
            }
        }
        return cursor == claims.size();
    }
}

int decodeRangeBits(ll upper_exclusive)
{
    if (upper_exclusive <= 1)
        return 1;
    int bits = 0;
    ll current = 1;
    while (current < upper_exclusive)
    {
        if (bits >= 126)
        {
            throw std::overflow_error("decode range width exceeds 126 bits");
        }
        current <<= 1;
        ++bits;
    }
    return std::max(bits, 1);
}

DecodeLookupResult proveDecodeRangeLookup(
    const std::vector<ll> &values,
    int range_bits,
    int threads,
    const MVKZG_ProverKey &pk,
    const MVKZG_VerifierKey &vk,
    LookupBackend backend,
    const std::vector<G1> &expected_value_commitments,
    G1WireEncoding encoding)
{
    if (values.empty())
        return {};
    const int bits = std::max(range_bits, 1);
    if (bits >= 127)
    {
        throw std::invalid_argument("decode range width is too large");
    }
    const ll upper = static_cast<ll>(1) << bits;
    for (ll value : values)
    {
        if (value < 0 || value >= upper)
        {
            throw std::out_of_range("decode range value is outside its width");
        }
    }

    const auto groups = zkdecode_detail::lookupGroups(
        {zkdecode_detail::LookupBatchKind::GreedyRange, values.size(), bits});
    std::vector<DecodeLookupRequest> requests;
    requests.reserve(groups.size());
    for (const auto &group : groups)
    {
        const ll mask = (ll(1) << group.table_bits) - 1;
        std::vector<ll> queries(values.size());
        for (std::size_t row = 0; row < values.size(); ++row)
            queries[row] = (values[row] >> group.bit_shift) & mask;
        const std::string id = group.table_bits == RangeLimbBits
                                  ? "decode/range/limb/" + std::to_string(group.group_index)
                                  : "decode/range/tail";
        requests.push_back(identity_request(id, std::move(queries), group.table_bits, {},
            &zkdecode_detail::lookupTableInputs(group)));
    }
    const bool reuse_original_query = bits <= RangeLimbBits;
    if (reuse_original_query)
        requests.front().expected_commitments = expected_value_commitments;
    DecodeLookupResult result = prove_requests(
        std::move(requests), threads, pk, vk, backend, encoding,
        reuse_original_query ? QueryCommitmentRole::InterfaceQuery : QueryCommitmentRole::GeneratedQueries,
        true);
    return result;
}

DecodeLookupResult proveDecodeExpDigitLookups(
    DecodeExpDigitsWitness witness,
    const Fr &tuple_compression_challenge,
    int threads,
    const MVKZG_ProverKey &pk,
    const MVKZG_VerifierKey &vk,
    LookupBackend backend,
    const DecodeTopKLinkProof &link_proof,
    G1WireEncoding encoding)
{
    if (!witness.success)
    {
        throw std::invalid_argument("cannot prove an invalid digit witness");
    }
    const DecodeExpDigitsConfig &config = witness.config;
    const auto &public_tables = decode_exp_digits_public_tables(config);
    std::vector<DecodeLookupRequest> requests;
    const auto groups = zkdecode_detail::lookupGroups(
        {zkdecode_detail::LookupBatchKind::ExpDigits, witness.digits.front().size(), 0});
    for (const auto &group : groups)
    {
        const int digit = group.digit;
        const auto input_column = static_cast<DecodeTopKLinkColumn>(
            static_cast<int>(DecodeTopKLinkColumn::Digit0) + digit);
        const auto &input_commitment = decode_topk_column_commitment(link_proof, input_column);
        if (!zkdecode_detail::lookupTableOutputs(group))
        {
            requests.push_back(identity_request(
                "decode/exp/digit/" + std::to_string(digit), std::move(witness.digits[digit]),
                group.table_bits, input_commitment, &public_tables.inputs[digit]));
            continue;
        }
        const bool middle = digit == config.nonlinear_segment;
        const int high = digit - config.nonlinear_segment - 1;
        const auto output_column = middle ? DecodeTopKLinkColumn::MiddleOutput
                                   : high == 0 ? DecodeTopKLinkColumn::HighZero0
                                               : DecodeTopKLinkColumn::HighZero1;
        requests.push_back(mapped_request(
            middle ? "decode/exp/middle" : "decode/exp/high/" + std::to_string(high),
            middle ? "decode/exp/middle/" + std::to_string(static_cast<long long>(witness.output_scale))
                   : "decode/exp/zero/" + std::to_string(group.table_bits),
            std::move(witness.digits[digit]),
            middle ? std::move(witness.nonlinear_outputs) : std::move(witness.high_zero_flags[high]),
            public_tables.inputs[digit], public_tables.outputs[digit], group.table_bits,
            tuple_compression_challenge, threads, input_commitment,
            decode_topk_column_commitment(link_proof, output_column)));
    }
    return prove_requests(
        std::move(requests), threads, pk, vk, backend, encoding,
        QueryCommitmentRole::InterfaceQuery);
}
