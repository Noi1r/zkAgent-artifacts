#include "zkdecode/zkdecode_proof.hpp"

namespace
{
    constexpr u64 FieldPayloadBytes = 32;

    u64 payload_bytes(const DecodeTopKLinkProof &proof)
    {
        u64 groups = proof.combined_opening.eval_proof.q_comm.size() +
                     proof.shift_opening.eval_proof.q_comm.size();
        // Input-linked proofs leave the external logit commitment empty.
        for (const auto &column : proof.column_commitments)
            groups += column.size();
        groups += proof.shift_commitments.size();
        groups += proof.separator_commitments.size();
        const u64 fields = proof.sumcheck_rounds.size() * (DecodeTopKLinkDegree + 1) +
                           DecodeTopKLinkColumns + 2;
        return groups * G1::getSerializedByteSize() + fields * FieldPayloadBytes;
    }

    u64 payload_bytes(const DecodeGreedyLinkProof &proof)
    {
        u64 groups = proof.combined_opening.eval_proof.q_comm.size() +
                     proof.lambda_opening.eval_proof.q_comm.size() +
                     proof.lambda_commitments.size();
        for (const auto &column : proof.column_commitments)
            groups += column.size();
        return groups * G1::getSerializedByteSize() +
               (DecodeGreedyLinkColumns + 1) * FieldPayloadBytes;
    }

    u64 payload_bytes(const DecodeSelectorProof &proof, bool include_shared_commitments)
    {
        const u64 groups = proof.probability_opening.eval_proof.q_comm.size() +
                           (include_shared_commitments ? proof.probability_commitments.size() : 0);
        const u64 fields = proof.total_mass_claims.size() + proof.prefix_claims.size() +
                           proof.next_prefix_claims.size() + proof.sumcheck_rounds.size() * 3 + 1;
        return groups * G1::getSerializedByteSize() + fields * FieldPayloadBytes;
    }

    u64 payload_bytes(const DecodeBatchPointZeroProof &proof, bool include_shared_commitments)
    {
        const u64 groups = proof.opening.eval_proof.q_comm.size() +
                           (include_shared_commitments ? proof.value_commitments.size() : 0);
        const u64 fields = proof.sumcheck_rounds.size() * 3 + 1;
        return groups * G1::getSerializedByteSize() + fields * FieldPayloadBytes;
    }
}

u64 zkDecodeProofPayloadBytes(const ZkDecodeProof &proof)
{
    u64 bytes = 0;
    if (proof.has_greedy_link)
        bytes += payload_bytes(proof.greedy_link);
    if (proof.has_greedy_zero)
        bytes += payload_bytes(proof.greedy_zero, !proof.has_greedy_link);
    if (proof.has_topk_link)
        bytes += payload_bytes(proof.topk_link);
    if (proof.has_selector)
        bytes += payload_bytes(proof.selector, !proof.has_topk_link);
    for (const auto &record : proof.lookup_proofs)
        bytes += record.proof_size_bytes;
    return bytes;
}
