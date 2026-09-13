#ifndef ZKAGENT_LOOKUP_INPUT_LINK_HPP
#define ZKAGENT_LOOKUP_INPUT_LINK_HPP

#include "gkr/input_reduction_link.hpp"
#include "lookup/lookup_bundle.hpp"

#include <cstdint>
#include <vector>

struct MVKZG_ProverKey;
struct MVKZG_VerifierKey;
class FiatShamirTranscript;

struct LookupInputLinkProof {
    std::vector<std::uint8_t> wire_bytes;
    u64 compressed_payload_bytes = 0;
};

struct LookupInputLinkTiming {
    double statement_seconds = 0;
    double coefficient_seconds = 0;
    double query_value_seconds = 0;
    double round_seconds = 0;
    double fold_seconds = 0;
    double opening_seconds = 0;
    std::size_t equality_tables = 0;
    std::size_t equality_values = 0;
    std::size_t equality_cache_hits = 0;
};

struct LookupInputLinkResult {
    bool verified = false;
    double prover_time = 0.0;
    double verifier_time = 0.0;
    u64 proof_size_bytes = 0;
    std::size_t query_claims = 0;
    std::size_t opening_buckets = 0;
    std::size_t opening_verification_batches = 0;
    std::size_t relation_claims = 0;
    InputReductionLink input_link;
    LookupInputLinkProof proof;
    LookupInputLinkTiming timing;
};

LookupInputLinkProof proveLookupInputLink(
    const LookupWitness& witness,
    const LookupBundleProof& lookup_proof,
    const MVKZG_ProverKey& pk,
    int threads,
    double* prover_time = nullptr,
    LookupInputLinkTiming* timing = nullptr);

LookupInputLinkProof proveLookupInputLink(
    const LookupWitness& witness,
    const LookupBundleProof& lookup_proof,
    const MVKZG_ProverKey& pk,
    int threads,
    FiatShamirTranscript& transcript,
    double* prover_time = nullptr,
    LookupInputLinkTiming* timing = nullptr);

bool verifyLookupInputLink(
    const LookupLayout& layout,
    const LookupBundleProof& lookup_proof,
    const LookupInputLinkProof& proof,
    const MVKZG_VerifierKey& vk,
    int threads,
    InputReductionLink* input_link,
    double* verifier_time = nullptr);

bool verifyLookupInputLink(
    const LookupLayout& layout,
    const LookupBundleProof& lookup_proof,
    const LookupInputLinkProof& proof,
    const MVKZG_VerifierKey& vk,
    int threads,
    InputReductionLink* input_link,
    FiatShamirTranscript& transcript,
    double* verifier_time = nullptr);

LookupInputLinkResult proveAndVerifyLookupInputLink(
    const LookupWitness& witness,
    const LookupBundleProof& lookup_proof,
    const MVKZG_ProverKey& pk,
    const MVKZG_VerifierKey& vk,
    int threads,
    FiatShamirTranscript* prover_transcript = nullptr,
    FiatShamirTranscript* verifier_transcript = nullptr);

#endif
