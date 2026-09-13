#ifndef ZKAGENT_LOOKUP_BUNDLE_HPP
#define ZKAGENT_LOOKUP_BUNDLE_HPP

#include "common/fiat_shamir.hpp"
#include "lookup/lookup_common.hpp"
#include "lookup/lookup_witness.hpp"

#include <cstdint>
#include <vector>
#include <memory>

struct MVKZG_ProverKey;
struct MVKZG_VerifierKey;
class FiatShamirTranscript;
struct ShoutProof;
struct LogupGkrProof;

enum class LookupBackend
{
    Shout,
    LogUpGkr,
};

enum class LookupOpeningVerification
{
    Individual,
    BatchMvKzg,
};

struct LookupMembershipWitness
{
    std::vector<LookupGroupSpec> groups;
    std::vector<std::vector<ll>> group_queries;
    std::vector<std::vector<ll>> group_query_outputs;
    std::vector<Fr> tuple_compression_challenges;
    std::vector<std::vector<G1>> group_query_commitments;
};

struct LookupGroupProverTiming
{
    double total_seconds = 0;
    double query_commit_seconds = 0;
    double auxiliary_commit_seconds = 0;
    double relation_seconds = 0;
    double sumcheck_seconds = 0;
    double opening_seconds = 0;
};

struct LookupGroupProofRecord
{
    std::size_t group_index = 0;
    std::size_t query_base = 0;
    std::size_t actual_query_count = 0;
    int table_size = 0;
    int query_count = 0;
    std::vector<std::uint8_t> wire_bytes;
    u64 proof_size_bytes = 0;
    LookupReadClaim read_claim;
    LookupGroupProverTiming timing;
    std::shared_ptr<const ShoutProof> shout_messages;
    std::shared_ptr<const LogupGkrProof> logup_messages;
};

struct LookupQueryChunkPlan
{
    std::size_t query_base = 0;
    std::size_t actual_count = 0;
    std::size_t padded_count = 0;
};

struct LookupGroupPackingPlan
{
    std::size_t group_index = 0;
    std::size_t table_size = 0;
    std::size_t actual_count = 0;
    std::size_t legacy_padded_count = 0;
    std::vector<LookupQueryChunkPlan> chunks;
};

struct LookupPackingPlan
{
    std::size_t max_chunks_per_group = 1;
    std::size_t proof_chunks = 0;
    std::uint64_t actual_query_slots = 0;
    std::uint64_t legacy_padded_query_slots = 0;
    std::uint64_t packed_query_slots = 0;
    std::uint64_t legacy_weighted_cells = 0;
    std::uint64_t packed_weighted_cells = 0;
    std::vector<LookupGroupPackingPlan> groups;
};

struct LookupBundleProof
{
    LookupBackend backend = LookupBackend::Shout;
    LookupOpeningVerification opening_verification =
        LookupOpeningVerification::Individual;
    bool verified = false;
    bool query_openings_deferred = false;
    double prover_time = 0.0;
    double query_commitment_time = 0.0;
    double auxiliary_commitment_time = 0.0;
    double witness_preparation_time = 0.0;
    double proof_materialization_time = 0.0;
    double relation_preparation_time = 0.0;
    double sumcheck_time = 0.0;
    double opening_prover_time = 0.0;
    double unattributed_prover_time = 0.0;
    double verifier_time = 0.0;
    std::size_t mvkzg_opening_claims = 0;
    std::size_t mvkzg_verification_batches = 0;
    double mvkzg_batch_verify_time = 0.0;
    double table_preprocess_time = 0.0;
    u64 proof_size_bytes = 0;
    LookupPackingPlan packing;
    std::vector<LookupGroupProofRecord> groups;
    std::vector<Fr> tuple_compression_challenges;

    double proverTimeWithPrecommittedQueries() const;
};

LookupBundleProof proveLookupBundle(
    const LookupMembershipWitness &, const MVKZG_ProverKey &,
    const MVKZG_VerifierKey &, int threads, LookupBackend,
    std::size_t max_chunks_per_group = 1,
    G1WireEncoding encoding = G1WireEncoding::Affine64);

LookupPackingPlan planLookupBundlePacking(
    const LookupWitness &witness,
    std::size_t max_chunks_per_group = 1);

LookupPackingPlan planLookupBundlePacking(
    const LookupMembershipWitness &witness,
    std::size_t max_chunks_per_group = 1);

std::size_t lookupBundleRequiredGenerators(
    const LookupWitness &witness,
    LookupBackend backend = LookupBackend::Shout,
    std::size_t max_chunks_per_group = 1);

std::size_t lookupBundleRequiredGenerators(
    const LookupMembershipWitness &witness,
    LookupBackend backend = LookupBackend::Shout,
    std::size_t max_chunks_per_group = 1);

LookupBundleProof proveAndVerifyLookupBundle(
    const LookupWitness &witness,
    const MVKZG_ProverKey &pk,
    const MVKZG_VerifierKey &vk,
    int threads, LookupBackend backend = LookupBackend::Shout,
    bool defer_query_openings = false,
    std::size_t max_chunks_per_group = 1,
    LookupOpeningVerification opening_verification =
        LookupOpeningVerification::Individual,
    FiatShamirTranscript *prover_transcript = nullptr,
    FiatShamirTranscript *verifier_transcript = nullptr,
    G1WireEncoding g1_encoding = G1WireEncoding::Affine64);

LookupBundleProof proveAndVerifyLookupBundle(
    const LookupMembershipWitness &witness,
    const MVKZG_ProverKey &pk,
    const MVKZG_VerifierKey &vk,
    int threads, LookupBackend backend = LookupBackend::Shout,
    std::size_t max_chunks_per_group = 1,
    LookupOpeningVerification opening_verification =
        LookupOpeningVerification::Individual,
    FiatShamirTranscript *prover_transcript = nullptr,
    FiatShamirTranscript *verifier_transcript = nullptr,
    G1WireEncoding g1_encoding = G1WireEncoding::Affine64);

#endif
