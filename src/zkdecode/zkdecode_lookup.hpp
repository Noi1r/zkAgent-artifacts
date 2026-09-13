#ifndef ZKAGENT_ZKDECODE_LOOKUP_HPP
#define ZKAGENT_ZKDECODE_LOOKUP_HPP

#include "lookup/lookup_bundle.hpp"
#include "zkdecode/zkdecode_exp_digits.hpp"
#include "zkdecode/zkdecode_arithmetic.hpp"

#include <vector>

struct ZkDecodeParameters;
struct ZkDecodeProof;

namespace zkdecode_detail {

constexpr int RangeLimbBits = 10;

enum class LookupBatchKind {
    GreedyRange, CandidateIn, CandidateOut, ExpDigits,
};

struct LookupBatch {
    LookupBatchKind kind;
    std::size_t value_count;
    int range_bits = 0;
};

struct LookupGroup {
    std::size_t group_index;
    std::size_t actual_count;
    int query_count;
    int table_bits;
    int bit_shift = 0;
    int digit = -1; // -1 is a range limb; 0..5 selects a fixed exp table.
};

// One ordered plan drives proving, verifier-key preparation and source bindings.
std::vector<LookupBatch> lookupPlan(const ZkDecodeParameters &parameters);
std::vector<LookupGroup> lookupGroups(const LookupBatch &batch);
const std::vector<ll> &lookupTableInputs(const LookupGroup &group);
const std::vector<ll> *lookupTableOutputs(const LookupGroup &group);
const std::vector<G1> &rangeCommitment(const ZkDecodeProof &proof, LookupBatchKind source);
bool verifyLookupBindings(const std::vector<LookupBatch> &plan, const ZkDecodeProof &proof,
                          const std::vector<LookupReadClaim> &claims, int threads);

} // namespace zkdecode_detail

struct DecodeLookupResult {
    double setup_time = 0.0;
    double prover_time = 0.0;
    std::vector<LookupGroupProofRecord> proof_records;
};

int decodeRangeBits(ll upper_exclusive);


DecodeLookupResult proveDecodeRangeLookup(
    const std::vector<ll>& values,
    int range_bits,
    int threads,
    const MVKZG_ProverKey& pk,
    const MVKZG_VerifierKey& vk,
    LookupBackend backend,
    const std::vector<G1>& expected_value_commitments,
    G1WireEncoding encoding = G1WireEncoding::Affine64);

DecodeLookupResult proveDecodeExpDigitLookups(
    DecodeExpDigitsWitness witness,
    const Fr& tuple_compression_challenge,
    int threads,
    const MVKZG_ProverKey& pk,
    const MVKZG_VerifierKey& vk,
    LookupBackend backend,
    const DecodeTopKLinkProof& link_proof,
    G1WireEncoding encoding = G1WireEncoding::Affine64);

#endif
