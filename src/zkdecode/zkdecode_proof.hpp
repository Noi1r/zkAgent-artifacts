#ifndef ZKAGENT_ZKDECODE_PROOF_HPP
#define ZKAGENT_ZKDECODE_PROOF_HPP

#include "lookup/lookup_bundle.hpp"
#include "zkdecode/zkdecode_output.hpp"
#include "zkdecode/zkdecode_arithmetic.hpp"

#include <vector>
#include <string>

enum class ZkDecodeProofKind {
    None,
    Greedy,
    TopK,
};

struct ZkDecodeParameters {
    ZkDecodeProofKind kind = ZkDecodeProofKind::None;
    LookupBackend lookup_backend = LookupBackend::LogUpGkr;
    int token_count = 0;
    int vocab_size = 0;
    int k = 0;
    ll bound = 0;
    ll theta = 0;
    ll epsilon_sm = 0;

    bool operator==(const ZkDecodeParameters& other) const {
        return kind == other.kind && lookup_backend == other.lookup_backend &&
               token_count == other.token_count && vocab_size == other.vocab_size &&
               k == other.k && bound == other.bound && theta == other.theta &&
               epsilon_sm == other.epsilon_sm;
    }
};

struct ZkDecodeStatement {
    ZkDecodeParameters parameters;
    std::vector<G1> logit_commitments;
    std::vector<int> output_tokens;
    std::vector<ll> sample_values;
};

struct ZkDecodeProof {
    ZkDecodeParameters parameters;
    std::vector<int> output_tokens;
    std::vector<ll> sample_values;

    bool has_greedy_link = false;
    DecodeGreedyLinkProof greedy_link;
    bool has_greedy_zero = false;
    DecodeBatchPointZeroProof greedy_zero;
    bool has_topk_link = false;
    DecodeTopKLinkProof topk_link;
    bool has_selector = false;
    DecodeSelectorProof selector;
    std::vector<LookupGroupProofRecord> lookup_proofs;
};

struct DecodeProofStats {
    // Generation status only; verification is a separate call.
    bool success = false;
    // Preserve the benchmark's preparation and online-prover boundaries.
    double setup_time = 0.0;
    double prover_time = 0.0;
    std::string message;
};

struct ZkDecodeProveResult {
    ZkDecodeProof proof;
    DecodeProofStats stats;
};

// Compressed cryptographic payload; shared commitments are counted once.
u64 zkDecodeProofPayloadBytes(const ZkDecodeProof &proof);

#endif
