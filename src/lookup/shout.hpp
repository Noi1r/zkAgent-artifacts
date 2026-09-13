

#ifndef ZKCNN_SHOUT_HPP
#define ZKCNN_SHOUT_HPP

#include "common/global_var.hpp"
#include "common/fiat_shamir.hpp"
#include "common/polynomial.h"
#include "pcs/hyrax.hpp"
#include "lookup/lookup_common.hpp"
#include <memory>
#include <vector>

struct MVKZG_ProverKey;
struct MVKZG_VerifierKey;

struct ShoutProof {

    std::vector<Fr> core_sumcheck_polys;

    std::vector<Fr> booleanity_sumcheck_polys;

    G1* RA_comm = nullptr;
    int RA_comm_size = 0;
    std::vector<G1> read_value_commitments;
    hyrax::OpeningProof read_value_opening;
    hyrax::OpeningProof RA_open_proof_core;
    hyrax::OpeningProof RA_open_proof;

    Fr ra_claim;
    Fr rv_claim;
    Fr RA_eval;

    std::vector<std::uint8_t> wire_bytes;
    G1WireEncoding g1_encoding = G1WireEncoding::Affine64;

};

// Compressed cryptographic payload, excluding wire framing and public metadata.
u64 shout_proof_size_bytes(const ShoutProof& proof);

struct ShoutProveTiming {
    double eq_ms = 0.0;
    double read_poly_ms = 0.0;
    double commit_ra_ms = 0.0;
    double commit_read_values_ms = 0.0;
    double finalize_ra_ms = 0.0;
    double finalize_read_values_ms = 0.0;
    double core_sumcheck_ms = 0.0;
    double booleanity_sumcheck_ms = 0.0;
    double open_read_value_ms = 0.0;
    double open_ra_ms = 0.0;
    double total_ms = 0.0;

    void reset() {
        eq_ms = 0.0;
        read_poly_ms = 0.0;
        commit_ra_ms = 0.0;
        commit_read_values_ms = 0.0;
        finalize_ra_ms = 0.0;
        finalize_read_values_ms = 0.0;
        core_sumcheck_ms = 0.0;
        booleanity_sumcheck_ms = 0.0;
        open_read_value_ms = 0.0;
        open_ra_ms = 0.0;
        total_ms = 0.0;
    }
};

struct ShoutVerifierChallenges {
    std::vector<Fr> r_cycle;
    std::vector<Fr> r_cycle_bool;
    std::vector<Fr> r_address_bool;
    std::vector<Fr> core_round_challenges;
    std::vector<Fr> booleanity_round_challenges;
    Fr z = Fr(0);
};

class ShoutProver {
public:
    ShoutProver(G1* generators, const MVKZG_ProverKey& pk)
        : g(generators), pk_(pk) {}

    ShoutProof prove_prepared_values(
        const std::vector<Fr>& lookup_table,
        const std::vector<int>& read_addresses,
        const std::vector<Fr>& read_values,
        int K,
        int T,
        int num_threads,
        ShoutProveTiming* timing,
        G1WireEncoding encoding,
        bool defer_read_value_opening,
        bool retain_wire_bytes,
        FiatShamirTranscript* shared_transcript = nullptr,
        const std::vector<G1>* precommitted_read_values = nullptr
    );

private:
    G1* g;
    const MVKZG_ProverKey& pk_;

    std::vector<Fr> compute_read_address_poly(
        const std::vector<int>& read_addresses,
        const std::vector<Fr>& E,
        int K,
        int num_threads
    );

    void prove_core_piop_hamming(
        std::vector<Fr>& F_poly,
        std::vector<Fr>& val_poly,
        Fr z,
        Fr& rv_claim,
        std::vector<Fr>& r_address,
        std::vector<Fr>& compressed_polys,
        int num_rounds,
        int num_threads,
        FiatShamirProofWriter& transcript
    );

    void prove_booleanity(
        const std::vector<int>& read_addresses,
        const std::vector<Fr>& r_address,
        std::vector<Fr>& D,
        std::vector<Fr>& G,
        Fr& ra_claim_prime,
        std::vector<Fr>& r_address_prime,
        std::vector<Fr>& compressed_polys,
        int K,
        int T,
        int num_threads,
        FiatShamirProofWriter& transcript
    );

    void bind_polynomial(
        std::vector<Fr>& poly,
        Fr r_j,
        int num_threads
    );

    void ensure_local_F_workspace(int num_threads, int K);

    std::vector<std::vector<Fr>> local_F_buffers_;
    std::vector<std::vector<unsigned char>> local_F_marks_;
    std::vector<std::vector<int>> local_F_touched_;
    int local_F_threads_ = 0;
    int local_F_K_ = 0;
    std::vector<Fr> bind_scratch_;
};

class ShoutVerifier {
public:
    explicit ShoutVerifier(const MVKZG_VerifierKey& vk) : vk_(vk) {}

    bool verify_serialized(
        const std::vector<std::uint8_t>& wire_bytes,
        const std::vector<Fr>& lookup_table,
        int K,
        int T,
        std::vector<G1>* parsed_read_value_commitments = nullptr,
        LookupReadClaim* read_claim = nullptr,
        bool defer_read_value_opening = false,
        hyrax::OpeningBatchVerifier* opening_batch = nullptr,
        FiatShamirTranscript* shared_transcript = nullptr);

    bool verify_in_memory(
        const ShoutProof& proof,
        const std::vector<Fr>& lookup_table,
        int K,
        int T,
        LookupReadClaim* read_claim = nullptr,
        bool defer_read_value_opening = false,
        hyrax::OpeningBatchVerifier* opening_batch = nullptr,
        FiatShamirTranscript* shared_transcript = nullptr);
private:
    const MVKZG_VerifierKey& vk_;

    bool verify_proof_object_impl(
        const ShoutProof& proof,
        const ShoutVerifierChallenges& challenges,
        const std::vector<Fr>& lookup_table,
        int K,
        int T,
        LookupReadClaim* read_claim,
        bool defer_read_value_opening,
        hyrax::OpeningBatchVerifier* opening_batch,
        long long transcript_time_us);
};

#endif
