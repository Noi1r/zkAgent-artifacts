#ifndef ZKAGENT_LOGUP_GKR_HPP
#define ZKAGENT_LOGUP_GKR_HPP

#include "common/global_var.hpp"
#include "common/fiat_shamir.hpp"
#include "gkr/circuit.h"
#include "common/polynomial.h"
#include "pcs/hyrax.hpp"
#include "lookup/lookup_common.hpp"

struct MVKZG_ProverKey;
struct MVKZG_VerifierKey;
constexpr int LogupGkrMaxVariables = 30;

struct SumcheckGkrGruenProof
{
    std::vector<Fr> round_polys;
};

struct LogupGkrLayerProof
{
    SumcheckGkrGruenProof sumcheck;
    Fr v_N0, v_N1, v_D0, v_D1;
};

enum class LogupGkrMode : std::uint64_t
{
    Values = 1,
};

struct LogupGkrProof
{
    LogupGkrMode mode = LogupGkrMode::Values;
    int m = 0, n = 0;
    int log_m = 0, log_n = 0;

    std::vector<G1> f_combined_comm;
    std::vector<G1> c_comm;
    std::vector<G1> t_combined_comm;

    Fr query_N_root, query_D_root;
    std::vector<LogupGkrLayerProof> query_layers;

    Fr table_N_root, table_D_root;
    std::vector<LogupGkrLayerProof> table_layers;

    hyrax::OpeningProof open_f_combined;
    hyrax::OpeningProof open_c;
    hyrax::OpeningProof open_t_combined;

    std::vector<std::uint8_t> wire_bytes;
    G1WireEncoding g1_encoding = G1WireEncoding::Affine64;
};

struct LogupGkrTableCommitment
{
    LogupGkrMode mode = LogupGkrMode::Values;
    int padded_n = 0;
    int log_n = 0;
    std::vector<G1> value_commitments;
};

struct LogupGkrProveTiming
{
    double query_commitment_ms = 0.0;
    double multiplicity_commitment_ms = 0.0;
    double table_commitment_ms = 0.0;
    double tree_build_ms = 0.0;
    double sumcheck_ms = 0.0;
    double opening_ms = 0.0;
    double total_ms = 0.0;
};

struct LogupGkrPreparedValueTable
{
    int padded_n = 0;
    int log_n = 0;
    // Trusted caller-owned table; keep it alive and immutable during proving.
    const std::vector<Fr> *values = nullptr;
};

struct LogupGkrPreparedValueWitness
{
    int padded_m = 0;
    int log_m = 0;
    std::vector<Fr> values;
    std::vector<ll> multiplicities;
};

class LogupGkrProver
{
public:
    LogupGkrProver(const MVKZG_ProverKey &pk,
                 const MVKZG_VerifierKey &vk)
        : pk_(pk), vk_(vk) {}

    G1 g[1 << (LogupGkrMaxVariables / 2)];

    const MVKZG_ProverKey &pk_;
    const MVKZG_VerifierKey &vk_;
    void init(std::size_t generator_count = std::size_t(1) << (LogupGkrMaxVariables / 2));

    LogupGkrProof logup_gkr_prove_prepared_values(
        const LogupGkrPreparedValueWitness &query,
        const LogupGkrPreparedValueTable &table,
        int thread_num,
        G1WireEncoding encoding = G1WireEncoding::Affine64,
        LogupGkrProveTiming *timing = nullptr,
        const LogupGkrTableCommitment *preprocessed_table = nullptr,
        bool defer_query_opening = false,
        bool retain_wire_bytes = true,
        FiatShamirTranscript *shared_transcript = nullptr,
        const std::vector<G1> *precommitted_query = nullptr);
    LogupGkrTableCommitment logup_gkr_commit_public_table(
        const ll *table, int n, int thread_num, LogupGkrMode mode);
    LogupGkrTableCommitment logup_gkr_commit_public_table(
        const Fr *table, int n, int thread_num, LogupGkrMode mode);
    bool logup_gkr_verify_in_memory_with_table(
        const LogupGkrProof &proof,
        const LogupGkrTableCommitment &table_commitment,
        LookupReadClaim *read_claim = nullptr,
        bool defer_query_opening = false,
        FiatShamirTranscript *shared_transcript = nullptr,
        hyrax::OpeningBatchVerifier *opening_batch = nullptr) const;
};
bool logup_gkr_verify_serialized(
    const std::vector<std::uint8_t> &wire_bytes,
    int expected_query_count,
    const LogupGkrTableCommitment &table_commitment,
    const MVKZG_VerifierKey &vk,
    LookupReadClaim *read_claim = nullptr,
    hyrax::OpeningBatchVerifier *opening_batch = nullptr);
bool logup_gkr_verify_in_memory(
    const LogupGkrProof &, const LogupGkrTableCommitment &,
    const MVKZG_VerifierKey &, LookupReadClaim *read_claim = nullptr,
    hyrax::OpeningBatchVerifier *opening_batch = nullptr);
// Compressed cryptographic payload, excluding wire framing and public metadata.
u64 logup_gkr_proof_size_bytes(const LogupGkrProof &proof);

#endif
