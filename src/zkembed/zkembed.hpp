#ifndef ZKAGENT_ZKEMBED_HPP
#define ZKAGENT_ZKEMBED_HPP

#include "pcs/srs.hpp"
#include "common/typedef.hpp"
#include "pcs/hyrax.hpp"
#include "gkr/input_reduction_link.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class FiatShamirTranscript;
class ZkEmbedFFTPlan;

using namespace mcl::bn;

struct ZkEmbedTiming
{
    long long compute_rlc_evals_ms = 0;
    long long commit_m_ms = 0;
    long long commit_a_ms = 0;
    long long commit_qa_ms = 0;
    long long commit_b_hyrax_ms = 0;
    long long sumcheck_ms = 0;
    long long hyrax_open_ms = 0;
    long long g2_rlc_ms = 0;
    long long lazy_load_ms = 0;
    std::size_t lazy_loaded_g1_points = 0;
    long long total_prove_ms = 0;
    long long prove_compute_ms = 0;

    long long verify_a_wellform_us = 0;
    long long verify_a0_us = 0;
    long long verify_sumcheck_us = 0;
    long long verify_g2_rlc_us = 0;
    long long total_verify_us = 0;

    void print(const std::string &scheme_name)
    {
        std::cout << "\n┌─────────────────────────────────────────────────────────────┐" << std::endl;
        std::cout << "│  " << std::left << std::setw(55) << (scheme_name + " Detailed Timing") << "│" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│  PROVE:                                                     │" << std::endl;
        std::cout << "│    Fold rows and compute reciprocals:" << std::setw(8) << compute_rlc_evals_ms << " ms            │" << std::endl;
        std::cout << "│    Commit multiplicities:           " << std::setw(8) << commit_m_ms << " ms            │" << std::endl;
        std::cout << "│    Commit table reciprocals:        " << std::setw(8) << commit_a_ms << " ms            │" << std::endl;
        std::cout << "│    Commit table quotient:           " << std::setw(8) << commit_qa_ms << " ms            │" << std::endl;
        std::cout << "│    Commit query witnesses:          " << std::setw(8) << commit_b_hyrax_ms << " ms            │" << std::endl;
        std::cout << "│    Sumcheck:                        " << std::setw(8) << sumcheck_ms << " ms            │" << std::endl;
        std::cout << "│    Commitment openings:             " << std::setw(8) << hyrax_open_ms << " ms            │" << std::endl;
        if (g2_rlc_ms > 0)
        {
            std::cout << "│    Fold table commitments:          " << std::setw(8) << g2_rlc_ms << " ms            │" << std::endl;
        }
        std::cout << "│    ────────────────────────────────────────                │" << std::endl;
        std::cout << "│    TOTAL PROVE:                     " << std::setw(8) << total_prove_ms << " ms            │" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│  VERIFY:                                                    │" << std::endl;
        std::cout << "│    Table reciprocal relation:       " << std::setw(8) << verify_a_wellform_us << " us            │" << std::endl;
        std::cout << "│    Grand-sum opening:               " << std::setw(8) << verify_a0_us << " us            │" << std::endl;
        std::cout << "│    Sumcheck + Hyrax opening:        " << std::setw(8) << verify_sumcheck_us << " us            │" << std::endl;
        if (verify_g2_rlc_us > 0)
        {
            std::cout << "│    Fold table commitments:          " << std::setw(8) << verify_g2_rlc_us << " us            │" << std::endl;
        }
        std::cout << "│    ────────────────────────────────────────                │" << std::endl;
        std::cout << "│    TOTAL VERIFY:                    " << std::setw(8) << total_verify_us << " us            │" << std::endl;
        std::cout << "└─────────────────────────────────────────────────────────────┘" << std::endl;
    }
};

struct SumcheckRoundPoly
{
    Fr coeffs[4];

    Fr evaluate(const Fr &x) const
    {

        Fr result = coeffs[3];
        Fr::mul(result, result, x);
        Fr::add(result, result, coeffs[2]);
        Fr::mul(result, result, x);
        Fr::add(result, result, coeffs[1]);
        Fr::mul(result, result, x);
        Fr::add(result, result, coeffs[0]);
        return result;
    }
};

class ZkEmbedOutputCommitment
{
public:
    static ZkEmbedOutputCommitment tensor(
        int variables, std::vector<G1> rows);
    static ZkEmbedOutputCommitment inputPrefix(
        int input_variables, std::vector<G1> input_rows,
        std::vector<G1> position_rows);

    int variables() const { return variables_; }
    bool isInputView() const { return input_view_; }
    // Standalone tensor rows; input views retain inputRows() and positionRows().
    const std::vector<G1> &rows() const { return rows_; }
    const std::vector<G1> &inputRows() const { return input_rows_; }
    const std::vector<G1> &positionRows() const { return position_rows_; }

private:
    int variables_ = -1;
    bool input_view_ = false;
    std::vector<G1> rows_;
    std::vector<G1> input_rows_;
    std::vector<G1> position_rows_;
};

struct ZkEmbedStatement
{
    std::vector<ll> query_indices;
    ZkEmbedOutputCommitment h0;
};

struct ZkEmbedDebugTrace
{
    Fr beta;
    Fr column_folding_challenge;
    Fr batching_challenge;
    std::vector<Fr> zero_check_point;
    std::vector<Fr> outer_sumcheck_challenges;
    std::vector<Fr> inner_sumcheck_challenges;
};

struct ZkEmbedProof
{
    G1 multiplicity_commitment;
    G1 table_reciprocal_commitment;
    G1 table_quotient_commitment;
    G1 table_zero_opening_commitment;
    Fr table_reciprocal_at_zero;

    std::vector<G1> query_reciprocal_commitments;

    std::vector<SumcheckRoundPoly> outer_sumcheck_polys;
    Fr query_reciprocal_eval;
    hyrax::OpeningProof query_reciprocal_opening;

    Fr h0_eval = Fr(0);
    hyrax::OpeningProof h0_opening;

    std::vector<SumcheckRoundPoly> inner_sumcheck_polys;
    Fr folded_h0_claim = Fr(0);

    std::size_t cryptographicPayloadBytes() const;
};

struct ZkEmbedOpeningClaim
{
    std::vector<Fr> point;
    Fr evaluation;
};

class ZkEmbed
{
public:
    static int requiredTableSize(int vocab_size);
    int embeddingDimension() const { return embed_dim; }
    int threadCount() const { return thread_num; }

    ZkEmbed(KZG_ProverKey &uni_pk, KZG_VerifierKey &uni_vk,
            MVKZG_ProverKey &mv_pk, const MVKZG_VerifierKey &mv_vk,
            int vocab_size, int embed_dim,
            int thread_num = 32);
    ~ZkEmbed();

    // When requested, return compute time for the caller to report instead of printing it.
    void preprocess(const ll *embedding_matrix, long long *preprocess_time_ms = nullptr,
                    long long *cache_save_ms = nullptr);

    std::vector<Fr> prepareWitness(
        std::size_t query_size, const std::vector<ll> &output_rows) const;

    ZkEmbedStatement makeStatement(
        const std::vector<ll> &query_indices,
        const std::vector<Fr> &h0_values) const;

    ZkEmbedProof prove(const ZkEmbedStatement &statement,
                       const std::vector<Fr> &h0_values,
                       ZkEmbedTiming &timing,
                       ZkEmbedDebugTrace *debug_trace = nullptr);
    ZkEmbedProof prove(const ZkEmbedStatement &statement,
                       const std::vector<Fr> &h0_values,
                       FiatShamirTranscript &transcript,
                       ZkEmbedTiming &timing,
                       ZkEmbedDebugTrace *debug_trace = nullptr);

    bool verify(const ZkEmbedStatement &statement,
                const ZkEmbedProof &proof,
                ZkEmbedTiming &timing,
                ZkEmbedDebugTrace *debug_trace = nullptr);

    ZkEmbedProof proveToInput(
        const ZkEmbedStatement &statement, const std::vector<Fr> &h0_values,
        FiatShamirTranscript &transcript, ZkEmbedTiming &timing,
        ZkEmbedDebugTrace *debug_trace = nullptr);
    std::unique_ptr<ZkEmbedOpeningClaim> reduceToInput(
        const ZkEmbedStatement &statement, const ZkEmbedProof &proof,
        FiatShamirTranscript &transcript, ZkEmbedTiming &timing,
        ZkEmbedDebugTrace *debug_trace = nullptr);
    bool verify(const ZkEmbedStatement &statement,
                const ZkEmbedProof &proof,
                FiatShamirTranscript &transcript,
                ZkEmbedTiming &timing,
                ZkEmbedDebugTrace *debug_trace = nullptr);

    // Required before prove/proveToInput; the working set must cover their queries and padding.
    void preloadProverDataForQueries(const std::vector<ll> &query_indices);
    std::size_t prefetchedProverRowCount() const
    {
        return prefetched_prover_rows_.indices.size();
    }
    void preloadVerifierData();

    static std::string cacheFilename(int vocab_size, int embed_dim);

private:
    int vocab_size;
    int embed_dim;
    int thread_num;
    int table_domain_size;

    KZG_ProverKey &uni_pk_;
    KZG_VerifierKey &uni_vk_;
    MVKZG_ProverKey &mv_pk_;
    const MVKZG_VerifierKey &mv_vk_;
    Fr omega;
    std::vector<Fr> omega_powers;

    std::vector<G1> com1_li_x;
    std::vector<G1> com1_li_0;
    G2 com2_vanishing;
    G2 com2_idx;

    std::vector<Fr> dft_w_vec;
    std::vector<Fr> scaled_dft_w_prime_vec;
    std::vector<G1> phi_i_i;

    std::vector<G2> com2_columns;
    std::vector<std::vector<G1>> com1_qi_columns;
    std::vector<G1> com1_qi_idx;

    ll *embedding_matrix_copy;

    int cache_fd;
    static constexpr size_t G1_SIZE = 96;
    static constexpr size_t G2_SIZE = 192;

    size_t offset_com1_li_x;
    size_t offset_com1_li_0;
    size_t offset_com2_vanishing;
    size_t offset_com2_idx;
    size_t offset_com2_columns;
    size_t offset_com1_qi_columns;
    size_t offset_com1_qi_idx;
    bool com2_loaded;

    struct PrefetchedProverRows
    {
        std::vector<int> indices;
        std::vector<G1> li_x;
        std::vector<G1> li_0;
        std::vector<G1> qi_idx;
        std::vector<std::vector<G1>> qi_columns;
    };
    PrefetchedProverRows prefetched_prover_rows_;

    void compute_shared_preprocess(const ZkEmbedFFTPlan &plan);
    void compute_w_vectors(const ZkEmbedFFTPlan &plan);
    void compute_index_data(const ZkEmbedFFTPlan &plan);
    void compute_column_data(const ll *embedding_matrix, const ZkEmbedFFTPlan &plan);

    bool save_cache(const std::string &filename);
    bool load_cache(const std::string &filename);

    void ensure_com2_loaded();
    std::vector<G1> read_g1_rows(size_t offset, const std::vector<int> &row_indices);
    std::vector<G1> read_qi_idx_rows(const std::vector<int> &row_indices);
    void batch_read_all_qi_columns(const std::vector<int> &row_indices,
                                   std::vector<std::vector<G1>> &out_qi_columns);
    void close_cache();

    ZkEmbedProof proveImpl(const ZkEmbedStatement &statement,
                           const std::vector<Fr> &h0_values, int num_threads,
                           FiatShamirTranscript &transcript,
                           ZkEmbedTiming &timing,
                           ZkEmbedDebugTrace *debug_trace);
    bool verifyImpl(const ZkEmbedStatement &statement,
                    const ZkEmbedProof &proof,
                    FiatShamirTranscript &transcript,
                    ZkEmbedTiming &timing,
                    ZkEmbedDebugTrace *debug_trace,
                    ZkEmbedOpeningClaim *pending_claim = nullptr);
    bool validOutput(const ZkEmbedStatement &statement) const;
    void validateWitness(const ZkEmbedStatement &statement,
                         const std::vector<Fr> &h0_values) const;
};

// Embedding and absolute-position binding to the GKR input.
struct ApeShape
{
    std::size_t padded_tokens = 0;
    std::size_t hidden = 0;
    int input_variables = 0;

    int tokenVariables() const;
    int columnVariables() const;
    int prefixVariables() const;
    std::size_t prefixRows() const;
    bool operator==(const ApeShape &other) const;
};

struct ApeProverKey
{
    ApeShape shape;
    std::vector<Fr> position_prefix;
};

class ApeVerifierKey
{
public:
    const ApeShape &shape() const { return shape_; }
    const std::vector<G1> &prefixCommitments() const { return prefix_commitments_; }
    void saveToFile(const std::string &path) const;
    static ApeVerifierKey loadFromFile(const std::string &path, const ApeShape &expected);

private:
    ApeShape shape_;
    std::vector<G1> prefix_commitments_;
    friend ApeVerifierKey prepareApeVerifierKey(
        const ApeProverKey &, const MVKZG_ProverKey &, int);
};

ApeProverKey prepareApeProverKey(
    const ApeShape &shape, const std::vector<std::int32_t> &position_table);
ApeVerifierKey prepareApeVerifierKey(
    const ApeProverKey &source, const MVKZG_ProverKey &pk, int threads = 1);

struct ZkEmbedInputProof
{
    ZkEmbedProof embedding;
    Fr position_evaluation = Fr(0);
    hyrax::OpeningProof position_opening;

    std::size_t cryptographicPayloadBytes() const;
};

ZkEmbedInputProof proveZkEmbedInputLink(
    ZkEmbed &embedding, const std::vector<ll> &queries,
    const std::vector<Fr> &h0_values, const std::vector<G1> &input_commitments,
    const ApeProverKey &ape_pk, const ApeVerifierKey &ape_vk,
    const MVKZG_ProverKey &pk, FiatShamirTranscript &transcript,
    InputReductionLink &pending_claim);

std::unique_ptr<InputReductionLink> verifyZkEmbedInputLink(
    ZkEmbed &embedding, const std::vector<ll> &queries,
    const std::vector<G1> &input_commitments, const ApeVerifierKey &ape_vk,
    const ZkEmbedInputProof &proof, const MVKZG_VerifierKey &vk,
    FiatShamirTranscript &transcript);

#endif
