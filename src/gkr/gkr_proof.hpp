#ifndef ZKAGENT_GKR_PROOF_HPP
#define ZKAGENT_GKR_PROOF_HPP

#include "common/fiat_shamir.hpp"
#include "pcs/hyrax.hpp"
#include "common/polynomial.h"

#include <cstddef>
#include <vector>

class prover;
class layeredCircuit;
class layer;
struct MVKZG_ProverKey;

struct GkrFconnProductProof
{
    std::vector<quadratic_poly> rounds;
    F input_evaluation = F_ZERO;
    F matrix_evaluation = F_ZERO;
    hyrax::OpeningProof matrix_opening;
};

struct GkrLayerProof
{
    bool provided = false;
    bool is_fconn = false;
    GkrFconnProductProof fconn;

    std::vector<quadratic_poly> phase1_rounds;
    F final_u0 = F_ZERO;
    F final_u1 = F_ZERO;

    bool has_phase2 = false;
    std::vector<quadratic_poly> phase2_rounds;
    F final_v0 = F_ZERO;
    F final_v1 = F_ZERO;
};

struct GkrInputReductionRound
{
    F at_zero = F_ZERO;
    F at_one = F_ZERO;
    F at_two = F_ZERO;
};

struct GkrInputOpeningProof
{
    bool provided = false;
    F evaluation = F_ZERO;
    hyrax::OpeningProof opening;
};

struct GkrProof
{
    F output_evaluation = F_ZERO;
    std::vector<GkrLayerProof> layers;
    std::vector<GkrInputReductionRound> input_reduction_rounds;
    F input_evaluation = F_ZERO;
    GkrInputOpeningProof input_opening;

    std::size_t cryptographicPayloadBytes() const;
};

struct GkrFconnVerifierParameters
{
    int rows = 0;
    int columns = 0;
    int input_rows = 0;
    int matrix_variables = 0;
    std::size_t input_offset = 0;
    std::vector<G1> matrix_commitments;
    std::vector<F> bias;

    // Full index validation belongs to trusted key/circuit preparation.
    bool matchesInputRouting(const layer &circuit, std::size_t input_values,
                             bool check_indices) const;
};

struct GkrVerifierKey
{
    std::vector<GkrFconnVerifierParameters> fconn;
};

struct GkrFconnProverParameters
{
    int matrix_variables = 0;
    std::vector<F> matrix_values;
    const std::vector<F> *prepared_input_values = nullptr;
    std::size_t prepared_offset = 0, prepared_count = 0;
    const F *matrixData() const;
    std::size_t matrixSize() const;
};

struct GkrProverKey
{
    std::vector<GkrFconnProverParameters> fconn;
};

struct GkrKeyBundle
{
    GkrProverKey prover;
    GkrVerifierKey verifier;
};

GkrKeyBundle prepareGkrKeys(
    const prover &source,
    const MVKZG_ProverKey &commitment_key,
    int threads = 1);

GkrKeyBundle prepareGkrKeysCached(
    const prover &source,
    const MVKZG_ProverKey &commitment_key,
    const MVKZG_VerifierKey &verifier_key,
    const std::string &cache_path,
    int threads = 1,
    bool refresh = false,
    bool *cache_loaded = nullptr);

GkrVerifierKey makeGkrBenchmarkVerifierKey(const prover &source);

struct GkrInputCommitment
{
    int variables = 0;
    std::vector<G1> rows;
};

GkrInputCommitment makeGkrInputCommitment(const prover &source);

struct GkrPublicContext
{
    int sequence_length = 0;
    int layers = 0;
    int heads = 0;
    int head_dimension = 0;
    int hidden_dimension = 0;
    int ffn_dimension = 0;
    int vocabulary_size = 0;
    bool merged = false;
};

void appendGkrPublicContext(
    FiatShamirTranscript &transcript,
    const layeredCircuit &circuit,
    const GkrVerifierKey &verifier_key,
    const GkrInputCommitment &input_commitment,
    const GkrPublicContext &context);

#endif
