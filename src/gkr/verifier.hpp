

#ifndef ZKCNN_CONVVERIFIER_HPP
#define ZKCNN_CONVVERIFIER_HPP

#include "gkr/prover.hpp"
#include "common/fiat_shamir.hpp"
#include "gkr/gkr_proof.hpp"
#include "pcs/hyrax.hpp"
#include "gkr/input_reduction_link.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
using std::unique_ptr;
struct MVKZG_ProverKey;
struct MVKZG_VerifierKey;
class CompiledWiringPlan;
class CompiledInputRoutingPlan;

struct InputReductionWorkspacePreparation
{
    std::size_t current_domain = 0;
    std::size_t next_domain = 0;
    std::size_t allocated_bytes = 0;
};

class verifier
{
public:
    prover *p;
    const layeredCircuit &C;

    verifier(prover *pr, const layeredCircuit &cir,
             const MVKZG_ProverKey *mv_pk = nullptr,
             const MVKZG_VerifierKey *mv_vk = nullptr,
             bool debug = false);
    verifier(const layeredCircuit &cir,
             GkrVerifierKey verifier_key,
             GkrInputCommitment input_commitment,
             const MVKZG_VerifierKey *mv_vk = nullptr,
             bool debug = false);

    bool prove(int commit_thread = 4);
    bool generateProof(int commit_thread = 4);
    bool verifyProof(
        const GkrProof &proof,
        FiatShamirTranscript *transcript);
    void setInputCommitmentKey(const InputCommitmentKey &key) { input_commitment_key_ = &key; }
    void setThreadCount(int threads)
    {
        thread_num_ = std::max(1, threads);
    }
    bool gkr_verified() const { return last_gkr_ok_; }
    bool input_reduction_verified() const { return last_input_reduction_ok_; }
    bool lookup_link_verified() const { return last_lookup_link_ok_; }
    bool commitment_checked() const { return last_commitment_checked_; }
    bool commitment_verified() const { return last_commitment_ok_; }
    bool input_open_reused_fold() const { return input_open_rt_reused_; }
    const GkrProof &proof() const { return proof_; }
    InputReductionWorkspacePreparation prepareInputReductionWorkspace();
    void range_prove(double range_prover_time_) { range_prover_time = range_prover_time_; };
    void setFiatShamirTranscript(FiatShamirTranscript &transcript);
    void setGkrKeys(
        const GkrProverKey &prover_key,
        const GkrVerifierKey &verifier_key)
    {
        gkr_prover_key_ = &prover_key;
        verifier_key_ = verifier_key;
    }
    void setInputReductionLink(InputReductionLink link);
    void setCompiledWiringPlan(const CompiledWiringPlan &plan);
    bool compiled_wiring_enabled() const
    {
        return compiled_wiring_plan_ != nullptr;
    }
    void setCompiledInputRoutingPlan(const CompiledInputRoutingPlan &plan);
    bool compiled_input_routing_enabled() const
    {
        return compiled_input_routing_plan_ != nullptr;
    }

    double verifier_time;
    double prover_time;
    double range_prover_time = 0;
    double matrix_time;
    double lm_head_prover_time;
    double lm_head_matrix_time;
    double gkr_sumcheck_phase1_time;
    double gkr_sumcheck_phase2_time;
    double gkr_init_time;
    double gkr_output_eval_time;
    double verifier_sumcheck_time;
    double verifier_wiring_time;
    double verifier_wiring_contraction_time;
    double verifier_challenge_time;
    double verifier_input_reduction_time;
    double transformer_prover_time;
    bool transformer_time_valid;
    double input_reduction_init_time;
    double input_reduction_sumcheck_time;
    std::size_t input_reduction_active_coefficients;
    bool input_reduction_sparse_coefficients;
    double input_reduction_coefficient_time;
    double input_link_injection_time;
    double input_link_verifier_time;
    double commit_open_time;
    double commit_open_verify_time;
    Commit_return comm;

private:
    const InputCommitmentKey *input_commitment_key_ = nullptr;
    const MVKZG_ProverKey *mv_pk = nullptr;
    const MVKZG_VerifierKey *mv_vk = nullptr;
    bool debug_ = false;
    int thread_num_ = 1;
    bool last_gkr_ok_ = false;
    bool last_input_reduction_ok_ = false;
    bool last_lookup_link_ok_ = false;
    bool last_commitment_checked_ = false;
    bool last_commitment_ok_ = false;
    vector<vector<F>> r_u, r_v;
    vector<F> final_claim_u0, final_claim_v0;
    bool verifyGKR();
    bool verifyInputClaimReduction();
    bool openCommit();

    F uni_value[2];
    F bin_value[3];

    F getFinalValue(const F &claim_u0, const F &claim_u1, const F &claim_v0, const F &claim_v1);

    F eval_in;
    std::unique_ptr<InputReductionLink> input_reduction_link_;
    const CompiledWiringPlan *compiled_wiring_plan_ = nullptr;
    vector<Fr> compiled_wiring_workspace_;
    const CompiledInputRoutingPlan *compiled_input_routing_plan_ = nullptr;
    vector<Fr> compiled_input_routing_workspace_;
    vector<Fr> input_open_rt_;
    vector<Fr> input_reduction_coefficient_workspace_;
    vector<Fr> input_reduction_value_workspace_;
    vector<Fr> input_reduction_next_coefficient_workspace_;
    vector<Fr> input_reduction_next_value_workspace_;
    vector<std::uint64_t> input_reduction_active_workspace_;
    vector<std::uint64_t> input_reduction_next_active_workspace_;
    vector<Fr> input_link_compiled_workspace_;
    bool input_open_rt_ready_ = false;
    bool input_open_rt_reused_ = false;
    GkrProof proof_;
    const GkrProverKey *gkr_prover_key_ = nullptr;
    GkrVerifierKey verifier_key_;
    std::vector<G1> public_commitment_rows_;
    bool generation_only_ = false;
    FiatShamirTranscript *fiat_shamir_transcript_ = nullptr;
    F nextChallenge(bool reject_one = false);
    void requireChallengeMode() const;
    void initializeState();
};

#endif
