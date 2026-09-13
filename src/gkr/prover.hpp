#ifndef ZKCNN_PROVER_HPP
#define ZKCNN_PROVER_HPP

#include "common/global_var.hpp"
#include "common/model_limits.hpp"
#include "gkr/circuit.h"
#include "common/polynomial.h"
#include "pcs/hyrax.hpp"
#include "gkr/input_commitment.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

struct InputCommitTiming
{
    double padding_seconds = 0;
    double row_commitment_seconds = 0;
};

struct InputReductionInitTiming
{
    double reset_seconds = 0;
    double equality_seconds = 0;
    double scatter_seconds = 0;
    std::size_t parallel_routes = 0;
    std::size_t serial_routes = 0;
};

struct Commit_return
{
    int l;
    const Fr *w;
    G1 G, *g;
    G1 *comm;
};

struct SumcheckInitTiming
{
    double phase1_total_seconds = 0;
    double phase1_workspace_seconds = 0;
    double phase1_witness_buffer_seconds = 0;
    double phase1_beta_seconds = 0;
    double phase1_relu_scale_seconds = 0;
    double phase1_unary_gate_seconds = 0;
    double phase1_binary_gate_seconds = 0;
    double phase2_total_seconds = 0;
    double phase2_workspace_seconds = 0;
    double phase2_witness_buffer_seconds = 0;
    double phase2_beta_seconds = 0;
    double phase2_unary_gate_seconds = 0;
    double phase2_binary_gate_seconds = 0;
    u64 phase1_calls = 0;
    u64 phase2_calls = 0;

    double movable_upper_bound_seconds() const
    {
        return phase1_workspace_seconds + phase1_witness_buffer_seconds + phase2_workspace_seconds + phase2_witness_buffer_seconds;
    }

    double total_seconds() const
    {
        return phase1_total_seconds + phase2_total_seconds;
    }
};

struct GkrWorkspacePreparation
{
    std::size_t branch_domain[2] = {1, 1};
    std::size_t beta_g_domain = 1;
    std::size_t beta_u_domain = 1;
    std::size_t input_claim_domain = 1;
    std::size_t allocated_bytes = 0;
};

class prover
{
public:
    int fc_row[model_limits::FconnCapacity];
    int fc_col[model_limits::FconnCapacity];
    int fc_start_id[model_limits::FconnCapacity];
    int fc_input_row[model_limits::FconnCapacity];
    int fc_input_col[model_limits::FconnCapacity];
    int fc_input_id[model_limits::FconnCapacity];
    int fc_bias_id[model_limits::FconnCapacity];
    int fc_real_row[model_limits::FconnCapacity];
    int fc_real_col[model_limits::FconnCapacity];
    int fc_real_input_row[model_limits::FconnCapacity];
    int fc_real_input_col[model_limits::FconnCapacity];
    int lm_fc_id = -1;
    u32 total[2], total_size[2];
    vector<linear_poly> mult_array[2];
    vector<linear_poly> V_mult[2];
    vector<linear_poly> tmp_mult_array[2];
    vector<linear_poly> tmp_V_mult[2];
    int **mat_val;
    bool prepared_model_values = false;
    void init();
    GkrWorkspacePreparation prepareGkrWorkspace();

    void sumcheckInitAll(const vector<F>::const_iterator &r_0_from_v);
    void sumcheckInit(const F &alpha_0, const F &beta_0,
                      const vector<F> *fresh_zero_root = nullptr);
    void sumcheckInitPhase1(const F &relu_rou_0);
    void sumcheckInitPhase2(
        const std::array<F, 2> *phase1_unary = nullptr);

    quadratic_poly sumcheckUpdate1(const F &previous_random);
    quadratic_poly sumcheckUpdate2(const F &previous_random);

    F Vres(const vector<F>::const_iterator &r, u32 output_size, u8 r_size, int layer_id, int start = 0);
    vector<F> input_claim_weights;
    const vector<std::uint64_t> &inputClaimActiveBits() const
    {
        return input_claim_active_bits;
    }
    vector<std::uint64_t> &mutableInputClaimActiveBits()
    {
        return input_claim_active_bits;
    }
    vector<u32> &mutableInputClaimActiveIndices()
    {
        return input_claim_active_indices;
    }
    std::size_t inputClaimActiveCount() const
    {
        return input_claim_active_indices.size();
    }
    void releaseInputClaimWeights();
    void sumcheckFinalize1(const F &previous_random, F &claim_0, F &claim_1);
    void sumcheckFinalize2(const F &previous_random, F &claim_0, F &claim_1);
    void setThreadCount(int threads) { thread_num = threads > 0 ? threads : 1; }
    int threadCount() const { return thread_num; }
    const SumcheckInitTiming &sumcheckInitTiming() const
    {
        return sumcheck_init_timing;
    }
    const InputCommitTiming &inputCommitTiming() const { return input_commit_timing_; }
    const InputReductionInitTiming &inputReductionInitTiming() const { return input_reduction_init_timing_; }

    void initInputClaimReduction(const vector<F> &s_u, const vector<F> &s_v,
                                 const vector<vector<F>> &r_uu,
                                 const vector<vector<F>> &r_vv);
    void commitInput(const vector<G1> &gens, int thread_n = 1,
                     const InputCommitmentKey *fixed_key = nullptr);

    layeredCircuit C;
    vector<vector<F>> val;
    F getCirValue(int layer_id, const vector<u32> &ori, u32 u);
    u64 proof_size = 0;
    vector<G> gens;
    Commit_return cc{};

private:
    std::vector<G1> input_commitment_rows_;
    SumcheckInitTiming sumcheck_init_timing;
    InputCommitTiming input_commit_timing_;
    InputReductionInitTiming input_reduction_init_timing_;
    quadratic_poly sumcheckUpdateEach(const F &previous_random, bool idx);
    quadratic_poly sumcheckUpdate(const F &previous_random, vector<F> &r_arr);

    vector<F>::iterator r_0, r_1;
    vector<vector<F>> r_u, r_v;
    vector<F> fresh_zero_root_point;

    vector<F> beta_g;

    vector<std::uint64_t> input_claim_active_bits;
    vector<u32> input_claim_active_indices;
    bool input_claim_workspace_initialized = false;

    F add_term;

    F V_u0, V_u1;

    F alpha, beta, relu_rou;
    int thread_num = 32;

    u8 round;
    int sumcheck_id;
};
#endif
