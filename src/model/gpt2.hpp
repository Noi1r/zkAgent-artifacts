

#ifndef GPT2_HPP
#define GPT2_HPP

#include "common/global_var.hpp"
#include "model/nonlinear_quant.hpp"
#include "common/model_limits.hpp"
#include "gkr/circuit.h"
#include "common/polynomial.h"
#include "gkr/prover.hpp"
#include "common/utils.hpp"
#include "zkembed/zkembed.hpp"
#include "model/quant_plan.hpp"
#include "model/linear_witness.hpp"
#include "model/attention_softmax.hpp"
#include "lookup/lookup_witness.hpp"
#include "gkr/input_commitment.hpp"
#include <vector>
#include <fstream>
#include <cmath>
#include <memory>
#include <string>

using std::ifstream;
using std::make_pair;
using std::pair;
using std::vector;

enum class GPT2CircuitProfile
{
    CompleteInference = 0,
    ZkGPTArtifactTransformer = 1,
};

struct GPT2CircuitProfileSpec
{
    const char *name;
    bool force_immediate_linear_requant;
    bool include_skip_connections;
    bool include_final_layer_norm;
    bool include_lm_head;
    bool include_external_lookups;
};

const GPT2CircuitProfileSpec &gpt2CircuitProfileSpec(
    GPT2CircuitProfile profile);

class GPT2
{
public:
    static constexpr int DefaultQBits = 20;

    GPT2(KZG_SRS &uni_srs,
         MVKZG_SRS &mv_srs,
         int depth,
         int headnum,
         int headdim,
         int attn_dim,
         int linear_dim,
         int seq_len,
         int vocab_size,
         int threads = 32);

    ~GPT2();

    void loadTokensFromFile(const char *path);
    void setTokens(const vector<int> &tokens);
    void setTokensNoPad(const vector<int> &tokens);
    int inputTokenCount() const { return static_cast<int>(input_tokens.size()); }

    void loadEmbedding(const char *path);
    void initRandomEmbedding();
    void setupZkEmbed();
    std::vector<Fr> prepareZkEmbedWitness() const;
    void computeEmbedding();

    void loadTransformerWeights(const char *model_dir);
    void loadPositionEmbedding(const char *path);
    void setCircuitProfile(GPT2CircuitProfile profile);
    GPT2CircuitProfile circuitProfile() const { return circuit_profile_; }
    const GPT2CircuitProfileSpec &circuitProfileSpec() const
    {
        return gpt2CircuitProfileSpec(circuit_profile_);
    }
    void prepareCircuitData(prover &pr);
    void buildCircuit(prover &pr, bool merge = false);
    bool saveCircuitCache(const char *path, prover &pr) const;
    bool loadCircuitCache(const char *path, prover &pr);
    void prepareWeightsForCache(prover &pr);
    void recomputeWitness(prover &pr);
    void create(prover &pr, bool merge = false);
    struct FreshFcTiming
    {
        double kernel_preparation_seconds = 0;
        double witness_seconds = 0;
        std::size_t evaluated_layers = 0;
    };
    const FreshFcTiming &freshFcTiming() const { return fresh_fc_timing_; }
    InputCommitmentLayout inputCommitmentLayout() const;
    InputCommitmentPreparation prepareInputCommitment(
        prover &pr, const std::string &key_file, const std::string &srs_cache,
        bool replace = false);
    void commitCircuit(prover &pr, const InputCommitmentKey &key);
    void merge_layer(prover &pr, i64 layer_id);

    int getQuantRangeBits() const { return quant_plan.loaded() ? quant_plan.maxRangeBits() : 0; }

    void setSaveIntermediates(const char *output_dir);
    void saveLayerOutput(const char *name, const ll *data, int rows, int cols);
    void dumpBlockOutputs(const char *output_dir);
    void dumpBlockStages(prover &pr, const char *output_dir);

    void setInput(const ll *input_data);
    void setSeqLen(int new_seq_len);

    bool validateCircuitProfile(const prover &pr,
                                std::string *error = nullptr) const;

    int layer_num;
    int headnum;
    int headdim;
    int attn_dim;
    int linear_dim;
    int len;
    int vocab_size;
    int threads;

    bool save_intermediates;
    std::string intermediates_dir;

    G1 *gens;
    G1 gens_base;
    int gens_size;

    KZG_SRS &uni_srs;
    MVKZG_SRS &mv_srs;

    void initGenerators();
    void ensureGenerators(const prover &pr);

    float *embedding_matrix;
    float *position_embedding;
    ll *quantized_embedding_matrix;
    std::unique_ptr<ZkEmbed> zkembed;

    vector<int> input_tokens;

    struct RoundRequant
    {
        ll c1 = 0;
        ll c2 = 0;
        int n = 1;
        int range_bits = DefaultQBits + 1;
    };
    struct FCRoundRequant
    {
        RoundRequant group[3];
        int group_count = 1;
        int group_width = 0;
    };
    GPT2QuantPlan quant_plan;
    vector<FCRoundRequant> fc_round_requants;
    vector<int> fc_round_offsets;
    vector<int> block_attn_residual_offsets;
    vector<int> block_ffn_residual_offsets;

    struct FCInputScale
    {
        int c, m;
        double scale;
    };
    vector<FCInputScale> fc_input_scales;

    int channel_in, channel_out;
    int q_offset;

    int ln_aux_start;
    int layer_norm_w_q_start[model_limits::LayerNormCapacity];
    int layer_norm_b_q_start[model_limits::LayerNormCapacity];
    int layer_norm_c1[model_limits::LayerNormCapacity];
    int layer_norm_e1[model_limits::LayerNormCapacity];
    int layer_norm_c2[model_limits::LayerNormCapacity];
    int layer_norm_e2[model_limits::LayerNormCapacity];

    int gelu_aux_start;

    int softmax_aux_start;

    i64 total_in_size;
    struct AttentionLookupGroup
    {
        std::string key;
        int input_bits = 0;
        bool mapped = false;
        std::vector<int> input_offsets;
        std::vector<int> output_offsets;
        std::vector<ll> table_outputs;
    };
    std::vector<AttentionLookupGroup> attention_lookup_groups;

    const vector<RangeQuerySpec> &rangeQuerySpecs() const
    {
        return range_query_specs;
    }
    LookupWitness buildAttentionLookupWitness(const prover &pr) const;
    LookupWitness buildTransformerLookupWitness(
        const prover &pr, unsigned full_limb_bits = 10) const;

    int lm_head_output_offset = 0;
    int embed_matrix_circuit_offset = 0;

    ll *getRangeProofInputs(int &out_count);

    vector<vector<F>>::iterator val;
    double quant_scale;

    struct fconKernel
    {
        i64 channel_in;
        i64 channel_out;
        i64 weight_start_id;
        fconKernel(i64 co, i64 ci) : channel_in(ci), channel_out(co), weight_start_id(0) {}
    };
    vector<fconKernel> full_conn;

    int *mat_values[model_limits::FconnCapacity];

    u32 SIZE;

private:
    struct GateWitnessSchedule
    {
        std::vector<u32> unary_offsets, unary_indices;
        std::vector<u32> binary_offsets, binary_indices;
        static GateWitnessSchedule build(const layer &circuit);
    };
    std::vector<GateWitnessSchedule> prepared_gate_schedules_;
    const layeredCircuit *prepared_gate_owner_ = nullptr;
    void prepareGateWitnessSchedules(const prover &pr);
    std::vector<PreparedAttentionSoftmax> prepared_attention_;
    std::vector<std::array<std::int64_t, 3>> prepared_qkv_bounds_;
    bool prepared_attention_ready_ = false;
    void prepareCachedAttention(const prover &pr);
    std::size_t ensureAttentionLookupGroup(const std::string &key, int bits,
                                           bool mapped, const std::vector<ll> &table);
    std::vector<zkagent_integer::PreparedLinearWitness> prepared_fc_witness_;
    const prover *prepared_weights_owner_ = nullptr;
    void prepareLinearWitnessKernels(const prover &pr, bool include_lm_head = true);
    FreshFcTiming fresh_fc_timing_;
    bool build_gates = true;
    GPT2CircuitProfile circuit_profile_ =
        GPT2CircuitProfile::CompleteInference;

    const ll *input_data = nullptr;

    ll *owned_embedded_input = nullptr;

    i64 matIdx(i64 row, i64 col, i64 cols) { return row * cols + col; }
    void initQuantPolicy(prover &pr);
    const QuantLinearPlan &linearPlanForFc(int fc_id, int group = 0) const;
    const QuantLayerNormPlan &layerNormPlan(int ln_id) const;

    void prepareWeights(prover &pr);
    void prepareFCWeights(i64 first_fc_id, int real_r, int real_c, int id, int ch_in, int ch_out);
    void prepareFCBias(prover &pr, int real_c, int id, int ch_out);
    void prepareLayerNormWeights(int ln_id, int ch_in);
    void prepareFinalLayerNormWeights(int ln_id, int ch_in);

    void initParam(prover &pr);
    void inputLayer(layer &circuit, int ch_in);
    void calcInputLayer(layer &circuit, int ch_in);

    void computeLayerNormWitness(int ln_id, int input_offset, int aux_start,
                                 int real_width, int padded_width, int real_rows,
                                 const bool *sparsity_map = nullptr);
    void computeRoundWitness(int fc_id, const vector<F> &accumulator,
                             int output_offset, int width);
    void computeResidualWitness(const vector<F> &accumulator, int residual_offset,
                                int output_offset, int width, const QuantResidualPlan &plan);
    zkagent_integer::GeluCoefficients computeGeluWitness(
        int input_offset, int aux_start, int real_width, int padded_width,
        const QuantBlockPlan &block);

    void ln_checker_layer1(layer &circuit, i64 &layer_id, int ln_id, int ey, int cy, int real_cn_in, int ch_in, bool *sparsity_map);
    void ln_checker_layer2(layer &circuit, i64 &layer_id, int ln_id, int ey, int cy, int ch_in, bool *sparsity_map);
    void ln_checker_layer3(layer &circuit, i64 &layer_id, int ln_id, int ey, int cy, int real_cn_in, int ch_in, bool *sparsity_map);

    void fullyConnLayer(layer &circuit, i64 &layer_id, int fc_id, int x_offset, int x_layer,
                        int ch_in, int ch_out);
    void roundLayer(layer &circuit, i64 &layer_id, int fc_id, int ch_out, bool *sparsity_map = nullptr);

    void delayedResidualAddLayers(prover &pr, i64 &layer_id, int ch_out,
                                  int residual_offset, const QuantResidualPlan &plan);

    void multi_head_matrix_QK(layer &circuit, i64 &layer_id, int ch_out);

    void softmax_layer_1(layer &circuit, i64 &layer_id, int ch_in,
                         int ch_out, int block_idx);
    void softmax_layer_2(layer &circuit, i64 &layer_id, int ch_in,
                         int block_idx);
    void softmax_layer_3(layer &circuit, i64 &layer_id, int block_idx);
    void recordAttentionLookup(
        const std::string &key, int input_bits, bool mapped,
        int input_offset, int output_offset,
        const std::vector<ll> &table_outputs);
    void recordPowerOfTwoRange(
        const std::string &id, std::size_t offset, std::size_t count,
        unsigned bit_width, std::size_t stride = 1);

    vector<RangeQuerySpec> range_query_specs;

    void gelu_checker_layer1(layer &circuit, i64 &layer_id, int block_idx,
                             int real_cn_out, int ch_out);
    void gelu_checker_layer2(layer &circuit, i64 &layer_id, int block_idx,
                             int real_cn_out, int ch_out);
    void gelu_checker_layer3(layer &circuit, i64 &layer_id, int real_cn_out, int ch_out);

    void buildLMHeadCircuit(prover &pr, i64 &layer_id);

    void lmHeadWideOutputLayer(layer &circuit, i64 &layer_id);

    void calcNormalLayer(const layer &circuit, i64 layer_id,
                         const GateWitnessSchedule *schedule = nullptr);
    void recomputeAuxValues(prover &pr);
};

#endif
