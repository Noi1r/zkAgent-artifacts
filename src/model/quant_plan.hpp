#pragma once

#include "common/global_var.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct QuantDyadic
{
    ll c1 = 0;
    ll c2 = 0;
    int n = 0;
    int range_bits = 0;
};

struct QuantLinearPlan
{
    std::string name;
    std::string mode;
    int in_dim = 0;
    int out_dim = 0;
    double input_scale = 0.0;
    ll input_bound = 0;
    double weight_scale = 0.0;
    double accumulator_scale = 0.0;
    ll accumulator_bound = 0;
    double output_scale = 0.0;
    ll output_bound = 0;
    QuantDyadic requant;
    std::vector<int32_t> weight;
    std::vector<int64_t> bias;
};

struct QuantLayerNormPlan
{
    std::string name;
    int width = 0;
    ll input_bound = 0;
    double normalized_scale = 0.0;
    double gamma_scale = 0.0;
    double beta_scale = 0.0;
    double output_scale = 0.0;
    ll accumulator_bound = 0;
    ll output_bound = 0;
    int term1_c = 0;
    int term1_e = 0;
    int term2_c = 0;
    int term2_e = 0;
    int sigma_range_bits = 0;
    int rounding_range_bits = 0;
    int variance_bits = 0;
    int q_input_sigma_range_bits = 0;
    int q_input_rounding_range_bits = 0;
    int q_input_variance_bits = 0;
    std::vector<int32_t> gamma;
    std::vector<int64_t> beta;
};

struct QuantResidualPlan
{
    std::string name;
    double left_scale = 0.0;
    ll left_bound = 0;
    double right_scale = 0.0;
    ll right_bound = 0;
    double output_scale = 0.0;
    ll output_bound = 0;
    QuantDyadic left;
    QuantDyadic right;
};

struct QuantAttentionPlan
{
    double score_scale = 0.0;
    ll score_bound = 0;
    int score_bits = 0;
    ll theta = 0;
    ll shift_upper_bound = 0;
    int shift_bits = 0;
    std::array<int, 6> digit_bits{{0, 0, 0, 0, 0, 0}};
    int least_segments = 3;
    int nonlinear_segment = 3;
    int most_zero_segments = 2;
    std::vector<int64_t> epsilon_by_count;
    int mass_bits = 0;
    ll accumulator_bound = 0;
    int accumulator_bits = 0;
    int rounding_bits = 0;
    ll output_bound = 0;
    int output_bits = 0;
};

struct QuantBlockPlan
{
    QuantLayerNormPlan ln1;
    QuantLinearPlan q;
    QuantLinearPlan k;
    QuantLinearPlan v;
    QuantAttentionPlan attention;
    QuantLinearPlan attn_proj;
    QuantResidualPlan attn_residual;
    QuantLayerNormPlan ln2;
    QuantLinearPlan ffn_up;
    int gelu_scale_c = 0;
    int gelu_scale_e = 0;
    double gelu_input_scale = 0.0;
    double gelu_output_scale = 0.0;
    int gelu_d1_range_bits = 0;
    int gelu_d2_range_bits = 0;
    int gelu_rounding_range_bits = 0;
    QuantLinearPlan ffn_down;
    QuantResidualPlan ffn_residual;
};

class GPT2QuantPlan
{
public:
    bool load(const std::string &model_dir, std::string *error = nullptr);
    bool validateDimensions(int layers, int heads, int head_dim, int hidden,
                            int ffn, int vocab, std::string *error = nullptr) const;
    bool loaded() const { return is_loaded_; }
    int maxRangeBits() const;

    int q_bits = 0;
    ll R = 0;
    int vocab_size = 0;
    int positions = 0;
    int hidden = 0;
    int layer_count = 0;
    int heads = 0;
    int head_dim = 0;
    int ffn = 0;

    double embedding_scale = 0.0;
    ll embedding_bound = 0;
    std::vector<int32_t> token_embedding;
    std::vector<int32_t> position_embedding;

    std::vector<QuantBlockPlan> blocks;
    QuantLayerNormPlan final_ln;
    QuantLinearPlan lm_head;
    double logit_scale = 0.0;
    ll logit_bound = 0;

    const std::string &artifactDir() const { return artifact_dir_; }

private:
    using Values = std::unordered_map<std::string, std::string>;

    bool is_loaded_ = false;
    std::string artifact_dir_;
    Values values_;

    bool fail(const std::string &message, std::string *error) const;
    bool parseManifest(const std::string &path, std::string *error);
    bool loadLinear(const std::string &name, QuantLinearPlan &out, std::string *error);
    bool loadLayerNorm(const std::string &name, QuantLayerNormPlan &out, std::string *error);
    bool loadResidual(const std::string &name, QuantResidualPlan &out, std::string *error);
    bool loadAttention(const std::string &name, QuantAttentionPlan &out,
                       std::string *error);

    bool getString(const std::string &key, std::string &out, std::string *error) const;
    bool getInt(const std::string &key, int &out, std::string *error) const;
    bool getI64(const std::string &key, ll &out, std::string *error) const;
    bool getDouble(const std::string &key, double &out, std::string *error) const;

    template <typename T>
    bool loadTensor(const std::string &key, std::vector<T> &out, std::string *error) const;
};
