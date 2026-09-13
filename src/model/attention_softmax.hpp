#pragma once

#include "model/quant_plan.hpp"

#include <array>
#include <string>
#include <vector>

struct AttentionSoftmaxRowWitness
{
    bool success = false;
    std::string message;
    ll zeta = 0;
    std::array<std::vector<ll>, 6> digits;
    std::vector<ll> middle_outputs;
    std::array<std::vector<ll>, 2> high_zero_flags;
    std::vector<ll> probabilities;
    ll total_mass = 0;
    ll lower_residual = 0;
    ll upper_residual = 0;
};

// Digit helpers consume an already validated plan and an in-range digit/segment.
int attention_digit_base(const QuantAttentionPlan &plan, int segment);
ll attention_digit_weight(const QuantAttentionPlan &plan, int segment);
ll attention_middle_table_output(const QuantAttentionPlan &plan, ll digit);

AttentionSoftmaxRowWitness build_attention_softmax_row(
    const std::vector<ll> &scores,
    const QuantAttentionPlan &plan);

bool verify_attention_softmax_row(
    const AttentionSoftmaxRowWitness &witness,
    const std::vector<ll> &scores,
    const QuantAttentionPlan &plan);

class PreparedAttentionSoftmax
{
public:
    explicit PreparedAttentionSoftmax(const QuantAttentionPlan &plan);
    AttentionSoftmaxRowWitness build(const std::vector<ll> &scores) const;
    bool verify(const AttentionSoftmaxRowWitness &, const std::vector<ll> &scores) const;
    const std::vector<ll> &middleTable() const { return middle_table_; }
    const std::vector<ll> &zeroTable(int high) const { return zero_tables_.at(high); }

private:
    QuantAttentionPlan plan_;
    std::vector<ll> middle_table_;
    std::array<std::vector<ll>, 2> zero_tables_;
};
