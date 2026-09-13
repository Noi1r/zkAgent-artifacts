#include "model/attention_softmax.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{

    ll round_nearest_signed(long double value)
    {
        if (!std::isfinite(value) || std::fabs(value) >= std::ldexp(1.0L, 120))
        {
            throw std::overflow_error("attention nearest-rounding input is invalid");
        }
        return value >= 0.0L
                   ? static_cast<ll>(std::floor(value + 0.5L))
                   : static_cast<ll>(std::ceil(value - 0.5L));
    }

    ll round_nonnegative(long double value)
    {
        if (!(value >= 0.0L) || !std::isfinite(value) || value >= std::ldexp(1.0L, 120))
        {
            throw std::overflow_error("attention positive rounding input is invalid");
        }
        return static_cast<ll>(std::floor(value + 0.5L));
    }

    int total_digit_bits(const QuantAttentionPlan &plan)
    {
        int total = 0;
        for (int bits : plan.digit_bits)
            total += bits;
        return total;
    }

    bool valid_plan(const QuantAttentionPlan &plan)
    {
        if (!(plan.score_scale > 0.0) || plan.score_bound <= 0 || plan.theta <= 0 || plan.shift_upper_bound <= 0 || plan.least_segments != 3 || plan.nonlinear_segment != 3 || plan.most_zero_segments != 2 || plan.epsilon_by_count.empty())
        {
            return false;
        }
        const int bits = total_digit_bits(plan);
        if (bits <= 0 || bits >= 63 || plan.shift_upper_bound >= (static_cast<ll>(1) << bits))
        {
            return false;
        }
        for (int width : plan.digit_bits)
        {
            if (width < 0 || width > 16)
                return false;
        }
        return true;
    }

}

int attention_digit_base(const QuantAttentionPlan &plan, int segment)
{
    return 1 << plan.digit_bits[static_cast<std::size_t>(segment)];
}

ll attention_digit_weight(const QuantAttentionPlan &plan, int segment)
{
    ll weight = 1;
    for (int i = 0; i < segment; ++i)
    {
        weight *= static_cast<ll>(attention_digit_base(plan, i));
    }
    return weight;
}

ll attention_middle_table_output(const QuantAttentionPlan &plan, ll digit)
{
    const int middle = plan.nonlinear_segment;
    const long double exponent =
        -static_cast<long double>(plan.score_scale) * static_cast<long double>(attention_digit_weight(plan, middle)) * static_cast<long double>(digit);
    const long double scaled =
        static_cast<long double>(plan.theta) * std::exp(exponent);
    return std::max<ll>(0, std::min<ll>(
                               plan.theta, round_nonnegative(scaled)));
}

static AttentionSoftmaxRowWitness build_attention_softmax_row_impl(
    const std::vector<ll> &scores,
    const QuantAttentionPlan &plan, const std::vector<ll> *middle_table)
{
    AttentionSoftmaxRowWitness result;
    if (scores.empty() || scores.size() > plan.epsilon_by_count.size())
    {
        result.message = "invalid attention Softmax row instance";
        return result;
    }
    try
    {
        ll maximum = scores[0];
        for (ll score : scores)
        {
            if (score < -plan.score_bound || score > plan.score_bound)
            {
                result.message = "attention score exceeds its public bound";
                return result;
            }
            maximum = std::max(maximum, score);
        }
        const long double alpha = static_cast<long double>(plan.score_scale);
        long double sum = 0.0L;
        for (ll score : scores)
        {
            sum += std::exp(alpha * static_cast<long double>(score - maximum));
        }
        result.zeta = round_nearest_signed(
            static_cast<long double>(maximum) + std::log(sum) / alpha);

        const std::size_t n = scores.size();
        for (auto &column : result.digits)
            column.assign(n, 0);
        result.middle_outputs.assign(n, 0);
        for (auto &column : result.high_zero_flags)
            column.assign(n, 1);
        result.probabilities.assign(n, 0);

        for (std::size_t i = 0; i < n; ++i)
        {
            ll magnitude = result.zeta - scores[i];
            if (magnitude < 0 || magnitude > plan.shift_upper_bound)
            {
                result.message = "attention exp magnitude exceeds its public bound";
                return result;
            }
            ll remaining = magnitude;
            for (int segment = 0; segment < 6; ++segment)
            {
                const int base = attention_digit_base(plan, segment);
                result.digits[static_cast<std::size_t>(segment)][i] =
                    remaining % static_cast<ll>(base);
                remaining /= static_cast<ll>(base);
            }
            if (remaining != 0)
            {
                result.message = "attention digit decomposition overflowed";
                return result;
            }
            const ll middle_digit = result.digits[static_cast<std::size_t>(plan.nonlinear_segment)][i];
            result.middle_outputs[i] = middle_table
                                           ? (*middle_table)[static_cast<std::size_t>(middle_digit)]
                                           : attention_middle_table_output(plan, middle_digit);
            bool keep = true;
            for (int high = 0; high < 2; ++high)
            {
                const int segment = plan.nonlinear_segment + 1 + high;
                const ll flag = result.digits[static_cast<std::size_t>(segment)][i] == 0 ? 1 : 0;
                result.high_zero_flags[static_cast<std::size_t>(high)][i] = flag;
                keep = keep && flag == 1;
            }
            result.probabilities[i] = keep ? result.middle_outputs[i] : 0;
            result.total_mass += result.probabilities[i];
        }

        const ll epsilon = static_cast<ll>(plan.epsilon_by_count[n - 1]);
        result.lower_residual = result.total_mass - (plan.theta - epsilon);
        result.upper_residual = plan.theta + epsilon - result.total_mass;
        if (result.lower_residual < 0 || result.upper_residual < 0)
        {
            result.message = "attention row mass exceeds its public tolerance";
            return result;
        }
    }
    catch (const std::exception &error)
    {
        result.message = error.what();
        return result;
    }
    result.success = true;
    result.message = "attention zkLLM digit witness constructed";
    return result;
}

static bool verify_attention_softmax_row_impl(
    const AttentionSoftmaxRowWitness &witness,
    const std::vector<ll> &scores,
    const QuantAttentionPlan &plan, const std::vector<ll> *middle_table)
{
    if (!witness.success || scores.empty() || scores.size() > plan.epsilon_by_count.size())
    {
        return false;
    }
    const std::size_t n = scores.size();
    if (witness.middle_outputs.size() != n || witness.probabilities.size() != n)
    {
        return false;
    }
    for (const auto &column : witness.digits)
    {
        if (column.size() != n)
            return false;
    }
    for (const auto &column : witness.high_zero_flags)
    {
        if (column.size() != n)
            return false;
    }
    ll total = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (scores[i] < -plan.score_bound || scores[i] > plan.score_bound)
        {
            return false;
        }
        ll magnitude = 0;
        ll weight = 1;
        for (int segment = 0; segment < 6; ++segment)
        {
            const ll digit = witness.digits[static_cast<std::size_t>(segment)][i];
            const int base = attention_digit_base(plan, segment);
            if (digit < 0 || digit >= base)
                return false;
            magnitude += weight * digit;
            weight *= static_cast<ll>(base);
        }
        if (magnitude != witness.zeta - scores[i] || magnitude < 0 || magnitude > plan.shift_upper_bound)
        {
            return false;
        }
        const ll middle = witness.digits[static_cast<std::size_t>(plan.nonlinear_segment)][i];
        if (witness.middle_outputs[i] != (middle_table ? (*middle_table)[static_cast<std::size_t>(middle)]
                                                       : attention_middle_table_output(plan, middle)))
        {
            return false;
        }
        ll keep = 1;
        for (int high = 0; high < 2; ++high)
        {
            const int segment = plan.nonlinear_segment + 1 + high;
            const ll expected = witness.digits[static_cast<std::size_t>(segment)][i] == 0 ? 1 : 0;
            if (witness.high_zero_flags[static_cast<std::size_t>(high)][i] != expected)
            {
                return false;
            }
            keep *= expected;
        }
        if (witness.probabilities[i] != witness.middle_outputs[i] * keep)
        {
            return false;
        }
        total += witness.probabilities[i];
    }
    const ll epsilon = static_cast<ll>(plan.epsilon_by_count[n - 1]);
    return total == witness.total_mass && witness.lower_residual == total - (plan.theta - epsilon) && witness.upper_residual == plan.theta + epsilon - total && witness.lower_residual >= 0 && witness.upper_residual >= 0;
}

AttentionSoftmaxRowWitness build_attention_softmax_row(
    const std::vector<ll> &scores, const QuantAttentionPlan &plan)
{
    if (!valid_plan(plan))
    {
        AttentionSoftmaxRowWitness result;
        result.message = "invalid attention Softmax plan";
        return result;
    }
    return build_attention_softmax_row_impl(scores, plan, nullptr);
}

bool verify_attention_softmax_row(const AttentionSoftmaxRowWitness &witness,
                                  const std::vector<ll> &scores, const QuantAttentionPlan &plan)
{
    return valid_plan(plan) && verify_attention_softmax_row_impl(witness, scores, plan, nullptr);
}

PreparedAttentionSoftmax::PreparedAttentionSoftmax(const QuantAttentionPlan &plan) : plan_(plan)
{
    if (!valid_plan(plan_))
        throw std::invalid_argument("invalid prepared attention plan");
    middle_table_.resize(attention_digit_base(plan_, plan_.nonlinear_segment));
    for (std::size_t i = 0; i < middle_table_.size(); ++i)
        middle_table_[i] = attention_middle_table_output(plan_, i);
    for (int high = 0; high < 2; ++high)
    {
        zero_tables_[high].assign(attention_digit_base(plan_, 4 + high), 0);
        zero_tables_[high][0] = 1;
    }
}
AttentionSoftmaxRowWitness PreparedAttentionSoftmax::build(const std::vector<ll> &scores) const
{
    return build_attention_softmax_row_impl(scores, plan_, &middle_table_);
}
bool PreparedAttentionSoftmax::verify(const AttentionSoftmaxRowWitness &witness,
                                      const std::vector<ll> &scores) const
{
    return verify_attention_softmax_row_impl(witness, scores, plan_, &middle_table_);
}
