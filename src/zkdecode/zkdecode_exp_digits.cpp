#include "zkdecode/zkdecode_exp_digits.hpp"
#include "common/bit_utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{

    int bit_length(ll value)
    {
        if (value < 0)
            value = -value;
        int bits = 0;
        while (value != 0)
        {
            value >>= 1;
            ++bits;
        }
        return std::max(bits, 1);
    }

    using zkagent_integer::is_power_of_two;

    bool valid_config(const DecodeExpDigitsConfig &config)
    {
        const DecodeExpDigitsConfig q20;
        return config.bases == q20.bases &&
            config.least_segments == q20.least_segments &&
            config.nonlinear_segment == q20.nonlinear_segment &&
            config.most_zero_segments == q20.most_zero_segments &&
            config.logit_scale == q20.logit_scale &&
            config.max_abs_logit == q20.max_abs_logit &&
            config.expected_theta == q20.expected_theta &&
            config.max_active_count == q20.max_active_count;
    }

    ll radix_product(const DecodeExpDigitsConfig &config)
    {
        ll product = 1;
        for (int base : config.bases)
        {
            if (bit_length(product) + decode_exp_digit_bits(base) >= 124)
            {
                throw std::overflow_error("mixed-radix product exceeds signed ll capacity");
            }
            product *= static_cast<ll>(base);
        }
        return product;
    }

    ll segment_weight(const DecodeExpDigitsConfig &config, int segment)
    {
        if (segment < 0 || segment >= static_cast<int>(config.bases.size()))
        {
            throw std::out_of_range("mixed-radix segment is out of range");
        }
        ll weight = 1;
        for (int k = 0; k < segment; ++k)
        {
            weight *= static_cast<ll>(config.bases[static_cast<size_t>(k)]);
        }
        return weight;
    }

    ll round_positive(long double value)
    {
        if (!(value >= 0.0L) || !std::isfinite(value) || value >= std::ldexp(1.0L, 120))
        {
            throw std::overflow_error("positive nearest-rounding input is invalid");
        }
        return static_cast<ll>(std::floor(value + 0.5L));
    }

    ll compute_table_output(
        ll digit, ll theta, const DecodeExpDigitsConfig &config)
    {
        const int middle = config.nonlinear_segment;
        const int base = config.bases[static_cast<size_t>(middle)];
        const ll weight = segment_weight(config, middle);
        const long double exponent =
            -static_cast<long double>(config.logit_scale) *
            static_cast<long double>(weight) * static_cast<long double>(digit);
        const long double scaled =
            static_cast<long double>(theta) * std::exp(exponent);
        return std::max<ll>(0, std::min<ll>(theta, round_positive(scaled)));
    }

    ll honest_shift(
        const std::vector<ll> &deltas,
        const std::vector<int> &active,
        const DecodeExpDigitsConfig &config)
    {
        ll maximum = 0;
        bool found = false;
        for (size_t i = 0; i < active.size(); ++i)
        {
            if (!active[i])
                continue;
            maximum = found ? std::max(maximum, deltas[i]) : deltas[i];
            found = true;
        }
        if (!found)
            throw std::invalid_argument("digit Softmax has no active entry");

        const long double alpha = static_cast<long double>(config.logit_scale);
        long double sum = 0.0L;
        for (size_t i = 0; i < active.size(); ++i)
        {
            if (!active[i])
                continue;
            sum += std::exp(alpha * static_cast<long double>(deltas[i] - maximum));
        }
        const ll extra = round_positive(std::log(sum) / alpha);
        return maximum + extra;
    }

}

const DecodeExpDigitsPublicTables &decode_exp_digits_public_tables(
    const DecodeExpDigitsConfig &config)
{
    if (!valid_config(config))
        throw std::invalid_argument("invalid digit public-table configuration");

    static const DecodeExpDigitsPublicTables tables = []
    {
        const DecodeExpDigitsConfig fixed;
        DecodeExpDigitsPublicTables result;
        for (std::size_t segment = 0; segment < result.inputs.size(); ++segment)
        {
            const int base = fixed.bases[segment];
            auto &inputs = result.inputs[segment];
            inputs.resize(static_cast<std::size_t>(base));
            for (int input = 0; input < base; ++input)
                inputs[static_cast<std::size_t>(input)] = input;
        }

        const std::size_t middle =
            static_cast<std::size_t>(fixed.nonlinear_segment);
        result.outputs[middle].resize(result.inputs[middle].size());
        for (std::size_t digit = 0; digit < result.outputs[middle].size(); ++digit)
        {
            result.outputs[middle][digit] = compute_table_output(
                static_cast<ll>(digit), fixed.expected_theta, fixed);
        }
        for (int high = 0; high < fixed.most_zero_segments; ++high)
        {
            const std::size_t segment = static_cast<std::size_t>(
                fixed.nonlinear_segment + 1 + high);
            result.outputs[segment].assign(result.inputs[segment].size(), 0);
            result.outputs[segment][0] = 1;
        }
        return result;
    }();
    return tables;
}

int decode_exp_digit_bits(int base)
{
    if (!is_power_of_two(base))
    {
        throw std::invalid_argument("digit base must be a power of two");
    }
    int bits = 0;
    while ((1 << bits) < base)
        ++bits;
    return bits;
}

ll decode_exp_digits_shift_upper_bound(
    int active_count,
    const DecodeExpDigitsConfig &config)
{
    if (!valid_config(config) || active_count <= 0)
    {
        throw std::invalid_argument("invalid digit shift-bound parameters");
    }
    const long double extra_real =
        std::log(static_cast<long double>(active_count)) / static_cast<long double>(config.logit_scale);
    const ll extra = round_positive(extra_real);
    if (bit_length(config.max_abs_logit) >= 122)
    {
        throw std::overflow_error("logit bound exceeds signed ll capacity");
    }
    return 2 * config.max_abs_logit + extra;
}

ll decode_exp_digits_table_output(
    ll digit,
    ll theta,
    const DecodeExpDigitsConfig &config)
{
    if (!valid_config(config) || theta <= 0 || theta != config.expected_theta)
    {
        throw std::invalid_argument("invalid digit exp-table parameters");
    }
    const std::size_t middle =
        static_cast<std::size_t>(config.nonlinear_segment);
    const int base = config.bases[middle];
    if (digit < 0 || digit >= base)
    {
        throw std::out_of_range("nonlinear exp digit is outside its table");
    }
    return decode_exp_digits_public_tables(config)
        .outputs[middle][static_cast<std::size_t>(digit)];
}

DecodeExpDigitsWitness build_decode_exp_digits_witness(
    const std::vector<ll> &deltas,
    const std::vector<int> &active,
    ll theta,
    const DecodeExpDigitsConfig &config)
{
    DecodeExpDigitsWitness result;
    result.config = config;
    result.output_scale = theta;
    if (!valid_config(config) || deltas.empty() || deltas.size() != active.size() || theta != config.expected_theta)
    {
        result.message = "invalid zkLLM digit-decomposition instance";
        return result;
    }

    try
    {
        int active_count = 0;
        const ll max_delta = 2 * config.max_abs_logit;
        for (size_t i = 0; i < active.size(); ++i)
        {
            if (active[i] != 0 && active[i] != 1)
            {
                result.message = "digit selector is not boolean";
                return result;
            }
            if (!active[i])
                continue;
            ++active_count;
            if (deltas[i] < 0 || deltas[i] > max_delta)
            {
                result.message = "active digit delta is outside the public logit bound";
                return result;
            }
        }
        if (active_count == 0)
        {
            result.message = "digit Softmax has no active entry";
            return result;
        }
        if (active_count > config.max_active_count)
        {
            result.message = "digit Softmax active count exceeds the Q20 contract";
            return result;
        }

        const ll radix_domain = radix_product(config);
        const ll shift_upper_bound = decode_exp_digits_shift_upper_bound(
            active_count, config);
        if (shift_upper_bound >= radix_domain)
        {
            result.message = "mixed-radix domain does not cover the public wide bound";
            return result;
        }
        result.shift = honest_shift(deltas, active, config);
        if (result.shift < 0 || result.shift > shift_upper_bound)
        {
            result.message = "honest digit shift is outside its public bound";
            return result;
        }

        const size_t n = deltas.size();
        const int segments = static_cast<int>(config.bases.size());
        result.digits.assign(
            static_cast<size_t>(segments), std::vector<ll>(n, 0));
        result.nonlinear_outputs.assign(n, 0);
        result.high_zero_flags.assign(
            static_cast<size_t>(config.most_zero_segments),
            std::vector<ll>(n, 1));
        result.unmasked_outputs.assign(n, 0);
        const auto &public_tables = decode_exp_digits_public_tables(config);

        for (size_t i = 0; i < n; ++i)
        {
            const ll magnitude = active[i] ? result.shift - deltas[i] : 0;
            if (magnitude < 0 || magnitude >= radix_domain)
            {
                result.message = "digit magnitude is outside the mixed-radix domain";
                return result;
            }
            ll remaining = magnitude;
            for (int k = 0; k < segments; ++k)
            {
                const int base = config.bases[static_cast<size_t>(k)];
                result.digits[static_cast<size_t>(k)][i] = remaining % base;
                remaining /= base;
            }
            if (remaining != 0)
            {
                result.message = "digit decomposition left a nonzero quotient";
                return result;
            }

            const ll middle_digit =
                result.digits[static_cast<size_t>(config.nonlinear_segment)][i];
            result.nonlinear_outputs[i] =
                public_tables.outputs[static_cast<std::size_t>(config.nonlinear_segment)]
                                     [static_cast<std::size_t>(middle_digit)];
            bool keep = true;
            for (int j = 0; j < config.most_zero_segments; ++j)
            {
                const int segment = config.nonlinear_segment + 1 + j;
                const ll digit = result.digits[static_cast<size_t>(segment)][i];
                const ll flag = digit == 0 ? 1 : 0;
                result.high_zero_flags[static_cast<size_t>(j)][i] = flag;
                keep = keep && flag == 1;
            }
            result.unmasked_outputs[i] = keep
                                             ? result.nonlinear_outputs[i]
                                             : 0;
        }
    }
    catch (const std::exception &error)
    {
        result.message = error.what();
        return result;
    }

    result.success = true;
    result.message = "Q20 zkLLM mixed-radix exp witness constructed";
    return result;
}

ll decode_exp_digits_recommended_epsilon(
    ll theta,
    int active_count,
    const DecodeExpDigitsConfig &config)
{
    if (!valid_config(config) || theta != config.expected_theta || active_count <= 0)
    {
        throw std::invalid_argument("invalid digit epsilon parameters");
    }
    const long double alpha = static_cast<long double>(config.logit_scale);
    const ll low_boundary = segment_weight(config, config.nonlinear_segment);
    const ll high_boundary = low_boundary * static_cast<ll>(config.bases[static_cast<size_t>(config.nonlinear_segment)]);
    const long double shift_and_low =
        static_cast<long double>(theta) * (std::exp(alpha * (static_cast<long double>(low_boundary - 1) + 0.5L)) - 1.0L);
    const long double table_rounding =
        static_cast<long double>(active_count) / 2.0L;
    const long double high_tail =
        static_cast<long double>(active_count) * static_cast<long double>(theta) * std::exp(-alpha * (static_cast<long double>(high_boundary) - 0.5L));
    return static_cast<ll>(std::ceil(
        shift_and_low + table_rounding + high_tail));
}
