#include "common/mle.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mle
{
    PrefixEvaluation prefix_and_equality(std::size_t count, const std::vector<mcl::bn::Fr> &point)
    {
        using mcl::bn::Fr;
        if (point.size() >= std::numeric_limits<std::size_t>::digits ||
            count > (std::size_t(1) << point.size()))
            throw std::invalid_argument("prefix exceeds its Boolean domain");
        if (count == (std::size_t(1) << point.size()))
            return {Fr(1), Fr(0)};
        PrefixEvaluation result;
        for (std::size_t bit = point.size(); bit-- > 0;)
        {
            if ((count >> bit) & 1)
            {
                result.prefix += result.equality * (Fr(1) - point[bit]);
                result.equality *= point[bit];
            }
            else
                result.equality *= Fr(1) - point[bit];
        }
        return result;
    }

    mcl::bn::Fr prefix_sum(std::size_t count, const std::vector<mcl::bn::Fr> &point)
    {
        return prefix_and_equality(count, point).prefix;
    }

    mcl::bn::Fr equality_at(std::size_t index, const std::vector<mcl::bn::Fr> &point)
    {
        using mcl::bn::Fr;
        if (point.size() >= std::numeric_limits<std::size_t>::digits ||
            index >= (std::size_t(1) << point.size()))
            throw std::out_of_range("equality index exceeds its Boolean domain");
        Fr result(1);
        for (std::size_t bit = 0; bit < point.size(); ++bit)
            result *= ((index >> bit) & 1) ? point[bit] : Fr(1) - point[bit];
        return result;
    }

    mcl::bn::Fr evaluate(std::vector<mcl::bn::Fr> values, const mcl::bn::Fr *point,
                        std::size_t variables, int threads)
    {
        using mcl::bn::Fr;
        if (variables >= std::numeric_limits<std::size_t>::digits ||
            values.size() != (std::size_t(1) << variables) ||
            (variables != 0 && point == nullptr))
            throw std::invalid_argument("MLE vector and point dimensions differ");
        std::size_t active = values.size();
        std::vector<Fr> scratch;
        for (std::size_t bit = 0; bit < variables; ++bit)
        {
            const std::size_t half = active / 2;
            const Fr coordinate = point[bit];
            if (threads > 1 && half >= 4096)
            {
                scratch.resize(half);
#pragma omp parallel for schedule(static) num_threads(threads)
                for (long long raw = 0; raw < static_cast<long long>(half); ++raw)
                {
                    const std::size_t i = static_cast<std::size_t>(raw);
                    scratch[i] = values[2 * i] + (values[2 * i + 1] - values[2 * i]) * coordinate;
                }
                values.swap(scratch);
            }
            else
            {
                // An in-place fold must remain serial because its reads alias later writes.
                for (std::size_t i = 0; i < half; ++i)
                    values[i] = values[2 * i] + (values[2 * i + 1] - values[2 * i]) * coordinate;
            }
            active = half;
        }
        return values[0];
    }

    mcl::bn::Fr evaluate(std::vector<mcl::bn::Fr> values,
                        const std::vector<mcl::bn::Fr> &point, int threads)
    {
        return evaluate(std::move(values), point.data(), point.size(), threads);
    }

    std::vector<mcl::bn::Fr> equality_weights(
        const mcl::bn::Fr *point, std::size_t variables,
        int requested_threads, const mcl::bn::Fr &initial)
    {
        using mcl::bn::Fr;
        if (variables >= std::numeric_limits<std::size_t>::digits ||
            (variables != 0 && point == nullptr))
            throw std::invalid_argument("invalid equality-table point dimension");

        std::vector<Fr> weights(std::size_t(1) << variables);
        weights[0] = initial;
        const int threads = std::max(1, requested_threads);
        std::size_t parallel_begin = variables;
        for (std::size_t bit = 0; bit < variables; ++bit)
        {
            const std::size_t active = std::size_t(1) << bit;
            if (threads > 1 && active >= static_cast<std::size_t>(threads) * 256)
            {
                parallel_begin = bit;
                break;
            }
            const Fr coordinate = point[bit];
            for (std::size_t i = 0; i < active; ++i)
            {
                const Fr base = weights[i];
                const Fr high = base * coordinate;
                weights[i] = base - high;
                weights[i + active] = high;
            }
        }
        if (parallel_begin < variables)
        {
#pragma omp parallel num_threads(threads)
            {
                for (std::size_t bit = parallel_begin; bit < variables; ++bit)
                {
                    const std::size_t active = std::size_t(1) << bit;
                    const Fr coordinate = point[bit];
#pragma omp for schedule(static)
                    for (long long raw = 0; raw < static_cast<long long>(active); ++raw)
                    {
                        const std::size_t i = static_cast<std::size_t>(raw);
                        const Fr base = weights[i];
                        const Fr high = base * coordinate;
                        weights[i] = base - high;
                        weights[i + active] = high;
                    }
                }
            }
        }
        return weights;
    }
}
