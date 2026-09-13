#pragma once

#include <mcl/bn256.hpp>
#include <cstddef>
#include <vector>

namespace mle
{
    struct PrefixEvaluation
    {
        mcl::bn::Fr prefix = mcl::bn::Fr(0);
        mcl::bn::Fr equality = mcl::bn::Fr(1);
    };

    PrefixEvaluation prefix_and_equality(std::size_t count, const std::vector<mcl::bn::Fr> &point);
    mcl::bn::Fr prefix_sum(std::size_t count, const std::vector<mcl::bn::Fr> &point);
    mcl::bn::Fr equality_at(std::size_t index, const std::vector<mcl::bn::Fr> &point);

    mcl::bn::Fr evaluate(std::vector<mcl::bn::Fr> values, const mcl::bn::Fr *point,
                        std::size_t variables, int threads = 1);
    mcl::bn::Fr evaluate(std::vector<mcl::bn::Fr> values,
                        const std::vector<mcl::bn::Fr> &point, int threads = 1);

    // point[0] corresponds to the least significant bit of the table index.
    // Entry i is initial * product_b (bit_b(i) ? point[b] : 1-point[b]).
    std::vector<mcl::bn::Fr> equality_weights(
        const mcl::bn::Fr *point, std::size_t variables,
        int threads = 1, const mcl::bn::Fr &initial = mcl::bn::Fr(1));

    inline std::vector<mcl::bn::Fr> equality_weights(
        const std::vector<mcl::bn::Fr> &point,
        int threads = 1, const mcl::bn::Fr &initial = mcl::bn::Fr(1))
    {
        return equality_weights(point.data(), point.size(), threads, initial);
    }
}
