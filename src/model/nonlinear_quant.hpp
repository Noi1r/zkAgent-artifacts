#pragma once

#include <mcl/gmp_util.hpp>
#include <cstddef>

namespace zkagent_integer {

struct RoundedNonlinear {
    __int128 value, lower, upper;
};

struct LayerNormCoefficients {
    __int128 a, b, denominator;
};

struct LayerNormVariance {
    mpz_class value;
    __int128 sigma, lower, upper;
};

LayerNormVariance layernorm_variance(const __int128* centered,
                                    std::size_t count, unsigned variance_bits);

LayerNormCoefficients layernorm_coefficients(int c1, int e1, int c2, int e2);
RoundedNonlinear layernorm_round(const LayerNormCoefficients& coefficients,
                                __int128 gamma, __int128 centered,
                                __int128 beta, __int128 sigma);

struct GeluCoefficients {
    __int128 c1, c2, c3, c4, c5, c6, c7;
};

struct GeluValue {
    RoundedNonlinear rounded;
    __int128 magnitude, selector, d1, d2;
};

GeluCoefficients gelu_coefficients(int scale_c, int scale_e);
GeluValue gelu_round(const GeluCoefficients& coefficients, __int128 input);

}
