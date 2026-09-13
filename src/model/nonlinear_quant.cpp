#include "model/nonlinear_quant.hpp"
#include "common/integer_sqrt.hpp"

#include <mcl/gmp_util.hpp>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace zkagent_integer
{
    namespace
    {
        using Big = mpz_class;

        Big wide(__int128 value)
        {
            const auto magnitude = value < 0 ? -static_cast<unsigned __int128>(value)
                                             : static_cast<unsigned __int128>(value);
            const uint64_t words[2] = {static_cast<uint64_t>(magnitude),
                                       static_cast<uint64_t>(magnitude >> 64)};
            Big result;
            mcl::gmp::setArray(result, words, 2);
            return value < 0 ? -result : result;
        }

        __int128 narrow(const Big &value)
        {
            const Big magnitude = mcl::gmp::abs(value);
            if (magnitude >= (Big(1) << 126))
                throw std::overflow_error("nonlinear coefficient/advice exceeds 126 bits");
            uint64_t words[2] = {};
            bool ok = false;
            mcl::gmp::getArray(&ok, words, 2, magnitude);
            if (!ok)
                throw std::overflow_error("nonlinear integer conversion failed");
            const __int128 result = static_cast<__int128>(
                (static_cast<unsigned __int128>(words[1]) << 64) | words[0]);
            return value < 0 ? -result : result;
        }

        Big power(int exponent)
        {
            if (exponent < 0 || exponent >= 126)
                throw std::overflow_error("nonlinear gate shift exceeds supported integer width");
            return Big(1) << exponent;
        }

        RoundedNonlinear rounded(const Big &numerator, const Big &denominator)
        {
            if (denominator <= 0)
                throw std::invalid_argument("non-positive nonlinear denominator");
            if (mcl::gmp::abs(numerator) * 2 + denominator + 2 >= (Big(1) << 253))
                throw std::overflow_error("nonlinear relation may wrap BN254 Fr");
            Big q = (mcl::gmp::abs(numerator) * 2 + denominator) / (denominator * 2);
            if (numerator < 0)
                q = -q;
            const Big residual = numerator - q * denominator;
            return {narrow(q), narrow(denominator / 2 + 1 - residual),
                    narrow(denominator / 2 + 1 + residual)};
        }
    }

    LayerNormVariance layernorm_variance(const __int128 *centered,
                                         std::size_t count, unsigned variance_bits)
    {
        if (!centered || count == 0 || variance_bits == 0 || variance_bits > 253)
            throw std::invalid_argument("invalid LayerNorm variance shape/width");

        if (variance_bits <= 126)
        {
            using U128 = unsigned __int128;
            const U128 limit = U128(1) << variance_bits;
            U128 value = 1;
            for (std::size_t i = 0; i < count; ++i)
            {
                const U128 a = centered[i] < 0 ? -U128(centered[i]) : U128(centered[i]);
                if (a >= (U128(1) << 63))
                    throw std::overflow_error("LayerNorm variance exceeds offline bound");
                value += a * a;
                if (value >= limit)
                    throw std::overflow_error("LayerNorm variance exceeds offline bound");
            }
            const __int128 v = static_cast<__int128>(value);
            const __int128 sigma = nearest_sqrt_i128(v);
            const __int128 delta = v - sigma * sigma;
            return {wide(v), sigma, delta + sigma - 1, sigma - delta};
        }

        Big value = 1;
        for (std::size_t i = 0; i < count; ++i)
        {
            const Big a = wide(centered[i]);
            value += a * a;
        }
        if (value >= (Big(1) << variance_bits))
            throw std::overflow_error("LayerNorm variance exceeds offline bound");

        Big root = Big(1) << ((mcl::gmp::getBitSize(value) + 1) / 2);
        for (;;)
        {
            const Big next = (root + value / root) >> 1;
            if (next >= root)
                break;
            root = next;
        }
        Big delta = value - root * root;
        if (delta > root)
        {
            delta -= root * 2 + 1;
            ++root;
        }
        return {value, narrow(root), narrow(delta + root - 1), narrow(root - delta)};
    }

    LayerNormCoefficients layernorm_coefficients(int c1, int e1, int c2, int e2)
    {
        if (c1 <= 0 || c2 <= 0 || e1 < -125 || e1 > 125 || e2 < -125 || e2 > 125)
            throw std::invalid_argument("unsupported LayerNorm coefficient representation");
        const int m = std::max({1, -e1, -e2});
        return {narrow(power(e1 + m) * c1), narrow(power(e2 + m) * c2), narrow(power(m))};
    }

    RoundedNonlinear layernorm_round(const LayerNormCoefficients &c,
                                     __int128 gamma, __int128 centered,
                                     __int128 beta, __int128 sigma)
    {
        if (sigma <= 0)
            throw std::invalid_argument("non-positive LayerNorm sigma");
        return rounded(wide(c.a) * wide(gamma) * wide(centered) + wide(c.b) * wide(beta) * wide(sigma),
                       wide(c.denominator) * wide(sigma));
    }

    GeluCoefficients gelu_coefficients(int c, int e)
    {
        if (c <= 0 || e < -125 || e > 125)
            throw std::invalid_argument("unsupported GELU coefficient representation");
        const int m1 = std::max({0, 8, 8 - e, 8 - 2 * e});
        const int m2 = std::max({0, 8, -e});
        const Big cx(c);
        const Big c1 = power(m1) * cx;
        const Big c2 = power(-8 + 2 * e + m1) * cx * 48 * cx * cx;
        const Big c3 = power(-8 + e + m1) * cx * 217 * cx;
        const Big c4 = power(-8 + m1) * cx * 252;
        const Big c6 = power(-8 + m2) * 615;
        const Big c7 = power(e + m2) * cx;
        narrow(c1 * 2 + 1);
        narrow(c6 * 2 + 1);
        narrow(c7 * 2);
        return {narrow(c1), narrow(c2), narrow(c3), narrow(c4), narrow(c1),
                narrow(c6), narrow(c7)};
    }

    GeluValue gelu_round(const GeluCoefficients &c, __int128 input)
    {
        const Big x = wide(input), a = mcl::gmp::abs(x);
        const Big delta = wide(c.c7) * a - wide(c.c6);
        const int t = delta <= 0 ? 1 : 0;
        const Big p = wide(c.c1) * (x + a) - wide(c.c2) * t * a * a * a + wide(c.c3) * t * a * a - wide(c.c4) * t * a;
        return {rounded(p, wide(c.c5) * 2), narrow(a), t, narrow(a + 1),
                narrow(Big(t) + delta * (1 - 2 * t))};
    }
}
