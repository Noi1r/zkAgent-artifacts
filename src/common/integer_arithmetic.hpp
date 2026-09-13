#pragma once

#include <limits>
#include <stdexcept>

namespace zkagent_integer
{
    template <class Result = __int128>
    Result floor_div_i128(__int128 numerator, __int128 denominator)
    {
        if (denominator <= 0)
            throw std::invalid_argument("integer division requires a positive denominator");
        __int128 quotient = numerator / denominator;
        if (numerator % denominator < 0)
            --quotient;
        if (quotient < std::numeric_limits<Result>::min() ||
            quotient > std::numeric_limits<Result>::max())
            throw std::overflow_error("integer quotient exceeds the result type");
        return static_cast<Result>(quotient);
    }

    // Round to nearest with half-integers rounded toward positive infinity.
    template <class Result = __int128, class Dyadic>
    Result round_dyadic(__int128 value, const Dyadic &scale)
    {
        return floor_div_i128<Result>(
            2 * scale.c1 * value + scale.c2,
            static_cast<__int128>(2) * scale.c2);
    }
}
