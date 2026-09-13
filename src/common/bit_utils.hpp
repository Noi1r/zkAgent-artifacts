#pragma once

#include <limits>
#include <stdexcept>

namespace zkagent_integer
{
    template <class Integer>
    bool is_power_of_two(Integer value)
    {
        return value > 0 && (value & (value - 1)) == 0;
    }

    // The one-element domain needs no variables; zero also maps to zero.
    template <class Integer>
    int ceil_log2(Integer value)
    {
        if (value <= 1)
            return 0;
        --value;
        int bits = 0;
        while (value != 0)
        {
            value >>= 1;
            ++bits;
        }
        return bits;
    }

    template <class Integer>
    int log2_power_of_two(
        Integer value,
        const char *error_message = "dimension is not a positive power of two")
    {
        if (!is_power_of_two(value))
            throw std::invalid_argument(error_message);
        return ceil_log2(value);
    }

    // Preserve the input type so a padded int cannot silently overflow.
    template <class Integer>
    Integer next_power_of_two(
        Integer value,
        const char *overflow_message = "power-of-two domain exceeds integer capacity")
    {
        Integer result = 1;
        while (result < value)
        {
            if (result > std::numeric_limits<Integer>::max() / 2)
                throw std::overflow_error(overflow_message);
            result <<= 1;
        }
        return result;
    }
}
