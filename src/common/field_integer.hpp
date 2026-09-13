#pragma once

#include <mcl/bn256.hpp>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace zkagent_integer
{
    inline bool decode_nonnegative_i128(const mcl::bn::Fr &value, __int128 bound,
                                        __int128 &result)
    {
        if (bound < 0)
            return false;
        std::uint8_t bytes[16] = {};
        const std::size_t count = value.getLittleEndian(bytes, sizeof(bytes));
        if (count == 0)
            return false;
        unsigned __int128 magnitude = 0;
        for (std::size_t i = count; i-- > 0;)
            magnitude = (magnitude << 8) | bytes[i];
        if (magnitude > static_cast<unsigned __int128>(bound))
            return false;
        result = static_cast<__int128>(magnitude);
        return true;
    }

    inline __int128 fr_to_i128(const mcl::bn::Fr &value)
    {
        const bool negative = value.isNegative();
        const mcl::bn::Fr magnitude = negative ? -value : value;
        __int128 result;
        if (!decode_nonnegative_i128(magnitude, std::numeric_limits<__int128>::max(), result))
            throw std::overflow_error("field magnitude exceeds signed 128-bit range");
        return negative ? -result : result;
    }

    inline bool decode_bounded_i64(const mcl::bn::Fr &value, std::int64_t bound,
                                   std::int64_t &result)
    {
        if (bound < 0)
            return false;
        const bool negative = value.isNegative();
        const mcl::bn::Fr absolute = negative ? -value : value;
        std::uint8_t bytes[8] = {};
        const std::size_t count = absolute.getLittleEndian(bytes, sizeof(bytes));
        if (count == 0)
            return false;
        std::uint64_t magnitude = 0;
        for (std::size_t i = count; i-- > 0;)
            magnitude = (magnitude << 8) | bytes[i];
        if (magnitude > static_cast<std::uint64_t>(bound))
            return false;
        result = negative ? -static_cast<std::int64_t>(magnitude) : static_cast<std::int64_t>(magnitude);
        return true;
    }

    inline mcl::bn::Fr fr_from_i128(__int128 value)
    {
        if (value >= std::numeric_limits<int64_t>::min() &&
            value <= std::numeric_limits<int64_t>::max())
            return mcl::bn::Fr(static_cast<int64_t>(value));
        const bool negative = value < 0;
        auto magnitude = negative ? -static_cast<unsigned __int128>(value)
                                  : static_cast<unsigned __int128>(value);
        uint8_t bytes[16] = {};
        for (auto &byte : bytes)
        {
            byte = static_cast<uint8_t>(magnitude);
            magnitude >>= 8;
        }
        mcl::bn::Fr result;
        result.setLittleEndianMod(bytes, sizeof(bytes));
        return negative ? -result : result;
    }

    inline mcl::bn::Fr fr_from_bigint(const mpz_class &value)
    {
        const mpz_class magnitude = mcl::gmp::abs(value);
        if (magnitude >= mcl::bn::Fr::getOp().mp)
            throw std::overflow_error("integer advice exceeds Fr modulus");
        mcl::bn::Fr result;
        result.setMpz(magnitude);
        return value < 0 ? -result : result;
    }
}
