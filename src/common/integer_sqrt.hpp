#pragma once

#include <stdexcept>


namespace zkagent_integer {

inline unsigned __int128 floor_sqrt_u128(unsigned __int128 value) {
    unsigned __int128 result = 0;
    unsigned __int128 bit = static_cast<unsigned __int128>(1) << 126;

    while (bit > value) bit >>= 2;
    while (bit != 0) {
        if (value >= result + bit) {
            value -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    return result;
}

inline __int128 nearest_sqrt_i128(__int128 value) {
    if (value <= 0) {
        throw std::invalid_argument("nearest_sqrt_i128 requires a positive input");
    }
    const unsigned __int128 input = static_cast<unsigned __int128>(value);
    unsigned __int128 root = floor_sqrt_u128(input);
    const unsigned __int128 remainder = input - root * root;
    if (remainder > root) ++root;
    return static_cast<__int128>(root);
}

}
