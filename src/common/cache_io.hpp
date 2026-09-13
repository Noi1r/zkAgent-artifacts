#pragma once

#include <array>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <type_traits>

namespace cache_io
{
    // Native POD cache fields and explicit little-endian fields are distinct formats.
    template <class T>
    void write_native(std::ostream &out, const T &value)
    {
        static_assert(std::is_trivially_copyable<T>::value, "cache field must be POD");
        out.write(reinterpret_cast<const char *>(&value), sizeof(value));
        if (!out)
            throw std::runtime_error("failed to write cache field");
    }

    template <class T>
    T read_native(std::istream &in)
    {
        static_assert(std::is_trivially_copyable<T>::value, "cache field must be POD");
        T value{};
        in.read(reinterpret_cast<char *>(&value), sizeof(value));
        if (!in)
            throw std::runtime_error("truncated cache field");
        return value;
    }

    inline void write_u64(std::ostream &out, std::uint64_t value)
    {
        for (int i = 0; i < 8; ++i)
        {
            out.put(static_cast<char>(value & 255));
            value >>= 8;
        }
    }

    inline std::uint64_t read_u64(std::istream &in)
    {
        std::uint64_t value = 0;
        for (int i = 0; i < 8; ++i)
        {
            const int byte = in.get();
            if (byte == std::char_traits<char>::eof())
                throw std::runtime_error("truncated little-endian cache field");
            value |= std::uint64_t(static_cast<unsigned char>(byte)) << (8 * i);
        }
        return value;
    }

    template <class Point, std::size_t Capacity = 256>
    void write_point(std::ostream &out, const Point &point)
    {
        std::array<char, Capacity> bytes{};
        const auto size = point.serialize(bytes.data(), bytes.size());
        if (!size)
            throw std::runtime_error("cache point serialization failed");
        write_u64(out, size);
        out.write(bytes.data(), size);
    }

    template <class Point, std::size_t Capacity = 256>
    Point read_point(std::istream &in)
    {
        std::array<char, Capacity> bytes{};
        const auto size = read_u64(in);
        if (!size || size > bytes.size())
            throw std::runtime_error("invalid cache point length");
        in.read(bytes.data(), size);
        Point point;
        if (!in || point.deserialize(bytes.data(), size) != size)
            throw std::runtime_error("invalid cache point encoding");
        return point;
    }
}
