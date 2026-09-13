#ifndef ZKAGENT_FIAT_SHAMIR_HPP
#define ZKAGENT_FIAT_SHAMIR_HPP

#include "common/global_var.hpp"

#include <cybozu/sha2.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

enum class G1WireEncoding : std::uint64_t
{
    Affine64 = 0,
    Compressed = 1,
};

inline const char *g1_wire_encoding_name(G1WireEncoding encoding)
{
    return encoding == G1WireEncoding::Affine64
               ? "affine64"
               : "compressed";
}

inline std::size_t g1_wire_size(G1WireEncoding encoding)
{
    if (encoding == G1WireEncoding::Affine64)
        return 64;
    return G1::getSerializedByteSize();
}

inline int g1_wire_io_mode(G1WireEncoding encoding)
{
    return encoding == G1WireEncoding::Affine64
               ? mcl::IoEcAffineSerialize
               : mcl::IoSerialize;
}

inline void encode_g1_vector_canonical_into(
    const G1 *values, std::size_t count, G1WireEncoding encoding,
    std::uint8_t *output, std::size_t output_size)
{
    if (values == nullptr && count != 0)
    {
        throw std::invalid_argument("cannot encode a null G1 vector");
    }
    const std::size_t point_size = g1_wire_size(encoding);
    if (count > std::numeric_limits<std::size_t>::max() / point_size)
    {
        throw std::length_error("G1 wire vector is too large");
    }
    const std::size_t required = count * point_size;
    if (output_size != required || (output == nullptr && required != 0))
    {
        throw std::invalid_argument("invalid canonical G1 output buffer");
    }
    bool already_normalized = true;
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!values[i].isNormalized())
        {
            already_normalized = false;
            break;
        }
    }
    if (already_normalized)
    {
        std::atomic<bool> valid(true);
#pragma omp parallel for schedule(static) if (count >= 1024)
        for (long long raw_i = 0;
             raw_i < static_cast<long long>(count); ++raw_i)
        {
            const std::size_t i = static_cast<std::size_t>(raw_i);
            const std::size_t written = values[i].serialize(
                output + i * point_size,
                point_size, g1_wire_io_mode(encoding));
            if (written != point_size)
                valid.store(false);
        }
        if (!valid.load())
        {
            throw std::runtime_error(
                "canonical G1 serialization returned an unexpected size");
        }
        return;
    }
    constexpr std::size_t NormalizeChunk = 4096;
    const std::size_t blocks =
        (count + NormalizeChunk - 1) / NormalizeChunk;
    std::atomic<bool> valid(true);
#pragma omp parallel if (count >= NormalizeChunk)
    {
        std::vector<G1> normalized(NormalizeChunk);
#pragma omp for schedule(static)
        for (long long block = 0;
             block < static_cast<long long>(blocks); ++block)
        {
            const std::size_t offset =
                static_cast<std::size_t>(block) * NormalizeChunk;
            const std::size_t chunk =
                std::min(NormalizeChunk, count - offset);
            G1::normalizeVec(normalized.data(), values + offset, chunk);
            for (std::size_t i = 0; i < chunk; ++i)
            {
                const std::size_t written = normalized[i].serialize(
                    output + (offset + i) * point_size,
                    point_size, g1_wire_io_mode(encoding));
                if (written != point_size)
                    valid.store(false);
            }
        }
    }
    if (!valid.load())
    {
        throw std::runtime_error(
            "canonical G1 serialization returned an unexpected size");
    }
}

inline std::vector<std::uint8_t> encode_g1_vector_canonical(
    const G1 *values, std::size_t count, G1WireEncoding encoding)
{
    const std::size_t point_size = g1_wire_size(encoding);
    if (count > std::numeric_limits<std::size_t>::max() / point_size)
    {
        throw std::length_error("G1 wire vector is too large");
    }
    std::vector<std::uint8_t> bytes(count * point_size);
    encode_g1_vector_canonical_into(
        values, count, encoding, bytes.data(), bytes.size());
    return bytes;
}

inline std::vector<G1> decode_g1_vector_canonical(
    const std::uint8_t *bytes, std::size_t byte_count,
    std::size_t point_count, G1WireEncoding encoding)
{
    const std::size_t point_size = g1_wire_size(encoding);
    if (point_count > std::numeric_limits<std::size_t>::max() / point_size || byte_count != point_count * point_size || (bytes == nullptr && byte_count != 0))
    {
        throw std::invalid_argument("invalid canonical G1-vector length");
    }
    std::vector<G1> values(point_count);
    std::atomic<bool> valid(true);
#pragma omp parallel for schedule(static) if (point_count >= 1024)
    for (long long raw_i = 0;
         raw_i < static_cast<long long>(point_count); ++raw_i)
    {
        const std::size_t i = static_cast<std::size_t>(raw_i);
        std::array<std::uint8_t, 96> canonical{};
        const std::uint8_t *encoded = bytes + i * point_size;
        const std::size_t consumed = values[i].deserialize(
            encoded, point_size, g1_wire_io_mode(encoding));
        if (consumed != point_size)
        {
            valid.store(false);
            continue;
        }
        const std::size_t written = values[i].serialize(
            canonical.data(), point_size, g1_wire_io_mode(encoding));
        if (written != point_size || std::memcmp(canonical.data(), encoded, point_size) != 0)
        {
            valid.store(false);
        }
    }
    if (!valid.load())
        throw std::invalid_argument("invalid G1 encoding");
    return values;
}

inline std::vector<std::uint8_t> encode_fr_vector_canonical(
    const Fr *values, std::size_t count)
{
    if (values == nullptr && count != 0)
    {
        throw std::invalid_argument("cannot encode a null Fr vector");
    }
    std::vector<std::uint8_t> bytes(count * 32);
    for (std::size_t i = 0; i < count; ++i)
    {
        const std::size_t written = values[i].serialize(
            bytes.data() + i * 32, 32);
        if (written != 32)
        {
            throw std::runtime_error(
                "canonical Fr serialization returned an unexpected size");
        }
    }
    return bytes;
}

inline std::vector<Fr> decode_fr_vector_canonical(
    const std::uint8_t *bytes, std::size_t byte_count,
    std::size_t element_count)
{
    if (element_count > std::numeric_limits<std::size_t>::max() / 32 || byte_count != element_count * 32 || (bytes == nullptr && byte_count != 0))
    {
        throw std::invalid_argument("invalid canonical Fr-vector length");
    }
    std::vector<Fr> values(element_count);
    std::array<std::uint8_t, 32> canonical{};
    for (std::size_t i = 0; i < element_count; ++i)
    {
        const std::uint8_t *encoded = bytes + i * 32;
        const std::size_t consumed = values[i].deserialize(encoded, 32);
        if (consumed != 32)
        {
            throw std::invalid_argument("invalid Fr encoding");
        }
        const std::size_t written = values[i].serialize(
            canonical.data(), canonical.size());
        if (written != 32 || std::memcmp(canonical.data(), encoded, 32) != 0)
        {
            throw std::invalid_argument("non-canonical Fr encoding");
        }
    }
    return values;
}

namespace fiat_shamir_detail
{
inline void store_u64(std::uint8_t *out, std::uint64_t value)
{
    for (int i = 7; i >= 0; --i)
    {
        out[i] = static_cast<std::uint8_t>(value);
        value >>= 8;
    }
}

inline std::uint64_t load_u64(const std::uint8_t *input)
{
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i)
        value = (value << 8) | input[i];
    return value;
}
}

class FiatShamirTranscript
{
public:
    void append_u64(std::uint64_t value)
    {
        std::array<std::uint8_t, 8> bytes{};
        fiat_shamir_detail::store_u64(bytes.data(), value);
        append_record(1, bytes.data(), bytes.size());
    }

    void append_fr(const Fr &value)
    {
        std::array<std::uint8_t, 32> bytes{};
        if (value.serialize(bytes.data(), bytes.size()) != bytes.size())
            throw std::runtime_error("Fr serialization failed");
        append_record(3, bytes.data(), bytes.size());
    }

    void append_g1(const G1 &value, G1WireEncoding encoding = G1WireEncoding::Affine64)
    {
        const auto bytes = encode_g1_vector_canonical(&value, 1, encoding);
        append_record(4, bytes.data(), bytes.size());
    }

    void append_g2(const G2 &value)
    {
        std::array<std::uint8_t, 192> bytes{};
        const std::size_t size = value.serialize(bytes.data(), bytes.size(), mcl::IoEcAffineSerialize);
        if (size == 0)
            throw std::runtime_error("G2 serialization failed");
        append_record(7, bytes.data(), size);
    }

    void append_fr_vector(const std::vector<Fr> &values)
    {
        const auto bytes = encode_fr_vector_canonical(values.data(), values.size());
        append_record(5, bytes.data(), bytes.size());
    }

    void append_g1_vector(const std::vector<G1> &values,
                          G1WireEncoding encoding = G1WireEncoding::Affine64)
    {
        append_g1_array(values.data(), values.size(), encoding);
    }

    void append_g1_array(const G1 *values, std::size_t count,
                         G1WireEncoding encoding = G1WireEncoding::Affine64)
    {
        const std::size_t point_size = g1_wire_size(encoding);
        if (count > (std::numeric_limits<std::size_t>::max() - 8) / point_size)
            throw std::length_error("G1 transcript vector is too large");
        std::vector<std::uint8_t> bytes(8 + count * point_size);
        fiat_shamir_detail::store_u64(bytes.data(), count);
        encode_g1_vector_canonical_into(values, count, encoding,
                                        bytes.data() + 8, bytes.size() - 8);
        append_record(6, bytes.data(), bytes.size());
    }

    Fr challenge()
    {
        const std::size_t field_bits = Fr::getModBitLen();
        if (field_bits == 0 || field_bits > 256)
            throw std::runtime_error("unsupported scalar-field bit length");
        static const auto modulus = scalar_modulus_bytes();
        for (;;)
        {
            // Type zero is reserved for squeezing; advance even when rejecting a sample.
            append_record(0, nullptr, 0);
            auto sample = state_;
            const std::size_t excess = 256 - field_bits;
            std::fill(sample.begin(), sample.begin() + excess / 8, 0);
            if (excess % 8)
                sample[excess / 8] &= static_cast<std::uint8_t>(0xffu >> (excess % 8));
            if (std::lexicographical_compare(sample.begin(), sample.end(),
                                             modulus.begin(), modulus.end()))
            {
                Fr result;
                result.setBigEndianMod(sample.data(), sample.size());
                return result;
            }
        }
    }

    Fr challenge_excluding(const std::vector<Fr> &excluded)
    {
        for (;;)
        {
            const Fr result = challenge();
            if (std::find(excluded.begin(), excluded.end(), result) == excluded.end())
                return result;
        }
    }

    Fr challenge_not_one()
    {
        Fr result;
        do { result = challenge(); } while (result == Fr(1));
        return result;
    }

    Fr nonzero_challenge()
    {
        Fr result;
        do { result = challenge(); } while (result.isZero());
        return result;
    }

    void absorb_encoded_record(const std::uint8_t *record, std::size_t size)
    {
        cybozu::Sha256 hash;
        hash.update(state_.data(), state_.size());
        if (hash.digest(state_.data(), state_.size(), record, size) != state_.size())
            throw std::runtime_error("Fiat--Shamir SHA-256 failed");
    }

private:
    std::array<std::uint8_t, 32> state_{};

    void append_record(std::uint8_t type, const std::uint8_t *payload, std::size_t size)
    {
        std::array<std::uint8_t, 9> header{};
        header[0] = type;
        fiat_shamir_detail::store_u64(header.data() + 1, size);
        cybozu::Sha256 hash;
        hash.update(state_.data(), state_.size());
        hash.update(header.data(), header.size());
        if (hash.digest(state_.data(), state_.size(), payload, size) != state_.size())
            throw std::runtime_error("Fiat--Shamir SHA-256 failed");
    }

    static std::array<std::uint8_t, 32> scalar_modulus_bytes()
    {
        std::array<std::uint8_t, 32> result{};
        const std::string decimal_modulus = Fr::getModulo();
        for (char digit_char : decimal_modulus)
        {
            if (digit_char < '0' || digit_char > '9')
            {
                throw std::runtime_error("invalid scalar-field modulus");
            }
            unsigned carry = static_cast<unsigned>(digit_char - '0');
            for (int i = static_cast<int>(result.size()) - 1;
                 i >= 0; --i)
            {
                const unsigned value =
                    static_cast<unsigned>(result[static_cast<std::size_t>(i)]) * 10u + carry;
                result[static_cast<std::size_t>(i)] =
                    static_cast<std::uint8_t>(value & 0xffu);
                carry = value >> 8;
            }
            if (carry != 0)
            {
                throw std::runtime_error(
                    "scalar-field modulus exceeds 256 bits");
            }
        }
        return result;
    }

};

class FiatShamirProofWriter
{
public:
    FiatShamirProofWriter(G1WireEncoding encoding, bool retain_wire_bytes = true,
                         FiatShamirTranscript *shared_transcript = nullptr)
        : owned_transcript_(shared_transcript ? nullptr : new FiatShamirTranscript()),
          transcript_(shared_transcript ? shared_transcript : owned_transcript_.get()),
          encoding_(encoding), retain_wire_bytes_(retain_wire_bytes)
    {
        write_u64(static_cast<std::uint64_t>(encoding));
    }

    FiatShamirTranscript &transcript() { return *transcript_; }
    G1WireEncoding encoding() const { return encoding_; }
    // Fr and compressed G1 payload only, independent of wire encoding/retention.
    std::size_t compressed_payload_bytes() const { return compressed_payload_bytes_; }

    void write_u64(std::uint64_t value)
    {
        append_record(1, 8, [&](std::uint8_t *payload) {
            fiat_shamir_detail::store_u64(payload, value);
        });
    }

    void write_fr(const Fr &value)
    {
        append_record(3, 32, [&](std::uint8_t *payload) {
            if (value.serialize(payload, 32) != 32)
                throw std::runtime_error("Fr serialization failed");
        });
        account_compressed_payload(32);
    }

    void write_g1_vector(const std::vector<G1> &values)
    {
        write_g1_array(values.data(), values.size());
    }

    void write_g1_array(const G1 *values, std::size_t count)
    {
        const std::size_t point_size = g1_wire_size(encoding_);
        if (count > (std::numeric_limits<std::size_t>::max() - 17) / point_size)
            throw std::length_error("G1 proof record is too large");
        append_record(6, 8 + count * point_size, [&](std::uint8_t *payload) {
            fiat_shamir_detail::store_u64(payload, count);
            encode_g1_vector_canonical_into(values, count, encoding_,
                                            payload + 8, count * point_size);
        });
        account_compressed_payload(count * g1_wire_size(G1WireEncoding::Compressed));
    }

    Fr challenge() { return transcript_->challenge(); }
    Fr challenge_excluding(const std::vector<Fr> &excluded)
    {
        return transcript_->challenge_excluding(excluded);
    }
    Fr challenge_not_one() { return transcript_->challenge_not_one(); }

    const std::vector<std::uint8_t> &bytes() const { return bytes_; }
    std::vector<std::uint8_t> take_bytes() { return std::move(bytes_); }

private:
    std::unique_ptr<FiatShamirTranscript> owned_transcript_;
    FiatShamirTranscript *transcript_;
    G1WireEncoding encoding_;
    bool retain_wire_bytes_;
    std::vector<std::uint8_t> bytes_;
    std::size_t compressed_payload_bytes_ = 0;

    void account_compressed_payload(std::size_t size)
    {
        if (size > std::numeric_limits<std::size_t>::max() - compressed_payload_bytes_)
            throw std::length_error("compressed proof payload is too large");
        compressed_payload_bytes_ += size;
    }

    template <class Fill>
    void append_record(std::uint8_t type, std::size_t payload_size, Fill &&fill)
    {
        if (payload_size > std::numeric_limits<std::size_t>::max() - 9)
            throw std::length_error("proof record is too large");
        const std::size_t record_size = 9 + payload_size;
        std::vector<std::uint8_t> temporary;
        auto &buffer = retain_wire_bytes_ ? bytes_ : temporary;
        const std::size_t offset = buffer.size();
        if (record_size > std::numeric_limits<std::size_t>::max() - offset)
            throw std::length_error("proof byte stream is too large");
        buffer.resize(offset + record_size);
        auto *record = buffer.data() + offset;
        record[0] = type;
        fiat_shamir_detail::store_u64(record + 1, payload_size);
        fill(record + 9);
        transcript_->absorb_encoded_record(record, record_size);
    }
};

class FiatShamirProofReader
{
public:
    FiatShamirProofReader(const std::vector<std::uint8_t> &bytes,
                         FiatShamirTranscript *shared_transcript = nullptr)
        : owned_transcript_(shared_transcript ? nullptr : new FiatShamirTranscript()),
          transcript_(shared_transcript ? shared_transcript : owned_transcript_.get()),
          bytes_(bytes)
    {
        const auto encoding = read_u64();
        if (encoding > static_cast<std::uint64_t>(G1WireEncoding::Compressed))
            throw std::invalid_argument("unsupported lookup-proof G1 encoding");
        encoding_ = static_cast<G1WireEncoding>(encoding);
    }

    FiatShamirTranscript &transcript() { return *transcript_; }
    G1WireEncoding encoding() const { return encoding_; }

    std::uint64_t read_u64()
    {
        return fiat_shamir_detail::load_u64(read_record(1, 8));
    }

    Fr read_fr()
    {
        return decode_fr_vector_canonical(read_record(3, 32), 32, 1)[0];
    }

    std::vector<G1> read_g1_vector(std::size_t expected_count)
    {
        const std::size_t point_size = g1_wire_size(encoding_);
        if (expected_count > (std::numeric_limits<std::size_t>::max() - 8) / point_size)
            throw std::invalid_argument("G1-vector payload is too large");
        const std::size_t byte_count = expected_count * point_size;
        const auto *payload = read_record(6, 8 + byte_count);
        if (fiat_shamir_detail::load_u64(payload) != expected_count)
            throw std::invalid_argument("unexpected G1-vector length");
        return decode_g1_vector_canonical(payload + 8, byte_count, expected_count, encoding_);
    }

    Fr challenge() { return transcript_->challenge(); }
    Fr challenge_excluding(const std::vector<Fr> &excluded)
    {
        return transcript_->challenge_excluding(excluded);
    }
    Fr challenge_not_one() { return transcript_->challenge_not_one(); }
    bool at_end() const { return offset_ == bytes_.size(); }

private:
    std::unique_ptr<FiatShamirTranscript> owned_transcript_;
    FiatShamirTranscript *transcript_;
    const std::vector<std::uint8_t> &bytes_;
    std::size_t offset_ = 0;
    G1WireEncoding encoding_ = G1WireEncoding::Affine64;

    const std::uint8_t *read_record(std::uint8_t type, std::size_t payload_size)
    {
        const std::size_t remaining = bytes_.size() - offset_;
        if (remaining < 9 || payload_size > remaining - 9)
            throw std::invalid_argument("truncated proof record");
        const auto *record = bytes_.data() + offset_;
        if (record[0] != type || fiat_shamir_detail::load_u64(record + 1) != payload_size)
            throw std::invalid_argument("unexpected proof record schema");
        offset_ += 9 + payload_size;
        transcript_->absorb_encoded_record(record, 9 + payload_size);
        return record + 9;
    }
};

#endif
