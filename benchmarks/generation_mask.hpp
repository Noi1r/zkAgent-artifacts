#pragma once

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// A generation mask is an explicit public, verifier-supplied statement about
// which sequence positions are generated. It is not inferred from prover
// metadata or automatically derived from a tool-call grammar.
namespace generation_mask {

// Positions are zero-based output-token positions. Position zero cannot be
// generated because it has no preceding token from which to predict it.
inline std::vector<std::size_t> load(const std::string& path,
                                     std::size_t sequence_length) {
    if (sequence_length == 0) {
        throw std::runtime_error("generation mask requires a nonempty sequence");
    }
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open generation mask: " + path);
    }

    std::vector<std::size_t> positions;
    std::size_t count = 0;
    bool comment = false;
    char ch;
    while (input.get(ch)) {
        if (comment) {
            if (ch == '\n' || ch == '\r') comment = false;
            continue;
        }
        if (ch == '#') {
            comment = true;
            continue;
        }
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' ||
            ch == '\f' || ch == '\v') {
            continue;
        }
        if (ch != '0' && ch != '1') {
            throw std::runtime_error("generation mask contains a non-bit character: " + path);
        }
        if (count == sequence_length) {
            throw std::runtime_error("generation mask has more bits than sequence length: " + path);
        }
        if (ch == '1') {
            if (count == 0) {
                throw std::runtime_error("generation mask cannot select token position 0: " + path);
            }
            positions.push_back(count);
        }
        ++count;
    }
    if (input.bad()) {
        throw std::runtime_error("cannot read generation mask: " + path);
    }
    if (count != sequence_length) {
        throw std::runtime_error("generation mask bit count does not match sequence length: " + path);
    }
    if (positions.empty()) {
        throw std::runtime_error("generation mask must select at least one token: " + path);
    }
    return positions;
}

// Construct the legacy contiguous generated suffix under the same constraints.
inline std::vector<std::size_t> suffix(std::size_t prompt_length,
                                       std::size_t sequence_length) {
    if (prompt_length == 0 || prompt_length >= sequence_length) {
        throw std::invalid_argument("generation suffix requires 0 < prompt length < sequence length");
    }
    std::vector<std::size_t> positions;
    positions.reserve(sequence_length - prompt_length);
    for (std::size_t position = prompt_length; position < sequence_length; ++position) {
        positions.push_back(position);
    }
    return positions;
}

// Partition the public ordered set without dropping or duplicating positions.
// A zero limit preserves the legacy single decode proof.
inline std::vector<std::vector<std::size_t>> partition(
    const std::vector<std::size_t>& positions, std::size_t max_batch_size) {
    if (positions.empty()) {
        throw std::invalid_argument("generation positions must not be empty");
    }
    for (std::size_t i = 0; i < positions.size(); ++i) {
        if (positions[i] == 0 || (i != 0 && positions[i] <= positions[i - 1])) {
            throw std::invalid_argument("generation positions must be positive and strictly increasing");
        }
    }
    if (max_batch_size == 0) return {positions};
    std::vector<std::vector<std::size_t>> batches;
    for (std::size_t begin = 0; begin < positions.size();) {
        const std::size_t count = std::min(max_batch_size, positions.size() - begin);
        batches.emplace_back(positions.begin() + begin, positions.begin() + begin + count);
        begin += count;
    }
    return batches;
}

inline std::size_t initialPromptLength(const std::vector<std::size_t>& positions) {
    if (positions.empty()) {
        throw std::invalid_argument("generation positions must not be empty");
    }
    return positions.front();
}

// Input positions must be strictly increasing (as returned by load/suffix).
inline std::string formatRanges(const std::vector<std::size_t>& positions) {
    std::ostringstream output;
    output << "0-based half-open: ";
    for (std::size_t i = 0; i < positions.size();) {
        const std::size_t begin = positions[i];
        std::size_t last = begin;
        ++i;
        while (i < positions.size() && positions[i] - last == 1) {
            last = positions[i++];
        }
        output << '[' << begin << ',' << last + 1 << ')';
        if (i < positions.size()) output << ", ";
    }
    return output.str();
}

} // namespace generation_mask
