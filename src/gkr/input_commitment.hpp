#pragma once

#include "pcs/hyrax.hpp"
#include "pcs/srs.hpp"
#include <string>

struct InputRowRange
{
    std::size_t first = 0, count = 0;
    bool operator==(const InputRowRange &rhs) const { return first == rhs.first && count == rhs.count; }
};

struct InputCommitmentLayout
{
    int variables = 0;
    std::size_t value_count = 0;
    std::vector<InputRowRange> fixed_rows;
    std::size_t columns() const;
    std::size_t rows() const;
    std::size_t fixedRowCount() const;
    std::size_t runtimeRowCount() const;
    bool isRuntimeRow(std::size_t row) const;
    void validate() const;
    bool operator==(const InputCommitmentLayout &rhs) const;
};

class InputCommitmentKey
{
public:
    const InputCommitmentLayout &layout() const { return layout_; }
    std::size_t fixedRowCount() const { return fixed_commitments_.size(); }
    bool matchesSrs(const MVKZG_VerifierKey &vk) const;
    bool matchesCommitment(int variables, const std::vector<G1> &rows) const;
    std::vector<G1> commitRuntime(const Fr *values, std::size_t value_count,
                                  G1 *generators, int threads) const;
    void saveToFile(const std::string &path) const;
    static InputCommitmentKey loadFromFile(const std::string &path,
                                           const InputCommitmentLayout &expected);

private:
    InputCommitmentLayout layout_;
    std::vector<G1> fixed_commitments_;
    G1 srs_g1_;
    G2 srs_g2_;
    std::vector<G2> srs_variables_;
    friend InputCommitmentKey prepareInputCommitmentKey(
        const InputCommitmentLayout &, const Fr *, std::size_t,
        const MVKZG_ProverKey &, const MVKZG_VerifierKey &, int);
};

InputCommitmentKey prepareInputCommitmentKey(
    const InputCommitmentLayout &layout,
    const Fr *values, std::size_t value_count,
    const MVKZG_ProverKey &pk, const MVKZG_VerifierKey &vk, int threads = 1);

struct InputCommitmentPreparation
{
    InputCommitmentKey key;
    bool loaded = false;
    double preparation_seconds = 0;
    double fixed_commit_seconds = 0;
};
