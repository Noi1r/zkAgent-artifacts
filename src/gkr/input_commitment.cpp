#include "gkr/input_commitment.hpp"
#include "common/cache_io.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <stdexcept>

std::size_t InputCommitmentLayout::columns() const { return std::size_t(1) << (variables - variables / 2); }
std::size_t InputCommitmentLayout::rows() const { return std::size_t(1) << (variables / 2); }
std::size_t InputCommitmentLayout::fixedRowCount() const
{
    std::size_t n = 0;
    for (const auto &span : fixed_rows)
        n += span.count;
    return n;
}
std::size_t InputCommitmentLayout::runtimeRowCount() const
{
    return (value_count + columns() - 1) / columns() - fixedRowCount();
}
bool InputCommitmentLayout::isRuntimeRow(std::size_t row) const
{
    if (row >= (value_count + columns() - 1) / columns())
        return false;
    for (const auto &span : fixed_rows)
        if (row >= span.first && row - span.first < span.count)
            return false;
    return true;
}
void InputCommitmentLayout::validate() const
{
    if (variables < 0 || variables > 30 || value_count == 0 || value_count > (std::size_t(1) << variables))
        throw std::invalid_argument("invalid input commitment shape");
    std::size_t previous_end = 0;
    for (const auto &span : fixed_rows)
    {
        if (!span.count || span.first < previous_end || span.first > rows() || span.count > rows() - span.first ||
            (span.first + span.count) * columns() > value_count)
            throw std::invalid_argument("fixed input rows overlap or exceed the input");
        previous_end = span.first + span.count;
    }
}
bool InputCommitmentLayout::operator==(const InputCommitmentLayout &rhs) const
{
    return variables == rhs.variables && value_count == rhs.value_count && fixed_rows == rhs.fixed_rows;
}

InputCommitmentKey prepareInputCommitmentKey(
    const InputCommitmentLayout &layout,
    const Fr *values, std::size_t value_count,
    const MVKZG_ProverKey &pk, const MVKZG_VerifierKey &vk, int threads)
{
    layout.validate();
    const int col_bits = layout.variables - layout.variables / 2;
    if (value_count < layout.value_count || !values || pk.srs_g1.size() < layout.columns() ||
        vk.vars < col_bits || vk.g2_variables.size() < static_cast<size_t>(col_bits) ||
        pk.srs_g1[0] != vk.g1_generator)
        throw std::invalid_argument("invalid fixed input key preparation data/SRS");
    InputCommitmentKey result;
    result.layout_ = layout;
    result.srs_g1_ = vk.g1_generator;
    result.srs_g2_ = vk.g2_generator;
    result.srs_variables_.assign(vk.g2_variables.begin(), vk.g2_variables.begin() + col_bits);
    for (const auto &span : layout.fixed_rows)
    {
        auto rows = hyrax::commit_rows(values, value_count, const_cast<G1 *>(pk.srs_g1.data()),
                                       col_bits, span.first, span.count, threads);
        result.fixed_commitments_.insert(result.fixed_commitments_.end(), rows.begin(), rows.end());
    }
    return result;
}
bool InputCommitmentKey::matchesSrs(const MVKZG_VerifierKey &vk) const
{
    return vk.vars >= static_cast<int>(srs_variables_.size()) &&
           vk.g1_generator == srs_g1_ && vk.g2_generator == srs_g2_ &&
           vk.g2_variables.size() >= srs_variables_.size() &&
           std::equal(srs_variables_.begin(), srs_variables_.end(), vk.g2_variables.begin());
}
bool InputCommitmentKey::matchesCommitment(int variables, const std::vector<G1> &rows) const
{
    if (variables != layout_.variables || rows.size() != layout_.rows())
        return false;
    std::size_t fixed = 0;
    for (const auto &span : layout_.fixed_rows)
        for (std::size_t i = 0; i < span.count; ++i)
            if (rows[span.first + i] != fixed_commitments_[fixed++])
                return false;
    for (std::size_t i = (layout_.value_count + layout_.columns() - 1) / layout_.columns(); i < rows.size(); ++i)
        if (!rows[i].isZero())
            return false;
    return true;
}
std::vector<G1> InputCommitmentKey::commitRuntime(
    const Fr *values, std::size_t value_count, G1 *generators, int threads) const
{
    layout_.validate();
    if (value_count != layout_.value_count || !values || !generators)
        throw std::invalid_argument("runtime input shape does not match model commitment key");
    std::vector<G1> result(layout_.rows());
    for (auto &point : result)
        point.clear();
    const auto commit_interval = [&](std::size_t first, std::size_t end)
    {
        if (first == end)
            return;
        auto rows = hyrax::commit_rows(values, value_count, generators,
                                       layout_.variables - layout_.variables / 2, first, end - first, threads);
        std::copy(rows.begin(), rows.end(), result.begin() + first);
    };
    std::size_t cursor = 0, fixed = 0;
    for (const auto &span : layout_.fixed_rows)
    {
        commit_interval(cursor, span.first);
        std::copy_n(fixed_commitments_.begin() + fixed, span.count, result.begin() + span.first);
        fixed += span.count;
        cursor = span.first + span.count;
    }
    commit_interval(cursor, (value_count + layout_.columns() - 1) / layout_.columns());
    return result;
}
void InputCommitmentKey::saveToFile(const std::string &path) const
{
    layout_.validate();
    if (fixed_commitments_.size() != layout_.fixedRowCount())
        throw std::invalid_argument("incomplete fixed input commitment key");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write("ICRK", 4);
    cache_io::write_u64(out, layout_.variables);
    cache_io::write_u64(out, layout_.value_count);
    cache_io::write_u64(out, layout_.fixed_rows.size());
    for (const auto &span : layout_.fixed_rows)
    {
        cache_io::write_u64(out, span.first);
        cache_io::write_u64(out, span.count);
    }
    cache_io::write_point(out, srs_g1_);
    cache_io::write_point(out, srs_g2_);
    cache_io::write_u64(out, srs_variables_.size());
    for (const auto &x : srs_variables_)
        cache_io::write_point(out, x);
    cache_io::write_u64(out, fixed_commitments_.size());
    for (const auto &x : fixed_commitments_)
        cache_io::write_point(out, x);
    if (!out)
        throw std::runtime_error("could not save input commitment key: " + path);
}
InputCommitmentKey InputCommitmentKey::loadFromFile(
    const std::string &path, const InputCommitmentLayout &expected)
{
    expected.validate();
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != "ICRK")
        throw std::runtime_error("invalid input commitment key header");
    InputCommitmentKey result;
    if (cache_io::read_u64(in) != static_cast<unsigned>(expected.variables) || cache_io::read_u64(in) != expected.value_count ||
        cache_io::read_u64(in) != expected.fixed_rows.size())
        throw std::runtime_error("input commitment key shape mismatch");
    for (const auto &span : expected.fixed_rows)
        if (cache_io::read_u64(in) != span.first || cache_io::read_u64(in) != span.count)
            throw std::runtime_error("input commitment key fixed rows mismatch");
    result.layout_ = expected;
    result.srs_g1_ = cache_io::read_point<G1>(in);
    result.srs_g2_ = cache_io::read_point<G2>(in);
    const std::size_t col_bits = expected.variables - expected.variables / 2;
    if (cache_io::read_u64(in) != col_bits)
        throw std::runtime_error("input commitment key SRS shape mismatch");
    for (std::size_t i = 0; i < col_bits; ++i)
        result.srs_variables_.push_back(cache_io::read_point<G2>(in));
    if (cache_io::read_u64(in) != expected.fixedRowCount())
        throw std::runtime_error("input commitment key row count mismatch");
    for (std::size_t i = 0; i < expected.fixedRowCount(); ++i)
        result.fixed_commitments_.push_back(cache_io::read_point<G1>(in));
    if (in.peek() != EOF)
        throw std::runtime_error("trailing input commitment key data");
    return result;
}
