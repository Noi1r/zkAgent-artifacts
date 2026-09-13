#include "gkr/input_tensor_view.hpp"
#include "common/bit_utils.hpp"

#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace
{
    constexpr char InputTensorViewType[] = "zkagent.gkr.input-tensor-view\n";

    int bits(std::size_t count)
    {
        if (count == 0 || count > (std::size_t(1) << 30))
            throw std::invalid_argument("invalid input tensor dimension");
        return zkagent_integer::ceil_log2(count);
    }
    std::array<std::uint64_t, 5> header(const InputTensorViewSpec &s)
    {
        return {{static_cast<std::uint64_t>(s.input_bits), s.offset, s.rows, s.columns, s.row_stride}};
    }
}

void InputTensorViewSpec::validate() const
{
    if (input_bits < 0 || input_bits > 30 || rows == 0 || columns == 0 || row_stride < columns ||
        offset >= inputDomain() || columns > inputDomain() - offset ||
        rows - 1 > (inputDomain() - offset - columns) / row_stride || localBits() > 30)
        throw std::invalid_argument("input tensor view exceeds the committed input");
}
int InputTensorViewSpec::rowBits() const { return bits(rows); }
int InputTensorViewSpec::columnBits() const { return bits(columns); }
std::size_t InputTensorViewSpec::inputDomain() const
{
    if (input_bits < 0 || input_bits > 30)
        throw std::invalid_argument("invalid input dimension");
    return std::size_t(1) << input_bits;
}
bool InputTensorViewSpec::operator==(const InputTensorViewSpec &o) const
{
    return input_bits == o.input_bits && offset == o.offset && rows == o.rows &&
           columns == o.columns && row_stride == o.row_stride;
}
CompiledInputTensorView CompiledInputTensorView::build(const InputTensorViewSpec &spec)
{
    spec.validate();
    CompiledInputTensorView result;
    result.spec_ = spec;
    std::vector<compiled_mle::Entry> entries;
    entries.reserve(spec.rows * spec.columns);
    const int column_bits = spec.columnBits(), local_bits = spec.localBits();
    for (std::size_t t = 0; t < spec.rows; ++t)
        for (std::size_t j = 0; j < spec.columns; ++j)
            entries.push_back({compiled_mle::encodeInterleaved(
                                   static_cast<u32>(spec.offset + t * spec.row_stride + j), spec.input_bits,
                                   static_cast<u32>((t << column_bits) + j), local_bits),
                               1});
    result.routing_ = compiled_mle::Polynomial::build(std::move(entries), spec.input_bits + local_bits);
    return result;
}
Fr CompiledInputTensorView::evaluate(const std::vector<Fr> &input_point,
                                     const std::vector<Fr> &local_point, int threads,
                                     std::vector<Fr> &workspace) const
{
    if (input_point.size() != std::size_t(spec_.input_bits) || local_point.size() != std::size_t(spec_.localBits()))
        throw std::invalid_argument("input tensor evaluation point has wrong dimensions");
    return routing_.evaluate(compiled_mle::interleavePoint(input_point, spec_.input_bits,
                                                           local_point, spec_.localBits()),
                             threads, workspace);
}
void CompiledInputTensorView::saveToFile(const std::string &path) const
{
    std::ofstream out(path, std::ios::binary);
    const auto h = header(spec_);
    out.write(InputTensorViewType, sizeof(InputTensorViewType) - 1);
    out.write(reinterpret_cast<const char *>(h.data()), sizeof(h));
    routing_.writeTo(out);
    if (!out)
        throw std::runtime_error("failed to save input tensor view");
}
CompiledInputTensorView CompiledInputTensorView::loadFromFile(
    const std::string &path, const InputTensorViewSpec &expected)
{
    expected.validate();
    std::ifstream in(path, std::ios::binary);
    std::array<char, sizeof(InputTensorViewType) - 1> cache_type{};
    std::array<std::uint64_t, 5> h{};
    in.read(cache_type.data(), cache_type.size());
    in.read(reinterpret_cast<char *>(h.data()), sizeof(h));
    if (!in || std::string(cache_type.data(), cache_type.size()) != InputTensorViewType ||
        h != header(expected))
        throw std::runtime_error("input tensor view cache has incompatible public metadata");
    CompiledInputTensorView result;
    result.spec_ = expected;
    result.routing_ = compiled_mle::Polynomial::readFrom(in);
    if (!in || result.routing_.variableCount() != std::size_t(expected.input_bits + expected.localBits()) ||
        in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("invalid input tensor view polynomial cache");
    return result;
}
