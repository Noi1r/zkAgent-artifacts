#ifndef ZKAGENT_INPUT_TENSOR_VIEW_HPP
#define ZKAGENT_INPUT_TENSOR_VIEW_HPP

#include "common/compiled_mle.hpp"
#include <string>

struct InputTensorViewSpec
{
    int input_bits = 0;
    std::size_t offset = 0;
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t row_stride = 0;
    void validate() const;
    int rowBits() const;
    int columnBits() const;
    int localBits() const { return rowBits() + columnBits(); }
    std::size_t inputDomain() const;
    bool operator==(const InputTensorViewSpec &other) const;
};

class CompiledInputTensorView
{
public:
    static CompiledInputTensorView build(const InputTensorViewSpec &spec);
    void saveToFile(const std::string &path) const;
    static CompiledInputTensorView loadFromFile(
        const std::string &path, const InputTensorViewSpec &expected);
    const InputTensorViewSpec &spec() const { return spec_; }
    std::size_t workspaceElements() const { return routing_.workspaceElements(); }
    std::size_t nodeCount() const { return routing_.nodeCount(); }
    Fr evaluate(const std::vector<Fr> &input_point, const std::vector<Fr> &local_point,
                int threads, std::vector<Fr> &workspace) const;

private:
    InputTensorViewSpec spec_;
    compiled_mle::Polynomial routing_;
};

#endif
