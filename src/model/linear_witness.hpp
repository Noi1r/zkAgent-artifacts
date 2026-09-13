#pragma once

#include <mcl/bn256.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace zkagent_integer {

struct LinearWitnessModelView {
    const int* weights = nullptr;
    std::size_t weight_size = 0;
    const mcl::bn::Fr* bias = nullptr;
    std::size_t bias_size = 0;
    std::size_t input_stride = 0;
    std::size_t output_stride = 0;
    std::size_t real_inputs = 0;
    std::size_t real_outputs = 0;
    __int128 input_bound = 0;
    __int128 accumulator_bound = 0;
};

class PreparedLinearWitness {
public:
    static PreparedLinearWitness prepare(const LinearWitnessModelView&, int threads);

    std::vector<mcl::bn::Fr> evaluate(const mcl::bn::Fr* input,
                                    std::size_t input_size,
                                    std::size_t rows, int threads) const;
    std::size_t inputStride() const { return input_stride_; }
    std::size_t outputStride() const { return output_stride_; }

private:
    PreparedLinearWitness() = default;
    std::size_t input_stride_ = 0, output_stride_ = 0;
    std::size_t real_inputs_ = 0, real_outputs_ = 0;
    std::int64_t input_bound_ = 0;
    std::vector<int> weights_;
    std::vector<std::int64_t> bias_;
};

}
