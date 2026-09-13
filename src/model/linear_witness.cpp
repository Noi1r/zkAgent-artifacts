#include "model/linear_witness.hpp"
#include "common/field_integer.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <omp.h>

namespace zkagent_integer
{
    namespace
    {
        using Fr = mcl::bn::Fr;
        using U128 = unsigned __int128;

        std::size_t product(std::size_t a, std::size_t b)
        {
            if (b && a > std::numeric_limits<std::size_t>::max() / b)
                throw std::overflow_error("linear witness shape overflows size_t");
            return a * b;
        }

    }

    PreparedLinearWitness PreparedLinearWitness::prepare(const LinearWitnessModelView &v,
                                                         int requested_threads)
    {
        static_assert(std::numeric_limits<int>::digits == 31, "prepared weights require int32");
        static_assert(sizeof(std::size_t) <= 8, "weight norms require <=64-bit sizes");
        if (!v.weights || !v.real_inputs || !v.real_outputs ||
            v.real_inputs > v.input_stride || v.real_outputs > v.output_stride ||
            v.input_bound <= 0 || v.input_bound > std::numeric_limits<std::int64_t>::max() ||
            v.accumulator_bound <= 0 || v.accumulator_bound >= (__int128(1) << 62))
            throw std::invalid_argument("FC witness needs a valid int64-bounded quantization plan");
        if (product(v.input_stride, v.output_stride) > v.weight_size ||
            (v.bias && v.bias_size < v.output_stride))
            throw std::invalid_argument("linear model buffer is too short");

        PreparedLinearWitness result;
        result.input_stride_ = v.input_stride;
        result.output_stride_ = v.output_stride;
        result.real_inputs_ = v.real_inputs;
        result.real_outputs_ = v.real_outputs;
        result.input_bound_ = static_cast<std::int64_t>(v.input_bound);
        result.weights_.resize(product(v.real_inputs, v.real_outputs));
        result.bias_.assign(v.real_outputs, 0);
        const auto accumulator_bound = static_cast<std::int64_t>(v.accumulator_bound);
        for (std::size_t o = 0; o < v.output_stride; ++o)
        {
            if (o < v.real_outputs)
            {
                if (v.bias && !decode_bounded_i64(v.bias[o], accumulator_bound, result.bias_[o]))
                    throw std::invalid_argument("bias exceeds the quantization accumulator bound");
            }
            else if (v.bias && !v.bias[o].isZero())
            {
                throw std::invalid_argument("nonzero linear bias padding");
            }
        }
        const int threads = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(std::max(1, requested_threads)), v.output_stride));
        std::vector<unsigned char> invalid(threads, 0);
#pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            unsigned char failed = 0;
#pragma omp for schedule(static)
            for (std::size_t o = 0; o < v.output_stride; ++o)
            {
                U128 norm = 0;
                const int *weights = v.weights + o * v.input_stride;
                for (std::size_t k = 0; k < v.input_stride; ++k)
                {
                    const std::int64_t w = weights[k];
                    if (o < v.real_outputs && k < v.real_inputs)
                    {
                        result.weights_[o * v.real_inputs + k] = weights[k];
                        norm += static_cast<std::uint64_t>(w < 0 ? -w : w);
                    }
                    else if (w != 0)
                    {
                        failed = 1;
                    }
                }
                if (o < v.real_outputs)
                {
                    const std::int64_t b = result.bias_[o];
                    const auto abs_b = static_cast<std::uint64_t>(b < 0 ? -b : b);
                    if (norm > (static_cast<U128>(accumulator_bound) - abs_b) /
                                   static_cast<U128>(result.input_bound_))
                        failed = 1;
                }
            }
            invalid[tid] = failed;
        }
        if (std::find(invalid.begin(), invalid.end(), 1) != invalid.end())
            throw std::invalid_argument("FC model padding or quantization bound is inconsistent");
        return result;
    }

    std::vector<Fr> PreparedLinearWitness::evaluate(const Fr *input,
                                                    std::size_t input_size,
                                                    std::size_t rows,
                                                    int requested_threads) const
    {
        if (!input || !rows ||
            product(rows, input_stride_) > input_size)
            throw std::invalid_argument("invalid prepared FC input shape");
        const std::size_t active_inputs = product(rows, real_inputs_);
        const std::size_t jobs = product(rows, real_outputs_);
        const std::size_t output_count = product(rows, output_stride_);
        const int threads = static_cast<int>(std::min<std::size_t>(
            static_cast<std::size_t>(std::max(1, requested_threads)), jobs));
        std::vector<std::int64_t> packed(active_inputs);
        std::vector<unsigned char> invalid(threads, 0);
#pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            unsigned char failed = 0;
#pragma omp for schedule(static)
            for (std::size_t i = 0; i < active_inputs; ++i)
            {
                const std::size_t row = i / real_inputs_, col = i % real_inputs_;
                if (!decode_bounded_i64(input[row * input_stride_ + col], input_bound_, packed[i]))
                    failed = 1;
            }
            invalid[tid] = failed;
        }
        if (std::find(invalid.begin(), invalid.end(), 1) != invalid.end())
            throw std::runtime_error("FC input exceeds its prepared quantization bound");

        std::vector<Fr> output(output_count, Fr(0));
#pragma omp parallel for schedule(static) num_threads(threads)
        for (std::size_t job = 0; job < jobs; ++job)
        {
            const std::size_t row = job / real_outputs_, o = job % real_outputs_;
            const auto *x = packed.data() + row * real_inputs_;
            const int *w = weights_.data() + o * real_inputs_;
            std::int64_t sum = bias_[o];
            for (std::size_t k = 0; k < real_inputs_; ++k)
                sum += x[k] * static_cast<std::int64_t>(w[k]);
            output[row * output_stride_ + o] = Fr(sum);
        }
        return output;
    }
}
