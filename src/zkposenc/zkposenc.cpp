#include "zkposenc/zkposenc.hpp"
#include "common/mle.hpp"

#include "common/global_var.hpp"
#include "common/fiat_shamir.hpp"
#include "common/utils.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include <omp.h>

using namespace std;
using namespace std::chrono;
using namespace mcl::bn;

namespace zkposenc
{

    static constexpr size_t SumcheckParallelThreshold = 1024;

    void ZkPosEncBenchmarkResult::print() const
    {
        cout << "  " << setw(12) << left << scheme_name
             << " | T=" << setw(4) << seq_len
             << " | d=" << setw(3) << head_dim
             << " | prepare=" << fixed << setprecision(2) << preparation_ms << "ms"
             << " | prove=" << setw(8) << fixed << setprecision(2)
             << prover_ms << "ms"
             << " | verify=" << setw(6) << verifier_ms << "ms"
             << " | payload=" << setw(6) << proof_size_kb << "KB" << endl;
    }

    static void require_even_head_dim(int head_dim)
    {
        if (head_dim <= 0 || (head_dim % 2) != 0)
        {
            throw invalid_argument("RoPE head_dim must be a positive even integer");
        }
    }

    static void build_rope_rotation_matrix(int position, int head_dim,
                                           double rope_base, double rope_scale,
                                           ll *matrix)
    {
        memset(matrix, 0, sizeof(ll) * head_dim * head_dim);
        for (int i = 0; i < head_dim / 2; ++i)
        {
            const double theta = pow(rope_base, -2.0 * i / head_dim);
            const double angle = position * theta;
            const ll cos_v = round(cos(angle) * rope_scale);
            const ll sin_v = round(sin(angle) * rope_scale);
            const int even = 2 * i;
            const int odd = even + 1;
            matrix[even * head_dim + even] = cos_v;
            matrix[even * head_dim + odd] = -sin_v;
            matrix[odd * head_dim + even] = sin_v;
            matrix[odd * head_dim + odd] = cos_v;
        }
    }

    class ThalerRoPEData
    {
    public:
        int seq_len;
        int head_dim;
        int heads;
        int padded_d;
        int padded_h;
        double rope_scale = 256.0;
        double rope_base = 10000.0;

        vector<F> q;
        vector<F> rotation;
        vector<F> output;

        ThalerRoPEData(int T, int d, int H)
            : seq_len(T), head_dim(d), heads(H)
        {
            require_even_head_dim(head_dim);
            if (seq_len <= 0 || heads <= 0)
            {
                throw invalid_argument("RoPE sequence length and head count must be positive");
            }
            padded_d = 1 << ceilPow2BitLength(head_dim);
            padded_h = 1 << ceilPow2BitLength(heads);
        }

        size_t qIndex(int position, int head, int coordinate) const
        {
            return (static_cast<size_t>(position) * padded_h + head) * padded_d + coordinate;
        }

        size_t rotationIndex(int position, int row, int column) const
        {
            return (static_cast<size_t>(position) * padded_d + row) * padded_d + column;
        }

        size_t outputIndex(int position, int head, int coordinate) const
        {
            return (static_cast<size_t>(position) * heads + head) * head_dim + coordinate;
        }

        void create()
        {
            q.assign(static_cast<size_t>(seq_len) * padded_h * padded_d, F_ZERO);
            rotation.assign(static_cast<size_t>(seq_len) * padded_d * padded_d,
                            F_ZERO);
            output.assign(static_cast<size_t>(seq_len) * heads * head_dim, F_ZERO);

            mt19937_64 rng(42);
            for (int position = 0; position < seq_len; ++position)
            {
                for (int head = 0; head < heads; ++head)
                {
                    for (int coordinate = 0; coordinate < head_dim; ++coordinate)
                    {
                        const ll sample = static_cast<ll>(rng() % 2000) - 1000;
                        q[qIndex(position, head, coordinate)] = sample;
                    }
                }
            }

            vector<ll> dense_matrix(static_cast<size_t>(head_dim) * head_dim);
            for (int position = 0; position < seq_len; ++position)
            {
                build_rope_rotation_matrix(position, head_dim, rope_base, rope_scale,
                                           dense_matrix.data());
                for (int row = 0; row < head_dim; ++row)
                {
                    for (int column = 0; column < head_dim; ++column)
                    {
                        rotation[rotationIndex(position, row, column)] =
                            dense_matrix[static_cast<size_t>(row) * head_dim + column];
                    }
                }
            }

            for (int position = 0; position < seq_len; ++position)
            {
                for (int head = 0; head < heads; ++head)
                {
                    for (int row = 0; row < head_dim; ++row)
                    {
                        F value = F_ZERO;
                        for (int column = 0; column < head_dim; ++column)
                        {
                            value += q[qIndex(position, head, column)] *
                                     rotation[rotationIndex(position, row, column)];
                        }
                        output[outputIndex(position, head, row)] = value;
                    }
                }
            }
        }
    };

    class StructuredHadamardData
    {
    public:
        int seq_len;
        int head_dim;
        int heads;
        int padded_t;
        int padded_h;
        int padded_d;
        double rope_scale = 256.0;
        double rope_base = 10000.0;

        vector<F> q;
        vector<F> cosine;
        vector<F> sine;
        vector<F> output;

        StructuredHadamardData(int T, int d, int H)
            : seq_len(T), head_dim(d), heads(H)
        {
            require_even_head_dim(head_dim);
            if (seq_len <= 0 || heads <= 0)
            {
                throw invalid_argument("RoPE sequence length and head count must be positive");
            }
            padded_t = 1 << ceilPow2BitLength(seq_len);
            padded_h = 1 << ceilPow2BitLength(heads);
            padded_d = 1 << ceilPow2BitLength(head_dim);
        }

        size_t size() const
        {
            return static_cast<size_t>(padded_t) * padded_h * padded_d;
        }

        size_t index(int position, int head, int coordinate) const
        {
            return (static_cast<size_t>(position) * padded_h + head) * padded_d + coordinate;
        }

        void create()
        {
            q.assign(size(), F_ZERO);
            const size_t coefficient_size = static_cast<size_t>(padded_t) * padded_d;
            cosine.assign(coefficient_size, F_ZERO);
            sine.assign(coefficient_size, F_ZERO);
            output.assign(size(), F_ZERO);

            mt19937_64 rng(42);
            for (int position = 0; position < seq_len; ++position)
            {
                for (int head = 0; head < heads; ++head)
                {
                    for (int coordinate = 0; coordinate < head_dim; ++coordinate)
                    {
                        const ll sample = static_cast<ll>(rng() % 2000) - 1000;
                        q[index(position, head, coordinate)] = sample;
                    }
                }
            }

            for (int position = 0; position < seq_len; ++position)
            {
                for (int pair = 0; pair < head_dim / 2; ++pair)
                {
                    const double theta = pow(rope_base, -2.0 * pair / head_dim);
                    const double angle = position * theta;
                    const ll cos_v = round(cos(angle) * rope_scale);
                    const ll sin_v = round(sin(angle) * rope_scale);
                    const int even = 2 * pair;
                    const int odd = even + 1;
                    const size_t coefficient = static_cast<size_t>(position) * padded_d + even;
                    cosine[coefficient] = cosine[coefficient + 1] = cos_v;
                    sine[coefficient] = sine[coefficient + 1] = sin_v;
                    for (int head = 0; head < heads; ++head)
                    {
                        const size_t even_index = index(position, head, even);
                        const size_t odd_index = index(position, head, odd);
                        output[even_index] = q[even_index] * cosine[coefficient] -
                                             q[odd_index] * sine[coefficient];
                        output[odd_index] = q[odd_index] * cosine[coefficient] +
                                            q[even_index] * sine[coefficient];
                    }
                }
            }
        }
    };

    using mle::equality_weights;

    struct RoPEShape
    {
        int seq_len;
        int head_dim;
        int heads;
    };

    struct QuadraticSumcheckMessage
    {
        F a = F_ZERO;
        F b = F_ZERO;
        F c = F_ZERO;

        F eval(const F &x) const { return (a * x + b) * x + c; }
        F booleanSum() const { return c + (a + b + c); }
    };

    template <size_t TerminalCount>
    struct RoPEProof
    {
        vector<QuadraticSumcheckMessage> rounds;
        array<F, TerminalCount> terminal_values;

        size_t payloadBytes() const
        {
            return (3 * rounds.size() + TerminalCount) * F_BYTE_SIZE;
        }
    };

    template <size_t TerminalCount>
    struct RoPEProverOutput
    {
        // The core protocol starts from this externally supplied output evaluation.
        // Its commitment/opening link is outside the Table-4 benchmark.
        F output_claim = F_ZERO;
        RoPEProof<TerminalCount> proof;
    };

    static void append_shape(FiatShamirTranscript &transcript, const RoPEShape &shape)
    {
        transcript.append_u64(shape.seq_len);
        transcript.append_u64(shape.head_dim);
        transcript.append_u64(shape.heads);
    }

    static vector<F> challenge_point(FiatShamirTranscript &transcript, int bits)
    {
        vector<F> point(static_cast<size_t>(bits));
        for (F &coordinate : point) coordinate = transcript.challenge();
        return point;
    }

    static F round_challenge(FiatShamirTranscript &transcript,
                              const QuadraticSumcheckMessage &message)
    {
        transcript.append_fr(message.a);
        transcript.append_fr(message.b);
        transcript.append_fr(message.c);
        return transcript.challenge();
    }

    static QuadraticSumcheckMessage combine_messages(
        const vector<QuadraticSumcheckMessage> &partials)
    {
        QuadraticSumcheckMessage message;
        for (const auto &local : partials)
        {
            message.a += local.a;
            message.b += local.b;
            message.c += local.c;
        }
        return message;
    }

    static RoPEProverOutput<2> prove_thaler_rope_protocol(const ThalerRoPEData &data,
                                                        int threads)
    {
        RoPEProverOutput<2> result;
        auto &proof = result.proof;
        const int thread_count = threads > 0 ? threads : 1;
        const int instance_count = data.seq_len * data.heads;
        const int inner_bits = ceilPow2BitLength(data.head_dim);
        const int batch_bits = ceilPow2BitLength(instance_count);
        const size_t padded_inner = size_t{1} << inner_bits;
        const size_t padded_batch = size_t{1} << batch_bits;
        proof.rounds.reserve(batch_bits + inner_bits);

        FiatShamirTranscript transcript;
        append_shape(transcript, {data.seq_len, data.head_dim, data.heads});
        const vector<F> output_point = challenge_point(transcript, inner_bits);
        const vector<F> batch_point = challenge_point(transcript, batch_bits);
        const vector<F> output_weights = equality_weights(output_point);
        const vector<F> batch_weights = equality_weights(batch_point);
        vector<F> left(padded_batch * padded_inner, F_ZERO);
        vector<F> right(padded_batch * padded_inner, F_ZERO);
        vector<F> output_claims(padded_batch, F_ZERO);

#pragma omp parallel for schedule(static) if (thread_count > 1 && padded_batch * padded_inner >= SumcheckParallelThreshold) num_threads(thread_count)
        for (int instance = 0; instance < instance_count; ++instance)
        {
            const int position = instance / data.heads;
            const int selected_head = instance % data.heads;
            const size_t position_offset = static_cast<size_t>(instance) * padded_inner;
            for (int inner = 0; inner < data.head_dim; ++inner)
            {
                left[position_offset + inner] =
                    data.q[data.qIndex(position, selected_head, inner)];
                F right_value = F_ZERO;
                for (int output = 0; output < data.head_dim; ++output)
                {
                    right_value += data.rotation[data.rotationIndex(position, output, inner)] *
                                   output_weights[output];
                }
                right[position_offset + inner] = right_value;
            }
            F output_evaluation = F_ZERO;
            for (int output = 0; output < data.head_dim; ++output)
            {
                output_evaluation +=
                    data.output[data.outputIndex(position, selected_head, output)] * output_weights[output];
            }
            output_claims[instance] = output_evaluation;
        }
        for (size_t position = 0; position < padded_batch; ++position)
            result.output_claim += batch_weights[position] * output_claims[position];

        // Every fold overwrites its active destination prefix before the swap.
        vector<F> reduced_left((padded_batch >> 1) * padded_inner);
        vector<F> reduced_right((padded_batch >> 1) * padded_inner);
        vector<QuadraticSumcheckMessage> local_messages(thread_count);
        size_t active_batch = padded_batch;
        for (int round = 0; round < batch_bits; ++round)
        {
            const size_t half = active_batch >> 1;
            const vector<F> tail_weights = equality_weights(
                batch_point.data() + round + 1, batch_bits - round - 1);
            fill(local_messages.begin(), local_messages.end(), QuadraticSumcheckMessage{});
#pragma omp parallel if (thread_count > 1 && half * padded_inner >= SumcheckParallelThreshold) num_threads(thread_count)
            {
                const int tid = omp_get_thread_num();
                QuadraticSumcheckMessage &local = local_messages[tid];
#pragma omp for schedule(static)
                for (size_t position = 0; position < half; ++position)
                {
                    const F weight = tail_weights[position];
                    const size_t even_base = (position << 1) * padded_inner;
                    const size_t odd_base = ((position << 1) | 1) * padded_inner;
                    for (size_t inner = 0; inner < padded_inner; ++inner)
                    {
                        const F f0 = left[even_base + inner];
                        const F f1 = left[odd_base + inner];
                        const F g0 = right[even_base + inner];
                        const F g1 = right[odd_base + inner];
                        const F df = f1 - f0;
                        const F dg = g1 - g0;
                        local.c += weight * f0 * g0;
                        local.b += weight * (df * g0 + f0 * dg);
                        local.a += weight * df * dg;
                    }
                }
            }
            proof.rounds.push_back(combine_messages(local_messages));
            const F challenge = round_challenge(transcript, proof.rounds.back());
#pragma omp parallel for schedule(static) if (thread_count > 1 && half * padded_inner >= SumcheckParallelThreshold) num_threads(thread_count)
            for (size_t position = 0; position < half; ++position)
            {
                const size_t target_base = position * padded_inner;
                const size_t even_base = (position << 1) * padded_inner;
                const size_t odd_base = ((position << 1) | 1) * padded_inner;
                for (size_t inner = 0; inner < padded_inner; ++inner)
                {
                    const F f0 = left[even_base + inner];
                    const F f1 = left[odd_base + inner];
                    const F g0 = right[even_base + inner];
                    const F g1 = right[odd_base + inner];
                    reduced_left[target_base + inner] = f0 + (f1 - f0) * challenge;
                    reduced_right[target_base + inner] = g0 + (g1 - g0) * challenge;
                }
            }
            left.swap(reduced_left);
            right.swap(reduced_right);
            active_batch = half;
        }

        size_t active_inner = padded_inner;
        for (int round = 0; round < inner_bits; ++round)
        {
            QuadraticSumcheckMessage message;
            const size_t half = active_inner >> 1;
            for (size_t i = 0; i < half; ++i)
            {
                const F f0 = left[i << 1];
                const F f1 = left[i << 1 | 1];
                const F g0 = right[i << 1];
                const F g1 = right[i << 1 | 1];
                const F df = f1 - f0;
                const F dg = g1 - g0;
                message.c += f0 * g0;
                message.b += df * g0 + f0 * dg;
                message.a += df * dg;
            }
            proof.rounds.push_back(message);
            const F challenge = round_challenge(transcript, message);
            // This short inner fold stays serial, so its in-place order is safe.
            for (size_t i = 0; i < half; ++i)
            {
                const F f0 = left[i << 1];
                const F f1 = left[i << 1 | 1];
                const F g0 = right[i << 1];
                const F g1 = right[i << 1 | 1];
                left[i] = f0 + (f1 - f0) * challenge;
                right[i] = g0 + (g1 - g0) * challenge;
            }
            active_inner = half;
        }
        proof.terminal_values = {{left[0], right[0]}};
        return result;
    }

    static RoPEProverOutput<4> prove_hadamard_rope_protocol(
        const StructuredHadamardData &data, int threads)
    {
        RoPEProverOutput<4> result;
        auto &proof = result.proof;
        const int thread_count = threads > 0 ? threads : 1;
        const int position_bits = ceilPow2BitLength(data.seq_len);
        const int head_bits = ceilPow2BitLength(data.heads);
        const int coordinate_bits = ceilPow2BitLength(data.head_dim);
        const int bit_length = position_bits + coordinate_bits;
        const size_t compact_size = static_cast<size_t>(data.padded_t) * data.padded_d;
        proof.rounds.reserve(bit_length);

        FiatShamirTranscript transcript;
        append_shape(transcript, {data.seq_len, data.head_dim, data.heads});
        vector<F> output_point = challenge_point(transcript, coordinate_bits);
        const vector<F> position_point = challenge_point(transcript, position_bits);
        const vector<F> head_point = challenge_point(transcript, head_bits);
        output_point.insert(output_point.end(), position_point.begin(), position_point.end());
        const vector<F> head_weights = equality_weights(head_point);
        const vector<F> output_weights = equality_weights(output_point);
        vector<F> q(compact_size, F_ZERO);
        vector<F> cosine(data.cosine);
        vector<F> permuted_q(compact_size, F_ZERO);
        vector<F> sine(data.sine);
        vector<F> local_claims(thread_count, F_ZERO);

#pragma omp parallel if (thread_count > 1 && compact_size >= SumcheckParallelThreshold) num_threads(thread_count)
        {
            const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
            for (size_t i = 0; i < compact_size; ++i)
            {
                const int position = static_cast<int>(i / data.padded_d);
                const int coordinate = static_cast<int>(i % data.padded_d);
                F q_value = F_ZERO;
                F output_value = F_ZERO;
                for (int head = 0; head < data.padded_h; ++head)
                {
                    const size_t source = data.index(position, head, coordinate);
                    q_value += head_weights[head] * data.q[source];
                    output_value += head_weights[head] * data.output[source];
                }
                q[i] = q_value;
                local_claims[tid] += output_weights[i] * output_value;
            }
        }
        for (const F &local : local_claims) result.output_claim += local;
#pragma omp parallel for schedule(static) if (thread_count > 1 && compact_size >= SumcheckParallelThreshold) num_threads(thread_count)
        for (size_t pair = 0; pair < compact_size / 2; ++pair)
        {
            const size_t even = pair << 1;
            const size_t odd = even | 1;
            permuted_q[even] = -q[odd];
            permuted_q[odd] = q[even];
        }

        vector<F> reduced_q(compact_size >> 1);
        vector<F> reduced_cosine(compact_size >> 1);
        vector<F> reduced_permuted_q(compact_size >> 1);
        vector<F> reduced_sine(compact_size >> 1);
        vector<QuadraticSumcheckMessage> local_messages(thread_count);
        size_t active = compact_size;
        for (int round = 0; round < bit_length; ++round)
        {
            const size_t half = active >> 1;
            const vector<F> tail_weights = equality_weights(
                output_point.data() + round + 1, bit_length - round - 1);
            fill(local_messages.begin(), local_messages.end(), QuadraticSumcheckMessage{});
#pragma omp parallel if (thread_count > 1 && half >= SumcheckParallelThreshold) num_threads(thread_count)
            {
                const int tid = omp_get_thread_num();
                QuadraticSumcheckMessage &local = local_messages[tid];
#pragma omp for schedule(static)
                for (size_t i = 0; i < half; ++i)
                {
                    const size_t even = i << 1;
                    const size_t odd = even | 1;
                    const F q0 = q[even];
                    const F q1 = q[odd];
                    const F c0 = cosine[even];
                    const F c1 = cosine[odd];
                    const F p0 = permuted_q[even];
                    const F p1 = permuted_q[odd];
                    const F s0 = sine[even];
                    const F s1 = sine[odd];
                    const F dq = q1 - q0;
                    const F dc = c1 - c0;
                    const F dp = p1 - p0;
                    const F ds = s1 - s0;
                    const F weight = tail_weights[i];
                    local.c += weight * (q0 * c0 + p0 * s0);
                    local.b += weight * (dq * c0 + q0 * dc + dp * s0 + p0 * ds);
                    local.a += weight * (dq * dc + dp * ds);
                }
            }
            proof.rounds.push_back(combine_messages(local_messages));
            const F challenge = round_challenge(transcript, proof.rounds.back());
#pragma omp parallel for schedule(static) if (thread_count > 1 && half >= SumcheckParallelThreshold) num_threads(thread_count)
            for (size_t i = 0; i < half; ++i)
            {
                const size_t even = i << 1;
                const size_t odd = even | 1;
                reduced_q[i] = q[even] + (q[odd] - q[even]) * challenge;
                reduced_cosine[i] = cosine[even] + (cosine[odd] - cosine[even]) * challenge;
                reduced_permuted_q[i] = permuted_q[even] + (permuted_q[odd] - permuted_q[even]) * challenge;
                reduced_sine[i] = sine[even] + (sine[odd] - sine[even]) * challenge;
            }
            q.swap(reduced_q);
            cosine.swap(reduced_cosine);
            permuted_q.swap(reduced_permuted_q);
            sine.swap(reduced_sine);
            active = half;
        }
        proof.terminal_values = {{q[0], cosine[0], permuted_q[0], sine[0]}};
        return result;
    }

    static bool valid_shape(const RoPEShape &shape)
    {
        return shape.seq_len > 0 && shape.heads > 0 && shape.head_dim > 0 &&
               shape.head_dim % 2 == 0 &&
               static_cast<uint64_t>(shape.seq_len) * shape.heads <= (uint64_t{1} << 30) &&
               shape.head_dim <= (1 << 30);
    }

    static bool verify_round(FiatShamirTranscript &transcript,
                              const QuadraticSumcheckMessage &message, F &claim,
                              const F *evaluation_point = nullptr)
    {
        const F expected = evaluation_point
            ? (F_ONE - *evaluation_point) * message.eval(F_ZERO) +
                  *evaluation_point * message.eval(F_ONE)
            : message.booleanSum();
        if (expected != claim) return false;
        claim = message.eval(round_challenge(transcript, message));
        return true;
    }

    static bool verify_thaler_rope_protocol(const RoPEShape &shape,
                                             const F &output_claim,
                                             const RoPEProof<2> &proof)
    {
        if (!valid_shape(shape)) return false;
        const int inner_bits = ceilPow2BitLength(shape.head_dim);
        const int batch_bits = ceilPow2BitLength(shape.seq_len * shape.heads);
        if (proof.rounds.size() != static_cast<size_t>(batch_bits + inner_bits))
            return false;
        FiatShamirTranscript transcript;
        append_shape(transcript, shape);
        challenge_point(transcript, inner_bits);
        const vector<F> batch_point = challenge_point(transcript, batch_bits);
        F claim = output_claim;
        for (int round = 0; round < batch_bits + inner_bits; ++round)
        {
            const F *evaluation_point = round < batch_bits ? &batch_point[round] : nullptr;
            if (!verify_round(transcript, proof.rounds[round], claim, evaluation_point))
                return false;
        }
        return claim == proof.terminal_values[0] * proof.terminal_values[1];
    }

    static bool verify_hadamard_rope_protocol(const RoPEShape &shape,
                                               const F &output_claim,
                                               const RoPEProof<4> &proof)
    {
        if (!valid_shape(shape)) return false;
        const int coordinate_bits = ceilPow2BitLength(shape.head_dim);
        const int position_bits = ceilPow2BitLength(shape.seq_len);
        const int head_bits = ceilPow2BitLength(shape.heads);
        const int bit_length = coordinate_bits + position_bits;
        if (proof.rounds.size() != static_cast<size_t>(bit_length)) return false;
        FiatShamirTranscript transcript;
        append_shape(transcript, shape);
        vector<F> output_point = challenge_point(transcript, coordinate_bits);
        const vector<F> position_point = challenge_point(transcript, position_bits);
        challenge_point(transcript, head_bits);
        output_point.insert(output_point.end(), position_point.begin(), position_point.end());
        F claim = output_claim;
        for (int round = 0; round < bit_length; ++round)
        {
            if (!verify_round(transcript, proof.rounds[round], claim, &output_point[round]))
                return false;
        }
        const auto &values = proof.terminal_values;
        return claim == values[0] * values[1] + values[2] * values[3];
    }

    template <class Data, class Prove, class Verify>
    static ZkPosEncBenchmarkResult benchmark_rope(const char *scheme_name,
        int seq_len, int head_dim, int heads, int threads, Prove prove, Verify verify)
    {
        using Clock = high_resolution_clock;
        ZkPosEncBenchmarkResult result;
        result.scheme_name = scheme_name;
        result.seq_len = seq_len;
        result.head_dim = head_dim;
        result.heads = heads;
        const RoPEShape shape{seq_len, head_dim, heads};
        if (!valid_shape(shape) || threads <= 0)
            throw invalid_argument("invalid RoPE benchmark dimensions or thread count");

        const auto preparation_start = Clock::now();
        Data data(seq_len, head_dim, heads);
        data.create();
        result.preparation_ms = duration<double, milli>(Clock::now() - preparation_start).count();
        double prover_ms = 0;
        double verifier_ms = 0;
        for (int run = 0; run < ZkPosEncTotalRuns; ++run)
        {
            const auto prover_start = Clock::now();
            const auto output = prove(data, threads);
            const auto prover_end = Clock::now();
            const auto verifier_start = Clock::now();
            const bool accepted = verify(shape, output.output_claim, output.proof);
            const auto verifier_end = Clock::now();
            if (!accepted)
                throw runtime_error(string(scheme_name) + " RoPE sumcheck verification failed");
            if (run >= ZkPosEncWarmupRuns)
            {
                prover_ms += duration<double, milli>(prover_end - prover_start).count();
                verifier_ms += duration<double, milli>(verifier_end - verifier_start).count();
            }
            result.sumcheck_rounds = static_cast<int>(output.proof.rounds.size());
            result.proof_size_kb = static_cast<double>(output.proof.payloadBytes()) / 1024.0;
        }
        result.prover_ms = prover_ms / ZkPosEncMeasuredRuns;
        result.verifier_ms = verifier_ms / ZkPosEncMeasuredRuns;
        return result;
    }

    ZkPosEncBenchmarkResult benchmarkThalerRoPE(int seq_len, int head_dim,
                                                int heads, int threads)
    {
        return benchmark_rope<ThalerRoPEData>("Thaler", seq_len, head_dim, heads, threads,
            prove_thaler_rope_protocol, verify_thaler_rope_protocol);
    }

    ZkPosEncBenchmarkResult benchmarkHadamardRoPE(int seq_len, int head_dim,
                                                  int heads, int threads)
    {
        return benchmark_rope<StructuredHadamardData>("Hadamard", seq_len, head_dim, heads, threads,
            prove_hadamard_rope_protocol, verify_hadamard_rope_protocol);
    }

}
