#include "gkr/verifier.hpp"
#include "common/mle.hpp"
#include "common/polynomial.h"

#include "pcs/srs.hpp"
#include "common/utils.hpp"
#include "gkr/wiring_compiled.hpp"
#include "gkr/input_routing_compiled.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace mcl::bn;

using VerificationClock = std::chrono::steady_clock;

namespace
{

    void append_quadratic_message(
        FiatShamirTranscript *transcript, const quadratic_poly &polynomial)
    {
        transcript->append_fr(polynomial.a);
        transcript->append_fr(polynomial.b);
        transcript->append_fr(polynomial.c);
    }

    Fr evaluate_bias(
        const vector<Fr> &bias, int column_count, int column_bits,
        const vector<F> &point, int row_bits)
    {
        if (bias.empty() || column_count <= 0 || column_bits < 0)
        {
            return F_ZERO;
        }
        const int evaluation_size = 1 << column_bits;
        std::vector<Fr> work(static_cast<std::size_t>(evaluation_size), F_ZERO);
        const int available = std::min(column_count, static_cast<int>(bias.size()));
        for (int i = 0; i < available; ++i)
        {
            work[static_cast<std::size_t>(i)] = bias[static_cast<std::size_t>(i)];
        }
        return mle::evaluate(std::move(work), point.data() + row_bits, column_bits);
    }

}

bool verifier::verifyProof(
    const GkrProof &proof, FiatShamirTranscript *transcript)
{
    const VerificationClock::time_point total_start =
        VerificationClock::now();
    verifier_time = 0.0;
    verifier_wiring_time = 0.0;
    verifier_wiring_contraction_time = 0.0;
    input_reduction_coefficient_time = 0.0;
    commit_open_verify_time = 0.0;
    if (C.size <= 0 || transcript == nullptr ||
        proof.layers.size() != static_cast<std::size_t>(C.size))
    {
        return false;
    }
    if (input_commitment_key_ &&
        (mv_vk == nullptr ||
         input_commitment_key_->layout().value_count != C.circuit[0].size ||
         !input_commitment_key_->matchesSrs(*mv_vk) ||
         !input_commitment_key_->matchesCommitment(comm.l, public_commitment_rows_)))
    {
        return false;
    }

    auto append_layer_final = [&](int layer_id, const layer &current,
                                  const GkrLayerProof &message)
    {
        transcript->append_u64(static_cast<std::uint64_t>(layer_id));
        transcript->append_u64(static_cast<std::uint64_t>(current.ty));
        transcript->append_fr(message.final_u0);
        transcript->append_fr(message.final_u1);
        transcript->append_fr(message.final_v0);
        transcript->append_fr(message.final_v1);
    };

    r_u.assign(static_cast<std::size_t>(C.size + 2), {});
    r_v.assign(static_cast<std::size_t>(C.size + 2), {});
    final_claim_u0.assign(static_cast<std::size_t>(C.size + 2), F_ZERO);
    final_claim_v0.assign(static_cast<std::size_t>(C.size + 2), F_ZERO);

    const int output_bits = C.circuit[C.size - 1].bit_length;
    r_u[C.size].resize(static_cast<std::size_t>(output_bits));
    for (int bit = 0; bit < output_bits; ++bit)
    {
        r_u[C.size][static_cast<std::size_t>(bit)] = transcript->challenge_not_one();
    }
    F previous_sum = proof.output_evaluation;
    transcript->append_fr(previous_sum);

    F alpha = F_ONE;
    F beta = F_ZERO;
    F final_u1 = previous_sum;
    F final_v1 = F_ZERO;
    std::vector<F> current_r0 = r_u[C.size];
    std::vector<F> current_r1;
    int remaining_fconn = static_cast<int>(verifier_key_.fconn.size());

    for (int layer_id = C.size - 1; layer_id > 0; --layer_id)
    {
        const layer &current = C.circuit[layer_id];
        if (debug_)
            std::cerr << "[gkr_replay] layer=" << layer_id
                      << " type=" << static_cast<int>(current.ty) << '\n';
        const GkrLayerProof &message =
            proof.layers[static_cast<std::size_t>(layer_id)];
        if (!message.provided)
            return false;
        if (requiresFreshZeroRoot(C, layer_id))
        {
            if (!alpha.isZero() || !beta.isZero() || !previous_sum.isZero())
            {
                return false;
            }
            current_r0.resize(static_cast<std::size_t>(current.bit_length));
            for (int bit = 0; bit < current.bit_length; ++bit)
            {
                current_r0[static_cast<std::size_t>(bit)] = transcript->challenge_not_one();
            }
            current_r1.clear();
            alpha = F_ONE;
            beta = F_ZERO;
            previous_sum = F_ZERO;
            final_u1 = F_ZERO;
            final_v1 = F_ZERO;
        }
        F relu_rou = F_ONE;
        if (current.zero_start_id < current.size)
            relu_rou = transcript->challenge();

        if (current.ty == layerType::FCONN)
        {
            if (!message.is_fconn || remaining_fconn <= 0 ||
                message.has_phase2 ||
                !message.phase1_rounds.empty() ||
                !message.phase2_rounds.empty())
            {
                return false;
            }
            const int fc_id = --remaining_fconn;
            const GkrFconnVerifierParameters &params =
                verifier_key_.fconn[static_cast<std::size_t>(fc_id)];
            if (params.rows <= 0 || params.columns <= 0 ||
                params.input_rows <= 0 ||
                !params.matchesInputRouting(current, C.circuit[0].size, false))
            {
                return false;
            }
            const int row_bits = ceilPow2BitLength(params.rows);
            const int input_bits = ceilPow2BitLength(params.input_rows);
            const int column_bits = ceilPow2BitLength(params.columns);
            if (message.fconn.rounds.size() !=
                static_cast<std::size_t>(row_bits))
            {
                return false;
            }
            r_u[layer_id].resize(row_bits + input_bits);
            r_v[layer_id].resize(row_bits + column_bits);
            for (int bit = row_bits; bit < row_bits + input_bits; ++bit)
            {
                const int source = bit - row_bits + column_bits;
                if (source < 0 ||
                    source >= static_cast<int>(current_r0.size()))
                {
                    return false;
                }
                r_u[layer_id][static_cast<std::size_t>(bit)] =
                    current_r0[static_cast<std::size_t>(source)];
            }
            for (int bit = row_bits; bit < row_bits + column_bits; ++bit)
            {
                const int source = bit - row_bits;
                if (source < 0 ||
                    source >= static_cast<int>(current_r0.size()))
                {
                    return false;
                }
                r_v[layer_id][static_cast<std::size_t>(bit)] =
                    current_r0[static_cast<std::size_t>(source)];
            }
            const F bias_evaluation = params.bias.empty()
                                          ? F_ZERO
                                          : evaluate_bias(
                                                params.bias, params.columns, column_bits,
                                                r_v[layer_id], row_bits);
            F product_claim = final_u1 - bias_evaluation;
            for (int round = 0; round < row_bits; ++round)
            {
                const quadratic_poly &polynomial =
                    message.fconn.rounds[static_cast<std::size_t>(round)];
                if (polynomial.eval(F_ZERO) + polynomial.eval(F_ONE) !=
                    product_claim)
                {
                    if (debug_)
                        std::cerr << "[gkr_replay] FCONN round mismatch: " << round << '\n';
                    return false;
                }
                append_quadratic_message(
                    transcript, polynomial);
                r_u[layer_id][static_cast<std::size_t>(round)] = transcript->challenge_not_one();
                r_v[layer_id][static_cast<std::size_t>(round)] =
                    r_u[layer_id][static_cast<std::size_t>(round)];

                product_claim = polynomial.eval(
                    r_u[layer_id][static_cast<std::size_t>(round)]);
            }
            if (product_claim !=
                message.fconn.input_evaluation *
                    message.fconn.matrix_evaluation)
            {
                return false;
            }
            if (params.matrix_commitments.empty())
            {
                return false;
            }
            else if (mv_vk == nullptr ||
                     params.matrix_variables != row_bits + column_bits ||
                     params.matrix_commitments.size() !=
                         (std::size_t(1) << (params.matrix_variables / 2)) ||
                     !hyrax::verify(
                         message.fconn.matrix_opening,
                         r_v[layer_id].data(),
                         message.fconn.matrix_evaluation,
                         params.matrix_commitments.data(),
                         params.matrix_variables, *mv_vk,
                         hyrax::Layout::InterleavedLowBits))
            {
                return false;
            }
            if (!params.matrix_commitments.empty())
            {
                transcript->append_g1_vector(message.fconn.matrix_opening.eval_proof.q_comm,
                    G1WireEncoding::Compressed);
            }
            if (message.final_u0 != message.fconn.input_evaluation ||
                message.final_u1 != F_ZERO ||
                message.final_v0 != message.fconn.matrix_evaluation ||
                message.final_v1 != F_ZERO)
            {
                return false;
            }
            final_claim_u0[layer_id] = message.final_u0;
            final_u1 = message.final_u1;
            final_claim_v0[layer_id] = message.final_v0;
            final_v1 = message.final_v1;
            append_layer_final(layer_id, current, message);

            if (~current.bit_length_u[1])
            {
                alpha = transcript->challenge();
            }
            else
            {
                alpha = F_ZERO;
            }
            if ((~current.bit_length_v[1]) ||
                current.ty == layerType::FFT)
            {
                beta = transcript->challenge();
            }
            else
            {
                beta = F_ZERO;
            }
            previous_sum = alpha * final_u1 + beta * final_v1;
            current_r0 = r_u[layer_id];
            current_r1 = r_v[layer_id];
            continue;
        }
        if (message.is_fconn ||
            message.phase1_rounds.size() != current.max_bl_u)
        {
            return false;
        }

        r_u[layer_id].resize(current.max_bl_u);
        F claim = previous_sum;
        for (int round = 0; round < current.max_bl_u; ++round)
        {
            const quadratic_poly &polynomial =
                message.phase1_rounds[static_cast<std::size_t>(round)];
            if (polynomial.eval(F_ZERO) + polynomial.eval(F_ONE) != claim)
            {
                if (debug_)
                    std::cerr << "[gkr_replay] phase1 round mismatch: " << round << '\n';
                return false;
            }
            append_quadratic_message(
                transcript, polynomial);
            r_u[layer_id][static_cast<std::size_t>(round)] = transcript->challenge_not_one();
            claim = polynomial.eval(
                r_u[layer_id][static_cast<std::size_t>(round)]);
        }
        final_claim_u0[layer_id] = message.final_u0;
        final_u1 = message.final_u1;

        const std::vector<F> wiring_r0 = current_r0;
        const std::vector<F> wiring_r1 = beta.isZero()
                                             ? std::vector<F>()
                                             : current_r1;
        const VerificationClock::time_point phase1_wiring_start =
            VerificationClock::now();
        if (compiled_wiring_enabled())
        {
            const std::array<Fr, 2> wiring_result =
                compiled_wiring_plan_->evaluatePhase1(
                    static_cast<std::size_t>(layer_id), alpha, beta,
                    relu_rou, wiring_r0, wiring_r1, r_u[layer_id],
                    thread_num_, compiled_wiring_workspace_);
            uni_value[0] = wiring_result[0];
            uni_value[1] = wiring_result[1];
            bin_value[0] = bin_value[1] = bin_value[2] = F_ZERO;
            verifier_wiring_contraction_time +=
                std::chrono::duration<double>(
                    VerificationClock::now() - phase1_wiring_start)
                    .count();
        }
        else
        {
            return false;
        }
        verifier_wiring_time += std::chrono::duration<double>(
                                    VerificationClock::now() - phase1_wiring_start)
                                    .count();

        if (current.need_phase2)
        {
            if (!message.has_phase2 ||
                message.phase2_rounds.size() != current.max_bl_v)
            {
                return false;
            }
            r_v[layer_id].resize(current.max_bl_v);
            for (int round = 0; round < current.max_bl_v; ++round)
            {
                const quadratic_poly &polynomial =
                    message.phase2_rounds[static_cast<std::size_t>(round)];
                if (polynomial.eval(F_ZERO) + polynomial.eval(F_ONE) !=
                    claim)
                {
                    if (debug_)
                        std::cerr << "[gkr_replay] phase2 round mismatch: " << round << '\n';
                    return false;
                }
                append_quadratic_message(
                    transcript, polynomial);
                r_v[layer_id][static_cast<std::size_t>(round)] = transcript->challenge_not_one();
                claim = polynomial.eval(
                    r_v[layer_id][static_cast<std::size_t>(round)]);
            }
            final_claim_v0[layer_id] = message.final_v0;
            final_v1 = message.final_v1;
            const VerificationClock::time_point phase2_wiring_start =
                VerificationClock::now();
            if (compiled_wiring_enabled())
            {
                const std::array<Fr, 3> wiring_result =
                    compiled_wiring_plan_->evaluatePhase2(
                        static_cast<std::size_t>(layer_id), alpha, beta,
                        relu_rou, wiring_r0, wiring_r1, r_u[layer_id],
                        r_v[layer_id], thread_num_,
                        compiled_wiring_workspace_);
                const std::array<Fr, 2> phase1_unary =
                    {{uni_value[0], uni_value[1]}};
                Fr at_zero = F_ONE;
                for (const Fr &coordinate : r_v[layer_id])
                {
                    at_zero *= F_ONE - coordinate;
                }
                const std::array<Fr, 2> compiled_unary =
                    {{phase1_unary[0] * at_zero,
                      phase1_unary[1] * at_zero}};
                uni_value[0] = compiled_unary[0];
                uni_value[1] = compiled_unary[1];
                bin_value[0] = wiring_result[0];
                bin_value[1] = wiring_result[1];
                bin_value[2] = wiring_result[2];
                verifier_wiring_contraction_time +=
                    std::chrono::duration<double>(
                        VerificationClock::now() - phase2_wiring_start)
                        .count();
            }
            else
            {
                return false;
            }
            verifier_wiring_time += std::chrono::duration<double>(
                                        VerificationClock::now() - phase2_wiring_start)
                                        .count();
        }
        else
        {
            if (message.has_phase2 || !message.phase2_rounds.empty() ||
                message.final_v0 != F_ZERO || message.final_v1 != F_ZERO)
            {
                return false;
            }
            final_claim_v0[layer_id] = message.final_v0;
            final_v1 = message.final_v1;
        }

        if (claim != getFinalValue(
                         final_claim_u0[layer_id], final_u1,
                         final_claim_v0[layer_id], final_v1))
        {
            if (debug_)
                std::cerr << "[gkr_replay] terminal wiring identity mismatch\n";
            return false;
        }
        append_layer_final(layer_id, current, message);

        if (~current.bit_length_u[1])
        {
            alpha = transcript->challenge();
        }
        else
        {
            alpha = F_ZERO;
        }
        if ((~current.bit_length_v[1]) || current.ty == layerType::FFT)
        {
            beta = transcript->challenge();
        }
        else
        {
            beta = F_ZERO;
        }
        previous_sum = alpha * final_u1 + beta * final_v1;
        current_r0 = r_u[layer_id];
        current_r1 = r_v[layer_id];
    }

    const layer &input_layer = C.circuit[0];
    std::vector<F> sig_u(static_cast<std::size_t>(C.size - 1));
    std::vector<F> sig_v(static_cast<std::size_t>(C.size - 1));
    for (int layer_id = 0; layer_id < C.size - 1; ++layer_id)
    {
        sig_u[static_cast<std::size_t>(layer_id)] = transcript->challenge();
    }
    for (int layer_id = 0; layer_id < C.size - 1; ++layer_id)
    {
        sig_v[static_cast<std::size_t>(layer_id)] = transcript->challenge();
    }

    F input_claim = F_ZERO;
    for (int layer_id = 1; layer_id < C.size; ++layer_id)
    {
        if (~C.circuit[layer_id].bit_length_u[0])
        {
            input_claim += sig_u[static_cast<std::size_t>(layer_id - 1)] * final_claim_u0[layer_id];
        }
        if (~C.circuit[layer_id].bit_length_v[0] &&
            C.circuit[layer_id].need_phase2)
        {
            input_claim += sig_v[static_cast<std::size_t>(layer_id - 1)] * final_claim_v0[layer_id];
        }
    }
    if (input_reduction_link_ != nullptr)
    {
        input_claim += input_reduction_link_->claimed_evaluation;
    }

    const int input_bits = input_layer.bit_length;
    if (proof.input_reduction_rounds.size() !=
        static_cast<std::size_t>(input_bits))
    {
        return false;
    }
    r_u[0].resize(static_cast<std::size_t>(input_bits));
    const int row_bits = input_bits / 2;
    const int column_bits = input_bits - row_bits;
    for (int round = 0; round < input_bits; ++round)
    {
        const int variable = round < row_bits
                                 ? column_bits + round
                                 : round - row_bits;
        const GkrInputReductionRound &message =
            proof.input_reduction_rounds[static_cast<std::size_t>(round)];
        if (message.at_zero + message.at_one != input_claim)
            return false;
        transcript->append_fr(message.at_zero);
        transcript->append_fr(message.at_one);
        transcript->append_fr(message.at_two);

        r_u[0][static_cast<std::size_t>(variable)] = transcript->challenge_not_one();

        input_claim = polynomial::evaluate_quadratic_samples(
            message.at_zero, message.at_one, message.at_two,
            r_u[0][static_cast<std::size_t>(variable)]);
    }
    transcript->append_fr(proof.input_evaluation);

    const VerificationClock::time_point coefficient_start =
        VerificationClock::now();
    if (!compiled_input_routing_enabled())
        return false;
    F coefficient_evaluation = compiled_input_routing_plan_->evaluate(
        r_u[0], r_u, r_v, sig_u, sig_v, thread_num_,
        compiled_input_routing_workspace_);
    if (input_reduction_link_ != nullptr)
    {
        coefficient_evaluation +=
            input_reduction_link_->coefficientEvaluationAtPoint(
                r_u[0], thread_num_, &input_link_compiled_workspace_);
    }
    input_reduction_coefficient_time = std::chrono::duration<double>(
                                           VerificationClock::now() - coefficient_start)
                                           .count();
    if (proof.input_evaluation * coefficient_evaluation != input_claim)
    {
        return false;
    }

    const GkrInputOpeningProof &opening = proof.input_opening;
    const VerificationClock::time_point opening_start =
        VerificationClock::now();
    if (opening.provided)
    {
        if (mv_vk == nullptr)
            return false;
        if (opening.evaluation != proof.input_evaluation)
            return false;
        if (!hyrax::verify(
                opening.opening, r_u[0].data(), opening.evaluation,
                comm.comm, comm.l, *mv_vk,
                hyrax::Layout::ContiguousRows))
        {
            return false;
        }
        transcript->append_g1_vector(opening.opening.eval_proof.q_comm,
            G1WireEncoding::Compressed);

    }
    else if (comm.comm)
    {
        return false;
    }

    commit_open_verify_time = std::chrono::duration<double>(
                                  VerificationClock::now() - opening_start)
                                  .count();
    verifier_time = std::chrono::duration<double>(
                        VerificationClock::now() - total_start)
                        .count();

    return true;
}
