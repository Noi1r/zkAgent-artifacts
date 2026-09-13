#include "lookup/logup_gkr.hpp"
#include "common/mle.hpp"
#include "common/bit_utils.hpp"
#include "common/field_integer.hpp"
#include "common/fiat_shamir.hpp"
#include "pcs/srs.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <omp.h>

using namespace mcl::bn;

namespace
{

    Fr evaluate_cubic_samples(Fr x0, Fr x1, Fr x2, Fr x3, Fr u)
    {
        Fr y = 1 / Fr(6) * ((-x0) * (u - 1) * (u - 2) * (u - 3) + 3 * x1 * u * (u - 2) * (u - 3) - 3 * x2 * u * (u - 1) * (u - 3) + x3 * u * (u - 1) * (u - 2));
        return y;
    }

    using zkagent_integer::ceil_log2;

    constexpr int LogupSumcheckParallelThreshold = 1024;

    using mle::equality_weights;

    Fr eval_eq_at_point(const std::vector<Fr> &r_eq, const std::vector<Fr> &r_point)
    {
        Fr result = Fr(1);
        int l = static_cast<int>(r_eq.size());
        for (int i = 0; i < l; i++)
        {
            result *= r_eq[i] * r_point[i] + (Fr(1) - r_eq[i]) * (Fr(1) - r_point[i]);
        }
        return result;
    }

    void build_gkr_layers(std::vector<std::vector<Fr>> &N_layers,
                          std::vector<std::vector<Fr>> &D_layers,
                          int M, int thread_num, bool unit_numerator_leaf = false)
    {
        const int nt = thread_num > 0 ? thread_num : 1;
        for (int l = 0; l < M; l++)
        {
            const int size_upper = 1 << (M - l - 1);
            N_layers[l + 1].resize(size_upper);
            D_layers[l + 1].resize(size_upper);
            const bool unit_numerator = unit_numerator_leaf && l == 0;
            const Fr *NL = N_layers[l].data();
            const Fr *DL = D_layers[l].data();
            Fr *NU = N_layers[l + 1].data();
            Fr *DU = D_layers[l + 1].data();
#pragma omp parallel for num_threads(nt) schedule(static) if (size_upper > 64)
            for (int k = 0; k < size_upper; k++)
            {
                NU[k] = unit_numerator
                            ? DL[2 * k + 1] + DL[2 * k]
                            : NL[2 * k] * DL[2 * k + 1] + NL[2 * k + 1] * DL[2 * k];
                DU[k] = DL[2 * k] * DL[2 * k + 1];
            }
        }
    }

    template <bool UnitNumerator>
    void gkr_layer_sumcheck_prove(
        int vars, const std::vector<Fr> &r_eq,
        std::vector<Fr> &N_lower, std::vector<Fr> &D_lower,
        Fr lambda,
        SumcheckGkrGruenProof &out_sumcheck,
        Fr &out_v_N0, Fr &out_v_N1, Fr &out_v_D0, Fr &out_v_D1,
        int thread_num,
        FiatShamirProofWriter &transcript, std::vector<Fr> &out_challenges)
    {
        const int half = 1 << vars;
        std::vector<Fr> N_e, N_o;
        if (!UnitNumerator)
        {
            N_e.resize(half);
            N_o.resize(half);
        }
        std::vector<Fr> D_e(half), D_o(half);
        const int requested_threads = std::max(thread_num, 1);
        const int split_threads = std::min(requested_threads, half);
        const auto split_pair = [&](int k)
        {
            if (!UnitNumerator)
            {
                N_e[k] = N_lower[2 * k];
                N_o[k] = N_lower[2 * k + 1];
            }
            D_e[k] = D_lower[2 * k];
            D_o[k] = D_lower[2 * k + 1];
        };
        if (half >= LogupSumcheckParallelThreshold && split_threads > 1)
        {
#pragma omp parallel for num_threads(split_threads) schedule(static)
            for (int k = 0; k < half; k++)
                split_pair(k);
        }
        else
        {
            for (int k = 0; k < half; k++)
                split_pair(k);
        }
        // Later layers and PCS openings do not use these consumed tree buffers.
        std::vector<Fr>().swap(N_lower);
        std::vector<Fr>().swap(D_lower);

        std::vector<Fr> eq_poly = (vars > 0) ? equality_weights(r_eq) : std::vector<Fr>{Fr(1)};

        out_sumcheck.round_polys.clear();
        out_sumcheck.round_polys.reserve(vars * 2);
        out_challenges.clear();
        out_challenges.reserve(vars);

        const int max_pair_count = std::max(1, half / 2);
        const int max_reduction_threads =
            std::min(requested_threads, max_pair_count);
        std::vector<Fr> local_sum0;
        std::vector<Fr> local_sum2;
        if (max_pair_count >= LogupSumcheckParallelThreshold &&
            max_reduction_threads > 1)
        {
            local_sum0.assign(
                static_cast<std::size_t>(max_reduction_threads), Fr(0));
            local_sum2.assign(
                static_cast<std::size_t>(max_reduction_threads), Fr(0));
        }

        std::vector<Fr> eq_scratch;
        std::vector<Fr> N_e_scratch;
        std::vector<Fr> N_o_scratch;
        std::vector<Fr> D_e_scratch;
        std::vector<Fr> D_o_scratch;

        const auto accumulate_pair = [&](int j, Fr &sum0, Fr &sum2)
        {
            const Fr eq_0 = eq_poly[j], eq_1 = eq_poly[j + 1];
            const Fr De0 = D_e[j], De1 = D_e[j + 1];
            const Fr Do0 = D_o[j], Do1 = D_o[j + 1];
            const Fr eq_2 = eq_1 + eq_1 - eq_0;
            const Fr De_2 = De1 + De1 - De0;
            const Fr Do_2 = Do1 + Do1 - Do0;
            if (UnitNumerator)
            {
                sum0 += eq_0 * (Do0 + De0 + lambda * De0 * Do0);
                sum2 += eq_2 * (Do_2 + De_2 + lambda * De_2 * Do_2);
            }
            else
            {
                const Fr Ne0 = N_e[j], Ne1 = N_e[j + 1];
                const Fr No0 = N_o[j], No1 = N_o[j + 1];
                const Fr Ne_2 = Ne1 + Ne1 - Ne0;
                const Fr No_2 = No1 + No1 - No0;
                sum0 += eq_0 * (Ne0 * Do0 + No0 * De0 + lambda * De0 * Do0);
                sum2 += eq_2 * (Ne_2 * Do_2 + No_2 * De_2 + lambda * De_2 * Do_2);
            }
        };

        for (int round_i = vars; round_i >= 1; round_i--)
        {
            const int current_len = 1 << round_i;
            Fr sum0 = Fr(0), sum2 = Fr(0);
            const int pair_count = current_len / 2;
            const int reduction_threads = std::min(requested_threads, pair_count);
            if (pair_count >= LogupSumcheckParallelThreshold &&
                reduction_threads > 1)
            {
                int actual_reduction_threads = 1;
#pragma omp parallel num_threads(reduction_threads)
                {
#pragma omp single
                    {
                        actual_reduction_threads = omp_get_num_threads();
                    }
                    const int tid = omp_get_thread_num();
                    Fr local0 = Fr(0);
                    Fr local2 = Fr(0);
#pragma omp for schedule(static)
                    for (int pair = 0; pair < pair_count; ++pair)
                    {
                        accumulate_pair(2 * pair, local0, local2);
                    }
                    local_sum0[static_cast<std::size_t>(tid)] = local0;
                    local_sum2[static_cast<std::size_t>(tid)] = local2;
                }
                for (int tid = 0; tid < actual_reduction_threads; ++tid)
                {
                    sum0 += local_sum0[static_cast<std::size_t>(tid)];
                    sum2 += local_sum2[static_cast<std::size_t>(tid)];
                }
            }
            else
            {
                for (int j = 0; j < current_len; j += 2)
                    accumulate_pair(j, sum0, sum2);
            }

            out_sumcheck.round_polys.push_back(sum0);
            out_sumcheck.round_polys.push_back(sum2);

            transcript.write_fr(sum0);
            transcript.write_fr(sum2);
            const Fr chal = transcript.challenge_excluding(std::vector<Fr>{Fr(1), Fr(1) / Fr(2)});
            out_challenges.push_back(chal);

            const int new_len = current_len / 2;
            eq_scratch.resize(static_cast<std::size_t>(new_len));
            if (!UnitNumerator)
            {
                N_e_scratch.resize(static_cast<std::size_t>(new_len));
                N_o_scratch.resize(static_cast<std::size_t>(new_len));
            }
            D_e_scratch.resize(static_cast<std::size_t>(new_len));
            D_o_scratch.resize(static_cast<std::size_t>(new_len));

            const auto fold_pair = [&](int j)
            {
                eq_scratch[static_cast<std::size_t>(j)] =
                    eq_poly[2 * j] + chal * (eq_poly[2 * j + 1] - eq_poly[2 * j]);
                if (!UnitNumerator)
                {
                    N_e_scratch[static_cast<std::size_t>(j)] =
                        N_e[2 * j] + chal * (N_e[2 * j + 1] - N_e[2 * j]);
                    N_o_scratch[static_cast<std::size_t>(j)] =
                        N_o[2 * j] + chal * (N_o[2 * j + 1] - N_o[2 * j]);
                }
                D_e_scratch[static_cast<std::size_t>(j)] =
                    D_e[2 * j] + chal * (D_e[2 * j + 1] - D_e[2 * j]);
                D_o_scratch[static_cast<std::size_t>(j)] =
                    D_o[2 * j] + chal * (D_o[2 * j + 1] - D_o[2 * j]);
            };
            const int fold_threads = std::min(requested_threads, new_len);
            if (new_len >= LogupSumcheckParallelThreshold && fold_threads > 1)
            {
#pragma omp parallel for num_threads(fold_threads) schedule(static)
                for (int j = 0; j < new_len; ++j)
                    fold_pair(j);
            }
            else
            {
                for (int j = 0; j < new_len; ++j)
                    fold_pair(j);
            }
            eq_poly.swap(eq_scratch);
            if (!UnitNumerator)
            {
                N_e.swap(N_e_scratch);
                N_o.swap(N_o_scratch);
            }
            D_e.swap(D_e_scratch);
            D_o.swap(D_o_scratch);
        }

        out_v_N0 = UnitNumerator ? Fr(1) : N_e[0];
        out_v_N1 = UnitNumerator ? Fr(1) : N_o[0];
        out_v_D0 = D_e[0];
        out_v_D1 = D_o[0];
    }

    bool verify_gkr_layer_sumcheck(
        const SumcheckGkrGruenProof &proof, const std::vector<Fr> &r_eq,
        Fr initial_claim, Fr lambda,
        Fr v_N0, Fr v_N1, Fr v_D0, Fr v_D1,
        const std::vector<Fr> &challenges)
    {
        if ((proof.round_polys.size() & 1u) != 0)
            return false;
        const int rounds = static_cast<int>(proof.round_polys.size() / 2);
        if ((int)r_eq.size() != rounds)
            return false;
        if (challenges.size() != static_cast<std::size_t>(rounds))
            return false;

        if (rounds == 0)
        {
            const Fr expected = v_N0 * v_D1 + v_N1 * v_D0 + lambda * v_D0 * v_D1;
            return initial_claim == expected;
        }

        std::vector<Fr> one_minus_2y(rounds);
        for (int i = 0; i < rounds; i++)
        {
            one_minus_2y[i] = Fr(1) - r_eq[i] - r_eq[i];
            if (one_minus_2y[i] == Fr(0))
                return false;
        }
        std::vector<Fr> one_minus_2y_inv(rounds);
        invVec(one_minus_2y_inv.data(), one_minus_2y.data(), rounds);

        std::vector<Fr> bs(rounds);
        std::vector<Fr> to_inv(3 * rounds);
        for (int i = 0; i < rounds; i++)
        {
            bs[i] = (Fr(1) - r_eq[i]) * one_minus_2y_inv[i];
            to_inv[3 * i + 0] = bs[i];
            to_inv[3 * i + 1] = bs[i] - Fr(1);
            to_inv[3 * i + 2] = bs[i] - Fr(2);
        }
        std::vector<Fr> inv_vals(3 * rounds);
        invVec(inv_vals.data(), to_inv.data(), 3 * rounds);

        Fr claim = initial_claim;
        for (int i = 0; i < rounds; i++)
        {
            const Fr sum0 = proof.round_polys[i * 2];
            const Fr sum2 = proof.round_polys[i * 2 + 1];
            const Fr sum1 = claim - sum0;

            const Fr challenge = challenges[static_cast<std::size_t>(i)];

            const Fr bs_m3 = bs[i] - Fr(3);
            const Fr coef_0 = -bs_m3 * inv_vals[3 * i + 0];
            const Fr three_bs_m3 = bs_m3 + bs_m3 + bs_m3;
            const Fr coef_1 = three_bs_m3 * inv_vals[3 * i + 1];
            const Fr coef_2 = -three_bs_m3 * inv_vals[3 * i + 2];
            const Fr sum3 = -(coef_0 * sum0 + coef_1 * sum1 + coef_2 * sum2);

            claim = evaluate_cubic_samples(
                sum0, sum1, sum2, sum3, challenge);
        }
        const Fr eq_val = eval_eq_at_point(r_eq, challenges);
        const Fr expected = eq_val * (v_N0 * v_D1 + v_N1 * v_D0 + lambda * v_D0 * v_D1);
        return claim == expected;
    }

    void prove_gkr_tree(
        std::vector<std::vector<Fr>> &N_layers,
        std::vector<std::vector<Fr>> &D_layers, int M,
        std::vector<LogupGkrLayerProof> &out_layers,
        std::vector<Fr> &out_leaf_point,
        int thread_num,
        FiatShamirProofWriter &transcript, bool unit_numerator_leaf = false)
    {
        out_layers.clear();
        out_layers.reserve(M);

        std::vector<Fr> r_claim;
        std::vector<Fr>().swap(N_layers[M]);
        std::vector<Fr>().swap(D_layers[M]);

        for (int upper = M; upper >= 1; upper--)
        {
            const int lower = upper - 1;
            const int vars = M - upper;

            LogupGkrLayerProof layer;
            const int layer_index = M - upper;
            transcript.transcript().append_u64(layer_index);
            const Fr lambda = transcript.challenge();

            std::vector<Fr> layer_challenges;
            const auto prove_layer = unit_numerator_leaf && lower == 0
                                         ? gkr_layer_sumcheck_prove<true>
                                         : gkr_layer_sumcheck_prove<false>;
            prove_layer(vars, r_claim,
                        N_layers[lower], D_layers[lower], lambda,
                        layer.sumcheck, layer.v_N0, layer.v_N1,
                        layer.v_D0, layer.v_D1,
                        thread_num, transcript, layer_challenges);

            transcript.write_fr(layer.v_N0);
            transcript.write_fr(layer.v_N1);
            transcript.write_fr(layer.v_D0);
            transcript.write_fr(layer.v_D1);
            const Fr alpha = transcript.challenge_excluding(std::vector<Fr>{Fr(1), Fr(1) / Fr(2)});

            std::vector<Fr> new_r;
            new_r.reserve(static_cast<size_t>(vars) + 1);
            new_r.push_back(alpha);
            new_r.insert(
                new_r.end(), layer_challenges.begin(), layer_challenges.end());
            r_claim = std::move(new_r);

            out_layers.push_back(std::move(layer));
        }

        out_leaf_point = std::move(r_claim);
    }
}

void LogupGkrProver::init(std::size_t generator_count)
{
    const std::size_t capacity = std::size_t(1) << (LogupGkrMaxVariables / 2);
    if (generator_count == 0 || generator_count > capacity)
    {
        throw std::invalid_argument("LogupGkrProver generator count is outside its capacity");
    }
    if (pk_.srs_g1.size() < generator_count)
    {
        std::cerr << "Error: MVKZG SRS too small for LogupGkrProver (need "
                  << generator_count
                  << ", have " << pk_.srs_g1.size() << ")" << std::endl;
        throw std::runtime_error("MVKZG SRS too small for LogupGkrProver");
    }
    hyrax::load_generators(
        g, pk_.srs_g1, static_cast<int>(generator_count));
}
namespace
{

    void append_logup_gkr_statement(
        FiatShamirTranscript &transcript, const LogupGkrProof &proof,
        bool defer_query_opening = false)
    {
        transcript.append_u64(static_cast<std::uint64_t>(proof.mode));
        transcript.append_u64(proof.m);
        transcript.append_u64(proof.n);
        transcript.append_u64(proof.log_m);
        transcript.append_u64(proof.log_n);
        transcript.append_u64(defer_query_opening ? 1 : 0);
    }

    Fr append_logup_gkr_combined_commitments(
        FiatShamirProofWriter &transcript, const LogupGkrProof &proof)
    {
        transcript.write_g1_vector(proof.f_combined_comm);
        transcript.write_g1_vector(proof.t_combined_comm);
        return transcript.challenge();
    }

    void append_logup_gkr_roots(
        FiatShamirProofWriter &transcript, const LogupGkrProof &proof)
    {
        transcript.write_fr(proof.query_N_root);
        transcript.write_fr(proof.query_D_root);
        transcript.write_fr(proof.table_N_root);
        transcript.write_fr(proof.table_D_root);
    }

    bool valid_logup_gkr_shape(const LogupGkrProof &proof)
    {
        if ((proof.mode != LogupGkrMode::Values) || proof.log_m < 0 || proof.log_n < 0 || proof.log_m > LogupGkrMaxVariables || proof.log_n > LogupGkrMaxVariables || proof.m != (1 << proof.log_m) || proof.n != (1 << proof.log_n) || proof.query_layers.size() != static_cast<std::size_t>(proof.log_m) || proof.table_layers.size() != static_cast<std::size_t>(proof.log_n))
        {
            return false;
        }
        const std::size_t query_rows = std::size_t(1) << (proof.log_m / 2);
        const std::size_t table_rows = std::size_t(1) << (proof.log_n / 2);
        if (proof.f_combined_comm.size() != query_rows || proof.c_comm.size() != table_rows || proof.t_combined_comm.size() != table_rows)
        {
            return false;
        }
        for (int layer = 0; layer < proof.log_m; ++layer)
        {
            if (proof.query_layers[static_cast<std::size_t>(layer)]
                    .sumcheck.round_polys.size() != static_cast<std::size_t>(2 * layer))
            {
                return false;
            }
        }
        for (int layer = 0; layer < proof.log_n; ++layer)
        {
            if (proof.table_layers[static_cast<std::size_t>(layer)]
                    .sumcheck.round_polys.size() != static_cast<std::size_t>(2 * layer))
            {
                return false;
            }
        }
        return true;
    }

    struct LogupGkrLayerCoins
    {
        Fr lambda = Fr(0);
        Fr alpha = Fr(0);
        std::vector<Fr> sumcheck;
    };

    struct ParsedLogupGkrWire
    {
        LogupGkrProof proof;
        Fr denominator_shift = Fr(0);
        std::vector<LogupGkrLayerCoins> query_coins;
        std::vector<LogupGkrLayerCoins> table_coins;
    };

    bool parse_logup_gkr_wire(
        const std::vector<std::uint8_t> &wire_bytes,
        LogupGkrMode mode,
        int m,
        int n,
        int log_m,
        int log_n,
        ParsedLogupGkrWire &parsed,
        bool defer_query_opening = false,
        FiatShamirTranscript *shared_transcript = nullptr)
    {
        if ((mode != LogupGkrMode::Values) || log_m < 0 || log_n < 0 || log_m > LogupGkrMaxVariables || log_n > LogupGkrMaxVariables || m != (1 << log_m) || n != (1 << log_n))
        {
            return false;
        }
        try
        {
            parsed.proof.mode = mode;
            parsed.proof.m = m;
            parsed.proof.n = n;
            parsed.proof.log_m = log_m;
            parsed.proof.log_n = log_n;
            const std::size_t query_rows = std::size_t(1) << (log_m / 2);
            const std::size_t table_rows = std::size_t(1) << (log_n / 2);
            FiatShamirProofReader reader(wire_bytes, shared_transcript);
            append_logup_gkr_statement(
                reader.transcript(), parsed.proof, defer_query_opening);

            parsed.proof.c_comm = reader.read_g1_vector(table_rows);
            parsed.proof.f_combined_comm = reader.read_g1_vector(query_rows);
            parsed.proof.t_combined_comm = reader.read_g1_vector(table_rows);
            parsed.denominator_shift = reader.challenge();

            parsed.proof.query_N_root = reader.read_fr();
            parsed.proof.query_D_root = reader.read_fr();
            parsed.proof.table_N_root = reader.read_fr();
            parsed.proof.table_D_root = reader.read_fr();

            auto parse_tree = [&](
                                  int M,
                                  std::vector<LogupGkrLayerProof> &layers,
                                  std::vector<LogupGkrLayerCoins> &coins)
            {
                layers.resize(static_cast<std::size_t>(M));
                coins.resize(static_cast<std::size_t>(M));
                for (int layer_index = 0; layer_index < M; ++layer_index)
                {
                    reader.transcript().append_u64(layer_index);
                    LogupGkrLayerCoins &layer_coins =
                        coins[static_cast<std::size_t>(layer_index)];
                    LogupGkrLayerProof &layer =
                        layers[static_cast<std::size_t>(layer_index)];
                    layer_coins.lambda = reader.challenge();
                    layer.sumcheck.round_polys.reserve(
                        static_cast<std::size_t>(2 * layer_index));
                    layer_coins.sumcheck.reserve(layer_index);
                    for (int round = 0; round < layer_index; ++round)
                    {
                        layer.sumcheck.round_polys.push_back(
                            reader.read_fr());
                        layer.sumcheck.round_polys.push_back(
                            reader.read_fr());
                        layer_coins.sumcheck.push_back(
                            reader.challenge_excluding(std::vector<Fr>{Fr(1), Fr(1) / Fr(2)}));
                    }
                    layer.v_N0 = reader.read_fr();
                    layer.v_N1 = reader.read_fr();
                    layer.v_D0 = reader.read_fr();
                    layer.v_D1 = reader.read_fr();
                    layer_coins.alpha = reader.challenge_excluding(std::vector<Fr>{Fr(1), Fr(1) / Fr(2)});
                }
            };

            parse_tree(
                log_m, parsed.proof.query_layers, parsed.query_coins);
            parse_tree(
                log_n, parsed.proof.table_layers, parsed.table_coins);

            const std::size_t query_opening_size =
                static_cast<std::size_t>(log_m - log_m / 2);
            const std::size_t table_opening_size =
                static_cast<std::size_t>(log_n - log_n / 2);
            if (!defer_query_opening)
            {
                parsed.proof.open_f_combined.eval_proof.q_comm =
                    reader.read_g1_vector(query_opening_size);
            }
            parsed.proof.open_c.eval_proof.q_comm = reader.read_g1_vector(table_opening_size);
            parsed.proof.open_t_combined.eval_proof.q_comm =
                reader.read_g1_vector(table_opening_size);
            if (!reader.at_end())
                return false;
            parsed.proof.g1_encoding = reader.encoding();
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    bool replay_logup_gkr_proof_object(
        const LogupGkrProof &proof,
        bool defer_query_opening,
        ParsedLogupGkrWire &parsed,
        FiatShamirTranscript *shared_transcript = nullptr)
    {
#ifndef NDEBUG
        assert(valid_logup_gkr_shape(proof));
        const std::size_t query_opening_size =
            static_cast<std::size_t>(proof.log_m - proof.log_m / 2);
        const std::size_t table_opening_size =
            static_cast<std::size_t>(proof.log_n - proof.log_n / 2);
        assert((defer_query_opening
                    ? proof.open_f_combined.eval_proof.q_comm.empty()
                    : proof.open_f_combined.eval_proof.q_comm.size() ==
                          query_opening_size));
        assert(proof.open_c.eval_proof.q_comm.size() == table_opening_size);
        assert(proof.open_t_combined.eval_proof.q_comm.size() ==
               table_opening_size);
#endif
        try
        {
            parsed = ParsedLogupGkrWire{};
            FiatShamirProofWriter transcript(
                proof.g1_encoding, false, shared_transcript);
            append_logup_gkr_statement(
                transcript.transcript(), proof, defer_query_opening);
            transcript.write_g1_vector(proof.c_comm);
            parsed.denominator_shift =
                append_logup_gkr_combined_commitments(transcript, proof);
            append_logup_gkr_roots(transcript, proof);

            auto replay_tree = [&](
                                   int layers_count,
                                   const std::vector<LogupGkrLayerProof> &layers,
                                   std::vector<LogupGkrLayerCoins> &coins)
            {
                coins.resize(static_cast<std::size_t>(layers_count));
                for (int layer_index = 0; layer_index < layers_count;
                     ++layer_index)
                {
                    transcript.transcript().append_u64(layer_index);
                    const LogupGkrLayerProof &layer =
                        layers[static_cast<std::size_t>(layer_index)];
                    LogupGkrLayerCoins &layer_coins =
                        coins[static_cast<std::size_t>(layer_index)];
                    layer_coins.lambda = transcript.challenge();
                    layer_coins.sumcheck.reserve(layer_index);
                    for (int round = 0; round < layer_index; ++round)
                    {
                        transcript.write_fr(layer.sumcheck.round_polys[static_cast<std::size_t>(2 * round)]);
                        transcript.write_fr(layer.sumcheck.round_polys[static_cast<std::size_t>(2 * round + 1)]);
                        layer_coins.sumcheck.push_back(
                            transcript.challenge_excluding(std::vector<Fr>{Fr(1), Fr(1) / Fr(2)}));
                    }
                    transcript.write_fr(layer.v_N0);
                    transcript.write_fr(layer.v_N1);
                    transcript.write_fr(layer.v_D0);
                    transcript.write_fr(layer.v_D1);
                    layer_coins.alpha = transcript.challenge_excluding(std::vector<Fr>{Fr(1), Fr(1) / Fr(2)});
                }
            };

            replay_tree(
                proof.log_m, proof.query_layers, parsed.query_coins);
            replay_tree(
                proof.log_n, proof.table_layers, parsed.table_coins);
            if (!defer_query_opening)
            {
                transcript.write_g1_vector(proof.open_f_combined.eval_proof.q_comm);
            }
            transcript.write_g1_vector(proof.open_c.eval_proof.q_comm);
            transcript.write_g1_vector(proof.open_t_combined.eval_proof.q_comm);
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    bool logup_gkr_verify_impl(
        const MVKZG_VerifierKey &vk,
        const LogupGkrProof &encoded_proof,
        const LogupGkrTableCommitment *table_commitment,
        LookupReadClaim *read_claim,
        bool defer_query_opening,
        FiatShamirTranscript *shared_transcript = nullptr,
        bool in_memory_fiat_shamir = false,
        hyrax::OpeningBatchVerifier *opening_batch = nullptr,
        const std::vector<std::uint8_t> *wire_only = nullptr)
    {
        if (read_claim != nullptr)
            *read_claim = LookupReadClaim{};
        auto fail = [](const char *msg)
        {
            std::cerr << "LogUp-GKR verify failed: " << msg << std::endl;
            return false;
        };
        ParsedLogupGkrWire parsed;
        const bool serialized = !in_memory_fiat_shamir;
        const LogupGkrProof *proof_view = nullptr;
        if (serialized)
        {
            if (!parse_logup_gkr_wire(
                    wire_only != nullptr ? *wire_only : encoded_proof.wire_bytes,
                    encoded_proof.mode,
                    encoded_proof.m, encoded_proof.n,
                    encoded_proof.log_m, encoded_proof.log_n, parsed,
                    defer_query_opening, shared_transcript))
            {
                return fail("wire_parse");
            }
            proof_view = &parsed.proof;
        }
        else
        {
            FiatShamirTranscript local_transcript;
            FiatShamirTranscript *transcript = shared_transcript != nullptr
                                                   ? shared_transcript
                                                   : &local_transcript;
            if (!replay_logup_gkr_proof_object(
                    encoded_proof, defer_query_opening, parsed, transcript))
            {
                return fail("fiat_shamir_replay");
            }
            proof_view = &encoded_proof;
        }
        const LogupGkrProof &proof = *proof_view;
        if (serialized && !valid_logup_gkr_shape(proof))
        {
            return fail("proof_shape");
        }
        if (read_claim != nullptr && proof.mode != LogupGkrMode::Values)
        {
            return fail("read_claim_requires_value_mode");
        }
        if (serialized && wire_only == nullptr &&
            (encoded_proof.f_combined_comm != proof.f_combined_comm || encoded_proof.c_comm != proof.c_comm || encoded_proof.t_combined_comm != proof.t_combined_comm))
        {
            return fail("wire_commitment_view_mismatch");
        }
        if (table_commitment != nullptr)
        {
            if (proof.mode != table_commitment->mode || proof.n != table_commitment->padded_n || proof.log_n != table_commitment->log_n || proof.t_combined_comm.size() != table_commitment->value_commitments.size())
            {
                return fail("public_table_shape");
            }
            for (std::size_t row = 0; row < proof.t_combined_comm.size(); ++row)
            {
                G1 expected = table_commitment->value_commitments[row];
                if (expected != proof.t_combined_comm[row])
                {
                    return fail("public_table_commitment");
                }
            }
        }

        const Fr denominator_shift = parsed.denominator_shift;

        if (proof.query_N_root * proof.table_D_root != proof.query_D_root * proof.table_N_root)
        {
            return fail("root_cross_ratio");
        }

        auto descend = [&](int M, const std::vector<LogupGkrLayerProof> &layers,
                           const std::vector<LogupGkrLayerCoins> &coins,
                           Fr N_root, Fr D_root,
                           std::vector<Fr> &out_r, Fr &out_n, Fr &out_d) -> bool
        {
            std::vector<Fr> r_claim;
            Fr n_claim = N_root;
            Fr d_claim = D_root;
            for (int upper = M; upper >= 1; --upper)
            {
                const int vars = M - upper;
                const int layer_index = M - upper;
                const auto &layer = layers[static_cast<std::size_t>(layer_index)];
                const auto &layer_coins =
                    coins[static_cast<std::size_t>(layer_index)];
                const Fr lambda = layer_coins.lambda;
                const Fr S = n_claim + lambda * d_claim;
                if (!verify_gkr_layer_sumcheck(
                        layer.sumcheck, r_claim, S, lambda,
                        layer.v_N0, layer.v_N1, layer.v_D0, layer.v_D1,
                        layer_coins.sumcheck))
                {
                    return false;
                }
                const Fr alpha = layer_coins.alpha;
                const Fr one_minus_alpha = Fr(1) - alpha;
                n_claim = one_minus_alpha * layer.v_N0 + alpha * layer.v_N1;
                d_claim = one_minus_alpha * layer.v_D0 + alpha * layer.v_D1;
                std::vector<Fr> new_r;
                new_r.reserve(static_cast<std::size_t>(vars) + 1);
                new_r.push_back(alpha);
                new_r.insert(
                    new_r.end(), layer_coins.sumcheck.begin(),
                    layer_coins.sumcheck.end());
                r_claim = std::move(new_r);
            }
            out_r = std::move(r_claim);
            out_n = n_claim;
            out_d = d_claim;
            return true;
        };

        std::vector<Fr> query_point;
        Fr query_n, query_d;
        if (!descend(
                proof.log_m, proof.query_layers,
                parsed.query_coins,
                proof.query_N_root, proof.query_D_root, query_point, query_n, query_d))
        {
            return fail("query_layer_sumcheck");
        }
        if (query_n != Fr(1))
            return fail("query_n_leaf_not_1");
        if (!defer_query_opening && opening_batch == nullptr && !hyrax::verify(proof.open_f_combined, query_point.data(), query_d - denominator_shift, proof.f_combined_comm.data(), proof.log_m, vk))
        {
            return fail("open_f_combined");
        }

        std::vector<Fr> table_point;
        Fr table_n, table_d;
        if (!descend(
                proof.log_n, proof.table_layers,
                parsed.table_coins,
                proof.table_N_root, proof.table_D_root, table_point, table_n, table_d))
        {
            return fail("table_layer_sumcheck");
        }
        if (opening_batch == nullptr && !hyrax::verify(
                                            proof.open_c, table_point.data(), table_n,
                                            proof.c_comm.data(),
                                            proof.log_n, vk))
        {
            return fail("open_c");
        }
        if (opening_batch == nullptr && !hyrax::verify(
                                            proof.open_t_combined, table_point.data(),
                                            table_d - denominator_shift,
                                            proof.t_combined_comm.data(), proof.log_n, vk))
        {
            return fail("open_t_combined");
        }
        if (opening_batch != nullptr &&
            ((!defer_query_opening && !opening_batch->append(
                                          proof.open_f_combined, query_point.data(),
                                          query_d - denominator_shift,
                                          proof.f_combined_comm.data(), proof.log_m)) ||
             !opening_batch->append(
                 proof.open_c, table_point.data(), table_n,
                 proof.c_comm.data(), proof.log_n) ||
             !opening_batch->append(
                 proof.open_t_combined, table_point.data(),
                 table_d - denominator_shift,
                 proof.t_combined_comm.data(), proof.log_n)))
        {
            return fail("opening_batch_append");
        }
        if (read_claim != nullptr)
        {
            read_claim->commitments = proof.f_combined_comm;
            read_claim->point = std::move(query_point);
            read_claim->evaluation = query_d - denominator_shift;
        }
        return true;
    }

}

LogupGkrProof LogupGkrProver::logup_gkr_prove_prepared_values(
    const LogupGkrPreparedValueWitness &query,
    const LogupGkrPreparedValueTable &table,
    int thread_num,
    G1WireEncoding encoding,
    LogupGkrProveTiming *timing,
    const LogupGkrTableCommitment *preprocessed_table,
    bool defer_query_opening,
    bool retain_wire_bytes,
    FiatShamirTranscript *shared_transcript,
    const std::vector<G1> *precommitted_query)
{
    if (timing != nullptr)
        *timing = LogupGkrProveTiming{};
    const auto total_start = std::chrono::steady_clock::now();

    LogupGkrProof proof;
    proof.mode = LogupGkrMode::Values;
    proof.log_m = query.log_m;
    proof.log_n = table.log_n;
    const int padded_m = query.padded_m;
    const int padded_n = table.padded_n;
    proof.m = padded_m;
    proof.n = padded_n;
    const std::vector<Fr> &query_values = query.values;
    const std::vector<ll> &multiplicities = query.multiplicities;
    const std::vector<Fr> &table_values = *table.values;

    const int row_m = 1 << (proof.log_m / 2);
    const int row_n = 1 << (proof.log_n / 2);
    if (precommitted_query != nullptr)
    {
        if (precommitted_query->size() != static_cast<std::size_t>(row_m))
        {
            throw std::invalid_argument(
                "precommitted LogUp query commitment has the wrong shape");
        }
        proof.f_combined_comm = *precommitted_query;
        G1::normalizeVec(
            proof.f_combined_comm.data(), proof.f_combined_comm.data(),
            proof.f_combined_comm.size());
    }
    else
    {
        auto commitment_start = std::chrono::steady_clock::now();
        G1 *query_commitment = hyrax::commit(
            query_values.data(), g, proof.log_m, thread_num);
        proof.f_combined_comm.assign(
            query_commitment, query_commitment + row_m);
        delete[] query_commitment;
        if (timing != nullptr)
        {
            timing->query_commitment_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - commitment_start)
                    .count();
        }
    }

    auto commitment_start = std::chrono::steady_clock::now();
    G1 *multiplicity_commitment = hyrax::commit(
        multiplicities.data(), g, proof.log_n, thread_num);
    proof.c_comm.assign(
        multiplicity_commitment, multiplicity_commitment + row_n);
    delete[] multiplicity_commitment;
    if (timing != nullptr)
    {
        timing->multiplicity_commitment_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - commitment_start)
                .count();
    }

    if (preprocessed_table != nullptr)
    {
        if (preprocessed_table->mode != LogupGkrMode::Values ||
            preprocessed_table->padded_n != padded_n ||
            preprocessed_table->log_n != proof.log_n ||
            preprocessed_table->value_commitments.size() != static_cast<std::size_t>(row_n))
        {
            throw std::invalid_argument(
                "preprocessed LogUp table commitment has the wrong shape");
        }
        proof.t_combined_comm = preprocessed_table->value_commitments;
    }
    else
    {
        commitment_start = std::chrono::steady_clock::now();
        G1 *table_value_commitment = hyrax::commit(
            table_values.data(), g, proof.log_n, thread_num);
        proof.t_combined_comm.assign(
            table_value_commitment, table_value_commitment + row_n);
        delete[] table_value_commitment;
        if (timing != nullptr)
        {
            timing->table_commitment_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - commitment_start)
                    .count();
        }
    }

    FiatShamirProofWriter transcript(
        encoding, retain_wire_bytes,
        shared_transcript);
    append_logup_gkr_statement(
        transcript.transcript(), proof, defer_query_opening);
    transcript.write_g1_vector(proof.c_comm);
    const Fr denominator_shift =
        append_logup_gkr_combined_commitments(transcript, proof);

    const auto tree_start = std::chrono::steady_clock::now();
    std::vector<std::vector<Fr>> query_N(
        static_cast<std::size_t>(proof.log_m) + 1);
    std::vector<std::vector<Fr>> query_D(
        static_cast<std::size_t>(proof.log_m) + 1);
    // The query numerator leaf is the public constant one.
    query_D[0].resize(static_cast<std::size_t>(padded_m));
    for (int i = 0; i < padded_m; ++i)
    {
        query_D[0][static_cast<std::size_t>(i)] =
            denominator_shift + query_values[static_cast<std::size_t>(i)];
    }
    build_gkr_layers(query_N, query_D, proof.log_m, thread_num, true);

    std::vector<std::vector<Fr>> table_N(
        static_cast<std::size_t>(proof.log_n) + 1);
    std::vector<std::vector<Fr>> table_D(
        static_cast<std::size_t>(proof.log_n) + 1);
    table_N[0].resize(static_cast<std::size_t>(padded_n));
    table_D[0].resize(static_cast<std::size_t>(padded_n));
    for (int j = 0; j < padded_n; ++j)
    {
        table_N[0][static_cast<std::size_t>(j)] =
            zkagent_integer::fr_from_i128(
                multiplicities[static_cast<std::size_t>(j)]);
        table_D[0][static_cast<std::size_t>(j)] =
            denominator_shift + table_values[static_cast<std::size_t>(j)];
    }
    build_gkr_layers(table_N, table_D, proof.log_n, thread_num);

    proof.query_N_root = proof.log_m == 0 ? Fr(1) : query_N[proof.log_m][0];
    proof.query_D_root = query_D[proof.log_m][0];
    proof.table_N_root = table_N[proof.log_n][0];
    proof.table_D_root = table_D[proof.log_n][0];
    append_logup_gkr_roots(transcript, proof);
    if (timing != nullptr)
    {
        timing->tree_build_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - tree_start)
                                    .count();
    }

    const auto sumcheck_start = std::chrono::steady_clock::now();
    std::vector<Fr> query_point, table_point;
    prove_gkr_tree(
        query_N, query_D, proof.log_m, proof.query_layers,
        query_point, thread_num, transcript, true);
    prove_gkr_tree(
        table_N, table_D, proof.log_n, proof.table_layers,
        table_point, thread_num, transcript);
    if (timing != nullptr)
    {
        timing->sumcheck_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - sumcheck_start)
                                  .count();
    }

    const auto opening_start = std::chrono::steady_clock::now();
    if (!defer_query_opening)
    {
        proof.open_f_combined = hyrax::open(
            query_values.data(), query_point.data(), proof.log_m, pk_);
    }
    proof.open_c = hyrax::open(
        multiplicities.data(), table_point.data(), proof.log_n, pk_);
    proof.open_t_combined = hyrax::open(
        table_values.data(), table_point.data(), proof.log_n, pk_);
    if (!defer_query_opening)
    {
        transcript.write_g1_vector(proof.open_f_combined.eval_proof.q_comm);
    }
    transcript.write_g1_vector(proof.open_c.eval_proof.q_comm);
    transcript.write_g1_vector(proof.open_t_combined.eval_proof.q_comm);
    if (timing != nullptr)
    {
        timing->opening_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - opening_start)
                                 .count();
    }
    proof.g1_encoding = encoding;
    proof.wire_bytes = transcript.take_bytes();
    if (timing != nullptr)
    {
        timing->total_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - total_start)
                               .count();
    }
    return proof;
}

LogupGkrTableCommitment LogupGkrProver::logup_gkr_commit_public_table(
    const ll *table, int n, int thread_num, LogupGkrMode mode)
{
    if (table == nullptr || n <= 0 || n > (1 << LogupGkrMaxVariables))
    {
        throw std::invalid_argument("public LogUp table must be nonempty");
    }
    if (mode != LogupGkrMode::Values)
    {
        throw std::invalid_argument("invalid public LogUp table mode");
    }
    if ((n & (n - 1)) == 0)
    {
        LogupGkrTableCommitment result;
        result.mode = mode;
        result.log_n = ceil_log2(n);
        result.padded_n = n;
        const int row_count = 1 << (result.log_n / 2);
        G1 *value_comm = hyrax::commit(
            table, g, result.log_n, std::max(thread_num, 1));
        result.value_commitments.assign(value_comm, value_comm + row_count);
        delete[] value_comm;
        return result;
    }
    std::vector<Fr> values(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        values[static_cast<std::size_t>(i)] =
            zkagent_integer::fr_from_i128(table[i]);
    }
    return logup_gkr_commit_public_table(
        values.data(), n, thread_num, mode);
}

LogupGkrTableCommitment LogupGkrProver::logup_gkr_commit_public_table(
    const Fr *table, int n, int thread_num, LogupGkrMode mode)
{
    if (table == nullptr || n <= 0 || n > (1 << LogupGkrMaxVariables))
    {
        throw std::invalid_argument("public LogUp table must be nonempty");
    }
    if (mode != LogupGkrMode::Values)
    {
        throw std::invalid_argument("invalid public LogUp table mode");
    }
    LogupGkrTableCommitment result;
    result.mode = mode;
    result.log_n = ceil_log2(n);
    result.padded_n = 1 << result.log_n;
    const int row_count = 1 << (result.log_n / 2);
    std::vector<Fr> padded_values;
    const Fr *values = table;
    if (n != result.padded_n)
    {
        padded_values.assign(result.padded_n, Fr(0));
        std::copy_n(table, n, padded_values.data());
        values = padded_values.data();
    }
    G1 *value_comm = hyrax::commit(
        values, g, result.log_n, std::max(thread_num, 1));
    result.value_commitments.assign(value_comm, value_comm + row_count);
    delete[] value_comm;
    return result;
}

bool logup_gkr_verify_serialized(
    const std::vector<std::uint8_t> &wire_bytes,
    int expected_query_count,
    const LogupGkrTableCommitment &table_commitment,
    const MVKZG_VerifierKey &vk,
    LookupReadClaim *read_claim,
    hyrax::OpeningBatchVerifier *opening_batch)
{
    if (expected_query_count <= 0 || expected_query_count > (1 << LogupGkrMaxVariables) ||
        (expected_query_count & (expected_query_count - 1)) != 0 ||
        table_commitment.mode != LogupGkrMode::Values)
        return false;
    LogupGkrProof statement;
    statement.mode = LogupGkrMode::Values;
    statement.m = expected_query_count;
    statement.log_m = ceil_log2(expected_query_count);
    statement.n = table_commitment.padded_n;
    statement.log_n = table_commitment.log_n;
    return logup_gkr_verify_impl(vk, statement, &table_commitment,
                                 read_claim, false, nullptr, false, opening_batch, &wire_bytes);
}

bool logup_gkr_verify_in_memory(
    const LogupGkrProof &proof, const LogupGkrTableCommitment &table,
    const MVKZG_VerifierKey &vk, LookupReadClaim *read_claim,
    hyrax::OpeningBatchVerifier *opening_batch)
{
    if (proof.mode != LogupGkrMode::Values || !valid_logup_gkr_shape(proof) ||
        proof.open_f_combined.eval_proof.q_comm.size() != std::size_t(proof.log_m - proof.log_m / 2) ||
        proof.open_c.eval_proof.q_comm.size() != std::size_t(proof.log_n - proof.log_n / 2) ||
        proof.open_t_combined.eval_proof.q_comm.size() != std::size_t(proof.log_n - proof.log_n / 2))
        return false;
    return logup_gkr_verify_impl(vk, proof, &table, read_claim,
                                 false, nullptr, true, opening_batch);
}

bool LogupGkrProver::logup_gkr_verify_in_memory_with_table(
    const LogupGkrProof &proof,
    const LogupGkrTableCommitment &table_commitment,
    LookupReadClaim *read_claim,
    bool defer_query_opening,
    FiatShamirTranscript *shared_transcript,
    hyrax::OpeningBatchVerifier *opening_batch) const
{
    return logup_gkr_verify_impl(
        vk_, proof, &table_commitment, read_claim,
        defer_query_opening, shared_transcript, true, opening_batch);
}

u64 logup_gkr_proof_size_bytes(const LogupGkrProof &proof)
{
    const u64 groups = proof.f_combined_comm.size() + proof.c_comm.size() + proof.t_combined_comm.size() + proof.open_f_combined.eval_proof.q_comm.size() + proof.open_c.eval_proof.q_comm.size() + proof.open_t_combined.eval_proof.q_comm.size();
    u64 fields = 4;
    for (const LogupGkrLayerProof &layer : proof.query_layers)
    {
        fields += layer.sumcheck.round_polys.size() + 4;
    }
    for (const LogupGkrLayerProof &layer : proof.table_layers)
    {
        fields += layer.sumcheck.round_polys.size() + 4;
    }
    return groups * g1_wire_size(G1WireEncoding::Compressed) + fields * 32ULL;
}
