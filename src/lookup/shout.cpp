

#include "common/bit_utils.hpp"
#include "common/polynomial.h"
#include "common/mle.hpp"
#include "lookup/shout.hpp"
#include "common/fiat_shamir.hpp"
#include "pcs/srs.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <omp.h>

namespace
{

    using zkagent_integer::is_power_of_two;

    using zkagent_integer::log2_power_of_two;

    bool valid_shout_dimensions(int K, int T)
    {
        return is_power_of_two(K) && is_power_of_two(T);
    }

    void append_shout_statement(
        FiatShamirTranscript &transcript,
        const std::vector<Fr> &lookup_table,
        int K,
        int T,
        bool defer_read_value_opening = false)
    {
        transcript.append_u64(K);
        transcript.append_u64(T);
        transcript.append_u64(defer_read_value_opening ? 1 : 0);
        transcript.append_fr_vector(lookup_table);
    }

    void append_shout_commitments(
        FiatShamirProofWriter &transcript, const ShoutProof &proof)
    {
        transcript.write_g1_array(proof.RA_comm,
            static_cast<std::size_t>(proof.RA_comm_size));
        transcript.write_g1_vector(proof.read_value_commitments);
    }

    void append_shout_opening(
        FiatShamirProofWriter &transcript, const hyrax::OpeningProof &proof)
    {
        transcript.write_g1_vector(proof.eval_proof.q_comm);
    }

    template <class Transcript>
    std::vector<Fr> derive_shout_point(
        Transcript &transcript, int length,
        bool exclude_one)
    {
        std::vector<Fr> point(static_cast<std::size_t>(length));
        for (int i = 0; i < length; ++i)
        {
            point[static_cast<std::size_t>(i)] = exclude_one
                                                     ? transcript.challenge_not_one()
                                                     : transcript.challenge();
        }
        return point;
    }

    void normalize_g1_array_in_place(G1 *points, std::size_t count)
    {
        if (count != 0)
            G1::normalizeVec(points, points, count);
    }

    void normalize_opening_in_place(hyrax::OpeningProof &proof)
    {
        normalize_g1_array_in_place(
            proof.eval_proof.q_comm.data(), proof.eval_proof.q_comm.size());
    }

}

void ShoutProver::ensure_local_F_workspace(int num_threads, int K)
{
    if (num_threads <= 0)
    {
        num_threads = 1;
    }
    if (local_F_threads_ != num_threads || local_F_K_ != K)
    {
        local_F_buffers_.assign(num_threads, std::vector<Fr>(K, Fr(0)));
        local_F_marks_.assign(num_threads, std::vector<unsigned char>(K, 0));
        local_F_touched_.assign(num_threads, std::vector<int>());
        local_F_threads_ = num_threads;
        local_F_K_ = K;
        return;
    }
    for (int t = 0; t < num_threads; t++)
    {
        local_F_touched_[t].clear();
    }
}

std::vector<Fr> ShoutProver::compute_read_address_poly(
    const std::vector<int> &read_addresses,
    const std::vector<Fr> &E,
    int K,
    int num_threads)
{
    std::vector<Fr> F(K, Fr(0));

    int total = static_cast<int>(read_addresses.size());
    if (total <= 0 || K <= 0)
    {
        return F;
    }

    int num_workers = std::max(1, std::min(num_threads, total));
    long long dense_cost = static_cast<long long>(K) * num_workers;

    bool use_sparse = dense_cost > static_cast<long long>(total) * 8;

    if (use_sparse)
    {
        std::vector<std::unordered_map<int, Fr>> local_maps(num_workers);
#pragma omp parallel num_threads(num_workers)
        {
            int tid = omp_get_thread_num();
            auto &local = local_maps[tid];
            int reserve_hint = (total + num_workers - 1) / num_workers;
            local.reserve(static_cast<size_t>(reserve_hint) * 2);

#pragma omp for schedule(static)
            for (int j = 0; j < total; j++)
            {
                int addr = read_addresses[j];
                if (addr < 0 || addr >= K)
                {
                    continue;
                }
                auto it = local.find(addr);
                if (it == local.end())
                {
                    local.emplace(addr, E[j]);
                }
                else
                {
                    it->second += E[j];
                }
            }
        }

        for (int t = 0; t < num_workers; t++)
        {
            for (const auto &kv : local_maps[t])
            {
                F[kv.first] += kv.second;
            }
        }
        return F;
    }

    ensure_local_F_workspace(num_workers, K);

#pragma omp parallel num_threads(num_workers)
    {
        int tid = omp_get_thread_num();
        auto &buf = local_F_buffers_[tid];
        auto &marks = local_F_marks_[tid];
        auto &touched = local_F_touched_[tid];

#pragma omp for schedule(static)
        for (int j = 0; j < total; j++)
        {
            int addr = read_addresses[j];
            if (addr < 0 || addr >= K)
            {
                continue;
            }
            if (!marks[addr])
            {
                marks[addr] = 1;
                touched.push_back(addr);
            }
            buf[addr] += E[j];
        }
    }

    for (int t = 0; t < num_workers; t++)
    {
        auto &buf = local_F_buffers_[t];
        auto &marks = local_F_marks_[t];
        auto &touched = local_F_touched_[t];
        for (int idx : touched)
        {
            F[idx] += buf[idx];
            buf[idx] = Fr(0);
            marks[idx] = 0;
        }
        touched.clear();
    }

    return F;
}

void ShoutProver::bind_polynomial(
    std::vector<Fr> &poly,
    Fr r_j,
    int num_threads)
{
    int len = poly.size();
    int new_len = len / 2;

    bind_scratch_.resize(new_len);

#pragma omp parallel for num_threads(num_threads) if (new_len >= 1024)
    for (int i = 0; i < new_len; i++)
    {
        const Fr left = poly[2 * i];
        const Fr right = poly[2 * i + 1];
        bind_scratch_[i] = left + r_j * (right - left);
    }

    poly.swap(bind_scratch_);
}

void ShoutProver::prove_core_piop_hamming(
    std::vector<Fr> &F_poly,
    std::vector<Fr> &val_poly,
    Fr z,
    Fr &rv_claim,
    std::vector<Fr> &r_address,
    std::vector<Fr> &compressed_polys,
    int num_rounds,
    int num_threads,
    FiatShamirProofWriter &transcript)
{
    Fr previous_claim = rv_claim + z;
    int max_threads = std::max(1, num_threads);
    std::vector<Fr> local_eval_0(max_threads, Fr(0));
    std::vector<Fr> local_eval_2(max_threads, Fr(0));

    for (int round = 0; round < num_rounds; round++)
    {
        int len = F_poly.size();
        int half_len = len / 2;

        int num_actual_threads = std::min(num_threads, half_len);
        std::fill_n(local_eval_0.begin(), num_actual_threads, Fr(0));
        std::fill_n(local_eval_2.begin(), num_actual_threads, Fr(0));

#pragma omp parallel num_threads(num_actual_threads) if (half_len >= 1024)
        {
            int tid = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < half_len; i++)
            {

                Fr F_0 = F_poly[2 * i];
                Fr val_0 = val_poly[2 * i];
                local_eval_0[tid] += F_0 * (z + val_0);

                Fr F_1 = F_poly[2 * i + 1];
                Fr val_1 = val_poly[2 * i + 1];
                Fr F_2 = F_1 + F_1 - F_0;
                Fr val_2 = val_1 + val_1 - val_0;
                local_eval_2[tid] += F_2 * (z + val_2);
            }
        }

        Fr eval_0 = Fr(0);
        Fr eval_2 = Fr(0);
        for (int t = 0; t < num_actual_threads; t++)
        {
            eval_0 += local_eval_0[t];
            eval_2 += local_eval_2[t];
        }

        compressed_polys.push_back(eval_0);
        compressed_polys.push_back(eval_2);

        Fr eval_1 = previous_claim - eval_0;
        transcript.write_fr(eval_0);
        transcript.write_fr(eval_2);
        const Fr r_j = transcript.challenge_not_one();
        r_address.push_back(r_j);

        previous_claim = polynomial::evaluate_quadratic_samples(eval_0, eval_1, eval_2, r_j);

        bind_polynomial(F_poly, r_j, num_threads);
        bind_polynomial(val_poly, r_j, num_threads);
    }
}

void ShoutProver::prove_booleanity(
    const std::vector<int> &read_addresses,
    const std::vector<Fr> &r_address,
    std::vector<Fr> &D,
    std::vector<Fr> &G,
    Fr &ra_claim_prime,
    std::vector<Fr> &r_address_prime,
    std::vector<Fr> &compressed_polys,
    int K,
    int T,
    int num_threads,
    FiatShamirProofWriter &transcript)
{
    const int log_K = log2_power_of_two(K);
    const int log_T = log2_power_of_two(T);

    int max_threads = std::max(1, num_threads);
    std::vector<Fr> local_eval_0(max_threads, Fr(0));
    std::vector<Fr> local_eval_2(max_threads, Fr(0));
    std::vector<Fr> local_eval_3(max_threads, Fr(0));

    std::vector<Fr> B = mle::equality_weights(r_address.data(), log2_power_of_two(K), num_threads);

    std::vector<Fr> F_table(K, Fr(0));
    F_table[0] = Fr(1);

    struct AddressRoundCoefficients
    {
        Fr at_0;
        Fr at_2_left;
        Fr at_2_right;
        Fr at_3_left;
        Fr at_3_right;
    };
    const auto address_coefficients = [](const Fr &base)
    {
        const Fr square = base * base;
        return AddressRoundCoefficients{
            square - base,
            square + base,
            Fr(4) * square - Fr(2) * base,
            Fr(4) * square + Fr(2) * base,
            Fr(9) * square - Fr(3) * base};
    };
    std::vector<AddressRoundCoefficients> coefficient_workspace;

    r_address_prime.clear();

    for (int round = 0; round < log_K; round++)
    {
        int m = round + 1;
        int len = B.size();
        int half_len = len / 2;
        const int half_size = 1 << (m - 1);
        // These coefficients depend only on the already bound address bits.
        // Cache them while multiple high-address blocks reuse them.
        const bool cache_coefficients = half_len > 1;
        if (cache_coefficients)
        {
            coefficient_workspace.resize(half_size);
#pragma omp parallel for num_threads(num_threads) if (half_size >= 512)
            for (int k_low = 0; k_low < half_size; ++k_low)
            {
                coefficient_workspace[k_low] = address_coefficients(F_table[k_low]);
            }
        }

        int num_actual_threads_bool = std::min(num_threads, half_len);
        std::fill_n(local_eval_0.begin(), num_actual_threads_bool, Fr(0));
        std::fill_n(local_eval_2.begin(), num_actual_threads_bool, Fr(0));
        std::fill_n(local_eval_3.begin(), num_actual_threads_bool, Fr(0));

#pragma omp parallel num_threads(num_actual_threads_bool) if (half_len >= 512)
        {
            int tid = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < half_len; i++)
            {
                Fr B_0 = B[2 * i];
                Fr B_1 = B[2 * i + 1];
                Fr B_2 = B_1 + B_1 - B_0;
                Fr B_3 = B_2 + B_1 - B_0;

                Fr inner_0 = Fr(0), inner_2 = Fr(0), inner_3 = Fr(0);

                int k_base = i << m;
                const auto accumulate = [&](int k_low, const AddressRoundCoefficients &coefficients)
                {
                    int k0 = k_base + k_low;
                    int k1 = k0 + half_size;
                    Fr G_k0 = G[k0];
                    Fr G_k1 = G[k1];

                    inner_0 += G_k0 * coefficients.at_0;
                    inner_2 += G_k0 * coefficients.at_2_left + G_k1 * coefficients.at_2_right;
                    inner_3 += G_k0 * coefficients.at_3_left + G_k1 * coefficients.at_3_right;
                };
                if (cache_coefficients)
                {
                    for (int k_low = 0; k_low < half_size; ++k_low)
                        accumulate(k_low, coefficient_workspace[k_low]);
                }
                else
                {
                    for (int k_low = 0; k_low < half_size; ++k_low)
                        accumulate(k_low, address_coefficients(F_table[k_low]));
                }

                local_eval_0[tid] += B_0 * inner_0;
                local_eval_2[tid] += B_2 * inner_2;
                local_eval_3[tid] += B_3 * inner_3;
            }
        }

        Fr eval_0 = Fr(0), eval_2 = Fr(0), eval_3 = Fr(0);
        for (int t = 0; t < num_actual_threads_bool; t++)
        {
            eval_0 += local_eval_0[t];
            eval_2 += local_eval_2[t];
            eval_3 += local_eval_3[t];
        }

        compressed_polys.push_back(eval_0);
        compressed_polys.push_back(eval_2);
        compressed_polys.push_back(eval_3);

        transcript.write_fr(eval_0);
        transcript.write_fr(eval_2);
        transcript.write_fr(eval_3);
        const Fr r_j = transcript.challenge_not_one();
        r_address_prime.push_back(r_j);

        bind_polynomial(B, r_j, num_threads);

        Fr one_minus_r = Fr(1) - r_j;
        int old_size = 1 << round;
#pragma omp parallel for num_threads(num_threads) if (old_size >= 512)
        for (int i = 0; i < old_size; i++)
        {
            Fr left = F_table[i] * one_minus_r;
            Fr right = F_table[i] * r_j;
            F_table[i] = left;
            F_table[i + old_size] = right;
        }
    }

    Fr eq_r_r = B[0];

    std::vector<Fr> H(T);
#pragma omp parallel for num_threads(num_threads) if (T >= 1024)
    for (int j = 0; j < T; j++)
    {
        H[j] = F_table[read_addresses[j]];
    }

    for (int round = 0; round < log_T; round++)
    {
        int len = H.size();
        int half_len = len / 2;

        int num_actual_threads_t = std::min(num_threads, half_len);
        std::fill_n(local_eval_0.begin(), num_actual_threads_t, Fr(0));
        std::fill_n(local_eval_2.begin(), num_actual_threads_t, Fr(0));
        std::fill_n(local_eval_3.begin(), num_actual_threads_t, Fr(0));

#pragma omp parallel num_threads(num_actual_threads_t) if (half_len >= 512)
        {
            int tid = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < half_len; i++)
            {
                Fr D_0 = D[2 * i];
                Fr D_1 = D[2 * i + 1];
                Fr H_0 = H[2 * i];
                Fr H_1 = H[2 * i + 1];

                local_eval_0[tid] += D_0 * (H_0 * H_0 - H_0);

                Fr H_2 = H_1 + H_1 - H_0;
                Fr D_2 = D_1 + D_1 - D_0;
                local_eval_2[tid] += D_2 * (H_2 * H_2 - H_2);

                Fr H_3 = H_2 + H_1 - H_0;
                Fr D_3 = D_2 + D_1 - D_0;
                local_eval_3[tid] += D_3 * (H_3 * H_3 - H_3);
            }
        }

        Fr eval_0 = Fr(0), eval_2 = Fr(0), eval_3 = Fr(0);
        for (int t = 0; t < num_actual_threads_t; t++)
        {
            eval_0 += local_eval_0[t];
            eval_2 += local_eval_2[t];
            eval_3 += local_eval_3[t];
        }

        eval_0 *= eq_r_r;
        eval_2 *= eq_r_r;
        eval_3 *= eq_r_r;

        compressed_polys.push_back(eval_0);
        compressed_polys.push_back(eval_2);
        compressed_polys.push_back(eval_3);

        transcript.write_fr(eval_0);
        transcript.write_fr(eval_2);
        transcript.write_fr(eval_3);
        const Fr r_j = transcript.challenge_not_one();
        r_address_prime.push_back(r_j);

        bind_polynomial(H, r_j, num_threads);
        bind_polynomial(D, r_j, num_threads);
    }

    ra_claim_prime = H[0];
}

ShoutProof ShoutProver::prove_prepared_values(
    const std::vector<Fr> &lookup_table,
    const std::vector<int> &read_addresses,
    const std::vector<Fr> &read_values,
    int K,
    int T,
    int num_threads,
    ShoutProveTiming *timing,
    G1WireEncoding encoding,
    bool defer_read_value_opening,
    bool retain_wire_bytes,
    FiatShamirTranscript *shared_transcript,
    const std::vector<G1> *precommitted_read_values)
{
    if (!valid_shout_dimensions(K, T))
    {
        throw std::invalid_argument("Shout requires positive power-of-two K and T");
    }
    if (lookup_table.size() != static_cast<size_t>(K))
    {
        throw std::invalid_argument("Shout lookup table size must equal K");
    }
    if (read_addresses.size() != static_cast<size_t>(T))
    {
        throw std::invalid_argument("Shout read-address count must equal T");
    }
    if (read_values.size() != static_cast<std::size_t>(T))
    {
        throw std::invalid_argument(
            "prepared Shout read-value count must equal T");
    }
    const int log_T = log2_power_of_two(T);
    const int read_value_rows = 1 << (log_T / 2);
    if (precommitted_read_values != nullptr &&
        precommitted_read_values->size() != static_cast<std::size_t>(read_value_rows))
    {
        throw std::invalid_argument(
            "precommitted Shout read-value commitment has the wrong shape");
    }
    ShoutProof proof;
    ShoutProveTiming *timing_out = timing;
    auto total_start = std::chrono::high_resolution_clock::time_point{};
    if (timing_out)
    {
        timing_out->reset();
        total_start = std::chrono::high_resolution_clock::now();
    }
    auto elapsed_ms = [](const std::chrono::high_resolution_clock::time_point &start)
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count() /
               1000.0;
    };
    std::chrono::high_resolution_clock::time_point stage_start;
    if (num_threads < 1)
    {
        num_threads = 1;
    }
    int max_threads = T;
    if (max_threads < 1)
    {
        max_threads = 1;
    }
    if (num_threads > max_threads)
    {
        num_threads = max_threads;
    }

    int log_K = log2_power_of_two(K);

    if (timing_out)
    {
        stage_start = std::chrono::high_resolution_clock::now();
    }
    const int log_KT = log_K + log_T;
    G1 *RA_comm = hyrax::commit_onehot(
        read_addresses.data(), T, K, g, num_threads);
    if (timing_out)
    {
        timing_out->commit_ra_ms = elapsed_ms(stage_start);
    }
    proof.RA_comm = RA_comm;
    proof.RA_comm_size = 1 << (log_KT / 2);

    if (precommitted_read_values != nullptr)
    {
        proof.read_value_commitments = *precommitted_read_values;
    }
    else
    {
        if (timing_out)
        {
            stage_start = std::chrono::high_resolution_clock::now();
        }
        G1 *read_value_commitment = hyrax::commit(
            read_values.data(), g, log_T, num_threads);
        proof.read_value_commitments.assign(
            read_value_commitment, read_value_commitment + read_value_rows);
        delete[] read_value_commitment;
        if (timing_out)
        {
            timing_out->commit_read_values_ms = elapsed_ms(stage_start);
        }
    }

    if (timing_out)
    {
        stage_start = std::chrono::high_resolution_clock::now();
    }
    normalize_g1_array_in_place(
        proof.RA_comm, static_cast<std::size_t>(proof.RA_comm_size));
    if (timing_out)
    {
        timing_out->finalize_ra_ms = elapsed_ms(stage_start);
        stage_start = std::chrono::high_resolution_clock::now();
    }
    normalize_g1_array_in_place(
        proof.read_value_commitments.data(),
        proof.read_value_commitments.size());
    if (timing_out)
    {
        timing_out->finalize_read_values_ms = elapsed_ms(stage_start);
    }

    FiatShamirProofWriter transcript(
        encoding, retain_wire_bytes,
        shared_transcript);
    append_shout_statement(
        transcript.transcript(), lookup_table, K, T,
        defer_read_value_opening);
    append_shout_commitments(
        transcript, proof);
    const std::vector<Fr> r_cycle = derive_shout_point(
        transcript, log_T, true);

    if (timing_out)
    {
        stage_start = std::chrono::high_resolution_clock::now();
    }
    std::vector<Fr> E = mle::equality_weights(r_cycle.data(), log2_power_of_two(T), num_threads);
    if (timing_out)
    {
        timing_out->eq_ms += elapsed_ms(stage_start);
    }

    if (timing_out)
    {
        stage_start = std::chrono::high_resolution_clock::now();
    }
    std::vector<Fr> F = compute_read_address_poly(
        read_addresses, E, K, num_threads);
    Fr rv_claim = Fr(0);
    for (int k = 0; k < K; ++k)
    {
        rv_claim += F[static_cast<std::size_t>(k)] * lookup_table[static_cast<std::size_t>(k)];
    }
    if (timing_out)
    {
        timing_out->read_poly_ms += elapsed_ms(stage_start);
    }
    proof.rv_claim = rv_claim;
    transcript.write_fr(proof.rv_claim);
    if (timing_out)
    {
        stage_start = std::chrono::high_resolution_clock::now();
    }
    if (!defer_read_value_opening)
    {
        proof.read_value_opening = hyrax::open(
            read_values.data(), r_cycle.data(), log_T, pk_);
        normalize_opening_in_place(proof.read_value_opening);
        append_shout_opening(
            transcript, proof.read_value_opening);
    }
    if (timing_out)
    {
        timing_out->open_read_value_ms = elapsed_ms(stage_start);
    }

    const Fr z = transcript.challenge();

    std::vector<Fr> val_poly = lookup_table;
    std::vector<Fr> r_address;
    std::vector<Fr> core_compressed_polys;
    r_address.reserve(log_K);
    core_compressed_polys.reserve(log_K * 2);

    if (timing_out)
    {
        stage_start = std::chrono::high_resolution_clock::now();
    }
    prove_core_piop_hamming(
        F, val_poly, z, rv_claim, r_address,
        core_compressed_polys, log_K, num_threads, transcript);
    if (timing_out)
    {
        timing_out->core_sumcheck_ms = elapsed_ms(stage_start);
    }

    proof.ra_claim = F[0];
    proof.core_sumcheck_polys = std::move(core_compressed_polys);

    if (timing_out)
    {
        stage_start = std::chrono::high_resolution_clock::now();
    }
    std::vector<Fr> r_address_full;
    r_address_full.reserve(log_K + log_T);
    r_address_full.insert(r_address_full.end(), r_address.begin(), r_address.end());
    r_address_full.insert(r_address_full.end(), r_cycle.begin(), r_cycle.end());
    hyrax::OneHotOpeningWorkspace onehot_opening_workspace;
    proof.RA_open_proof_core = hyrax::open_onehot(
        read_addresses.data(), T, K, r_address_full.data(), pk_, num_threads,
        &onehot_opening_workspace);
    normalize_opening_in_place(proof.RA_open_proof_core);
    transcript.write_fr(proof.ra_claim);
    append_shout_opening(
        transcript, proof.RA_open_proof_core);
    if (timing_out)
    {
        timing_out->open_ra_ms += elapsed_ms(stage_start);
    }

    std::vector<Fr> booleanity_compressed_polys;
    std::vector<Fr> r_address_prime;
    Fr ra_claim_prime;
    booleanity_compressed_polys.reserve((log_K + log_T) * 3);
    r_address_prime.reserve(log_K + log_T);

    const std::vector<Fr> r_cycle_bool = derive_shout_point(
        transcript, log_T, false);
    const std::vector<Fr> r_address_bool = derive_shout_point(
        transcript, log_K, false);

    if (timing_out)
    {
        stage_start = std::chrono::high_resolution_clock::now();
    }
    std::vector<Fr> D = mle::equality_weights(r_cycle_bool.data(), log2_power_of_two(T), num_threads);
    if (timing_out)
    {
        timing_out->eq_ms += elapsed_ms(stage_start);
    }

    if (timing_out)
    {
        stage_start = std::chrono::high_resolution_clock::now();
    }
    std::vector<Fr> G = compute_read_address_poly(read_addresses, D, K, num_threads);
    if (timing_out)
    {
        timing_out->read_poly_ms += elapsed_ms(stage_start);
    }

    if (timing_out)
    {
        stage_start = std::chrono::high_resolution_clock::now();
    }
    prove_booleanity(
        read_addresses, r_address_bool, D, G, ra_claim_prime,
        r_address_prime, booleanity_compressed_polys, K, T, num_threads,
        transcript);
    if (timing_out)
    {
        timing_out->booleanity_sumcheck_ms = elapsed_ms(stage_start);
    }

    proof.booleanity_sumcheck_polys = std::move(booleanity_compressed_polys);
    proof.RA_eval = ra_claim_prime;

    if (timing_out)
    {
        stage_start = std::chrono::high_resolution_clock::now();
    }
    proof.RA_open_proof = hyrax::open_onehot(
        read_addresses.data(), T, K, r_address_prime.data(), pk_, num_threads,
        &onehot_opening_workspace);
    normalize_opening_in_place(proof.RA_open_proof);
    transcript.write_fr(proof.RA_eval);
    append_shout_opening(
        transcript, proof.RA_open_proof);
    if (timing_out)
    {
        timing_out->open_ra_ms += elapsed_ms(stage_start);
        timing_out->total_ms = elapsed_ms(total_start);
    }

    proof.g1_encoding = encoding;
    proof.wire_bytes = transcript.take_bytes();
    return proof;
}

namespace
{

    struct ParsedShoutWire
    {
        ShoutProof proof;
        std::vector<G1> ra_commitments;
        ShoutVerifierChallenges challenges;
    };

    bool parse_shout_wire(
        const std::vector<std::uint8_t> &wire_bytes,
        const std::vector<Fr> &lookup_table,
        int K,
        int T,
        bool defer_read_value_opening,
        ParsedShoutWire &parsed,
        FiatShamirTranscript *shared_transcript = nullptr)
    {
        if (!valid_shout_dimensions(K, T) || lookup_table.size() != static_cast<std::size_t>(K))
        {
            return false;
        }
        try
        {
            const int log_K = log2_power_of_two(K);
            const int log_T = log2_power_of_two(T);
            const int log_KT = log_K + log_T;
            const std::size_t ra_rows = std::size_t(1) << (log_KT / 2);
            const std::size_t value_rows = std::size_t(1) << (log_T / 2);
            const std::size_t value_opening_size =
                static_cast<std::size_t>(log_T - log_T / 2);
            const std::size_t ra_opening_size =
                static_cast<std::size_t>(log_KT - log_KT / 2);

            FiatShamirProofReader reader(wire_bytes, shared_transcript);
            append_shout_statement(
                reader.transcript(), lookup_table, K, T,
                defer_read_value_opening);

            parsed.ra_commitments = reader.read_g1_vector(ra_rows);
            parsed.proof.RA_comm = parsed.ra_commitments.data();
            parsed.proof.RA_comm_size = static_cast<int>(ra_rows);
            parsed.proof.read_value_commitments = reader.read_g1_vector(value_rows);

            parsed.challenges.r_cycle = derive_shout_point(
                reader, log_T, true);
            parsed.proof.rv_claim = reader.read_fr();
            if (!defer_read_value_opening)
            {
                parsed.proof.read_value_opening.eval_proof.q_comm =
                    reader.read_g1_vector(value_opening_size);
            }
            parsed.challenges.z = reader.challenge();

            parsed.proof.core_sumcheck_polys.clear();
            parsed.proof.core_sumcheck_polys.reserve(
                static_cast<std::size_t>(2 * log_K));
            parsed.challenges.core_round_challenges.clear();
            parsed.challenges.core_round_challenges.reserve(log_K);
            for (int round = 0; round < log_K; ++round)
            {
                parsed.proof.core_sumcheck_polys.push_back(
                    reader.read_fr());
                parsed.proof.core_sumcheck_polys.push_back(
                    reader.read_fr());
                parsed.challenges.core_round_challenges.push_back(
                    reader.challenge_not_one());
            }
            parsed.proof.ra_claim = reader.read_fr();
            parsed.proof.RA_open_proof_core.eval_proof.q_comm =
                reader.read_g1_vector(ra_opening_size);

            parsed.challenges.r_cycle_bool = derive_shout_point(
                reader, log_T, false);
            parsed.challenges.r_address_bool = derive_shout_point(
                reader, log_K, false);

            parsed.proof.booleanity_sumcheck_polys.clear();
            parsed.proof.booleanity_sumcheck_polys.reserve(
                static_cast<std::size_t>(3 * (log_K + log_T)));
            parsed.challenges.booleanity_round_challenges.clear();
            parsed.challenges.booleanity_round_challenges.reserve(log_K + log_T);
            for (int round = 0; round < log_K; ++round)
            {
                parsed.proof.booleanity_sumcheck_polys.push_back(
                    reader.read_fr());
                parsed.proof.booleanity_sumcheck_polys.push_back(
                    reader.read_fr());
                parsed.proof.booleanity_sumcheck_polys.push_back(
                    reader.read_fr());
                parsed.challenges.booleanity_round_challenges.push_back(
                    reader.challenge_not_one());
            }
            for (int round = 0; round < log_T; ++round)
            {
                parsed.proof.booleanity_sumcheck_polys.push_back(
                    reader.read_fr());
                parsed.proof.booleanity_sumcheck_polys.push_back(
                    reader.read_fr());
                parsed.proof.booleanity_sumcheck_polys.push_back(
                    reader.read_fr());
                parsed.challenges.booleanity_round_challenges.push_back(
                    reader.challenge_not_one());
            }
            parsed.proof.RA_eval = reader.read_fr();
            parsed.proof.RA_open_proof.eval_proof.q_comm =
                reader.read_g1_vector(ra_opening_size);
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

    bool replay_shout_proof_object(
        const ShoutProof &proof,
        const std::vector<Fr> &lookup_table,
        int K,
        int T,
        bool defer_read_value_opening,
        ShoutVerifierChallenges &challenges,
        FiatShamirTranscript *shared_transcript = nullptr)
    {
        if (!valid_shout_dimensions(K, T) ||
            lookup_table.size() != static_cast<std::size_t>(K))
        {
            return false;
        }
        try
        {
            const int log_K = log2_power_of_two(K);
            const int log_T = log2_power_of_two(T);
            const int log_KT = log_K + log_T;
            const std::size_t ra_rows = std::size_t(1) << (log_KT / 2);
            const std::size_t value_rows = std::size_t(1) << (log_T / 2);
            const std::size_t value_opening_size =
                static_cast<std::size_t>(log_T - log_T / 2);
            const std::size_t ra_opening_size =
                static_cast<std::size_t>(log_KT - log_KT / 2);
            const std::uint64_t encoding =
                static_cast<std::uint64_t>(proof.g1_encoding);
            if (encoding >
                    static_cast<std::uint64_t>(G1WireEncoding::Compressed) ||
                proof.RA_comm == nullptr ||
                proof.RA_comm_size != static_cast<int>(ra_rows) ||
                proof.read_value_commitments.size() != value_rows ||
                proof.core_sumcheck_polys.size() !=
                    static_cast<std::size_t>(2 * log_K) ||
                proof.booleanity_sumcheck_polys.size() !=
                    static_cast<std::size_t>(3 * log_KT) ||
                proof.RA_open_proof_core.eval_proof.q_comm.size() !=
                    ra_opening_size ||
                proof.RA_open_proof.eval_proof.q_comm.size() !=
                    ra_opening_size ||
                (!defer_read_value_opening &&
                 proof.read_value_opening.eval_proof.q_comm.size() !=
                     value_opening_size) ||
                (defer_read_value_opening &&
                 !proof.read_value_opening.eval_proof.q_comm.empty()))
            {
                return false;
            }

            FiatShamirProofWriter transcript(
                proof.g1_encoding, false, shared_transcript);
            append_shout_statement(
                transcript.transcript(), lookup_table, K, T,
                defer_read_value_opening);
            append_shout_commitments(
                transcript, proof);

            challenges.r_cycle = derive_shout_point(
                transcript, log_T, true);
            transcript.write_fr(proof.rv_claim);
            if (!defer_read_value_opening)
            {
                append_shout_opening(
                    transcript, proof.read_value_opening);
            }
            challenges.z = transcript.challenge();

            challenges.core_round_challenges.clear();
            challenges.core_round_challenges.reserve(log_K);
            for (int round = 0; round < log_K; ++round)
            {
                transcript.write_fr(proof.core_sumcheck_polys[static_cast<std::size_t>(2 * round)]);
                transcript.write_fr(proof.core_sumcheck_polys[static_cast<std::size_t>(2 * round + 1)]);
                challenges.core_round_challenges.push_back(
                    transcript.challenge_not_one());
            }
            transcript.write_fr(proof.ra_claim);
            append_shout_opening(
                transcript, proof.RA_open_proof_core);

            challenges.r_cycle_bool = derive_shout_point(
                transcript, log_T, false);
            challenges.r_address_bool = derive_shout_point(
                transcript, log_K, false);

            challenges.booleanity_round_challenges.clear();
            challenges.booleanity_round_challenges.reserve(log_KT);
            std::size_t polynomial_index = 0;
            for (int round = 0; round < log_K; ++round)
            {
                transcript.write_fr(proof.booleanity_sumcheck_polys[polynomial_index++]);
                transcript.write_fr(proof.booleanity_sumcheck_polys[polynomial_index++]);
                transcript.write_fr(proof.booleanity_sumcheck_polys[polynomial_index++]);
                challenges.booleanity_round_challenges.push_back(
                    transcript.challenge_not_one());
            }
            for (int round = 0; round < log_T; ++round)
            {
                transcript.write_fr(proof.booleanity_sumcheck_polys[polynomial_index++]);
                transcript.write_fr(proof.booleanity_sumcheck_polys[polynomial_index++]);
                transcript.write_fr(proof.booleanity_sumcheck_polys[polynomial_index++]);
                challenges.booleanity_round_challenges.push_back(
                    transcript.challenge_not_one());
            }
            transcript.write_fr(proof.RA_eval);
            append_shout_opening(
                transcript, proof.RA_open_proof);
            return polynomial_index == proof.booleanity_sumcheck_polys.size();
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

}

bool verify_core_sumcheck_impl(
    const std::vector<Fr> &compressed_polys,
    Fr expected_claim,
    Fr z,
    const std::vector<Fr> &lookup_table,
    Fr ra_claim,
    int log_K,
    const std::vector<Fr> &round_challenges,
    std::vector<Fr> &r_address_out)
{
    if ((int)compressed_polys.size() != log_K * 2)
    {
        return false;
    }

    const Fr inv_two = Fr(1) / Fr(2);
    Fr previous_claim = expected_claim;
    r_address_out.clear();
    r_address_out.reserve(log_K);
    if (round_challenges.size() != static_cast<std::size_t>(log_K))
    {
        return false;
    }

    for (int round = 0; round < log_K; round++)
    {
        Fr eval_0 = compressed_polys[2 * round];
        Fr eval_2 = compressed_polys[2 * round + 1];
        Fr eval_1 = previous_claim - eval_0;
        const Fr r_j = round_challenges[static_cast<std::size_t>(round)];
        r_address_out.push_back(r_j);

        Fr term0 = eval_0 * (r_j - Fr(1)) * (r_j - Fr(2)) * inv_two;
        Fr term1 = eval_1 * r_j * (r_j - Fr(2)) * Fr(-1);
        Fr term2 = eval_2 * r_j * (r_j - Fr(1)) * inv_two;
        previous_claim = term0 + term1 + term2;
    }

    int K_size = lookup_table.size();
    std::vector<Fr> eq_evals(K_size, Fr(0));
    eq_evals[0] = Fr(1);
    int size = 1;
    for (int i = 0; i < log_K; i++)
    {
        Fr r_i = r_address_out[i];
        Fr one_minus_r = Fr(1) - r_i;
        for (int j = size - 1; j >= 0; j--)
        {
            eq_evals[j + size] = eq_evals[j] * r_i;
            eq_evals[j] = eq_evals[j] * one_minus_r;
        }
        size *= 2;
    }

    Fr val_at_r = Fr(0);
    for (int k = 0; k < K_size; k++)
    {
        val_at_r += lookup_table[k] * eq_evals[k];
    }

    Fr expected_final = ra_claim * (z + val_at_r);
    if (expected_final != previous_claim)
    {
        std::cerr << "[Verify] Core sumcheck final check failed" << std::endl;
        return false;
    }

    return true;
}

bool verify_booleanity_sumcheck_impl(
    const std::vector<Fr> &compressed_polys,
    const std::vector<Fr> &r_address,
    const std::vector<Fr> &r_cycle,
    Fr RA_eval,
    int log_K,
    int log_T,
    const std::vector<Fr> &round_challenges,
    std::vector<Fr> &r_address_prime_out)
{
    int total_rounds = log_K + log_T;
    if ((int)compressed_polys.size() != total_rounds * 3)
    {
        return false;
    }

    const Fr inv_six = Fr(1) / Fr(6);
    const Fr inv_two = Fr(3) * inv_six;
    Fr previous_claim = Fr(0);
    int poly_idx = 0;
    r_address_prime_out.clear();
    r_address_prime_out.reserve(total_rounds);
    if (round_challenges.size() != static_cast<std::size_t>(total_rounds))
    {
        return false;
    }

    for (int round = 0; round < log_K; round++)
    {
        Fr eval_0 = compressed_polys[poly_idx++];
        Fr eval_2 = compressed_polys[poly_idx++];
        Fr eval_3 = compressed_polys[poly_idx++];
        Fr eval_1 = previous_claim - eval_0;
        const Fr r_j = round_challenges[static_cast<std::size_t>(round)];
        r_address_prime_out.push_back(r_j);

        Fr term0 = eval_0 * (r_j - Fr(1)) * (r_j - Fr(2)) * (r_j - Fr(3)) * (-inv_six);
        Fr term1 = eval_1 * r_j * (r_j - Fr(2)) * (r_j - Fr(3)) * inv_two;
        Fr term2 = eval_2 * r_j * (r_j - Fr(1)) * (r_j - Fr(3)) * (-inv_two);
        Fr term3 = eval_3 * r_j * (r_j - Fr(1)) * (r_j - Fr(2)) * inv_six;
        previous_claim = term0 + term1 + term2 + term3;
    }

    Fr eq_r_address = Fr(1);
    for (int i = 0; i < log_K; i++)
    {
        Fr r_i = r_address[i];
        Fr r_prime_i = r_address_prime_out[i];
        eq_r_address *= (r_i * r_prime_i + (Fr(1) - r_i) * (Fr(1) - r_prime_i));
    }

    for (int round = 0; round < log_T; round++)
    {
        Fr eval_0 = compressed_polys[poly_idx++];
        Fr eval_2 = compressed_polys[poly_idx++];
        Fr eval_3 = compressed_polys[poly_idx++];
        Fr eval_1 = previous_claim - eval_0;
        const int round_index = log_K + round;
        const Fr r_j = round_challenges[static_cast<std::size_t>(round_index)];
        r_address_prime_out.push_back(r_j);

        Fr term0 = eval_0 * (r_j - Fr(1)) * (r_j - Fr(2)) * (r_j - Fr(3)) * (-inv_six);
        Fr term1 = eval_1 * r_j * (r_j - Fr(2)) * (r_j - Fr(3)) * inv_two;
        Fr term2 = eval_2 * r_j * (r_j - Fr(1)) * (r_j - Fr(3)) * (-inv_two);
        Fr term3 = eval_3 * r_j * (r_j - Fr(1)) * (r_j - Fr(2)) * inv_six;
        previous_claim = term0 + term1 + term2 + term3;
    }

    Fr eq_r_cycle = Fr(1);
    for (int i = 0; i < log_T; i++)
    {
        Fr r_i = r_cycle[i];
        Fr r_prime_i = r_address_prime_out[log_K + i];
        eq_r_cycle *= (r_i * r_prime_i + (Fr(1) - r_i) * (Fr(1) - r_prime_i));
    }

    Fr expected_final = eq_r_address * eq_r_cycle * (RA_eval * RA_eval - RA_eval);

    if (expected_final != previous_claim)
    {
        return false;
    }

    return true;
}

bool ShoutVerifier::verify_serialized(
    const std::vector<std::uint8_t> &wire_bytes,
    const std::vector<Fr> &lookup_table,
    int K,
    int T,
    std::vector<G1> *parsed_read_value_commitments,
    LookupReadClaim *read_claim,
    bool defer_read_value_opening,
    hyrax::OpeningBatchVerifier *opening_batch,
    FiatShamirTranscript *shared_transcript)
{
    const auto transcript_start = std::chrono::high_resolution_clock::now();
    ParsedShoutWire parsed;
    if (!parse_shout_wire(
            wire_bytes, lookup_table, K, T,
            defer_read_value_opening, parsed,
            shared_transcript))
    {
        return false;
    }
    const long long transcript_time_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - transcript_start)
            .count();
    if (!verify_proof_object_impl(
            parsed.proof, parsed.challenges, lookup_table, K, T,
            read_claim, defer_read_value_opening, opening_batch,
            transcript_time_us))
    {
        return false;
    }
    if (parsed_read_value_commitments != nullptr)
    {
        *parsed_read_value_commitments = parsed.proof.read_value_commitments;
    }
    return true;
}

bool ShoutVerifier::verify_in_memory(
    const ShoutProof &proof,
    const std::vector<Fr> &lookup_table,
    int K,
    int T,
    LookupReadClaim *read_claim,
    bool defer_read_value_opening,
    hyrax::OpeningBatchVerifier *opening_batch,
    FiatShamirTranscript *shared_transcript)
{
    const auto transcript_start = std::chrono::high_resolution_clock::now();
    ShoutVerifierChallenges challenges;
    if (!replay_shout_proof_object(
            proof, lookup_table, K, T,
            defer_read_value_opening, challenges,
            shared_transcript))
    {
        return false;
    }
    const long long transcript_time_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - transcript_start)
            .count();
    return verify_proof_object_impl(
        proof, challenges, lookup_table, K, T, read_claim,
        defer_read_value_opening, opening_batch, transcript_time_us);
}

bool ShoutVerifier::verify_proof_object_impl(
    const ShoutProof &proof,
    const ShoutVerifierChallenges &challenges,
    const std::vector<Fr> &lookup_table,
    int K,
    int T,
    LookupReadClaim *read_claim,
    bool defer_read_value_opening,
    hyrax::OpeningBatchVerifier *opening_batch,
    long long transcript_time_us)
{
    if (!valid_shout_dimensions(K, T) ||
        lookup_table.size() != static_cast<std::size_t>(K))
    {
        return false;
    }
    const bool timing_enabled =
        std::getenv("SHOUT_VERIFY_TIMING") != nullptr;
    const auto algebraic_start = std::chrono::high_resolution_clock::now();
    auto stage_start = algebraic_start;

    const int log_K = log2_power_of_two(K);
    const int log_T = log2_power_of_two(T);
    const int log_KT = log_K + log_T;
    const int expected_ra_comm_size = 1 << (log_KT / 2);
    const int expected_read_value_rows = 1 << (log_T / 2);
    if (proof.RA_comm == nullptr ||
        proof.RA_comm_size != expected_ra_comm_size ||
        proof.read_value_commitments.size() !=
            static_cast<std::size_t>(expected_read_value_rows) ||
        proof.core_sumcheck_polys.size() !=
            static_cast<std::size_t>(2 * log_K) ||
        proof.booleanity_sumcheck_polys.size() !=
            static_cast<std::size_t>(3 * log_KT) ||
        challenges.r_cycle.size() != static_cast<std::size_t>(log_T) ||
        challenges.r_cycle_bool.size() != static_cast<std::size_t>(log_T) ||
        challenges.r_address_bool.size() != static_cast<std::size_t>(log_K) ||
        challenges.core_round_challenges.size() !=
            static_cast<std::size_t>(log_K) ||
        challenges.booleanity_round_challenges.size() !=
            static_cast<std::size_t>(log_KT))
    {
        return false;
    }

    const std::vector<Fr> &r_cycle = challenges.r_cycle;
    if (!defer_read_value_opening && opening_batch == nullptr &&
        !hyrax::verify(
            proof.read_value_opening, r_cycle.data(), proof.rv_claim,
            proof.read_value_commitments.data(), log_T, vk_))
    {
        return false;
    }
    const auto read_open_end = std::chrono::high_resolution_clock::now();
    const long long read_open_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            read_open_end - stage_start)
            .count();
    stage_start = read_open_end;

    std::vector<Fr> r_address;
    if (!verify_core_sumcheck_impl(
            proof.core_sumcheck_polys,
            proof.rv_claim + challenges.z,
            challenges.z,
            lookup_table,
            proof.ra_claim,
            log_K,
            challenges.core_round_challenges,
            r_address))
    {
        return false;
    }
    const auto core_end = std::chrono::high_resolution_clock::now();
    const long long core_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            core_end - stage_start)
            .count();
    stage_start = core_end;

    std::vector<Fr> r_address_full;
    r_address_full.reserve(log_KT);
    r_address_full.insert(
        r_address_full.end(), r_address.begin(), r_address.end());
    r_address_full.insert(
        r_address_full.end(), r_cycle.begin(), r_cycle.end());
    if (opening_batch == nullptr && !hyrax::verify(
                                        proof.RA_open_proof_core, r_address_full.data(), proof.ra_claim,
                                        proof.RA_comm, log_KT, vk_))
    {
        return false;
    }
    const auto core_open_end = std::chrono::high_resolution_clock::now();
    const long long core_open_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            core_open_end - stage_start)
            .count();
    stage_start = core_open_end;

    std::vector<Fr> r_address_prime;
    if (!verify_booleanity_sumcheck_impl(
            proof.booleanity_sumcheck_polys,
            challenges.r_address_bool,
            challenges.r_cycle_bool,
            proof.RA_eval,
            log_K,
            log_T,
            challenges.booleanity_round_challenges,
            r_address_prime))
    {
        return false;
    }
    const auto booleanity_end = std::chrono::high_resolution_clock::now();
    const long long booleanity_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            booleanity_end - stage_start)
            .count();
    stage_start = booleanity_end;

    if (r_address_prime.size() != static_cast<std::size_t>(log_KT))
    {
        return false;
    }
    if (opening_batch == nullptr && !hyrax::verify(
                                        proof.RA_open_proof, r_address_prime.data(), proof.RA_eval,
                                        proof.RA_comm, log_KT, vk_))
    {
        return false;
    }
    if (opening_batch != nullptr)
    {
        if ((!defer_read_value_opening && !opening_batch->append(
                                              proof.read_value_opening,
                                              r_cycle.data(), proof.rv_claim,
                                              proof.read_value_commitments.data(), log_T)) ||
            !opening_batch->append(
                proof.RA_open_proof_core,
                r_address_full.data(), proof.ra_claim,
                proof.RA_comm, log_KT) ||
            !opening_batch->append(
                proof.RA_open_proof,
                r_address_prime.data(), proof.RA_eval,
                proof.RA_comm, log_KT))
        {
            return false;
        }
    }
    const auto final_open_end = std::chrono::high_resolution_clock::now();
    const long long final_open_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            final_open_end - stage_start)
            .count();
    const long long algebraic_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            final_open_end - algebraic_start)
            .count();

    if (timing_enabled)
    {
        std::cout << "[Shout::verify] K=" << K << " T=" << T
                  << " logK=" << log_K << " logT=" << log_T << "\n";
        std::cout << "[Shout::verify] transcript+challenge: "
                  << transcript_time_us << " us\n";
        std::cout << "[Shout::verify] read_value_open: "
                  << read_open_us << " us\n";
        std::cout << "[Shout::verify] core_sumcheck: "
                  << core_us << " us\n";
        std::cout << "[Shout::verify] ra_open_core: "
                  << core_open_us << " us\n";
        std::cout << "[Shout::verify] booleanity_sumcheck: "
                  << booleanity_us << " us\n";
        std::cout << "[Shout::verify] ra_open_verify: "
                  << final_open_us << " us\n";
        std::cout << "[Shout::verify] total: "
                  << (transcript_time_us + algebraic_us) << " us\n";
    }

    if (read_claim != nullptr)
    {
        read_claim->commitments = proof.read_value_commitments;
        read_claim->point = r_cycle;
        read_claim->evaluation = proof.rv_claim;
    }
    return true;
}

u64 shout_proof_size_bytes(const ShoutProof &proof)
{
    const size_t groups = static_cast<size_t>(proof.RA_comm_size) + proof.read_value_commitments.size() + proof.read_value_opening.eval_proof.q_comm.size() + proof.RA_open_proof_core.eval_proof.q_comm.size() + proof.RA_open_proof.eval_proof.q_comm.size();
    const size_t fields = proof.core_sumcheck_polys.size() + proof.booleanity_sumcheck_polys.size() + 3;
    return static_cast<u64>(groups * g1_wire_size(G1WireEncoding::Compressed) + fields * 32);
}
