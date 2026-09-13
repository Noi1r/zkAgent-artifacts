#include "gkr/prover.hpp"
#include "common/field_integer.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <limits>
#include <omp.h>
#include "common/timer.hpp"
#include <common/utils.hpp>

static vector<F> beta_u;
using namespace mcl::bn;
using zkagent_integer::fr_from_i128;

namespace
{

    void add_scaled_field(F &accumulator, const F &value, ll scale)
    {
        if (scale == 1)
        {
            accumulator += value;
        }
        else if (scale == -1)
        {
            accumulator -= value;
        }
        else if (scale != 0)
        {
            accumulator += value * fr_from_i128(scale);
        }
    }

    struct InputScatterWorkspace
    {
        std::vector<u32> order;
        std::vector<std::size_t> cursors;
        std::vector<std::size_t> offsets;
        std::vector<std::vector<u32>> newly_active;
    };

    bool scatter_input_route(
        const std::vector<u32> &mapping, u32 count, const std::vector<F> &beta,
        std::vector<F> &weights, std::vector<std::uint64_t> &active_bits,
        std::vector<u32> &active_indices, int threads, InputScatterWorkspace &scratch)
    {
        if (threads <= 1 || count < (1U << 15))
            return false;
        if (count > mapping.size() || count > beta.size())
            throw std::invalid_argument("invalid input coefficient route");
        u32 lo = std::numeric_limits<u32>::max(), hi = 0;
#pragma omp parallel for reduction(min : lo) reduction(max : hi) num_threads(threads)
        for (u32 source = 0; source < count; ++source)
        {
            lo = std::min(lo, mapping[source]);
            hi = std::max(hi, mapping[source]);
        }
        if (hi >= weights.size())
            throw std::invalid_argument("input coefficient destination outside domain");
        const std::size_t first_word = (std::size_t(lo) >> 6) & ~std::size_t(7);
        const std::size_t word_count = (std::size_t(hi) >> 6) - first_word + 1;
        const std::size_t desired_buckets = std::size_t(threads) * 4;
        const std::size_t words_per_bucket =
            ((word_count + desired_buckets - 1) / desired_buckets + 7) & ~std::size_t(7);
        const std::size_t buckets = (word_count + words_per_bucket - 1) / words_per_bucket;
        if (buckets < 2)
            return false;
        scratch.order.resize(count);
        scratch.cursors.assign(std::size_t(threads) * buckets, 0);
        scratch.offsets.resize(buckets + 1);
        scratch.newly_active.resize(buckets);
        const auto bucket_for = [&](u32 target)
        {
            return ((std::size_t(target) >> 6) - first_word) / words_per_bucket;
        };
#pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num();
            const int team_size = omp_get_num_threads();
            const std::size_t start = std::size_t(count) * tid / team_size;
            const std::size_t end = std::size_t(count) * (tid + 1) / team_size;
            std::size_t *cursors = scratch.cursors.data() + std::size_t(tid) * buckets;
            for (std::size_t source = start; source < end; ++source)
                ++cursors[bucket_for(mapping[source])];
#pragma omp barrier
#pragma omp single
            {
                std::size_t offset = 0;
                for (std::size_t bucket = 0; bucket < buckets; ++bucket)
                {
                    scratch.offsets[bucket] = offset;
                    for (int worker = 0; worker < team_size; ++worker)
                    {
                        auto &cursor = scratch.cursors[std::size_t(worker) * buckets + bucket];
                        const std::size_t size = cursor;
                        cursor = offset;
                        offset += size;
                    }
                }
                scratch.offsets[buckets] = offset;
            }
            for (std::size_t source = start; source < end; ++source)
                scratch.order[cursors[bucket_for(mapping[source])]++] = static_cast<u32>(source);
#pragma omp barrier
#pragma omp for schedule(dynamic, 1)
            for (std::size_t bucket = 0; bucket < buckets; ++bucket)
            {
                auto &newly_active = scratch.newly_active[bucket];
                newly_active.clear();
                for (std::size_t position = scratch.offsets[bucket];
                     position < scratch.offsets[bucket + 1]; ++position)
                {
                    const u32 source = scratch.order[position];
                    const u32 target = mapping[source];
                    weights[target] += beta[source];
                    auto &word = active_bits[std::size_t(target) >> 6];
                    const std::uint64_t mask = std::uint64_t(1) << (target & 63U);
                    if ((word & mask) == 0)
                    {
                        word |= mask;
                        newly_active.push_back(target);
                    }
                }
            }
        }
        for (const auto &newly_active : scratch.newly_active)
            active_indices.insert(active_indices.end(), newly_active.begin(), newly_active.end());
        return true;
    }

}

linear_poly interpolate(const F &zero_v, const F &one_v)
{
    return {one_v - zero_v, zero_v};
}

F prover::getCirValue(int layer_id, const vector<u32> &ori, u32 u)
{
    return !layer_id ? val[0][ori[u]] : val[layer_id][u];
}

void prover::init()
{
    proof_size = 0;
    sumcheck_init_timing = SumcheckInitTiming{};
    r_u.resize(C.size + 1);
    r_v.resize(C.size + 1);

    int max_bl = 0;
    for (int i = 0; i < C.size; ++i)
    {
        auto &layer = C.circuit[i];
        max_bl = std::max(max_bl, (int)layer.max_bl_u);
        max_bl = std::max(max_bl, (int)layer.max_bl_v);
        max_bl = std::max(max_bl, (int)layer.bit_length);
    }

    const int INITIAL_SIZE = std::min(max_bl + 1, 20);
    const size_t initial_array_size = 1ULL << INITIAL_SIZE;

    V_mult[0].resize(initial_array_size);
    V_mult[1].resize(initial_array_size);
    mult_array[0].resize(initial_array_size);
    mult_array[1].resize(initial_array_size);
    tmp_V_mult[0].resize(initial_array_size);
    tmp_V_mult[1].resize(initial_array_size);
    tmp_mult_array[0].resize(initial_array_size);
    tmp_mult_array[1].resize(initial_array_size);

    input_claim_weights.clear();
    input_claim_active_bits.clear();
    input_claim_active_indices.clear();
    input_claim_workspace_initialized = false;
}

GkrWorkspacePreparation prover::prepareGkrWorkspace()
{
    GkrWorkspacePreparation result;
    auto domain = [](int bits) -> std::size_t
    {
        if (bits < 0)
            return 0;
        if (bits >= static_cast<int>(sizeof(std::size_t) * 8))
        {
            throw std::length_error("GKR workspace bit length is too large");
        }
        return std::size_t(1) << bits;
    };

    for (int layer_id = 1; layer_id < C.size; ++layer_id)
    {
        const layer &cur = C.circuit[layer_id];
        if (cur.ty == layerType::FCONN)
        {
            continue;
        }
        result.beta_g_domain = std::max(
            result.beta_g_domain, domain(cur.bit_length));
        result.beta_u_domain = std::max(
            result.beta_u_domain, domain(cur.max_bl_u));
        for (int branch = 0; branch < 2; ++branch)
        {
            result.branch_domain[branch] = std::max(
                result.branch_domain[branch],
                domain(cur.bit_length_u[branch]));
            if (cur.need_phase2)
            {
                result.branch_domain[branch] = std::max(
                    result.branch_domain[branch],
                    domain(cur.bit_length_v[branch]));
            }
        }
    }

    if (C.size > 0)
    {
        result.input_claim_domain = domain(C.circuit[0].bit_length);
    }

    for (int branch = 0; branch < 2; ++branch)
    {
        const std::size_t n = result.branch_domain[branch];
        V_mult[branch].resize(n);
        mult_array[branch].resize(n);
        tmp_V_mult[branch].resize(n);
        tmp_mult_array[branch].resize(n);
    }
    beta_g.resize(result.beta_g_domain);
    beta_u.resize(result.beta_u_domain);
    input_claim_weights.resize(result.input_claim_domain);
    const int input_threads = thread_num > 0 ? thread_num : 1;
#pragma omp parallel for schedule(static) num_threads(input_threads) if (result.input_claim_domain >= \
                                                                             static_cast<std::size_t>(input_threads) * 256)
    for (long long raw = 0;
         raw < static_cast<long long>(result.input_claim_domain); ++raw)
    {
        input_claim_weights[static_cast<std::size_t>(raw)].clear();
    }
    input_claim_active_bits.assign(
        (result.input_claim_domain + 63) / 64, 0);
    std::size_t input_route_upper_bound = 0;
    for (int layer_id = 1; layer_id < C.size; ++layer_id)
    {
        const layer &cur = C.circuit[layer_id];
        if (~cur.bit_length_u[0])
        {
            input_route_upper_bound += cur.size_u[0];
        }
        if (~cur.bit_length_v[0] && cur.need_phase2)
        {
            input_route_upper_bound += cur.size_v[0];
        }
    }
    input_claim_active_indices.reserve(std::min(
        result.input_claim_domain, input_route_upper_bound));
    input_claim_workspace_initialized = true;

    result.allocated_bytes =
        4 * (result.branch_domain[0] + result.branch_domain[1]) * sizeof(linear_poly) + (result.beta_g_domain + result.beta_u_domain + result.input_claim_domain) * sizeof(F) + input_claim_active_bits.size() * sizeof(std::uint64_t) + input_claim_active_indices.capacity() * sizeof(u32);
    return result;
}

void prover::sumcheckInitAll(const vector<F>::const_iterator &r_0_from_v)
{
    sumcheck_id = C.size;
    i8 last_bl = C.circuit[sumcheck_id - 1].bit_length;
    r_u[sumcheck_id].resize(last_bl);
    for (int i = 0; i < last_bl; ++i)
        r_u[sumcheck_id][i] = r_0_from_v[i];
}

void prover::sumcheckInit(const F &alpha_0, const F &beta_0,
                          const vector<F> *fresh_zero_root)
{
    alpha = alpha_0;
    beta = beta_0;
    if (fresh_zero_root != nullptr)
    {
        if (!requiresFreshZeroRoot(C, sumcheck_id - 1) ||
            fresh_zero_root->size() != static_cast<std::size_t>(
                                           C.circuit[sumcheck_id - 1].bit_length) ||
            alpha != F_ONE || beta != F_ZERO)
        {
            throw std::invalid_argument("invalid orphan zero-root initialization");
        }
        fresh_zero_root_point = *fresh_zero_root;
        r_0 = fresh_zero_root_point.begin();
        r_1 = fresh_zero_root_point.end();
    }
    else
    {
        r_0 = r_u[sumcheck_id].begin();
        r_1 = r_v[sumcheck_id].begin();
    }
    --sumcheck_id;
}
void prover::sumcheckInitPhase1(const F &relu_rou_0)
{
    timer total_timer;
    total_timer.start();
    auto &cur = C.circuit[sumcheck_id];

    timer workspace_timer;
    workspace_timer.start();
    total[0] = ~cur.bit_length_u[0] ? 1ULL << cur.bit_length_u[0] : 0;
    total_size[0] = cur.size_u[0];
    total[1] = ~cur.bit_length_u[1] ? 1ULL << cur.bit_length_u[1] : 0;
    total_size[1] = cur.size_u[1];

    const int thread_n = thread_num > 0 ? thread_num : 1;
    const u32 OmpThreshold = 1U << 14;
    r_u[sumcheck_id].resize(cur.max_bl_u);
    const std::size_t beta_g_required = 1ULL << cur.bit_length;
    if (beta_g.size() < beta_g_required)
    {
        beta_g.resize(beta_g_required);
    }
    relu_rou = relu_rou_0;
    add_term.clear();

    for (int b = 0; b < 2; ++b)
    {
        if (total[b] > mult_array[b].size())
        {
            mult_array[b].resize(total[b]);
            V_mult[b].resize(total[b]);
            tmp_mult_array[b].resize(total[b]);
            tmp_V_mult[b].resize(total[b]);
        }
    }
    workspace_timer.stop();

    timer witness_buffer_timer;
    witness_buffer_timer.start();
    for (int b = 0; b < 2; ++b)
    {
        const u32 total_b = total[b];
        if (!total_b)
            continue;
        const u32 size_b = cur.size_u[b];
        auto dep = !b ? 0 : sumcheck_id - 1;
        if (dep == 0)
        {
#pragma omp parallel for schedule(static) if (total_b >= OmpThreshold && thread_n > 1) num_threads(thread_n)
            for (u32 u = 0; u < total_b; ++u)
            {
                mult_array[b][u].clear();
                if (u < size_b)
                    V_mult[b][u] = val[0][cur.ori_id_u[u]];
                else
                    V_mult[b][u].clear();
            }
        }
        else
        {
#pragma omp parallel for schedule(static) if (total_b >= OmpThreshold && thread_n > 1) num_threads(thread_n)
            for (u32 u = 0; u < total_b; ++u)
            {
                mult_array[b][u].clear();
                if (u < size_b)
                    V_mult[b][u] = val[dep][u];
                else
                    V_mult[b][u].clear();
            }
        }
    }
    witness_buffer_timer.stop();

    timer beta_timer;
    beta_timer.start();
    initBetaTable(beta_g, cur.bit_length, r_0, r_1, alpha, beta, thread_n);
    beta_timer.stop();

    timer relu_scale_timer;
    relu_scale_timer.start();
    if (!relu_rou.isOne() && cur.zero_start_id < cur.size)
    {
        const u64 relu_start = cur.zero_start_id;
        const u64 relu_end = 1ULL << cur.bit_length;
#pragma omp parallel for schedule(static) if ((relu_end - relu_start) >= OmpThreshold && thread_n > 1) num_threads(thread_n)
        for (u64 g = relu_start; g < relu_end; ++g)
            beta_g[g] = beta_g[g] * relu_rou;
    }
    relu_scale_timer.stop();

    timer unary_gate_timer;
    unary_gate_timer.start();
    if (cur.uni_interval.size() >= 2 && thread_n > 1)
    {
        const int interval_count = static_cast<int>(cur.uni_interval.size());
        const int actual_threads = std::min(thread_n, interval_count);
#pragma omp parallel for schedule(static) num_threads(actual_threads)
        for (int j = 0; j < interval_count; ++j)
        {
            const int l = cur.uni_interval[j].first;
            const int r = cur.uni_interval[j].second;
            int i = l;
            while (i < r)
            {
                const auto &first = cur.uni_gates[i];
                const bool idx = first.lu != 0;
                const u32 target = first.u;
                F subtotal = F_ZERO;
                do
                {
                    const auto &gate = cur.uni_gates[i];
                    add_scaled_field(subtotal, beta_g[gate.g], gate.sc);
                    ++i;
                } while (i < r &&
                         (cur.uni_gates[i].lu != 0) == idx &&
                         cur.uni_gates[i].u == target);
                mult_array[idx][target].b += subtotal;
            }
        }
    }
    else
        for (auto &gate : cur.uni_gates)
        {
            bool idx = gate.lu != 0;
            add_scaled_field(
                mult_array[idx][gate.u].b, beta_g[gate.g], gate.sc);
        }
    unary_gate_timer.stop();

    timer binary_gate_timer;
    binary_gate_timer.start();
    const auto &bin_intervals_p1 = !cur.bin_interval_u.empty() ? cur.bin_interval_u : cur.bin_interval;
    if (bin_intervals_p1.size() >= 2 && thread_n > 1)
    {
        const int interval_count = static_cast<int>(bin_intervals_p1.size());
        const int actual_threads = std::min(thread_n, interval_count);
#pragma omp parallel for schedule(static) num_threads(actual_threads)
        for (int j = 0; j < interval_count; ++j)
        {
            const int l = bin_intervals_p1[j].first;
            const int r = bin_intervals_p1[j].second;
            int i = l;
            while (i < r)
            {
                const auto &first = cur.bin_gates[i];
                const bool idx =
                    first.getLayerIdU(sumcheck_id) != 0;
                const u32 target = first.u;
                F subtotal = F_ZERO;
                do
                {
                    const auto &gate = cur.bin_gates[i];
                    const auto layer_v =
                        gate.getLayerIdV(sumcheck_id);
                    const F &val_lv = !layer_v
                                          ? val[0][cur.ori_id_v[gate.v]]
                                          : val[layer_v][gate.v];
                    add_scaled_field(
                        subtotal, val_lv * beta_g[gate.g], gate.sc);
                    ++i;
                } while (i < r &&
                         (cur.bin_gates[i].getLayerIdU(sumcheck_id) != 0) == idx &&
                         cur.bin_gates[i].u == target);
                mult_array[idx][target].b += subtotal;
            }
        }
    }
    else
        for (auto &gate : cur.bin_gates)
        {
            bool idx = gate.getLayerIdU(sumcheck_id) != 0;
            auto val_lv = getCirValue(gate.getLayerIdV(sumcheck_id), cur.ori_id_v, gate.v);
            add_scaled_field(
                mult_array[idx][gate.u].b,
                val_lv * beta_g[gate.g], gate.sc);
        }
    binary_gate_timer.stop();

    round = 0;
    total_timer.stop();
    sumcheck_init_timing.phase1_total_seconds += total_timer.elapse_sec();
    sumcheck_init_timing.phase1_workspace_seconds +=
        workspace_timer.elapse_sec();
    sumcheck_init_timing.phase1_witness_buffer_seconds +=
        witness_buffer_timer.elapse_sec();
    sumcheck_init_timing.phase1_beta_seconds += beta_timer.elapse_sec();
    sumcheck_init_timing.phase1_relu_scale_seconds +=
        relu_scale_timer.elapse_sec();
    sumcheck_init_timing.phase1_unary_gate_seconds +=
        unary_gate_timer.elapse_sec();
    sumcheck_init_timing.phase1_binary_gate_seconds +=
        binary_gate_timer.elapse_sec();
    ++sumcheck_init_timing.phase1_calls;
}

void prover::sumcheckInitPhase2(
    const std::array<F, 2> *phase1_unary)
{
    timer total_timer;
    total_timer.start();
    auto &cur = C.circuit[sumcheck_id];
    timer workspace_timer;
    workspace_timer.start();
    total[0] = ~cur.bit_length_v[0] ? 1ULL << cur.bit_length_v[0] : 0;
    total_size[0] = cur.size_v[0];
    total[1] = ~cur.bit_length_v[1] ? 1ULL << cur.bit_length_v[1] : 0;
    total_size[1] = cur.size_v[1];

    const int thread_n = thread_num > 0 ? thread_num : 1;
    const u32 OmpThreshold = 1U << 14;
    r_v[sumcheck_id].resize(cur.max_bl_v);

    const std::size_t beta_u_required = 1ULL << cur.max_bl_u;
    if (beta_u.size() < beta_u_required)
    {
        beta_u.resize(beta_u_required);
    }

    add_term.clear();

    for (int b = 0; b < 2; ++b)
    {
        if (total[b] > mult_array[b].size())
        {
            mult_array[b].resize(total[b]);
            V_mult[b].resize(total[b]);
            tmp_mult_array[b].resize(total[b]);
            tmp_V_mult[b].resize(total[b]);
        }
    }
    workspace_timer.stop();

    timer beta_timer;
    beta_timer.start();
    initBetaTable(beta_u, cur.max_bl_u, r_u[sumcheck_id].begin(), F_ONE, thread_n);
    beta_timer.stop();

    timer witness_buffer_timer;
    witness_buffer_timer.start();
    for (int b = 0; b < 2; ++b)
    {
        const u32 total_b = total[b];
        if (!total_b)
            continue;
        const u32 size_b = cur.size_v[b];
        auto dep = !b ? 0 : sumcheck_id - 1;
        if (dep == 0)
        {
#pragma omp parallel for schedule(static) if (total_b >= OmpThreshold && thread_n > 1) num_threads(thread_n)
            for (u32 v = 0; v < total_b; ++v)
            {
                mult_array[b][v].clear();
                if (v < size_b)
                    V_mult[b][v] = val[0][cur.ori_id_v[v]];
                else
                    V_mult[b][v].clear();
            }
        }
        else
        {
#pragma omp parallel for schedule(static) if (total_b >= OmpThreshold && thread_n > 1) num_threads(thread_n)
            for (u32 v = 0; v < total_b; ++v)
            {
                mult_array[b][v].clear();
                if (v < size_b)
                    V_mult[b][v] = val[dep][v];
                else
                    V_mult[b][v].clear();
            }
        }
    }
    witness_buffer_timer.stop();

    timer unary_gate_timer;
    unary_gate_timer.start();
    if (phase1_unary != nullptr)
    {
        add_term = (*phase1_unary)[0] * V_u0 + (*phase1_unary)[1] * V_u1;
    }
    else if (cur.uni_interval.size() >= 2 && thread_n > 1)
    {
        const int interval_count = static_cast<int>(cur.uni_interval.size());
        const int actual_threads = std::min(thread_n, interval_count);
        std::vector<F> local_sum(actual_threads);
#pragma omp parallel num_threads(actual_threads)
        {
            const int tid = omp_get_thread_num();
            local_sum[tid].clear();
#pragma omp for schedule(static)
            for (int j = 0; j < interval_count; ++j)
            {
                const int l = cur.uni_interval[j].first;
                const int r = cur.uni_interval[j].second;
                F ss;
                ss.clear();
                for (int i = l; i < r; ++i)
                {
                    auto &gate = cur.uni_gates[i];
                    auto V_u = !gate.lu ? V_u0 : V_u1;
                    ss += beta_g[gate.g] * beta_u[gate.u] * V_u * fr_from_i128(gate.sc);
                }
                local_sum[tid] += ss;
            }
        }
        for (int i = 0; i < actual_threads; ++i)
            add_term += local_sum[i];
    }
    else
        for (auto &gate : cur.uni_gates)
        {
            auto V_u = !gate.lu ? V_u0 : V_u1;
            add_term = add_term + beta_g[gate.g] * beta_u[gate.u] * V_u * fr_from_i128(gate.sc);
        }
    unary_gate_timer.stop();

    timer binary_gate_timer;
    binary_gate_timer.start();
    const auto &bin_intervals_p2 = !cur.bin_interval_v.empty() ? cur.bin_interval_v : cur.bin_interval;
    const auto &bin_gates_p2 = !cur.bin_gates_v.empty() ? cur.bin_gates_v : cur.bin_gates;
    if (bin_intervals_p2.size() >= 2 && thread_n > 1)
    {
        const int interval_count = static_cast<int>(bin_intervals_p2.size());
        const int actual_threads = std::min(thread_n, interval_count);
#pragma omp parallel for schedule(static) num_threads(actual_threads)
        for (int j = 0; j < interval_count; ++j)
        {
            const int l = bin_intervals_p2[j].first;
            const int r = bin_intervals_p2[j].second;
            int i = l;
            while (i < r)
            {
                const auto &first = bin_gates_p2[i];
                const bool idx =
                    first.getLayerIdV(sumcheck_id) != 0;
                const u32 target = first.v;
                F subtotal[2] = {F_ZERO, F_ZERO};
                bool present[2] = {false, false};
                do
                {
                    const auto &gate = bin_gates_p2[i];
                    const bool source_branch = gate.getLayerIdU(sumcheck_id) != 0;
                    present[source_branch] = true;
                    add_scaled_field(
                        subtotal[source_branch],
                        beta_g[gate.g] * beta_u[gate.u],
                        gate.sc);
                    ++i;
                } while (i < r &&
                         (bin_gates_p2[i].getLayerIdV(sumcheck_id) != 0) == idx &&
                         bin_gates_p2[i].v == target);
                if (present[0])
                    mult_array[idx][target].b += subtotal[0] * V_u0;
                if (present[1])
                    mult_array[idx][target].b += subtotal[1] * V_u1;
            }
        }
    }
    else
        for (auto &gate : cur.bin_gates)
        {
            bool idx = gate.getLayerIdV(sumcheck_id);
            auto V_u = !gate.getLayerIdU(sumcheck_id) ? V_u0 : V_u1;
            add_scaled_field(
                mult_array[idx][gate.v].b,
                beta_g[gate.g] * beta_u[gate.u] * V_u, gate.sc);
        }
    binary_gate_timer.stop();

    round = 0;
    total_timer.stop();
    sumcheck_init_timing.phase2_total_seconds += total_timer.elapse_sec();
    sumcheck_init_timing.phase2_workspace_seconds +=
        workspace_timer.elapse_sec();
    sumcheck_init_timing.phase2_witness_buffer_seconds +=
        witness_buffer_timer.elapse_sec();
    sumcheck_init_timing.phase2_beta_seconds += beta_timer.elapse_sec();
    sumcheck_init_timing.phase2_unary_gate_seconds +=
        unary_gate_timer.elapse_sec();
    sumcheck_init_timing.phase2_binary_gate_seconds +=
        binary_gate_timer.elapse_sec();
    ++sumcheck_init_timing.phase2_calls;
}

void prover::initInputClaimReduction(
    const vector<F> &s_u, const vector<F> &s_v,
    const vector<vector<F>> &r_uu, const vector<vector<F>> &r_vv)
{
    input_reduction_init_timing_ = InputReductionInitTiming{};
    const auto reset_start = std::chrono::steady_clock::now();
    sumcheck_id = 0;
    total[1] = (1ULL << C.circuit[sumcheck_id].bit_length);
    total_size[1] = (1ULL << C.circuit[sumcheck_id].bit_length);
    const int input_threads = thread_num > 0 ? thread_num : 1;
    if (!input_claim_workspace_initialized ||
        input_claim_weights.size() != total[1] ||
        input_claim_active_bits.size() != (total[1] + 63) / 64)
    {
        input_claim_weights.resize(total[1]);
#pragma omp parallel for schedule(static) num_threads(input_threads) if (total[1] >= static_cast<std::size_t>(input_threads) * 256)
        for (long long raw = 0;
             raw < static_cast<long long>(total[1]); ++raw)
        {
            input_claim_weights[static_cast<std::size_t>(raw)].clear();
        }
        input_claim_active_bits.assign((total[1] + 63) / 64, 0);
        input_claim_active_indices.clear();
        input_claim_workspace_initialized = true;
    }
    else
    {
#pragma omp parallel for schedule(static) num_threads(input_threads) if (input_claim_active_indices.size() >= \
                                                                             static_cast<std::size_t>(input_threads) * 256)
        for (long long raw = 0;
             raw < static_cast<long long>(input_claim_active_indices.size());
             ++raw)
        {
            input_claim_weights[input_claim_active_indices[static_cast<std::size_t>(raw)]].clear();
        }
        std::fill(input_claim_active_bits.begin(),
                  input_claim_active_bits.end(), 0);
        input_claim_active_indices.clear();
    }

    input_reduction_init_timing_.reset_seconds = std::chrono::duration<double>(
                                                     std::chrono::steady_clock::now() - reset_start)
                                                     .count();
    const auto add_input_coefficient = [&](u32 index,
                                           const F &coefficient)
    {
        const std::size_t word = static_cast<std::size_t>(index) >> 6;
        const std::uint64_t mask =
            std::uint64_t(1) << (index & 63U);
        if ((input_claim_active_bits[word] & mask) == 0)
        {
            input_claim_active_bits[word] |= mask;
            input_claim_active_indices.push_back(index);
        }
        input_claim_weights[index] += coefficient;
    };

    InputScatterWorkspace scatter_workspace;
    r_u[0].resize(C.circuit[0].bit_length);

    i8 max_bl = 0;
    for (int i = sumcheck_id + 1; i < C.size; ++i)
        max_bl = max(max_bl, max(C.circuit[i].bit_length_u[0], C.circuit[i].bit_length_v[0]));
    beta_g.resize(1ULL << max_bl);
    for (int i = sumcheck_id + 1; i < C.size; ++i)
    {
        i8 bit_length_i = C.circuit[i].bit_length_u[0];
        u32 size_i = C.circuit[i].size_u[0];

        if (~bit_length_i)
        {
            r_u[i].resize(C.circuit[i].max_bl_u);
            for (int j = 0; j < C.circuit[i].max_bl_u; j++)
                r_u[i][j] = r_uu[i][j];
            auto stage_start = std::chrono::steady_clock::now();
            initBetaTable(beta_g, bit_length_i, r_u[i].begin(), s_u[i - 1], thread_num);
            input_reduction_init_timing_.equality_seconds += std::chrono::duration<double>(
                                                                 std::chrono::steady_clock::now() - stage_start)
                                                                 .count();
            stage_start = std::chrono::steady_clock::now();
            if (scatter_input_route(C.circuit[i].ori_id_u, size_i, beta_g,
                                    input_claim_weights, input_claim_active_bits, input_claim_active_indices,
                                    input_threads, scatter_workspace))
            {
                ++input_reduction_init_timing_.parallel_routes;
            }
            else
            {
                for (u32 hu = 0; hu < size_i; ++hu)
                    add_input_coefficient(C.circuit[i].ori_id_u[hu], beta_g[hu]);
                ++input_reduction_init_timing_.serial_routes;
            }
            input_reduction_init_timing_.scatter_seconds += std::chrono::duration<double>(
                                                                std::chrono::steady_clock::now() - stage_start)
                                                                .count();
        }
        bit_length_i = C.circuit[i].bit_length_v[0];
        size_i = C.circuit[i].size_v[0];

        if (~bit_length_i && C.circuit[i].need_phase2)
        {
            r_v[i].resize(C.circuit[i].max_bl_v);
            for (int j = 0; j < C.circuit[i].max_bl_v; j++)
                r_v[i][j] = r_vv[i][j];
            auto stage_start = std::chrono::steady_clock::now();
            initBetaTable(beta_g, bit_length_i, r_v[i].begin(), s_v[i - 1], thread_num);
            input_reduction_init_timing_.equality_seconds += std::chrono::duration<double>(
                                                                 std::chrono::steady_clock::now() - stage_start)
                                                                 .count();
            stage_start = std::chrono::steady_clock::now();
            if (scatter_input_route(C.circuit[i].ori_id_v, size_i, beta_g,
                                    input_claim_weights, input_claim_active_bits, input_claim_active_indices,
                                    input_threads, scatter_workspace))
            {
                ++input_reduction_init_timing_.parallel_routes;
            }
            else
            {
                for (u32 hv = 0; hv < size_i; ++hv)
                    add_input_coefficient(C.circuit[i].ori_id_v[hv], beta_g[hv]);
                ++input_reduction_init_timing_.serial_routes;
            }
            input_reduction_init_timing_.scatter_seconds += std::chrono::duration<double>(
                                                                std::chrono::steady_clock::now() - stage_start)
                                                                .count();
        }
    }
    round = 0;
}

void prover::releaseInputClaimWeights()
{
    vector<F>().swap(input_claim_weights);
    vector<std::uint64_t>().swap(input_claim_active_bits);
    vector<u32>().swap(input_claim_active_indices);
    input_claim_workspace_initialized = false;
}

quadratic_poly prover::sumcheckUpdate1(const F &previous_random)
{
    return sumcheckUpdate(previous_random, r_u[sumcheck_id]);
}

quadratic_poly prover::sumcheckUpdate2(const F &previous_random)
{
    return sumcheckUpdate(previous_random, r_v[sumcheck_id]);
}

quadratic_poly prover::sumcheckUpdate(const F &previous_random, vector<F> &r_arr)
{
    if (round)
        r_arr.at(round - 1) = previous_random;
    ++round;
    quadratic_poly ret;

    add_term = add_term * (F_ONE - previous_random);
    for (int b = 0; b < 2; ++b)
        ret = ret + sumcheckUpdateEach(previous_random, b);
    ret = ret + quadratic_poly(F_ZERO, -add_term, add_term);

    proof_size += F_BYTE_SIZE * 3;
    return ret;
}

quadratic_poly prover::sumcheckUpdateEach(const F &previous_random, bool idx)
{
    auto &tmp_mult = mult_array[idx];
    auto &tmp_v = V_mult[idx];
    auto &tmp_mult_2 = tmp_mult_array[idx];
    auto &tmp_v_2 = tmp_V_mult[idx];

    if (total[idx] == 1)
    {
        tmp_v[0] = tmp_v[0].eval(previous_random);
        tmp_mult[0] = tmp_mult[0].eval(previous_random);
        add_term = add_term + tmp_v[0].b * tmp_mult[0].b;
    }

    quadratic_poly ret;
    ret.clear();
    if (total[idx] < (1 << 15))
    {
        for (u32 i = 0; i < (total[idx] >> 1); ++i)
        {
            u32 g0 = i << 1, g1 = i << 1 | 1;
            if (g0 >= total_size[idx])
            {
                tmp_v[i].clear();
                tmp_mult[i].clear();
                continue;
            }
            tmp_v[i] = interpolate(tmp_v[g0].eval(previous_random), tmp_v[g1].eval(previous_random));
            tmp_mult[i] = interpolate(tmp_mult[g0].eval(previous_random), tmp_mult[g1].eval(previous_random));
            ret = ret + tmp_mult[i] * tmp_v[i];
        }
    }
    else
    {
        int total_work = (total[idx] >> 1);
        int max_work = std::min<int>(total_work, (total_size[idx] + 1) >> 1);
        const int thread_n = thread_num > 0 ? thread_num : 1;
        if (thread_n > 1 && max_work > 1)
        {
            const int actual_threads = std::min(thread_n, max_work);
            std::vector<quadratic_poly> local_sums(actual_threads);
#pragma omp parallel num_threads(actual_threads)
            {
                const int tid = omp_get_thread_num();
                quadratic_poly sum;
                sum.clear();
#pragma omp for schedule(static)
                for (int i = 0; i < max_work; ++i)
                {
                    u32 g0 = static_cast<u32>(i) << 1;
                    u32 g1 = g0 | 1;
                    tmp_v_2[i] = interpolate(tmp_v[g0].eval(previous_random), tmp_v[g1].eval(previous_random));
                    tmp_mult_2[i] = interpolate(tmp_mult[g0].eval(previous_random), tmp_mult[g1].eval(previous_random));
                    sum = sum + tmp_mult_2[i] * tmp_v_2[i];
                }
#pragma omp for schedule(static)
                for (int i = max_work; i < total_work; ++i)
                {
                    tmp_v_2[i].clear();
                    tmp_mult_2[i].clear();
                }
                local_sums[tid] = sum;
            }
            for (int j = 0; j < actual_threads; ++j)
                ret = ret + local_sums[j];
        }
        else
        {
            for (int i = 0; i < total_work; ++i)
            {
                if (i >= max_work)
                {
                    tmp_v_2[i].clear();
                    tmp_mult_2[i].clear();
                    continue;
                }
                u32 g0 = static_cast<u32>(i) << 1;
                u32 g1 = g0 | 1;
                tmp_v_2[i] = interpolate(tmp_v[g0].eval(previous_random), tmp_v[g1].eval(previous_random));
                tmp_mult_2[i] = interpolate(tmp_mult[g0].eval(previous_random), tmp_mult[g1].eval(previous_random));
                ret = ret + tmp_mult_2[i] * tmp_v_2[i];
            }
        }
        tmp_mult.swap(tmp_mult_2);
        tmp_v.swap(tmp_v_2);
    }

    total[idx] >>= 1;
    total_size[idx] = (total_size[idx] + 1) >> 1;

    return ret;
}

F prover::Vres(const vector<F>::const_iterator &r, u32 output_size, u8 r_size, int layer_id, int start)
{
    vector<F> output(output_size);
    for (u32 i = 0; i < output_size; ++i)
        output[i] = val[layer_id][i + start];
    u32 whole = 1ULL << r_size;
    for (int i = 0; i < r_size; ++i)
    {
        for (u32 j = 0; j < (whole >> 1); ++j)
        {
            if (j > 0)
                output[j].clear();
            if ((j << 1) < output_size)
                output[j] = output[j << 1] * (F_ONE - r[i]);
            if ((j << 1 | 1) < output_size)
                output[j] = output[j] + output[j << 1 | 1] * (r[i]);
        }
        whole >>= 1;
    }
    F res = output[0];

    proof_size += F_BYTE_SIZE;
    return res;
}

void prover::sumcheckFinalize1(const F &previous_random, F &claim_0, F &claim_1)
{
    if (round > 0)
        r_u[sumcheck_id].at(round - 1) = previous_random;
    V_u0 = claim_0 = total[0] ? V_mult[0][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_u[0]) ? V_mult[0][0].b
                                                                                                               : F_ZERO;
    V_u1 = claim_1 = total[1] ? V_mult[1][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_u[1]) ? V_mult[1][0].b
                                                                                                               : F_ZERO;

    proof_size += F_BYTE_SIZE * 2;
}

void prover::sumcheckFinalize2(const F &previous_random, F &claim_0, F &claim_1)
{
    if (round > 0)
        r_v[sumcheck_id].at(round - 1) = previous_random;
    claim_0 = total[0] ? V_mult[0][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_v[0]) ? V_mult[0][0].b
                                                                                                        : F_ZERO;
    claim_1 = total[1] ? V_mult[1][0].eval(previous_random) : (~C.circuit[sumcheck_id].bit_length_v[1]) ? V_mult[1][0].b
                                                                                                        : F_ZERO;

    proof_size += F_BYTE_SIZE * 2;
}

void prover::commitInput(const vector<G1> &gens, int thr, const InputCommitmentKey *fixed_key)
{
    using Clock = std::chrono::steady_clock;
    input_commit_timing_ = InputCommitTiming{};
    auto stage_start = Clock::now();
    const std::size_t logical_size = C.circuit[0].size;
    if (val[0].size() < logical_size)
        throw std::invalid_argument("input witness is shorter than the circuit input");
    if (C.circuit[0].size != (1ULL << C.circuit[0].bit_length))
    {
        const auto padded_size = std::size_t(1) << C.circuit[0].bit_length;
        if (logical_size > padded_size)
            throw std::invalid_argument("circuit input exceeds its declared Boolean domain");
        const int padding_threads = std::max(1, thr);
        if (val[0].capacity() < padded_size)
        {
            std::vector<F> expanded(padded_size);
#pragma omp parallel for schedule(static) num_threads(padding_threads) if (padding_threads > 1 && padded_size >= (1U << 16))
            for (long long raw = 0; raw < static_cast<long long>(padded_size); ++raw)
            {
                const std::size_t i = static_cast<std::size_t>(raw);
                if (i < logical_size)
                    expanded[i] = val[0][i];
                else
                    expanded[i].clear();
            }
            val[0].swap(expanded);
        }
        else
        {
            val[0].resize(padded_size);
#pragma omp parallel for schedule(static) num_threads(padding_threads) if (padding_threads > 1 && padded_size - logical_size >= (1U << 16))
            for (long long raw = static_cast<long long>(logical_size);
                 raw < static_cast<long long>(padded_size); ++raw)
                val[0][static_cast<std::size_t>(raw)].clear();
        }
    }

    input_commit_timing_.padding_seconds = std::chrono::duration<double>(
                                               Clock::now() - stage_start)
                                               .count();
    stage_start = Clock::now();
    const int l = C.circuit[0].bit_length;
    const std::size_t columns = std::size_t(1) << (l - l / 2);
    if (gens.size() < columns + 1)
        throw std::invalid_argument("GKR input commitment generators too small");
    if (fixed_key)
    {
        if (fixed_key->layout().variables != l || fixed_key->layout().value_count != C.circuit[0].size)
            throw std::invalid_argument("fixed input key does not match circuit input shape");
        input_commitment_rows_ = fixed_key->commitRuntime(
            val[0].data(), C.circuit[0].size, const_cast<G1 *>(gens.data()), thr);
    }
    else
    {
        input_commitment_rows_ = hyrax::commit_rows(
            val[0].data(), C.circuit[0].size, const_cast<G1 *>(gens.data()),
            l - l / 2, 0, std::size_t(1) << (l / 2), thr);
    }
    input_commit_timing_.row_commitment_seconds = std::chrono::duration<double>(
                                                      Clock::now() - stage_start)
                                                      .count();
    cc.comm = input_commitment_rows_.data();
    cc.G = gens.back();
    cc.g = (G1 *)gens.data();
    cc.l = l;
    cc.w = val[0].data();
}
