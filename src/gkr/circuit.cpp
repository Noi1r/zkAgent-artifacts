#include "gkr/circuit.h"
#include "common/utils.hpp"
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace
{

    template <typename GateVec, typename Less>
    bool stable_sort_if_needed(GateVec &gates, Less less)
    {
        if (std::is_sorted(gates.begin(), gates.end(), less))
            return false;
        std::stable_sort(gates.begin(), gates.end(), less);
        return true;
    }

    template <typename GateVec, typename IdxFn, typename KeyFn>
    void build_intervals_by_key(const GateVec &gates, IdxFn idx_fn, KeyFn key_fn,
                                std::vector<std::pair<int, int>> &intervals,
                                int max_intervals_per_source)
    {
        intervals.clear();
        if (gates.size() < 2)
            return;
        if (max_intervals_per_source <= 0)
            throw std::invalid_argument("gate interval count must be positive");

        auto append_intervals = [&](int start, int end)
        {
            if (end <= start)
                return;
            int unique = 0;
            decltype(key_fn(gates[start])) last_key{};
            bool first = true;
            for (int i = start; i < end; ++i)
            {
                const auto key = key_fn(gates[i]);
                if (first || key != last_key)
                {
                    ++unique;
                    last_key = key;
                    first = false;
                }
            }

            const int interval_cap = std::min(max_intervals_per_source, unique);
            if (interval_cap <= 1)
            {
                intervals.emplace_back(start, end);
                return;
            }

            const int keys_per_interval =
                (unique + interval_cap - 1) / interval_cap;
            int interval_start = start;
            int seen = 0;
            first = true;
            for (int i = start; i < end; ++i)
            {
                const auto key = key_fn(gates[i]);
                if (first || key != last_key)
                {
                    ++seen;
                    last_key = key;
                    first = false;
                }
                if (seen >= keys_per_interval && i + 1 < end &&
                    key_fn(gates[i + 1]) != key)
                {
                    intervals.emplace_back(interval_start, i + 1);
                    interval_start = i + 1;
                    seen = 0;
                }
            }
            if (interval_start < end)
                intervals.emplace_back(interval_start, end);
        };

        int segment_start = 0;
        for (int i = 1; i < static_cast<int>(gates.size()); ++i)
        {
            if (idx_fn(gates[i]) != idx_fn(gates[i - 1]))
            {
                append_intervals(segment_start, i);
                segment_start = i;
            }
        }
        append_intervals(segment_start, static_cast<int>(gates.size()));
    }

}

GateIntervalPreparation prepareParallelGateIntervals(
    layer &current, int max_intervals_per_source)
{
    GateIntervalPreparation result;

    const auto unary_category = [](const uniGate &gate)
    {
        return gate.lu != 0;
    };
    const auto unary_less = [&](const uniGate &lhs, const uniGate &rhs)
    {
        const int lhs_category = unary_category(lhs);
        const int rhs_category = unary_category(rhs);
        if (lhs_category != rhs_category)
            return lhs_category < rhs_category;
        if (lhs.u != rhs.u)
            return lhs.u < rhs.u;
        return lhs.g < rhs.g;
    };
    result.unary_reordered =
        stable_sort_if_needed(current.uni_gates, unary_less);
    build_intervals_by_key(
        current.uni_gates, unary_category,
        [](const uniGate &gate)
        { return gate.u; }, current.uni_interval,
        max_intervals_per_source);
    result.unary_intervals = current.uni_interval.size();

    const auto binary_u_category = [](const binGate &gate)
    {
        return gate.l != 0;
    };
    const auto binary_u_less = [&](const binGate &lhs, const binGate &rhs)
    {
        const int lhs_category = binary_u_category(lhs);
        const int rhs_category = binary_u_category(rhs);
        if (lhs_category != rhs_category)
            return lhs_category < rhs_category;
        if (lhs.u != rhs.u)
            return lhs.u < rhs.u;
        return lhs.g < rhs.g;
    };
    result.binary_u_reordered =
        stable_sort_if_needed(current.bin_gates, binary_u_less);
    build_intervals_by_key(
        current.bin_gates, binary_u_category,
        [](const binGate &gate)
        { return gate.u; }, current.bin_interval_u,
        max_intervals_per_source);
    result.binary_u_intervals = current.bin_interval_u.size();

    if (!current.bin_gates_v.empty())
    {
        const auto binary_v_category = [](const binGate &gate)
        {
            return (gate.l & 1) != 0;
        };
        const auto binary_v_less = [&](const binGate &lhs,
                                       const binGate &rhs)
        {
            const int lhs_category = binary_v_category(lhs);
            const int rhs_category = binary_v_category(rhs);
            if (lhs_category != rhs_category)
                return lhs_category < rhs_category;
            if (lhs.v != rhs.v)
                return lhs.v < rhs.v;
            return lhs.g < rhs.g;
        };
        result.binary_v_reordered =
            stable_sort_if_needed(current.bin_gates_v, binary_v_less);
        build_intervals_by_key(
            current.bin_gates_v, binary_v_category,
            [](const binGate &gate)
            { return gate.v; },
            current.bin_interval_v, max_intervals_per_source);
    }
    else
    {
        current.bin_interval_v.clear();
    }
    result.binary_v_intervals = current.bin_interval_v.size();
    return result;
}

bool layeredCircuit::saveToFile(const char *path) const
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        return false;

    f.write((char *)&size, sizeof(size));
    for (int i = 0; i < size; ++i)
    {
        const auto &l = circuit[i];
        f.write((char *)&l.ty, sizeof(l.ty));
        f.write((char *)&l.size, sizeof(l.size));
        f.write((char *)&l.bit_length, sizeof(l.bit_length));
        f.write((char *)l.size_u, sizeof(l.size_u));
        f.write((char *)l.size_v, sizeof(l.size_v));
        f.write((char *)l.bit_length_u, sizeof(l.bit_length_u));
        f.write((char *)l.bit_length_v, sizeof(l.bit_length_v));
        f.write((char *)&l.max_bl_u, sizeof(l.max_bl_u));
        f.write((char *)&l.max_bl_v, sizeof(l.max_bl_v));
        f.write((char *)&l.need_phase2, sizeof(l.need_phase2));
        f.write((char *)&l.zero_start_id, sizeof(l.zero_start_id));
        f.write((char *)&l.fft_bit_length, sizeof(l.fft_bit_length));

        auto writeVec = [&](const auto &v)
        {
            size_t n = v.size();
            f.write((char *)&n, sizeof(n));
            if (n)
                f.write((char *)v.data(), n * sizeof(v[0]));
        };
        writeVec(l.uni_gates);
        writeVec(l.bin_gates);
        writeVec(l.bin_gates_v);
        writeVec(l.ori_id_u);
        writeVec(l.ori_id_v);
    }
    return f.good();
}

bool layeredCircuit::loadFromFile(const char *path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;

    f.read((char *)&size, sizeof(size));
    circuit.resize(size);
    for (int i = 0; i < size; ++i)
    {
        auto &l = circuit[i];
        f.read((char *)&l.ty, sizeof(l.ty));
        f.read((char *)&l.size, sizeof(l.size));
        f.read((char *)&l.bit_length, sizeof(l.bit_length));
        f.read((char *)l.size_u, sizeof(l.size_u));
        f.read((char *)l.size_v, sizeof(l.size_v));
        f.read((char *)l.bit_length_u, sizeof(l.bit_length_u));
        f.read((char *)l.bit_length_v, sizeof(l.bit_length_v));
        f.read((char *)&l.max_bl_u, sizeof(l.max_bl_u));
        f.read((char *)&l.max_bl_v, sizeof(l.max_bl_v));
        f.read((char *)&l.need_phase2, sizeof(l.need_phase2));
        f.read((char *)&l.zero_start_id, sizeof(l.zero_start_id));
        f.read((char *)&l.fft_bit_length, sizeof(l.fft_bit_length));

        auto readVec = [&](auto &v)
        {
            size_t n;
            f.read((char *)&n, sizeof(n));
            v.resize(n);
            if (n)
                f.read((char *)v.data(), n * sizeof(v[0]));
        };
        readVec(l.uni_gates);
        readVec(l.bin_gates);
        readVec(l.bin_gates_v);
        readVec(l.ori_id_u);
        readVec(l.ori_id_v);
    }
    return f.good();
}

void layeredCircuit::initSubset()
{
    if (g_debug)
    {
        std::cerr << "initSubset: circuit[0].size = " << circuit[0].size << std::endl;
    }
    vector<int> visited_uidx(circuit[0].size);
    vector<u64> subset_uidx(circuit[0].size);
    vector<int> visited_vidx(circuit[0].size);
    vector<u64> subset_vidx(circuit[0].size);
    for (int i = 1; i < size; ++i)
    {
        auto &cur = circuit[i], &lst = circuit[i - 1];
        bool has_pre_layer_u = circuit[i].ty == layerType::FFT || circuit[i].ty == layerType::IFFT;
        bool has_pre_layer_v = false;
        for (auto &gate : cur.uni_gates)
        {
            if (!gate.lu)
            {
                if (visited_uidx[gate.u] != i)
                {
                    visited_uidx[gate.u] = i;
                    subset_uidx[gate.u] = cur.size_u[0];
                    cur.ori_id_u.push_back(gate.u);
                    ++cur.size_u[0];
                }
                gate.u = subset_uidx[gate.u];
            }
            has_pre_layer_u |= (gate.lu != 0);
        }
        for (auto &gate : cur.bin_gates)
        {
            if (!gate.getLayerIdU(i))
            {
                if (visited_uidx[gate.u] != i)
                {
                    visited_uidx[gate.u] = i;
                    subset_uidx[gate.u] = cur.size_u[0];
                    cur.ori_id_u.push_back(gate.u);
                    ++cur.size_u[0];
                }
                gate.u = subset_uidx[gate.u];
            }
            if (!gate.getLayerIdV(i))
            {
                if (visited_vidx[gate.v] != i)
                {
                    visited_vidx[gate.v] = i;
                    subset_vidx[gate.v] = cur.size_v[0];
                    cur.ori_id_v.push_back(gate.v);
                    ++cur.size_v[0];
                }
                gate.v = subset_vidx[gate.v];
            }
            has_pre_layer_u |= (gate.getLayerIdU(i) != 0);
            has_pre_layer_v |= (gate.getLayerIdV(i) != 0);
        }
        if (!cur.bin_gates_v.empty())
        {
            for (auto &gate : cur.bin_gates_v)
            {
                if (!gate.getLayerIdU(i))
                    gate.u = subset_uidx[gate.u];
                if (!gate.getLayerIdV(i))
                    gate.v = subset_vidx[gate.v];
            }
        }
        cur.bit_length_u[0] = ceilPow2BitLength(cur.size_u[0]);
        cur.bit_length_v[0] = ceilPow2BitLength(cur.size_v[0]);

        if (has_pre_layer_u)
            switch (cur.ty)
            {
            case layerType::FFT:
                cur.size_u[1] = 1ULL << (cur.fft_bit_length - 1);
                cur.bit_length_u[1] = cur.fft_bit_length - 1;
                break;
            case layerType::IFFT:
                cur.size_u[1] = 1ULL << cur.fft_bit_length;
                cur.bit_length_u[1] = cur.fft_bit_length;
                break;
            default:
                cur.size_u[1] = lst.size;
                cur.bit_length_u[1] = lst.bit_length;
                break;
            }
        else
        {
            cur.size_u[1] = 0;
            cur.bit_length_u[1] = -1;
        }
        if (has_pre_layer_v)
        {
            if (cur.ty == layerType::DOT_PROD)
            {
                cur.size_v[1] = lst.size >> cur.fft_bit_length;
                cur.bit_length_v[1] = lst.bit_length - cur.fft_bit_length;
            }
            else
            {
                cur.size_v[1] = lst.size;
                cur.bit_length_v[1] = lst.bit_length;
            }
        }
        else
        {
            cur.size_v[1] = 0;
            cur.bit_length_v[1] = -1;
        }
        cur.updateSize();
    }
}

void layeredCircuit::init(u8 q_bit_size, int _layer_sz)
{
    size = _layer_sz;
    circuit.resize(size);
}
