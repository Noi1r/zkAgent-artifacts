#include "gkr/input_routing_compiled.hpp"
#include "common/cache_io.hpp"
#include "common/parallel_build.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace
{

    constexpr char InputRoutingPlanType[] = "zkagent.gkr.input-routing-plan\n";

    compiled_mle::Polynomial compile_route(
        const std::vector<u32> &routing,
        int input_bits,
        int local_bits,
        int threads,
        CompiledInputRouteTiming *timing)
    {
        if (local_bits < 0 ||
            (local_bits < 32 && routing.size() >
                                    (std::size_t(1) << local_bits)))
        {
            throw std::invalid_argument(
                "compiled input route has an invalid local domain");
        }
        if (!routing.empty() && routing.size() - 1 >
                                    static_cast<std::size_t>(std::numeric_limits<u32>::max()))
        {
            throw std::length_error("compiled input route local index exceeds 32 bits");
        }
        const auto encoding_start = std::chrono::steady_clock::now();
        std::vector<compiled_mle::Entry> entries(routing.size());
        const std::size_t chunks = std::min<std::size_t>(
            static_cast<std::size_t>(threads), 1 + routing.size() / (1U << 16));
        compiled_mle::detail::parallelBuildTasks(chunks, threads, [&](std::size_t chunk)
                                                 {
        for (std::size_t local = routing.size() * chunk / chunks;
             local < routing.size() * (chunk + 1) / chunks; ++local) {
            entries[local] = {
                compiled_mle::encodeInterleaved(
                    routing[local], input_bits,
                    static_cast<u32>(local), local_bits),
                1};
        } });
        if (timing)
            timing->encoding_seconds = std::chrono::duration<double>(
                                           std::chrono::steady_clock::now() - encoding_start)
                                           .count();
        return compiled_mle::Polynomial::build(
            std::move(entries),
            static_cast<std::size_t>(input_bits + local_bits),
            threads, timing ? &timing->polynomial : nullptr);
    }

}

CompiledInputRoutingPlan CompiledInputRoutingPlan::build(
    const layeredCircuit &circuit,
    int threads,
    std::vector<std::array<CompiledInputRouteTiming, 2>> *timings)
{
    if (circuit.size <= 0 ||
        circuit.circuit.size() != static_cast<std::size_t>(circuit.size) ||
        circuit.circuit[0].bit_length < 0 || threads <= 0)
    {
        throw std::invalid_argument(
            "compiled input routing received an invalid circuit");
    }

    CompiledInputRoutingPlan result;
    result.input_bits_ = circuit.circuit[0].bit_length;
    result.layers_.resize(circuit.circuit.size());
    if (timings)
    {
        timings->clear();
        timings->resize(circuit.circuit.size());
    }
    result.stats_.layers = circuit.circuit.size();
    const auto accumulate = [&](const RoutePlan &route,
                                std::size_t source_entries)
    {
        if (!route.enabled)
            return;
        ++result.stats_.route_polynomials;
        result.stats_.source_entries += source_entries;
        result.stats_.nonterminal_nodes += route.polynomial.nodeCount();
        result.stats_.terminal_values += route.polynomial.terminalCount();
        result.stats_.max_workspace_elements = std::max(
            result.stats_.max_workspace_elements,
            route.polynomial.workspaceElements());
        result.stats_.logical_storage_bytes +=
            route.polynomial.logicalStorageBytes();
    };
    compiled_mle::detail::parallelBuildTasks(
        circuit.circuit.size() * 2, threads, [&](std::size_t index)
        {
        const std::size_t layer_id = index / 2;
        if (layer_id == 0) return;
        const layer& source = circuit.circuit[layer_id];
        LayerPlan& target = result.layers_[layer_id];

        if (index % 2 == 0 && source.bit_length_u[0] != -1) {
            if (source.ori_id_u.size() != source.size_u[0]) {
                throw std::invalid_argument(
                    "compiled U input route has inconsistent public metadata");
            }
            target.u.enabled = true;
            target.u.local_bits = source.bit_length_u[0];
            target.u.source_entries = source.ori_id_u.size();
            target.u.polynomial = compile_route(
                source.ori_id_u, result.input_bits_, target.u.local_bits,
                threads,
                timings ? &(*timings)[layer_id][0] : nullptr);
        }
        if (index % 2 != 0 && source.need_phase2 && source.bit_length_v[0] != -1) {
            if (source.ori_id_v.size() != source.size_v[0]) {
                throw std::invalid_argument(
                    "compiled V input route has inconsistent public metadata");
            }
            target.v.enabled = true;
            target.v.local_bits = source.bit_length_v[0];
            target.v.source_entries = source.ori_id_v.size();
            target.v.polynomial = compile_route(
                source.ori_id_v, result.input_bits_, target.v.local_bits,
                threads,
                timings ? &(*timings)[layer_id][1] : nullptr);
        } });
    for (const LayerPlan &layer_plan : result.layers_)
    {
        accumulate(layer_plan.u, layer_plan.u.source_entries);
        accumulate(layer_plan.v, layer_plan.v.source_entries);
    }
    return result;
}

void CompiledInputRoutingPlan::saveToFile(const std::string &path) const
{
    const std::string temporary = path + ".tmp";
    try
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error(
                "failed to create compiled input-routing plan: " + temporary);
        }
        output.write(InputRoutingPlanType, sizeof(InputRoutingPlanType) - 1);
        cache_io::write_native(output, static_cast<std::int32_t>(input_bits_));
        cache_io::write_native(output, static_cast<std::uint64_t>(layers_.size()));
        const auto write_route = [&](const RoutePlan &route)
        {
            cache_io::write_native(
                output, static_cast<std::uint8_t>(route.enabled ? 1 : 0));
            cache_io::write_native(output, static_cast<std::int32_t>(route.local_bits));
            cache_io::write_native(
                output, static_cast<std::uint64_t>(route.source_entries));
            route.polynomial.writeTo(output);
        };
        for (const LayerPlan &layer_plan : layers_)
        {
            write_route(layer_plan.u);
            write_route(layer_plan.v);
        }
        output.close();
        if (!output || std::rename(temporary.c_str(), path.c_str()) != 0)
        {
            throw std::runtime_error(
                "failed to finalize compiled input-routing plan: " + path);
        }
    }
    catch (...)
    {
        std::remove(temporary.c_str());
        throw;
    }
}

CompiledInputRoutingPlan CompiledInputRoutingPlan::loadFromFile(
    const std::string &path,
    const layeredCircuit &circuit)
{
    if (circuit.size <= 0 ||
        circuit.circuit.size() != static_cast<std::size_t>(circuit.size) ||
        circuit.circuit[0].bit_length < 0)
    {
        throw std::invalid_argument(
            "compiled input-routing load received an invalid circuit");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error(
            "compiled input-routing plan is missing: " + path);
    }
    std::array<char, sizeof(InputRoutingPlanType) - 1> plan_type{};
    input.read(plan_type.data(), plan_type.size());
    if (!input || std::string(plan_type.data(), plan_type.size()) != InputRoutingPlanType)
    {
        throw std::runtime_error(
            "compiled input-routing plan has an incompatible format");
    }
    const int input_bits = cache_io::read_native<std::int32_t>(input);
    const std::uint64_t layer_count = cache_io::read_native<std::uint64_t>(input);
    if (input_bits != circuit.circuit[0].bit_length ||
        layer_count != circuit.circuit.size())
    {
        throw std::runtime_error(
            "compiled input-routing plan has incompatible dimensions");
    }

    CompiledInputRoutingPlan result;
    result.input_bits_ = input_bits;
    result.layers_.resize(static_cast<std::size_t>(layer_count));
    result.stats_.layers = result.layers_.size();
    const auto read_route = [&](RoutePlan &route, bool expected_enabled,
                                int expected_bits,
                                std::size_t expected_entries)
    {
        const std::uint8_t enabled = cache_io::read_native<std::uint8_t>(input);
        if (enabled > 1)
        {
            throw std::runtime_error(
                "compiled input-routing plan has an invalid route flag");
        }
        route.enabled = enabled != 0;
        route.local_bits = cache_io::read_native<std::int32_t>(input);
        const std::uint64_t source_entries =
            cache_io::read_native<std::uint64_t>(input);
        if (source_entries > std::numeric_limits<std::size_t>::max())
        {
            throw std::runtime_error(
                "compiled input-routing source count is too large");
        }
        route.source_entries = static_cast<std::size_t>(source_entries);
        route.polynomial = compiled_mle::Polynomial::readFrom(
            input);
        const std::size_t expected_variables = route.enabled
                                                   ? static_cast<std::size_t>(input_bits + expected_bits)
                                                   : 0;
        if (route.enabled != expected_enabled ||
            route.local_bits != (expected_enabled ? expected_bits : 0) ||
            route.source_entries != expected_entries ||
            route.polynomial.variableCount() != expected_variables)
        {
            throw std::runtime_error(
                "compiled input-routing plan does not match circuit metadata");
        }
        if (route.enabled)
        {
            ++result.stats_.route_polynomials;
            result.stats_.source_entries += route.source_entries;
            result.stats_.nonterminal_nodes += route.polynomial.nodeCount();
            result.stats_.terminal_values += route.polynomial.terminalCount();
            result.stats_.max_workspace_elements = std::max(
                result.stats_.max_workspace_elements,
                route.polynomial.workspaceElements());
            result.stats_.logical_storage_bytes +=
                route.polynomial.logicalStorageBytes();
        }
    };
    for (std::size_t layer_id = 0; layer_id < result.layers_.size(); ++layer_id)
    {
        const layer &source = circuit.circuit[layer_id];
        const bool has_u = layer_id != 0 && source.bit_length_u[0] != -1;
        const bool has_v = layer_id != 0 && source.need_phase2 &&
                           source.bit_length_v[0] != -1;
        read_route(
            result.layers_[layer_id].u, has_u,
            has_u ? source.bit_length_u[0] : 0,
            has_u ? source.ori_id_u.size() : 0);
        read_route(
            result.layers_[layer_id].v, has_v,
            has_v ? source.bit_length_v[0] : 0,
            has_v ? source.ori_id_v.size() : 0);
    }
    if (input.peek() != std::char_traits<char>::eof())
    {
        throw std::runtime_error(
            "compiled input-routing plan has trailing data");
    }
    return result;
}

Fr CompiledInputRoutingPlan::evaluate(
    const std::vector<Fr> &input_point,
    const std::vector<std::vector<Fr>> &layer_u_points,
    const std::vector<std::vector<Fr>> &layer_v_points,
    const std::vector<Fr> &sigma_u,
    const std::vector<Fr> &sigma_v,
    int threads,
    std::vector<Fr> &workspace) const
{
    if (input_point.size() < static_cast<std::size_t>(input_bits_) ||
        layer_u_points.size() < layers_.size() ||
        layer_v_points.size() < layers_.size() ||
        sigma_u.size() + 1 < layers_.size() ||
        sigma_v.size() + 1 < layers_.size() ||
        threads <= 0 ||
        workspace.size() < stats_.max_workspace_elements)
    {
        throw std::invalid_argument(
            "compiled input routing received incompatible online state");
    }

    Fr result = Fr(0);
    for (std::size_t layer_id = 1; layer_id < layers_.size(); ++layer_id)
    {
        const LayerPlan &current = layers_[layer_id];
        if (current.u.enabled && !sigma_u[layer_id - 1].isZero())
        {
            const std::vector<Fr> point = compiled_mle::interleavePoint(
                input_point, input_bits_,
                layer_u_points[layer_id], current.u.local_bits);
            result += sigma_u[layer_id - 1] *
                      current.u.polynomial.evaluate(point, threads, workspace);
        }
        if (current.v.enabled && !sigma_v[layer_id - 1].isZero())
        {
            const std::vector<Fr> point = compiled_mle::interleavePoint(
                input_point, input_bits_,
                layer_v_points[layer_id], current.v.local_bits);
            result += sigma_v[layer_id - 1] *
                      current.v.polynomial.evaluate(point, threads, workspace);
        }
    }
    return result;
}
