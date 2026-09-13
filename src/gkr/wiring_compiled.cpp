#include "gkr/wiring_compiled.hpp"
#include "common/cache_io.hpp"
#include "common/parallel_build.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace
{

    constexpr char WiringPlanType[] = "zkagent.gkr.wiring-plan\n";

    void accumulate_stats(
        const compiled_mle::Polynomial &polynomial,
        CompiledWiringStats &stats)
    {
        if (polynomial.empty())
            return;
        ++stats.polynomial_groups;
        stats.nonterminal_nodes += polynomial.nodeCount();
        stats.terminal_values += polynomial.terminalCount();
        stats.max_workspace_elements = std::max(
            stats.max_workspace_elements, polynomial.workspaceElements());
        stats.logical_storage_bytes += polynomial.logicalStorageBytes();
    }

    template <std::size_t Groups, class Gate, class Classify, class Encode>
    std::array<std::vector<compiled_mle::Entry>, Groups> encode_gates(
        const std::vector<Gate> &gates, int threads, Classify classify, Encode encode)
    {
        std::array<std::vector<compiled_mle::Entry>, Groups> result;
        if (gates.size() < (1U << 16))
        {
            for (const Gate &gate : gates)
                result[classify(gate)].push_back({encode(gate), gate.sc});
            return result;
        }
        const std::size_t chunks = std::min<std::size_t>(
            static_cast<std::size_t>(threads), 1 + gates.size() / (1U << 16));
        std::vector<std::array<std::size_t, Groups>> offsets(chunks);
        compiled_mle::detail::parallelBuildTasks(chunks, threads, [&](std::size_t chunk)
                                                 {
        std::array<std::size_t, Groups> counts{};
        for (std::size_t i = gates.size() * chunk / chunks;
             i < gates.size() * (chunk + 1) / chunks; ++i)
            ++counts[classify(gates[i])];
        offsets[chunk] = counts; });
        std::array<std::size_t, Groups> sizes{};
        for (std::size_t group = 0; group < Groups; ++group)
        {
            for (std::size_t chunk = 0; chunk < chunks; ++chunk)
            {
                const std::size_t count = offsets[chunk][group];
                offsets[chunk][group] = sizes[group];
                sizes[group] += count;
            }
        }
        compiled_mle::detail::parallelBuildTasks(Groups, threads, [&](std::size_t group)
                                                 { result[group].resize(sizes[group]); });
        compiled_mle::detail::parallelBuildTasks(chunks, threads, [&](std::size_t chunk)
                                                 {
        auto positions = offsets[chunk];
        for (std::size_t i = gates.size() * chunk / chunks;
             i < gates.size() * (chunk + 1) / chunks; ++i) {
            const Gate& gate = gates[i];
            const std::size_t group = classify(gate);
            result[group][positions[group]++] = {encode(gate), gate.sc};
        } });
        return result;
    }

}

void CompiledWiringPlan::saveToFile(const std::string &path) const
{
    const std::string temporary = path + ".tmp";
    try
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error(
                "failed to create compiled wiring plan: " + temporary);
        }
        output.write(WiringPlanType, sizeof(WiringPlanType) - 1);
        cache_io::write_native(output, static_cast<std::uint64_t>(layers_.size()));
        for (const LayerPlan &layer_plan : layers_)
        {
            cache_io::write_native(
                output, static_cast<std::uint8_t>(layer_plan.enabled ? 1 : 0));
            cache_io::write_native(output, static_cast<std::int32_t>(layer_plan.output_bits));
            cache_io::write_native(output, static_cast<std::int32_t>(layer_plan.input_bits_u));
            cache_io::write_native(output, static_cast<std::int32_t>(layer_plan.input_bits_v));
            for (const compiled_mle::Polynomial &polynomial :
                 layer_plan.unary)
            {
                polynomial.writeTo(output);
            }
            for (const compiled_mle::Polynomial &polynomial :
                 layer_plan.binary)
            {
                polynomial.writeTo(output);
            }
        }
        output.close();
        if (!output || std::rename(temporary.c_str(), path.c_str()) != 0)
        {
            throw std::runtime_error(
                "failed to finalize compiled wiring plan: " + path);
        }
    }
    catch (...)
    {
        std::remove(temporary.c_str());
        throw;
    }
}

CompiledWiringPlan CompiledWiringPlan::loadFromFile(
    const std::string &path,
    const layeredCircuit &circuit)
{
    if (circuit.size < 0 ||
        circuit.circuit.size() != static_cast<std::size_t>(circuit.size))
    {
        throw std::invalid_argument(
            "compiled wiring load received an invalid circuit");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("compiled wiring plan is missing: " + path);
    }
    std::array<char, sizeof(WiringPlanType) - 1> plan_type{};
    input.read(plan_type.data(), plan_type.size());
    if (!input || std::string(plan_type.data(), plan_type.size()) != WiringPlanType)
    {
        throw std::runtime_error(
            "compiled wiring plan has an incompatible format");
    }
    const std::uint64_t layer_count = cache_io::read_native<std::uint64_t>(input);
    if (layer_count != circuit.circuit.size())
    {
        throw std::runtime_error(
            "compiled wiring plan has the wrong layer count");
    }

    CompiledWiringPlan result;
    result.layers_.resize(static_cast<std::size_t>(layer_count));
    result.stats_.layers = result.layers_.size();
    for (std::size_t layer_id = 0; layer_id < result.layers_.size(); ++layer_id)
    {
        const layer &source = circuit.circuit[layer_id];
        LayerPlan &target = result.layers_[layer_id];
        const std::uint8_t enabled = cache_io::read_native<std::uint8_t>(input);
        if (enabled > 1)
        {
            throw std::runtime_error(
                "compiled wiring plan has an invalid layer flag");
        }
        target.enabled = enabled != 0;
        target.output_bits = cache_io::read_native<std::int32_t>(input);
        target.input_bits_u = cache_io::read_native<std::int32_t>(input);
        target.input_bits_v = cache_io::read_native<std::int32_t>(input);
        const bool expected_enabled =
            layer_id != 0 && source.ty != layerType::FCONN;
        const int expected_v = source.need_phase2 ? source.max_bl_v : 0;
        if (target.enabled != expected_enabled ||
            (target.enabled &&
             (target.output_bits != source.bit_length ||
              target.input_bits_u != source.max_bl_u ||
              target.input_bits_v != expected_v)))
        {
            throw std::runtime_error(
                "compiled wiring plan does not match circuit metadata");
        }
        for (compiled_mle::Polynomial &polynomial : target.unary)
        {
            polynomial = compiled_mle::Polynomial::readFrom(
                input);
            if (polynomial.variableCount() != static_cast<std::size_t>(
                                                  std::max(0, target.output_bits + target.input_bits_u)))
            {
                throw std::runtime_error(
                    "compiled unary wiring plan has the wrong dimension");
            }
            accumulate_stats(polynomial, result.stats_);
        }
        for (compiled_mle::Polynomial &polynomial : target.binary)
        {
            polynomial = compiled_mle::Polynomial::readFrom(
                input);
            if (polynomial.variableCount() != static_cast<std::size_t>(
                                                  std::max(0, target.output_bits + target.input_bits_u +
                                                                  target.input_bits_v)))
            {
                throw std::runtime_error(
                    "compiled binary wiring plan has the wrong dimension");
            }
            accumulate_stats(polynomial, result.stats_);
        }
    }
    if (input.peek() != std::char_traits<char>::eof())
    {
        throw std::runtime_error(
            "compiled wiring plan has trailing data");
    }
    return result;
}

CompiledWiringPlan CompiledWiringPlan::build(
    const layeredCircuit &circuit,
    int threads,
    std::vector<CompiledWiringLayerTiming> *timings)
{
    if (circuit.size < 0 ||
        circuit.circuit.size() != static_cast<std::size_t>(circuit.size) || threads <= 0)
    {
        throw std::invalid_argument(
            "compiled wiring received an invalid circuit");
    }

    CompiledWiringPlan result;
    result.layers_.resize(circuit.circuit.size());
    if (timings)
    {
        timings->clear();
        timings->resize(circuit.circuit.size());
    }
    compiled_mle::detail::parallelBuildTasks(
        circuit.circuit.size(), threads, [&](std::size_t layer_id)
        {
        if (layer_id == 0) return;
        const layer& source = circuit.circuit[layer_id];
        if (source.ty == layerType::FCONN) return;
        if (source.bit_length < 0 || source.max_bl_u < 0 ||
            (source.need_phase2 && source.max_bl_v < 0)) {
            throw std::invalid_argument(
                "compiled wiring layer has invalid dimensions");
        }

        LayerPlan& target = result.layers_[layer_id];
        target.enabled = true;
        target.output_bits = source.bit_length;
        target.input_bits_u = source.max_bl_u;
        target.input_bits_v = source.need_phase2 ? source.max_bl_v : 0;

        const auto encoding_start = std::chrono::steady_clock::now();
        const compiled_mle::InterleavedIndexEncoder unary_encoder(
            target.output_bits, target.input_bits_u);
        const compiled_mle::InterleavedIndexEncoder binary_encoder(
            target.output_bits, target.input_bits_u, target.input_bits_v);
        auto unary = encode_gates<4>(source.uni_gates, threads,
            [&](const uniGate& gate) -> std::size_t {
                return (gate.lu == 0 ? 0 : 2) + (gate.g >= source.zero_start_id ? 1 : 0);
            }, [&](const uniGate& gate) { return unary_encoder.encode(gate.g, gate.u); });
        auto binary = encode_gates<6>(source.bin_gates, threads,
            [&](const binGate& gate) -> std::size_t {
                if (gate.l > 2)
                    throw std::invalid_argument("compiled wiring binary source category is invalid");
                return static_cast<std::size_t>(gate.l) * 2 +
                    (gate.g >= source.zero_start_id ? 1 : 0);
            }, [&](const binGate& gate) { return binary_encoder.encode(gate.g, gate.u, gate.v); });

        if (timings) (*timings)[layer_id].encoding_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - encoding_start).count();
        compiled_mle::detail::parallelBuildTasks(
            unary.size() + binary.size(), threads, [&](std::size_t index) {
            if (index < unary.size()) {
                target.unary[index] = compiled_mle::Polynomial::build(
                    std::move(unary[index]),
                    static_cast<std::size_t>(
                        target.output_bits + target.input_bits_u),
                    threads,
                    timings ? &(*timings)[layer_id].unary[index] : nullptr);
            } else {
                const std::size_t group = index - unary.size();
                target.binary[group] = compiled_mle::Polynomial::build(
                    std::move(binary[group]),
                    static_cast<std::size_t>(
                        target.output_bits + target.input_bits_u +
                        target.input_bits_v),
                    threads,
                    timings ? &(*timings)[layer_id].binary[group] : nullptr,
                     true);
            }
        }); });
    for (const LayerPlan &layer_plan : result.layers_)
    {
        for (const auto &polynomial : layer_plan.unary)
            accumulate_stats(polynomial, result.stats_);
        for (const auto &polynomial : layer_plan.binary)
            accumulate_stats(polynomial, result.stats_);
    }
    result.stats_.layers = result.layers_.size();
    return result;
}

std::array<Fr, 2> CompiledWiringPlan::evaluatePhase1(
    std::size_t layer_id,
    const Fr &alpha,
    const Fr &beta,
    const Fr &zero_zone_scale,
    const std::vector<Fr> &output_point_0,
    const std::vector<Fr> &output_point_1,
    const std::vector<Fr> &input_point,
    int threads,
    std::vector<Fr> &workspace) const
{
    if (layer_id >= layers_.size() || !layers_[layer_id].enabled)
    {
        throw std::invalid_argument(
            "compiled wiring phase 1 received an unsupported layer");
    }
    const LayerPlan &current = layers_[layer_id];
    const std::vector<Fr> point_0 = alpha.isZero()
                                        ? std::vector<Fr>()
                                        : compiled_mle::interleavePoint(
                                              output_point_0, current.output_bits,
                                              input_point, current.input_bits_u);
    const std::vector<Fr> point_1 = beta.isZero()
                                        ? std::vector<Fr>()
                                        : compiled_mle::interleavePoint(
                                              output_point_1, current.output_bits,
                                              input_point, current.input_bits_u);

    std::array<Fr, 2> result = {{Fr(0), Fr(0)}};
    for (std::size_t category = 0; category < result.size(); ++category)
    {
        const auto at_point = [&](const std::vector<Fr> &point)
        {
            const Fr active = current.unary[category * 2].evaluate(
                point, threads, workspace);
            const Fr zero = current.unary[category * 2 + 1].evaluate(
                point, threads, workspace);
            return active + zero_zone_scale * zero;
        };
        if (!alpha.isZero())
            result[category] += alpha * at_point(point_0);
        if (!beta.isZero())
            result[category] += beta * at_point(point_1);
    }
    return result;
}

std::array<Fr, 3> CompiledWiringPlan::evaluatePhase2(
    std::size_t layer_id,
    const Fr &alpha,
    const Fr &beta,
    const Fr &zero_zone_scale,
    const std::vector<Fr> &output_point_0,
    const std::vector<Fr> &output_point_1,
    const std::vector<Fr> &input_point_u,
    const std::vector<Fr> &input_point_v,
    int threads,
    std::vector<Fr> &workspace) const
{
    if (layer_id >= layers_.size() || !layers_[layer_id].enabled)
    {
        throw std::invalid_argument(
            "compiled wiring phase 2 received an unsupported layer");
    }
    const LayerPlan &current = layers_[layer_id];
    const std::vector<Fr> point_0 = alpha.isZero()
                                        ? std::vector<Fr>()
                                        : compiled_mle::interleavePoint(
                                              output_point_0, current.output_bits,
                                              input_point_u, current.input_bits_u,
                                              input_point_v, current.input_bits_v);
    const std::vector<Fr> point_1 = beta.isZero()
                                        ? std::vector<Fr>()
                                        : compiled_mle::interleavePoint(
                                              output_point_1, current.output_bits,
                                              input_point_u, current.input_bits_u,
                                              input_point_v, current.input_bits_v);

    std::array<Fr, 3> result = {{Fr(0), Fr(0), Fr(0)}};
    for (std::size_t category = 0; category < result.size(); ++category)
    {
        const auto at_point = [&](const std::vector<Fr> &point)
        {
            const Fr active = current.binary[category * 2].evaluate(
                point, threads, workspace);
            const Fr zero = current.binary[category * 2 + 1].evaluate(
                point, threads, workspace);
            return active + zero_zone_scale * zero;
        };
        if (!alpha.isZero())
            result[category] += alpha * at_point(point_0);
        if (!beta.isZero())
            result[category] += beta * at_point(point_1);
    }
    return result;
}
