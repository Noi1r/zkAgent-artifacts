#include "lookup/lookup_input_routing_compiled.hpp"
#include "common/cache_io.hpp"
#include "common/bit_utils.hpp"

#include "common/field_integer.hpp"
#include "lookup/lookup_common.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace
{

    constexpr std::array<char, 8> LookupRoutingMagic{{'Z', 'K', 'L', 'R', 'O', 'U', 'T', 'E'}};

    int log2_power_of_two(std::size_t value)
    {
        return zkagent_integer::log2_power_of_two(
            value,
            "compiled lookup routing requires a power-of-two input domain");
    }

    int local_bits_for_rows(std::size_t rows)
    {
        if (rows == 0)
            throw std::invalid_argument("compiled lookup routing received an empty relation");
        if (rows > (std::size_t(1) << (8 * sizeof(std::size_t) - 1)))
            throw std::length_error("compiled lookup routing local domain is too large");
        return zkagent_integer::ceil_log2(rows);
    }

    compiled_mle::Polynomial compile_route(
        const std::vector<std::size_t> &offsets,
        int input_bits,
        int local_bits)
    {
        std::vector<compiled_mle::Entry> entries;
        entries.reserve(offsets.size());
        for (std::size_t local = 0; local < offsets.size(); ++local)
        {
            if (offsets[local] > std::numeric_limits<u32>::max() ||
                local > std::numeric_limits<u32>::max())
            {
                throw std::length_error(
                    "compiled lookup routing index exceeds 32 bits");
            }
            entries.push_back({compiled_mle::encodeInterleaved(
                                   static_cast<u32>(offsets[local]), input_bits,
                                   static_cast<u32>(local), local_bits),
                               1});
        }
        return compiled_mle::Polynomial::build(
            std::move(entries),
            static_cast<std::size_t>(input_bits + local_bits));
    }

}

CompiledLookupInputRoutingPlan CompiledLookupInputRoutingPlan::build(
    const LookupLayout &layout)
{
    if (!zkagent_integer::is_power_of_two(layout.inputDomainSize()))
    {
        throw std::invalid_argument(
            "compiled lookup routing received an invalid layout");
    }

    CompiledLookupInputRoutingPlan result;
    result.input_domain_size_ = layout.inputDomainSize();
    result.input_bits_ = log2_power_of_two(result.input_domain_size_);
    result.affine_.resize(layout.affineBindings().size());
    result.reconstruction_.resize(
        layout.reconstructionBindings().size());

    const auto account = [&](const compiled_mle::Polynomial &polynomial,
                             std::size_t entries)
    {
        ++result.stats_.route_polynomials;
        result.stats_.source_entries += entries;
        result.stats_.nonterminal_nodes += polynomial.nodeCount();
        result.stats_.terminal_values += polynomial.terminalCount();
        result.stats_.max_workspace_elements = std::max(
            result.stats_.max_workspace_elements,
            polynomial.workspaceElements());
        result.stats_.logical_storage_bytes +=
            polynomial.logicalStorageBytes();
    };

    for (std::size_t index = 0;
         index < layout.affineBindings().size(); ++index)
    {
        const LookupAffineBinding &binding =
            layout.affineBindings()[index];
        AffinePlan &plan = result.affine_[index];
        plan.local_bits = local_bits_for_rows(binding.input_offsets.size());
        plan.input = compile_route(
            binding.input_offsets, result.input_bits_, plan.local_bits);
        account(plan.input, binding.input_offsets.size());
        if (!binding.output_offsets.empty())
        {
            if (binding.output_offsets.size() !=
                binding.input_offsets.size())
            {
                throw std::invalid_argument(
                    "compiled affine lookup routing has mismatched maps");
            }
            plan.output = compile_route(
                binding.output_offsets, result.input_bits_, plan.local_bits);
            account(plan.output, binding.output_offsets.size());
        }
    }

    for (std::size_t index = 0;
         index < layout.reconstructionBindings().size(); ++index)
    {
        const LookupReconstructionBinding &binding =
            layout.reconstructionBindings()[index];
        ReconstructionPlan &plan = result.reconstruction_[index];
        plan.local_bits = local_bits_for_rows(binding.input_offsets.size());
        plan.input_coefficient =
            zkagent_integer::fr_from_i128(binding.input_coefficient);
        plan.input = compile_route(
            binding.input_offsets, result.input_bits_, plan.local_bits);
        account(plan.input, binding.input_offsets.size());
    }

    return result;
}

void CompiledLookupInputRoutingPlan::saveToFile(const std::string &path) const
{
    const std::string temporary = path + ".tmp";
    try
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out)
            throw std::runtime_error("cannot create lookup routing cache");
        out.write(LookupRoutingMagic.data(), LookupRoutingMagic.size());
        cache_io::write_native(out, static_cast<std::uint64_t>(input_domain_size_));
        cache_io::write_native(out, static_cast<std::uint64_t>(affine_.size()));
        cache_io::write_native(out, static_cast<std::uint64_t>(reconstruction_.size()));
        for (const auto &plan : affine_)
        {
            plan.input.writeTo(out);
            plan.output.writeTo(out);
        }
        for (const auto &plan : reconstruction_)
            plan.input.writeTo(out);
        out.close();
        if (!out || std::rename(temporary.c_str(), path.c_str()) != 0)
            throw std::runtime_error("cannot finalize lookup routing cache");
    }
    catch (...)
    {
        std::remove(temporary.c_str());
        throw;
    }
}

CompiledLookupInputRoutingPlan CompiledLookupInputRoutingPlan::loadFromFile(
    const std::string &path, const LookupLayout &layout)
{
    if (!zkagent_integer::is_power_of_two(layout.inputDomainSize()))
        throw std::invalid_argument("invalid lookup routing cache layout");
    std::ifstream in(path, std::ios::binary);
    std::array<char, 8> magic{};
    in.read(magic.data(), magic.size());
    if (!in || magic != LookupRoutingMagic)
        throw std::runtime_error("missing/incompatible lookup routing cache");
    const auto domain = cache_io::read_native<std::uint64_t>(in);
    const auto affine_count = cache_io::read_native<std::uint64_t>(in);
    const auto reconstruction_count = cache_io::read_native<std::uint64_t>(in);
    CompiledLookupInputRoutingPlan result;
    if (!in || domain != layout.inputDomainSize() || affine_count != layout.affineBindings().size() ||
        reconstruction_count != layout.reconstructionBindings().size())
        throw std::runtime_error("lookup routing cache dimensions do not match");
    result.input_domain_size_ = layout.inputDomainSize();
    result.input_bits_ = log2_power_of_two(result.input_domain_size_);
    result.affine_.resize(static_cast<std::size_t>(affine_count));
    result.reconstruction_.resize(static_cast<std::size_t>(reconstruction_count));
    auto read = [&](std::size_t rows, bool enabled)
    {
        auto polynomial = compiled_mle::Polynomial::readFrom(in);
        const std::size_t variables = enabled ? result.input_bits_ + local_bits_for_rows(rows) : 0;
        if (polynomial.variableCount() != variables)
            throw std::runtime_error("lookup routing cache polynomial dimension mismatch");
        if (enabled)
        {
            ++result.stats_.route_polynomials;
            result.stats_.source_entries += rows;
            result.stats_.nonterminal_nodes += polynomial.nodeCount();
            result.stats_.terminal_values += polynomial.terminalCount();
            result.stats_.max_workspace_elements = std::max(result.stats_.max_workspace_elements, polynomial.workspaceElements());
            result.stats_.logical_storage_bytes += polynomial.logicalStorageBytes();
        }
        return polynomial;
    };
    for (std::size_t i = 0; i < result.affine_.size(); ++i)
    {
        const auto &binding = layout.affineBindings()[i];
        auto &plan = result.affine_[i];
        plan.local_bits = local_bits_for_rows(binding.input_offsets.size());
        plan.input = read(binding.input_offsets.size(), true);
        plan.output = read(binding.output_offsets.size(), !binding.output_offsets.empty());
    }
    for (std::size_t i = 0; i < result.reconstruction_.size(); ++i)
    {
        const auto &binding = layout.reconstructionBindings()[i];
        auto &plan = result.reconstruction_[i];
        plan.local_bits = local_bits_for_rows(binding.input_offsets.size());
        plan.input_coefficient =
            zkagent_integer::fr_from_i128(binding.input_coefficient);
        plan.input = read(binding.input_offsets.size(), true);
    }
    if (in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("trailing lookup routing cache bytes");
    return result;
}

Fr CompiledLookupInputRoutingPlan::evaluate(
    const std::vector<Fr> &input_point,
    const std::vector<InputAffineRelationClaim> &affine_relations,
    const std::vector<InputReconstructionRelationClaim> &
        reconstruction_relations,
    int threads,
    std::vector<Fr> &workspace) const
{
    if (input_point.size() != static_cast<std::size_t>(input_bits_) ||
        affine_relations.size() != affine_.size() ||
        reconstruction_relations.size() != reconstruction_.size() ||
        threads <= 0 ||
        workspace.size() < stats_.max_workspace_elements)
    {
        throw std::invalid_argument(
            "compiled lookup routing received incompatible online state");
    }

    Fr result = Fr(0);
    for (const InputAffineRelationClaim &relation : affine_relations)
    {
        if (relation.binding_index >= affine_.size())
        {
            throw std::out_of_range(
                "compiled affine lookup relation index is invalid");
        }
        const AffinePlan &plan = affine_[relation.binding_index];
        if (relation.point.size() !=
            static_cast<std::size_t>(plan.local_bits))
        {
            throw std::invalid_argument(
                "compiled affine lookup relation point has wrong dimension");
        }
        const std::vector<Fr> point = compiled_mle::interleavePoint(
            input_point, input_bits_, relation.point, plan.local_bits);
        Fr relation_evaluation = relation.input_coefficient *
                                 plan.input.evaluate(point, threads, workspace);
        if (!plan.output.empty())
        {
            relation_evaluation += relation.output_coefficient *
                                   plan.output.evaluate(point, threads, workspace);
        }
        result += relation.rho * relation_evaluation;
    }

    for (const InputReconstructionRelationClaim &relation :
         reconstruction_relations)
    {
        if (relation.binding_index >= reconstruction_.size())
        {
            throw std::out_of_range(
                "compiled reconstruction lookup relation index is invalid");
        }
        const ReconstructionPlan &plan =
            reconstruction_[relation.binding_index];
        if (relation.point.size() !=
            static_cast<std::size_t>(plan.local_bits))
        {
            throw std::invalid_argument(
                "compiled reconstruction lookup relation point has wrong dimension");
        }
        const std::vector<Fr> point = compiled_mle::interleavePoint(
            input_point, input_bits_, relation.point, plan.local_bits);
        result += relation.rho * plan.input_coefficient *
                  plan.input.evaluate(point, threads, workspace);
    }
    return result;
}
