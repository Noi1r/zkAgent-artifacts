#include "lookup/lookup_witness.hpp"
#include "common/field_integer.hpp"
#include "common/bit_utils.hpp"

#include "lookup/lookup_common.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{

    constexpr unsigned MaxHostRangeBits = 126;

    void validate_spans(const std::vector<InputSpan> &spans)
    {
        if (spans.empty())
        {
            throw std::invalid_argument("range query must contain at least one input span");
        }
        for (const InputSpan &span : spans)
        {
            if (span.count == 0 || span.stride == 0)
            {
                throw std::invalid_argument("range-query spans require positive count and stride");
            }
            const std::size_t steps = span.count - 1;
            if (steps != 0 &&
                span.stride > (std::numeric_limits<std::size_t>::max() - span.offset) / steps)
            {
                throw std::overflow_error("range-query span endpoint overflows size_t");
            }
        }
    }

}

RangeQuerySpec::RangeQuerySpec(
    std::string id, std::vector<InputSpan> spans, unsigned bit_width)
    : id_(std::move(id)),
      spans_(std::move(spans)),
      bit_width_(bit_width)
{
    if (id_.empty())
        throw std::invalid_argument("range-query id must not be empty");
    validate_spans(spans_);
    if (bit_width_ == 0 || bit_width_ > MaxHostRangeBits)
    {
        throw std::invalid_argument("range bit width must be in [1,126]");
    }
}

RangeQuerySpec RangeQuerySpec::powerOfTwo(
    std::string id, std::vector<InputSpan> spans, unsigned bit_width)
{
    return RangeQuerySpec(std::move(id), std::move(spans), bit_width);
}

std::size_t RangeQuerySpec::valueCount() const
{
    std::size_t result = 0;
    for (const InputSpan &span : spans_)
    {
        if (span.count > std::numeric_limits<std::size_t>::max() - result)
        {
            throw std::overflow_error("range-query value count overflows size_t");
        }
        result += span.count;
    }
    return result;
}

void RangeQuerySpec::validateInputDomain(std::size_t input_domain_size) const
{
    for (const InputSpan &span : spans_)
    {
        const std::size_t endpoint = span.offset + (span.count - 1) * span.stride;
        if (endpoint >= input_domain_size)
        {
            throw std::out_of_range("range-query span lies outside the GKR input domain");
        }
    }
}

LookupGroupSpec::LookupGroupSpec(
    std::string id, std::string table_id, std::vector<ll> table_values,
    std::size_t actual_count, ll padding_value)
    : id_(std::move(id)),
      table_id_(std::move(table_id)),
      table_values_(std::move(table_values)),
      actual_count_(actual_count),
      padded_count_(zkagent_integer::next_power_of_two(
          actual_count, "lookup query padding overflows size_t")),
      padding_value_(padding_value)
{
    if (id_.empty() || table_id_.empty())
    {
        throw std::invalid_argument("lookup group and table ids must not be empty");
    }
    if (table_values_.empty() ||
        !zkagent_integer::is_power_of_two(table_values_.size()))
    {
        throw std::invalid_argument("lookup table size must be a positive power of two");
    }
    if (actual_count_ == 0)
    {
        throw std::invalid_argument("lookup group must contain at least one real query");
    }
    if (std::find(table_values_.begin(), table_values_.end(), padding_value_) == table_values_.end())
    {
        throw std::invalid_argument("lookup padding value must belong to its table");
    }
}

LookupGroupSpec LookupGroupSpec::randomTuple(
    std::string id, std::string table_id,
    std::vector<ll> table_inputs, std::vector<ll> table_outputs,
    std::size_t actual_count, ll padding_input)
{
    LookupGroupSpec result(
        std::move(id), std::move(table_id), std::move(table_inputs),
        actual_count, padding_input);
    if (table_outputs.size() != result.table_values_.size())
    {
        throw std::invalid_argument(
            "random tuple lookup input/output tables must have equal size");
    }
    result.table_outputs_ = std::move(table_outputs);
    result.random_tuple_compression_ = true;
    return result;
}

LookupLayout::LookupLayout(
    std::string id, std::size_t input_domain_size,
    std::vector<LookupGroupSpec> groups,
    std::vector<LookupAffineBinding> affine_bindings,
    std::vector<LookupReconstructionBinding> reconstruction_bindings)
    : id_(std::move(id)),
      input_domain_size_(input_domain_size),
      groups_(std::move(groups)),
      affine_bindings_(std::move(affine_bindings)),
      reconstruction_bindings_(std::move(reconstruction_bindings))
{
    if (id_.empty() || input_domain_size_ == 0 || groups_.empty())
    {
        throw std::invalid_argument("lookup layout requires an id, input domain, and groups");
    }
    std::unordered_set<std::string> group_ids;
    for (const LookupGroupSpec &group : groups_)
    {
        if (!group_ids.insert(group.id()).second)
        {
            throw std::invalid_argument("lookup layout contains duplicate group ids");
        }
    }
    if (affine_bindings_.empty() && reconstruction_bindings_.empty())
    {
        throw std::invalid_argument("lookup layout must contain link bindings");
    }
    for (const LookupAffineBinding &binding : affine_bindings_)
    {
        if (binding.id.empty() || binding.group_index >= groups_.size() ||
            binding.input_offsets.empty() || binding.input_coefficient == 0 ||
            binding.query_base + binding.input_offsets.size() > groups_[binding.group_index].actualCount())
        {
            throw std::invalid_argument("invalid affine lookup binding");
        }
        if (binding.random_tuple_compression !=
            groups_[binding.group_index].usesRandomTupleCompression())
        {
            throw std::invalid_argument(
                "affine lookup binding has the wrong tuple-compression mode");
        }
        if ((binding.random_tuple_compression
                 ? binding.output_offsets.empty()
                 : ((binding.output_coefficient == 0) != binding.output_offsets.empty())) ||
            (binding.random_tuple_compression && binding.output_coefficient != 0) ||
            (!binding.output_offsets.empty() &&
             binding.output_offsets.size() != binding.input_offsets.size()))
        {
            throw std::invalid_argument("inconsistent affine lookup output binding");
        }
        for (std::size_t index : binding.input_offsets)
        {
            if (index >= input_domain_size_)
            {
                throw std::invalid_argument("affine lookup input lies outside GKR input domain");
            }
        }
        for (std::size_t index : binding.output_offsets)
        {
            if (index >= input_domain_size_)
            {
                throw std::invalid_argument("affine lookup output lies outside GKR input domain");
            }
        }
    }
    for (const LookupReconstructionBinding &binding : reconstruction_bindings_)
    {
        if (binding.id.empty() || binding.input_offsets.empty() ||
            binding.input_coefficient == 0 || binding.lookup_terms.empty())
        {
            throw std::invalid_argument("invalid lookup reconstruction binding");
        }
        for (std::size_t input_index : binding.input_offsets)
        {
            if (input_index >= input_domain_size_)
            {
                throw std::invalid_argument(
                    "reconstruction input lies outside GKR input domain");
            }
        }
        for (const LookupReconstructionTerm &term : binding.lookup_terms)
        {
            if (term.group_index >= groups_.size() || term.coefficient == 0 ||
                term.query_base + binding.input_offsets.size() > groups_[term.group_index].actualCount())
            {
                throw std::invalid_argument("invalid reconstruction lookup term");
            }
        }
    }
}

std::size_t LookupLayoutBuilder::addGroup(LookupGroupSpec group)
{
    groups_.push_back(std::move(group));
    return groups_.size() - 1;
}

void LookupLayoutBuilder::addAffineBinding(LookupAffineBinding binding)
{
    affine_bindings_.push_back(std::move(binding));
}

void LookupLayoutBuilder::addReconstructionBinding(
    LookupReconstructionBinding binding)
{
    reconstruction_bindings_.push_back(std::move(binding));
}

LookupLayout LookupLayoutBuilder::finalize() &&
{
    return LookupLayout(std::move(id_), input_domain_size_,
                        std::move(groups_), std::move(affine_bindings_),
                        std::move(reconstruction_bindings_));
}

namespace
{
    std::vector<ll> take_queries(const std::vector<ll> &values, std::size_t count)
    {
        return std::vector<ll>(values.begin(), values.begin() + count);
    }
    std::vector<ll> take_queries(std::vector<ll> &values, std::size_t count)
    {
        values.resize(count);
        return std::move(values);
    }

    template <class Components>
    LookupWitness merge_lookup_witnesses(std::string id, Components &components)
    {
        if (components.empty())
        {
            throw std::invalid_argument("lookup-layout merge requires at least one component");
        }
        const std::size_t input_domain_size = components[0].layout.inputDomainSize();
        for (const LookupWitness &component : components)
        {
            const bool outputs_absent = component.group_query_outputs.empty();
            if (component.layout.inputDomainSize() != input_domain_size ||
                component.group_queries.size() != component.layout.groups().size() ||
                (!outputs_absent && component.group_query_outputs.size() != component.layout.groups().size()))
            {
                throw std::invalid_argument("incompatible lookup-layout component");
            }
        }

        struct MergedGroup
        {
            std::string table_id;
            std::vector<ll> table_values;
            std::vector<ll> table_outputs;
            bool random_tuple_compression = false;
            ll padding_value = 0;
            std::vector<ll> queries;
            std::vector<ll> query_outputs;

            MergedGroup(std::string id, std::vector<ll> table,
                        std::vector<ll> outputs, bool random_tuple, ll padding)
                : table_id(std::move(id)), table_values(std::move(table)),
                  table_outputs(std::move(outputs)),
                  random_tuple_compression(random_tuple),
                  padding_value(padding) {}
        };
        std::vector<MergedGroup> merged_groups;
        std::vector<std::vector<std::pair<std::size_t, std::size_t>>> remap;
        remap.resize(components.size());

        for (std::size_t component_index = 0;
             component_index < components.size(); ++component_index)
        {
            auto &component = components[component_index];
            remap[component_index].resize(component.layout.groups().size());
            for (std::size_t old_group = 0;
                 old_group < component.layout.groups().size(); ++old_group)
            {
                const LookupGroupSpec &source = component.layout.groups()[old_group];
                std::size_t target_group = merged_groups.size();
                for (std::size_t candidate = 0; candidate < merged_groups.size(); ++candidate)
                {
                    if (merged_groups[candidate].table_values == source.tableValues() &&
                        merged_groups[candidate].table_outputs == source.tableOutputs() &&
                        merged_groups[candidate].random_tuple_compression == source.usesRandomTupleCompression() &&
                        merged_groups[candidate].padding_value == source.paddingValue())
                    {
                        target_group = candidate;
                        break;
                    }
                }
                if (target_group == merged_groups.size())
                {
                    merged_groups.emplace_back(
                        source.tableId(), source.tableValues(), source.tableOutputs(),
                        source.usesRandomTupleCompression(), source.paddingValue());
                }
                MergedGroup &target = merged_groups[target_group];
                auto &source_queries = component.group_queries[old_group];
                if (source_queries.size() != source.paddedCount())
                {
                    throw std::invalid_argument("lookup witness query vector has wrong padded size");
                }
                const std::size_t query_base = target.queries.size();
                if (target.queries.empty())
                    target.queries = take_queries(source_queries, source.actualCount());
                else
                    target.queries.insert(target.queries.end(), source_queries.begin(),
                                          source_queries.begin() + source.actualCount());
                if (source.usesRandomTupleCompression())
                {
                    if (component.group_query_outputs.empty() ||
                        component.group_query_outputs[old_group].size() != source.paddedCount())
                    {
                        throw std::invalid_argument(
                            "random tuple lookup witness output vector has wrong padded size");
                    }
                    auto &source_outputs = component.group_query_outputs[old_group];
                    target.query_outputs.insert(
                        target.query_outputs.end(), source_outputs.begin(),
                        source_outputs.begin() + source.actualCount());
                }
                else if (!component.group_query_outputs.empty() &&
                         !component.group_query_outputs[old_group].empty())
                {
                    throw std::invalid_argument(
                        "scalar lookup group unexpectedly has tuple outputs");
                }
                remap[component_index][old_group] = {target_group, query_base};
            }
        }

        LookupLayoutBuilder builder(std::move(id), input_domain_size);
        std::vector<std::vector<ll>> queries;
        std::vector<std::vector<ll>> query_outputs;
        queries.reserve(merged_groups.size());
        query_outputs.reserve(merged_groups.size());
        for (std::size_t group = 0; group < merged_groups.size(); ++group)
        {
            MergedGroup &source = merged_groups[group];
            const std::size_t actual_count = source.queries.size();
            const std::string group_id = "lookup/group/" + std::to_string(group);
            builder.addGroup(source.random_tuple_compression
                                 ? LookupGroupSpec::randomTuple(
                                       group_id, source.table_id, source.table_values,
                                       source.table_outputs, actual_count, source.padding_value)
                                 : LookupGroupSpec(
                                       group_id, source.table_id, source.table_values,
                                       actual_count, source.padding_value));
            source.queries.resize(
                zkagent_integer::next_power_of_two(
                    actual_count, "lookup query padding overflows size_t"),
                source.padding_value);
            queries.push_back(std::move(source.queries));
            if (source.random_tuple_compression)
            {
                const auto padding = std::find(
                    source.table_values.begin(), source.table_values.end(),
                    source.padding_value);
                if (padding == source.table_values.end())
                {
                    throw std::logic_error("tuple lookup padding input is absent");
                }
                const std::size_t padding_index = static_cast<std::size_t>(
                    padding - source.table_values.begin());
                source.query_outputs.resize(
                    zkagent_integer::next_power_of_two(
                        actual_count, "lookup query padding overflows size_t"),
                    source.table_outputs[padding_index]);
                query_outputs.push_back(std::move(source.query_outputs));
            }
            else
            {
                query_outputs.emplace_back();
            }
        }

        for (std::size_t component_index = 0;
             component_index < components.size(); ++component_index)
        {
            const LookupWitness &component = components[component_index];
            for (const LookupAffineBinding &old_binding :
                 component.layout.affineBindings())
            {
                LookupAffineBinding binding = old_binding;
                const auto target = remap[component_index][binding.group_index];
                binding.group_index = target.first;
                binding.query_base += target.second;
                builder.addAffineBinding(std::move(binding));
            }
            for (const LookupReconstructionBinding &old_binding :
                 component.layout.reconstructionBindings())
            {
                LookupReconstructionBinding binding = old_binding;
                for (LookupReconstructionTerm &term : binding.lookup_terms)
                {
                    const auto target = remap[component_index][term.group_index];
                    term.group_index = target.first;
                    term.query_base += target.second;
                }
                builder.addReconstructionBinding(std::move(binding));
            }
        }
        return {std::move(builder).finalize(), std::move(queries),
                std::move(query_outputs)};
    }

}

LookupWitness mergeLookupWitnesses(std::string id, const std::vector<LookupWitness> &components)
{
    return merge_lookup_witnesses(std::move(id), components);
}
LookupWitness mergeLookupWitnesses(std::string id, std::vector<LookupWitness> &&components)
{
    return merge_lookup_witnesses(std::move(id), components);
}

namespace
{

    using zkagent_integer::fr_to_i128;

    ll limb_mask(unsigned bits)
    {
        return (static_cast<ll>(1) << bits) - 1;
    }

    std::vector<ll> identity_table(unsigned bits)
    {
        const std::size_t size = std::size_t(1) << bits;
        std::vector<ll> table(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            table[i] = static_cast<ll>(i);
        }
        return table;
    }

}

RangeLookupWitness buildRangeLookupWitness(
    const std::vector<RangeQuerySpec> &specs,
    const std::vector<Fr> &input_layer,
    std::size_t input_domain_size,
    unsigned full_limb_bits, int threads)
{
    if (specs.empty())
    {
        throw std::invalid_argument(
            "range lookup requires at least one query specification");
    }
    if (full_limb_bits == 0 || full_limb_bits > 20)
    {
        throw std::invalid_argument(
            "range lookup limb width must be in [1,20]");
    }
    if (input_domain_size < input_layer.size())
    {
        throw std::invalid_argument(
            "range lookup input domain is smaller than its value vector");
    }
    if (threads <= 0)
    {
        throw std::invalid_argument("range lookup requires positive threads");
    }

    struct Limb
    {
        std::size_t group, base;
        unsigned consumed, bits;
    };
    struct Plan
    {
        std::vector<Limb> limbs;
    };
    struct Group
    {
        unsigned bits;
        std::size_t count;
    };

    std::vector<Plan> plans(specs.size());
    std::vector<LookupReconstructionBinding> bindings(specs.size());
    std::vector<Group> groups;
    std::array<int, 21> by_width;
    by_width.fill(-1);
    for (std::size_t index = 0; index < specs.size(); ++index)
    {
        const auto &spec = specs[index];
        spec.validateInputDomain(input_layer.size());
        const std::size_t count = spec.valueCount();
        auto &binding = bindings[index];
        binding.id = spec.id() + "/reconstruction";
        binding.input_offsets.reserve(count);
        for (const auto &span : spec.spans())
        {
            for (std::size_t item = 0; item < span.count; ++item)
            {
                binding.input_offsets.push_back(
                    span.offset + item * span.stride);
            }
        }

        const unsigned limb_count =
            (spec.bitWidth() + full_limb_bits - 1) / full_limb_bits;
        for (unsigned limb = 0; limb < limb_count; ++limb)
        {
            const unsigned consumed = limb * full_limb_bits;
            const unsigned bits =
                std::min(full_limb_bits, spec.bitWidth() - consumed);
            if (by_width[bits] < 0)
            {
                by_width[bits] = static_cast<int>(groups.size());
                groups.push_back({bits, 0});
            }
            const std::size_t group =
                static_cast<std::size_t>(by_width[bits]);
            const std::size_t base = groups[group].count;
            if (count > std::numeric_limits<std::size_t>::max() - base)
            {
                throw std::length_error("range lookup bucket size overflow");
            }
            groups[group].count += count;
            plans[index].limbs.push_back({group, base, consumed, bits});
            binding.lookup_terms.push_back(
                {group, base, -(static_cast<ll>(1) << consumed)});
        }
    }

    LookupLayoutBuilder builder("range-layout", input_domain_size);
    std::vector<std::vector<ll>> queries(groups.size());
    for (std::size_t i = 0; i < groups.size(); ++i)
    {
        const auto &group = groups[i];
        builder.addGroup(LookupGroupSpec(
            "lookup/group/" + std::to_string(i),
            "identity/" + std::to_string(group.bits),
            identity_table(group.bits), group.count, 0));
        queries[i].assign(
            zkagent_integer::next_power_of_two(group.count), 0);
    }

    struct Tile
    {
        std::size_t spec, begin, end;
    };
    std::vector<Tile> tiles;
    constexpr std::size_t chunk = 8192;
    for (std::size_t spec = 0; spec < specs.size(); ++spec)
    {
        for (std::size_t begin = 0;
             begin < bindings[spec].input_offsets.size(); begin += chunk)
        {
            tiles.push_back(
                {spec, begin,
                 std::min(begin + chunk,
                          bindings[spec].input_offsets.size())});
        }
    }

    std::vector<std::exception_ptr> errors(tiles.size());
#pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t job = 0; job < tiles.size(); ++job)
    {
        try
        {
            const auto &tile = tiles[job];
            const auto &spec = specs[tile.spec];
            const auto &offsets = bindings[tile.spec].input_offsets;
            const ll limit = static_cast<ll>(1) << spec.bitWidth();
            for (std::size_t i = tile.begin; i < tile.end; ++i)
            {
                const ll value = fr_to_i128(input_layer[offsets[i]]);
                if (value < 0 || value >= limit)
                {
                    throw std::out_of_range(
                        "range-query witness violates its immutable bound: " +
                        spec.id());
                }
                for (const auto &limb : plans[tile.spec].limbs)
                {
                    queries[limb.group][limb.base + i] =
                        (value >> limb.consumed) & limb_mask(limb.bits);
                }
            }
        }
        catch (...)
        {
            errors[job] = std::current_exception();
        }
    }
    for (const auto &error : errors)
    {
        if (error)
        {
            std::rethrow_exception(error);
        }
    }

    for (auto &binding : bindings)
    {
        builder.addReconstructionBinding(std::move(binding));
    }
    return {std::move(builder).finalize(), std::move(queries)};
}
