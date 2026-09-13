#ifndef ZKAGENT_LOOKUP_WITNESS_HPP
#define ZKAGENT_LOOKUP_WITNESS_HPP

#include "common/global_var.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

struct InputSpan
{
    std::size_t offset = 0;
    std::size_t count = 0;
    std::size_t stride = 1;

    InputSpan() = default;
    InputSpan(std::size_t offset_value, std::size_t count_value,
              std::size_t stride_value = 1)
        : offset(offset_value), count(count_value), stride(stride_value) {}
};

class RangeQuerySpec
{
public:
    static RangeQuerySpec powerOfTwo(
        std::string id, std::vector<InputSpan> spans, unsigned bit_width);

    const std::string &id() const { return id_; }
    const std::vector<InputSpan> &spans() const { return spans_; }
    unsigned bitWidth() const { return bit_width_; }
    std::size_t valueCount() const;
    void validateInputDomain(std::size_t input_domain_size) const;

private:
    RangeQuerySpec(std::string id, std::vector<InputSpan> spans,
                   unsigned bit_width);

    std::string id_;
    std::vector<InputSpan> spans_;
    unsigned bit_width_ = 0;
};

struct LookupReconstructionTerm
{
    std::size_t group_index = 0;
    std::size_t query_base = 0;
    ll coefficient = 0;

    LookupReconstructionTerm() = default;
    LookupReconstructionTerm(std::size_t group, std::size_t base, ll scalar)
        : group_index(group), query_base(base), coefficient(scalar) {}
};

struct LookupAffineBinding
{
    std::string id;
    std::size_t group_index = 0;
    std::size_t query_base = 0;
    std::vector<std::size_t> input_offsets;
    std::vector<std::size_t> output_offsets;
    ll input_coefficient = 1;
    ll output_coefficient = 0;
    ll constant = 0;
    bool random_tuple_compression = false;
};

struct LookupReconstructionBinding
{
    std::string id;
    std::vector<std::size_t> input_offsets;
    ll input_coefficient = 1;
    std::vector<LookupReconstructionTerm> lookup_terms;
    ll constant = 0;
};

class LookupGroupSpec
{
public:
    LookupGroupSpec(std::string id, std::string table_id,
                    std::vector<ll> table_values,
                    std::size_t actual_count, ll padding_value);
    static LookupGroupSpec randomTuple(
        std::string id, std::string table_id,
        std::vector<ll> table_inputs, std::vector<ll> table_outputs,
        std::size_t actual_count, ll padding_input);

    const std::string &id() const { return id_; }
    const std::string &tableId() const { return table_id_; }
    const std::vector<ll> &tableValues() const { return table_values_; }
    bool usesRandomTupleCompression() const
    {
        return random_tuple_compression_;
    }
    const std::vector<ll> &tableOutputs() const { return table_outputs_; }
    std::size_t actualCount() const { return actual_count_; }
    std::size_t paddedCount() const { return padded_count_; }
    ll paddingValue() const { return padding_value_; }

private:
    std::string id_;
    std::string table_id_;
    std::vector<ll> table_values_;
    std::vector<ll> table_outputs_;
    bool random_tuple_compression_ = false;
    std::size_t actual_count_ = 0;
    std::size_t padded_count_ = 0;
    ll padding_value_ = 0;
};

class LookupLayout
{
public:
    LookupLayout(std::string id, std::size_t input_domain_size,
                 std::vector<LookupGroupSpec> groups,
                 std::vector<LookupAffineBinding> affine_bindings,
                 std::vector<LookupReconstructionBinding>
                     reconstruction_bindings);

    const std::string &id() const { return id_; }
    std::size_t inputDomainSize() const { return input_domain_size_; }
    const std::vector<LookupGroupSpec> &groups() const { return groups_; }
    const std::vector<LookupAffineBinding> &affineBindings() const
    {
        return affine_bindings_;
    }
    const std::vector<LookupReconstructionBinding> &
    reconstructionBindings() const
    {
        return reconstruction_bindings_;
    }

private:
    std::string id_;
    std::size_t input_domain_size_ = 0;
    std::vector<LookupGroupSpec> groups_;
    std::vector<LookupAffineBinding> affine_bindings_;
    std::vector<LookupReconstructionBinding> reconstruction_bindings_;
};

class LookupLayoutBuilder
{
public:
    LookupLayoutBuilder(std::string id, std::size_t input_domain_size)
        : id_(std::move(id)), input_domain_size_(input_domain_size) {}

    std::size_t addGroup(LookupGroupSpec group);
    void addAffineBinding(LookupAffineBinding binding);
    void addReconstructionBinding(LookupReconstructionBinding binding);
    LookupLayout finalize() &&;

private:
    std::string id_;
    std::size_t input_domain_size_ = 0;
    std::vector<LookupGroupSpec> groups_;
    std::vector<LookupAffineBinding> affine_bindings_;
    std::vector<LookupReconstructionBinding> reconstruction_bindings_;
};

struct LookupWitness
{
    LookupLayout layout;
    std::vector<std::vector<ll>> group_queries;
    std::vector<std::vector<ll>> group_query_outputs;
};

LookupWitness mergeLookupWitnesses(
    std::string id, const std::vector<LookupWitness> &components);
LookupWitness mergeLookupWitnesses(
    std::string id, std::vector<LookupWitness> &&components);

using RangeLookupWitness = LookupWitness;

RangeLookupWitness buildRangeLookupWitness(
    const std::vector<RangeQuerySpec> &specs,
    const std::vector<Fr> &input_layer,
    std::size_t input_domain_size,
    unsigned full_limb_bits = 10, int threads = 1);

#endif
