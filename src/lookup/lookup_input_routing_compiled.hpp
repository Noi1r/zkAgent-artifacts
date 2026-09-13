#ifndef ZKAGENT_LOOKUP_INPUT_ROUTING_COMPILED_HPP
#define ZKAGENT_LOOKUP_INPUT_ROUTING_COMPILED_HPP

#include "common/compiled_mle.hpp"
#include "gkr/input_reduction_link.hpp"
#include "lookup/lookup_witness.hpp"

#include <cstddef>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct CompiledLookupInputRoutingStats
{
    std::size_t route_polynomials = 0;
    std::size_t source_entries = 0;
    std::size_t nonterminal_nodes = 0;
    std::size_t terminal_values = 0;
    std::size_t max_workspace_elements = 0;
    std::size_t logical_storage_bytes = 0;
};

class CompiledLookupInputRoutingPlan
{
public:
    CompiledLookupInputRoutingPlan() = default;
    CompiledLookupInputRoutingPlan(
        const CompiledLookupInputRoutingPlan &) = delete;
    CompiledLookupInputRoutingPlan &operator=(
        const CompiledLookupInputRoutingPlan &) = delete;
    CompiledLookupInputRoutingPlan(
        CompiledLookupInputRoutingPlan &&) noexcept = default;
    CompiledLookupInputRoutingPlan &operator=(
        CompiledLookupInputRoutingPlan &&) noexcept = default;

    static CompiledLookupInputRoutingPlan build(
        const LookupLayout &layout);

    void saveToFile(const std::string &path) const;
    static CompiledLookupInputRoutingPlan loadFromFile(
        const std::string &path, const LookupLayout &layout);

    Fr evaluate(
        const std::vector<Fr> &input_point,
        const std::vector<InputAffineRelationClaim> &affine_relations,
        const std::vector<InputReconstructionRelationClaim> &
            reconstruction_relations,
        int threads,
        std::vector<Fr> &workspace) const;

    std::size_t inputDomainSize() const { return input_domain_size_; }
    const CompiledLookupInputRoutingStats &stats() const { return stats_; }

private:
    struct AffinePlan
    {
        int local_bits = 0;
        compiled_mle::Polynomial input;
        compiled_mle::Polynomial output;
    };

    struct ReconstructionPlan
    {
        int local_bits = 0;
        Fr input_coefficient = Fr(0);
        compiled_mle::Polynomial input;
    };

    int input_bits_ = 0;
    std::size_t input_domain_size_ = 0;
    std::vector<AffinePlan> affine_;
    std::vector<ReconstructionPlan> reconstruction_;
    CompiledLookupInputRoutingStats stats_;
};

#endif
