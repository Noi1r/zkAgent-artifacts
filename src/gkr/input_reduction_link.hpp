#ifndef ZKAGENT_INPUT_REDUCTION_LINK_HPP
#define ZKAGENT_INPUT_REDUCTION_LINK_HPP

#include "lookup/lookup_witness.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

class CompiledLookupInputRoutingPlan;
class CompiledInputTensorView;

struct InputReductionCoefficientTerm
{
    std::size_t index = 0;
    Fr coefficient = Fr(0);
};

struct InputAffineRelationClaim
{
    std::size_t binding_index = 0;
    std::vector<Fr> point;
    Fr rho = Fr(0);
    Fr input_coefficient = Fr(1);
    Fr output_coefficient = Fr(0);
};

struct InputReconstructionRelationClaim
{
    std::size_t binding_index = 0;
    std::vector<Fr> point;
    Fr rho = Fr(0);
};

struct InputPrefixEvaluationClaim
{
    std::vector<Fr> point;
    Fr weight = Fr(0);
};

struct InputTensorEvaluationClaim
{
    const CompiledInputTensorView *view = nullptr;
    std::vector<Fr> point;
    Fr weight = Fr(0);
};

struct InputReductionLink
{
    std::vector<InputReductionCoefficientTerm> terms;
    const LookupLayout *layout = nullptr;
    const CompiledLookupInputRoutingPlan *compiled_routing_plan = nullptr;
    std::vector<InputAffineRelationClaim> affine_relations;
    std::vector<InputReconstructionRelationClaim> reconstruction_relations;
    std::vector<InputPrefixEvaluationClaim> prefix_evaluations;
    std::vector<InputTensorEvaluationClaim> tensor_evaluations;
    std::size_t input_domain_size = 0;
    Fr claimed_evaluation = Fr(0);
    std::size_t query_claims = 0;
    std::size_t relation_claims = 0;

    std::size_t domainSize() const { return input_domain_size; }
    void addPrefixEvaluation(const std::vector<Fr> &point,
                             const Fr &evaluation, const Fr &weight);
    void addTensorEvaluation(const CompiledInputTensorView &view,
                             const std::vector<Fr> &point,
                             const Fr &evaluation, const Fr &weight);
    void appendEvaluationClaims(const InputReductionLink &other);
    std::size_t compiledWorkspaceElements() const;
    void addTo(std::vector<Fr> &coefficients, int threads = 1) const;
    void addTo(
        std::vector<Fr> &coefficients,
        std::vector<std::uint64_t> &active_bits,
        std::vector<u32> &active_indices,
        int threads = 1) const;
    Fr coefficientEvaluationAtPoint(
        const std::vector<Fr> &input_point,
        int threads = 1,
        std::vector<Fr> *compiled_workspace = nullptr) const;
};

#endif
