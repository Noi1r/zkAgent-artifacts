#include "gkr/input_reduction_link.hpp"
#include "common/mle.hpp"
#include "common/field_integer.hpp"
#include "lookup/lookup_input_routing_compiled.hpp"
#include "gkr/input_tensor_view.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace
{

    using zkagent_integer::fr_from_i128;
    using mle::equality_weights;

    void add_scaled(Fr &target, const Fr &weight, ll coefficient)
    {
        if (coefficient == 1)
        {
            target += weight;
        }
        else if (coefficient == -1)
        {
            target -= weight;
        }
        else if (coefficient != 0)
        {
            target += weight * fr_from_i128(coefficient);
        }
    }

    Fr input_equality_at(const std::vector<Fr> &point, std::size_t index)
    {
        return mle::equality_at(index, point);
    }

    Fr affine_relation_evaluation(
        const LookupAffineBinding &binding,
        const InputAffineRelationClaim &relation,
        const std::vector<Fr> &input_point,
        int requested_threads)
    {
        const std::vector<Fr> weights = equality_weights(
            relation.point, requested_threads, relation.rho);
        if (binding.input_offsets.size() > weights.size())
        {
            throw std::invalid_argument(
                "affine input-link relation has the wrong point dimension");
        }
        const int threads = std::max(
            1, std::min<int>(requested_threads,
                             static_cast<int>(binding.input_offsets.size())));
        std::vector<Fr> partial(static_cast<std::size_t>(threads), Fr(0));
        const Fr &input_coefficient = relation.input_coefficient;
        const Fr &output_coefficient = relation.output_coefficient;
#pragma omp parallel num_threads(threads) if (binding.input_offsets.size() >= (std::size_t(1) << 16))
        {
            const int thread = omp_get_thread_num();
            Fr subtotal = Fr(0);
#pragma omp for schedule(static)
            for (long long raw = 0;
                 raw < static_cast<long long>(binding.input_offsets.size());
                 ++raw)
            {
                const std::size_t row = static_cast<std::size_t>(raw);
                subtotal += weights[row] * input_coefficient * input_equality_at(input_point, binding.input_offsets[row]);
                if (!binding.output_offsets.empty())
                {
                    subtotal += weights[row] * output_coefficient * input_equality_at(input_point, binding.output_offsets[row]);
                }
            }
            partial[static_cast<std::size_t>(thread)] = subtotal;
        }
        Fr result = Fr(0);
        for (const Fr &subtotal : partial)
            result += subtotal;
        return result;
    }

    Fr reconstruction_relation_evaluation(
        const LookupReconstructionBinding &binding,
        const InputReconstructionRelationClaim &relation,
        const std::vector<Fr> &input_point,
        int requested_threads)
    {
        const std::vector<Fr> weights = equality_weights(
            relation.point, requested_threads, relation.rho);
        if (binding.input_offsets.size() > weights.size())
        {
            throw std::invalid_argument(
                "reconstruction input-link relation has the wrong point dimension");
        }
        const int threads = std::max(
            1, std::min<int>(requested_threads,
                             static_cast<int>(binding.input_offsets.size())));
        std::vector<Fr> partial(static_cast<std::size_t>(threads), Fr(0));
        const Fr coefficient = fr_from_i128(binding.input_coefficient);
#pragma omp parallel num_threads(threads) if (binding.input_offsets.size() >= (std::size_t(1) << 16))
        {
            const int thread = omp_get_thread_num();
            Fr subtotal = Fr(0);
#pragma omp for schedule(static)
            for (long long raw = 0;
                 raw < static_cast<long long>(binding.input_offsets.size());
                 ++raw)
            {
                const std::size_t row = static_cast<std::size_t>(raw);
                subtotal += weights[row] * coefficient * input_equality_at(input_point, binding.input_offsets[row]);
            }
            partial[static_cast<std::size_t>(thread)] = subtotal;
        }
        Fr result = Fr(0);
        for (const Fr &subtotal : partial)
            result += subtotal;
        return result;
    }

}

namespace
{

    void mark_active(
        std::size_t index,
        std::vector<std::uint64_t> *active_bits,
        std::vector<u32> *active_indices)
    {
        if (active_bits == nullptr || active_indices == nullptr)
            return;
        if (index > std::numeric_limits<u32>::max() ||
            active_bits->size() <= (index >> 6))
        {
            throw std::out_of_range(
                "input-link sparse support lies outside its domain");
        }
        const std::uint64_t mask =
            std::uint64_t(1) << (index & 63);
        std::uint64_t &word = (*active_bits)[index >> 6];
        if ((word & mask) == 0)
        {
            word |= mask;
            active_indices->push_back(static_cast<u32>(index));
        }
    }

    void add_link_coefficients(
        const InputReductionLink &link,
        std::vector<Fr> &coefficients,
        std::vector<std::uint64_t> *active_bits,
        std::vector<u32> *active_indices,
        int requested_threads)
    {
        if (coefficients.size() != link.input_domain_size)
        {
            throw std::invalid_argument(
                "input-link coefficient domain does not match GKR input");
        }
        if ((active_bits == nullptr) != (active_indices == nullptr) ||
            (active_bits != nullptr &&
             active_bits->size() < (coefficients.size() + 63) / 64))
        {
            throw std::invalid_argument(
                "input-link sparse support has the wrong domain");
        }
        for (const InputReductionCoefficientTerm &term : link.terms)
        {
            if (term.index >= coefficients.size())
            {
                throw std::out_of_range(
                    "input-link coefficient lies outside the GKR input domain");
            }
            mark_active(term.index, active_bits, active_indices);
            coefficients[term.index] += term.coefficient;
        }
        for (const InputPrefixEvaluationClaim &claim : link.prefix_evaluations)
        {
            if (claim.point.size() >= sizeof(std::size_t) * 8 ||
                (std::size_t(1) << claim.point.size()) > coefficients.size())
            {
                throw std::invalid_argument("input prefix claim exceeds its input domain");
            }
            const auto weights = equality_weights(claim.point, requested_threads, claim.weight);
            for (std::size_t index = 0; index < weights.size(); ++index)
            {
                mark_active(index, active_bits, active_indices);
                coefficients[index] += weights[index];
            }
        }
        for (const auto &claim : link.tensor_evaluations)
        {
            if (!claim.view || claim.view->spec().inputDomain() != link.input_domain_size ||
                claim.point.size() != std::size_t(claim.view->spec().localBits()))
                throw std::invalid_argument("invalid input tensor claim");
            const auto &spec = claim.view->spec();
            const auto split = claim.point.begin() + spec.columnBits();
            const auto columns = equality_weights(std::vector<Fr>(claim.point.begin(), split), requested_threads);
            const auto rows = equality_weights(std::vector<Fr>(split, claim.point.end()), requested_threads, claim.weight);
            for (std::size_t t = 0; t < spec.rows; ++t)
                for (std::size_t j = 0; j < spec.columns; ++j)
                {
                    const std::size_t index = spec.offset + t * spec.row_stride + j;
                    mark_active(index, active_bits, active_indices);
                    coefficients[index] += rows[t] * columns[j];
                }
        }
        if (link.layout == nullptr)
            return;

        for (const InputAffineRelationClaim &relation : link.affine_relations)
        {
            const LookupAffineBinding &binding =
                link.layout->affineBindings().at(relation.binding_index);
            const std::vector<Fr> weights = equality_weights(
                relation.point, requested_threads, relation.rho);
            if (binding.input_offsets.size() > weights.size())
            {
                throw std::invalid_argument(
                    "affine input-link relation has the wrong point dimension");
            }
            for (std::size_t row = 0;
                 row < binding.input_offsets.size(); ++row)
            {
                mark_active(
                    binding.input_offsets[row], active_bits, active_indices);
                coefficients.at(binding.input_offsets[row]) +=
                    weights[row] * relation.input_coefficient;
                if (!binding.output_offsets.empty())
                {
                    mark_active(
                        binding.output_offsets[row], active_bits, active_indices);
                    coefficients.at(binding.output_offsets[row]) +=
                        weights[row] * relation.output_coefficient;
                }
            }
        }

        for (const InputReconstructionRelationClaim &relation :
             link.reconstruction_relations)
        {
            const LookupReconstructionBinding &binding =
                link.layout->reconstructionBindings().at(
                    relation.binding_index);
            const std::vector<Fr> weights = equality_weights(
                relation.point, requested_threads, relation.rho);
            if (binding.input_offsets.size() > weights.size())
            {
                throw std::invalid_argument(
                    "reconstruction input-link relation has the wrong point dimension");
            }
            for (std::size_t row = 0;
                 row < binding.input_offsets.size(); ++row)
            {
                mark_active(
                    binding.input_offsets[row], active_bits, active_indices);
                add_scaled(coefficients.at(binding.input_offsets[row]),
                           weights[row], binding.input_coefficient);
            }
        }
    }

}

void InputReductionLink::addTo(
    std::vector<Fr> &coefficients, int requested_threads) const
{
    add_link_coefficients(
        *this, coefficients, nullptr, nullptr, requested_threads);
}

void InputReductionLink::addTo(
    std::vector<Fr> &coefficients,
    std::vector<std::uint64_t> &active_bits,
    std::vector<u32> &active_indices,
    int requested_threads) const
{
    add_link_coefficients(
        *this, coefficients, &active_bits, &active_indices,
        requested_threads);
}

Fr InputReductionLink::coefficientEvaluationAtPoint(
    const std::vector<Fr> &input_point,
    int requested_threads,
    std::vector<Fr> *compiled_workspace) const
{
    if (input_point.size() >= sizeof(std::size_t) * 8 ||
        (std::size_t(1) << input_point.size()) != input_domain_size)
    {
        throw std::invalid_argument(
            "input-link point has the wrong domain size");
    }
    Fr result = Fr(0);
    for (const InputPrefixEvaluationClaim &claim : prefix_evaluations)
    {
        if (claim.point.size() > input_point.size())
        {
            throw std::invalid_argument("input prefix has too many evaluation coordinates");
        }
        Fr coefficient = claim.weight;
        for (std::size_t bit = 0; bit < input_point.size(); ++bit)
        {
            coefficient *= bit < claim.point.size()
                               ? (claim.point[bit] * input_point[bit] +
                                  (Fr(1) - claim.point[bit]) * (Fr(1) - input_point[bit]))
                               : Fr(1) - input_point[bit];
        }
        result += coefficient;
    }
    for (const InputReductionCoefficientTerm &term : terms)
    {
        if (term.index >= input_domain_size)
        {
            throw std::out_of_range(
                "input-link coefficient lies outside the GKR input domain");
        }
        result += term.coefficient * input_equality_at(
                                         input_point, term.index);
    }
    for (const auto &claim : tensor_evaluations)
    {
        if (!claim.view || claim.view->spec().inputDomain() != input_domain_size)
            throw std::invalid_argument("invalid input tensor routing source");
        std::vector<Fr> local_workspace;
        auto *workspace = compiled_workspace;
        if (workspace == nullptr)
        {
            local_workspace.resize(claim.view->workspaceElements());
            workspace = &local_workspace;
        }
        result += claim.weight * claim.view->evaluate(input_point, claim.point, requested_threads, *workspace);
    }
    if (layout == nullptr)
        return result;
    if (compiled_routing_plan != nullptr)
    {
        if (compiled_workspace == nullptr ||
            compiled_routing_plan->inputDomainSize() != input_domain_size)
        {
            throw std::invalid_argument(
                "compiled input-link routing is missing its workspace");
        }
        result += compiled_routing_plan->evaluate(
            input_point, affine_relations, reconstruction_relations,
            requested_threads, *compiled_workspace);
        return result;
    }
    for (const InputAffineRelationClaim &relation : affine_relations)
    {
        result += affine_relation_evaluation(
            layout->affineBindings().at(relation.binding_index),
            relation, input_point, requested_threads);
    }
    for (const InputReconstructionRelationClaim &relation :
         reconstruction_relations)
    {
        result += reconstruction_relation_evaluation(
            layout->reconstructionBindings().at(relation.binding_index),
            relation, input_point, requested_threads);
    }
    return result;
}

void InputReductionLink::addPrefixEvaluation(
    const std::vector<Fr> &point, const Fr &evaluation, const Fr &weight)
{
    if (input_domain_size == 0 || (input_domain_size & (input_domain_size - 1)) != 0 ||
        point.size() >= sizeof(std::size_t) * 8 ||
        (std::size_t(1) << point.size()) > input_domain_size)
    {
        throw std::invalid_argument("invalid aligned input-prefix claim");
    }
    prefix_evaluations.push_back({point, weight});
    claimed_evaluation += weight * evaluation;
    ++relation_claims;
}

void InputReductionLink::appendEvaluationClaims(const InputReductionLink &other)
{
    if (other.input_domain_size != input_domain_size || other.layout != nullptr ||
        !other.terms.empty() || !other.affine_relations.empty() ||
        !other.reconstruction_relations.empty())
    {
        throw std::invalid_argument("only same-domain evaluation claims can be appended here");
    }
    prefix_evaluations.insert(prefix_evaluations.end(),
                              other.prefix_evaluations.begin(), other.prefix_evaluations.end());
    tensor_evaluations.insert(tensor_evaluations.end(), other.tensor_evaluations.begin(), other.tensor_evaluations.end());
    claimed_evaluation += other.claimed_evaluation;
    relation_claims += other.relation_claims;
}

void InputReductionLink::addTensorEvaluation(const CompiledInputTensorView &view,
                                             const std::vector<Fr> &point, const Fr &value, const Fr &weight)
{
    if (view.spec().inputDomain() != input_domain_size || point.size() != std::size_t(view.spec().localBits()))
        throw std::invalid_argument("incompatible input tensor evaluation");
    tensor_evaluations.push_back({&view, point, weight});
    claimed_evaluation += weight * value;
    ++relation_claims;
}
std::size_t InputReductionLink::compiledWorkspaceElements() const
{
    std::size_t count = compiled_routing_plan ? compiled_routing_plan->stats().max_workspace_elements : 0;
    for (const auto &claim : tensor_evaluations)
    {
        if (!claim.view)
            throw std::invalid_argument("missing input tensor routing plan");
        count = std::max(count, claim.view->workspaceElements());
    }
    return count;
}
