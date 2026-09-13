#ifndef ZKAGENT_INPUT_ROUTING_COMPILED_HPP
#define ZKAGENT_INPUT_ROUTING_COMPILED_HPP

#include "gkr/circuit.h"
#include "common/compiled_mle.hpp"

#include <cstddef>
#include <array>
#include <string>
#include <vector>

struct CompiledInputRoutingStats
{
    std::size_t layers = 0;
    std::size_t route_polynomials = 0;
    std::size_t source_entries = 0;
    std::size_t nonterminal_nodes = 0;
    std::size_t terminal_values = 0;
    std::size_t max_workspace_elements = 0;
    std::size_t logical_storage_bytes = 0;
};

struct CompiledInputRouteTiming
{
    double encoding_seconds = 0;
    compiled_mle::BuildTiming polynomial;
};

class CompiledInputRoutingPlan
{
public:
    CompiledInputRoutingPlan() = default;
    CompiledInputRoutingPlan(const CompiledInputRoutingPlan &) = delete;
    CompiledInputRoutingPlan &operator=(const CompiledInputRoutingPlan &) = delete;
    CompiledInputRoutingPlan(CompiledInputRoutingPlan &&) noexcept = default;
    CompiledInputRoutingPlan &operator=(CompiledInputRoutingPlan &&) noexcept = default;

    static CompiledInputRoutingPlan build(
        const layeredCircuit &circuit,
        int threads = 1,
        std::vector<std::array<CompiledInputRouteTiming, 2>> *timings = nullptr);

    void saveToFile(const std::string &path) const;
    static CompiledInputRoutingPlan loadFromFile(
        const std::string &path,
        const layeredCircuit &circuit);

    Fr evaluate(
        const std::vector<Fr> &input_point,
        const std::vector<std::vector<Fr>> &layer_u_points,
        const std::vector<std::vector<Fr>> &layer_v_points,
        const std::vector<Fr> &sigma_u,
        const std::vector<Fr> &sigma_v,
        int threads,
        std::vector<Fr> &workspace) const;

    const CompiledInputRoutingStats &stats() const { return stats_; }
    std::size_t layerCount() const { return layers_.size(); }
    int inputBits() const { return input_bits_; }

private:
    struct RoutePlan
    {
        bool enabled = false;
        int local_bits = 0;
        std::size_t source_entries = 0;
        compiled_mle::Polynomial polynomial;
    };

    struct LayerPlan
    {
        RoutePlan u;
        RoutePlan v;
    };

    int input_bits_ = 0;
    std::vector<LayerPlan> layers_;
    CompiledInputRoutingStats stats_;
};

#endif
