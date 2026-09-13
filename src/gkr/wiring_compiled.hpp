#ifndef ZKAGENT_WIRING_COMPILED_HPP
#define ZKAGENT_WIRING_COMPILED_HPP

#include "gkr/circuit.h"
#include "common/compiled_mle.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct CompiledWiringStats
{
    std::size_t layers = 0;
    std::size_t polynomial_groups = 0;
    std::size_t nonterminal_nodes = 0;
    std::size_t terminal_values = 0;
    std::size_t max_workspace_elements = 0;
    std::size_t logical_storage_bytes = 0;
};

struct CompiledWiringLayerTiming
{
    double encoding_seconds = 0;
    std::array<compiled_mle::BuildTiming, 4> unary;
    std::array<compiled_mle::BuildTiming, 6> binary;
};

class CompiledWiringPlan
{
public:
    CompiledWiringPlan() = default;
    CompiledWiringPlan(const CompiledWiringPlan &) = delete;
    CompiledWiringPlan &operator=(const CompiledWiringPlan &) = delete;
    CompiledWiringPlan(CompiledWiringPlan &&) noexcept = default;
    CompiledWiringPlan &operator=(CompiledWiringPlan &&) noexcept = default;

    static CompiledWiringPlan build(
        const layeredCircuit &circuit,
        int threads = 1,
        std::vector<CompiledWiringLayerTiming> *timings = nullptr);

    void saveToFile(const std::string &path) const;
    static CompiledWiringPlan loadFromFile(
        const std::string &path,
        const layeredCircuit &circuit);

    std::array<Fr, 2> evaluatePhase1(
        std::size_t layer_id,
        const Fr &alpha,
        const Fr &beta,
        const Fr &zero_zone_scale,
        const std::vector<Fr> &output_point_0,
        const std::vector<Fr> &output_point_1,
        const std::vector<Fr> &input_point,
        int threads,
        std::vector<Fr> &workspace) const;

    std::array<Fr, 3> evaluatePhase2(
        std::size_t layer_id,
        const Fr &alpha,
        const Fr &beta,
        const Fr &zero_zone_scale,
        const std::vector<Fr> &output_point_0,
        const std::vector<Fr> &output_point_1,
        const std::vector<Fr> &input_point_u,
        const std::vector<Fr> &input_point_v,
        int threads,
        std::vector<Fr> &workspace) const;

    const CompiledWiringStats &stats() const { return stats_; }
    std::size_t layerCount() const { return layers_.size(); }

private:
    struct LayerPlan
    {
        bool enabled = false;
        int output_bits = 0;
        int input_bits_u = 0;
        int input_bits_v = 0;
        std::array<compiled_mle::Polynomial, 4> unary;
        std::array<compiled_mle::Polynomial, 6> binary;
    };

    std::vector<LayerPlan> layers_;
    CompiledWiringStats stats_;
};

#endif
