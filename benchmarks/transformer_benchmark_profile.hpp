#ifndef ZKAGENT_TRANSFORMER_BENCHMARK_PROFILE_HPP
#define ZKAGENT_TRANSFORMER_BENCHMARK_PROFILE_HPP

#include "model/gpt2.hpp"

#include <stdexcept>
#include <string>

enum class TransformerBenchmarkProfile {
    Table5ZkGPTStandalone,
    Table6Greedy,
};

struct TransformerBenchmarkProfileSpec {
    const char* name;
    GPT2CircuitProfile circuit_profile;
    bool require_lookup_input_link;
    bool use_compiled_wiring_plan;
    bool use_compiled_input_routing_plan;
};

inline const TransformerBenchmarkProfileSpec& transformerBenchmarkProfileSpec(
    TransformerBenchmarkProfile profile) {
    static const TransformerBenchmarkProfileSpec Table5ZkGPTStandalone{
        "table5-zkgpt-standalone",
        GPT2CircuitProfile::ZkGPTArtifactTransformer,
        false, true, true,
    };
    static const TransformerBenchmarkProfileSpec Table6Greedy{
        "table6-greedy", GPT2CircuitProfile::CompleteInference,
        true, true, true,
    };
    switch (profile) {
        case TransformerBenchmarkProfile::Table5ZkGPTStandalone:
            return Table5ZkGPTStandalone;
        case TransformerBenchmarkProfile::Table6Greedy:
            return Table6Greedy;
    }
    throw std::invalid_argument("unknown Transformer benchmark profile");
}

inline TransformerBenchmarkProfile parseTransformerBenchmarkProfile(
    const std::string& name) {
    if (name == "table5-zkgpt-standalone") {
        return TransformerBenchmarkProfile::Table5ZkGPTStandalone;
    }
    if (name == "table6-greedy") return TransformerBenchmarkProfile::Table6Greedy;
    throw std::invalid_argument("unknown Transformer benchmark profile: " + name);
}

#endif
