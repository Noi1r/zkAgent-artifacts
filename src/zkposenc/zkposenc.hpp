#ifndef ZKAGENT_ZKPOSENC_HPP
#define ZKAGENT_ZKPOSENC_HPP

#include <string>

namespace zkposenc {

constexpr int ZkPosEncWarmupRuns = 1;
constexpr int ZkPosEncMeasuredRuns = 10;
constexpr int ZkPosEncTotalRuns =
    ZkPosEncWarmupRuns + ZkPosEncMeasuredRuns;

struct ZkPosEncBenchmarkResult {
    std::string scheme_name;
    int seq_len = 0;
    int head_dim = 0;
    int heads = 0;

    int sumcheck_rounds = 0;

    double preparation_ms = 0;
    double prover_ms = 0;
    double verifier_ms = 0;
    double proof_size_kb = 0;

    void print() const;
};

ZkPosEncBenchmarkResult benchmarkThalerRoPE(
    int seq_len, int head_dim, int heads, int threads);
ZkPosEncBenchmarkResult benchmarkHadamardRoPE(
    int seq_len, int head_dim, int heads, int threads);

}

#endif
