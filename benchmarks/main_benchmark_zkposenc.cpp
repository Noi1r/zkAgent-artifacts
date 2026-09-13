#include "zkposenc/zkposenc.hpp"
#include "common/global_var.hpp"
#include "benchmark_output.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace mcl::bn;
using namespace zkposenc;

void print_header()
{
    cout << string(100, '=') << endl;
    cout << "RoPE (Rotary Positional Embedding) Benchmark" << endl;
    cout << "Comparing Thaler and Hadamard" << endl;
    cout << string(100, '=') << endl;
}

static bool is_number(const string &s)
{
    if (s.empty())
    {
        return false;
    }
    for (char c : s)
    {
        if (!isdigit(static_cast<unsigned char>(c)))
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    if (argc == 2 || argc > 4)
    {
        cerr << "Usage: " << argv[0] << " [seq_len head_dim [heads]]" << endl;
        return 2;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (!is_number(argv[i]))
        {
            cerr << "Arguments must be positive integers: seq_len head_dim heads" << endl;
            return 2;
        }
    }
    initPairing(mcl::BN_SNARK1);

    print_header();

    constexpr int threads = 1;
    int heads = 12;

    vector<int> seq_lens;
    vector<int> head_dims;

    if (argc >= 3)
    {

        int test_seq = atoi(argv[1]);
        int test_dim = atoi(argv[2]);
        seq_lens.push_back(test_seq);
        head_dims.push_back(test_dim);

        if (argc >= 4)
            heads = atoi(argv[3]);
    }
    else
    {

        seq_lens = {32, 128, 512};
        head_dims = {64, 128};
    }
    if (heads <= 0)
    {
        cerr << "Head count must be positive." << endl;
        return 2;
    }
    for (int seq_len : seq_lens)
    {
        if (seq_len <= 0)
        {
            cerr << "Sequence length must be positive." << endl;
            return 2;
        }
    }
    for (int head_dim : head_dims)
    {
        if (head_dim <= 0 || (head_dim % 2) != 0)
        {
            cerr << "Invalid head dimension " << head_dim
                 << ": RoPE requires a positive even head dimension." << endl;
            return 2;
        }
    }

    const string report_path = benchmark_output::reportPath("table4");
    cout << "\nConfiguration:" << endl;
    cout << "  threads: " << threads << endl;
    cout << "  heads (H): " << heads
         << (argc == 1 ? "; final Llama-2-7B case: 32" : "") << endl;
    cout << "  scope: one Q/K tensor; sumcheck core only" << endl;
    cout << "  execution: prover, then independent verifier replay" << endl;
    cout << "  preparation: input, coefficient, and output construction; reported separately" << endl;
    cout << "  excluded: claim links, commitments, and openings" << endl;
    cout << "  runs: " << ZkPosEncTotalRuns << " (" << ZkPosEncWarmupRuns << " warmup + "
         << ZkPosEncMeasuredRuns << " averaged)" << endl;
    cout << endl;

    vector<ZkPosEncBenchmarkResult> results;
    auto run_case = [&](int seq_len, int head_dim, int case_heads)
    {
        cout << "\nSeq Length: " << seq_len << ", Head Dimension: " << head_dim
             << ", Heads: " << case_heads << endl;
        auto thaler = benchmarkThalerRoPE(seq_len, head_dim, case_heads, threads);
        thaler.print();
        results.push_back(thaler);
        auto hadamard = benchmarkHadamardRoPE(seq_len, head_dim, case_heads, threads);
        hadamard.print();
        results.push_back(hadamard);
        cout << "  Prover ratio (Thaler/Hadamard): "
             << thaler.prover_ms / hadamard.prover_ms << "x" << endl;
    };
    for (int head_dim : head_dims)
    {
        for (int seq_len : seq_lens)
            run_case(seq_len, head_dim, heads);
    }
    if (argc == 1)
    {
        cout << "\nLlama-2-7B RoPE scale (one Q/K tensor)" << endl;
        run_case(512, 128, 32);
    }
    ofstream md(report_path);
    if (!md)
        throw runtime_error("Cannot open RoPE benchmark report: " + report_path);
    md << "# RoPE Benchmark Results\n\n"
       << "Thaler uses stacked matrix-vector sumcheck over T*H instances. "
       << "Hadamard reduces the head dimension and proves the two-product identity "
       << "over T*d elements.\n\n"
       << "Scope: one Q/K tensor; sumcheck core only. Preparation measures input, "
       << "coefficient, and output construction once. Prove and verify measure separate "
       << "complete calls; proving includes challenge-dependent reductions, and verification "
       << "replays the proof with its own Fiat-Shamir transcript.\n\n"
       << "The initial output evaluation is supplied as the core claim. Payload counts "
       << "sumcheck messages and terminal evaluations; claim links, commitments, openings, "
       << "and serialization framing are excluded.\n\n"
       << "Threads: " << threads
       << "; warmup runs: " << ZkPosEncWarmupRuns
       << "; measured runs: " << ZkPosEncMeasuredRuns << ".\n\n"
       << "| Scheme | T | d | H | Rounds | Prepare (ms) | Prove (ms) | Verify (ms) | Payload (KB) |\n"
       << "|---|---|---|---|---|---|---|---|---|\n";
    for (const auto &r : results)
    {
        md << "| " << r.scheme_name << " | " << r.seq_len << " | " << r.head_dim
           << " | " << r.heads
           << " | " << r.sumcheck_rounds << " | " << fixed << setprecision(2)
           << r.preparation_ms << " | " << r.prover_ms << " | " << r.verifier_ms
           << " | " << r.proof_size_kb << " |\n";
    }
    md.close();
    if (!md)
        throw runtime_error("Failed to write RoPE benchmark report: " + report_path);
    cout << "Results written to " << report_path << endl;
    return 0;
}
