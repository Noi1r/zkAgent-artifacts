#include "zkembed/zkembed.hpp"
#include "common/omp_threads.hpp"
#include "benchmark_output.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

using namespace mcl::bn;
using namespace std;
using namespace std::chrono;

namespace {

constexpr int Gpt2VocabSize = 50257;
constexpr int Gpt2EmbedDim = 768;

int env_int(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    return (raw && *raw) ? std::atoi(raw) : fallback;
}

bool file_exists(const string& path) {
    ifstream file(path, ios::binary);
    return file.good();
}

string embedding_path() {
    const char* configured = std::getenv("GPT2_EMBEDDING_PATH");
    if (configured && *configured) return configured;
    return "./model/gpt2-124m/quantized/tensors/embedding_token_i32.bin";
}

vector<ll> load_or_generate_embedding(int vocab_size, int embed_dim,
                                      string& provenance) {
    const string path = embedding_path();
    if (vocab_size == Gpt2VocabSize && embed_dim == Gpt2EmbedDim &&
        file_exists(path)) {
        ifstream file(path, ios::binary);
        const size_t count = static_cast<size_t>(vocab_size) * embed_dim;
        vector<int32_t> encoded(count);
        file.read(reinterpret_cast<char*>(encoded.data()),
                  static_cast<std::streamsize>(count * sizeof(int32_t)));
        if (!file || file.peek() != std::ifstream::traits_type::eof()) {
            throw runtime_error("GPT-2 int32 embedding file has an unexpected size");
        }
        vector<ll> result(count);
        for (size_t i = 0; i < count; i++) result[i] = encoded[i];
        provenance = "GPT-2 quantized table: " + path;
        return result;
    }

    mt19937_64 rng(42);
    uniform_int_distribution<int64_t> value_dist(0, 65536);
    vector<ll> result(static_cast<size_t>(vocab_size) * embed_dim);
    for (ll& value : result) value = value_dist(rng);
    provenance = "deterministic synthetic table (shape-only benchmark)";
    return result;
}

struct Result {
    int query_size = 0;
    double prove_ms = 0;
    double verify_us = 0;
    double compute_ms = 0;
    double proof_bytes = 0;
};

}

int main(int argc, char** argv) {
    initPairing(mcl::BN_SNARK1);

    int vocab_size = Gpt2VocabSize;
    int embed_dim = Gpt2EmbedDim;
    int threads = env_int("THREADS", env_int("OMP_NUM_THREADS", 32));
    int start_exp = env_int("START_EXP", 5);
    int end_exp = env_int("END_EXP", 10);
    int warmups = env_int("WARMUPS", 1);
    int runs = env_int("RUNS", 10);

    vector<int> positional;
    for (int i = 1; i < argc; i++) {
        const string argument = argv[i];
        if (argument == "--start-exp" && i + 1 < argc) {
            start_exp = std::atoi(argv[++i]);
        } else if (argument == "--end-exp" && i + 1 < argc) {
            end_exp = std::atoi(argv[++i]);
        } else {
            positional.push_back(std::atoi(argv[i]));
        }
    }
    if (!positional.empty()) vocab_size = positional[0];
    if (positional.size() > 1) embed_dim = positional[1];
    if (positional.size() > 2) threads = positional[2];
    if (vocab_size <= 0 || embed_dim <= 0 || threads <= 0 || runs <= 0 ||
        warmups < 0 || start_exp < 0 || end_exp < start_exp || end_exp > 30) {
        cerr << "Invalid benchmark parameters" << endl;
        return 2;
    }
    const ScopedOmpThreads thread_scope(threads);

    string provenance;
    vector<ll> embedding =
        load_or_generate_embedding(vocab_size, embed_dim, provenance);

    const int table_size = ZkEmbed::requiredTableSize(vocab_size);
    int column_variables = 0;
    while ((std::size_t(1) << column_variables) < static_cast<std::size_t>(embed_dim)) {
        ++column_variables;
    }
    const int max_output_variables = end_exp + column_variables;
    if (max_output_variables > 30) {
        cerr << "Requested output tensor exceeds the supported domain" << endl;
        return 2;
    }

    const string report_path = benchmark_output::reportPath("table2");
    auto srs_start = high_resolution_clock::now();
    SRSPlan srs_plan;
    srs_plan.mv_vars = (max_output_variables + 1) / 2;
    // Use the same univariate KZG SRS as the embedding-table preprocessing cache.
    srs_plan.uni_size = table_size;
    SRSBundle srs;
    srs.setup_cached(srs_plan, "cache");
    auto srs_end = high_resolution_clock::now();

    ZkEmbed zkembed(
        srs.uni.pk, srs.uni.vk, srs.mv.pk, srs.mv.vk,
        vocab_size, embed_dim, threads);
    auto preprocess_start = high_resolution_clock::now();
    zkembed.preprocess(embedding.data());
    auto preprocess_end = high_resolution_clock::now();
    zkembed.preloadVerifierData();

    cout << "zkEmbed online benchmark\n"
         << "  table: " << vocab_size << " x " << embed_dim << '\n'
         << "  padded table domain: " << table_size << '\n'
         << "  data: " << provenance << '\n'
         << "  prover storage: query rows preloaded before each batch\n"
         << "  threads: " << threads << ", warmups: " << warmups
         << ", measured runs: " << runs << '\n';

    vector<Result> results;
    mt19937_64 rng(7);
    uniform_int_distribution<int> index_dist(0, vocab_size - 1);

    for (int exponent = start_exp; exponent <= end_exp; exponent++) {
        const int query_size = 1 << exponent;
        vector<ll> queries(query_size);
        vector<ll> outputs(static_cast<size_t>(query_size) * embed_dim);
        for (int i = 0; i < query_size; i++) {
            queries[i] = index_dist(rng);
            const size_t source = static_cast<size_t>(queries[i]) * embed_dim;
            const size_t destination = static_cast<size_t>(i) * embed_dim;
            std::copy_n(embedding.data() + source, embed_dim,
                        outputs.data() + destination);
        }

        Result result;
        result.query_size = query_size;
        zkembed.preloadProverDataForQueries(queries);
        for (int iteration = 0; iteration < warmups + runs; iteration++) {
            const auto h0_values = zkembed.prepareWitness(queries.size(), outputs);
            ZkEmbedTiming prove_timing;
            const auto prove_start = high_resolution_clock::now();
            ZkEmbedStatement statement =
                zkembed.makeStatement(queries, h0_values);
            ZkEmbedProof proof =
                zkembed.prove(statement, h0_values, prove_timing);
            const auto prove_end = high_resolution_clock::now();
            if (prove_timing.lazy_loaded_g1_points != 0 ||
                prove_timing.lazy_load_ms != 0) {
                throw runtime_error("zkEmbed performed cache reads after query preload");
            }

            ZkEmbedTiming verify_timing;
            const auto verify_start = high_resolution_clock::now();
            const bool accepted =
                zkembed.verify(statement, proof, verify_timing);
            const auto verify_end = high_resolution_clock::now();
            if (!accepted) {
                cerr << "zkEmbed verification failed for T=" << query_size << endl;
                return 1;
            }

            if (iteration >= warmups) {
                result.prove_ms += duration<double, milli>(prove_end - prove_start).count();
                result.verify_us += duration<double, micro>(verify_end - verify_start).count();
                result.compute_ms += prove_timing.prove_compute_ms;
                result.proof_bytes += proof.cryptographicPayloadBytes();
            }
        }
        result.prove_ms /= runs;
        result.verify_us /= runs;
        result.compute_ms /= runs;
        result.proof_bytes /= runs;
        results.push_back(result);

        cout << "  T=" << setw(4) << query_size
             << ": prove=" << fixed << setprecision(3) << result.prove_ms
             << " ms, verify=" << result.verify_us << " us, proof="
             << (result.proof_bytes / 1024.0) << " KB" << endl;
    }

    ofstream report(report_path);
    if (!report) throw runtime_error("Cannot open benchmark report: " + report_path);
    report << "# zkEmbed online benchmark\n\n"
           << "- Table: `" << vocab_size << " x " << embed_dim << "`\n"
           << "- Padded table domain: `" << table_size << "`\n"
           << "- Data: " << provenance << "\n"
           << "- Threads: `" << threads << "`\n"
           << "- Online proving is timed after query-dependent preload; no cache reads occur during proving.\n"
           << "- SRS setup: `"
           << duration<double, milli>(srs_end - srs_start).count() << " ms`\n"
           << "- Table preprocessing: `"
           << duration<double, milli>(preprocess_end - preprocess_start).count()
           << " ms`\n\n"
           << "| Queries | Prove (ms) | Verify (us) | Proof (KB) | Prove compute (ms) |\n"
           << "|---:|---:|---:|---:|---:|\n";
    for (const Result& result : results) {
        report << "| " << result.query_size << " | " << fixed << setprecision(3)
               << result.prove_ms << " | " << result.verify_us << " | "
               << (result.proof_bytes / 1024.0) << " | "
               << result.compute_ms << " |\n";
    }
    report.close();
    if (!report) throw runtime_error("Failed to write benchmark report: " + report_path);
    cout << "Results written to " << report_path << endl;
    return 0;
}
