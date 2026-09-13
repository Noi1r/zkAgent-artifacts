

#include "zkembed/zkembed.hpp"
#include "common/omp_threads.hpp"
#include "benchmark_output.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sys/stat.h>
#include <vector>

using namespace std;
using namespace mcl::bn;

const int GPT2_SMALL_VOCAB_SIZE = 50257;
const int GPT2_SMALL_EMBED_DIM = 768;

const int GPT2_MEDIUM_VOCAB_SIZE = 50257;
const int GPT2_MEDIUM_EMBED_DIM = 1024;

const int GPT2_XL_VOCAB_SIZE = 50257;
const int GPT2_XL_EMBED_DIM = 1600;

const int GPT3_VOCAB_SIZE = 50257;
const int GPT3_EMBED_DIM = 12288;

const int DEEPSEEK_V2_VOCAB_SIZE = 102400;
const int DEEPSEEK_V2_EMBED_DIM = 5120;

const int DEEPSEEK_V3_VOCAB_SIZE = 129280;
const int DEEPSEEK_V3_EMBED_DIM = 7168;

const int TINYLLAMA_VOCAB_SIZE = 32000;
const int TINYLLAMA_EMBED_DIM = 2048;

struct PreprocessTiming {
    long long srs_ms = 0;
    long long preprocess_ms = 0;
    long long cache_save_ms = 0;
    uint64_t cache_bytes = 0;

    void print(const string& name) const {
        cout << "\n=== " << name << " Timing ===" << endl;
        cout << "  SRS setup (excluded): " << setw(8) << srs_ms << " ms" << endl;
        cout << "  Preprocess:           " << setw(8) << preprocess_ms << " ms" << endl;
        cout << "  Cache save (excluded):" << setw(8) << cache_save_ms << " ms" << endl;
        cout << "  Auxiliary data:       " << fixed << setprecision(3)
             << (cache_bytes / 1073741824.0) << " GiB" << endl;
    }
};

ll* generate_random_embedding(int vocab_size, int embed_dim, int seed = 42) {
    cout << "Generating random embedding matrix..." << endl;

    mt19937 gen(seed);
    uniform_int_distribution<ll> dist(0, 65536);

    ll* embedding = new ll[(long long)vocab_size * embed_dim];
    for (long long i = 0; i < (long long)vocab_size * embed_dim; i++) {
        embedding[i] = dist(gen);
    }

    cout << "Generated " << vocab_size << " x " << embed_dim << " random embedding matrix" << endl;
    cout << "Memory used: " << ((long long)vocab_size * embed_dim * sizeof(ll) / (1024 * 1024)) << " MB" << endl;

    return embedding;
}

ll* load_embedding(const string& filepath, int vocab_size, int embed_dim) {
    cout << "Loading embedding from: " << filepath << endl;

    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        cerr << "Error: Cannot open file " << filepath << endl;
        return nullptr;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    long expected_size = (long)vocab_size * embed_dim * sizeof(int32_t);
    if (file_size != expected_size) {
        cerr << "Error: File size mismatch. Expected " << expected_size
             << " bytes, got " << file_size << " bytes" << endl;
        fclose(f);
        return nullptr;
    }

    int32_t* temp_embedding = new int32_t[(long long)vocab_size * embed_dim];
    size_t read_count = fread(temp_embedding, sizeof(int32_t), (size_t)vocab_size * embed_dim, f);
    fclose(f);

    if (read_count != (size_t)vocab_size * embed_dim) {
        cerr << "Error: Read " << read_count << " elements, expected "
             << (long long)vocab_size * embed_dim << endl;
        delete[] temp_embedding;
        return nullptr;
    }

    ll* embedding = new ll[(long long)vocab_size * embed_dim];
    ll min_val = temp_embedding[0], max_val = temp_embedding[0];
    for (long long i = 0; i < (long long)vocab_size * embed_dim; i++) {
        embedding[i] = (ll)temp_embedding[i];
        if (temp_embedding[i] < min_val) min_val = temp_embedding[i];
        if (temp_embedding[i] > max_val) max_val = temp_embedding[i];
    }
    delete[] temp_embedding;

    cout << "Loaded " << vocab_size << " x " << embed_dim << " embedding matrix" << endl;
    cout << "Memory used: " << (file_size / (1024 * 1024)) << " MB (file), "
         << ((long long)vocab_size * embed_dim * sizeof(ll) / (1024 * 1024)) << " MB (memory)" << endl;
    cout << "Value range: [" << (long long)min_val << ", " << (long long)max_val << "]" << endl;

    return embedding;
}

void run_benchmark(const string& model_name, const string& embedding_path,
                   int vocab_size, int embed_dim, int threads,
                   PreprocessTiming& timing) {
    cout << "\n" << string(70, '=') << endl;
    cout << "BENCHMARK: " << model_name << endl;
    cout << "  vocab_size=" << vocab_size << ", embed_dim=" << embed_dim << endl;
    cout << string(70, '=') << endl;

    ll* embedding = nullptr;
    if (!embedding_path.empty()) {
        embedding = load_embedding(embedding_path, vocab_size, embed_dim);
        if (!embedding) {
            cerr << "Failed to load embedding from " << embedding_path << endl;
            cerr << "Falling back to random generation..." << endl;
            embedding = generate_random_embedding(vocab_size, embed_dim);
        }
    } else {
        embedding = generate_random_embedding(vocab_size, embed_dim);
    }

    if (!embedding) {
        cerr << "Failed to create embedding matrix for " << model_name << endl;
        return;
    }

    string cache_file = ZkEmbed::cacheFilename(vocab_size, embed_dim);
    remove(cache_file.c_str());
    cout << "\n--- Fresh Setup (cache deleted) ---" << endl;

    int table_N = ZkEmbed::requiredTableSize(vocab_size);

    auto srs_start = chrono::high_resolution_clock::now();
    SRSPlan srs_plan;
    srs_plan.uni_size = table_N;
    SRSBundle srs;
    srs.setup_cached(srs_plan, "cache");
    auto srs_end = chrono::high_resolution_clock::now();
    timing.srs_ms = chrono::duration_cast<chrono::milliseconds>(srs_end - srs_start).count();

    ZkEmbed zkembed(
        srs.uni.pk, srs.uni.vk, srs.mv.pk, srs.mv.vk,
        vocab_size, embed_dim, threads);
    zkembed.preprocess(embedding, &timing.preprocess_ms, &timing.cache_save_ms);
    struct stat cache_stat;
    if (stat(cache_file.c_str(), &cache_stat) == 0) {
        timing.cache_bytes = static_cast<uint64_t>(cache_stat.st_size);
    }
    cout << "\n  Preprocess time: " << timing.preprocess_ms << " ms" << endl;
    cout << "  Cache save time (excluded): " << timing.cache_save_ms << " ms" << endl;
    cout << "  Auxiliary data size: " << fixed << setprecision(3)
         << (timing.cache_bytes / 1073741824.0) << " GiB" << endl;

    delete[] embedding;
}

int main(int argc, char* argv[]) {
    initPairing(mcl::BN_SNARK1);

    const char* thread_env = getenv("THREADS");
    if (!thread_env || !*thread_env) thread_env = getenv("OMP_NUM_THREADS");
    const int threads = thread_env && *thread_env ? std::atoi(thread_env) : 32;
    if (threads <= 0) {
        cerr << "Thread count must be positive" << endl;
        return 2;
    }
    const ScopedOmpThreads thread_scope(threads);

    cout << "Embedding Table Preprocessing Benchmark" << endl;
    cout << "========================================" << endl;
    cout << "Threads: " << threads << endl;

    string gpt2_embedding_path = "./model/gpt2-124m/quantized/tensors/embedding_token_i32.bin";

    const char* env_gpt2 = getenv("GPT2_EMBEDDING_PATH");
    if (env_gpt2) gpt2_embedding_path = env_gpt2;

    struct ModelConfig {
        string key;
        string name;
        string embedding_path;
        int vocab_size;
        int embed_dim;
    };

    vector<ModelConfig> models = {
        {"gpt2-small", "GPT-2 Small", gpt2_embedding_path, GPT2_SMALL_VOCAB_SIZE, GPT2_SMALL_EMBED_DIM},
        {"tinyllama", "TinyLlama", "", TINYLLAMA_VOCAB_SIZE, TINYLLAMA_EMBED_DIM},
        {"gpt2-medium", "GPT-2 Medium", "", GPT2_MEDIUM_VOCAB_SIZE, GPT2_MEDIUM_EMBED_DIM},
        {"gpt2-xl", "GPT-2 XL", "", GPT2_XL_VOCAB_SIZE, GPT2_XL_EMBED_DIM},
        {"gpt3", "GPT-3", "", GPT3_VOCAB_SIZE, GPT3_EMBED_DIM},
        {"deepseek-v2", "DeepSeek V2", "", DEEPSEEK_V2_VOCAB_SIZE, DEEPSEEK_V2_EMBED_DIM},
        {"deepseek-v3", "DeepSeek V3", "", DEEPSEEK_V3_VOCAB_SIZE, DEEPSEEK_V3_EMBED_DIM},
    };

    const char* requested_model = getenv("ZKEMBED_MODEL");
    if (requested_model && *requested_model) {
        vector<ModelConfig> selected;
        for (const auto& model : models) {
            if (model.key == requested_model) selected.push_back(model);
        }
        if (selected.empty()) {
            cerr << "Unknown ZKEMBED_MODEL='" << requested_model << "'" << endl;
            cerr << "Expected one of:";
            for (const auto& model : models) cerr << " " << model.key;
            cerr << endl;
            return 2;
        }
        models.swap(selected);
    }

    const string report_path = benchmark_output::reportPath(
        requested_model && string(requested_model) == "gpt2-small" ? "table1" : "table3");
    vector<PreprocessTiming> timings;

    for (const auto& model : models) {
        PreprocessTiming timing;
        run_benchmark(model.name, model.embedding_path, model.vocab_size, model.embed_dim, threads, timing);
        timings.push_back(timing);
    }

    cout << "\n" << string(70, '=') << endl;
    cout << "Generating markdown report..." << endl;
    cout << string(70, '=') << endl;

    ofstream md_file(report_path);
    if (!md_file) throw runtime_error("Cannot open benchmark report: " + report_path);
    md_file << "# Embedding Table Preprocessing Benchmark\n\n";
    md_file << "Benchmark of ZkEmbed preprocessing time for different embedding tables.\n\n";
    md_file << "## Test Configuration\n\n";
    md_file << "- **Threads**: " << threads << "\n";

    for (const auto& model : models) {
        md_file << "- **" << model.name << "**: vocab_size=" << model.vocab_size
                << ", embed_dim=" << model.embed_dim << "\n";
    }

    md_file << "\n## Results\n\n";
    md_file << "| Model | vocab_size | embed_dim | table_N | Preprocess (ms) | Cache save, excluded (ms) | Auxiliary data (GiB) |\n";
    md_file << "|-------|------------|-----------|---------|----------------:|--------------------------:|---------------------:|\n";

    for (size_t i = 0; i < models.size(); i++) {
        const auto& model = models[i];
        const auto& t = timings[i];
        int table_N = ZkEmbed::requiredTableSize(model.vocab_size);

        md_file << "| " << model.name
                << " | " << model.vocab_size
                << " | " << model.embed_dim
                << " | " << table_N
                << " | " << t.preprocess_ms
                << " | " << t.cache_save_ms
                << " | " << fixed << setprecision(3)
                << (t.cache_bytes / 1073741824.0) << " |\n";
    }

    md_file << "\n## Notes\n\n";
    md_file << "- Preprocess includes FFT planning and table precomputation; input copying and cache I/O are excluded\n";
    md_file << "- SRS setup time is **excluded** from preprocess time (SRS is cached separately)\n";
    md_file << "- Cache files are stored in `cache/zkembed_vocab<vocab>_dim<dim>.bin`\n";
    md_file << "- Existing cache is deleted before each benchmark to measure fresh computation\n";

    md_file.close();
    if (!md_file) throw runtime_error("Failed to write benchmark report: " + report_path);

    cout << "\nResults written to " << report_path << endl;

    return 0;
}
