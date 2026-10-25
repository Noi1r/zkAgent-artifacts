#undef NDEBUG
#include "model/gpt2.hpp"
#include "common/global_var.hpp"
#include "lookup/lookup_bundle.hpp"
#include "lookup/lookup_input_link.hpp"
#include "zkembed/zkembed.hpp"
#include "zkdecode/zkdecode_input_link.hpp"
#include "zkdecode/zkdecode_prover.hpp"
#include "zkdecode/zkdecode_verifier.hpp"
#include "zkdecode/zkdecode_lookup.hpp"
#include "lookup/lookup_input_routing_compiled.hpp"
#include "gkr/input_routing_compiled.hpp"
#include "transformer_benchmark_profile.hpp"
#include "generation_mask.hpp"
#include "benchmark_report.hpp"
#include "gkr/verifier.hpp"
#include "gkr/wiring_compiled.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <omp.h>

#ifdef __linux__
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

using namespace mcl::bn;
using namespace std;

namespace {

void check_transcript_stage(const FiatShamirTranscript& prover,
                             const FiatShamirTranscript& verifier,
                             const char* stage, bool enabled) {
    if (!enabled) return;
    auto p = prover;
    auto v = verifier;
    if (p.challenge() != v.challenge()) {
        throw std::runtime_error(std::string("transcript diverged after ") + stage);
    }
    std::cout << "[benchmark_transcript] stage=" << stage << " aligned=yes\n";
}

const char* configure_numa_policy_for_affinity() {
#ifdef __linux__
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0) {
        return "default-affinity-unavailable";
    }
    const int place_count = omp_get_num_places();
    for (int place = 0; place < place_count; ++place) {
        const int count = omp_get_place_num_procs(place);
        if (count <= 0) continue;
        std::vector<int> cpus(static_cast<std::size_t>(count));
        omp_get_place_proc_ids(place, cpus.data());
        for (int cpu : cpus) {
            if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(cpu, &affinity);
        }
    }

    constexpr int MaxNodes = 64;
    constexpr int WordBits = static_cast<int>(sizeof(unsigned long) * 8);
    unsigned long node_mask[(MaxNodes + WordBits - 1) / WordBits] = {};
    int node_count = 0;
    int highest_node = -1;
    for (int node = 0; node < MaxNodes; ++node) {
        bool used = false;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &affinity)) continue;
            const std::string path = "/sys/devices/system/cpu/cpu"
                + std::to_string(cpu) + "/node" + std::to_string(node);
            if (access(path.c_str(), F_OK) == 0) {
                used = true;
                break;
            }
        }
        if (used) {
            node_mask[node / WordBits] |=
                1UL << static_cast<unsigned>(node % WordBits);
            ++node_count;
            highest_node = node;
        }
    }
    if (node_count <= 1) return "local";
    const long status = syscall(
        SYS_set_mempolicy, MPOL_INTERLEAVE, node_mask, highest_node + 1);
    return status == 0 ? "interleave" : "default-interleave-failed";
#else
    return "platform-default";
#endif
}

bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

void ensure_directory(const string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        if ((st.st_mode & S_IFDIR) == 0) {
            throw runtime_error("cache path exists but is not a directory: " + path);
        }
        return;
    }
    if (mkdir(path.c_str(), 0755) != 0) {
        throw runtime_error("failed to create cache directory: " + path);
    }
}

void read_model_config(const string& model_dir, int& depth, int& heads,
                       int& head_dim, int& hidden, int& ffn, int& vocab) {
    ifstream input(model_dir + "/config.txt");
    if (!input) {
        throw runtime_error("failed to open model config: " + model_dir + "/config.txt");
    }
    string line;
    while (getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t eq = line.find('=');
        if (eq == string::npos) continue;
        const string key = line.substr(0, eq);
        const int value = atoi(line.substr(eq + 1).c_str());
        if (key == "n_layer") depth = value;
        else if (key == "n_head") heads = value;
        else if (key == "head_dim") head_dim = value;
        else if (key == "n_embd") hidden = value;
        else if (key == "ffn_dim") ffn = value;
        else if (key == "vocab_size") vocab = value;
    }
    if (head_dim <= 0 && heads > 0) head_dim = hidden / heads;
}

vector<int> read_exact_tokens(const string& path, int seq_len, int vocab_size) {
    ifstream input(path);
    if (!input) {
        throw runtime_error("failed to open token file: " + path);
    }
    vector<int> tokens;
    int token = 0;
    while (input >> token) {
        if (token < 0 || token >= vocab_size) {
            throw runtime_error("token ID outside the configured vocabulary: " +
                                to_string(token));
        }
        tokens.push_back(token);
    }
    if (!input.eof()) {
        throw runtime_error("token file contains a non-integer value: " + path);
    }
    if (tokens.size() != static_cast<size_t>(seq_len)) {
        throw runtime_error("token file must contain exactly seq_len token IDs");
    }
    return tokens;
}

const char* lookup_backend_name(LookupBackend backend) {
    switch (backend) {
        case LookupBackend::Shout:
            return "shout";
        case LookupBackend::LogUpGkr:
            return "logup-gkr";
    }
    throw invalid_argument("unknown lookup backend");
}

const char* lookup_opening_verification_name(
    LookupOpeningVerification verification) {
    switch (verification) {
        case LookupOpeningVerification::Individual:
            return "individual";
        case LookupOpeningVerification::BatchMvKzg:
            return "batch-mvkzg";
    }
    throw invalid_argument("unknown lookup opening verification mode");
}

TransformerBenchmarkProfile default_benchmark_profile() {
#ifdef ZKAGENT_TABLE6_DEFAULT
    return TransformerBenchmarkProfile::Table6Greedy;
#else
    return TransformerBenchmarkProfile::Table5ZkGPTStandalone;
#endif
}

void print_usage(const char* program) {
    cout << "Usage: " << program << " --model-dir DIR --token-file FILE [options]\n"
         << "\nDefault benchmark: " << transformerBenchmarkProfileSpec(default_benchmark_profile()).name << "\n"
         << "Table 5 circuit relation: zkgpt-artifact-transformer\n"
         << "  Includes Transformer arithmetic, actual range/mapped lookups, and the GKR input opening.\n"
         << "  Excludes skip connections, final LayerNorm, LM head, embedding proof, decode,\n"
         << "  and shift consistency. In the default Table 5 profile, lookup proofs are\n"
         << "  standalone and are not linked to the GKR input. Public wiring and input-routing\n"
         << "  coefficient evaluations are included in verifier time. Prover and verifier\n"
         << "  totals follow the roles that execute the work.\n\n"
         << "Options:\n"
         << "  --seq-len N             Full-sequence length (default: 32)\n"
         << "  --prompt-len N          Table 6 suffix-only generation; alternative to --generation-mask\n"
         << "  --generation-mask FILE  Public mask: T bits (0=input, 1=generated); permits interleaved spans\n"
         << "                          Supply exactly one of --prompt-len and --generation-mask.\n"
         << "  --decode-batch-size N   Table 6 maximum generated tokens per decode proof (default: 0, one proof)\n"
         << "  --threads N             OpenMP threads (default: 32)\n"
         << "  --depth N               Override model layer count\n"
         << "  --heads N               Override attention head count\n"
         << "  --head-dim N            Override head dimension\n"
         << "  --hidden-dim N          Override hidden dimension\n"
         << "  --ffn-dim N             Override FFN dimension\n"
         << "  --vocab-size N          Override vocabulary size\n"
         << "  --lookup-backend NAME   shout or logup-gkr (default: shout)\n"
         << "  --proof-profile NAME    table5-zkgpt-standalone (default)\n"
         << "                          or table6-greedy\n"
         << "  benchmark_inference defaults to table6-greedy: complete circuit +\n"
         << "  embedding/APE/lookups + input-linked greedy decoding and public shift check.\n"
         << "  --no-merge              Disable the zkGPT-style merged circuit\n"
         << "  --use-cache             Load a compatible prebuilt circuit when available\n"
         << "  --update-cache          Rebuild and replace the circuit cache\n"
         << "  --prepare-circuit-only  Prepare/cache the circuit and public plans\n"
         << "  --witness-only         Measure cache-hit witness generation and exit without proofs\n"
         << "  --prepare-lookup-routing-only  Prepare the lookup route cache and exit before online work\n"
         << "  --cache-dir DIR         Cache directory (default: ./cache)\n"
         << "  --debug                 Show internal protocol diagnostics before the report\n";
    cout << "  --online-profile FILE   Save online stage/chunk timings and FS checkpoints as CSV\n";
}

}

int main(int argc, char** argv) {
    configure_numa_policy_for_affinity();
    initPairing(mcl::BN_SNARK1);

    string model_dir;
    string token_file;
    string generation_mask_file;
    string online_profile_file;
    string cache_dir = "./cache";
    int seq_len = 32;
    int prompt_len = -1;
    std::size_t decode_batch_size = 0;
    bool prompt_len_provided = false;
    bool explicit_generation_mask = false;
    int threads = 32;
    int depth = 0;
    int heads = 0;
    int head_dim = 0;
    int hidden = 0;
    int ffn = 0;
    int vocab = 0;
    constexpr int lookup_max_chunks = 2;
    bool merge = true;
    bool use_cache = false;
    bool update_cache = false;
    bool prepare_only = false;
    bool witness_only = false;
    bool lookup_routing_only = false;
    bool debug = false;
    LookupBackend lookup_backend = LookupBackend::Shout;
    constexpr LookupOpeningVerification lookup_opening_verification =
        LookupOpeningVerification::BatchMvKzg;
    TransformerBenchmarkProfile benchmark_profile = default_benchmark_profile();

    bool depth_override = false;
    bool heads_override = false;
    bool head_dim_override = false;
    bool hidden_override = false;
    bool ffn_override = false;
    bool vocab_override = false;

    for (int i = 1; i < argc; ++i) {
        auto require_value = [&](const char* option) -> const char* {
            if (i + 1 >= argc) throw invalid_argument(string(option) + " requires a value");
            return argv[++i];
        };
        if (strcmp(argv[i], "--model-dir") == 0) model_dir = require_value(argv[i]);
        else if (strcmp(argv[i], "--token-file") == 0) token_file = require_value(argv[i]);
        else if (strcmp(argv[i], "--online-profile") == 0) online_profile_file = require_value(argv[i]);
        else if (strcmp(argv[i], "--seq-len") == 0) seq_len = atoi(require_value(argv[i]));
        else if (strcmp(argv[i], "--prompt-len") == 0) {
            prompt_len = atoi(require_value(argv[i]));
            prompt_len_provided = true;
        }
        else if (strcmp(argv[i], "--generation-mask") == 0) {
            generation_mask_file = require_value(argv[i]);
            explicit_generation_mask = true;
        }
        else if (strcmp(argv[i], "--decode-batch-size") == 0) {
            const string value = require_value(argv[i]);
            std::size_t parsed = 0;
            bool valid = !value.empty();
            for (const char digit : value) {
                if (digit < '0' || digit > '9' ||
                    parsed > (std::numeric_limits<std::size_t>::max() - (digit - '0')) / 10) {
                    valid = false;
                    break;
                }
                parsed = parsed * 10 + (digit - '0');
            }
            if (!valid) {
                cerr << "--decode-batch-size requires a nonnegative integer" << endl;
                return 1;
            }
            decode_batch_size = parsed;
        }
        else if (strcmp(argv[i], "--threads") == 0) threads = atoi(require_value(argv[i]));
        else if (strcmp(argv[i], "--depth") == 0) { depth = atoi(require_value(argv[i])); depth_override = true; }
        else if (strcmp(argv[i], "--heads") == 0) { heads = atoi(require_value(argv[i])); heads_override = true; }
        else if (strcmp(argv[i], "--head-dim") == 0) { head_dim = atoi(require_value(argv[i])); head_dim_override = true; }
        else if (strcmp(argv[i], "--hidden-dim") == 0) { hidden = atoi(require_value(argv[i])); hidden_override = true; }
        else if (strcmp(argv[i], "--ffn-dim") == 0) { ffn = atoi(require_value(argv[i])); ffn_override = true; }
        else if (strcmp(argv[i], "--vocab-size") == 0) { vocab = atoi(require_value(argv[i])); vocab_override = true; }
        else if (strcmp(argv[i], "--lookup-backend") == 0) {
            const string value = require_value(argv[i]);
            if (value == "shout") lookup_backend = LookupBackend::Shout;
            else if (value == "logup-gkr") lookup_backend = LookupBackend::LogUpGkr;
            else {
                cerr << "Unknown lookup backend: " << value << endl;
                print_usage(argv[0]);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--proof-profile") == 0) {
            benchmark_profile = parseTransformerBenchmarkProfile(
                require_value(argv[i]));
        }
        else if (strcmp(argv[i], "--cache-dir") == 0) cache_dir = require_value(argv[i]);
        else if (strcmp(argv[i], "--no-merge") == 0) merge = false;
        else if (strcmp(argv[i], "--use-cache") == 0) use_cache = true;
        else if (strcmp(argv[i], "--update-cache") == 0) { use_cache = true; update_cache = true; }
        else if (strcmp(argv[i], "--prepare-circuit-only") == 0) {
            use_cache = true;
            prepare_only = true;
        } else if (strcmp(argv[i], "--witness-only") == 0) witness_only = true;
        else if (strcmp(argv[i], "--prepare-lookup-routing-only") == 0) lookup_routing_only = true;
        else if (strcmp(argv[i], "--debug") == 0) debug = true;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            cerr << "Unknown option: " << argv[i] << endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        benchmark_report::QuietLibraryOutput library_output(debug);
        benchmark_report::Writer report;
        setSrsProgressOutput(std::cout);
        if (lookup_routing_only && (!use_cache || update_cache || prepare_only || witness_only))
            throw invalid_argument("--prepare-lookup-routing-only requires --use-cache without other preparation modes or updates");
        if (witness_only && (!use_cache || update_cache || prepare_only))
            throw invalid_argument("--witness-only requires --use-cache, without cache updates or preparation-only mode");
        if (model_dir.empty()) throw invalid_argument("--model-dir is required");
        if (token_file.empty() && !prepare_only) {
            throw invalid_argument("--token-file is required unless --prepare-circuit-only is used");
        }
        const bool require_decode = benchmark_profile == TransformerBenchmarkProfile::Table6Greedy;
        const bool full_inference = require_decode;
        if (require_decode && prompt_len_provided == explicit_generation_mask)
            throw invalid_argument("table6-greedy requires exactly one of --prompt-len and --generation-mask");
        if (require_decode && !explicit_generation_mask && (prompt_len <= 0 || prompt_len >= seq_len))
            throw invalid_argument("table6-greedy requires 1 <= --prompt-len < --seq-len");
        if (!require_decode && decode_batch_size != 0)
            throw invalid_argument("--decode-batch-size is only used by table6-greedy");
        if (!require_decode && (prompt_len_provided || explicit_generation_mask))
            throw invalid_argument("generation selection is only used by table6-greedy");
        if (seq_len <= 0 || (!full_inference && !is_power_of_two(seq_len))) {
            throw invalid_argument("sequence length must be positive (Table 5 also requires a power of two)");
        }
        if (threads <= 0) throw invalid_argument("--threads must be positive");

        // The caller supplies the public generation statement, not the prover.
        // Both proof construction and verification use this exact position set.
        const std::vector<std::size_t> generation_positions = !require_decode
            ? std::vector<std::size_t>{}
            : explicit_generation_mask
                ? generation_mask::load(generation_mask_file, static_cast<std::size_t>(seq_len))
                : generation_mask::suffix(static_cast<std::size_t>(prompt_len), static_cast<std::size_t>(seq_len));

        const auto decode_position_batches = require_decode
            ? generation_mask::partition(generation_positions, decode_batch_size)
            : std::vector<std::vector<std::size_t>>{};

        int config_depth = 0, config_heads = 0, config_head_dim = 0;
        int config_hidden = 0, config_ffn = 0, config_vocab = 0;
        read_model_config(model_dir, config_depth, config_heads, config_head_dim,
                          config_hidden, config_ffn, config_vocab);
        if (!depth_override) depth = config_depth;
        if (!heads_override) heads = config_heads;
        if (!head_dim_override) head_dim = config_head_dim;
        if (!hidden_override) hidden = config_hidden;
        if (!ffn_override) ffn = config_ffn;
        if (!vocab_override) vocab = config_vocab;
        if (depth <= 0 || heads <= 0 || head_dim <= 0 || hidden <= 0 ||
            ffn <= 0 || vocab <= 0 || hidden != heads * head_dim) {
            throw invalid_argument("invalid GPT-2 dimensions after config/CLI resolution");
        }

        omp_set_num_threads(threads);
        g_debug = debug;

        const TransformerBenchmarkProfileSpec& benchmark_spec =
            transformerBenchmarkProfileSpec(benchmark_profile);
        const bool require_lookup_link =
            benchmark_spec.require_lookup_input_link;
        if (lookup_routing_only && !require_lookup_link)
            throw invalid_argument("lookup route preparation requires an input-linked profile");
        const bool use_compiled_wiring =
            benchmark_spec.use_compiled_wiring_plan;
        const bool use_compiled_input_routing =
            benchmark_spec.use_compiled_input_routing_plan;
        if (!use_compiled_wiring) {
            throw logic_error(
                "benchmark profile has no public wiring evaluator");
        }
        if (!use_compiled_input_routing) {
            throw logic_error(
                "benchmark profile has no public input-routing evaluator");
        }
        SRSBundle srs;
        MVKZG_SRS& input_srs = srs.mv;
        GPT2 model(srs.uni, input_srs, depth, heads, head_dim,
                   hidden, ffn, seq_len, vocab, threads);
        model.setCircuitProfile(benchmark_spec.circuit_profile);
        model.loadTransformerWeights(model_dir.c_str());
        if (prepare_only) model.setTokensNoPad(vector<int>(seq_len, 0));
        else model.setTokensNoPad(read_exact_tokens(token_file, seq_len, vocab));
        if (model.inputTokenCount() != seq_len) {
            throw runtime_error("token file must contain exactly seq_len valid token IDs");
        }
        model.computeEmbedding();

        prover p;
        const string profile_name = model.circuitProfileSpec().name;
        const string cache_file = cache_dir + "/circuit_" + profile_name +
            "_d" + to_string(depth) + "_h" + to_string(heads) +
            "_hd" + to_string(head_dim) + "_ad" + to_string(hidden) +
            "_ld" + to_string(ffn) + "_sl" + to_string(seq_len) +
            "_m" + to_string(merge ? 1 : 0) + ".cache";
        const string wiring_plan_file = cache_file + ".wiring-plan";
        const string input_routing_plan_file =
            cache_file + ".input-routing-plan";
        CompiledWiringPlan compiled_wiring;
        CompiledInputRoutingPlan compiled_input_routing;
        bool cache_loaded = false;
        double circuit_and_plans_seconds = 0.0;
        auto print_configuration = [&]() {
            report.section("Run Configuration");
            report.field("table", full_inference ? 6 : 5);
            report.field("profile", benchmark_spec.name);
            report.field("mode", witness_only ? "witness-only" :
                lookup_routing_only ? "lookup-routing-preparation" :
                prepare_only ? "circuit-preparation" : "full");
            report.field("model", "GPT-2, " + to_string(depth) + " layers, Q" + to_string(model.quant_plan.q_bits));
            report.field("sequence", seq_len);
            if (model.len != seq_len) report.field("padded_sequence", model.len);
            report.field("prompt", require_decode
                ? to_string(generation_mask::initialPromptLength(generation_positions)) : "N/A");
            if (require_decode) {
                report.field("generation selection", explicit_generation_mask ? "public mask" : "contiguous suffix");
                report.field("generated tokens", generation_positions.size());
                string batch_sizes;
                for (const auto& positions : decode_position_batches) {
                    if (!batch_sizes.empty()) batch_sizes += ',';
                    batch_sizes += to_string(positions.size());
                }
                report.field("decode batch sizes", batch_sizes);
                report.field("decode range limb bits", zkdecode_detail::RangeLimbBits);
                if (explicit_generation_mask)
                    report.field("generation positions", generation_mask::formatRanges(generation_positions));
            }
            report.field("threads", threads);
            report.field("lookup", string(lookup_backend == LookupBackend::Shout ? "SHOUT" : "LogUp-GKR") +
                ", max-chunks=" + to_string(lookup_max_chunks));
            report.field("lookup_openings", lookup_opening_verification_name(lookup_opening_verification));
            report.field("transcript", "Fiat-Shamir");
            report.field("cache", cache_loaded ? "hit" : "fresh");
        };
        auto print_unverified_completion = [&](const char* status) {
            report.section("Verification Status");
            report.field("proof_generated", "NO");
            report.field("verification", "NOT RUN");
            report.section("Final Result");
            report.field("status", status);
        };
        const auto preparation_start = chrono::steady_clock::now();
        if (use_cache && !update_cache) {
            cache_loaded = model.loadCircuitCache(cache_file.c_str(), p);
        }
        if (cache_loaded) {
            if (!prepare_only) {
                model.prepareWeightsForCache(p);
                model.recomputeWitness(p);
            }
        }
        if (lookup_routing_only && !cache_loaded)
            throw runtime_error("lookup route preparation requires a compatible circuit cache hit");
        if (witness_only) {
            if (!cache_loaded)
                throw runtime_error("witness-only measurement requires a compatible cache hit; no circuit rebuild was performed");
            const double preparation_total_seconds = chrono::duration<double>(
                chrono::steady_clock::now() - preparation_start).count();
            library_output.finish(true);
            print_configuration();
            report.section("Preparation");
            report.seconds("total", preparation_total_seconds);
            report.seconds("circuit_and_plans", circuit_and_plans_seconds);
            print_unverified_completion("WITNESS READY");
            return 0;
        }
        if (!cache_loaded) {
            const auto circuit_start = chrono::steady_clock::now();
            model.prepareCircuitData(p);
            model.buildCircuit(p, merge);
            if (use_cache) {
                ensure_directory(cache_dir);
                if (!model.saveCircuitCache(cache_file.c_str(), p)) {
                    throw runtime_error("failed to save circuit cache: " + cache_file);
                }
            }
            circuit_and_plans_seconds += chrono::duration<double>(
                chrono::steady_clock::now() - circuit_start).count();
        }

        if (!prepare_only) {
            string profile_error;
            if (!model.validateCircuitProfile(p, &profile_error)) {
                throw runtime_error("profile validation failed: " + profile_error);
            }
        }

        if (use_compiled_wiring) {
            if (cache_loaded && !prepare_only &&
                ifstream(wiring_plan_file, ios::binary).good()) {
                compiled_wiring = CompiledWiringPlan::loadFromFile(
                    wiring_plan_file, p.C);
            } else {
                const auto plan_start = chrono::steady_clock::now();
                compiled_wiring = CompiledWiringPlan::build(p.C, threads);
                if (use_cache) {
                    compiled_wiring.saveToFile(wiring_plan_file);
                }
                circuit_and_plans_seconds += chrono::duration<double>(
                    chrono::steady_clock::now() - plan_start).count();
            }
        }
        if (use_compiled_input_routing) {
            if (cache_loaded && !prepare_only &&
                ifstream(input_routing_plan_file, ios::binary).good()) {
                compiled_input_routing =
                    CompiledInputRoutingPlan::loadFromFile(
                        input_routing_plan_file, p.C);
            } else {
                const auto plan_start = chrono::steady_clock::now();
                compiled_input_routing =
                    CompiledInputRoutingPlan::build(p.C, threads);
                if (use_cache) {
                    compiled_input_routing.saveToFile(
                        input_routing_plan_file);
                }
                circuit_and_plans_seconds += chrono::duration<double>(
                    chrono::steady_clock::now() - plan_start).count();
            }
        }
        ensure_directory(cache_dir);
        const string model_key_file = cache_file + ".model.vk";
        const auto model_input = model.prepareInputCommitment(p, model_key_file, cache_dir, update_cache);
        const double initial_preparation_seconds = chrono::duration<double>(
            chrono::steady_clock::now() - preparation_start).count();

        struct DecodeBatch {
            CompiledInputTensorView view;
            ZkDecodeStatement statement;
            ZkDecodeVerifierKey key;
            std::vector<std::vector<ll>> logits;
            ZkDecodeProof proof;
            InputReductionLink prover_claim;
            std::unique_ptr<InputReductionLink> verifier_claim;
        };
        std::vector<DecodeBatch> decode_batches(decode_position_batches.size());
        double decode_routing_seconds = 0.0, decode_key_seconds = 0.0, decode_witness_seconds = 0.0;
        if (require_decode) {
            if (model.lm_head_output_offset <= 0 || model.quant_plan.logit_bound < 0 ||
                model.quant_plan.logit_bound >= (ll(1) << 125) - 1)
                throw invalid_argument("LM-head source/bound is not supported by the native decoder");
            for (std::size_t index = 0; index < decode_position_batches.size(); ++index) {
                auto& batch = decode_batches[index];
                const auto& positions = decode_position_batches[index];
                auto& params = batch.statement.parameters;
                params.kind = ZkDecodeProofKind::Greedy;
                params.lookup_backend = lookup_backend;
                params.token_count = static_cast<int>(positions.size());
                params.vocab_size = vocab;
                params.bound = model.quant_plan.logit_bound + 1;
                InputTensorViewSpec view_spec{
                    p.C.circuit[0].bit_length,
                    std::size_t(model.lm_head_output_offset),
                    positions.size(), std::size_t(vocab), std::size_t(vocab)};
                string view_file;
                if (explicit_generation_mask || decode_batch_size != 0) {
                    FiatShamirTranscript mask_cache_key;
                    mask_cache_key.append_u64(0x47454e4d41534b31ULL); // GENMASK1
                    mask_cache_key.append_u64(static_cast<std::size_t>(seq_len));
                    mask_cache_key.append_u64(positions.size());
                    const auto domain = view_spec.inputDomain();
                    if (view_spec.offset >= domain || view_spec.columns > domain - view_spec.offset)
                        throw invalid_argument("LM-head source exceeds the committed input");
                    for (const auto position : positions) {
                        if (position - 1 > (domain - view_spec.offset - view_spec.columns) / view_spec.row_stride)
                            throw invalid_argument("generated token's logit row exceeds the committed input");
                        view_spec.row_offsets.push_back(view_spec.offset + (position - 1) * view_spec.row_stride);
                        mask_cache_key.append_u64(position);
                    }
                    view_file = cache_file + ".logits-mask-" + mask_cache_key.challenge().getStr(16) + ".plan";
                } else {
                    view_spec.offset += std::size_t(prompt_len - 1) * view_spec.row_stride;
                    view_file = cache_file + ".logits-prompt" + to_string(prompt_len) + ".plan";
                }
                view_spec.validate();
                const auto start = chrono::steady_clock::now();
                if (cache_loaded && use_cache && !update_cache && ifstream(view_file, ios::binary).good()) {
                    batch.view = CompiledInputTensorView::loadFromFile(view_file, view_spec);
                } else {
                    batch.view = CompiledInputTensorView::build(view_spec);
                    if (use_cache) { ensure_directory(cache_dir); batch.view.saveToFile(view_file); }
                }
                decode_routing_seconds += chrono::duration<double>(chrono::steady_clock::now() - start).count();
                if (!prepare_only) {
                    const auto witness_start = chrono::steady_clock::now();
                    batch.statement.output_tokens.reserve(positions.size());
                    for (const auto position : positions)
                        batch.statement.output_tokens.push_back(model.input_tokens[position]);
                    batch.logits = prepareZkDecodeInputWitness(p.val[0], view_spec, params.bound);
                    decode_witness_seconds += chrono::duration<double>(chrono::steady_clock::now() - witness_start).count();
                }
            }
        }
        auto prepare_decode_key = [&]() {
            const auto start = chrono::steady_clock::now();
            for (auto& batch : decode_batches)
                batch.key = prepareZkDecodeVerifierKey(batch.statement.parameters, input_srs.pk, input_srs.vk, threads);
            if (require_decode)
                decode_key_seconds = chrono::duration<double>(chrono::steady_clock::now() - start).count();
        };

        ApeShape ape_shape;
        ApeProverKey ape_pk;
        ApeVerifierKey ape_vk;
        string ape_key_file;
        string ape_key_access = "not-applicable";
        double ape_prepare_seconds = 0.0;
        if (full_inference) {
            ape_shape = {std::size_t(1) << ceilPow2BitLength(static_cast<u32>(seq_len)),
                         static_cast<std::size_t>(hidden), p.C.circuit[0].bit_length};
            ape_key_file = cache_dir + "/ape_t" + to_string(ape_shape.padded_tokens)
                + "_d" + to_string(hidden) + "_in" + to_string(ape_shape.input_variables)
                + ".vk";
        }
        auto prepare_ape = [&]() {
            const auto start = chrono::steady_clock::now();
            ape_pk = prepareApeProverKey(ape_shape, model.quant_plan.position_embedding);
            if (use_cache && !update_cache && ifstream(ape_key_file, ios::binary).good()) {
                ape_vk = ApeVerifierKey::loadFromFile(ape_key_file, ape_shape);
                ape_key_access = "vk-load";
            } else {
                ape_vk = prepareApeVerifierKey(ape_pk, input_srs.pk, threads);
                ape_key_access = "prepared";
                if (use_cache) {
                    ensure_directory(cache_dir);
                    ape_vk.saveToFile(ape_key_file);
                }
            }
            ape_prepare_seconds = chrono::duration<double>(chrono::steady_clock::now() - start).count();
        };
        if (prepare_only && full_inference) {
            SRSPlan key_plan;
            key_plan.mv_vars = (ape_shape.input_variables + 1) / 2;
            for (const auto& batch : decode_batches)
                key_plan.mv_vars = max(key_plan.mv_vars,
                    MVKZG_SRS::min_vars_for_size(zkDecodeRequiredGenerators(batch.statement.parameters)));
            srs.setup_cached(key_plan, cache_dir);
            prepare_ape();
            prepare_decode_key();

        }

        if (prepare_only) {
            const double preparation_total_seconds = chrono::duration<double>(
                chrono::steady_clock::now() - preparation_start).count();
            library_output.finish(true);
            print_configuration();
            report.section("Preparation");
            report.seconds("total", preparation_total_seconds);
            report.seconds("circuit_and_plans", circuit_and_plans_seconds);
            print_unverified_completion("CACHE READY");
            return 0;
        }

        const auto lookup_witness_start = chrono::steady_clock::now();
        const size_t gkr_input_values = p.val[0].size();
        LookupWitness lookup_witness = model.buildTransformerLookupWitness(p, 10);
        if (p.val[0].size() != gkr_input_values) {
            throw runtime_error(
                "lookup witness generation modified the committed GKR input");
        }
        const double lookup_witness_seconds = chrono::duration<double>(
            chrono::steady_clock::now() - lookup_witness_start).count();
        CompiledLookupInputRoutingPlan compiled_lookup_input_routing;
        double compiled_lookup_input_routing_prepare_seconds = 0.0;
        if (require_lookup_link) {
            const auto link_routing_start = chrono::steady_clock::now();
            const std::string lookup_plan_file = cache_file + ".lookup-routing.plan";
            if (use_cache && !update_cache && std::ifstream(lookup_plan_file, std::ios::binary).good()) {
                compiled_lookup_input_routing = CompiledLookupInputRoutingPlan::loadFromFile(
                    lookup_plan_file, lookup_witness.layout);
            } else {
                compiled_lookup_input_routing = CompiledLookupInputRoutingPlan::build(lookup_witness.layout);
                if (use_cache) compiled_lookup_input_routing.saveToFile(lookup_plan_file);
            }
            compiled_lookup_input_routing_prepare_seconds =
                chrono::duration<double>(
                    chrono::steady_clock::now() - link_routing_start)
                    .count();
        }
        if (lookup_routing_only) {
            const double preparation_total_seconds = chrono::duration<double>(
                chrono::steady_clock::now() - preparation_start).count();
            library_output.finish(true);
            print_configuration();
            report.section("Preparation");
            report.seconds("total", preparation_total_seconds);
            report.seconds("circuit_and_plans", circuit_and_plans_seconds);
            print_unverified_completion("LOOKUP ROUTING READY");
            return 0;
        }
        const LookupPackingPlan requested_lookup_packing =
            planLookupBundlePacking(
                lookup_witness, static_cast<std::size_t>(lookup_max_chunks));

        long long uni_gates = 0;
        long long bin_gates = 0;
        int fconn_layers = 0;
        for (const layer& circuit : p.C.circuit) {
            uni_gates += static_cast<long long>(circuit.uni_gates.size());
            bin_gates += static_cast<long long>(circuit.bin_gates.size());
            if (circuit.ty == layerType::FCONN) ++fconn_layers;
        }

        const int input_log = p.C.circuit[0].bit_length;
        const size_t input_columns = size_t(1)
            << (input_log - input_log / 2);
        size_t generator_count = max<size_t>(size_t(1) << 12,
            max(input_columns, lookupBundleRequiredGenerators(
                lookup_witness, lookup_backend,
                static_cast<std::size_t>(lookup_max_chunks))));
        for (const auto& batch : decode_batches)
            generator_count = max(generator_count, zkDecodeRequiredGenerators(batch.statement.parameters));
        const int mv_vars = MVKZG_SRS::min_vars_for_size(generator_count);
        const auto srs_start = chrono::steady_clock::now();
        ensure_directory(cache_dir);
        SRSPlan srs_plan;
        srs_plan.mv_vars = mv_vars;
        if (full_inference) srs_plan.uni_size = ZkEmbed::requiredTableSize(vocab);
        srs.setup_cached(srs_plan, cache_dir);
        const double srs_seconds = chrono::duration<double>(
            chrono::steady_clock::now() - srs_start).count();
        prepare_decode_key();

        std::vector<Fr> h0_values;
        const std::vector<ll> embedding_queries(model.input_tokens.begin(), model.input_tokens.end());
        double embedding_prepare_seconds = 0.0;
        if (full_inference) {
            prepare_ape();
            const auto start = chrono::steady_clock::now();
            model.setupZkEmbed();
            model.zkembed->preloadProverDataForQueries(embedding_queries);
            model.zkembed->preloadVerifierData();
            h0_values = model.prepareZkEmbedWitness();
            embedding_prepare_seconds = chrono::duration<double>(chrono::steady_clock::now() - start).count();

        }

        model.ensureGenerators(p);
        const auto commit_start = chrono::steady_clock::now();
        model.commitCircuit(p, model_input.key);
        const double commit_seconds = chrono::duration<double>(
            chrono::steady_clock::now() - commit_start).count();

        FiatShamirTranscript prover_transcript;
        FiatShamirTranscript verifier_transcript;
        std::vector<std::pair<std::string, std::string>> profile_checkpoints;
        auto record_profile_checkpoint = [&](const char* stage) {
            if (online_profile_file.empty()) return;
            auto prover_copy = prover_transcript;
            auto verifier_copy = verifier_transcript;
            const Fr checkpoint = prover_copy.challenge();
            if (checkpoint != verifier_copy.challenge())
                throw runtime_error(string("profile transcript mismatch after ") + stage);
            profile_checkpoints.emplace_back(stage, checkpoint.getStr(16));
        };
        const auto gkr_key_start = chrono::steady_clock::now();
        GkrKeyBundle gkr_keys;
        string gkr_key_access;
        {
            if (use_cache) {
                bool loaded = false;
                const string key_path = cache_file + ".gkr.vk";
                gkr_keys = prepareGkrKeysCached(p, input_srs.pk, input_srs.vk,
                    key_path, threads, update_cache, &loaded);
                gkr_key_access = loaded ? "cache-load" : "prepared-and-saved";
            } else {
                gkr_keys = prepareGkrKeys(p, input_srs.pk, threads);
                gkr_key_access = "prepared";
            }
        }
        const double gkr_key_seconds = chrono::duration<double>(
            chrono::steady_clock::now() - gkr_key_start).count();
        const GkrInputCommitment gkr_input_commitment =
            makeGkrInputCommitment(p);
        double context_prover_seconds = 0.0, context_verifier_seconds = 0.0;
        {
            GkrPublicContext gkr_context;
            gkr_context.sequence_length = seq_len;
            gkr_context.layers = depth;
            gkr_context.heads = heads;
            gkr_context.head_dimension = head_dim;
            gkr_context.hidden_dimension = hidden;
            gkr_context.ffn_dimension = ffn;
            gkr_context.vocabulary_size = vocab;
            gkr_context.merged = merge;
            const auto context_start = chrono::steady_clock::now();
            appendGkrPublicContext(
                prover_transcript, p.C, gkr_keys.verifier,
                gkr_input_commitment, gkr_context);
            const auto context_middle = chrono::steady_clock::now();
            appendGkrPublicContext(
                verifier_transcript, p.C, gkr_keys.verifier,
                gkr_input_commitment, gkr_context);
            context_prover_seconds = chrono::duration<double>(context_middle - context_start).count();
            context_verifier_seconds = chrono::duration<double>(chrono::steady_clock::now() - context_middle).count();
        }
        ZkEmbedInputProof embedding_proof;
        InputReductionLink embedding_prover_claim;
        std::unique_ptr<InputReductionLink> embedding_verifier_claim;
        double embedding_prover_seconds = 0.0, embedding_verifier_seconds = 0.0;
        if (full_inference) {
            const auto start = chrono::steady_clock::now();
            embedding_proof = proveZkEmbedInputLink(
                *model.zkembed, embedding_queries, h0_values, gkr_input_commitment.rows,
                ape_pk, ape_vk, input_srs.pk, prover_transcript, embedding_prover_claim);
            const auto middle = chrono::steady_clock::now();
            embedding_verifier_claim = verifyZkEmbedInputLink(
                *model.zkembed, embedding_queries, gkr_input_commitment.rows,
                ape_vk, embedding_proof, input_srs.vk, verifier_transcript);
            embedding_prover_seconds = chrono::duration<double>(middle - start).count();
            embedding_verifier_seconds = chrono::duration<double>(chrono::steady_clock::now() - middle).count();
            if (!embedding_verifier_claim) throw runtime_error("embedding/APE proof failed");
        }
        check_transcript_stage(prover_transcript, verifier_transcript, "embedding", debug);
        record_profile_checkpoint("embedding");
        double decode_prover_seconds = 0.0, decode_setup_seconds = 0.0;
        double decode_verifier_seconds = 0.0;
        std::size_t decode_payload_bytes = 0;
        bool decode_verified = false;
        bool shift_verified = false;
        if (require_decode) {
            // Reconstruct the expected partition from the verifier's public positions,
            // never from proof metadata. Every expected batch must verify and link.
            const auto verifier_batches = generation_mask::partition(generation_positions, decode_batch_size);
            if (decode_batches.empty() || decode_batches.size() != verifier_batches.size())
                throw runtime_error("decode batch count does not cover the public generation positions");
            for (std::size_t index = 0; index < verifier_batches.size(); ++index) {
                auto& batch = decode_batches[index];
                const ZkDecodeInputContext decode_context{batch.view, gkr_input_commitment.rows, model.quant_plan.logit_scale};
                ZkDecodeProver decoder(input_srs.pk, input_srs.vk, threads);
                const auto input_size_before = p.val[0].size();
                auto generated = proveZkDecodeInputLink(decoder, batch.logits, batch.statement, batch.key,
                    decode_context, prover_transcript, batch.prover_claim);
                if (!generated.stats.success) throw runtime_error("native decode proof generation failed");
                decode_prover_seconds += generated.stats.prover_time;
                decode_setup_seconds += generated.stats.setup_time;
                batch.proof = std::move(generated.proof);
                decode_payload_bytes += zkDecodeProofPayloadBytes(batch.proof);
                if (p.val[0].size() != input_size_before) throw logic_error("decoder modified GKR input shape");
                const auto start = chrono::steady_clock::now();
                batch.verifier_claim = verifyZkDecodeInputLink(batch.statement, batch.proof, batch.key,
                    decode_context, input_srs.vk, verifier_transcript);
                const bool batch_shift_verified = batch.verifier_claim && checkDecodeShiftConsistency(
                    model.input_tokens, verifier_batches[index], batch.proof.output_tokens);
                decode_verifier_seconds += chrono::duration<double>(chrono::steady_clock::now() - start).count();
                if (!batch_shift_verified) throw runtime_error("native decode/shift checks failed");
            }
            decode_verified = true;
            shift_verified = true;
        }
        check_transcript_stage(prover_transcript, verifier_transcript, "decode", debug);
        record_profile_checkpoint("decode");
        const LookupBundleProof lookup_proof = proveAndVerifyLookupBundle(
            lookup_witness, input_srs.pk, input_srs.vk, threads, lookup_backend,
             require_lookup_link,
            static_cast<std::size_t>(lookup_max_chunks),
            lookup_opening_verification,
            &prover_transcript,
            &verifier_transcript,
            G1WireEncoding::Compressed);
        check_transcript_stage(prover_transcript, verifier_transcript, "lookup", debug);
        record_profile_checkpoint("lookup");
        if (lookup_proof.query_openings_deferred != require_lookup_link) {
            throw runtime_error(
                "lookup backend used the wrong query-opening policy");
        }
        if (!require_lookup_link) {
            // Standalone lookups are complete before allocating GKR workspaces.
            std::vector<std::vector<ll>>().swap(lookup_witness.group_queries);
            std::vector<std::vector<ll>>().swap(lookup_witness.group_query_outputs);
        }
        LookupInputLinkResult lookup_link;
        if (require_lookup_link) {
            lookup_link = proveAndVerifyLookupInputLink(
                lookup_witness, lookup_proof, input_srs.pk, input_srs.vk,
                threads, &prover_transcript, &verifier_transcript);
            if (!lookup_link.verified) {
                throw runtime_error("lookup input link proof failed");
            }
            lookup_link.input_link.compiled_routing_plan =
                &compiled_lookup_input_routing;
        }
        check_transcript_stage(prover_transcript, verifier_transcript, "lookup-link", debug);
        record_profile_checkpoint("lookup-link");
        InputReductionLink prover_input_link = lookup_link.input_link;
        InputReductionLink verifier_input_link = lookup_link.input_link;
        if (full_inference) {
            prover_input_link.appendEvaluationClaims(embedding_prover_claim);
            verifier_input_link.appendEvaluationClaims(*embedding_verifier_claim);
        }
        if (require_decode) {
            for (const auto& batch : decode_batches) {
                if (!batch.verifier_claim) throw runtime_error("missing verified decode batch input claim");
                prover_input_link.appendEvaluationClaims(batch.prover_claim);
                verifier_input_link.appendEvaluationClaims(*batch.verifier_claim);
            }
        }
        verifier v(&p, p.C, &input_srs.pk, &input_srs.vk, debug);
        v.setGkrKeys(gkr_keys.prover, gkr_keys.verifier);
        v.setFiatShamirTranscript(prover_transcript);
        if (require_lookup_link) {
            v.setInputReductionLink(prover_input_link);
        }
        if (use_compiled_wiring) {
            v.setCompiledWiringPlan(compiled_wiring);
        }

        const auto workspace_start = chrono::steady_clock::now();
        p.prepareGkrWorkspace();
        v.prepareInputReductionWorkspace();
        const double workspace_seconds = chrono::duration<double>(
            chrono::steady_clock::now() - workspace_start).count();

        const bool gkr_generated = v.generateProof(threads);
        const double gkr_online_prover_seconds = v.prover_time + v.verifier_time;
        const GkrProof gkr_proof = v.proof();
        verifier gkr_checker(
            p.C, gkr_keys.verifier, gkr_input_commitment,
            &input_srs.vk, debug);
        gkr_checker.setThreadCount(threads);
        gkr_checker.setInputCommitmentKey(model_input.key);
        gkr_checker.setThreadCount(threads);
        if (require_lookup_link) {
            gkr_checker.setInputReductionLink(verifier_input_link);
        }
        if (use_compiled_wiring) {
            gkr_checker.setCompiledWiringPlan(compiled_wiring);
        }
        if (use_compiled_input_routing) {
            gkr_checker.setCompiledInputRoutingPlan(compiled_input_routing);
        }
        const auto gkr_verify_start = chrono::steady_clock::now();
        const bool gkr_replayed = gkr_checker.verifyProof(
            gkr_proof,
            &verifier_transcript);
        const double gkr_replay_verifier_seconds =
            chrono::duration<double>(
                chrono::steady_clock::now() - gkr_verify_start).count();
        const bool transcript_aligned =
            prover_transcript.challenge() ==
                verifier_transcript.challenge();
        const bool gkr_verified =
            gkr_generated && gkr_replayed && transcript_aligned;
        const bool link_verified = require_lookup_link &&
            lookup_link.verified && v.lookup_link_verified() && gkr_verified;
        const bool composition_verified =
            !require_lookup_link || link_verified;
        const bool verified =
            gkr_verified && lookup_proof.verified && composition_verified &&
            (!full_inference || embedding_verifier_claim != nullptr) &&
            (!require_decode || (decode_verified && shift_verified));
        record_profile_checkpoint("gkr");

        const double role_prover_seconds =
            gkr_online_prover_seconds + lookup_proof.prover_time
            + (require_lookup_link ? lookup_link.prover_time : 0.0)
            + (full_inference ? commit_seconds + embedding_prover_seconds : 0.0)
            + context_prover_seconds
            + decode_prover_seconds;
        const double role_verifier_seconds =
            gkr_replay_verifier_seconds + lookup_proof.verifier_time
            + (require_lookup_link ? lookup_link.verifier_time : 0.0)
            + (full_inference ? embedding_verifier_seconds : 0.0)
            + context_verifier_seconds
            + decode_verifier_seconds;
        const double preparation_total_seconds =
            initial_preparation_seconds
            + decode_routing_seconds
            + decode_key_seconds
            + decode_witness_seconds
            + decode_setup_seconds
            + ape_prepare_seconds
            + lookup_witness_seconds
            + compiled_lookup_input_routing_prepare_seconds
            + srs_seconds
            + embedding_prepare_seconds
            + gkr_key_seconds
            + workspace_seconds
            + lookup_proof.witness_preparation_time
            + lookup_proof.table_preprocess_time
            + (full_inference ? 0.0 : commit_seconds);
        std::size_t input_commitment_bytes = 0;
        if (full_inference) {
            for (std::size_t row = 0; row < gkr_input_commitment.rows.size(); ++row) {
                if (!model_input.key.layout().isRuntimeRow(row)) continue;
                input_commitment_bytes += G1::getSerializedByteSize();
            }
        }
        const std::size_t proof_payload_bytes = gkr_proof.cryptographicPayloadBytes()
            + lookup_proof.proof_size_bytes + input_commitment_bytes
            + (full_inference ? embedding_proof.cryptographicPayloadBytes() : 0)
            + decode_payload_bytes
            + (require_lookup_link ? lookup_link.proof_size_bytes : 0);

        library_output.finish(verified);
        print_configuration();
        report.section("Preparation");
        report.seconds("total", preparation_total_seconds);
        report.seconds("circuit_and_plans", circuit_and_plans_seconds);

        report.section("Circuit Size");
        report.field("FCONN layers", fconn_layers);
        report.field("unary gates", uni_gates);
        report.field("binary gates", bin_gates);
        report.field("input values", p.C.circuit[0].size);
        report.field("lookup groups", lookup_witness.layout.groups().size());
        report.field("lookup chunks", requested_lookup_packing.proof_chunks);

        report.section("Component Timings");
        report.seconds("gkr_prover", gkr_online_prover_seconds);
        report.seconds("gkr_replay_verifier", gkr_replay_verifier_seconds);
        report.seconds("lookup_prover", lookup_proof.prover_time);
        report.seconds("lookup_verifier", lookup_proof.verifier_time);
        if (require_lookup_link) {
            report.seconds("lookup_link_prover", lookup_link.prover_time);
            report.seconds("lookup_link_verifier", lookup_link.verifier_time);
        }
        if (full_inference) {
            report.seconds("input_commit", commit_seconds);
            report.seconds("embedding_prover", embedding_prover_seconds);
            report.seconds("embedding_verifier", embedding_verifier_seconds);
        }
        if (require_decode) {
            report.seconds("decode_prover", decode_prover_seconds);
            report.seconds("decode_verifier", decode_verifier_seconds);
        }

        report.section("Verification Status");
        report.field("GKR replay", gkr_generated && gkr_replayed ? "PASS" : "FAIL");
        report.field("transcript aligned", transcript_aligned ? "YES" : "NO");
        if (full_inference) {
            report.field("embedding linked", embedding_verifier_claim && gkr_verified ? "YES" : "NO");
        }
        report.field("lookup verified", lookup_proof.verified ? "YES" : "NO");
        if (require_lookup_link) {
            report.field("lookup linked", link_verified ? "YES" : "NO");
        }
        if (require_decode) {
            report.field("decode verified", decode_verified && gkr_verified ? "YES" : "NO");
            report.field("shift verified", shift_verified ? "YES" : "NO");
        }

        report.section("Final Result");
        report.field("status", verified ? "PASS" : "FAIL");
        report.seconds("prover", role_prover_seconds);
        report.seconds("verifier", role_verifier_seconds);
        report.field("proof_payload_bytes", proof_payload_bytes);
        report.field("scope", require_decode ? (explicit_generation_mask
                ? "embedding-ape-complete-gkr-lookup-masked-greedy-shift"
                : "embedding-ape-complete-gkr-lookup-greedy-shift") :
            full_inference ? "embedding-ape-complete-gkr-lookup" : "transformer-components");
        if (!online_profile_file.empty()) {
            std::ofstream profile(online_profile_file);
            if (!profile) throw runtime_error("cannot create online profile: " + online_profile_file);
            profile << std::setprecision(17) << "component,index,metric,value,unit\n";
            auto metric = [&](const char* component, long long index, const char* name,
                              double value, const char* unit = "s") {
                profile << component << ',' << index << ',' << name << ',' << value << ',' << unit << '\n';
            };
            metric("total", -1, "prover", role_prover_seconds);
            metric("total", -1, "verifier", role_verifier_seconds);
            metric("total", -1, "proof_payload_bytes", proof_payload_bytes, "bytes");
            metric("input", -1, "total", commit_seconds);
            metric("input", -1, "padding", p.inputCommitTiming().padding_seconds);
            metric("input", -1, "rows", p.inputCommitTiming().row_commitment_seconds);
            metric("input", -1, "runtime_rows", model_input.key.layout().runtimeRowCount(), "count");
            metric("input", -1, "fixed_rows", model_input.key.layout().fixedRowCount(), "count");
            metric("input", -1, "columns", model_input.key.layout().columns(), "count");
            metric("link", -1, "total", lookup_link.prover_time);
            metric("link", -1, "statement", lookup_link.timing.statement_seconds);
            metric("link", -1, "coefficients", lookup_link.timing.coefficient_seconds);
            metric("link", -1, "query_values", lookup_link.timing.query_value_seconds);
            metric("link", -1, "rounds", lookup_link.timing.round_seconds);
            metric("link", -1, "folds", lookup_link.timing.fold_seconds);
            metric("link", -1, "opening", lookup_link.timing.opening_seconds);
            metric("link", -1, "equality_tables", lookup_link.timing.equality_tables, "count");
            metric("link", -1, "equality_values", lookup_link.timing.equality_values, "count");
            metric("link", -1, "equality_cache_hits", lookup_link.timing.equality_cache_hits, "count");
            const auto& init = p.sumcheckInitTiming();
            const auto& input_init = p.inputReductionInitTiming();
            metric("gkr_init", -1, "phase1_total", init.phase1_total_seconds);
            metric("gkr_init", -1, "phase1_workspace", init.phase1_workspace_seconds);
            metric("gkr_init", -1, "phase1_witness_buffer", init.phase1_witness_buffer_seconds);
            metric("gkr_init", -1, "phase1_beta", init.phase1_beta_seconds);
            metric("gkr_init", -1, "phase1_relu_scale", init.phase1_relu_scale_seconds);
            metric("gkr_init", -1, "phase1_unary_gate", init.phase1_unary_gate_seconds);
            metric("gkr_init", -1, "phase1_binary_gate", init.phase1_binary_gate_seconds);
            metric("gkr_init", -1, "phase2_total", init.phase2_total_seconds);
            metric("gkr_init", -1, "phase2_workspace", init.phase2_workspace_seconds);
            metric("gkr_init", -1, "phase2_witness_buffer", init.phase2_witness_buffer_seconds);
            metric("gkr_init", -1, "phase2_beta", init.phase2_beta_seconds);
            metric("gkr_init", -1, "phase2_unary_gate", init.phase2_unary_gate_seconds);
            metric("gkr_init", -1, "phase2_binary_gate", init.phase2_binary_gate_seconds);
            metric("input_reduction_init", -1, "reset", input_init.reset_seconds);
            metric("input_reduction_init", -1, "equality", input_init.equality_seconds);
            metric("input_reduction_init", -1, "scatter", input_init.scatter_seconds);
            metric("input_reduction_init", -1, "parallel_routes", input_init.parallel_routes, "count");
            metric("input_reduction_init", -1, "serial_routes", input_init.serial_routes, "count");
            metric("verifier", -1, "lookup_total", lookup_proof.verifier_time);
            metric("verifier", -1, "lookup_batch_pcs", lookup_proof.mvkzg_batch_verify_time);
            metric("verifier", -1, "gkr_total", gkr_replay_verifier_seconds);
            metric("verifier", -1, "gkr_wiring", gkr_checker.verifier_wiring_time);
            metric("verifier", -1, "gkr_input_coefficient", gkr_checker.input_reduction_coefficient_time);
            metric("verifier", -1, "gkr_opening", gkr_checker.commit_open_verify_time);
            metric("lookup", -1, "total", lookup_proof.prover_time);
            for (std::size_t i = 0; i < lookup_proof.groups.size(); ++i) {
                const auto& group = lookup_proof.groups[i];
                metric("lookup", i, "group_index", group.group_index, "count");
                metric("lookup", i, "table_size", group.table_size, "count");
                metric("lookup", i, "padded_queries", group.query_count, "count");
                metric("lookup", i, "actual_queries", group.actual_query_count, "count");
                metric("lookup", i, "total", group.timing.total_seconds);
                metric("lookup", i, "query_commit", group.timing.query_commit_seconds);
                metric("lookup", i, "auxiliary_commit", group.timing.auxiliary_commit_seconds);
                metric("lookup", i, "relation", group.timing.relation_seconds);
                metric("lookup", i, "sumcheck", group.timing.sumcheck_seconds);
                metric("lookup", i, "opening", group.timing.opening_seconds);
            }
            for (const auto& checkpoint : profile_checkpoints)
                profile << "transcript,-1," << checkpoint.first << ',' << checkpoint.second << ",hex\n";
            profile.close();
            if (!profile) throw runtime_error("failed to save online profile");
        }
        return verified ? 0 : 1;
    } catch (const exception& error) {
        benchmark_report::Writer report;
        report.section("Final Result");
        report.field("status", "ERROR");
        cerr << "Error: " << error.what() << endl;
        return 1;
    }
}
