
#include "common/bit_utils.hpp"
#include "common/integer_arithmetic.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <omp.h>

#include "model/quant_plan.hpp"
#include "model/nonlinear_quant.hpp"
#include "model/attention_softmax.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::pair;
using std::string;
using std::vector;

using i64 = long long;
using i128 = __int128;

struct Model {
    int vocab_size = 1024;
    int attn_dim = 128;
    int headnum = 4;
    int headdim = 32;
    int linear_dim = 256;
    int layer_num = 1;

    GPT2QuantPlan quant_plan;
};

struct TraceWriter {
    string directory;

    explicit TraceWriter(string path) : directory(std::move(path)) {
        if (directory.empty()) return;
        if (::mkdir(directory.c_str(), 0755) != 0 && errno != EEXIST) {
            throw std::runtime_error("cannot create trace directory: " + directory);
        }
    }

    void save(const string& name, const vector<i64>& values, int rows, int cols) const {
        if (directory.empty()) return;
        if (values.size() != static_cast<size_t>(rows) * cols) {
            throw std::runtime_error(name + ": trace shape mismatch");
        }
        const string path = directory + "/" + name + ".bin";
        std::ofstream output(path, std::ios::binary);
        if (!output) throw std::runtime_error("cannot create trace file: " + path);
        output.write(reinterpret_cast<const char*>(values.data()),
                     values.size() * sizeof(values[0]));
        if (!output) throw std::runtime_error("short write to trace file: " + path);
    }
};

using zkagent_integer::next_power_of_two;

static void load_config(const string& path, Model& model,
                        bool& vocab_set, bool& attn_set, bool& depth_set,
                        bool& head_set, bool& headdim_set, bool& ffn_set) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return;
    }
    string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t eq = line.find('=');
        if (eq == string::npos) continue;
        string key = line.substr(0, eq);
        int value = std::atoi(line.substr(eq + 1).c_str());
        if (key == "vocab_size" && !vocab_set) model.vocab_size = value;
        else if (key == "n_embd" && !attn_set) model.attn_dim = value;
        else if (key == "n_layer" && !depth_set) model.layer_num = value;
        else if (key == "n_head" && !head_set) model.headnum = value;
        else if (key == "head_dim" && !headdim_set) model.headdim = value;
        else if (key == "ffn_dim" && !ffn_set) model.linear_dim = value;
    }
}

static void init_model(Model& model, const string& model_dir) {
    std::string error;
    if (!model.quant_plan.load(model_dir, &error)) {
        throw std::runtime_error(
            error + ". Run quant/prepare_gpt2_quant.py --model-dir " + model_dir);
    }
    if (!model.quant_plan.validateDimensions(
            model.layer_num, model.headnum, model.headdim, model.attn_dim,
            model.linear_dim, model.vocab_size, &error)) {
        throw std::runtime_error(error);
    }
}

static vector<i64> layernorm_q(const Model& model, const vector<i64>& input, int len,
                               const QuantLayerNormPlan& plan, int active_rows) {
    const int n = model.attn_dim;
    const auto coefficients = zkagent_integer::layernorm_coefficients(
        plan.term1_c, plan.term1_e, plan.term2_c, plan.term2_e);
    vector<i64> output((size_t)len * n, 0);
    #pragma omp parallel for
    for (int p = 0; p < active_rows; p++) {
        i128 sum = 0;
        for (int j = 0; j < n; j++) sum += input[(size_t)p * n + j];
        vector<i128> centered(n);
        for (int j = 0; j < n; j++) {
            centered[j] = (i128)input[(size_t)p * n + j] * n - sum;
        }
        const auto variance = zkagent_integer::layernorm_variance(
            centered.data(), centered.size(), plan.variance_bits);
        for (int j = 0; j < n; j++) {
            const auto rounded = zkagent_integer::layernorm_round(
                coefficients, plan.gamma[j], centered[j], plan.beta[j], variance.sigma);
            if (rounded.value < std::numeric_limits<i64>::min() ||
                rounded.value > std::numeric_limits<i64>::max())
                throw std::overflow_error("LayerNorm result does not fit int64");
            const i64 q = static_cast<i64>(rounded.value);
            if (std::llabs(q) > static_cast<i64>(plan.output_bound)) {
                throw std::runtime_error("LayerNorm output exceeds offline bound");
            }
            output[(size_t)p * n + j] = q;
        }
    }
    return output;
}

static vector<i64> fc_accumulator(const vector<i64>& input, int len,
                                  const QuantLinearPlan& plan) {
    vector<i64> output((size_t)len * plan.out_dim, 0);
    #pragma omp parallel for collapse(2)
    for (int p = 0; p < len; p++) {
        for (int o = 0; o < plan.out_dim; o++) {
            i128 sum = plan.bias[o];
            const int32_t* weight = plan.weight.data() + (size_t)o * plan.in_dim;
            const i64* row = input.data() + (size_t)p * plan.in_dim;
            for (int i = 0; i < plan.in_dim; i++) sum += (i128)row[i] * weight[i];
            if (sum < std::numeric_limits<i64>::min() ||
                sum > std::numeric_limits<i64>::max()) {
                throw std::runtime_error("linear accumulator does not fit int64");
            }
            i64 value = static_cast<i64>(sum);
            if (std::llabs(value) > static_cast<i64>(plan.accumulator_bound)) {
                throw std::runtime_error("linear accumulator exceeds offline bound");
            }
            output[(size_t)p * plan.out_dim + o] = value;
        }
    }
    return output;
}

static vector<i64> fc_immediate(const vector<i64>& input, int len,
                                const QuantLinearPlan& plan) {
    vector<i64> output = fc_accumulator(input, len, plan);
    #pragma omp parallel for
    for (size_t i = 0; i < output.size(); i++) {
        output[i] = zkagent_integer::round_dyadic<i64>(output[i], plan.requant);
        if (std::llabs(output[i]) > static_cast<i64>(plan.output_bound)) {
            throw std::runtime_error("requantized linear output exceeds R");
        }
    }
    return output;
}

static vector<i64> align_residual(const vector<i64>& left,
                                  const vector<i64>& right,
                                  const QuantResidualPlan& plan) {
    if (left.size() != right.size()) throw std::runtime_error("residual shape mismatch");
    vector<i64> output(left.size());
    #pragma omp parallel for
    for (size_t i = 0; i < left.size(); i++) {
        i64 q_left = zkagent_integer::round_dyadic<i64>(left[i], plan.left);
        i64 q_right = zkagent_integer::round_dyadic<i64>(right[i], plan.right);
        i128 sum = (i128)q_left + q_right;
        if (sum < -plan.output_bound || sum > plan.output_bound) {
            throw std::runtime_error("residual output exceeds offline bound");
        }
        output[i] = static_cast<i64>(sum);
    }
    return output;
}

static vector<i64> gelu_q(const vector<i64>& input, int len, int width,
                          const QuantBlockPlan& block) {
    const auto coefficients = zkagent_integer::gelu_coefficients(
        block.gelu_scale_c, block.gelu_scale_e);
    vector<i64> output((size_t)len * width, 0);
    #pragma omp parallel for
    for (size_t index = 0; index < output.size(); index++) {
        const auto result = zkagent_integer::gelu_round(coefficients, input[index]);
        if (result.rounded.value < std::numeric_limits<i64>::min() ||
            result.rounded.value > std::numeric_limits<i64>::max())
            throw std::overflow_error("GELU result does not fit int64");
        const i64 y = static_cast<i64>(result.rounded.value);
        output[index] = y;
    }
    return output;
}

static vector<i64> forward_transformer(const Model& model, const vector<int>& tokens, int len,
                                       const TraceWriter* trace = nullptr) {
    const GPT2QuantPlan& plan = model.quant_plan;
    const int real_len = (int)tokens.size();
    const int n = model.attn_dim;
    vector<i64> x_q((size_t)len * n, 0);
    if (len > plan.positions) throw std::runtime_error("sequence exceeds quantized position table");
    for (int token : tokens) {
        if (token < 0 || token >= model.vocab_size) {
            throw std::runtime_error("token outside vocabulary");
        }
    }
    #pragma omp parallel for
    for (int p = 0; p < real_len; p++) {
        for (int j = 0; j < n; j++) {
            x_q[(size_t)p * n + j] =
                plan.token_embedding[(size_t)tokens[p] * n + j] +
                plan.position_embedding[(size_t)p * n + j];
        }
    }

    for (int layer_index = 0; layer_index < model.layer_num; layer_index++) {
        const QuantBlockPlan& block = plan.blocks[layer_index];
        vector<i64> residual = x_q;
        vector<i64> ln1 = layernorm_q(model, x_q, len, block.ln1, real_len);
        vector<i64> Q = fc_immediate(ln1, len, block.q);
        vector<i64> K = fc_immediate(ln1, len, block.k);
        vector<i64> V = fc_immediate(ln1, len, block.v);
        if (trace) {
            const string prefix = "block_" + std::to_string(layer_index) + "_";
            trace->save(prefix + "ln1", ln1, len, n);
            vector<i64> qkv((size_t)len * 3 * n);
            for (int row = 0; row < len; ++row) {
                std::copy(Q.begin() + (size_t)row * n,
                          Q.begin() + (size_t)(row + 1) * n,
                          qkv.begin() + (size_t)row * 3 * n);
                std::copy(K.begin() + (size_t)row * n,
                          K.begin() + (size_t)(row + 1) * n,
                          qkv.begin() + (size_t)row * 3 * n + n);
                std::copy(V.begin() + (size_t)row * n,
                          V.begin() + (size_t)(row + 1) * n,
                          qkv.begin() + (size_t)row * 3 * n + 2 * n);
            }
            trace->save(prefix + "qkv", qkv, len, 3 * n);
        }

        vector<i64> attention((size_t)len * n, 0);
        #pragma omp parallel for collapse(2) schedule(dynamic, 1)
        for (int head = 0; head < model.headnum; head++) {
            for (int i = 0; i < len; i++) {
                const int head_start = head * model.headdim;
                vector<i64> scores((size_t)i + 1);
                for (int j = 0; j <= i; j++) {
                    i128 dot = 0;
                    for (int d = 0; d < model.headdim; d++) {
                        dot += (i128)Q[(size_t)i * n + head_start + d] *
                               K[(size_t)j * n + head_start + d];
                    }
                    if (dot < -block.attention.score_bound
                        || dot > block.attention.score_bound
                        || dot < std::numeric_limits<i64>::min()
                        || dot > std::numeric_limits<i64>::max()) {
                        throw std::runtime_error("attention score exceeds offline bound");
                    }
                    scores[j] = static_cast<i64>(dot);
                }
                AttentionSoftmaxRowWitness softmax = build_attention_softmax_row(
                    std::vector<ll>(scores.begin(), scores.end()), block.attention);
                if (!softmax.success || !verify_attention_softmax_row(
                        softmax, std::vector<ll>(scores.begin(), scores.end()),
                        block.attention)) {
                    throw std::runtime_error(
                        "attention Softmax witness failed: " + softmax.message);
                }
                for (int d = 0; d < model.headdim; d++) {
                    i128 weighted = 0;
                    for (int j = 0; j <= i; j++) {
                        weighted += (i128)V[(size_t)j * n + head_start + d]
                                  * softmax.probabilities[static_cast<size_t>(j)];
                    }
                    if (weighted < -block.attention.accumulator_bound
                        || weighted > block.attention.accumulator_bound) {
                        throw std::runtime_error(
                            "attention accumulator exceeds offline bound");
                    }
                    i64 output = zkagent_integer::floor_div_i128<i64>(
                        2 * weighted + block.attention.theta,
                        2 * block.attention.theta);
                    if (std::llabs(output)
                        > static_cast<i64>(block.attention.output_bound)) {
                        throw std::runtime_error(
                            "attention output exceeds offline zkLLM bound");
                    }
                    attention[(size_t)i * n + head_start + d] = output;
                }
            }
        }

        vector<i64> projection = fc_accumulator(attention, len, block.attn_proj);
        x_q = align_residual(residual, projection, block.attn_residual);
        if (trace) {
            const string prefix = "block_" + std::to_string(layer_index) + "_";
            trace->save(prefix + "softmax", attention, len, n);
            trace->save(prefix + "attn_resid", x_q, len, n);
        }

        residual = x_q;
        vector<i64> ln2 = layernorm_q(model, x_q, len, block.ln2, real_len);
        vector<i64> up = fc_immediate(ln2, len, block.ffn_up);
        vector<i64> activated = gelu_q(up, len, model.linear_dim, block);
        vector<i64> down = fc_accumulator(activated, len, block.ffn_down);
        x_q = align_residual(residual, down, block.ffn_residual);
        if (trace) {
            const string prefix = "block_" + std::to_string(layer_index) + "_";
            trace->save(prefix + "ln2", ln2, len, n);
            trace->save(prefix + "ffn_up", up, len, model.linear_dim);
            trace->save(prefix + "gelu", activated, len, model.linear_dim);
            trace->save(prefix + "ffn_resid", x_q, len, n);
        }
    }

    vector<i64> final_ln = layernorm_q(model, x_q, len, plan.final_ln, real_len);
    if (trace) trace->save("final_ln", final_ln, len, n);
    return final_ln;
}

static int predict_token_at(const Model& model, const vector<i64>& x_q, int pos) {
    int n = model.attn_dim;
    const i64* row = x_q.data() + (size_t)pos * n;
    const QuantLinearPlan& head = model.quant_plan.lm_head;
#ifdef _OPENMP
    int thread_count = omp_get_max_threads();
    vector<i64> local_max((size_t)thread_count, std::numeric_limits<i64>::min());
    vector<int> local_idx((size_t)thread_count, -1);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        i64 max_logit = std::numeric_limits<i64>::min();
        int max_idx = -1;
        #pragma omp for nowait
        for (int v = 0; v < model.vocab_size; v++) {
            i128 sum = 0;
            const int32_t* emb = head.weight.data() + (size_t)v * n;
            for (int j = 0; j < n; j++) {
                sum += (i128)row[j] * emb[j];
            }
            sum += head.bias[v];
            i64 q = static_cast<i64>(sum);
            if (q > max_logit || (q == max_logit && (max_idx < 0 || v < max_idx))) {
                max_logit = q;
                max_idx = v;
            }
        }
        local_max[tid] = max_logit;
        local_idx[tid] = max_idx;
    }

    i64 max_logit = std::numeric_limits<i64>::min();
    int max_idx = 0;
    for (int t = 0; t < thread_count; t++) {
        if (local_idx[t] < 0) {
            continue;
        }
        if (local_max[t] > max_logit || (local_max[t] == max_logit && local_idx[t] < max_idx)) {
            max_logit = local_max[t];
            max_idx = local_idx[t];
        }
    }
#else
    i64 max_logit = std::numeric_limits<i64>::min();
    int max_idx = 0;
    for (int v = 0; v < model.vocab_size; v++) {
        i128 sum = 0;
        const int32_t* emb = head.weight.data() + (size_t)v * n;
        for (int j = 0; j < n; j++) {
            sum += (i128)row[j] * emb[j];
        }
        sum += head.bias[v];
        i64 q = static_cast<i64>(sum);
        if (q > max_logit) {
            max_logit = q;
            max_idx = v;
        }
    }
#endif
    return max_idx;
}

static vector<int> generate_sequence(const Model& model, int seq_len, int start_token) {
    vector<int> tokens;
    tokens.reserve(seq_len);
    tokens.push_back(start_token);

    for (int step = 1; step < seq_len; step++) {
        int cur_len = (int)tokens.size();
        int padded_len = next_power_of_two(cur_len);
        vector<i64> x_q = forward_transformer(model, tokens, padded_len);
        int next_token = predict_token_at(model, x_q, cur_len - 1);
        tokens.push_back(next_token);
    }
    return tokens;
}

static void write_tokens(const string& path, const vector<int>& tokens) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    for (int t : tokens) {
        out << t << "\n";
    }
}

static vector<int> load_prompt_tokens(const string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open token file: " + path);
    }
    vector<int> tokens;
    string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        tokens.push_back(std::atoi(line.c_str()));
    }
    return tokens;
}

static vector<int> parse_prompt_tokens(const string& text) {
    vector<int> tokens;
    string buf;
    for (char ch : text) {
        if (ch == ',' || ch == ' ' || ch == '\t' || ch == '\n') {
            if (!buf.empty()) {
                tokens.push_back(std::atoi(buf.c_str()));
                buf.clear();
            }
        } else {
            buf.push_back(ch);
        }
    }
    if (!buf.empty()) {
        tokens.push_back(std::atoi(buf.c_str()));
    }
    return tokens;
}

static vector<int> filter_prompt_tokens(const vector<int>& tokens, int vocab_size) {
    vector<int> filtered;
    filtered.reserve(tokens.size());
    for (int t : tokens) {
        if (t >= 0 && t < vocab_size) {
            filtered.push_back(t);
        } else {
            cerr << "Warning: token " << t << " out of range [0, " << vocab_size << "), skipping" << endl;
        }
    }
    return filtered;
}

static vector<int> generate_from_prompt(const Model& model, const vector<int>& prompt, int seq_len) {
    vector<int> tokens = prompt;
    if ((int)tokens.size() >= seq_len) {
        tokens.resize(seq_len);
        return tokens;
    }

    while ((int)tokens.size() < seq_len) {
        int cur_len = (int)tokens.size();
        int padded_len = next_power_of_two(cur_len);
        vector<i64> x_q = forward_transformer(model, tokens, padded_len);
        int next_token = predict_token_at(model, x_q, cur_len - 1);
        tokens.push_back(next_token);
    }
    return tokens;
}

static void print_usage(const char* prog) {
    cout << "Usage: " << prog << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  --model-dir D       Model directory (required)" << endl;
    cout << "  --seq-len N         Sequence length (default: 4)" << endl;
    cout << "  --prompt-tokens F   Prompt token file (one ID per line)" << endl;
    cout << "  --prompt LIST       Prompt token IDs (comma/space-separated)" << endl;
    cout << "  --output F          Output token file" << endl;
    cout << "  --lengths LIST      Comma-separated lengths (batch mode)" << endl;
    cout << "  --output-dir D      Output directory for batch mode" << endl;
    cout << "  --trace-dir D       Dump the final full-sequence integer trace" << endl;
    cout << "  --start-token N     Start token when max-start-tokens <= 1 (default: 0)" << endl;
    cout << "  --max-start-tokens N  Search [0, N) start tokens (default: 20)" << endl;
    cout << "  --min-unique N      Early stop when unique >= N (default: 0)" << endl;
    cout << "  --threads N         Number of threads (default: 128)" << endl;
    cout << "  --depth N           Override layer count" << endl;
    cout << "  --heads N           Override head count" << endl;
    cout << "  --head-dim N        Override head dim" << endl;
    cout << "  --hidden-dim N      Override hidden dim" << endl;
    cout << "  --ffn-dim N         Override FFN dim" << endl;
    cout << "  --vocab-size N      Override vocab size" << endl;
    cout << "  --help              Show this help" << endl;
}

static vector<int> parse_lengths(const string& text) {
    vector<int> result;
    string token;
    for (char ch : text) {
        if (ch == ',' || ch == ' ') {
            if (!token.empty()) {
                result.push_back(std::atoi(token.c_str()));
                token.clear();
            }
        } else {
            token.push_back(ch);
        }
    }
    if (!token.empty()) {
        result.push_back(std::atoi(token.c_str()));
    }
    result.erase(std::remove(result.begin(), result.end(), 0), result.end());
    if (result.empty()) {
        throw std::runtime_error("No valid lengths provided");
    }
    return result;
}

int main(int argc, char** argv) {
    string model_dir;
    int seq_len = 4;
    string output_path;
    string lengths_text;
    string output_dir = "./benchmark_tokens";
    string trace_dir;
    int start_token = 0;
    int max_start_tokens = 20;
    int min_unique = 0;
    int threads = 128;
    string prompt_tokens_path;
    string prompt_inline;

    Model model;
    bool vocab_set = false;
    bool attn_set = false;
    bool depth_set = false;
    bool head_set = false;
    bool headdim_set = false;
    bool ffn_set = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) {
            seq_len = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prompt-tokens") == 0 && i + 1 < argc) {
            prompt_tokens_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt_inline = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--lengths") == 0 && i + 1 < argc) {
            lengths_text = argv[++i];
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (strcmp(argv[i], "--trace-dir") == 0 && i + 1 < argc) {
            trace_dir = argv[++i];
        } else if (strcmp(argv[i], "--start-token") == 0 && i + 1 < argc) {
            start_token = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-start-tokens") == 0 && i + 1 < argc) {
            max_start_tokens = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--min-unique") == 0 && i + 1 < argc) {
            min_unique = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
            model.layer_num = std::atoi(argv[++i]);
            depth_set = true;
        } else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) {
            model.headnum = std::atoi(argv[++i]);
            head_set = true;
        } else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) {
            model.headdim = std::atoi(argv[++i]);
            headdim_set = true;
        } else if (strcmp(argv[i], "--hidden-dim") == 0 && i + 1 < argc) {
            model.attn_dim = std::atoi(argv[++i]);
            attn_set = true;
        } else if (strcmp(argv[i], "--ffn-dim") == 0 && i + 1 < argc) {
            model.linear_dim = std::atoi(argv[++i]);
            ffn_set = true;
        } else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) {
            model.vocab_size = std::atoi(argv[++i]);
            vocab_set = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (model_dir.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    if (threads <= 0) {
        threads = 1;
    }
    omp_set_num_threads(threads);

    string config_path = model_dir + "/config.txt";
    load_config(config_path, model, vocab_set, attn_set, depth_set,
                head_set, headdim_set, ffn_set);

    if (!headdim_set && attn_set && model.headnum > 0) {
        model.headdim = model.attn_dim / model.headnum;
    }
    if (model.attn_dim != model.headnum * model.headdim) {
        cerr << "Warning: hidden_dim (" << model.attn_dim << ") != heads (" << model.headnum
             << ") * head_dim (" << model.headdim << ")" << endl;
    }

    try {
        init_model(model, model_dir);
    } catch (const std::exception& ex) {
        cerr << ex.what() << endl;
        return 1;
    }

    vector<int> prompt_tokens;
    if (!prompt_inline.empty() || !prompt_tokens_path.empty()) {
        try {
            if (!prompt_inline.empty()) {
                prompt_tokens = parse_prompt_tokens(prompt_inline);
            } else {
                prompt_tokens = load_prompt_tokens(prompt_tokens_path);
            }
        } catch (const std::exception& ex) {
            cerr << ex.what() << endl;
            return 1;
        }
        prompt_tokens = filter_prompt_tokens(prompt_tokens, model.vocab_size);
        if (prompt_tokens.empty()) {
            cerr << "No valid prompt tokens after filtering" << endl;
            return 1;
        }
    }

    vector<int> lengths;
    if (!lengths_text.empty()) {
        try {
            lengths = parse_lengths(lengths_text);
        } catch (const std::exception& ex) {
            cerr << ex.what() << endl;
            return 1;
        }
    } else {
        lengths.push_back(seq_len);
    }
    if (!trace_dir.empty() && lengths.size() != 1) {
        cerr << "--trace-dir requires a single --seq-len run" << endl;
        return 1;
    }

    for (int L : lengths) {
        if (L <= 0) {
            cerr << "Invalid length: " << L << endl;
            return 1;
        }
        vector<int> best_tokens;
        int best_unique = -1;
        if (!prompt_tokens.empty()) {
            if ((int)prompt_tokens.size() > L) {
                cerr << "Warning: prompt length " << prompt_tokens.size()
                     << " exceeds target length " << L << ", truncating prompt" << endl;
            }
            best_tokens = generate_from_prompt(model, prompt_tokens, L);
        } else if (max_start_tokens <= 1) {
            best_tokens = generate_sequence(model, L, start_token);
        } else {
            for (int s = 0; s < max_start_tokens; s++) {
                vector<int> seq = generate_sequence(model, L, s);
                int unique = (int)std::set<int>(seq.begin(), seq.end()).size();
                if (unique > best_unique) {
                    best_unique = unique;
                    best_tokens = std::move(seq);
                }
                if (min_unique > 0 && best_unique >= min_unique) {
                    break;
                }
            }
        }

        string out_path = output_path;
        if (!lengths_text.empty()) {
            out_path = output_dir + "/tokens_len" + std::to_string(L) + ".txt";
        } else if (out_path.empty()) {
            out_path = "./benchmark_tokens/tokens_len" + std::to_string(L) + ".txt";
        }
        try {
            write_tokens(out_path, best_tokens);
        } catch (const std::exception& ex) {
            cerr << ex.what() << endl;
            return 1;
        }

        if (!trace_dir.empty()) {
            const int padded_len = next_power_of_two((int)best_tokens.size());
            TraceWriter trace(trace_dir);
            forward_transformer(
                model, best_tokens, padded_len, &trace);
        }
        cout << "Wrote " << best_tokens.size() << " tokens to " << out_path << endl;
    }

    return 0;
}
