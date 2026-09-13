#undef NDEBUG

#include "common/global_var.hpp"
#include "common/omp_threads.hpp"
#include "pcs/srs.hpp"
#include "zkdecode/zkdecode_prover.hpp"
#include "zkdecode/zkdecode_verifier.hpp"
#include "common/bit_utils.hpp"
#include "pcs/hyrax.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mcl/bn256.hpp>

using namespace mcl::bn;
using namespace std;

namespace
{

    string i128ToString(ll value)
    {
        if (value == 0)
        {
            return "0";
        }
        bool negative = value < 0;
        ll current = negative ? -value : value;
        string out;
        while (current > 0)
        {
            int digit = static_cast<int>(current % 10);
            out.push_back(static_cast<char>('0' + digit));
            current /= 10;
        }
        if (negative)
        {
            out.push_back('-');
        }
        reverse(out.begin(), out.end());
        return out;
    }

    vector<ll> makeDemoLogits(int vocab, std::mt19937_64 &rng, ll bound)
    {
        vector<ll> logits(vocab, 0);
        const ll span = min<ll>(max<ll>(bound / 8, 64), bound - 1);
        std::uniform_int_distribution<int64_t> distribution(
            -static_cast<int64_t>(span), static_cast<int64_t>(span));
        for (ll &value : logits)
            value = static_cast<ll>(distribution(rng));
        return logits;
    }

    pair<int, ll> pickTokenFromMass(const vector<ll> &mass, ll sample_value)
    {
        ll prefix = 0;
        for (size_t i = 0; i < mass.size(); ++i)
        {
            if (sample_value < prefix + mass[i])
            {
                return {static_cast<int>(i), prefix};
            }
            prefix += mass[i];
        }
        return {-1, prefix};
    }

    void printBenchmarkResult(const char *mode, const DecodeProofStats &stats,
                              const ZkDecodeVerifyStats &verification,
                              const ZkDecodeProof &proof,
                              int vocab, int tokens, int topk, int threads,
                              bool shout)
    {
        cout << "DECODE_RESULT mode=" << mode
             << " backend=" << (shout ? "SHOUT" : "LogUp-GKR")
             << " vocab=" << vocab << " tokens=" << tokens
             << " k=" << topk << " threads=" << threads
             << setprecision(12)
             << " prover_seconds=" << stats.prover_time
             << " verifier_seconds=" << verification.total_seconds
             << " proof_bytes=" << zkDecodeProofPayloadBytes(proof)
             << " verified=" << (verification.success ? "yes" : "no")
             << setprecision(3) << endl;
    }

    struct BenchmarkVerifierSetup
    {
        ZkDecodeStatement statement;
        ZkDecodeVerifierKey key;
        double seconds = 0.0;
    };

    BenchmarkVerifierSetup prepareVerifier(
        const vector<vector<ll>> &logits, const vector<int> &outputs,
        const vector<ll> &samples, LookupBackend backend, bool topk, int k,
        ll bound, ll epsilon, const MVKZG_ProverKey &pk,
        const MVKZG_VerifierKey &vk, int threads)
    {
        const auto start = chrono::steady_clock::now();
        BenchmarkVerifierSetup setup;
        auto &p = setup.statement.parameters;
        p.kind = topk ? ZkDecodeProofKind::TopK : ZkDecodeProofKind::Greedy;
        p.lookup_backend = backend;
        p.token_count = static_cast<int>(logits.size());
        p.vocab_size = static_cast<int>(logits.front().size());
        p.k = topk ? k : 0;
        p.bound = bound;
        p.theta = topk ? ZkDecodeProbabilityScale : 0;
        p.epsilon_sm = topk ? epsilon : 0;
        setup.key = prepareZkDecodeVerifierKey(p, pk, vk, threads);
        setup.statement.output_tokens = outputs;
        setup.statement.sample_values = samples;

        // Fix the public source commitment from the benchmark inputs, independently of the proof.
        const int token_bits = zkagent_integer::ceil_log2(p.token_count);
        const int vocab_bits = zkagent_integer::ceil_log2(p.vocab_size);
        const int variables = token_bits + vocab_bits;
        const size_t width = size_t(1) << vocab_bits;
        vector<ll> padded((size_t(1) << token_bits) * width, 0);
        for (size_t t = 0; t < logits.size(); ++t)
            copy(logits[t].begin(), logits[t].end(), padded.begin() + t * width);
        const size_t columns = size_t(1) << (variables - variables / 2);
        if (pk.srs_g1.size() < columns)
            throw invalid_argument("SRS is too small for the public logit commitment");
        vector<G1> generators(pk.srs_g1.begin(), pk.srs_g1.begin() + columns);
        G1 *commitments = hyrax::commit(padded.data(), generators.data(), variables, threads);
        setup.statement.logit_commitments.assign(
            commitments, commitments + (size_t(1) << (variables / 2)));
        delete[] commitments;
        setup.seconds = chrono::duration<double>(chrono::steady_clock::now() - start).count();
        return setup;
    }

    void print_usage(const char *program, ostream &out = cout)
    {
        out << "Usage: " << program << " [options]\n"
            << "  --decode greedy|topk               Decoding mode\n"
            << "  --vocab-size N                     Vocabulary size\n"
            << "  --tokens N                         Decoding steps in the batch\n"
            << "  --k N                              Top-k candidate count\n"
            << "  --threads N                        Proof-generation threads\n"
            << "  --lookup-backend shout|logup-gkr    Lookup backend\n"
            << "  --decode-seed N                    Synthetic-input and sampling seed\n"
            << "  --bound-bits N                     Log2 of the public logit bound\n"
            << "  --bound B                          Exact bound; nonzero overrides --bound-bits\n"
            << "  --epsilon-sm N                     Row-mass tolerance; -1 derives Q20 default\n"
            << "  --lookup-g1-encoding compressed|affine64\n"
            << "  --help, -h                         Show this help\n";
    }

    template <class Integer>
    Integer parse_integer(const char *value, const char *option)
    {
        size_t consumed = 0;
        long long parsed;
        try
        {
            parsed = stoll(value, &consumed);
        }
        catch (const exception &)
        {
            throw invalid_argument(string(option) + " requires an integer");
        }
        if (consumed != strlen(value) || parsed < numeric_limits<Integer>::min() ||
            parsed > numeric_limits<Integer>::max())
        {
            throw invalid_argument(string(option) + " has an invalid or out-of-range integer");
        }
        return static_cast<Integer>(parsed);
    }

}

int main(int argc, char **argv)
{
    string decode_mode = "greedy";
    int vocab = 1024;
    int topk = 40;
    constexpr double tau = ZkDecodeTemperature;
    constexpr ll theta = ZkDecodeProbabilityScale;
    int epsilon_sm = -1;
    int threads = 8;
    int decode_seed = 20260421;
    int bound_bits = 20;
    long long bound_value = 0;
    int tokens = 1;
    LookupBackend lookup_backend = LookupBackend::LogUpGkr;
    string lookup_g1_encoding = "compressed";

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            const char *option = argv[i];
            auto require_value = [&]() -> const char *
            {
                if (i + 1 >= argc)
                    throw invalid_argument(string(option) + " requires a value");
                return argv[++i];
            };
            if (strcmp(option, "--decode") == 0)
                decode_mode = require_value();
            else if (strcmp(option, "--vocab-size") == 0)
                vocab = parse_integer<int>(require_value(), option);
            else if (strcmp(option, "--tokens") == 0)
                tokens = parse_integer<int>(require_value(), option);
            else if (strcmp(option, "--k") == 0)
                topk = parse_integer<int>(require_value(), option);
            else if (strcmp(option, "--threads") == 0)
                threads = parse_integer<int>(require_value(), option);
            else if (strcmp(option, "--decode-seed") == 0)
                decode_seed = parse_integer<int>(require_value(), option);
            else if (strcmp(option, "--bound-bits") == 0)
                bound_bits = parse_integer<int>(require_value(), option);
            else if (strcmp(option, "--bound") == 0)
                bound_value = parse_integer<long long>(require_value(), option);
            else if (strcmp(option, "--epsilon-sm") == 0)
                epsilon_sm = parse_integer<int>(require_value(), option);
            else if (strcmp(option, "--lookup-g1-encoding") == 0)
                lookup_g1_encoding = require_value();
            else if (strcmp(option, "--lookup-backend") == 0)
            {
                const string backend = require_value();
                if (backend == "shout")
                    lookup_backend = LookupBackend::Shout;
                else if (backend == "logup-gkr")
                    lookup_backend = LookupBackend::LogUpGkr;
                else
                    throw invalid_argument("--lookup-backend must be shout or logup-gkr");
            }
            else if (strcmp(option, "--help") == 0 || strcmp(option, "-h") == 0)
            {
                print_usage(argv[0]);
                return 0;
            }
            else
                throw invalid_argument(string("Unknown option: ") + option);
        }
    }
    catch (const exception &error)
    {
        cerr << "Error: " << error.what() << endl;
        print_usage(argv[0], cerr);
        return 1;
    }

    if (decode_mode != "greedy" && decode_mode != "topk")
    {
        cerr << "Error: decode must be one of greedy|topk" << endl;
        return 1;
    }
    if (lookup_g1_encoding != "affine64" && lookup_g1_encoding != "compressed")
    {
        cerr << "Error: --lookup-g1-encoding must be affine64 or compressed" << endl;
        return 1;
    }
    if (vocab <= 0 || topk <= 0 || topk > vocab || epsilon_sm < -1 || threads <= 0 || bound_bits <= 0 || bound_bits >= 60 || bound_value < 0 || tokens <= 0 || static_cast<std::size_t>(tokens) > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(vocab))
    {
        cerr << "Error: invalid benchmark parameters" << endl;
        return 1;
    }
    const bool use_shout = lookup_backend == LookupBackend::Shout;
    const ScopedOmpThreads thread_scope(threads);
    initPairing(mcl::BN_SNARK1);

    const ll bound = bound_value > 0
                         ? static_cast<ll>(bound_value)
                         : static_cast<ll>(1) << bound_bits;
    std::mt19937_64 rng(static_cast<uint64_t>(decode_seed));
    vector<ll> logits = makeDemoLogits(vocab, rng, bound);
    const std::size_t query_count =
        static_cast<std::size_t>(vocab) * static_cast<std::size_t>(std::max(tokens, 1));
    const int query_vars = MVKZG_SRS::min_vars_for_size(
        std::max<std::size_t>(query_count, 1));
    const int shout_column_vars = (10 + query_vars + 1) / 2;
    SRSPlan srs_plan;
    srs_plan.mv_vars = std::max(15, shout_column_vars);
    SRSBundle srs;
    srs.setup_cached(srs_plan, "cache");
    ZkDecodeProver decoder(srs.mv.pk, srs.mv.vk, threads);
    decoder.setLookupBackend(lookup_backend);
    decoder.setLookupG1Encoding(
        lookup_g1_encoding == "compressed"
            ? G1WireEncoding::Compressed
            : G1WireEncoding::Affine64);
    if (decode_mode == "greedy")
        epsilon_sm = 0;
    else if (epsilon_sm < 0)
        epsilon_sm = static_cast<int>(
            decode_exp_digits_recommended_epsilon(theta, topk, DecodeExpDigitsConfig{}));
    decoder.setSoftmaxEpsilon(epsilon_sm);

    cout << fixed << setprecision(3);
    cout << "DECODE_CONFIG mode=" << decode_mode
         << " backend=" << (use_shout ? "SHOUT" : "LogUp-GKR")
         << " vocab=" << vocab
         << " tokens=" << tokens
         << " k=" << topk
         << " threads=" << threads
         << " tau=" << tau
         << " theta=" << i128ToString(theta)
         << " softmax=Q20-zkllm-digits"
         << " epsilon_sm=" << epsilon_sm
         << " bound=" << i128ToString(bound)
         << " lookup_g1_encoding=" << lookup_g1_encoding
         << " seed=" << decode_seed
         << " rng=mt19937_64"
         << " commitment_accounting=precommitted-advice-microbenchmark"
         << endl;

    if (decode_mode == "greedy")
    {

        vector<vector<ll>> logits_all(tokens);
        vector<int> outputs_all(tokens);
        logits_all[0] = logits;
        outputs_all[0] = static_cast<int>(max_element(logits.begin(), logits.end()) - logits.begin());
        for (int t = 1; t < tokens; ++t)
        {
            logits_all[t] = makeDemoLogits(vocab, rng, bound);
            outputs_all[t] = static_cast<int>(max_element(logits_all[t].begin(), logits_all[t].end()) - logits_all[t].begin());
        }
        const auto verifier_setup = prepareVerifier(
            logits_all, outputs_all, {}, lookup_backend, false, 0,
            bound, 0, srs.mv.pk, srs.mv.vk, threads);
        auto generated = decoder.proveGreedyBatch(logits_all, outputs_all, bound);
        const auto &stats = generated.stats;
        ZkDecodeVerifyStats verification;
        if (stats.success)
            verifyZkDecodeProof(verifier_setup.statement, generated.proof,
                                verifier_setup.key, srs.mv.vk, &verification);
        cout << "DECODE_PREPARATION prover_seconds=" << stats.setup_time
             << " verifier_seconds=" << verifier_setup.seconds << endl;
        if (!stats.success)
            cerr << "Error: zkDecode proof generation failed: " << stats.message << endl;
        else if (!verification.success)
            cerr << "Error: zkDecode verification failed: " << verification.message << endl;
        printBenchmarkResult("greedy", stats, verification, generated.proof,
                             vocab, tokens, topk, threads, use_shout);
        return verification.success ? 0 : 1;
    }
    else
    {

        vector<vector<ll>> logits_all(tokens);
        vector<int> outputs_all(tokens);
        vector<ll> samples_all(tokens);

        logits_all[0] = logits;
        for (int t = 0; t < tokens; ++t)
        {
            if (t > 0)
            {
                logits_all[t] = makeDemoLogits(vocab, rng, bound);
            }
            vector<ll> mass = decoder.computeSamplingMassTopK(
                logits_all[t], topk);
            ll total_mass = accumulate(mass.begin(), mass.end(), static_cast<ll>(0));
            samples_all[t] = static_cast<ll>(
                rng() % static_cast<uint64_t>(max<ll>(total_mass, 1)));
            pair<int, ll> picked = pickTokenFromMass(mass, samples_all[t]);
            if (picked.first < 0)
            {
                cerr << "Error: failed to derive a valid top-k sample at token " << t << endl;
                return 1;
            }
            outputs_all[t] = picked.first;
        }
        const auto verifier_setup = prepareVerifier(
            logits_all, outputs_all, samples_all, lookup_backend, true, topk,
            bound, epsilon_sm, srs.mv.pk, srs.mv.vk, threads);
        auto generated = decoder.proveTopKBatch(
            logits_all, outputs_all, topk, samples_all, bound);
        const auto &stats = generated.stats;
        ZkDecodeVerifyStats verification;
        if (stats.success)
            verifyZkDecodeProof(verifier_setup.statement, generated.proof,
                                verifier_setup.key, srs.mv.vk, &verification);
        cout << "DECODE_PREPARATION prover_seconds=" << stats.setup_time
             << " verifier_seconds=" << verifier_setup.seconds << endl;
        if (!stats.success)
            cerr << "Error: zkDecode proof generation failed: " << stats.message << endl;
        else if (!verification.success)
            cerr << "Error: zkDecode verification failed: " << verification.message << endl;
        printBenchmarkResult("topk", stats, verification, generated.proof,
                             vocab, tokens, topk, threads, use_shout);
        return verification.success ? 0 : 1;
    }
}
