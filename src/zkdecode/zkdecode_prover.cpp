#include "zkdecode/zkdecode_prover.hpp"
#include "zkdecode/zkdecode_protocol.hpp"
#include "zkdecode/zkdecode_input_link.hpp"
#include "common/fiat_shamir.hpp"
#include "common/omp_threads.hpp"

#include "zkdecode/zkdecode_arithmetic.hpp"
#include "zkdecode/zkdecode_lookup.hpp"
#include "zkdecode/zkdecode_output.hpp"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

using std::vector;

namespace
{

    using Clock = std::chrono::steady_clock;

    struct SelectionResult
    {
        vector<int> active;
        vector<ll> shifted_logits;
        vector<ll> quantized_mass;
        ll separator = 0;
        ll total_mass = 0;
        DecodeExpDigitsWitness exp_digits;
    };

    void appendLookupProofs(
        ZkDecodeProof &proof, DecodeProofStats &stats, DecodeLookupResult lookup)
    {
        stats.setup_time += lookup.setup_time;
        stats.prover_time += lookup.prover_time;
        proof.lookup_proofs.insert(
            proof.lookup_proofs.end(),
            std::make_move_iterator(lookup.proof_records.begin()),
            std::make_move_iterator(lookup.proof_records.end()));
    }

    double secondsSince(const Clock::time_point &start)
    {
        return std::chrono::duration<double>(Clock::now() - start).count();
    }

    using zkagent_integer::next_power_of_two;

    SelectionResult buildTopKSelection(
        const vector<ll> &logits,
        int k,
        const DecodeExpDigitsConfig &config)
    {
        constexpr ll theta = ZkDecodeProbabilityScale;
        SelectionResult result;
        const int vocab = static_cast<int>(logits.size());
        result.active.assign(vocab, 0);
        result.shifted_logits.assign(vocab, 0);

        vector<int> order(vocab);
        std::iota(order.begin(), order.end(), 0);
        k = std::max(1, std::min(k, vocab));
        std::nth_element(order.begin(), order.begin() + k - 1, order.end(),
                         [&](int lhs, int rhs)
                         {
                             if (logits[lhs] != logits[rhs])
                                 return logits[lhs] > logits[rhs];
                             return lhs < rhs;
                         });
        result.separator = logits[order[k - 1]];
        for (int i = 0; i < vocab; ++i)
        {
            result.shifted_logits[i] = logits[i] - result.separator;
        }
        for (int rank = 0; rank < k; ++rank)
        {
            const int idx = order[rank];
            result.active[idx] = 1;
        }

        result.exp_digits = build_decode_exp_digits_witness(
            result.shifted_logits, result.active, theta, config);
        if (!result.exp_digits.success)
            return result;
        result.quantized_mass = std::move(result.exp_digits.unmasked_outputs);
        for (int i = 0; i < vocab; ++i)
        {
            if (!result.active[i])
                result.quantized_mass[i] = 0;
            result.total_mass += result.quantized_mass[i];
        }
        return result;
    }

    struct PreparedTopKColumns
    {
        int vocab_domain = 0;
        vector<ll> candidate_in;
        vector<ll> candidate_out;
        DecodeExpDigitsWitness digits;
        bool candidates_valid = true;
        bool softmax_valid = true;
        ll epsilon = 0;
    };

    PreparedTopKColumns prepareTopKColumns(
        vector<SelectionResult> &selections, const vector<int> &outputs,
        int vocab, int k, ll bound, const DecodeExpDigitsConfig &config,
        ll requested_epsilon, double &prover_time)
    {
        PreparedTopKColumns result;
        const int tokens = static_cast<int>(selections.size());
        result.vocab_domain = next_power_of_two(vocab);
        const size_t domain = static_cast<size_t>(next_power_of_two(tokens)) * result.vocab_domain;
        const auto start = Clock::now();
        result.candidate_in.assign(domain, 0);
        result.candidate_out.assign(domain, 0);
        for (int token = 0; token < tokens; ++token)
        {
            const auto &selection = selections[token];
            int active_count = 0;
            if (!selection.active[outputs[token]])
                result.candidates_valid = false;
            for (int index = 0; index < vocab; ++index)
            {
                const int active = selection.active[index];
                active_count += active;
                const ll delta = selection.shifted_logits[index];
                const ll inside = active ? delta : 0;
                const ll outside = active ? 0 : -delta;
                if ((active != 0 && active != 1) || inside < 0 || inside >= 2 * bound ||
                    outside < 0 || outside >= 2 * bound)
                    result.candidates_valid = false;
                const size_t target = static_cast<size_t>(token) * result.vocab_domain + index;
                result.candidate_in[target] = inside;
                result.candidate_out[target] = outside;
            }
            if (active_count != k)
                result.candidates_valid = false;
        }
        result.epsilon = requested_epsilon >= 0
                             ? requested_epsilon
                             : decode_exp_digits_recommended_epsilon(
                                   ZkDecodeProbabilityScale, k, config);
        const auto &first = selections.front().exp_digits;
        result.digits.success = first.success;
        result.digits.config = first.config;
        result.digits.output_scale = first.output_scale;
        result.digits.nonlinear_outputs.assign(domain, ZkDecodeProbabilityScale);
        result.digits.digits.assign(config.bases.size(), vector<ll>(domain, 0));
        result.digits.high_zero_flags.assign(config.most_zero_segments, vector<ll>(domain, 1));
        for (int token = 0; token < tokens; ++token)
        {
            auto &selection = selections[token];
            auto &digits = selection.exp_digits;
            if (!digits.success || selection.total_mass <= 0)
            {
                result.softmax_valid = false;
                result.digits.success = false;
                continue;
            }
            const size_t base = static_cast<size_t>(token) * result.vocab_domain;
            std::copy(digits.nonlinear_outputs.begin(), digits.nonlinear_outputs.end(),
                      result.digits.nonlinear_outputs.begin() + base);
            for (size_t segment = 0; segment < digits.digits.size(); ++segment)
                std::copy(digits.digits[segment].begin(), digits.digits[segment].end(),
                          result.digits.digits[segment].begin() + base);
            for (size_t high = 0; high < digits.high_zero_flags.size(); ++high)
                std::copy(digits.high_zero_flags[high].begin(), digits.high_zero_flags[high].end(),
                          result.digits.high_zero_flags[high].begin() + base);
            // The shared padded columns own these values for arithmetic and lookup.
            vector<ll>().swap(digits.nonlinear_outputs);
            vector<vector<ll>>().swap(digits.digits);
            vector<vector<ll>>().swap(digits.high_zero_flags);
        }
        prover_time += secondsSince(start);
        return result;
    }

    DecodeTopKLinkWitnessView buildTopKLinkWitnessView(
        const vector<vector<ll>> &logits_per_token,
        const vector<SelectionResult> &selections,
        const PreparedTopKColumns &columns, int k)
    {
        DecodeTopKLinkWitnessView view;
        const int token_count = static_cast<int>(logits_per_token.size());
        if (token_count <= 0 || selections.size() != logits_per_token.size() || !columns.digits.success)
        {
            return view;
        }
        view.token_count = token_count;
        view.vocab_size = static_cast<int>(logits_per_token[0].size());
        view.k = k;
        view.logits.resize(static_cast<size_t>(token_count));
        view.deltas.resize(static_cast<size_t>(token_count));
        view.selectors.resize(static_cast<size_t>(token_count));
        view.candidate_in.resize(static_cast<size_t>(token_count));
        view.candidate_out.resize(static_cast<size_t>(token_count));
        view.middle_outputs.resize(static_cast<size_t>(token_count));
        view.high_zero_0.resize(static_cast<size_t>(token_count));
        view.high_zero_1.resize(static_cast<size_t>(token_count));
        view.probabilities.resize(static_cast<size_t>(token_count));
        for (auto &digit : view.digits)
        {
            digit.resize(static_cast<size_t>(token_count));
        }
        view.separators.resize(static_cast<size_t>(token_count));
        view.shifts.resize(static_cast<size_t>(token_count));

        for (int token = 0; token < token_count; ++token)
        {
            const size_t t = static_cast<size_t>(token);
            const auto &selection = selections[t];
            if (static_cast<int>(logits_per_token[t].size()) != view.vocab_size ||
                static_cast<int>(selection.shifted_logits.size()) != view.vocab_size ||
                static_cast<int>(selection.active.size()) != view.vocab_size ||
                static_cast<int>(selection.quantized_mass.size()) != view.vocab_size)
            {
                return DecodeTopKLinkWitnessView{};
            }
            view.logits[t] = logits_per_token[t].data();
            view.deltas[t] = selection.shifted_logits.data();
            view.selectors[t] = selection.active.data();
            const size_t base = t * columns.vocab_domain;
            view.candidate_in[t] = columns.candidate_in.data() + base;
            view.candidate_out[t] = columns.candidate_out.data() + base;
            for (int digit = 0; digit < 6; ++digit)
            {
                view.digits[static_cast<size_t>(digit)][t] =
                    columns.digits.digits[static_cast<size_t>(digit)].data() + base;
            }
            view.middle_outputs[t] = columns.digits.nonlinear_outputs.data() + base;
            view.high_zero_0[t] = columns.digits.high_zero_flags[0].data() + base;
            view.high_zero_1[t] = columns.digits.high_zero_flags[1].data() + base;
            view.probabilities[t] = selection.quantized_mass.data();
            view.separators[t] = selection.separator;
            view.shifts[t] = selection.exp_digits.shift;
        }
        return view;
    }

    struct PreparedTopKWitness
    {
        vector<SelectionResult> selections;
        PreparedTopKColumns columns;

        void releaseCandidates()
        {
#if defined(__GLIBC__)
            malloc_trim(0);
#endif
            vector<ll>().swap(columns.candidate_in);
            vector<ll>().swap(columns.candidate_out);
        }

        void releaseSelectionInputs()
        {
            for (auto &selection : selections)
            {
                vector<ll>().swap(selection.shifted_logits);
                vector<int>().swap(selection.active);
            }
#if defined(__GLIBC__)
            malloc_trim(0);
#endif
        }
    };

    PreparedTopKWitness prepareTopKWitness(
        const vector<vector<ll>> &logits, const vector<int> &outputs, int k, ll bound,
        const DecodeExpDigitsConfig &config, ll epsilon, double &prover_time)
    {
        PreparedTopKWitness witness;
        witness.selections.reserve(logits.size());
        for (const auto &row : logits)
            witness.selections.push_back(buildTopKSelection(row, k, config));
        witness.columns = prepareTopKColumns(witness.selections, outputs,
            static_cast<int>(logits.front().size()), k, bound, config, epsilon, prover_time);
        return witness;
    }

}

ZkDecodeProver::ZkDecodeProver(const MVKZG_ProverKey &pk,
                               const MVKZG_VerifierKey &vk,
                               int threads)
    : pk_(pk), vk_(vk), threads_(threads) {}

ZkDecodeProveResult ZkDecodeProver::proveGreedyBatch(
    const vector<vector<ll>> &logits, const vector<int> &outputs, ll bound)
{
    return proveGreedyBatchImpl(logits, outputs, bound, zkdecode_detail::ExecutionContext::standalone());
}
ZkDecodeProveResult ZkDecodeProver::proveGreedyBatchToInput(
    const vector<vector<ll>> &logits, const vector<int> &outputs, ll bound,
    FiatShamirTranscript &transcript, DecodeLogitEvaluation &deferred_logit_claim)
{
    deferred_logit_claim = DecodeLogitEvaluation{};
    return proveGreedyBatchImpl(logits, outputs, bound, zkdecode_detail::ExecutionContext::linked(transcript, deferred_logit_claim));
}
ZkDecodeProveResult ZkDecodeProver::proveGreedyBatchImpl(
    const vector<vector<ll>> &logits_per_token,
    const vector<int> &output_per_token,
    ll bound, const zkdecode_detail::ExecutionContext &execution)
{
    const ScopedOmpThreads thread_scope(threads_);
    ZkDecodeProveResult result;
    auto &stats = result.stats;
    auto &proof = result.proof;
    const int T = static_cast<int>(logits_per_token.size());
    proof.parameters.kind = ZkDecodeProofKind::Greedy;
    proof.parameters.token_count = T;
    proof.parameters.vocab_size = T > 0
                                 ? static_cast<int>(logits_per_token[0].size())
                                 : 0;
    proof.parameters.bound = bound;
    proof.parameters.lookup_backend = lookup_backend_;
    proof.output_tokens = output_per_token;
    if (T == 0 || static_cast<int>(output_per_token.size()) != T || bound <= 0)
    {
        stats.message = "invalid batched greedy instance";
        return result;
    }
    const int N = static_cast<int>(logits_per_token[0].size());
    if (N == 0)
    {
        stats.message = "empty logits";
        return result;
    }

    for (int t = 0; t < T; ++t)
    {
        if (static_cast<int>(logits_per_token[t].size()) != N)
        {
            stats.message = "inconsistent N across tokens";
            return result;
        }
        int yt = output_per_token[t];
        if (yt < 0 || yt >= N)
        {
            stats.message = "invalid output token in batched greedy";
            return result;
        }
    }

    vector<ll> lambdas(static_cast<size_t>(T));
    vector<vector<ll>> neg_delta_rows(
        static_cast<size_t>(T), vector<ll>(static_cast<size_t>(N), 0));
    for (int t = 0; t < T; ++t)
    {
        int yt = output_per_token[t];
        ll lambda_t = logits_per_token[t][yt];
        lambdas[static_cast<size_t>(t)] = lambda_t;
        for (int i = 0; i < N; ++i)
        {
            neg_delta_rows[static_cast<size_t>(t)][static_cast<size_t>(i)] =
                lambda_t - logits_per_token[t][i];
        }
    }

    DecodeGreedyLinkResult greedy_link;
    DecodeBatchPointZeroResult zero_opening;
    DecodeGreedyLinkWitnessView view;
    view.token_count = T;
    view.vocab_size = N;
    view.lambdas = lambdas;
    view.logits.resize(static_cast<size_t>(T));
    view.negative_deltas.resize(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t)
    {
        view.logits[static_cast<size_t>(t)] =
            logits_per_token[static_cast<size_t>(t)].data();
        view.negative_deltas[static_cast<size_t>(t)] =
            neg_delta_rows[static_cast<size_t>(t)].data();
    }
    // Standalone components keep their independent Fiat-Shamir streams.
    // Input-linked components share the caller's transcript and deferred logit opening.
    FiatShamirTranscript link_transcript;
    FiatShamirTranscript zero_transcript;
    greedy_link = prove_decode_greedy_link(
        view, pk_, vk_, threads_, execution.transcript(link_transcript), execution.deferredLogitClaim());
    zero_opening = prove_decode_batch_point_zero(
        neg_delta_rows, output_per_token, pk_, vk_, threads_,
        execution.transcript(zero_transcript),
        &decode_greedy_negative_delta_commitment(greedy_link.proof));
    const bool arithmetic_ok = greedy_link.success && zero_opening.success;
    stats.prover_time = greedy_link.prover_time + zero_opening.prover_time;
    proof.has_greedy_link = true;
    proof.has_greedy_zero = true;
    proof.greedy_zero = std::move(zero_opening.proof);
    proof.greedy_link = std::move(greedy_link.proof);

    const Clock::time_point cand_begin = Clock::now();
    bool cand_ok = arithmetic_ok;
    const int vocab_domain = next_power_of_two(N);
    vector<ll> neg_deltas(
        static_cast<size_t>(next_power_of_two(T)) * vocab_domain, 0);
    for (int t = 0; t < T; ++t)
    {
        for (int i = 0; i < N; ++i)
        {
            const ll nd = neg_delta_rows[static_cast<size_t>(t)][static_cast<size_t>(i)];
            neg_deltas[static_cast<size_t>(t) * vocab_domain + i] = nd;
            if ((i == output_per_token[t] && nd != 0) || nd < 0 || nd >= 2 * bound)
                cand_ok = false;
        }
    }
    stats.prover_time += secondsSince(cand_begin);
    if (cand_ok)
    {
        for (const auto &batch : zkdecode_detail::lookupPlan(proof.parameters))
            appendLookupProofs(proof, stats, proveDecodeRangeLookup(
                neg_deltas, batch.range_bits, threads_, pk_, vk_, lookup_backend_,
                zkdecode_detail::rangeCommitment(proof, batch.kind), lookup_g1_encoding_));
    }
    stats.success = cand_ok;
    stats.message = !arithmetic_ok ? "greedy arithmetic proof generation failed"
                    : !cand_ok ? "greedy candidate range is invalid"
                               : "batched greedy proof completed";
    return result;
}

ZkDecodeProveResult ZkDecodeProver::proveTopKBatch(
    const vector<vector<ll>> &logits, const vector<int> &outputs, int k,
    const vector<ll> &samples, ll bound)
{
    return proveTopKBatchImpl(logits, outputs, k, samples, bound, zkdecode_detail::ExecutionContext::standalone());
}
ZkDecodeProveResult ZkDecodeProver::proveTopKBatchToInput(
    const vector<vector<ll>> &logits, const vector<int> &outputs, int k,
    const vector<ll> &samples, ll bound,
    FiatShamirTranscript &transcript, DecodeLogitEvaluation &deferred_logit_claim)
{
    deferred_logit_claim = DecodeLogitEvaluation{};
    return proveTopKBatchImpl(logits, outputs, k, samples, bound, zkdecode_detail::ExecutionContext::linked(transcript, deferred_logit_claim));
}
ZkDecodeProveResult ZkDecodeProver::proveTopKBatchImpl(
    const vector<vector<ll>> &logits_per_token,
    const vector<int> &output_per_token,
    int k,
    const vector<ll> &sample_value_per_token,
    ll bound, const zkdecode_detail::ExecutionContext &execution)
{
    const ScopedOmpThreads thread_scope(threads_);
    constexpr ll theta = ZkDecodeProbabilityScale;
    ZkDecodeProveResult result;
    auto &stats = result.stats;
    auto &proof = result.proof;
    const int T = static_cast<int>(logits_per_token.size());
    proof.parameters.kind = ZkDecodeProofKind::TopK;
    proof.parameters.token_count = T;
    proof.parameters.vocab_size = T > 0
                                 ? static_cast<int>(logits_per_token[0].size())
                                 : 0;
    proof.parameters.k = k;
    proof.parameters.bound = bound;
    proof.parameters.theta = theta;
    proof.parameters.lookup_backend = lookup_backend_;
    proof.output_tokens = output_per_token;
    proof.sample_values = sample_value_per_token;
    if (T == 0 ||
        static_cast<int>(output_per_token.size()) != T ||
        static_cast<int>(sample_value_per_token.size()) != T ||
        k <= 0 ||
        bound <= 0 ||
        k > exp_digits_config_.max_active_count ||
        bound > exp_digits_config_.max_abs_logit)
    {
        stats.message = "invalid batched top-k instance";
        return result;
    }
    const int N = static_cast<int>(logits_per_token[0].size());
    if (N == 0 || k > N)
    {
        stats.message = "empty logits";
        return result;
    }
    for (int t = 0; t < T; ++t)
    {
        if (static_cast<int>(logits_per_token[t].size()) != N || output_per_token[t] < 0 || output_per_token[t] >= N || sample_value_per_token[t] < 0)
        {
            stats.message = "inconsistent N or invalid output token";
            return result;
        }
    }

    auto witness = prepareTopKWitness(logits_per_token, output_per_token, k, bound,
                                      exp_digits_config_, softmax_epsilon_, stats.prover_time);
    auto &selections = witness.selections;
    auto &columns = witness.columns;
    proof.parameters.epsilon_sm = columns.epsilon;
    DecodeTopKLinkResult link_result;
    DecodeTopKLinkWitnessView link_witness = buildTopKLinkWitnessView(
        logits_per_token, selections, columns, k);
    FiatShamirTranscript link_transcript;
    link_result = prove_decode_topk_link_view(
        link_witness, pk_, vk_, threads_,
        execution.transcript(link_transcript), execution.deferredLogitClaim());
    stats.prover_time += link_result.prover_time;
    proof.has_topk_link = true;
    const bool arith_ok = link_result.success;
    proof.topk_link = std::move(link_result.proof);
    const Fr tuple_compression_challenge = arith_ok
                                               ? decode_topk_tuple_challenge(
                                                     proof.topk_link, T, N, k)
                                               : Fr(0);

    const bool candidate_ok = arith_ok && columns.candidates_valid;
    const bool softmax_ok = candidate_ok && columns.softmax_valid;
    link_witness = DecodeTopKLinkWitnessView{};
    using Batch = zkdecode_detail::LookupBatchKind;
    for (const auto &batch : zkdecode_detail::lookupPlan(proof.parameters))
    {
        const vector<ll> *range_values = nullptr;
        switch (batch.kind)
        {
        case Batch::CandidateIn:
            if (candidate_ok) range_values = &columns.candidate_in;
            break;
        case Batch::CandidateOut:
            if (candidate_ok) range_values = &columns.candidate_out;
            break;
        case Batch::ExpDigits:
            witness.releaseCandidates();
            if (softmax_ok)
                appendLookupProofs(proof, stats, proveDecodeExpDigitLookups(
                    std::move(columns.digits), tuple_compression_challenge,
                    threads_, pk_, vk_, lookup_backend_, proof.topk_link, lookup_g1_encoding_));
            break;
        case Batch::GreedyRange:
            throw std::logic_error("greedy lookup in a TopK plan");
        }
        if (range_values)
            appendLookupProofs(proof, stats, proveDecodeRangeLookup(
                *range_values, batch.range_bits, threads_, pk_, vk_, lookup_backend_,
                zkdecode_detail::rangeCommitment(proof, batch.kind), lookup_g1_encoding_));
    }
    witness.releaseSelectionInputs();

    if (softmax_ok)
    {
        const auto begin = Clock::now();
        vector<vector<ll>> probability_mass_per_token;
        probability_mass_per_token.reserve(T);
        for (auto &selection : selections)
        {
            probability_mass_per_token.push_back(std::move(selection.quantized_mass));
        }
        stats.prover_time += secondsSince(begin);
        const auto &probability_commitment = decode_topk_probability_commitment(proof.topk_link);
        FiatShamirTranscript selector_transcript;
        auto selector = prove_decode_selector(
            probability_mass_per_token, output_per_token, pk_, vk_, threads_,
            execution.transcript(selector_transcript), &probability_commitment);
        proof.has_selector = selector.success;
        stats.success = selector.success;
        stats.prover_time += selector.prover_time;
        proof.selector = std::move(selector.proof);
    }

    stats.message = !arith_ok ? "top-k arithmetic proof generation failed"
                    : !candidate_ok ? "top-k candidate range is invalid"
                    : !softmax_ok ? "top-k softmax witness is invalid"
                    : !stats.success ? "selector proof generation failed"
                                     : "batched top-k proof completed";
    return result;
}

void ZkDecodeProver::setLookupBackend(LookupBackend backend)
{
    if (backend != LookupBackend::Shout && backend != LookupBackend::LogUpGkr)
    {
        throw std::invalid_argument("unsupported zkDecode lookup backend");
    }
    lookup_backend_ = backend;
}

void ZkDecodeProver::setSoftmaxEpsilon(ll epsilon)
{
    if (epsilon < -1)
    {
        throw std::invalid_argument("softmax epsilon must be -1 or non-negative");
    }
    softmax_epsilon_ = epsilon;
}

vector<ll> ZkDecodeProver::computeSamplingMassTopK(
    const vector<ll> &logits, int k) const
{
    if (logits.empty() || k <= 0 || k > static_cast<int>(logits.size()) || k > exp_digits_config_.max_active_count)
    {
        return {};
    }
    SelectionResult selection = buildTopKSelection(
        logits, k, exp_digits_config_);
    if (!selection.exp_digits.success)
        return {};
    return std::move(selection.quantized_mass);
}

ZkDecodeProveResult proveZkDecodeInputLink(
    ZkDecodeProver &decoder, const std::vector<std::vector<ll>> &logits,
    const ZkDecodeStatement &statement, const ZkDecodeVerifierKey &key,
    const ZkDecodeInputContext &context, FiatShamirTranscript &transcript,
    InputReductionLink &deferred_input_claim)
{
    const auto context_start = std::chrono::steady_clock::now();
    zkdecode_input_detail::append_context(statement, key, context, transcript);
    const double context_seconds = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - context_start)
                                       .count();
    const auto &p = statement.parameters;
    if (logits.size() != std::size_t(p.token_count))
        throw std::invalid_argument("wrong decode row count");
    decoder.setLookupBackend(p.lookup_backend);
    decoder.setSoftmaxEpsilon(p.epsilon_sm);
    DecodeLogitEvaluation evaluation;
    auto produced = p.kind == ZkDecodeProofKind::Greedy
                        ? decoder.proveGreedyBatchToInput(logits, statement.output_tokens, p.bound, transcript, evaluation)
                        : decoder.proveTopKBatchToInput(logits, statement.output_tokens, p.k,
                                                        statement.sample_values, p.bound, transcript, evaluation);
    if (!produced.stats.success)
    {
        throw std::runtime_error(
            "failed to produce input-bound decode proof: " + produced.stats.message);
    }
    const auto claim_start = std::chrono::steady_clock::now();
    zkdecode_input_detail::bind_logit_claim(deferred_input_claim, context, evaluation, transcript);
    const double binding_seconds = context_seconds + std::chrono::duration<double>(
                                                         std::chrono::steady_clock::now() - claim_start)
                                                         .count();
    produced.stats.prover_time += binding_seconds;
    return produced;
}
