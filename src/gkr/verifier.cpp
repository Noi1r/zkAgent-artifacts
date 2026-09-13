

#include "gkr/verifier.hpp"
#include "common/mle.hpp"
#include "common/polynomial.h"
#include "gkr/wiring_compiled.hpp"
#include "gkr/input_routing_compiled.hpp"
#include "lookup/lookup_input_routing_compiled.hpp"
#include "pcs/srs.hpp"
#include "common/timer.hpp"
#include "common/global_var.hpp"
#include <common/utils.hpp>
#include <gkr/circuit.h>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <omp.h>

using namespace std;
using namespace mcl::bn;

namespace
{

    struct ProductRoundSums
    {
        Fr at_zero = Fr(0);
        Fr at_one = Fr(0);
        Fr at_two = Fr(0);
    };

    std::size_t bit_words(std::size_t elements)
    {
        return (elements + 63) / 64;
    }

    bool support_contains(
        const std::vector<std::uint64_t> &support,
        std::size_t index)
    {
        return (support[index >> 6] &
                (std::uint64_t(1) << (index & 63))) != 0;
    }

    void fold_sparse_support(
        const std::vector<std::uint64_t> &support,
        std::size_t source_size,
        std::size_t pair_blocks,
        std::size_t block_width,
        int requested_threads,
        std::vector<std::uint64_t> &next_support)
    {
        const std::size_t output_size = pair_blocks * block_width;
        if (source_size != output_size * 2 ||
            support.size() < bit_words(source_size))
        {
            throw std::invalid_argument(
                "sparse input-reduction support has inconsistent dimensions");
        }
        const std::size_t output_words = bit_words(output_size);
        next_support.resize(output_words);
        const int threads = std::max(1, requested_threads);

        if (block_width >= 64 && block_width % 64 == 0 &&
            output_size % 64 == 0)
        {
            const std::size_t block_words = block_width / 64;
#pragma omp parallel for schedule(static) num_threads(threads) if (output_words >= static_cast<std::size_t>(threads) * 256)
            for (long long raw = 0;
                 raw < static_cast<long long>(output_words); ++raw)
            {
                const std::size_t output_word = static_cast<std::size_t>(raw);
                const std::size_t pair = output_word / block_words;
                const std::size_t offset = output_word % block_words;
                const std::size_t left = (2 * pair) * block_words + offset;
                next_support[output_word] =
                    support[left] | support[left + block_words];
            }
            return;
        }

#pragma omp parallel for schedule(static) num_threads(threads) if (output_words >= static_cast<std::size_t>(threads) * 16)
        for (long long raw = 0;
             raw < static_cast<long long>(output_words); ++raw)
        {
            const std::size_t output_word = static_cast<std::size_t>(raw);
            const std::size_t begin = output_word * 64;
            const std::size_t end = std::min(output_size, begin + 64);
            std::uint64_t word = 0;
            for (std::size_t output = begin; output < end; ++output)
            {
                const std::size_t pair = output / block_width;
                const std::size_t offset = output % block_width;
                const std::size_t left =
                    (2 * pair) * block_width + offset;
                const std::size_t right = left + block_width;
                if (support_contains(support, left) ||
                    support_contains(support, right))
                {
                    word |= std::uint64_t(1) << (output - begin);
                }
            }
            next_support[output_word] = word;
        }
    }

    ProductRoundSums fold_sparse_coefficient_dense_value(
        const std::vector<Fr> &coefficient,
        const std::vector<Fr> &value,
        const std::vector<std::uint64_t> &support,
        std::size_t pair_blocks,
        std::size_t block_width,
        const Fr *challenge,
        bool compute_sums,
        int requested_threads,
        std::vector<Fr> &next_coefficient,
        std::vector<Fr> &next_value,
        std::vector<std::uint64_t> &next_support)
    {
        const std::size_t output_size = pair_blocks * block_width;
        const std::size_t source_size = output_size * 2;
        if (coefficient.size() != source_size || value.size() != source_size ||
            support.size() < bit_words(source_size))
        {
            throw std::invalid_argument(
                "sparse input-reduction fold received inconsistent dimensions");
        }
        if (challenge != nullptr)
        {
            next_coefficient.resize(output_size);
            next_value.resize(output_size);
            fold_sparse_support(
                support, source_size, pair_blocks, block_width,
                requested_threads, next_support);
        }

        const int threads = std::max(1, requested_threads);
        const bool parallel = threads > 1 &&
                              output_size >= static_cast<std::size_t>(threads) * 256;
        std::vector<ProductRoundSums> local(static_cast<std::size_t>(threads));
        const Fr two(2);

#pragma omp parallel num_threads(threads) if (parallel)
        {
            const int tid = omp_get_thread_num();
            ProductRoundSums sums;
#pragma omp for schedule(static)
            for (long long raw = 0;
                 raw < static_cast<long long>(output_size); ++raw)
            {
                const std::size_t output = static_cast<std::size_t>(raw);
                const std::size_t pair = output / block_width;
                const std::size_t offset = output % block_width;
                const std::size_t left =
                    (2 * pair) * block_width + offset;
                const std::size_t right = left + block_width;
                const bool left_active = support_contains(support, left);
                const bool right_active = support_contains(support, right);
                const bool active = left_active || right_active;

                const Fr &v0 = value[left];
                const Fr &v1 = value[right];
                if (compute_sums && active)
                {
                    const Fr c0 = left_active ? coefficient[left] : F_ZERO;
                    const Fr c1 = right_active ? coefficient[right] : F_ZERO;
                    sums.at_zero += c0 * v0;
                    sums.at_one += c1 * v1;
                    sums.at_two += (two * c1 - c0) * (two * v1 - v0);
                }
                if (challenge != nullptr)
                {
                    next_value[output] =
                        v0 + (v1 - v0) * *challenge;
                    if (active)
                    {
                        const Fr c0 = left_active
                                          ? coefficient[left]
                                          : F_ZERO;
                        const Fr c1 = right_active
                                          ? coefficient[right]
                                          : F_ZERO;
                        next_coefficient[output] =
                            c0 + (c1 - c0) * *challenge;
                    }
                }
            }
            local[static_cast<std::size_t>(tid)] = sums;
        }

        ProductRoundSums result;
        for (const ProductRoundSums &sums : local)
        {
            result.at_zero += sums.at_zero;
            result.at_one += sums.at_one;
            result.at_two += sums.at_two;
        }
        return result;
    }

    void append_quadratic_message(
        FiatShamirTranscript *transcript, const quadratic_poly &polynomial)
    {
        transcript->append_fr(polynomial.a);
        transcript->append_fr(polynomial.b);
        transcript->append_fr(polynomial.c);
    }

    ProductRoundSums fold_and_sum_next_dimension(
        const std::vector<Fr> &coefficient,
        const std::vector<Fr> &value,
        const std::vector<std::uint64_t> &support,
        std::size_t pair_blocks,
        std::size_t block_width,
        std::size_t next_block_width,
        const Fr &challenge,
        int requested_threads,
        std::vector<Fr> &next_coefficient,
        std::vector<Fr> &next_value,
        std::vector<std::uint64_t> &next_support)
    {
        const std::size_t output_size = pair_blocks * block_width;
        if (block_width == 0 || next_block_width == 0 ||
            coefficient.size() != output_size * 2 ||
            value.size() != output_size * 2 ||
            output_size % (2 * next_block_width) != 0 ||
            (next_block_width != block_width &&
             (next_block_width != 1 || pair_blocks != 1)) ||
            support.size() < bit_words(output_size * 2))
        {
            throw std::invalid_argument(
                "pipelined input-reduction fold has inconsistent dimensions");
        }
        next_coefficient.resize(output_size);
        next_value.resize(output_size);
        fold_sparse_support(
            support, output_size * 2, pair_blocks, block_width,
            requested_threads, next_support);
        const std::size_t next_pairs = output_size / 2;
        const int threads = std::max(1, requested_threads);
        const bool parallel = threads > 1 &&
                              next_pairs >= static_cast<std::size_t>(threads) * 256;
        std::vector<ProductRoundSums> local(static_cast<std::size_t>(threads));
        const Fr two(2);

#pragma omp parallel num_threads(threads) if (parallel)
        {
            const int tid = omp_get_thread_num();
            ProductRoundSums sums;
            auto fold_entry = [&](std::size_t left, std::size_t output, Fr &c, Fr &v)
            {
                const std::size_t right = left + block_width;
                const Fr &v0 = value[left];
                const Fr &v1 = value[right];
                v = v0 + (v1 - v0) * challenge;
                next_value[output] = v;
                const bool left_active = support_contains(support, left);
                const bool right_active = support_contains(support, right);
                if (left_active || right_active)
                {
                    const Fr c0 = left_active ? coefficient[left] : F_ZERO;
                    const Fr c1 = right_active ? coefficient[right] : F_ZERO;
                    c = c0 + (c1 - c0) * challenge;
                    next_coefficient[output] = c;
                    return true;
                }
                c = F_ZERO;
                return false;
            };
#pragma omp for schedule(static)
            for (long long raw = 0;
                 raw < static_cast<long long>(next_pairs); ++raw)
            {
                const std::size_t output = static_cast<std::size_t>(raw);
                std::size_t target_left, target_right, source_left, source_right;
                if (block_width == 1)
                {
                    target_left = 2 * output;
                    target_right = target_left + 1;
                    source_left = 4 * output;
                    source_right = source_left + 2;
                }
                else if (next_block_width == 1)
                {
                    target_left = 2 * output;
                    target_right = target_left + 1;
                    source_left = target_left;
                    source_right = target_right;
                }
                else
                {
                    const std::size_t pair = output / block_width;
                    const std::size_t offset = output % block_width;
                    target_left = 2 * pair * block_width + offset;
                    target_right = target_left + block_width;
                    source_left = 4 * pair * block_width + offset;
                    source_right = source_left + 2 * block_width;
                }
                Fr c0, c1, v0, v1;
                const bool left_active = fold_entry(source_left, target_left, c0, v0);
                const bool right_active = fold_entry(source_right, target_right, c1, v1);
                if (left_active || right_active)
                {
                    sums.at_zero += c0 * v0;
                    sums.at_one += c1 * v1;
                    sums.at_two += (two * c1 - c0) * (two * v1 - v0);
                }
            }
            local[static_cast<std::size_t>(tid)] = sums;
        }
        ProductRoundSums result;
        for (const ProductRoundSums &sums : local)
        {
            result.at_zero += sums.at_zero;
            result.at_one += sums.at_one;
            result.at_two += sums.at_two;
        }
        return result;
    }

}

verifier::verifier(prover *pr, const layeredCircuit &cir,
                   const MVKZG_ProverKey *mv_pk_in,
                   const MVKZG_VerifierKey *mv_vk_in,
                   bool debug) : p(pr), C(cir), mv_pk(mv_pk_in), mv_vk(mv_vk_in), debug_(debug)
{
    comm = p->cc;
    initializeState();
    p->init();
}

verifier::verifier(
    const layeredCircuit &cir,
    GkrVerifierKey verifier_key,
    GkrInputCommitment input_commitment,
    const MVKZG_VerifierKey *mv_vk_in,
    bool debug)
    : p(nullptr), C(cir), mv_pk(nullptr), mv_vk(mv_vk_in), debug_(debug),
      verifier_key_(std::move(verifier_key))
{
    comm = Commit_return{};
    public_commitment_rows_ = std::move(input_commitment.rows);
    comm.l = input_commitment.variables;
    comm.comm = public_commitment_rows_.empty()
                    ? nullptr
                    : public_commitment_rows_.data();
    initializeState();
}

void verifier::initializeState()
{
    final_claim_u0.resize(C.size + 2);
    final_claim_v0.resize(C.size + 2);
    prover_time = verifier_time = 0;
    matrix_time = 0;
    lm_head_prover_time = 0;
    lm_head_matrix_time = 0;
    gkr_sumcheck_phase1_time = 0;
    gkr_sumcheck_phase2_time = 0;
    gkr_init_time = 0;
    gkr_output_eval_time = 0;
    verifier_sumcheck_time = 0;
    verifier_wiring_time = 0;
    verifier_wiring_contraction_time = 0;
    verifier_challenge_time = 0;
    verifier_input_reduction_time = 0;
    transformer_prover_time = 0;
    transformer_time_valid = false;
    input_reduction_init_time = 0;
    input_reduction_sumcheck_time = 0;
    input_reduction_active_coefficients = 0;
    input_reduction_sparse_coefficients = false;
    input_reduction_coefficient_time = 0;
    input_link_injection_time = 0;
    input_link_verifier_time = 0;
    commit_open_time = 0;
    commit_open_verify_time = 0;
    r_u.resize(C.size + 2);
    r_v.resize(C.size + 2);
}

void verifier::setFiatShamirTranscript(FiatShamirTranscript &transcript)
{
    fiat_shamir_transcript_ = &transcript;
}

void verifier::setCompiledWiringPlan(const CompiledWiringPlan &plan)
{
    if (plan.layerCount() != C.circuit.size())
    {
        throw std::invalid_argument(
            "compiled wiring plan does not match the circuit layer count");
    }
    compiled_wiring_plan_ = &plan;
    compiled_wiring_workspace_.resize(
        plan.stats().max_workspace_elements);
}

void verifier::setCompiledInputRoutingPlan(
    const CompiledInputRoutingPlan &plan)
{
    if (plan.layerCount() != C.circuit.size() || C.circuit.empty() ||
        plan.inputBits() != C.circuit[0].bit_length)
    {
        throw std::invalid_argument(
            "compiled input-routing plan does not match the circuit");
    }
    compiled_input_routing_plan_ = &plan;
    compiled_input_routing_workspace_.resize(
        plan.stats().max_workspace_elements);
}

void verifier::setInputReductionLink(InputReductionLink link)
{
    input_link_compiled_workspace_.resize(link.compiledWorkspaceElements());
    input_reduction_link_.reset(
        new InputReductionLink(std::move(link)));
}

F verifier::nextChallenge(bool reject_one)
{
    return reject_one
               ? fiat_shamir_transcript_->challenge_not_one()
               : fiat_shamir_transcript_->challenge();
}

void verifier::requireChallengeMode() const
{
    if (fiat_shamir_transcript_ == nullptr)
    {
        throw std::logic_error(
            "GKR proof generation requires a Fiat--Shamir transcript");
    }
}

InputReductionWorkspacePreparation verifier::prepareInputReductionWorkspace()
{
    InputReductionWorkspacePreparation result;
    if (C.size <= 0)
        return result;
    const int bits = C.circuit[0].bit_length;
    if (bits < 0 || bits >= static_cast<int>(sizeof(std::size_t) * 8))
    {
        throw std::length_error(
            "input-reduction workspace bit length is too large");
    }
    const std::size_t domain = std::size_t(1) << bits;
    result.current_domain = domain > 1 ? domain / 2 : 0;
    result.next_domain = domain > 2 ? domain / 4 : 0;

    input_reduction_coefficient_workspace_.resize(result.current_domain);
    input_reduction_value_workspace_.resize(result.current_domain);
    input_reduction_next_coefficient_workspace_.resize(result.next_domain);
    input_reduction_next_value_workspace_.resize(result.next_domain);
    input_reduction_active_workspace_.resize(bit_words(result.current_domain));
    input_reduction_next_active_workspace_.resize(
        bit_words(result.next_domain));
    input_open_rt_.reserve(
        std::size_t(1) << (bits - bits / 2));

    result.allocated_bytes =
        2 * (result.current_domain + result.next_domain) * sizeof(Fr) + (input_reduction_active_workspace_.size() +
                                                                         input_reduction_next_active_workspace_.size()) *
                                                                            sizeof(std::uint64_t);
    return result;
}

F verifier::getFinalValue(const F &claim_u0, const F &claim_u1, const F &claim_v0, const F &claim_v1)
{
    auto test_value = bin_value[0] * (claim_u0 * claim_v0) + bin_value[1] * (claim_u1 * claim_v1) + bin_value[2] * (claim_u1 * claim_v0) + uni_value[0] * claim_u0 + uni_value[1] * claim_u1;

    return test_value;
}

bool verifier::prove(int commit_thread)
{
    if (p == nullptr)
    {
        throw std::logic_error(
            "verifier-only instance cannot generate a GKR proof");
    }
    requireChallengeMode();
    proof_ = GkrProof{};
    proof_.layers.resize(static_cast<std::size_t>(C.size));
    last_gkr_ok_ = false;
    last_input_reduction_ok_ = false;
    last_lookup_link_ok_ = false;
    last_commitment_checked_ = false;
    last_commitment_ok_ = false;
    input_open_rt_reused_ = false;
    thread_num_ = std::max(1, commit_thread);
    p->setThreadCount(thread_num_);
    if (debug_)
    {
        cout << "Circuit Initiation finished" << endl;
        cout << "Proving Service started" << endl;
    }
    if (!verifyGKR())
    {
        cerr << "verifyGKR failed, aborting" << endl;
        return false;
    }
    last_gkr_ok_ = true;
    if (!verifyInputClaimReduction())
    {
        cerr << "verifyInputClaimReduction failed, aborting" << endl;
        return false;
    }
    last_input_reduction_ok_ = true;
    last_lookup_link_ok_ = input_reduction_link_ != nullptr;
    if (comm.comm && comm.g && comm.w && comm.l > 0)
    {
        last_commitment_checked_ = true;
        last_commitment_ok_ = openCommit();
    }
    else
    {
        cout << "Skipping commitment opening (no commit provided)..." << endl;
        last_commitment_ok_ = true;
    }
    if (debug_)
    {
        cout << "Our results:" << endl;
    }
    cout << "Matrix multiplication Prover time: " << matrix_time << "s" << endl;
    if (transformer_time_valid)
    {
        cout << "Transformer+Final LN Prover time (GKR): " << transformer_prover_time << "s" << endl;
        cout << "LM Head Prover time (GKR): " << lm_head_prover_time << "s";
        if (lm_head_matrix_time > 0)
        {
            cout << " (matrix: " << lm_head_matrix_time << "s, matrix is included in Matrix multiplication Prover time)";
        }
        cout << endl;
    }
    else
    {
        cout << "Transformer+Final LN Prover time (GKR): N/A" << endl;
        cout << "LM Head Prover time (GKR): N/A" << endl;
    }
    double total_prover_time = prover_time + range_prover_time;
    double accounted_time = gkr_output_eval_time + gkr_sumcheck_phase1_time +
                            gkr_sumcheck_phase2_time + gkr_init_time +
                            input_reduction_init_time + input_reduction_sumcheck_time + commit_open_time +
                            matrix_time + range_prover_time;
    double unaccounted_time = total_prover_time - accounted_time;
    if (unaccounted_time < 0 && unaccounted_time > -1e-9)
        unaccounted_time = 0;
    double verifier_accounted_time = verifier_challenge_time +
                                     verifier_sumcheck_time +
                                     verifier_wiring_time +
                                     verifier_input_reduction_time +
                                     input_reduction_coefficient_time +
                                     commit_open_verify_time;
    double verifier_unaccounted_time = verifier_time - verifier_accounted_time;
    if (verifier_unaccounted_time < 0 && verifier_unaccounted_time > -1e-9)
        verifier_unaccounted_time = 0;
    cout << "Total Prover time: " << total_prover_time << "s" << endl;
    cout << "  - Output MLE evaluation: " << gkr_output_eval_time << "s" << endl;
    cout << "  - GKR sumcheck Phase 1: " << gkr_sumcheck_phase1_time << "s" << endl;
    cout << "  - GKR sumcheck Phase 2: " << gkr_sumcheck_phase2_time << "s" << endl;
    cout << "  - GKR init (sumcheckInit*): " << gkr_init_time << "s" << endl;
    cout << "  - Input-claim reduction init: " << input_reduction_init_time << "s" << endl;
    cout << "    - Lookup link coefficient injection: " << input_link_injection_time << "s" << endl;
    cout << "  - Input-claim reduction sumcheck: " << input_reduction_sumcheck_time << "s" << endl;
    cout << "  - Commitment open: " << commit_open_time << "s" << endl;
    cout << "  - Commitment open (verifier): " << commit_open_verify_time << "s" << endl;
    cout << "  - Matrix multiplication: " << matrix_time << "s" << endl;
    cout << "  - Range proof (LogUp): " << range_prover_time << "s" << endl;
    cout << "  (Unaccounted: " << unaccounted_time << "s)" << endl;
    cout << "Verifier time: " << verifier_time << "s" << endl;
    cout << "  - Challenge handling: " << verifier_challenge_time << "s" << endl;
    cout << "  - Sumcheck claim checks: " << verifier_sumcheck_time << "s" << endl;
    cout << "  - Wiring predicates: " << verifier_wiring_time << "s" << endl;
    cout << "    - Compiled-plan evaluation: "
         << verifier_wiring_contraction_time << "s" << endl;
    cout << "  - Input-claim reduction checks: " << verifier_input_reduction_time << "s" << endl;
    cout << "  - Input-claim coefficient evaluation: " << input_reduction_coefficient_time << "s" << endl;
    cout << "    - Compiled routing/link evaluation: "
         << input_reduction_coefficient_time << "s" << endl;
    cout << "    - Lookup link coefficient evaluation: " << input_link_verifier_time << "s" << endl;
    cout << "  - Commitment opening: " << commit_open_verify_time << "s" << endl;
    cout << "  (Unaccounted: " << verifier_unaccounted_time << "s)" << endl;
    cout << "Proof size: "
         << (double)proof_.cryptographicPayloadBytes() / 1024.0
         << "KB" << endl;
    return last_commitment_ok_;
}

bool verifier::generateProof(int commit_thread)
{
    if (generation_only_)
    {
        throw std::logic_error("nested GKR proof generation is not allowed");
    }
    generation_only_ = true;
    try
    {
        const bool ok = prove(commit_thread);
        generation_only_ = false;
        return ok;
    }
    catch (...)
    {
        generation_only_ = false;
        throw;
    }
}

Fr *init_book_keeping(int m, int n, vector<Fr> &vec, int offset, vector<Fr> &ra)
{
    assert(ra.size() == m + n);
    Fr *pa[30];
    for (int j = 0; j <= n; j++)
    {
        pa[j] = new Fr[1 << (m + n - j)];
    }
    int t = min((1 << (m + n)), (int)vec.size() - offset);
    for (int j = t; j < (1 << (m + n)); ++j)
        pa[0][j].clear();

    for (int j = 0; j < t; j++)
        pa[0][j] = vec[j + offset];
    for (int j = 1; j <= n; j++)
    {
        for (int kp = 0; kp < (1 << (m + n - j)); kp++)
        {
            pa[j][kp] = pa[j - 1][kp + (1 << (m + n - j))] * ra[m + n - j] + pa[j - 1][kp] * (1 - ra[m + n - j]);
        }
    }
    for (int j = 0; j < n; j++)
    {
        delete[] pa[j];
    }
    return pa[n];
}

static Fr eval_bias_mle(const vector<Fr> &val, int bias_offset, int col_count, int k_bits,
                        const vector<F> &r_v, int m_bits)
{
    if (bias_offset < 0 || col_count <= 0 || k_bits < 0)
    {
        return F_ZERO;
    }
    if (bias_offset >= (int)val.size())
    {
        return F_ZERO;
    }
    int eval_size = 1 << k_bits;
    std::vector<Fr> work(eval_size);
    int avail = min(col_count, (int)val.size() - bias_offset);
    for (int i = 0; i < eval_size; ++i)
    {
        work[i] = (i < avail) ? val[bias_offset + i] : F_ZERO;
    }
    return mle::evaluate(std::move(work), r_v.data() + m_bits, k_bits);
}
static Fr *init_book_keeping_fast_impl(
    int m, int n, const int *vec, vector<Fr> &ra, int requested_threads)
{
    assert(ra.size() == m + n);

    int n1 = (n + 1) / 2;
    int n2 = n / 2;

    Fr *ret = new Fr[1 << m];
    std::vector<Fr> eq1(1 << n1), eq2(1 << n2);

    for (int j = 0; j < (1 << n1); j++)
    {
        eq1[j] = 1;
        for (int i = 0; i < n1; i++)
        {
            if (j & (1 << (n1 - 1 - i)))
                eq1[j] *= ra[m + n - 1 - i];
            else
                eq1[j] *= 1 - ra[m + n - 1 - i];
        }
    }

    for (int j = 0; j < (1 << n2); j++)
    {
        eq2[j] = 1;
        for (int i = 0; i < n2; i++)
        {
            if (j & (1 << (n2 - 1 - i)))
                eq2[j] *= ra[m + n2 - 1 - i];
            else
                eq2[j] *= 1 - ra[m + n2 - 1 - i];
        }
    }

    constexpr int limb_bits = 10;
    constexpr int limb_radix = 1 << limb_bits;
    constexpr std::uint64_t limb_mask = limb_radix - 1;
    constexpr int table_limb_count = 2;
    const int range_r = 1 << n2;
    const std::size_t per_r =
        static_cast<std::size_t>(table_limb_count) * limb_radix;
    std::vector<Fr> digit_multiples(static_cast<std::size_t>(range_r) * per_r);
    std::vector<Fr> high_bases(range_r);

    int num_threads = std::min(std::max(1, requested_threads), range_r);
#pragma omp parallel for schedule(static) num_threads(num_threads) if (range_r >= num_threads * 2)
    for (int R = 0; R < range_r; ++R)
    {
        Fr limb_base = eq2[R];
        for (int limb = 0; limb < table_limb_count; ++limb)
        {
            Fr *table = digit_multiples.data() +
                        static_cast<std::size_t>(R) * per_r +
                        static_cast<std::size_t>(limb) * limb_radix;
            table[0].clear();
            for (int digit = 1; digit < limb_radix; ++digit)
            {
                table[digit] = table[digit - 1] + limb_base;
            }
            for (int bit = 0; bit < limb_bits; ++bit)
            {
                limb_base += limb_base;
            }
        }
        high_bases[R] = limb_base;
    }

    const int range_kp = 1 << m;
    num_threads = std::min(std::max(1, requested_threads), range_kp);
#pragma omp parallel for schedule(static) num_threads(num_threads) if (range_kp >= num_threads * 2)
    for (int kp = 0; kp < range_kp; ++kp)
    {
        ret[kp] = 0;
        Fr sum = 0;
        for (int L = 0; L < (1 << n1); L++)
        {
            Fr sumL = 0;
            for (int R = 0; R < (1 << n2); R++)
            {
                int j = ((L << n2) + R) << m;
                const int value = vec[kp + j];
                if (value == 0)
                {
                    continue;
                }
                std::uint64_t magnitude = value < 0
                                              ? static_cast<std::uint64_t>(-static_cast<std::int64_t>(value))
                                              : static_cast<std::uint64_t>(value);
                Fr term = F_ZERO;
                for (int limb = 0; limb < table_limb_count; ++limb)
                {
                    const auto digit = static_cast<unsigned>(magnitude & limb_mask);
                    if (digit != 0)
                    {
                        const std::size_t table_index =
                            static_cast<std::size_t>(R) * per_r +
                            static_cast<std::size_t>(limb) * limb_radix + digit;
                        term += digit_multiples[table_index];
                    }
                    magnitude >>= limb_bits;
                }
                if (magnitude == 1)
                {
                    term += high_bases[R];
                }
                else if (magnitude > 1)
                {
                    term += high_bases[R] * Fr(static_cast<long long>(magnitude));
                }
                if (value > 0)
                    sumL += term;
                else
                    sumL -= term;
            }
            sum += eq1[L] * sumL;
        }
        ret[kp] = sum;
    }
    return ret;
}

static bool sum_check_product(const Fr *f, const Fr *g, int m,
                              const Fr &ans, pair<Fr, Fr> &result,
                              GkrFconnProductProof *proof,
                              int round,
                              const std::function<Fr(
                                  int, const quadratic_poly &)> &
                                  round_challenge)
{
    if (m < 0)
    {
        if (ans != f[0] * g[0])
        {
            cerr << "FCONN sumcheck claim mismatch for a one-element product" << endl;
            return false;
        }
        result = make_pair(f[0], g[0]);
        if (proof != nullptr)
        {
            proof->input_evaluation = f[0];
            proof->matrix_evaluation = g[0];
        }
        return true;
    }

    Fr A = 0, B = 0, C = 0;
    vector<Fr> zf(1 << m), zg(1 << m);
    for (int k = 0; k < (1 << m); k++)
    {
        zf[k] = 2 * f[k * 2 + 1] - f[k * 2];
        zg[k] = 2 * g[k * 2 + 1] - g[k * 2];
        C += zf[k] * zg[k];
    }
    for (int k = 0; k < (1 << (m + 1)); k++)
    {
        if (k & 1)
            B += f[k] * g[k];
        else
        {
            A += f[k] * g[k];
        }
    }
    Fr a = (C + A) / 2 - B, b = 2 * B - C / 2 - 3 * A / 2, c = A;
    const quadratic_poly message(a, b, c);
    if (proof != nullptr)
        proof->rounds.push_back(message);
    Fr g1 = a + b + c, g0 = c;
    if (!(ans - (g1 + g0)).isZero())
    {
        cerr << "FCONN sumcheck claim mismatch at round " << m << endl;
        return false;
    }
    if (!(a + b + c - B).isZero())
    {
        cerr << "FCONN sumcheck polynomial check failed at x=1, round "
             << m << endl;
        return false;
    }
    if (!(4 * a + 2 * b + c - C).isZero())
    {
        cerr << "FCONN sumcheck polynomial check failed at x=2, round "
             << m << endl;
        return false;
    }
    Fr send_r = round_challenge(round, message);
    vector<Fr> new_f(1 << m), new_g(1 << m);
    for (int k = 0; k < (1 << m); k++)
    {
        new_f[k] = (1 - send_r) * f[k * 2] + (send_r)*f[k * 2 + 1];
        new_g[k] = (1 - send_r) * g[k * 2] + (send_r)*g[k * 2 + 1];
    }
    Fr new_ans = send_r * send_r * a + send_r * b + c;
    if (m == 0)
    {
        if (new_ans != new_f[0] * new_g[0])
        {
            cerr << "FCONN sumcheck final oracle mismatch" << endl;
            return false;
        }
        result = make_pair(new_f[0], new_g[0]);
        if (proof != nullptr)
        {
            proof->input_evaluation = new_f[0];
            proof->matrix_evaluation = new_g[0];
        }
        return true;
    }
    return sum_check_product(new_f.data(), new_g.data(), m - 1,
                             new_ans, result, proof, round + 1,
                             round_challenge);
}

bool verifier::verifyGKR()
{
    F alpha = F_ONE, beta = F_ZERO, relu_rou, final_claim_u1,
      final_claim_v1 = F_ZERO;
    timer ptimer, vtimer, mat_timer;
    transformer_prover_time = 0;
    transformer_time_valid = false;
    lm_head_prover_time = 0;
    int lm_head_layer_idx = -1;
    int lm_head_round_idx = -1;
    long long max_fconn_size = -1;
    if (p->lm_fc_id >= 0)
    {
        for (int i = 1; i < C.size; ++i)
        {
            const auto ty = C.circuit[i].ty;
            if (ty == layerType::FCONN && C.circuit[i].size > max_fconn_size)
            {
                max_fconn_size = C.circuit[i].size;
                lm_head_layer_idx = i;
            }
        }
    }
    if (lm_head_layer_idx >= 0 && lm_head_layer_idx + 1 < C.size &&
        C.circuit[lm_head_layer_idx + 1].ty == layerType::RELU)
    {
        lm_head_round_idx = lm_head_layer_idx + 1;
    }
    transformer_time_valid = (lm_head_layer_idx >= 0);
    vector<char> include_layer(C.size, 1);
    if (!include_layer.empty())
    {
        include_layer[0] = 0;
    }
    if (lm_head_layer_idx >= 0 && lm_head_layer_idx < C.size)
    {
        include_layer[lm_head_layer_idx] = 0;
    }
    if (lm_head_round_idx >= 0 && lm_head_round_idx < C.size)
    {
        include_layer[lm_head_round_idx] = 0;
    }
    vtimer.start();
    r_u[C.size].resize(C.circuit[C.size - 1].bit_length);
    for (i8 i = 0; i < C.circuit[C.size - 1].bit_length; ++i)
        r_u[C.size][i] = nextChallenge(true);
    vtimer.stop();
    double dt_initial_challenge = vtimer.elapse_sec();
    verifier_time += dt_initial_challenge;
    verifier_challenge_time += dt_initial_challenge;
    vector<F>::const_iterator r_0 = r_u[C.size].begin();
    vector<F>::const_iterator r_1;

    ptimer.start();
    auto previousSum = p->Vres(r_0, C.circuit[C.size - 1].size,
                               C.circuit[C.size - 1].bit_length, C.size - 1);
    ptimer.stop();
    double dt_output_eval = ptimer.elapse_sec();
    prover_time += dt_output_eval;
    gkr_output_eval_time += dt_output_eval;
    ptimer.start();
    p->sumcheckInitAll(r_0);
    ptimer.stop();
    double dt_init_all = ptimer.elapse_sec();
    prover_time += dt_init_all;
    gkr_init_time += dt_init_all;
    if (transformer_time_valid && !include_layer.empty() && include_layer[C.size - 1])
    {
        transformer_prover_time += dt_init_all;
    }
    else if (lm_head_layer_idx >= 0 &&
             (C.size - 1 == lm_head_layer_idx || C.size - 1 == lm_head_round_idx))
    {
        lm_head_prover_time += dt_init_all;
    }
    int fc_cnt = 0, now_fc_id;
    Fr prev_u1 = 0;
    for (int i = C.size - 1; i; --i)
    {
        if (C.circuit[i].ty == layerType::FCONN)
            fc_cnt++;
    }

    final_claim_u1 = previousSum;
    proof_.output_evaluation = previousSum;
    fiat_shamir_transcript_->append_fr(previousSum);

    for (int i = C.size - 1; i; --i)
    {
        auto &cur = C.circuit[i];
        GkrLayerProof &layer_proof =
            proof_.layers[static_cast<std::size_t>(i)];
        layer_proof.provided = true;
        double prover_before = prover_time;
        vector<F> fresh_zero_root;
        const bool restart_zero_root = requiresFreshZeroRoot(C, i);
        if (restart_zero_root)
        {
            if (!alpha.isZero() || !beta.isZero() || !previousSum.isZero())
            {
                cerr << "orphan zero-root would overwrite an incoming claim"
                     << endl;
                return false;
            }
            vtimer.start();
            fresh_zero_root.resize(cur.bit_length);
            for (int bit = 0; bit < cur.bit_length; ++bit)
            {
                fresh_zero_root[bit] = nextChallenge(true);
            }
            r_0 = fresh_zero_root.cbegin();
            r_1 = fresh_zero_root.cend();
            alpha = F_ONE;
            beta = F_ZERO;
            previousSum = F_ZERO;
            final_claim_u1 = F_ZERO;
            final_claim_v1 = F_ZERO;
            vtimer.stop();
            const double dt_zero_root = vtimer.elapse_sec();
            verifier_time += dt_zero_root;
            verifier_challenge_time += dt_zero_root;
        }
        ptimer.start();
        p->sumcheckInit(alpha, beta,
                        restart_zero_root ? &fresh_zero_root : nullptr);
        ptimer.stop();
        double dt_init_layer = ptimer.elapse_sec();
        prover_time += dt_init_layer;
        gkr_init_time += dt_init_layer;

        vtimer.start();
        if (cur.zero_start_id < cur.size)
        {
            relu_rou = nextChallenge();
        }
        else
            relu_rou = F_ONE;
        vtimer.stop();
        double dt_relu_challenge = vtimer.elapse_sec();
        verifier_time += dt_relu_challenge;
        verifier_challenge_time += dt_relu_challenge;
        F previousRandom = F_ZERO;
        if (C.circuit[i].ty == layerType::FCONN)
        {
            layer_proof.is_fconn = true;
            now_fc_id = --fc_cnt;
            int n, m, k;
            n = ceilPow2BitLength(p->fc_input_row[now_fc_id]);
            k = ceilPow2BitLength(p->fc_col[now_fc_id]);
            m = ceilPow2BitLength(p->fc_row[now_fc_id]);
            vtimer.start();
            r_u[i].resize(m + n);
            r_v[i].resize(m + k);
            for (int j = m; j < m + n; ++j)
                r_u[i][j] = r_0[j - m + k];
            for (int j = m; j < m + k; ++j)
                r_v[i][j] = r_0[j - m];
            vtimer.stop();
            double dt_fc_challenge = vtimer.elapse_sec();
            verifier_time += dt_fc_challenge;
            verifier_challenge_time += dt_fc_challenge;

            prev_u1 = final_claim_u1;
            previousRandom = F_ZERO;
            ptimer.start();
            mat_timer.start();
            const char *dbg = std::getenv("DEBUG_FCONN_SUMCHECK");
            int dbg_id = -1;
            if (const char *dbg_id_env = std::getenv("DEBUG_FCONN_ID"))
            {
                dbg_id = std::atoi(dbg_id_env);
            }
            if (dbg && *dbg != '0' && (dbg_id < 0 || dbg_id == now_fc_id))
            {
                const auto input_offset = p->fc_input_id[now_fc_id];
                const auto val0_size = p->val[0].size();
                const size_t avail = (input_offset < (int)val0_size)
                                         ? (val0_size - input_offset)
                                         : 0;
                cerr << "[FCONN sumcheck] layer=" << i
                     << " fc_id=" << now_fc_id
                     << " m=" << m << " n=" << n << " k=" << k
                     << " input_offset=" << input_offset
                     << " val0_size=" << val0_size
                     << " avail=" << avail
                     << " in_rows=" << p->fc_input_row[now_fc_id]
                     << " in_cols=" << p->fc_input_col[now_fc_id]
                     << " row=" << p->fc_row[now_fc_id]
                     << " col=" << p->fc_col[now_fc_id]
                     << " real_in_rows=" << p->fc_real_input_row[now_fc_id]
                     << " real_in_cols=" << p->fc_real_input_col[now_fc_id]
                     << " real_row=" << p->fc_real_row[now_fc_id]
                     << " real_col=" << p->fc_real_col[now_fc_id]
                     << " mat_val=" << (p->mat_val && p->mat_val[now_fc_id] ? "set" : "null")
                     << endl;
                if (avail >= 4)
                {
                    cerr << "  input[0..3]="
                         << p->val[0][input_offset + 0] << " "
                         << p->val[0][input_offset + 1] << " "
                         << p->val[0][input_offset + 2] << " "
                         << p->val[0][input_offset + 3] << endl;
                }
            }
            Fr *pa = init_book_keeping(m, n, p->val[0], p->fc_input_id[now_fc_id], r_u[i]);
            Fr *pb = init_book_keeping_fast_impl(
                m, k, p->mat_val[now_fc_id], r_v[i], thread_num_);
            Fr bias_eval = eval_bias_mle(p->val[0], p->fc_bias_id[now_fc_id],
                                         p->fc_col[now_fc_id], k, r_v[i], m);
            pair<Fr, Fr> oracle;
            const auto fconn_round_challenge =
                [&](int round, const quadratic_poly &message)
            {
                append_quadratic_message(
                    fiat_shamir_transcript_, message);
                r_u[i][round] = nextChallenge(true);
                r_v[i][round] = r_u[i][round];

                return r_u[i][round];
            };
            const bool fconn_sumcheck_ok = sum_check_product(
                pa, pb, m - 1, prev_u1 - bias_eval, oracle,
                &layer_proof.fconn, 0, fconn_round_challenge);
            delete[] pa;
            delete[] pb;
            if (!fconn_sumcheck_ok)
            {
                mat_timer.stop();
                ptimer.stop();
                cerr << "FCONN verification failed at circuit layer " << i
                     << ", fc_id=" << now_fc_id << endl;
                return false;
            }
            if (gkr_prover_key_ != nullptr && mv_pk != nullptr)
            {
                if (now_fc_id < 0 ||
                    static_cast<std::size_t>(now_fc_id) >=
                        gkr_prover_key_->fconn.size())
                {
                    return false;
                }
                const GkrFconnProverParameters &matrix =
                    gkr_prover_key_->fconn[static_cast<std::size_t>(now_fc_id)];
                if (matrix.matrix_variables != m + k ||
                    matrix.matrixSize() !=
                        (std::size_t(1) << (m + k)))
                {
                    return false;
                }
                layer_proof.fconn.matrix_opening = hyrax::open(
                    matrix.matrixData(), r_v[i].data(), m + k,
                    *mv_pk, hyrax::Layout::InterleavedLowBits);
                fiat_shamir_transcript_->append_g1_vector(layer_proof.fconn.matrix_opening.eval_proof.q_comm,
                    G1WireEncoding::Compressed);

            }
            else
            {
                cerr << "Fiat--Shamir FCONN proof requires prepared GKR keys"
                     << endl;
                return false;
            }
            mat_timer.stop();
            ptimer.stop();
            double dt_mat = mat_timer.elapse_sec();
            double dt_fc = ptimer.elapse_sec();
            matrix_time += dt_mat;
            prover_time += dt_fc;
            gkr_init_time += dt_fc - dt_mat;
            if (p->lm_fc_id >= 0 && now_fc_id == p->lm_fc_id)
            {
                lm_head_matrix_time += dt_mat;
            }
            final_claim_u0[i] = oracle.first;
            final_claim_u1 = 0;
            final_claim_v0[i] = oracle.second;
            final_claim_v1 = 0;
            layer_proof.final_u0 = final_claim_u0[i];
            layer_proof.final_u1 = final_claim_u1;
            layer_proof.final_v0 = final_claim_v0[i];
            layer_proof.final_v1 = final_claim_v1;
        }
        else
        {
            ptimer.start();
            p->sumcheckInitPhase1(relu_rou);
            ptimer.stop();
            double dt_init_p1 = ptimer.elapse_sec();
            prover_time += dt_init_p1;
            gkr_init_time += dt_init_p1;
            vtimer.start();
            r_u[i].resize(cur.max_bl_u);
            vtimer.stop();
            double dt_phase1_challenge = vtimer.elapse_sec();
            verifier_time += dt_phase1_challenge;
            verifier_challenge_time += dt_phase1_challenge;
            for (i8 j = 0; j < cur.max_bl_u; ++j)
            {
                F cur_claim, nxt_claim;
                ptimer.start();
                quadratic_poly poly = p->sumcheckUpdate1(previousRandom);
                layer_proof.phase1_rounds.push_back(poly);
                ptimer.stop();
                double dt_p1 = ptimer.elapse_sec();
                prover_time += dt_p1;
                gkr_sumcheck_phase1_time += dt_p1;

                vtimer.start();
                append_quadratic_message(
                    fiat_shamir_transcript_, poly);
                r_u[i][j] = nextChallenge(true);

                cur_claim = poly.eval(F_ZERO) + poly.eval(F_ONE);
                nxt_claim = poly.eval(r_u[i][j]);

                if (cur_claim != previousSum)
                {
                    cerr << "cur_claim=" << cur_claim << " previousSum=" << previousSum << endl;
                    const char *layer_names[] = {
                        "INPUT", "FFT", "IFFT", "ADD_BIAS", "RELU", "Sqr", "OPT_AVG_POOL",
                        "MAX_POOL", "AVG_POOL", "DOT_PROD", "PADDING", "FCONN", "LAYER_NORM_1",
                        "LAYER_NORM_2", "LAYER_NORM_3", "GELU_1", "GELU_2", "GELU_3",
                        "MHA_QK", "SOFTMAX_1", "SOFTMAX_2", "SOFTMAX_3", "PLACE_HOLDER",
                        "ROUND", "RESIDUAL_ADD"};
                    int ty_idx = static_cast<int>(cur.ty);
                    fprintf(stderr, "Verification fail, phase1, circuit %d (%s), current bit %d\n",
                            i, (ty_idx >= 0 && ty_idx < 25) ? layer_names[ty_idx] : "UNKNOWN", j);
                    fprintf(stderr, "  layer size=%d, max_bl_u=%d, max_bl_v=%d\n",
                            cur.size, cur.max_bl_u, cur.max_bl_v);

                    return false;
                }
                vtimer.stop();
                double dt_verify_p1 = vtimer.elapse_sec();
                verifier_time += dt_verify_p1;
                verifier_sumcheck_time += dt_verify_p1;
                previousRandom = r_u[i][j];
                previousSum = nxt_claim;
            }
            prev_u1 = final_claim_u1;
            ptimer.start();
            p->sumcheckFinalize1(previousRandom, final_claim_u0[i], final_claim_u1);
            ptimer.stop();
            double dt_finalize_p1 = ptimer.elapse_sec();
            prover_time += dt_finalize_p1;
            gkr_init_time += dt_finalize_p1;
            layer_proof.final_u0 = final_claim_u0[i];
            layer_proof.final_u1 = final_claim_u1;

            if (!generation_only_)
            {
                if (!compiled_wiring_enabled())
                {
                    cerr << "GKR verification requires a compiled wiring plan"
                         << endl;
                    return false;
                }
                const vector<F> wiring_r0(
                    r_0, r_0 + cur.bit_length);
                vector<F> wiring_r1;
                if (!beta.isZero())
                {
                    wiring_r1.assign(r_1, r_1 + cur.bit_length);
                }
                vtimer.start();
                const std::array<Fr, 2> wiring_result =
                    compiled_wiring_plan_->evaluatePhase1(
                        static_cast<std::size_t>(i), alpha, beta,
                        relu_rou, wiring_r0, wiring_r1, r_u[i],
                        thread_num_, compiled_wiring_workspace_);
                uni_value[0] = wiring_result[0];
                uni_value[1] = wiring_result[1];
                bin_value[0] = bin_value[1] = bin_value[2] = F_ZERO;
                vtimer.stop();
                double dt_wiring_contraction_p1 = vtimer.elapse_sec();
                verifier_time += dt_wiring_contraction_p1;
                verifier_wiring_time += dt_wiring_contraction_p1;
                verifier_wiring_contraction_time +=
                    dt_wiring_contraction_p1;
            }

            if (cur.need_phase2)
            {
                layer_proof.has_phase2 = true;
                vtimer.start();
                r_v[i].resize(cur.max_bl_v);
                vtimer.stop();
                double dt_phase2_challenge = vtimer.elapse_sec();
                verifier_time += dt_phase2_challenge;
                verifier_challenge_time += dt_phase2_challenge;

                ptimer.start();
                std::array<F, 2> phase1_unary_for_prover;
                const std::array<F, 2> *phase1_unary_ptr = nullptr;
                if (generation_only_ && compiled_wiring_enabled())
                {
                    const vector<F> wiring_r0(
                        r_0, r_0 + cur.bit_length);
                    vector<F> wiring_r1;
                    if (!beta.isZero())
                    {
                        wiring_r1.assign(r_1, r_1 + cur.bit_length);
                    }
                    const std::array<Fr, 2> wiring_result =
                        compiled_wiring_plan_->evaluatePhase1(
                            static_cast<std::size_t>(i), alpha, beta,
                            relu_rou, wiring_r0, wiring_r1, r_u[i],
                            thread_num_, compiled_wiring_workspace_);
                    uni_value[0] = wiring_result[0];
                    uni_value[1] = wiring_result[1];
                }
                if (compiled_wiring_enabled())
                {
                    phase1_unary_for_prover =
                        {{uni_value[0], uni_value[1]}};
                    phase1_unary_ptr = &phase1_unary_for_prover;
                }
                p->sumcheckInitPhase2(phase1_unary_ptr);
                ptimer.stop();
                double dt_init_p2 = ptimer.elapse_sec();
                prover_time += dt_init_p2;
                gkr_init_time += dt_init_p2;
                previousRandom = F_ZERO;
                for (int j = 0; j < cur.max_bl_v; ++j)
                {
                    ptimer.start();
                    quadratic_poly poly = p->sumcheckUpdate2(previousRandom);
                    layer_proof.phase2_rounds.push_back(poly);
                    ptimer.stop();
                    double dt_p2 = ptimer.elapse_sec();
                    prover_time += dt_p2;
                    gkr_sumcheck_phase2_time += dt_p2;
                    vtimer.start();
                    append_quadratic_message(
                        fiat_shamir_transcript_, poly);
                    r_v[i][j] = nextChallenge(true);

                    F cur_claim2 = poly.eval(F_ZERO) + poly.eval(F_ONE);
                    if (cur_claim2 != previousSum)
                    {
                        fprintf(stderr, "Verification fail, phase2, circuit level %d, current bit %d, total is %d\n", i, j,
                                cur.max_bl_v);

                        return false;
                    }
                    vtimer.stop();
                    double dt_verify_p2 = vtimer.elapse_sec();
                    verifier_time += dt_verify_p2;
                    verifier_sumcheck_time += dt_verify_p2;
                    previousRandom = r_v[i][j];
                    previousSum = poly.eval(previousRandom);
                }
                ptimer.start();
                p->sumcheckFinalize2(previousRandom, final_claim_v0[i], final_claim_v1);
                ptimer.stop();
                double dt_finalize_p2 = ptimer.elapse_sec();
                prover_time += dt_finalize_p2;
                gkr_init_time += dt_finalize_p2;
                layer_proof.final_v0 = final_claim_v0[i];
                layer_proof.final_v1 = final_claim_v1;

                if (!generation_only_)
                {
                    if (!compiled_wiring_enabled())
                    {
                        cerr << "GKR verification requires a compiled wiring plan"
                             << endl;
                        return false;
                    }
                    const vector<F> wiring_r0(
                        r_0, r_0 + cur.bit_length);
                    vector<F> wiring_r1;
                    if (!beta.isZero())
                    {
                        wiring_r1.assign(r_1, r_1 + cur.bit_length);
                    }
                    vtimer.start();
                    const std::array<Fr, 3> wiring_result =
                        compiled_wiring_plan_->evaluatePhase2(
                            static_cast<std::size_t>(i), alpha, beta,
                            relu_rou, wiring_r0, wiring_r1,
                            r_u[i], r_v[i], thread_num_,
                            compiled_wiring_workspace_);
                    Fr at_zero = F_ONE;
                    for (const Fr &coordinate : r_v[i])
                    {
                        at_zero *= F_ONE - coordinate;
                    }
                    uni_value[0] *= at_zero;
                    uni_value[1] *= at_zero;
                    bin_value[0] = wiring_result[0];
                    bin_value[1] = wiring_result[1];
                    bin_value[2] = wiring_result[2];
                    vtimer.stop();
                    double dt_wiring_contraction_p2 = vtimer.elapse_sec();
                    verifier_time += dt_wiring_contraction_p2;
                    verifier_wiring_time += dt_wiring_contraction_p2;
                    verifier_wiring_contraction_time +=
                        dt_wiring_contraction_p2;
                }
            }
            else
            {
                final_claim_v0[i] = F_ZERO;
                final_claim_v1 = F_ZERO;
                layer_proof.final_v0 = F_ZERO;
                layer_proof.final_v1 = F_ZERO;
            }
        }

        vtimer.start();
        if (C.circuit[i].ty != layerType::FCONN && !generation_only_)
        {
            F test_value = getFinalValue(final_claim_u0[i], final_claim_u1, final_claim_v0[i], final_claim_v1);
            if (previousSum != test_value)
            {
                const char *layer_names[] = {
                    "INPUT", "FFT", "IFFT", "ADD_BIAS", "RELU", "Sqr", "OPT_AVG_POOL",
                    "MAX_POOL", "AVG_POOL", "DOT_PROD", "PADDING", "FCONN", "LAYER_NORM_1",
                    "LAYER_NORM_2", "LAYER_NORM_3", "GELU_1", "GELU_2", "GELU_3",
                    "MHA_QK", "SOFTMAX_1", "SOFTMAX_2", "SOFTMAX_3", "PLACE_HOLDER",
                    "ROUND", "RESIDUAL_ADD"};
                int ty_idx = static_cast<int>(cur.ty);
                fprintf(stderr, "Verification fail at getFinalValue, layer %d/%d (%s)\n",
                        i, C.size - 1, (ty_idx >= 0 && ty_idx < 25) ? layer_names[ty_idx] : "UNKNOWN");
                fprintf(stderr, "  previousSum = %s\n", previousSum.getStr().c_str());
                fprintf(stderr, "  test_value  = %s\n", test_value.getStr().c_str());
                fprintf(stderr, "  final_claim_u0[%d] = %s\n", i, final_claim_u0[i].getStr().c_str());
                fprintf(stderr, "  final_claim_u1    = %s\n", final_claim_u1.getStr().c_str());
                fprintf(stderr, "  final_claim_v0[%d] = %s\n", i, final_claim_v0[i].getStr().c_str());
                fprintf(stderr, "  final_claim_v1    = %s\n", final_claim_v1.getStr().c_str());
                fprintf(stderr, "  bin_value[0] = %s\n", bin_value[0].getStr().c_str());
                fprintf(stderr, "  bin_value[1] = %s\n", bin_value[1].getStr().c_str());
                fprintf(stderr, "  bin_value[2] = %s\n", bin_value[2].getStr().c_str());
                fprintf(stderr, "  uni_value[0] = %s\n", uni_value[0].getStr().c_str());
                fprintf(stderr, "  uni_value[1] = %s\n", uni_value[1].getStr().c_str());
                fprintf(stderr, "  need_phase2 = %d, max_bl_v = %d\n", cur.need_phase2, cur.max_bl_v);
                fprintf(stderr, "  layer size=%d, max_bl_u=%d\n", cur.size, cur.max_bl_u);

                return false;
            }
        }
        fiat_shamir_transcript_->append_u64(static_cast<std::uint64_t>(i));
        fiat_shamir_transcript_->append_u64(static_cast<std::uint64_t>(cur.ty));
        fiat_shamir_transcript_->append_fr(final_claim_u0[i]);
        fiat_shamir_transcript_->append_fr(final_claim_u1);
        fiat_shamir_transcript_->append_fr(final_claim_v0[i]);
        fiat_shamir_transcript_->append_fr(final_claim_v1);

        if (~cur.bit_length_u[1])
            alpha = nextChallenge();
        else
            alpha.clear();
        if ((~cur.bit_length_v[1]) || cur.ty == layerType::FFT)
            beta = nextChallenge();
        else
            beta.clear();
        previousSum = alpha * final_claim_u1 + beta * final_claim_v1;

        vtimer.stop();
        double dt_verify_final = vtimer.elapse_sec();
        verifier_time += dt_verify_final;
        verifier_sumcheck_time += dt_verify_final;

        r_0 = r_u[i].begin();
        r_1 = r_v[i].begin();
        if (transformer_time_valid && include_layer[i])
        {
            transformer_prover_time += (prover_time - prover_before);
        }
        if (lm_head_layer_idx >= 0 &&
            (i == lm_head_layer_idx || i == lm_head_round_idx))
        {
            lm_head_prover_time += (prover_time - prover_before);
        }
    }

    return true;
}

bool verifier::verifyInputClaimReduction()
{
    timer ptimer, vtimer;
    auto &cur = C.circuit[0];

    vtimer.start();
    vector<F> sig_u(C.size - 1);
    for (int i = 0; i < C.size - 1; ++i)
        sig_u[i] = nextChallenge();
    vector<F> sig_v(C.size - 1);
    for (int i = 0; i < C.size - 1; ++i)
        sig_v[i] = nextChallenge();
    r_u[0].resize(cur.bit_length);
    F previousSum = F_ZERO;
    for (int i = 1; i < C.size; ++i)
    {
        if (~C.circuit[i].bit_length_u[0])
            previousSum = previousSum + sig_u[i - 1] * final_claim_u0[i];

        if (~C.circuit[i].bit_length_v[0] && C.circuit[i].need_phase2)
            previousSum = previousSum + sig_v[i - 1] * final_claim_v0[i];
    }
    vtimer.stop();
    double dt_input_reduction_challenges = vtimer.elapse_sec();
    verifier_time += dt_input_reduction_challenges;
    verifier_input_reduction_time += dt_input_reduction_challenges;

    if (std::getenv("VERIFY_INPUT_REDUCTION_DEBUG") ||
        std::getenv("VERIFY_LASSO_DEBUG"))
    {
        const char *layer_names[] = {
            "INPUT", "FFT", "IFFT", "ADD_BIAS", "RELU", "Sqr", "OPT_AVG_POOL",
            "MAX_POOL", "AVG_POOL", "DOT_PROD", "PADDING", "FCONN", "LAYER_NORM_1",
            "LAYER_NORM_2", "LAYER_NORM_3", "GELU_1", "GELU_2", "GELU_3",
            "MHA_QK", "SOFTMAX_1", "SOFTMAX_2", "SOFTMAX_3", "PLACE_HOLDER",
            "ROUND", "RESIDUAL_ADD"};
        vector<F> beta_probe;
        for (int i = 1; i < C.size; ++i)
        {
            int ty_idx = static_cast<int>(C.circuit[i].ty);
            const char *tn = (ty_idx >= 0 && ty_idx < 25) ? layer_names[ty_idx] : "UNKNOWN";

            if (~C.circuit[i].bit_length_u[0])
            {
                beta_probe.assign(1ULL << C.circuit[i].bit_length_u[0], F_ZERO);
                initBetaTable(beta_probe, C.circuit[i].bit_length_u[0], r_u[i].begin(), F_ONE, thread_num_);
                F expect = F_ZERO;
                for (u32 j = 0; j < C.circuit[i].size_u[0]; ++j)
                    expect = expect + beta_probe[j] * p->val[0][C.circuit[i].ori_id_u[j]];
                if (expect != final_claim_u0[i])
                {
                    fprintf(stderr, "[Input-reduction probe] layer %d (%s) U mismatch: claim=%s expect=%s size_u0=%u bl_u0=%d max_bl_u=%d\n",
                            i, tn, final_claim_u0[i].getStr().c_str(), expect.getStr().c_str(),
                            (unsigned)C.circuit[i].size_u[0], (int)C.circuit[i].bit_length_u[0], (int)C.circuit[i].max_bl_u);
                }
            }
            if (~C.circuit[i].bit_length_v[0])
            {
                beta_probe.assign(1ULL << C.circuit[i].bit_length_v[0], F_ZERO);
                initBetaTable(beta_probe, C.circuit[i].bit_length_v[0], r_v[i].begin(), F_ONE, thread_num_);
                F expect = F_ZERO;
                for (u32 j = 0; j < C.circuit[i].size_v[0]; ++j)
                    expect = expect + beta_probe[j] * p->val[0][C.circuit[i].ori_id_v[j]];
                if (expect != final_claim_v0[i])
                {
                    fprintf(stderr, "[Input-reduction probe] layer %d (%s) V mismatch: claim=%s expect=%s size_v0=%u bl_v0=%d max_bl_v=%d\n",
                            i, tn, final_claim_v0[i].getStr().c_str(), expect.getStr().c_str(),
                            (unsigned)C.circuit[i].size_v[0], (int)C.circuit[i].bit_length_v[0], (int)C.circuit[i].max_bl_v);
                }
            }
        }
    }

    ptimer.start();
    p->initInputClaimReduction(sig_u, sig_v, r_u, r_v);
    input_reduction_sparse_coefficients = true;
    if (input_reduction_link_ != nullptr)
    {
        if (input_reduction_link_->domainSize() != p->input_claim_weights.size())
        {
            cerr << "Input-reduction lookup link has the wrong domain size" << endl;
            return false;
        }
        timer link_timer;
        link_timer.start();
        input_reduction_link_->addTo(
            p->input_claim_weights,
            p->mutableInputClaimActiveBits(),
            p->mutableInputClaimActiveIndices(), thread_num_);
        link_timer.stop();
        input_link_injection_time += link_timer.elapse_sec();
        previousSum += input_reduction_link_->claimed_evaluation;
    }
    input_reduction_active_coefficients = p->inputClaimActiveCount();
    ptimer.stop();
    double dt_input_reduction_init = ptimer.elapse_sec();
    prover_time += dt_input_reduction_init;
    input_reduction_init_time += dt_input_reduction_init;
    const int n = C.circuit[0].bit_length;
    if (n < 0 || n >= static_cast<int>(sizeof(std::size_t) * 8))
    {
        cerr << "verifyInputClaimReduction: bit_length too large for allocation: "
             << n << endl;
        return false;
    }
    const std::size_t domain = std::size_t(1) << n;
    if (p->input_claim_weights.size() != domain || p->val[0].size() < domain)
    {
        cerr << "verifyInputClaimReduction: input vectors have the wrong domain"
             << endl;
        return false;
    }

    const int row_bits = n / 2;
    const int column_bits = n - row_bits;
    std::size_t rows = std::size_t(1) << row_bits;
    const std::size_t columns = std::size_t(1) << column_bits;
    const int fold_threads = std::max(1, p->threadCount());
    std::vector<Fr> &coefficient =
        input_reduction_coefficient_workspace_;
    std::vector<Fr> &value = input_reduction_value_workspace_;
    std::vector<Fr> &next_coefficient =
        input_reduction_next_coefficient_workspace_;
    std::vector<Fr> &next_value = input_reduction_next_value_workspace_;
    std::vector<std::uint64_t> &active_support =
        input_reduction_active_workspace_;
    std::vector<std::uint64_t> &next_active_support =
        input_reduction_next_active_workspace_;
    bool have_rolling_state = false;
    input_open_rt_.clear();
    input_open_rt_ready_ = false;

    const double verifier_before_input_reduction_sumcheck = verifier_time;
    ptimer.start();
    int protocol_round = 0;
    auto consume_round = [&](const ProductRoundSums &sums,
                             const Fr &challenge) -> bool
    {
        proof_.input_reduction_rounds.push_back(
            {sums.at_zero, sums.at_one, sums.at_two});
        vtimer.start();
        if (previousSum != sums.at_zero + sums.at_one)
        {
            cerr << "Input-claim reduction sumcheck failed at round "
                 << protocol_round << "/" << n << endl;
            cerr << "  previousSum = " << previousSum.getStr() << endl;
            cerr << "  g0+g1 = "
                 << (sums.at_zero + sums.at_one).getStr() << endl;
            vtimer.stop();
            const double dt = vtimer.elapse_sec();
            verifier_time += dt;
            verifier_input_reduction_time += dt;
            return false;
        }
        previousSum = polynomial::evaluate_quadratic_samples(
            sums.at_zero, sums.at_one, sums.at_two, challenge);
        ++protocol_round;
        vtimer.stop();
        const double dt = vtimer.elapse_sec();
        verifier_time += dt;
        verifier_input_reduction_time += dt;
        return true;
    };

    ProductRoundSums prefetched_sums;
    bool have_prefetched_sums = false;
    auto fold_next_dimension = [&](std::size_t pair_blocks,
                                   std::size_t block_width,
                                   int variable) -> bool
    {
        const bool first_fold = !have_rolling_state;
        const std::vector<Fr> &source_coefficient = first_fold
                                                        ? p->input_claim_weights
                                                        : coefficient;
        const std::vector<Fr> &source_value = first_fold
                                                  ? p->val[0]
                                                  : value;
        const std::vector<std::uint64_t> &source_support =
            first_fold ? p->inputClaimActiveBits() : active_support;
        std::vector<Fr> &target_coefficient = first_fold
                                                  ? coefficient
                                                  : next_coefficient;
        std::vector<Fr> &target_value = first_fold ? value : next_value;
        std::vector<std::uint64_t> &target_support = first_fold
                                                         ? active_support
                                                         : next_active_support;
        ProductRoundSums sums;

        if (have_prefetched_sums)
        {
            sums = prefetched_sums;
            have_prefetched_sums = false;
        }
        else
        {
            sums = fold_sparse_coefficient_dense_value(
                source_coefficient, source_value, source_support,
                pair_blocks, block_width, nullptr, true,
                fold_threads, target_coefficient, target_value,
                target_support);
        }

        fiat_shamir_transcript_->append_fr(sums.at_zero);
        fiat_shamir_transcript_->append_fr(sums.at_one);
        fiat_shamir_transcript_->append_fr(sums.at_two);
        r_u[0][variable] = nextChallenge(true);

        if (protocol_round + 1 < n)
        {
            const std::size_t next_block_width =
                protocol_round + 1 < row_bits ? columns : 1;
            prefetched_sums = fold_and_sum_next_dimension(
                source_coefficient, source_value, source_support,
                pair_blocks, block_width, next_block_width,
                r_u[0][variable], fold_threads,
                target_coefficient, target_value, target_support);
            have_prefetched_sums = true;
        }
        else
        {
            fold_sparse_coefficient_dense_value(
                source_coefficient, source_value, source_support,
                pair_blocks, block_width, &r_u[0][variable], false,
                fold_threads, target_coefficient, target_value,
                target_support);
        }

        if (!first_fold)
        {
            coefficient.swap(next_coefficient);
            value.swap(next_value);
            active_support.swap(next_active_support);
        }
        else
        {
            p->releaseInputClaimWeights();
        }
        have_rolling_state = true;
        return consume_round(sums, r_u[0][variable]);
    };

    for (int row_bit = 0; row_bit < row_bits; ++row_bit)
    {
        const int variable = column_bits + row_bit;
        if (!fold_next_dimension(rows / 2, columns, variable))
        {
            ptimer.stop();
            return false;
        }
        rows /= 2;
    }

    if (have_rolling_state)
    {
        input_open_rt_ = value;
    }
    else
    {
        input_open_rt_.assign(p->val[0].begin(), p->val[0].begin() + columns);
    }
    input_open_rt_ready_ = input_open_rt_.size() == columns;

    std::size_t active_columns = columns;
    for (int column_bit = 0; column_bit < column_bits; ++column_bit)
    {
        if (!fold_next_dimension(
                active_columns / 2, 1, column_bit))
        {
            ptimer.stop();
            return false;
        }
        active_columns /= 2;
    }

    if (n == 0)
    {
        if (previousSum != p->input_claim_weights[0] * p->val[0][0])
        {
            ptimer.stop();
            cerr << "Input-claim reduction final product check failed" << endl;
            return false;
        }
        eval_in = p->val[0][0];
    }
    else
    {
        if (coefficient.size() != 1 || value.size() != 1 ||
            active_support.empty())
        {
            ptimer.stop();
            cerr << "Input-claim reduction final state has an invalid shape" << endl;
            return false;
        }
        const F terminal_coefficient = !support_contains(active_support, 0)
                                           ? F_ZERO
                                           : coefficient[0];
        if (previousSum != terminal_coefficient * value[0])
        {
            ptimer.stop();
            cerr << "Input-claim reduction final product check failed" << endl;
            return false;
        }
        eval_in = value[0];
    }
    proof_.input_evaluation = eval_in;
    fiat_shamir_transcript_->append_fr(eval_in);

    ptimer.stop();
    const double dt_input_reduction_total = ptimer.elapse_sec();
    const double dt_input_reduction_nested_verifier =
        verifier_time - verifier_before_input_reduction_sumcheck;
    double dt_input_reduction_prover =
        dt_input_reduction_total - dt_input_reduction_nested_verifier;
    if (dt_input_reduction_prover < 0 &&
        dt_input_reduction_prover > -1e-9)
    {
        dt_input_reduction_prover = 0;
    }
    prover_time += dt_input_reduction_prover;
    input_reduction_sumcheck_time += dt_input_reduction_prover;
    if (generation_only_)
    {
        r_u.resize(1);
        r_v.clear();
        return true;
    }
    if (!compiled_input_routing_enabled())
    {
        cerr << "Input-claim verification requires a compiled routing plan"
             << endl;
        return false;
    }

    timer coefficient_total_timer;
    coefficient_total_timer.start();
    F gr = compiled_input_routing_plan_->evaluate(
        r_u[0], r_u, r_v, sig_u, sig_v, thread_num_,
        compiled_input_routing_workspace_);
    if (input_reduction_link_ != nullptr)
    {
        timer link_timer;
        link_timer.start();
        gr += input_reduction_link_->coefficientEvaluationAtPoint(
            r_u[0], thread_num_, &input_link_compiled_workspace_);
        link_timer.stop();
        input_link_verifier_time += link_timer.elapse_sec();
    }
    coefficient_total_timer.stop();

    double dt_input_coefficient = coefficient_total_timer.elapse_sec();
    verifier_time += dt_input_coefficient;
    input_reduction_coefficient_time += dt_input_coefficient;

    vtimer.start();
    if (eval_in * gr != previousSum)
    {
        cerr << "Input-claim reduction failed: eval_in * coefficient != claim" << endl;
        return false;
    }
    vtimer.stop();
    double dt_input_reduction_final_check = vtimer.elapse_sec();
    verifier_time += dt_input_reduction_final_check;
    verifier_input_reduction_time += dt_input_reduction_final_check;

    if (debug_)
    {
        fprintf(stderr, "GKR and input-claim reduction verification passed\n");
    }

    r_u.resize(1);
    r_v.clear();

    sig_u.clear();
    sig_v.clear();
    return true;
}

bool verifier::openCommit()
{
    if (!mv_pk || !mv_vk)
    {
        cerr << "Error: MVKZG SRS is required for Hyrax opening" << endl;
        return false;
    }

    const int row_variables = comm.l / 2;
    const int column_variables = comm.l - row_variables;
    const int expected_rt_size = 1 << column_variables;
    const Fr *precomputed_rt =
        input_open_rt_ready_ &&
                input_open_rt_.size() == static_cast<std::size_t>(expected_rt_size)
            ? input_open_rt_.data()
            : nullptr;
    input_open_rt_reused_ = precomputed_rt != nullptr;

    timer prover_open_timer;
    prover_open_timer.start();
    const hyrax::OpeningProof opening = hyrax::open(
        comm.w, r_u[0].data(), comm.l, *mv_pk,
        hyrax::Layout::ContiguousRows, precomputed_rt);
    prover_open_timer.stop();
    proof_.input_opening.provided = true;
    proof_.input_opening.evaluation = eval_in;
    proof_.input_opening.opening = opening;
    fiat_shamir_transcript_->append_g1_vector(opening.eval_proof.q_comm,
        G1WireEncoding::Compressed);
    if (generation_only_)
    {
        const double prover_open_seconds = prover_open_timer.elapse_sec();
        prover_time += prover_open_seconds;
        commit_open_time += prover_open_seconds;
        input_open_rt_.clear();
        input_open_rt_.shrink_to_fit();
        input_open_rt_ready_ = false;
        return true;
    }

    const std::vector<Fr> row_weights = hyrax::equality_weights(
        r_u[0].data() + column_variables, row_variables);
    timer verifier_open_timer;
    verifier_open_timer.start();
    hyrax::VerifyTiming opening_timing;
    const bool ok = hyrax::verify(
        opening, r_u[0].data(), eval_in, comm.comm, comm.l, *mv_vk,
        hyrax::Layout::ContiguousRows, row_weights.data(), &opening_timing);
    verifier_open_timer.stop();

    const double verifier_open_seconds =
        opening_timing.row_commitment_seconds +
        opening_timing.pcs_pairing_seconds;
    const double prover_open_seconds = prover_open_timer.elapse_sec();
    prover_time += prover_open_seconds;
    commit_open_time += prover_open_seconds;
    verifier_time += verifier_open_seconds;
    commit_open_verify_time += verifier_open_seconds;
    input_open_rt_.clear();
    input_open_rt_.shrink_to_fit();
    input_open_rt_ready_ = false;
    return ok;
}
