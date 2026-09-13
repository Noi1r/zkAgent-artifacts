#include "pcs/mvkzg.hpp"
#include "common/fiat_shamir.hpp"
#include "pcs/srs.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

using namespace mcl::bn;

void mvkzg_eval_derive_batch_weights(
    const std::vector<MVKZGEvalBatchItem> &items,
    FiatShamirTranscript &transcript,
    std::vector<Fr> &weights)
{
    transcript.append_u64(items.size());
    for (const MVKZGEvalBatchItem &item : items)
    {
        transcript.append_u64(item.point.size());
        transcript.append_fr_vector(item.point);
        transcript.append_g1(item.commitment,
                             G1WireEncoding::Compressed);
        transcript.append_fr(item.evaluation);
        transcript.append_g1_vector(item.proof.q_comm,
                                    G1WireEncoding::Compressed);
    }
    weights.resize(items.size());
    for (std::size_t j = 0; j < items.size(); ++j)
    {
        weights[j] = transcript.nonzero_challenge();
    }
}

static bool mvkzg_eval_verify_scaled(const Fr *s, int vars,
                                     const G1 &comm, const Fr &eval,
                                     const std::vector<G1> &q_comm,
                                     const MVKZG_VerifierKey &vk,
                                     double *verify_time_sec)
{
    const G1 &g1_gen = vk.g1_generator;
    const G2 &g2_gen = vk.g2_generator;

    G1 lhs;
    G1 tmp;
    G1::mul(tmp, g1_gen, eval);
    G1::sub(lhs, comm, tmp);

    std::vector<G1> g1_pts;
    std::vector<G2> g2_pts;
    g1_pts.reserve(static_cast<std::size_t>(vars) + 1);
    g2_pts.reserve(static_cast<std::size_t>(vars) + 1);
    g1_pts.push_back(lhs);
    g2_pts.push_back(g2_gen);

    for (int i = 0; i < vars; i++)
    {
        g1_pts.push_back(q_comm[i]);
        G2 g2_term;
        G2::mul(g2_term, g2_gen, s[i]);
        G2::sub(g2_term, vk.g2_variables[static_cast<std::size_t>(i)],
                g2_term);
        G2::neg(g2_term, g2_term);
        g2_pts.push_back(g2_term);
    }

    auto start = std::chrono::high_resolution_clock::now();
    GT prod;
    millerLoopVec(prod, g1_pts.data(), g2_pts.data(), g1_pts.size());
    finalExp(prod, prod);
    auto end = std::chrono::high_resolution_clock::now();
    if (verify_time_sec)
    {
        *verify_time_sec =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    }
    return prod.isOne();
}

bool mvkzg_eval_prove(const Fr *coeffs, int vars, const Fr *r,
                      const MVKZG_ProverKey &pk,
                      MVKZGEvalProof &proof)
{
    if (vars < 0)
    {
        return false;
    }
    std::size_t n = std::size_t(1) << vars;
    if (pk.vars < vars || pk.srs_g1.size() < n)
    {
        return false;
    }

    if (vars == 0)
    {
        proof.q_comm.clear();
        return true;
    }

    std::vector<Fr> s(vars);
    for (int i = 0; i < vars; i++)
    {
        Fr denom = Fr(1) - r[i];
        if (denom.isZero())
        {
            return false;
        }
        Fr inv_d;
        Fr::inv(inv_d, denom);
        s[i] = r[i] * inv_d;
    }

    std::vector<Fr> work(coeffs, coeffs + n);
    proof.q_comm.resize(vars);

    for (int i = vars - 1; i >= 0; i--)
    {
        std::size_t half = std::size_t(1) << i;
        const Fr *q_coeff = work.data() + half;
        if (half > 0)
        {
            G1::mulVecMT(proof.q_comm[i],
                         const_cast<G1 *>(pk.srs_g1.data()),
                         q_coeff, half, 0);
        }
        else
        {
            proof.q_comm[i].clear();
        }
        for (std::size_t j = 0; j < half; j++)
        {
            Fr tmp;
            Fr::mul(tmp, s[i], work[half + j]);
            Fr::add(work[j], work[j], tmp);
        }
        work.resize(half);
    }

    return true;
}

bool mvkzg_eval_verify(const Fr *r, int vars,
                       const G1 &comm, const Fr &eval,
                       const MVKZGEvalProof &proof,
                       const MVKZG_VerifierKey &vk,
                       double *verify_time_sec)
{
    if (vars < 0)
    {
        return false;
    }
    if ((int)proof.q_comm.size() != vars)
    {
        return false;
    }

    if (vk.vars < vars ||
        vk.g2_variables.size() < static_cast<std::size_t>(vars))
    {
        return false;
    }

    if (vars == 0)
    {
        G1 expected;
        G1::mul(expected, vk.g1_generator, eval);
        return expected == comm;
    }

    std::vector<Fr> s(vars);
    Fr scale = Fr(1);
    for (int i = 0; i < vars; i++)
    {
        Fr denom = Fr(1) - r[i];
        if (denom.isZero())
        {
            return false;
        }
        scale *= denom;
        Fr inv_d;
        Fr::inv(inv_d, denom);
        s[i] = r[i] * inv_d;
    }

    Fr inv_scale;
    Fr::inv(inv_scale, scale);
    Fr eval_scaled = eval * inv_scale;

    return mvkzg_eval_verify_scaled(s.data(), vars, comm, eval_scaled,
                                    proof.q_comm, vk, verify_time_sec);
}

bool mvkzg_eval_batch_verify(
    const std::vector<MVKZGEvalBatchItem> &items,
    const MVKZG_VerifierKey &vk,
    double *verify_time_sec,
    const std::vector<Fr> *batch_weights,
    FiatShamirTranscript *shared_transcript)
{
    const auto start = std::chrono::high_resolution_clock::now();
    auto finish = [&](bool result)
    {
        if (verify_time_sec != nullptr)
        {
            *verify_time_sec = std::chrono::duration_cast<
                                   std::chrono::duration<double>>(
                                   std::chrono::high_resolution_clock::now() - start)
                                   .count();
        }
        return result;
    };

    if (items.empty())
    {
        return finish(false);
    }

    std::size_t max_vars = 0;
    std::size_t total_vars = 0;
    for (const MVKZGEvalBatchItem &item : items)
    {
        const std::size_t vars = item.point.size();
        if (item.proof.q_comm.size() != vars ||
            vars >= std::numeric_limits<std::size_t>::digits)
        {
            return finish(false);
        }
        if (vk.vars < static_cast<int>(vars) ||
            vk.g2_variables.size() < vars ||
            vars > std::numeric_limits<std::size_t>::max() - total_vars)
        {
            return finish(false);
        }
        max_vars = std::max(max_vars, vars);
        total_vars += vars;
    }

    std::vector<Fr> weights(items.size());
    if (batch_weights != nullptr)
    {
        if (batch_weights->size() != items.size())
            return finish(false);
        weights = *batch_weights;
        for (const Fr &weight : weights)
        {
            if (weight.isZero())
                return finish(false);
        }
    }
    else
    {
        FiatShamirTranscript local_transcript;
        FiatShamirTranscript &transcript = shared_transcript != nullptr
                                               ? *shared_transcript
                                               : local_transcript;
        mvkzg_eval_derive_batch_weights(items, transcript, weights);
    }

    std::vector<Fr> denominators;
    denominators.reserve(total_vars);
    for (const MVKZGEvalBatchItem &item : items)
    {
        for (const Fr &coordinate : item.point)
        {
            const Fr denominator = Fr(1) - coordinate;
            if (denominator.isZero())
                return finish(false);
            denominators.push_back(denominator);
        }
    }
    std::vector<Fr> inverse_denominators(total_vars);
    if (total_vars != 0)
    {
        std::vector<Fr> prefixes(total_vars);
        Fr product = Fr(1);
        for (std::size_t i = 0; i < total_vars; ++i)
        {
            prefixes[i] = product;
            product *= denominators[i];
        }
        Fr inverse_product;
        Fr::inv(inverse_product, product);
        for (std::size_t i = total_vars; i-- > 0;)
        {
            inverse_denominators[i] = inverse_product * prefixes[i];
            inverse_product *= denominators[i];
        }
    }

    std::vector<std::vector<Fr>> subset_points(items.size());
    std::vector<Fr> scaled_evaluations(items.size());
    std::size_t coordinate_offset = 0;
    for (std::size_t j = 0; j < items.size(); ++j)
    {
        const MVKZGEvalBatchItem &item = items[j];
        subset_points[j].resize(item.point.size());
        Fr inverse_scale = Fr(1);
        for (std::size_t i = 0; i < item.point.size(); ++i)
        {
            const Fr inverse = inverse_denominators[coordinate_offset++];
            subset_points[j][i] = item.point[i] * inverse;
            inverse_scale *= inverse;
        }
        scaled_evaluations[j] = item.evaluation * inverse_scale;
    }

    std::vector<G1> base_points;
    std::vector<Fr> base_scalars;
    base_points.reserve(items.size() + total_vars);
    base_scalars.reserve(items.size() + total_vars);
    Fr combined_evaluation = Fr(0);
    for (std::size_t j = 0; j < items.size(); ++j)
    {
        base_points.push_back(items[j].commitment);
        base_scalars.push_back(weights[j]);
        combined_evaluation += weights[j] * scaled_evaluations[j];
        for (std::size_t i = 0; i < items[j].point.size(); ++i)
        {
            base_points.push_back(items[j].proof.q_comm[i]);
            base_scalars.push_back(weights[j] * subset_points[j][i]);
        }
    }

    G1 combined_base;
    if (base_points.size() == 1)
    {
        G1::mul(combined_base, base_points[0], base_scalars[0]);
    }
    else
    {
        G1::mulVecMT(combined_base, base_points.data(),
                     base_scalars.data(), base_points.size(), 0);
    }
    G1 evaluation_term;
    G1::mul(evaluation_term, vk.g1_generator, combined_evaluation);
    G1::sub(combined_base, combined_base, evaluation_term);

    std::vector<G1> g1_terms;
    std::vector<G2> g2_terms;
    g1_terms.reserve(max_vars + 1);
    g2_terms.reserve(max_vars + 1);
    g1_terms.push_back(combined_base);
    g2_terms.push_back(vk.g2_generator);

    std::vector<G1> coordinate_points;
    std::vector<Fr> coordinate_scalars;
    coordinate_points.reserve(items.size());
    coordinate_scalars.reserve(items.size());
    for (std::size_t i = 0; i < max_vars; ++i)
    {
        coordinate_points.clear();
        coordinate_scalars.clear();
        for (std::size_t j = 0; j < items.size(); ++j)
        {
            if (i >= items[j].point.size())
                continue;
            coordinate_points.push_back(items[j].proof.q_comm[i]);
            coordinate_scalars.push_back(weights[j]);
        }
        G1 combined_quotient;
        if (coordinate_points.size() == 1)
        {
            G1::mul(combined_quotient, coordinate_points[0],
                    coordinate_scalars[0]);
        }
        else
        {
            G1::mulVecMT(combined_quotient, coordinate_points.data(),
                         coordinate_scalars.data(),
                         coordinate_points.size(), 0);
        }
        G1::neg(combined_quotient, combined_quotient);
        g1_terms.push_back(combined_quotient);
        g2_terms.push_back(vk.g2_variables[i]);
    }

    GT product;
    millerLoopVec(product, g1_terms.data(), g2_terms.data(),
                  g1_terms.size());
    finalExp(product, product);
    return finish(product.isOne());
}
