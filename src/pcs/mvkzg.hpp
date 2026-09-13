#ifndef MVKZG_HPP
#define MVKZG_HPP

#include <mcl/bn256.hpp>

#include <vector>

struct MVKZG_ProverKey;
struct MVKZG_VerifierKey;
class FiatShamirTranscript;

struct MVKZGEvalProof
{
    std::vector<mcl::bn::G1> q_comm;
};

struct MVKZGEvalBatchItem
{
    std::vector<mcl::bn::Fr> point;
    mcl::bn::G1 commitment;
    mcl::bn::Fr evaluation;
    MVKZGEvalProof proof;
};

void mvkzg_eval_derive_batch_weights(
    const std::vector<MVKZGEvalBatchItem> &items,
    FiatShamirTranscript &transcript,
    std::vector<mcl::bn::Fr> &weights);

bool mvkzg_eval_prove(const mcl::bn::Fr *coeffs, int vars,
                      const mcl::bn::Fr *r,
                      const MVKZG_ProverKey &pk,
                      MVKZGEvalProof &proof);

bool mvkzg_eval_verify(const mcl::bn::Fr *r, int vars,
                       const mcl::bn::G1 &comm,
                       const mcl::bn::Fr &eval,
                       const MVKZGEvalProof &proof,
                       const MVKZG_VerifierKey &vk,
                       double *verify_time_sec = nullptr);

bool mvkzg_eval_batch_verify(
    const std::vector<MVKZGEvalBatchItem> &items,
    const MVKZG_VerifierKey &vk,
    double *verify_time_sec = nullptr,
    const std::vector<mcl::bn::Fr> *batch_weights = nullptr,
    FiatShamirTranscript *shared_transcript = nullptr);

#endif
