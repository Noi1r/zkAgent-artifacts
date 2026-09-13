#ifndef ZKAGENT_ZKDECODE_INPUT_LINK_HPP
#define ZKAGENT_ZKDECODE_INPUT_LINK_HPP

#include "gkr/input_tensor_view.hpp"
#include "gkr/input_reduction_link.hpp"
#include "zkdecode/zkdecode_proof.hpp"
#include <memory>

class ZkDecodeProver;
class ZkDecodeVerifierKey;
struct ZkDecodeVerifyStats;

struct ZkDecodeInputContext
{
    const CompiledInputTensorView &logits;
    const std::vector<G1> &input_commitments;
    double logit_scale = 0;
};

std::vector<std::vector<ll>> prepareZkDecodeInputWitness(
    const std::vector<Fr> &input, const InputTensorViewSpec &view, ll bound);

std::size_t zkDecodeRequiredGenerators(const ZkDecodeParameters &parameters);

ZkDecodeProveResult proveZkDecodeInputLink(
    ZkDecodeProver &decoder, const std::vector<std::vector<ll>> &logits,
    const ZkDecodeStatement &statement, const ZkDecodeVerifierKey &key,
    const ZkDecodeInputContext &context, FiatShamirTranscript &transcript,
    InputReductionLink &deferred_input_claim);

std::unique_ptr<InputReductionLink> verifyZkDecodeInputLink(
    const ZkDecodeStatement &statement, const ZkDecodeProof &proof, const ZkDecodeVerifierKey &key,
    const ZkDecodeInputContext &context, const MVKZG_VerifierKey &vk, FiatShamirTranscript &transcript,
    ZkDecodeVerifyStats *stats = nullptr);

bool checkDecodeShiftConsistency(const std::vector<int> &transcript_tokens,
                                 std::size_t prompt_length,
                                 const std::vector<int> &verified_outputs);

namespace zkdecode_input_detail
{
    void append_context(const ZkDecodeStatement &statement, const ZkDecodeVerifierKey &key,
                        const ZkDecodeInputContext &context, FiatShamirTranscript &transcript);
    void bind_logit_claim(InputReductionLink &deferred_input_claim, const ZkDecodeInputContext &context,
                         const DecodeLogitEvaluation &evaluation, FiatShamirTranscript &transcript);
}

#endif
