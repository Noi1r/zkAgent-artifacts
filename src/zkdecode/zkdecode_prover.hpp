#ifndef ZKAGENT_ZKDECODE_PROVER_HPP
#define ZKAGENT_ZKDECODE_PROVER_HPP

#include "zkdecode/zkdecode_exp_digits.hpp"
#include "zkdecode/zkdecode_proof.hpp"
#include <string>
#include <vector>

struct MVKZG_ProverKey;
struct MVKZG_VerifierKey;

namespace zkdecode_detail { class ExecutionContext; }

// Builds decode proof artifacts. Success reports generation only; callers verify separately.
class ZkDecodeProver {
public:
    explicit ZkDecodeProver(const MVKZG_ProverKey& pk,
                            const MVKZG_VerifierKey& vk,
                            int threads = 32);

    ZkDecodeProveResult proveGreedyBatch(
        const std::vector<std::vector<ll>> &logits_per_token,
        const std::vector<int> &output_per_token,
        ll bound
    );

    ZkDecodeProveResult proveTopKBatch(
        const std::vector<std::vector<ll>> &logits_per_token,
        const std::vector<int> &output_per_token,
        int k,
        const std::vector<ll> &sample_value_per_token,
        ll bound
    );

    ZkDecodeProveResult proveGreedyBatchToInput(
        const std::vector<std::vector<ll>>& logits, const std::vector<int>& outputs, ll bound,
        FiatShamirTranscript& transcript, DecodeLogitEvaluation& deferred_logit_claim);
    ZkDecodeProveResult proveTopKBatchToInput(
        const std::vector<std::vector<ll>>& logits, const std::vector<int>& outputs, int k,
        const std::vector<ll>& samples, ll bound,
        FiatShamirTranscript& transcript, DecodeLogitEvaluation& deferred_logit_claim);

    std::vector<ll> computeSamplingMassTopK(
        const std::vector<ll> &logits, int k) const;

    void setLookupBackend(LookupBackend backend);

    void setLookupG1Encoding(G1WireEncoding encoding) {
        lookup_g1_encoding_ = encoding;
    }

    void setSoftmaxEpsilon(ll epsilon);

private:
    ZkDecodeProveResult proveGreedyBatchImpl(
        const std::vector<std::vector<ll>>& logits, const std::vector<int>& outputs, ll bound,
        const zkdecode_detail::ExecutionContext& execution);
    ZkDecodeProveResult proveTopKBatchImpl(
        const std::vector<std::vector<ll>>& logits, const std::vector<int>& outputs, int k,
        const std::vector<ll>& samples, ll bound,
        const zkdecode_detail::ExecutionContext& execution);
    const MVKZG_ProverKey& pk_;
    const MVKZG_VerifierKey& vk_;
    int threads_;
    LookupBackend lookup_backend_ = LookupBackend::LogUpGkr;
    G1WireEncoding lookup_g1_encoding_ = G1WireEncoding::Compressed;
    ll softmax_epsilon_ = -1;
    DecodeExpDigitsConfig exp_digits_config_;
};

#endif
