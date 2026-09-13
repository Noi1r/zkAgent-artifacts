#ifndef ZKAGENT_ZKDECODE_VERIFIER_HPP
#define ZKAGENT_ZKDECODE_VERIFIER_HPP

#include "zkdecode/zkdecode_proof.hpp"
#include "lookup/logup_gkr.hpp"
#include "zkdecode/zkdecode_lookup.hpp"

#include <memory>
#include <string>

namespace zkdecode_detail { class ExecutionContext; }

struct ZkDecodeVerifyStats {
    bool success = false;
    bool input_claim_pending = false;
    double total_seconds = 0.0;
    std::string message;
};

class ZkDecodeVerifierKey {
public:
    const ZkDecodeParameters& parameters() const { return parameters_; }
    std::size_t lookupProofCount() const { return groups_.size(); }

private:
    struct PreparedTable {
        std::vector<Fr> values;
        LogupGkrTableCommitment logup;
    };
    struct Group {
        zkdecode_detail::LookupGroup spec;
        std::shared_ptr<const PreparedTable> table;
        std::shared_ptr<const PreparedTable> tuple_table_outputs;

        bool usesRandomTupleCompression() const {
            return tuple_table_outputs != nullptr;
        }
    };
    ZkDecodeParameters parameters_;
    int threads_ = 1;
    std::vector<zkdecode_detail::LookupBatch> lookup_plan_;
    std::vector<Group> groups_;
    friend ZkDecodeVerifierKey prepareZkDecodeVerifierKey(
        const ZkDecodeParameters&, const MVKZG_ProverKey&,
        const MVKZG_VerifierKey&, int);
    friend bool verifyZkDecodeProof(
        const ZkDecodeStatement&, const ZkDecodeProof&,
        const ZkDecodeVerifierKey&, const MVKZG_VerifierKey&,
        ZkDecodeVerifyStats*);
    friend bool reduceZkDecodeProof(
        const ZkDecodeStatement&, const ZkDecodeProof&,
        const ZkDecodeVerifierKey&, const MVKZG_VerifierKey&,
        FiatShamirTranscript&, DecodeLogitEvaluation&, ZkDecodeVerifyStats*);
    bool verifyImpl(const ZkDecodeStatement& statement, const ZkDecodeProof& proof,
        const MVKZG_VerifierKey& vk, ZkDecodeVerifyStats* stats,
        const zkdecode_detail::ExecutionContext& execution) const;
};

ZkDecodeVerifierKey prepareZkDecodeVerifierKey(
    const ZkDecodeParameters& parameters,
    const MVKZG_ProverKey& pk, const MVKZG_VerifierKey& vk,
    int threads = 1);

bool verifyZkDecodeProof(
    const ZkDecodeStatement& statement, const ZkDecodeProof& proof,
    const ZkDecodeVerifierKey& key, const MVKZG_VerifierKey& vk,
    ZkDecodeVerifyStats* stats = nullptr);

bool reduceZkDecodeProof(
    const ZkDecodeStatement& statement, const ZkDecodeProof& proof, const ZkDecodeVerifierKey& key,
    const MVKZG_VerifierKey& vk, FiatShamirTranscript& transcript, DecodeLogitEvaluation& deferred_logit_claim,
    ZkDecodeVerifyStats* stats = nullptr);

#endif
