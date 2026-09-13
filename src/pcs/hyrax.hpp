#ifndef ZKAGENT_HYRAX_HPP
#define ZKAGENT_HYRAX_HPP

#include "pcs/mvkzg.hpp"
#include "common/typedef.hpp"

#include <cstdint>
#include <mcl/bn256.hpp>
#include <vector>

struct MVKZG_ProverKey;
struct MVKZG_VerifierKey;

namespace hyrax
{

    using mcl::bn::Fr;
    using mcl::bn::G1;

    enum class Layout : std::uint8_t
    {
        InterleavedLowBits,
        ContiguousRows,
    };

    struct OpeningProof
    {
        MVKZGEvalProof eval_proof;
    };

    // Scratch metadata shared only by openings of the same unchanged one-hot
    // address vector within one proof. Challenge-dependent folding values are
    // deliberately not retained. A shape change rebuilds the metadata; callers
    // must not reuse a ready workspace after mutating the addresses in place.
    struct OneHotOpeningWorkspace
    {
        int query_count = 0;
        int table_size = 0;
        std::vector<int> col_offsets;
        std::vector<int> bucket_rows;
        bool buckets_ready = false;
    };

    struct VerifyTiming
    {
        double row_commitment_seconds = 0.0;
        double pcs_pairing_seconds = 0.0;
    };

    struct RowCommitmentTerm
    {
        const std::vector<G1> *rows = nullptr;
        Fr coefficient = Fr(0);
    };

    struct CommitmentPair
    {
        std::vector<G1> first;
        std::vector<G1> second;
    };

    class OpeningBatchVerifier
    {
    public:
        bool append(const OpeningProof &proof,
                    const Fr *point,
                    const Fr &evaluation,
                    const G1 *row_commitments,
                    int variables,
                    Layout layout = Layout::InterleavedLowBits,
                    const Fr *precomputed_row_weights = nullptr);

        bool append_combined(const OpeningProof &proof,
                             const Fr *point,
                             const Fr &evaluation,
                             const std::vector<RowCommitmentTerm> &terms,
                             int variables,
                             int threads = 1,
                             Layout layout = Layout::InterleavedLowBits);

        bool verify(const MVKZG_VerifierKey &vk,
                    double *verify_time_sec = nullptr,
                    FiatShamirTranscript *shared_transcript = nullptr) const;
        bool verify_with_weights(const MVKZG_VerifierKey &vk,
                                 const std::vector<Fr> &weights,
                                 double *verify_time_sec = nullptr) const;
        void derive_weights(
            FiatShamirTranscript &transcript,
            std::vector<Fr> &weights) const;

        std::size_t size() const { return items_.size(); }

    private:
        void append_folded(const OpeningProof &proof, const Fr *point,
                           const Fr &evaluation, const G1 &commitment,
                           int variables, Layout layout);
        std::vector<MVKZGEvalBatchItem> items_;
    };

    G1 load_generators(G1 *generators, const std::vector<G1> &srs_g1, int count);

    G1 *commit(const ll *values, G1 *generators, int variables,
               int threads = 1,
               Layout layout = Layout::InterleavedLowBits);
    CommitmentPair commit_pair(
        const ll *first, const ll *second, G1 *generators, int variables,
        int threads = 1,
        Layout layout = Layout::InterleavedLowBits);
    G1 *commit(const Fr *values, G1 *generators, int variables,
               int threads = 1,
               Layout layout = Layout::InterleavedLowBits);
    std::vector<G1> commit_rows(const Fr *values, std::size_t value_count,
                                G1 *generators, int column_variables,
                                std::size_t first_row, std::size_t row_count,
                                int threads = 1);
    G1 *commit_onehot(const int *indices, int query_count, int table_size,
                      G1 *generators, int threads = 1);

    Fr evaluate(const Fr *values, const Fr *point, int variables);
    std::vector<Fr> equality_weights(const Fr *point, int variables);

    OpeningProof open(const ll *values, const Fr *point, int variables,
                      const MVKZG_ProverKey &pk,
                      Layout layout = Layout::InterleavedLowBits,
                      const Fr *precomputed_folded = nullptr);
    OpeningProof open(const Fr *values, const Fr *point, int variables,
                      const MVKZG_ProverKey &pk,
                      Layout layout = Layout::InterleavedLowBits,
                      const Fr *precomputed_folded = nullptr);
    OpeningProof open_folded(const std::vector<Fr> &folded, const Fr *point,
                             int variables, const MVKZG_ProverKey &pk,
                             Layout layout = Layout::InterleavedLowBits);
    OpeningProof open_onehot(const int *indices, int query_count, int table_size,
                             const Fr *point, const MVKZG_ProverKey &pk,
                             int threads = 1,
                             OneHotOpeningWorkspace *workspace = nullptr);

    bool verify(const OpeningProof &proof, const Fr *point,
                const Fr &evaluation, const G1 *row_commitments,
                int variables, const MVKZG_VerifierKey &vk,
                Layout layout = Layout::InterleavedLowBits,
                const Fr *precomputed_row_weights = nullptr,
                VerifyTiming *timing = nullptr);

}

#endif
