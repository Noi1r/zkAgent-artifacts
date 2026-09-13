#include "zkembed/zkembed.hpp"
#include "common/cache_io.hpp"
#include "common/bit_utils.hpp"
#include "common/fiat_shamir.hpp"
#include "common/omp_threads.hpp"
#include "pcs/hyrax.hpp"
#include "pcs/mvkzg.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <omp.h>

using namespace std;
using namespace mcl::bn;

namespace
{

    constexpr char EmbeddingCacheType[] = "zkagent.embedding.cache\n";

    void batch_invert(std::vector<Fr> &values)
    {
        if (values.empty()) return;
        std::vector<Fr> prefixes(values.size());
        Fr product(1);
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if (values[i].isZero())
                throw std::invalid_argument("zkEmbed: reciprocal denominator is zero");
            prefixes[i] = product;
            product *= values[i];
        }
        Fr inverse;
        Fr::inv(inverse, product);
        for (std::size_t i = values.size(); i-- > 0;)
        {
            const Fr value = values[i];
            values[i] = inverse * prefixes[i];
            inverse *= value;
        }
    }

    // MLE of [r, r^2, ..., r^count, 0, ...], with low-bit-first coordinates.
    Fr geometric_prefix_mle(const Fr &ratio, const std::vector<Fr> &point,
                            std::size_t count)
    {
        Fr full(1), prefix(0), power(ratio);
        for (std::size_t bit = 0; bit < point.size(); ++bit)
        {
            const Fr low = Fr(1) - point[bit];
            const Fr high = point[bit] * power;
            // Before this step, prefix covers count mod 2^bit lower-bit entries.
            if ((count >> bit) & 1)
                prefix = low * full + high * prefix;
            else
                prefix *= low;
            full *= low + high;
            Fr::sqr(power, power);
        }
        return ratio * (count == (std::size_t(1) << point.size()) ? full : prefix);
    }

    int padded_variables(std::size_t size)
    {
        if (size == 0 || size > (std::size_t(1) << 30))
        {
            throw std::invalid_argument("zkEmbed: unsupported tensor dimension");
        }
        int variables = 0;
        while ((std::size_t(1) << variables) < size)
            ++variables;
        return variables;
    }

    std::size_t commitment_rows(int variables)
    {
        if (variables < 0 || variables > 30)
        {
            throw std::invalid_argument("zkEmbed: unsupported PCS dimension");
        }
        return std::size_t(1) << (variables / 2);
    }

    std::vector<Fr> tensor_point(const std::vector<Fr> &row_point,
                                 const std::vector<Fr> &column_point,
                                 int variables)
    {
        std::vector<Fr> point(column_point);
        point.insert(point.end(), row_point.begin(), row_point.end());
        point.resize(static_cast<std::size_t>(variables), Fr(0));
        return point;
    }

    std::vector<Fr> opening_weights(const ZkEmbedProof &proof, bool input_view,
                                    FiatShamirTranscript &transcript)
    {
        transcript.append_u64(input_view ? 1 : 2);
        transcript.append_g1_vector(proof.query_reciprocal_opening.eval_proof.q_comm,
                                    G1WireEncoding::Compressed);
        if (!input_view)
        {
            transcript.append_g1_vector(proof.h0_opening.eval_proof.q_comm,
                                        G1WireEncoding::Compressed);
        }
        std::vector<Fr> weights(input_view ? 1 : 2);
        for (Fr &weight : weights)
            weight = transcript.nonzero_challenge();
        return weights;
    }

    void append_zkembed_statement(FiatShamirTranscript &transcript,
                                  int vocab_size, int embed_dim, int table_size,
                                  const ll *query_indices, int query_size,
                                  int query_domain, int column_domain,
                                  const G2 &table_vanishing_commitment,
                                  const G2 &table_index_commitment,
                                  const std::vector<G2> &table_column_commitments,
                                  const ZkEmbedOutputCommitment &output)
    {
        transcript.append_u64(static_cast<std::uint64_t>(vocab_size));
        transcript.append_u64(static_cast<std::uint64_t>(embed_dim));
        transcript.append_u64(static_cast<std::uint64_t>(table_size));
        transcript.append_u64(static_cast<std::uint64_t>(query_size));
        transcript.append_u64(static_cast<std::uint64_t>(query_domain));
        transcript.append_u64(static_cast<std::uint64_t>(column_domain));
        for (int i = 0; i < query_size; i++)
        {
            transcript.append_u64(static_cast<std::uint64_t>(query_indices[i]));
        }

        transcript.append_g2(table_vanishing_commitment);
        transcript.append_g2(table_index_commitment);
        transcript.append_u64(static_cast<std::uint64_t>(table_column_commitments.size()));
        for (const G2 &commitment : table_column_commitments)
        {
            transcript.append_g2(commitment);
        }

        transcript.append_u64(static_cast<std::uint64_t>(output.variables()));
        transcript.append_u64(output.isInputView() ? 1 : 0);
        transcript.append_u64(static_cast<std::uint64_t>(hyrax::Layout::ContiguousRows));
        if (output.isInputView())
        {
            transcript.append_g1_vector(output.inputRows());
            transcript.append_g1_vector(output.positionRows());
        }
        else
        {
            transcript.append_g1_vector(output.rows());
        }
    }

    void append_zkembed_zero_check_commitments(
        FiatShamirTranscript &transcript,
        const ZkEmbedProof &proof, const Fr &grand_sum,
        int b_commitment_count)
    {
        transcript.append_g1(proof.table_reciprocal_commitment);
        transcript.append_g1(proof.table_quotient_commitment);
        transcript.append_g1(proof.table_zero_opening_commitment);
        transcript.append_fr(proof.table_reciprocal_at_zero);
        transcript.append_fr(grand_sum);
        transcript.append_g1_array(proof.query_reciprocal_commitments.data(),
                                   static_cast<std::size_t>(b_commitment_count));
    }

    void append_sumcheck_round(
        FiatShamirTranscript &transcript, int round, const SumcheckRoundPoly &poly)
    {
        transcript.append_u64(static_cast<std::uint64_t>(round));
        for (int i = 0; i < 4; ++i)
        {
            transcript.append_fr(poly.coeffs[i]);
        }
    }

}

ZkEmbedOutputCommitment ZkEmbedOutputCommitment::tensor(
    int variables, std::vector<G1> rows)
{
    if (rows.size() != commitment_rows(variables))
    {
        throw std::invalid_argument("zkEmbed: tensor commitment shape mismatch");
    }
    ZkEmbedOutputCommitment output;
    output.variables_ = variables;
    output.rows_ = std::move(rows);
    return output;
}

ZkEmbedOutputCommitment ZkEmbedOutputCommitment::inputPrefix(
    int input_variables, std::vector<G1> input_rows,
    std::vector<G1> position_rows)
{
    if (input_rows.size() != commitment_rows(input_variables) ||
        position_rows.empty() || position_rows.size() > input_rows.size())
    {
        throw std::invalid_argument("zkEmbed: input/position commitment shape mismatch");
    }
    ZkEmbedOutputCommitment output;
    output.variables_ = input_variables;
    output.input_view_ = true;
    output.input_rows_ = std::move(input_rows);
    output.position_rows_ = std::move(position_rows);
    return output;
}

std::size_t ZkEmbedProof::cryptographicPayloadBytes() const
{
    const std::size_t group_elements = 4 + query_reciprocal_commitments.size() +
                                       query_reciprocal_opening.eval_proof.q_comm.size() +
                                       h0_opening.eval_proof.q_comm.size();
    const std::size_t field_elements =
        4 + 4 * (outer_sumcheck_polys.size() + inner_sumcheck_polys.size());
    return group_elements * G1::getSerializedByteSize() + field_elements * 32;
}

namespace
{
    template <class Function>
    void preprocess_for(int count, int threads, const Function &function)
    {
        if (threads <= 1)
        {
            for (int i = 0; i < count; ++i) function(i);
            return;
        }
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int i = 0; i < count; ++i) function(i);
    }
}

class ZkEmbedFFTPlan
{
public:
    explicit ZkEmbedFFTPlan(int size) : size_(size), levels_(0)
    {
        if (size <= 0 || (size & (size - 1)) != 0 || size > (1 << 28))
            throw std::invalid_argument("unsupported zkEmbed FFT domain");
        reversed_.resize(size);
        for (int n = size; n > 1; n >>= 1) ++levels_;
        Fr omega;
        omega.setStr("19103219067921713944291392827692070036145651957329286315305642004821462161904");
        for (int i = levels_; i < 28; ++i) Fr::sqr(omega, omega);
        root_ = omega;
        Fr inverse_omega;
        Fr::inv(inverse_omega, omega);
        Fr::inv(inverse_size_, Fr(size));
        for (int i = 1; i < size; ++i)
            reversed_[i] = (reversed_[i >> 1] >> 1) |
                           ((i & 1) << (levels_ - 1));
        build_twiddles(forward_, omega);
        build_twiddles(inverse_, inverse_omega);
    }

    int size() const { return size_; }
    const Fr &inverseSize() const { return inverse_size_; }
    const Fr &root() const { return root_; }

    template <class Value>
    void transform(Value *output, const Value *input, bool inverse, int threads) const
    {
        if (output == input)
            throw std::invalid_argument("zkEmbed FFT requires distinct input/output buffers");
        const auto &twiddles = inverse ? inverse_ : forward_;
        const auto butterfly = [&](int index, int level)
        {
            const int half = 1 << (level - 1);
            const int j = index & (half - 1);
            const int left = ((index >> (level - 1)) << level) + j;
            const int right = left + half;
            Value product, original = output[left];
            Value::mul(product, output[right], twiddles[level][j]);
            Value::add(output[left], original, product);
            Value::sub(output[right], original, product);
        };
        if (threads <= 1)
        {
            for (int i = 0; i < size_; ++i) output[reversed_[i]] = input[i];
            for (int level = 1; level <= levels_; ++level)
                for (int i = 0; i < size_ / 2; ++i) butterfly(i, level);
            if (inverse)
                for (int i = 0; i < size_; ++i)
                    Value::mul(output[i], output[i], inverse_size_);
            return;
        }
#pragma omp parallel num_threads(threads)
        {
#pragma omp for schedule(static)
            for (int i = 0; i < size_; ++i) output[reversed_[i]] = input[i];
            for (int level = 1; level <= levels_; ++level)
            {
#pragma omp for schedule(static)
                for (int i = 0; i < size_ / 2; ++i) butterfly(i, level);
            }
            if (inverse)
            {
#pragma omp for schedule(static)
                for (int i = 0; i < size_; ++i)
                    Value::mul(output[i], output[i], inverse_size_);
            }
        }
    }

private:
    void build_twiddles(std::vector<std::vector<Fr>> &table, const Fr &root)
    {
        table.resize(levels_ + 1);
        for (int level = 1; level <= levels_; ++level)
        {
            const int width = 1 << level;
            Fr step;
            Fr::pow(step, root, size_ / width);
            auto &values = table[level];
            values.resize(width / 2);
            values[0] = Fr(1);
            for (int j = 1; j < width / 2; ++j)
                Fr::mul(values[j], values[j - 1], step);
        }
    }

    int size_, levels_;
    Fr root_, inverse_size_;
    std::vector<int> reversed_;
    std::vector<std::vector<Fr>> forward_, inverse_;
};

namespace
{
    struct ColumnPreprocessWorkspace
    {
        explicit ColumnPreprocessWorkspace(int n)
            : evaluations(n, Fr(0)), coefficients(n), convolution_input(n),
              convolution_output(n), points(n), transformed_points(n) {}
        std::vector<Fr> evaluations, coefficients, convolution_input, convolution_output;
        std::vector<G1> points, transformed_points;
    };

    void preprocess_column(const ZkEmbedFFTPlan &plan, KZG_ProverKey &key,
                           const std::vector<G1> &lagrange,
                           const std::vector<G1> &phi,
                           const std::vector<Fr> &dft_w,
                           const std::vector<Fr> &scaled_dft_w_prime,
                           ColumnPreprocessWorkspace &work,
                           G2 &commitment, std::vector<G1> &quotients, int threads)
    {
        const int n = plan.size();
        plan.transform(work.coefficients.data(), work.evaluations.data(), true, threads);
        if (threads <= 1)
            G2::mulVec(commitment, key.srs_g2.data(), work.coefficients.data(), n);
        else
            G2::mulVecMT(commitment, key.srs_g2.data(), work.coefficients.data(), n, threads);
        quotients.resize(n);
        preprocess_for(n, threads, [&](int i) {
            G1::mul(work.points[i], lagrange[i], work.evaluations[i]);
        });
        plan.transform(work.transformed_points.data(), work.points.data(), false, threads);
        preprocess_for(n, threads, [&](int k) {
            G1::mul(work.points[k], work.transformed_points[k], dft_w[k]);
            // FFT(f)[k] = N * IFFT(f)[-k mod N]; N is folded into the kernel.
            Fr::mul(work.convolution_input[k], work.coefficients[k == 0 ? 0 : n - k],
                    scaled_dft_w_prime[k]);
        });
        plan.transform(quotients.data(), work.points.data(), true, threads);
        plan.transform(work.convolution_output.data(), work.convolution_input.data(), true, threads);
        preprocess_for(n, threads, [&](int i) {
            G1 term2, term3;
            G1::mul(term2, lagrange[i], work.convolution_output[i]);
            G1::mul(term3, phi[i], work.evaluations[i]);
            G1::add(quotients[i], quotients[i], term2);
            G1::add(quotients[i], quotients[i], term3);
        });
    }
}

int ZkEmbed::requiredTableSize(int vocab_size)
{
    if (vocab_size <= 0)
    {
        throw std::invalid_argument("ZkEmbed: vocab_size must be positive");
    }

    int domain_size = 1;
    while (domain_size <= vocab_size)
    {
        if (domain_size > (1 << 29))
        {
            throw std::overflow_error("ZkEmbed: table domain is too large");
        }
        domain_size <<= 1;
    }
    return domain_size;
}

ZkEmbed::ZkEmbed(KZG_ProverKey &uni_pk, KZG_VerifierKey &uni_vk,
                 MVKZG_ProverKey &mv_pk, const MVKZG_VerifierKey &mv_vk,
                 int v_size, int e_dim, int t_num)
    : vocab_size(v_size), embed_dim(e_dim), thread_num(t_num),
      uni_pk_(uni_pk), uni_vk_(uni_vk), mv_pk_(mv_pk), mv_vk_(mv_vk),
      embedding_matrix_copy(nullptr), cache_fd(-1),
      offset_com1_li_x(0), offset_com1_li_0(0),
      offset_com2_vanishing(0), offset_com2_idx(0), offset_com2_columns(0),
      offset_com1_qi_columns(0), offset_com1_qi_idx(0), com2_loaded(false)
{
    table_domain_size = requiredTableSize(vocab_size);
    if (embed_dim <= 0 || thread_num <= 0)
    {
        throw std::invalid_argument("zkEmbed: invalid embedding dimension or thread count");
    }
}

ZkEmbed::~ZkEmbed()
{
    if (embedding_matrix_copy)
        delete[] embedding_matrix_copy;
    close_cache();
}

void ZkEmbed::compute_w_vectors(const ZkEmbedFFTPlan &plan)
{
    const int N = table_domain_size;
    const Fr half = Fr(N - 1) / Fr(2);
    dft_w_vec.resize(N);
    scaled_dft_w_prime_vec.resize(N);
    phi_i_i.resize(N);
    std::vector<G1> phi_coefficients(N);
    preprocess_for(N, thread_num, [&](int k) {
        // DFT(w)[k] = (k - (N-1)/2)/N, and w_prime is the cyclic reversal of w.
        dft_w_vec[k] = (Fr(k) - half) * plan.inverseSize();
        scaled_dft_w_prime_vec[k] = Fr(k == 0 ? 0 : N - k) - half;
        // phi = IFFT(( -sum(w) - DFT(w) ) * SRS).
        const Fr coefficient = Fr(N - 1 - k) * plan.inverseSize();
        G1::mul(phi_coefficients[k], uni_pk_.srs_g1[k], coefficient);
    });
    plan.transform(phi_i_i.data(), phi_coefficients.data(), true, thread_num);
}

void ZkEmbed::compute_shared_preprocess(const ZkEmbedFFTPlan &plan)
{
    const int N = table_domain_size;
    G2::sub(com2_vanishing, uni_pk_.srs_g2[N], uni_pk_.srs_g2[0]);
    com1_li_x.resize(N);
    plan.transform(com1_li_x.data(), uni_pk_.srs_g1.data(), true, thread_num);
    com1_li_0.resize(N);
    G1 scaled_x_nm1;
    G1::mul(scaled_x_nm1, uni_pk_.srs_g1[N - 1], plan.inverseSize());
    preprocess_for(N, thread_num, [&](int i) {
        const Fr &omega_neg_i = omega_powers[i == 0 ? 0 : N - i];
        G1 scaled_li;
        G1::mul(scaled_li, com1_li_x[i], omega_neg_i);
        G1::sub(com1_li_0[i], scaled_li, scaled_x_nm1);
    });
    compute_w_vectors(plan);
}

void ZkEmbed::compute_index_data(const ZkEmbedFFTPlan &plan)
{
    ColumnPreprocessWorkspace work(table_domain_size);
    preprocess_for(table_domain_size, thread_num, [&](int i) {
        work.evaluations[i] = Fr(i);
    });
    preprocess_column(plan, uni_pk_, com1_li_x, phi_i_i, dft_w_vec,
                      scaled_dft_w_prime_vec, work, com2_idx, com1_qi_idx, thread_num);
}

void ZkEmbed::compute_column_data(const ll *embedding_mat, const ZkEmbedFFTPlan &plan)
{
    const int N = table_domain_size;
    const int d = embed_dim;
    com2_columns.resize(d);
    com1_qi_columns.resize(d);
    const auto fill_column = [&](ColumnPreprocessWorkspace &work, int j, int threads) {
        preprocess_for(vocab_size, threads, [&](int i) {
            work.evaluations[i] = Fr(embedding_mat[static_cast<long long>(i) * d + j]);
        });
    };
    if (d == 1 || thread_num == 1)
    {
        ColumnPreprocessWorkspace work(N);
        for (int j = 0; j < d; ++j)
        {
            fill_column(work, j, thread_num);
            preprocess_column(plan, uni_pk_, com1_li_x, phi_i_i, dft_w_vec,
                              scaled_dft_w_prime_vec, work, com2_columns[j],
                              com1_qi_columns[j], thread_num);
        }
        return;
    }
    const int workers = std::min(thread_num, d);
#pragma omp parallel num_threads(workers)
    {
        ColumnPreprocessWorkspace work(N);
#pragma omp for schedule(static)
        for (int j = 0; j < d; ++j)
        {
            fill_column(work, j, 1);
            preprocess_column(plan, uni_pk_, com1_li_x, phi_i_i, dft_w_vec,
                              scaled_dft_w_prime_vec, work, com2_columns[j],
                              com1_qi_columns[j], 1);
        }
    }
}

void ZkEmbed::preprocess(const ll *embedding_mat, long long *preprocess_time_ms,
                         long long *cache_save_ms)
{
    const ScopedOmpThreads thread_scope(thread_num);
    if (!embedding_mat)
    {
        throw std::invalid_argument("ZkEmbed::preprocess: null embedding table");
    }
    std::cout << "\n=== ZkEmbed Setup ===" << std::endl;
    std::cout << "  vocab_size=" << vocab_size << ", embed_dim=" << embed_dim
              << ", table_domain_size=" << table_domain_size << std::endl;

    long long size = (long long)vocab_size * embed_dim;
    close_cache();
    prefetched_prover_rows_ = PrefetchedProverRows{};
    if (embedding_matrix_copy)
    {
        delete[] embedding_matrix_copy;
        embedding_matrix_copy = nullptr;
    }
    embedding_matrix_copy = new ll[size];
    memcpy(embedding_matrix_copy, embedding_mat, sizeof(ll) * size);

    if ((int)uni_pk_.srs_g1.size() < table_domain_size || (int)uni_pk_.srs_g2.size() < table_domain_size + 1)
    {
        std::cerr << "Error: KZG SRS too small for ZkEmbed table (need g1 >= "
                  << table_domain_size << ", g2 >= " << (table_domain_size + 1) << ", have g1="
                  << uni_pk_.srs_g1.size() << ", g2=" << uni_pk_.srs_g2.size() << ")" << std::endl;
        throw std::runtime_error("KZG SRS too small for ZkEmbed table");
    }
    if (preprocess_time_ms)
        *preprocess_time_ms = 0;
    if (cache_save_ms)
        *cache_save_ms = 0;

    std::string cache_file = cacheFilename(vocab_size, embed_dim);
    if (load_cache(cache_file))
    {
        return;
    }

    std::cout << "  Computing preprocess data..." << std::endl;
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    {
        const ZkEmbedFFTPlan plan(table_domain_size);
        omega = plan.root();
        omega_powers.resize(table_domain_size);
        omega_powers[0] = Fr(1);
        for (int i = 1; i < table_domain_size; ++i)
            Fr::mul(omega_powers[i], omega_powers[i - 1], omega);
        compute_shared_preprocess(plan);
        compute_index_data(plan);
        compute_column_data(embedding_mat, plan);
    }

    omega_powers.clear();
    dft_w_vec.clear();
    scaled_dft_w_prime_vec.clear();
    phi_i_i.clear();
    com2_loaded = true;
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        preprocess_end - preprocess_start).count();
    if (preprocess_time_ms)
        *preprocess_time_ms = elapsed_ms;
    else
        std::cout << "  Preprocess: " << elapsed_ms << " ms" << std::endl;

    auto save_start = std::chrono::high_resolution_clock::now();
    if (!save_cache(cache_file))
    {
        throw std::runtime_error("ZkEmbed: failed to save preprocessing cache");
    }
    auto save_end = std::chrono::high_resolution_clock::now();
    if (cache_save_ms)
    {
        *cache_save_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(save_end - save_start).count();
    }
}

std::string ZkEmbed::cacheFilename(int vocab_size, int embed_dim)
{
    return "cache/zkembed_vocab" + std::to_string(vocab_size) +
           "_dim" + std::to_string(embed_dim) + ".bin";
}

namespace
{

    template <class ColumnData>
    void write_cache_g1_columns(
        std::ofstream &out, std::size_t rows, std::size_t columns,
        std::size_t slot_bytes, int requested_threads,
        const ColumnData &column_data)
    {
        if (rows == 0 || columns == 0)
            return;
        constexpr std::size_t BufferBytes = 32ULL * 1024 * 1024;
        constexpr std::size_t NormalizeBatch = 1024;
        const std::size_t encoded_bytes = 2 * G1::Fp::getByteSize();
        if (encoded_bytes > slot_bytes)
        {
            throw std::runtime_error("ZkEmbed: G1 encoding exceeds cache slot");
        }
        const std::size_t chunk_rows = std::min(rows, std::max<std::size_t>(
                                                          1, BufferBytes / slot_bytes / columns));
        std::vector<std::uint8_t> buffer(chunk_rows * columns * slot_bytes, 0);
        for (std::size_t begin = 0; begin < rows; begin += chunk_rows)
        {
            const std::size_t count = std::min(chunk_rows, rows - begin);
            const std::size_t batches =
                (count + NormalizeBatch - 1) / NormalizeBatch;
            const std::size_t tasks = columns * batches;
            const int threads = static_cast<int>(std::min<std::size_t>(
                std::max(1, requested_threads), tasks));
            std::atomic<bool> valid(true);
#pragma omp parallel num_threads(threads) if (threads > 1)
            {
                std::vector<G1> normalized(std::min(count, NormalizeBatch));
#pragma omp for schedule(static)
                for (long long task = 0; task < static_cast<long long>(tasks); ++task)
                {
                    const std::size_t column = static_cast<std::size_t>(task) / batches;
                    const std::size_t first =
                        (static_cast<std::size_t>(task) % batches) * NormalizeBatch;
                    const std::size_t batch_size =
                        std::min(NormalizeBatch, count - first);
                    G1::normalizeVec(normalized.data(),
                                     column_data(column) + begin + first, batch_size);
                    for (std::size_t i = 0; i < batch_size; ++i)
                    {
                        std::uint8_t *destination = buffer.data() + ((first + i) * columns + column) * slot_bytes;
                        if (normalized[i].serialize(destination, slot_bytes,
                                                    mcl::IoEcAffineSerialize) != encoded_bytes)
                        {
                            valid.store(false, std::memory_order_relaxed);
                        }
                    }
                }
            }
            if (!valid.load(std::memory_order_relaxed))
            {
                throw std::runtime_error("ZkEmbed: G1 cache serialization failed");
            }
            out.write(reinterpret_cast<const char *>(buffer.data()),
                      static_cast<std::streamsize>(count * columns * slot_bytes));
        }
    }

}

bool ZkEmbed::save_cache(const std::string &filename)
{
    std::cout << "  Saving cache to " << filename << "..." << std::endl;

    if (mkdir("cache", 0755) != 0 && errno != EEXIST)
    {
        return false;
    }

    const std::string temporary = filename + ".tmp";
    try
    {
        std::ofstream ofs(temporary, std::ios::binary | std::ios::trunc);
        if (!ofs)
        {
            std::cerr << "  Failed to open cache file for writing: " << filename << std::endl;
            return false;
        }
        ofs.exceptions(std::ios::badbit | std::ios::failbit);

        ofs.write(EmbeddingCacheType, sizeof(EmbeddingCacheType) - 1);
        ofs.write(reinterpret_cast<const char *>(&vocab_size), sizeof(vocab_size));
        ofs.write(reinterpret_cast<const char *>(&embed_dim), sizeof(embed_dim));
        ofs.write(reinterpret_cast<const char *>(&table_domain_size), sizeof(table_domain_size));

        std::streampos offset_position = ofs.tellp();
        size_t dummy_offset = 0;
        ofs.write(reinterpret_cast<const char *>(&dummy_offset), sizeof(size_t));
        ofs.write(reinterpret_cast<const char *>(&dummy_offset), sizeof(size_t));
        ofs.write(reinterpret_cast<const char *>(&dummy_offset), sizeof(size_t));
        ofs.write(reinterpret_cast<const char *>(&dummy_offset), sizeof(size_t));

        auto write_g1_vector = [&](const std::vector<G1> &values)
        {
            write_cache_g1_columns(ofs, values.size(), 1, G1_SIZE, thread_num,
                                   [&](std::size_t)
                                   { return values.data(); });
        };

        auto write_g2 = [&](const G2 &g2)
        {
            uint8_t buf[G2_SIZE] = {};
            if (g2.serialize(buf, sizeof(buf), mcl::IoEcAffineSerialize) != 2 * G2::Fp::getByteSize())
            {
                throw std::runtime_error("ZkEmbed: G2 cache serialization failed");
            }
            ofs.write(reinterpret_cast<const char *>(buf), sizeof(buf));
        };

        write_g2(com2_vanishing);
        write_g2(com2_idx);

        size_t actual_offset_com1_li_x = ofs.tellp();
        write_g1_vector(com1_li_x);

        size_t actual_offset_com1_li_0 = ofs.tellp();
        write_g1_vector(com1_li_0);

        for (const auto &g2 : com2_columns)
            write_g2(g2);

        size_t actual_offset_com1_qi_columns = ofs.tellp();
        write_cache_g1_columns(ofs, table_domain_size, embed_dim, G1_SIZE, thread_num,
                               [&](std::size_t column)
                               { return com1_qi_columns[column].data(); });

        size_t actual_offset_com1_qi_idx = ofs.tellp();
        write_g1_vector(com1_qi_idx);

        ofs.seekp(offset_position);
        ofs.write(reinterpret_cast<const char *>(&actual_offset_com1_li_x), sizeof(size_t));
        ofs.write(reinterpret_cast<const char *>(&actual_offset_com1_li_0), sizeof(size_t));
        ofs.write(reinterpret_cast<const char *>(&actual_offset_com1_qi_columns), sizeof(size_t));
        ofs.write(reinterpret_cast<const char *>(&actual_offset_com1_qi_idx), sizeof(size_t));

        ofs.close();
        if (std::rename(temporary.c_str(), filename.c_str()) != 0)
        {
            throw std::runtime_error("ZkEmbed: cannot publish preprocessing cache");
        }
        std::cout << "  Cache saved successfully (lean interleaved format)" << std::endl;
        return true;
    }
    catch (const std::exception &error)
    {
        std::remove(temporary.c_str());
        std::cerr << "  Failed to save cache: " << error.what() << std::endl;
        return false;
    }
}

bool ZkEmbed::load_cache(const std::string &filename)
{
    std::cout << "  Loading cache from " << filename << "..." << std::endl;

    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
    {
        std::cout << "  Cache file not found" << std::endl;
        return false;
    }

    char cache_type[sizeof(EmbeddingCacheType) - 1] = {};
    ifs.read(cache_type, sizeof(cache_type));
    if (!ifs || std::memcmp(cache_type, EmbeddingCacheType, sizeof(cache_type)) != 0)
    {
        std::cerr << "  Incompatible embedding cache format; rebuild required" << std::endl;
        return false;
    }
    int cached_vocab_size = 0, cached_embed_dim = 0, cached_table_domain_size = 0;
    ifs.read(reinterpret_cast<char *>(&cached_vocab_size), sizeof(cached_vocab_size));
    ifs.read(reinterpret_cast<char *>(&cached_embed_dim), sizeof(cached_embed_dim));
    ifs.read(reinterpret_cast<char *>(&cached_table_domain_size), sizeof(cached_table_domain_size));
    if (!ifs || cached_vocab_size != vocab_size || cached_embed_dim != embed_dim || cached_table_domain_size != table_domain_size)
    {
        std::cerr << "  Cache parameters mismatch" << std::endl;
        return false;
    }

    ifs.read(reinterpret_cast<char *>(&offset_com1_li_x), sizeof(size_t));
    ifs.read(reinterpret_cast<char *>(&offset_com1_li_0), sizeof(size_t));
    ifs.read(reinterpret_cast<char *>(&offset_com1_qi_columns), sizeof(size_t));
    ifs.read(reinterpret_cast<char *>(&offset_com1_qi_idx), sizeof(size_t));
    if (!ifs)
        return false;

    size_t data_start = static_cast<size_t>(ifs.tellg());
    offset_com2_vanishing = data_start;
    offset_com2_idx = data_start + G2_SIZE;
    offset_com2_columns = offset_com1_qi_columns - static_cast<size_t>(embed_dim) * G2_SIZE;

    ifs.close();

    cache_fd = open(filename.c_str(), O_RDONLY);
    if (cache_fd < 0)
    {
        std::cerr << "  Failed to open cache file for lazy loading: " << filename << std::endl;
        return false;
    }

    com2_columns.clear();
    com2_loaded = false;

    std::cout << "  Cache loaded successfully (interleaved; query preload required)"
              << std::endl;
    return true;
}

void ZkEmbed::close_cache()
{
    if (cache_fd >= 0)
    {
        ::close(cache_fd);
        cache_fd = -1;
    }
}

void ZkEmbed::ensure_com2_loaded()
{
    if (com2_loaded)
    {
        return;
    }
    if (cache_fd < 0)
    {

        if (embed_dim > 0 && com2_columns.empty())
        {
            std::cerr << "Error: com2_columns missing without cache file" << std::endl;
            return;
        }
        com2_loaded = true;
        return;
    }

    auto read_g2_at = [&](G2 &g2, size_t offset)
    {
        uint8_t buf[G2_SIZE];
        ssize_t bytes_read = pread(cache_fd, buf, G2_SIZE, offset);
        if (bytes_read != (ssize_t)G2_SIZE)
        {
            std::cerr << "Error: failed to read G2 at offset " << offset << std::endl;
            return false;
        }
        g2.deserialize(buf, sizeof(buf), mcl::IoEcAffineSerialize);
        return true;
    };

    if (!read_g2_at(com2_vanishing, offset_com2_vanishing))
    {
        return;
    }
    if (!read_g2_at(com2_idx, offset_com2_idx))
    {
        return;
    }

    com2_columns.resize(embed_dim);
    if (embed_dim > 0)
    {
        size_t bytes_to_read = static_cast<size_t>(embed_dim) * G2_SIZE;
        std::vector<uint8_t> buffer(bytes_to_read);
        ssize_t bytes_read = pread(cache_fd, buffer.data(), bytes_to_read, offset_com2_columns);
        if (bytes_read != (ssize_t)bytes_to_read)
        {
            std::cerr << "Error: failed to read com2_columns from cache" << std::endl;
            return;
        }
#pragma omp parallel for
        for (int j = 0; j < embed_dim; j++)
        {
            com2_columns[j].deserialize(buffer.data() + static_cast<size_t>(j) * G2_SIZE,
                                        G2_SIZE, mcl::IoEcAffineSerialize);
        }
    }

    com2_loaded = true;
}

void ZkEmbed::preloadVerifierData()
{
    const ScopedOmpThreads thread_scope(thread_num);
    ensure_com2_loaded();
}

void ZkEmbed::preloadProverDataForQueries(
    const std::vector<ll> &query_indices)
{
    const ScopedOmpThreads thread_scope(thread_num);
    padded_variables(query_indices.size());
    if (!embedding_matrix_copy)
    {
        throw std::logic_error(
            "ZkEmbed::preloadProverDataForQueries: setup must be called first");
    }
    std::vector<int> rows;
    rows.reserve(query_indices.size() + 1);
    for (ll index : query_indices)
    {
        if (index < 0 || index >= vocab_size)
        {
            throw std::out_of_range(
                "ZkEmbed::preloadProverDataForQueries: query outside vocabulary");
        }
        rows.push_back(static_cast<int>(index));
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    if ((query_indices.size() & (query_indices.size() - 1)) != 0)
    {
        rows.push_back(vocab_size);
    }
    ensure_com2_loaded();
    if (!com2_loaded)
    {
        throw std::runtime_error(
            "ZkEmbed::preloadProverDataForQueries: verifier commitments unavailable");
    }
    if (rows == prefetched_prover_rows_.indices)
        return;

    PrefetchedProverRows loaded;
    loaded.indices = std::move(rows);
    if (cache_fd < 0)
    {
        const std::size_t domain = static_cast<std::size_t>(table_domain_size);
        if (com1_li_x.size() != domain || com1_li_0.size() != domain ||
            com1_qi_idx.size() != domain ||
            com1_qi_columns.size() != static_cast<std::size_t>(embed_dim))
            throw std::logic_error("ZkEmbed: resident preprocessing data unavailable");
        for (const auto &column : com1_qi_columns)
            if (column.size() != domain)
                throw std::logic_error("ZkEmbed: resident preprocessing column incomplete");
        const std::size_t count = loaded.indices.size();
        loaded.li_x.resize(count);
        loaded.li_0.resize(count);
        loaded.qi_idx.resize(count);
        loaded.qi_columns.resize(embed_dim);
        for (auto &column : loaded.qi_columns) column.resize(count);
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < count; ++i)
        {
            const int row = loaded.indices[i];
            loaded.li_x[i] = com1_li_x[row];
            loaded.li_0[i] = com1_li_0[row];
            loaded.qi_idx[i] = com1_qi_idx[row];
        }
#pragma omp parallel for schedule(static)
        for (int j = 0; j < embed_dim; ++j)
            for (std::size_t i = 0; i < count; ++i)
                loaded.qi_columns[j][i] = com1_qi_columns[j][loaded.indices[i]];
    }
    else
    {
        loaded.li_x = read_g1_rows(offset_com1_li_x, loaded.indices);
        loaded.li_0 = read_g1_rows(offset_com1_li_0, loaded.indices);
        loaded.qi_idx = read_qi_idx_rows(loaded.indices);
        batch_read_all_qi_columns(loaded.indices, loaded.qi_columns);
    }
    prefetched_prover_rows_ = std::move(loaded);
}

namespace
{
    void read_cache_row_exact(int fd, void *destination, std::size_t bytes,
                              std::size_t offset)
    {
        std::size_t completed = 0;
        auto *output = static_cast<std::uint8_t *>(destination);
        while (completed < bytes)
        {
            const ssize_t count = pread(fd, output + completed, bytes - completed,
                                        static_cast<off_t>(offset + completed));
            if (count < 0 && errno == EINTR)
                continue;
            if (count <= 0)
                throw std::runtime_error("ZkEmbed: truncated or unreadable cache row");
            completed += static_cast<std::size_t>(count);
        }
    }
}

std::vector<G1> ZkEmbed::read_g1_rows(size_t offset, const std::vector<int> &row_indices)
{
    int num_rows = row_indices.size();
    std::vector<G1> result(num_rows);
    if (num_rows == 0)
        return result;
    if (cache_fd < 0)
    {
        throw std::runtime_error("ZkEmbed: cache file not open for row loading");
    }

    std::vector<std::pair<int, int>> sorted_rows(num_rows);
    for (int i = 0; i < num_rows; i++)
    {
        if (row_indices[i] < 0 || row_indices[i] >= table_domain_size)
            throw std::out_of_range("ZkEmbed: cached row outside table domain");
        sorted_rows[i] = {row_indices[i], i};
    }
    std::sort(sorted_rows.begin(), sorted_rows.end());

    bool contiguous = true;
    for (int i = 1; i < num_rows && contiguous; i++)
    {
        if (sorted_rows[i].first != sorted_rows[i - 1].first + 1)
        {
            contiguous = false;
        }
    }

    if (contiguous && num_rows > 1)
    {

        size_t total_bytes = num_rows * G1_SIZE;
        std::vector<uint8_t> buffer(total_bytes);
        size_t start_offset = offset + static_cast<size_t>(sorted_rows[0].first) * G1_SIZE;
        read_cache_row_exact(cache_fd, buffer.data(), total_bytes, start_offset);
#pragma omp parallel for
        for (int i = 0; i < num_rows; i++)
        {
            int orig_pos = sorted_rows[i].second;
            result[orig_pos].deserialize(buffer.data() + i * G1_SIZE, G1_SIZE, mcl::IoEcAffineSerialize);
        }
        return result;
    }

    std::vector<uint8_t> buffer(num_rows * G1_SIZE);
    for (int i = 0; i < num_rows; i++)
    {
        int row_idx = sorted_rows[i].first;
        size_t row_offset = offset + static_cast<size_t>(row_idx) * G1_SIZE;
        read_cache_row_exact(cache_fd, buffer.data() + i * G1_SIZE, G1_SIZE, row_offset);
    }

#pragma omp parallel for
    for (int i = 0; i < num_rows; i++)
    {
        int orig_pos = sorted_rows[i].second;
        result[orig_pos].deserialize(buffer.data() + i * G1_SIZE, G1_SIZE, mcl::IoEcAffineSerialize);
    }

    return result;
}

std::vector<G1> ZkEmbed::read_qi_idx_rows(const std::vector<int> &row_indices)
{
    return read_g1_rows(offset_com1_qi_idx, row_indices);
}

void ZkEmbed::batch_read_all_qi_columns(const std::vector<int> &row_indices,
                                        std::vector<std::vector<G1>> &out_qi_columns)
{
    int num_rows = row_indices.size();
    int d = embed_dim;
    out_qi_columns.resize(d);

    if (num_rows == 0)
    {
        for (int j = 0; j < d; j++)
            out_qi_columns[j].resize(0);
        return;
    }
    if (cache_fd < 0)
        throw std::runtime_error("ZkEmbed: cache file not open for column loading");

    std::vector<std::pair<int, int>> sorted_rows(num_rows);
    for (int i = 0; i < num_rows; i++)
    {
        if (row_indices[i] < 0 || row_indices[i] >= table_domain_size)
            throw std::out_of_range("ZkEmbed: cached row outside table domain");
        sorted_rows[i] = {row_indices[i], i};
    }
    std::sort(sorted_rows.begin(), sorted_rows.end());

    bool contiguous = true;
    for (int i = 1; i < num_rows && contiguous; i++)
    {
        if (sorted_rows[i].first != sorted_rows[i - 1].first + 1)
        {
            contiguous = false;
        }
    }

    for (int j = 0; j < d; j++)
    {
        out_qi_columns[j].resize(num_rows);
    }

    size_t bytes_per_row = d * G1_SIZE;
    size_t total_bytes = num_rows * bytes_per_row;
    std::vector<uint8_t> all_buffer(total_bytes);

    if (contiguous && num_rows > 1)
    {
        size_t start_offset = offset_com1_qi_columns +
                              static_cast<size_t>(sorted_rows[0].first) * bytes_per_row;
        read_cache_row_exact(cache_fd, all_buffer.data(), total_bytes, start_offset);
    }
    else
    {
        for (int i = 0; i < num_rows; i++)
        {
            int row_idx = sorted_rows[i].first;
            size_t row_offset = offset_com1_qi_columns +
                                static_cast<size_t>(row_idx) * bytes_per_row;
            read_cache_row_exact(cache_fd, all_buffer.data() + i * bytes_per_row,
                                 bytes_per_row, row_offset);
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < d; j++)
        {
            int orig_pos = sorted_rows[i].second;
            uint8_t *ptr = all_buffer.data() + i * bytes_per_row + j * G1_SIZE;
            out_qi_columns[j][orig_pos].deserialize(
                ptr, G1_SIZE, mcl::IoEcAffineSerialize);
        }
    }
}

std::vector<Fr> ZkEmbed::prepareWitness(
    std::size_t query_size, const std::vector<ll> &output_rows) const
{
    const int row_vars = padded_variables(query_size);
    const int col_vars = padded_variables(static_cast<std::size_t>(embed_dim));
    if (row_vars + col_vars > 30 ||
        output_rows.size() != query_size * static_cast<std::size_t>(embed_dim))
    {
        throw std::invalid_argument("zkEmbed: output witness shape mismatch");
    }
    const std::size_t columns = std::size_t(1) << col_vars;
    std::vector<Fr> values(std::size_t(1) << (row_vars + col_vars), Fr(0));
    for (std::size_t row = 0; row < query_size; ++row)
    {
        for (int col = 0; col < embed_dim; ++col)
        {
            values[row * columns + col] = Fr(output_rows[row * embed_dim + col]);
        }
    }
    return values;
}

bool ZkEmbed::validOutput(const ZkEmbedStatement &statement) const
{
    try
    {
        const int row_vars = padded_variables(statement.query_indices.size());
        const int col_vars = padded_variables(static_cast<std::size_t>(embed_dim));
        const int tensor_vars = row_vars + col_vars;
        const int source_vars = statement.h0.variables();
        if (tensor_vars > 30 || source_vars < tensor_vars || source_vars > 30 ||
            (statement.h0.isInputView() ? statement.h0.inputRows() : statement.h0.rows()).size() !=
                commitment_rows(source_vars))
            return false;
        if (!statement.h0.isInputView())
            return source_vars == tensor_vars;
        const int source_col_vars = source_vars - source_vars / 2;
        const std::size_t prefix_rows = std::size_t(1) << std::max(0, tensor_vars - source_col_vars);
        return statement.h0.positionRows().size() == prefix_rows;
    }
    catch (const std::exception &)
    {
        return false;
    }
}

void ZkEmbed::validateWitness(const ZkEmbedStatement &statement,
                              const std::vector<Fr> &h0_values) const
{
    if (!validOutput(statement))
    {
        throw std::invalid_argument("zkEmbed: output commitment/view shape mismatch");
    }
    const int variables = padded_variables(statement.query_indices.size()) +
                          padded_variables(static_cast<std::size_t>(embed_dim));
    if (h0_values.size() != (std::size_t(1) << variables))
    {
        throw std::invalid_argument("zkEmbed: prepared witness shape mismatch");
    }
    const int u_vars = padded_variables(statement.query_indices.size());
    const int needed = statement.h0.isInputView()
                           ? u_vars - u_vars / 2
                           : variables - variables / 2;
    if (mv_pk_.srs_g1.size() < (std::size_t(1) << needed))
    {
        throw std::invalid_argument("zkEmbed: prover key too small for query commitments");
    }
}

ZkEmbedStatement ZkEmbed::makeStatement(
    const std::vector<ll> &query_indices, const std::vector<Fr> &h0_values) const
{
    const ScopedOmpThreads thread_scope(thread_num);
    const int variables = padded_variables(query_indices.size()) +
                          padded_variables(static_cast<std::size_t>(embed_dim));
    if (variables > 30 || h0_values.size() != (std::size_t(1) << variables) ||
        mv_pk_.srs_g1.size() < (std::size_t(1) << (variables - variables / 2)))
    {
        throw std::invalid_argument("zkEmbed: tensor witness or prover key has the wrong size");
    }
    G1 *rows = hyrax::commit(h0_values.data(), mv_pk_.srs_g1.data(),
                             variables, thread_num, hyrax::Layout::ContiguousRows);
    std::vector<G1> commitments(rows, rows + commitment_rows(variables));
    delete[] rows;
    return {query_indices, ZkEmbedOutputCommitment::tensor(variables, std::move(commitments))};
}

ZkEmbedProof ZkEmbed::prove(const ZkEmbedStatement &statement,
                            const std::vector<Fr> &h0_values, ZkEmbedTiming &timing,
                            ZkEmbedDebugTrace *debug_trace)
{
    FiatShamirTranscript transcript;
    return prove(statement, h0_values, transcript, timing, debug_trace);
}

ZkEmbedProof ZkEmbed::prove(const ZkEmbedStatement &statement,
                            const std::vector<Fr> &h0_values,
                            FiatShamirTranscript &transcript, ZkEmbedTiming &timing,
                            ZkEmbedDebugTrace *debug_trace)
{
    validateWitness(statement, h0_values);
    if (statement.h0.isInputView())
    {
        throw std::invalid_argument("zkEmbed: use proveToInput for an input view");
    }
    return proveImpl(statement, h0_values, thread_num, transcript, timing, debug_trace);
}

ZkEmbedProof ZkEmbed::proveToInput(
    const ZkEmbedStatement &statement, const std::vector<Fr> &h0_values,
    FiatShamirTranscript &transcript, ZkEmbedTiming &timing,
    ZkEmbedDebugTrace *debug_trace)
{
    validateWitness(statement, h0_values);
    if (!statement.h0.isInputView())
    {
        throw std::invalid_argument("zkEmbed: proveToInput requires an input view");
    }
    return proveImpl(statement, h0_values, thread_num, transcript, timing, debug_trace);
}

ZkEmbedProof ZkEmbed::proveImpl(const ZkEmbedStatement &statement,
                                const std::vector<Fr> &h0_values, int num_threads,
                                FiatShamirTranscript &transcript,
                                ZkEmbedTiming &timing,
                                ZkEmbedDebugTrace *debug_trace)
{
    const ScopedOmpThreads thread_scope(num_threads);
    auto total_start = std::chrono::high_resolution_clock::now();
    timing.lazy_load_ms = 0;
    timing.lazy_loaded_g1_points = 0;
    timing.prove_compute_ms = 0;

    ZkEmbedProof proof;
    int N = table_domain_size;
    int d = embed_dim;
    const ll *query_indices = statement.query_indices.data();
    const int query_size = static_cast<int>(statement.query_indices.size());

    if (query_size <= 0)
    {
        throw std::invalid_argument("ZkEmbed::prove: query_size must be positive");
    }
    if (!embedding_matrix_copy)
    {
        throw std::logic_error("ZkEmbed::prove: setup must be called first");
    }
    for (int i = 0; i < query_size; i++)
    {
        if (query_indices[i] < 0 || query_indices[i] >= vocab_size)
        {
            throw std::out_of_range(
                "ZkEmbed::prove: query index outside vocabulary");
        }
    }

    int query_domain = 1;
    int query_l = 0;
    while (query_domain < query_size)
    {
        query_domain <<= 1;
        query_l++;
    }

    int col_domain = 1;
    int col_l = 0;
    while (col_domain < d)
    {
        col_domain <<= 1;
        col_l++;
    }
    const int h0_rownum = query_domain;
    const int h0_colnum = col_domain;
    const int expected_h0_l = query_l + col_l;

    if (!com2_loaded || prefetched_prover_rows_.indices.empty())
        throw std::logic_error("ZkEmbed: call preloadProverDataForQueries before proving");

    const int dummy_index = vocab_size;
    const int padding_count = query_domain - query_size;

    std::vector<int> sorted_query_indices(static_cast<std::size_t>(query_size));
    for (int i = 0; i < query_size; ++i)
    {
        sorted_query_indices[static_cast<std::size_t>(i)] =
            static_cast<int>(query_indices[i]);
    }
    std::sort(sorted_query_indices.begin(), sorted_query_indices.end());
    std::vector<ll> mult_counts;
    std::vector<int> mult_indices;
    mult_counts.reserve(
        statement.query_indices.size() + (padding_count > 0 ? 1 : 0));
    mult_indices.reserve(mult_counts.capacity());
    for (int row : sorted_query_indices)
    {
        if (mult_indices.empty() || mult_indices.back() != row)
        {
            mult_indices.push_back(row);
            mult_counts.push_back(1);
        }
        else
        {
            ++mult_counts.back();
        }
    }
    if (padding_count > 0)
    {
        mult_indices.push_back(dummy_index);
        mult_counts.push_back(padding_count);
    }
    int m_num = mult_indices.size();

    auto t0 = std::chrono::high_resolution_clock::now();

    const auto &prefetched = prefetched_prover_rows_;
    const bool prefetched_same_order = prefetched.indices == mult_indices;
    std::vector<std::size_t> prefetched_slots;
    if (!prefetched_same_order)
    {
        prefetched_slots.reserve(mult_indices.size());
        for (int row : mult_indices)
        {
            const auto found = std::lower_bound(prefetched.indices.begin(), prefetched.indices.end(), row);
            if (found == prefetched.indices.end() || *found != row)
                throw std::logic_error("ZkEmbed: preloaded rows do not cover this query batch");
            prefetched_slots.push_back(static_cast<std::size_t>(found - prefetched.indices.begin()));
        }
    }
    const auto prefetched_slot = [&](std::size_t i) {
        return prefetched_same_order ? i : prefetched_slots[i];
    };

    append_zkembed_statement(
        transcript, vocab_size, embed_dim, table_domain_size,
        query_indices, query_size, query_domain, col_domain,
        com2_vanishing, com2_idx, com2_columns,
        statement.h0);

    const Fr r = transcript.challenge();
    if (debug_trace)
        debug_trace->column_folding_challenge = r;

    std::vector<G1> m_points(m_num);
    std::vector<Fr> m_scalars(m_num);

#pragma omp parallel for
    for (int i = 0; i < m_num; i++)
    {
        m_points[i] = prefetched.li_x[prefetched_slot(i)];
        m_scalars[i] = Fr(mult_counts[i]);
    }
    G1::mulVecMT(proof.multiplicity_commitment, m_points.data(), m_scalars.data(), m_num, num_threads);

    auto t1 = std::chrono::high_resolution_clock::now();
    timing.commit_m_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    transcript.append_g1(proof.multiplicity_commitment);
    const Fr beta = transcript.challenge();
    if (debug_trace)
        debug_trace->beta = beta;

    auto t2 = std::chrono::high_resolution_clock::now();

    std::vector<Fr> r_powers(d);
    if (d > 0)
    {
        r_powers[0] = r;
        for (int j = 1; j < d; j++)
        {
            Fr::mul(r_powers[j], r_powers[j - 1], r);
        }
    }

    const std::size_t denominator_count = static_cast<std::size_t>(m_num) + query_size +
                                          (padding_count > 0 ? 1 : 0);
    std::vector<Fr> denominators(denominator_count);
    std::vector<Fr> f_star_evals(query_domain, Fr(0));
    std::vector<Fr> b_star_evals(query_domain, Fr(0));

#pragma omp parallel for schedule(static) num_threads(num_threads) if (num_threads > 1 && static_cast<std::size_t>(m_num) * d >= 4096)
    for (int i = 0; i < m_num; ++i)
    {
        const int row_idx = mult_indices[i];
        Fr t_star_val(row_idx);
        if (row_idx < vocab_size)
        {
            const ll *row = embedding_matrix_copy + static_cast<std::size_t>(row_idx) * d;
            for (int j = 0; j < d; ++j)
                t_star_val += r_powers[j] * Fr(row[j]);
        }
        denominators[i] = t_star_val + beta;
    }
#pragma omp parallel for schedule(static) num_threads(num_threads) if (num_threads > 1 && static_cast<std::size_t>(query_size) * d >= 4096)
    for (int k = 0; k < query_size; ++k)
    {
        Fr f_star_val(query_indices[k]);
        const Fr *row = h0_values.data() + static_cast<std::size_t>(k) * col_domain;
        for (int j = 0; j < d; ++j)
            f_star_val += r_powers[j] * row[j];
        f_star_evals[k] = f_star_val;
        denominators[static_cast<std::size_t>(m_num) + k] = f_star_val + beta;
    }
    if (padding_count > 0) denominators.back() = Fr(dummy_index) + beta;
    batch_invert(denominators);

    std::vector<Fr> a_values(m_num);
    Fr sum_a_star(0);
    for (int i = 0; i < m_num; ++i)
    {
        a_values[i] = Fr(mult_counts[i]) * denominators[i];
        sum_a_star += a_values[i];
    }
    proof.table_reciprocal_at_zero = sum_a_star / Fr(N);
    Fr grand_sum(0);
    for (int k = 0; k < query_size; ++k)
    {
        b_star_evals[k] = denominators[static_cast<std::size_t>(m_num) + k];
        grand_sum += b_star_evals[k];
    }
    if (padding_count > 0)
    {
        const Fr dummy_value(dummy_index), dummy_inv = denominators.back();
        for (int k = query_size; k < query_domain; ++k)
        {
            f_star_evals[k] = dummy_value;
            b_star_evals[k] = dummy_inv;
        }
        grand_sum += Fr(padding_count) * dummy_inv;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    timing.compute_rlc_evals_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    std::vector<G1> a_points_li_0(m_num);
#pragma omp parallel for
    for (int i = 0; i < m_num; ++i)
        a_points_li_0[i] = prefetched.li_0[prefetched_slot(i)];
    G1::mulVecMT(proof.table_reciprocal_commitment, m_points.data(), a_values.data(), m_num, num_threads);
    G1::mulVecMT(proof.table_zero_opening_commitment, a_points_li_0.data(), a_values.data(), m_num, num_threads);

    auto t4 = std::chrono::high_resolution_clock::now();
    timing.commit_a_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

    int total_points = m_num * d + m_num;
    std::vector<G1> all_qi_points(total_points);
    std::vector<Fr> all_scalars(total_points);

#pragma omp parallel for collapse(2)
    for (int j = 0; j < d; j++)
    {
        for (int i = 0; i < m_num; i++)
        {
            int idx = j * m_num + i;
            all_qi_points[idx] = prefetched.qi_columns[j][prefetched_slot(i)];
            Fr::mul(all_scalars[idx], r_powers[j], a_values[i]);
        }
    }
    int idx_base = m_num * d;
#pragma omp parallel for
    for (int i = 0; i < m_num; i++)
    {
        all_qi_points[idx_base + i] = prefetched.qi_idx[prefetched_slot(i)];
        all_scalars[idx_base + i] = a_values[i];
    }
    G1::mulVecMT(proof.table_quotient_commitment, all_qi_points.data(), all_scalars.data(), total_points, num_threads);

    auto t5 = std::chrono::high_resolution_clock::now();
    timing.commit_qa_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();

    G1 *reciprocal_commitments = hyrax::commit(
        b_star_evals.data(), mv_pk_.srs_g1.data(), query_l, num_threads);
    const int reciprocal_commitment_count = 1 << (query_l / 2);
    proof.query_reciprocal_commitments.assign(
        reciprocal_commitments,
        reciprocal_commitments + reciprocal_commitment_count);
    delete[] reciprocal_commitments;

    auto t6 = std::chrono::high_resolution_clock::now();
    timing.commit_b_hyrax_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();
    timing.g2_rlc_ms = 0;

    auto t7 = std::chrono::high_resolution_clock::now();

    const int b_commitment_count = reciprocal_commitment_count;
    append_zkembed_zero_check_commitments(
        transcript, proof, grand_sum, b_commitment_count);

    const Fr alpha = transcript.challenge();
    if (debug_trace)
        debug_trace->batching_challenge = alpha;

    std::vector<Fr> zero_check_point(query_l);
    for (int i = 0; i < query_l; i++)
    {
        zero_check_point[i] = transcript.challenge();
    }
    if (debug_trace)
        debug_trace->zero_check_point = zero_check_point;

    std::vector<Fr> B_arr(b_star_evals);
    std::vector<Fr> f_arr(std::move(f_star_evals));
    std::vector<Fr> eq_arr = hyrax::equality_weights(zero_check_point.data(), query_l);
    const int b_row_variables = query_l / 2;
    std::vector<Fr> b_open_folded;
    if (b_row_variables == 0) b_open_folded = B_arr;

    std::vector<Fr> outer_sumcheck_challenges;
    outer_sumcheck_challenges.reserve(query_l);
    std::vector<Fr> inner_sumcheck_challenges;
    inner_sumcheck_challenges.reserve(col_l);
    int current_size = query_domain;

    for (int round = 0; round < query_l; round++)
    {
        int half_size = current_size / 2;

        Fr c0 = Fr(0), c1 = Fr(0), c2 = Fr(0), c3 = Fr(0);

        for (int j = 0; j < half_size; j++)
        {
            Fr B_0 = B_arr[2 * j];
            Fr B_1 = B_arr[2 * j + 1];
            Fr f_0 = f_arr[2 * j];
            Fr f_1 = f_arr[2 * j + 1];
            Fr eq_0 = eq_arr[2 * j];
            Fr eq_1 = eq_arr[2 * j + 1];

            Fr dB, df, deq;
            Fr::sub(dB, B_1, B_0);
            Fr::sub(df, f_1, f_0);
            Fr::sub(deq, eq_1, eq_0);

            Fr f0_beta;
            Fr::add(f0_beta, f_0, beta);

            Fr wf_c0, wf_c1, wf_c2;
            Fr::mul(wf_c0, B_0, f0_beta);
            Fr::sub(wf_c0, wf_c0, Fr(1));

            Fr t1, t2;
            Fr::mul(t1, B_0, df);
            Fr::mul(t2, dB, f0_beta);
            Fr::add(wf_c1, t1, t2);

            Fr::mul(wf_c2, dB, df);

            Fr a_eq0, a_deq;
            Fr::mul(a_eq0, alpha, eq_0);
            Fr::mul(a_deq, alpha, deq);

            Fr term0, term1, term2, term3;
            Fr::mul(term0, a_eq0, wf_c0);
            Fr::mul(t1, a_eq0, wf_c1);
            Fr::mul(t2, a_deq, wf_c0);
            Fr::add(term1, t1, t2);
            Fr::mul(t1, a_eq0, wf_c2);
            Fr::mul(t2, a_deq, wf_c1);
            Fr::add(term2, t1, t2);
            Fr::mul(term3, a_deq, wf_c2);

            Fr g_c0, g_c1;
            Fr::add(g_c0, B_0, term0);
            Fr::add(g_c1, dB, term1);

            Fr::add(c0, c0, g_c0);
            Fr::add(c1, c1, g_c1);
            Fr::add(c2, c2, term2);
            Fr::add(c3, c3, term3);
        }

        SumcheckRoundPoly round_poly;
        round_poly.coeffs[0] = c0;
        round_poly.coeffs[1] = c1;
        round_poly.coeffs[2] = c2;
        round_poly.coeffs[3] = c3;
        proof.outer_sumcheck_polys.push_back(round_poly);

        append_sumcheck_round(transcript, round, round_poly);
        Fr r_i = transcript.challenge_not_one();
        outer_sumcheck_challenges.push_back(r_i);

        Fr one_minus_r;
        Fr::sub(one_minus_r, Fr(1), r_i);

        for (int j = 0; j < half_size; j++)
        {
            Fr tmp;
            Fr::mul(B_arr[j], one_minus_r, B_arr[2 * j]);
            Fr::mul(tmp, r_i, B_arr[2 * j + 1]);
            Fr::add(B_arr[j], B_arr[j], tmp);

            Fr::mul(f_arr[j], one_minus_r, f_arr[2 * j]);
            Fr::mul(tmp, r_i, f_arr[2 * j + 1]);
            Fr::add(f_arr[j], f_arr[j], tmp);

            Fr::mul(eq_arr[j], one_minus_r, eq_arr[2 * j]);
            Fr::mul(tmp, r_i, eq_arr[2 * j + 1]);
            Fr::add(eq_arr[j], eq_arr[j], tmp);
        }

        if (round + 1 == b_row_variables)
            b_open_folded.assign(B_arr.begin(), B_arr.begin() + half_size);
        current_size = half_size;
    }

    proof.query_reciprocal_eval = B_arr[0];

    auto t8 = std::chrono::high_resolution_clock::now();
    long long outer_sumcheck_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t7).count();

    auto t9 = std::chrono::high_resolution_clock::now();

    proof.query_reciprocal_opening = hyrax::open(
        b_star_evals.data(), outer_sumcheck_challenges.data(), query_l, mv_pk_,
        hyrax::Layout::InterleavedLowBits, b_open_folded.data());
    auto t10 = std::chrono::high_resolution_clock::now();
    long long hyrax_b_open_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t10 - t9).count();

    const std::vector<Fr> eq_x = hyrax::equality_weights(
        outer_sumcheck_challenges.data(), query_l);
    std::vector<Fr> h_eval(h0_colnum, Fr(0));
    constexpr int column_tile = 64;
    const int tiles = (h0_colnum + column_tile - 1) / column_tile;
    const int fold_threads = std::min(num_threads, tiles);
#pragma omp parallel for schedule(static) num_threads(fold_threads) if (fold_threads > 1 && static_cast<std::size_t>(h0_rownum) * h0_colnum >= 4096)
    for (int tile = 0; tile < tiles; ++tile)
    {
        const int begin = tile * column_tile;
        const int end = std::min(h0_colnum, begin + column_tile);
        for (int x = 0; x < h0_rownum; ++x)
        {
            const Fr &weight = eq_x[x];
            const Fr *row = h0_values.data() + static_cast<std::size_t>(x) * h0_colnum;
            for (int y = begin; y < end; ++y)
                h_eval[y] += weight * row[y];
        }
    }

    std::vector<Fr> w_rho(h0_colnum, Fr(0));
    for (int j = 0; j < d; j++)
    {
        w_rho[j] = r_powers[j];
    }

    proof.folded_h0_claim = Fr(0);
    for (int y = 0; y < h0_colnum; y++)
    {
        Fr term;
        Fr::mul(term, w_rho[y], h_eval[y]);
        Fr::add(proof.folded_h0_claim, proof.folded_h0_claim, term);
    }

    transcript.append_fr(proof.query_reciprocal_eval);
    transcript.append_fr(proof.folded_h0_claim);

    auto inner_sumcheck_start = std::chrono::high_resolution_clock::now();

    std::vector<Fr> w_arr(std::move(w_rho));
    std::vector<Fr> h_arr(h_eval);
    int inner_size = h0_colnum;
    Fr inv2;
    Fr::inv(inv2, Fr(2));

    for (int round = 0; round < col_l; round++)
    {
        int half_size = inner_size / 2;
        Fr g0 = Fr(0);
        Fr g1 = Fr(0);
        Fr g2 = Fr(0);

        for (int j = 0; j < half_size; j++)
        {
            Fr w0 = w_arr[2 * j];
            Fr w1 = w_arr[2 * j + 1];
            Fr h0 = h_arr[2 * j];
            Fr h1 = h_arr[2 * j + 1];

            Fr dW, dH;
            Fr::sub(dW, w1, w0);
            Fr::sub(dH, h1, h0);

            Fr term0, term1, term2;
            Fr::mul(term0, w0, h0);
            Fr::mul(term1, w1, h1);

            Fr w2, h2;
            Fr::add(w2, w0, dW);
            Fr::add(w2, w2, dW);
            Fr::add(h2, h0, dH);
            Fr::add(h2, h2, dH);
            Fr::mul(term2, w2, h2);

            Fr::add(g0, g0, term0);
            Fr::add(g1, g1, term1);
            Fr::add(g2, g2, term2);
        }

        SumcheckRoundPoly round_poly;
        round_poly.coeffs[0] = g0;

        Fr tmp, c2;
        Fr::mul(tmp, Fr(2), g1);
        Fr::sub(c2, g2, tmp);
        Fr::add(c2, c2, g0);
        Fr::mul(round_poly.coeffs[2], c2, inv2);

        Fr c1;
        Fr::sub(c1, g1, g0);
        Fr::sub(round_poly.coeffs[1], c1, round_poly.coeffs[2]);
        round_poly.coeffs[3] = Fr(0);
        proof.inner_sumcheck_polys.push_back(round_poly);

        append_sumcheck_round(transcript, round, round_poly);
        Fr r_i = transcript.challenge_not_one();
        inner_sumcheck_challenges.push_back(r_i);

        Fr one_minus_r;
        Fr::sub(one_minus_r, Fr(1), r_i);
        for (int j = 0; j < half_size; j++)
        {
            Fr tmp2;
            Fr::mul(w_arr[j], one_minus_r, w_arr[2 * j]);
            Fr::mul(tmp2, r_i, w_arr[2 * j + 1]);
            Fr::add(w_arr[j], w_arr[j], tmp2);

            Fr::mul(h_arr[j], one_minus_r, h_arr[2 * j]);
            Fr::mul(tmp2, r_i, h_arr[2 * j + 1]);
            Fr::add(h_arr[j], h_arr[j], tmp2);
        }
        inner_size = half_size;
    }

    auto inner_sumcheck_end = std::chrono::high_resolution_clock::now();
    long long inner_sumcheck_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inner_sumcheck_end - inner_sumcheck_start).count();

    auto h0_open_start = std::chrono::high_resolution_clock::now();
    proof.h0_eval = h_arr[0];
    transcript.append_fr(proof.h0_eval);
    if (!statement.h0.isInputView())
    {
        const std::vector<Fr> point = tensor_point(
            outer_sumcheck_challenges, inner_sumcheck_challenges, expected_h0_l);
        const int pcs_col_vars = expected_h0_l - expected_h0_l / 2;
        std::vector<Fr> folded;
        const Fr *precomputed_h0 = nullptr;
        if (pcs_col_vars == col_l)
            precomputed_h0 = h_eval.data();
        else if (pcs_col_vars < col_l)
        {
            const int columns = 1 << pcs_col_vars;
            folded.assign(static_cast<std::size_t>(columns), Fr(0));
            const auto weights = hyrax::equality_weights(
                col_l == 0 ? nullptr : inner_sumcheck_challenges.data() + pcs_col_vars,
                col_l - pcs_col_vars);
            for (std::size_t row = 0; row < weights.size(); ++row)
            {
                for (int col = 0; col < columns; ++col)
                {
                    folded[col] += weights[row] * h_eval[row * columns + col];
                }
            }
        }
        if (!folded.empty()) precomputed_h0 = folded.data();
        proof.h0_opening = hyrax::open(
            h0_values.data(), point.data(), expected_h0_l, mv_pk_,
            hyrax::Layout::ContiguousRows, precomputed_h0);
    }
    opening_weights(proof, statement.h0.isInputView(), transcript);
    auto h0_open_end = std::chrono::high_resolution_clock::now();

    timing.sumcheck_ms = outer_sumcheck_ms + inner_sumcheck_ms;
    timing.hyrax_open_ms = hyrax_b_open_ms +
                           std::chrono::duration_cast<std::chrono::milliseconds>(h0_open_end - h0_open_start).count();

    if (debug_trace)
    {
        debug_trace->outer_sumcheck_challenges = outer_sumcheck_challenges;
        debug_trace->inner_sumcheck_challenges = inner_sumcheck_challenges;
    }

    auto t11 = std::chrono::high_resolution_clock::now();
    timing.total_prove_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t11 - total_start).count();
    long long compute_ms = timing.total_prove_ms - timing.lazy_load_ms;
    timing.prove_compute_ms = (compute_ms > 0) ? compute_ms : 0;
    return proof;
}

bool ZkEmbed::verify(const ZkEmbedStatement &statement,
                     const ZkEmbedProof &proof,
                     ZkEmbedTiming &timing,
                     ZkEmbedDebugTrace *debug_trace)
{
    FiatShamirTranscript transcript;
    return verify(statement, proof, transcript, timing, debug_trace);
}

bool ZkEmbed::verify(const ZkEmbedStatement &statement,
                     const ZkEmbedProof &proof,
                     FiatShamirTranscript &transcript,
                     ZkEmbedTiming &timing,
                     ZkEmbedDebugTrace *debug_trace)
{
    if (statement.h0.isInputView())
        return false;
    try
    {
        return verifyImpl(statement, proof, transcript, timing, debug_trace);
    }
    catch (const std::exception &)
    {
        return false;
    }
}

std::unique_ptr<ZkEmbedOpeningClaim> ZkEmbed::reduceToInput(
    const ZkEmbedStatement &statement, const ZkEmbedProof &proof,
    FiatShamirTranscript &transcript, ZkEmbedTiming &timing,
    ZkEmbedDebugTrace *debug_trace)
{
    if (!statement.h0.isInputView())
        return nullptr;
    std::unique_ptr<ZkEmbedOpeningClaim> claim(new ZkEmbedOpeningClaim);
    try
    {
        if (!verifyImpl(statement, proof, transcript, timing, debug_trace, claim.get()))
            return nullptr;
    }
    catch (const std::exception &)
    {
        return nullptr;
    }
    return claim;
}

bool ZkEmbed::verifyImpl(const ZkEmbedStatement &statement,
                         const ZkEmbedProof &proof,
                         FiatShamirTranscript &transcript,
                         ZkEmbedTiming &timing,
                         ZkEmbedDebugTrace *debug_trace,
                         ZkEmbedOpeningClaim *pending_claim)
{
    const ScopedOmpThreads thread_scope(thread_num);
    if (!validOutput(statement) ||
        statement.h0.isInputView() != (pending_claim != nullptr) ||
        (pending_claim && !proof.h0_opening.eval_proof.q_comm.empty()))
        return false;
    auto total_start = std::chrono::high_resolution_clock::now();
    int d = embed_dim;
    const ll *query_indices = statement.query_indices.data();
    const int query_size = static_cast<int>(statement.query_indices.size());

    if (query_size <= 0)
    {
        std::cout << "  [FAIL] Invalid query input" << std::endl;
        return false;
    }
    int expected_query_domain = 1;
    int expected_mu = 0;
    while (expected_query_domain < query_size)
    {
        expected_query_domain <<= 1;
        expected_mu++;
    }
    const int mu = expected_mu;
    if (static_cast<int>(proof.outer_sumcheck_polys.size()) != mu)
    {
        std::cout << "  [FAIL] Query-domain metadata mismatch" << std::endl;
        return false;
    }
    for (int i = 0; i < query_size; i++)
    {
        if (query_indices[i] < 0 || query_indices[i] >= vocab_size)
        {
            std::cout << "  [FAIL] Query index outside vocabulary" << std::endl;
            return false;
        }
    }

    ensure_com2_loaded();
    if (!com2_loaded)
    {
        std::cout << "  [FAIL] Verifier cache load failed" << std::endl;
        return false;
    }

    int col_domain = 1;
    int col_l = 0;
    while (col_domain < embed_dim)
    {
        col_domain <<= 1;
        col_l++;
    }
    const int expected_h0_l = expected_mu + col_l;
    const int b_commitment_count = 1 << (expected_mu / 2);
    if (proof.query_reciprocal_commitments.size() !=
        static_cast<std::size_t>(b_commitment_count))
        return false;

    append_zkembed_statement(
        transcript, vocab_size, embed_dim, table_domain_size,
        query_indices, query_size, expected_query_domain, col_domain,
        com2_vanishing, com2_idx, com2_columns,
        statement.h0);

    const Fr column_folding_challenge = transcript.challenge();
    if (debug_trace)
    {
        debug_trace->column_folding_challenge = column_folding_challenge;
    }

    transcript.append_g1(proof.multiplicity_commitment);
    const Fr beta = transcript.challenge();
    if (debug_trace)
        debug_trace->beta = beta;

    const Fr grand_sum =
        proof.table_reciprocal_at_zero * Fr(table_domain_size);

    append_zkembed_zero_check_commitments(
        transcript, proof, grand_sum, b_commitment_count);
    const Fr alpha = transcript.challenge();
    if (debug_trace)
        debug_trace->batching_challenge = alpha;

    std::vector<Fr> zero_check_point(expected_mu);
    for (int i = 0; i < expected_mu; i++)
    {
        zero_check_point[i] = transcript.challenge();
    }
    if (debug_trace)
        debug_trace->zero_check_point = zero_check_point;

    const G1 &g1_gen = uni_vk_.g1_generator;
    const G2 &g2_gen = uni_vk_.g2_generator;

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<Fr> r_powers(d);
    if (d > 0)
    {
        r_powers[0] = column_folding_challenge;
        for (int j = 1; j < d; j++)
        {
            Fr::mul(r_powers[j], r_powers[j - 1], column_folding_challenge);
        }
    }

    G2 combined_columns;
    const int g2_msm_threads = std::min(thread_num, d / 64);
    if (g2_msm_threads > 1)
    {
        const ScopedOmpThreads msm_thread_scope(g2_msm_threads);
        G2::mulVecMT(combined_columns, com2_columns.data(), r_powers.data(), d,
                     g2_msm_threads);
    }
    else
        G2::mulVec(combined_columns, com2_columns.data(), r_powers.data(), d);

    G2 computed_t_star;
    G2::add(computed_t_star, com2_idx, combined_columns);

    auto t1 = std::chrono::high_resolution_clock::now();
    timing.verify_g2_rlc_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    G2 beta_g2;
    G2::mul(beta_g2, g2_gen, beta);
    G2 t_star_plus_beta;
    G2::add(t_star_plus_beta, computed_t_star, beta_g2);

    G1 neg_m, neg_qa_star;
    G1::neg(neg_m, proof.multiplicity_commitment);
    G1::neg(neg_qa_star, proof.table_quotient_commitment);

    GT check1_result;
    G1 g1_pts[3] = {proof.table_reciprocal_commitment, neg_m, neg_qa_star};
    G2 g2_pts[3] = {t_star_plus_beta, g2_gen, com2_vanishing};
    millerLoopVec(check1_result, g1_pts, g2_pts, 3);
    finalExp(check1_result, check1_result);

    auto t2 = std::chrono::high_resolution_clock::now();
    timing.verify_a_wellform_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    if (!check1_result.isOne())
    {
        std::cout << "  [FAIL] Check 1: A* wellformedness" << std::endl;
        return false;
    }

    G1 a_star_minus_a0;
    G1 a0_g1;
    G1::mul(a0_g1, g1_gen, proof.table_reciprocal_at_zero);
    G1::sub(a_star_minus_a0, proof.table_reciprocal_commitment, a0_g1);

    G1 neg_zero_opening;
    G1::neg(neg_zero_opening, proof.table_zero_opening_commitment);
    G1 zero_g1_pts[2] = {a_star_minus_a0, neg_zero_opening};
    G2 zero_g2_pts[2] = {g2_gen, uni_vk_.g2_tau};
    GT check_zero_result;
    millerLoopVec(check_zero_result, zero_g1_pts, zero_g2_pts, 2);
    finalExp(check_zero_result, check_zero_result);

    auto t3 = std::chrono::high_resolution_clock::now();
    timing.verify_a0_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    if (!check_zero_result.isOne())
    {
        std::cout << "  [FAIL] Check 2: A*(0) consistency" << std::endl;
        return false;
    }

    Fr claimed_sum = grand_sum;
    std::vector<Fr> outer_sumcheck_challenges;
    outer_sumcheck_challenges.reserve(mu);

    for (int round = 0; round < mu; round++)
    {
        const SumcheckRoundPoly &g_i = proof.outer_sumcheck_polys[round];

        Fr g_0 = g_i.coeffs[0];
        Fr g_1;
        Fr::add(g_1, g_i.coeffs[0], g_i.coeffs[1]);
        Fr::add(g_1, g_1, g_i.coeffs[2]);
        Fr::add(g_1, g_1, g_i.coeffs[3]);

        Fr sum_check;
        Fr::add(sum_check, g_0, g_1);

        if (sum_check != claimed_sum)
        {
            std::cout << "  [FAIL] Check 3: Sumcheck round " << round << " failed" << std::endl;
            return false;
        }

        append_sumcheck_round(transcript, round, g_i);
        const Fr r_i = transcript.challenge_not_one();
        outer_sumcheck_challenges.push_back(r_i);

        claimed_sum = g_i.evaluate(r_i);
    }

    Fr eq_rho_r = Fr(1);
    for (int i = 0; i < mu; i++)
    {
        Fr rho_r, eq_i;
        Fr::mul(rho_r, zero_check_point[i], outer_sumcheck_challenges[i]);
        Fr::mul(eq_i, rho_r, Fr(2));
        Fr::sub(eq_i, eq_i, zero_check_point[i]);
        Fr::sub(eq_i, eq_i, outer_sumcheck_challenges[i]);
        Fr::add(eq_i, eq_i, Fr(1));
        Fr::mul(eq_rho_r, eq_rho_r, eq_i);
    }

    int query_domain = expected_query_domain;

    Fr inner_claim = proof.folded_h0_claim;
    if ((int)proof.inner_sumcheck_polys.size() != col_l)
    {
        std::cout << "  [FAIL] Check 4: Inner sumcheck rounds mismatch" << std::endl;
        return false;
    }
    std::vector<Fr> inner_sumcheck_challenges;
    inner_sumcheck_challenges.reserve(col_l);

    transcript.append_fr(proof.query_reciprocal_eval);
    transcript.append_fr(proof.folded_h0_claim);

    for (int round = 0; round < col_l; round++)
    {
        const SumcheckRoundPoly &g_i = proof.inner_sumcheck_polys[round];
        if (g_i.coeffs[3] != Fr(0))
            return false;
        Fr g_0 = g_i.coeffs[0];
        Fr g_1;
        Fr::add(g_1, g_i.coeffs[0], g_i.coeffs[1]);
        Fr::add(g_1, g_1, g_i.coeffs[2]);

        Fr sum_check;
        Fr::add(sum_check, g_0, g_1);
        if (sum_check != inner_claim)
        {
            std::cout << "  [FAIL] Check 4: Inner sumcheck round " << round << " failed" << std::endl;
            return false;
        }

        append_sumcheck_round(transcript, round, g_i);
        const Fr r_i = transcript.challenge_not_one();
        inner_sumcheck_challenges.push_back(r_i);

        inner_claim = g_i.evaluate(r_i);
    }

    const std::vector<Fr> eq_x = hyrax::equality_weights(
        outer_sumcheck_challenges.data(), mu);

    Fr t_eval = Fr(0);
    for (int i = 0; i < query_size; i++)
    {
        Fr term;
        Fr::mul(term, eq_x[i], Fr(query_indices[i]));
        Fr::add(t_eval, t_eval, term);
    }
    for (int i = query_size; i < query_domain; i++)
    {
        Fr term;
        Fr::mul(term, eq_x[i], Fr(vocab_size));
        Fr::add(t_eval, t_eval, term);
    }

    const Fr w_final = geometric_prefix_mle(
        column_folding_challenge, inner_sumcheck_challenges, d);

    transcript.append_fr(proof.h0_eval);
    Fr expected_inner;
    Fr::mul(expected_inner, w_final, proof.h0_eval);
    if (inner_claim != expected_inner)
    {
        std::cout << "  [FAIL] Check 4: Inner sumcheck final claim mismatch" << std::endl;
        return false;
    }

    Fr a_rho_final;
    Fr::add(a_rho_final, t_eval, proof.folded_h0_claim);

    Fr f_plus_beta;
    Fr::add(f_plus_beta, a_rho_final, beta);
    Fr wellform;
    Fr::mul(wellform, proof.query_reciprocal_eval, f_plus_beta);
    Fr::sub(wellform, wellform, Fr(1));

    Fr alpha_eq_wf;
    Fr::mul(alpha_eq_wf, alpha, eq_rho_r);
    Fr::mul(alpha_eq_wf, alpha_eq_wf, wellform);
    Fr expected_final;
    Fr::add(expected_final, proof.query_reciprocal_eval, alpha_eq_wf);

    if (claimed_sum != expected_final)
    {
        std::cout << "  [FAIL] Check 3: Final sumcheck claim mismatch" << std::endl;
        return false;
    }

    if (debug_trace)
    {
        debug_trace->outer_sumcheck_challenges = outer_sumcheck_challenges;
        debug_trace->inner_sumcheck_challenges = inner_sumcheck_challenges;
    }

    const std::vector<Fr> point = tensor_point(
        outer_sumcheck_challenges, inner_sumcheck_challenges, statement.h0.variables());
    hyrax::OpeningBatchVerifier batch;
    if (!batch.append(proof.query_reciprocal_opening,
                      outer_sumcheck_challenges.data(), proof.query_reciprocal_eval,
                      proof.query_reciprocal_commitments.data(), mu))
        return false;
    if (!pending_claim && !batch.append(
                              proof.h0_opening, point.data(), proof.h0_eval, statement.h0.rows().data(),
                              expected_h0_l, hyrax::Layout::ContiguousRows))
        return false;
    const auto weights = opening_weights(proof, statement.h0.isInputView(), transcript);
    if (!batch.verify_with_weights(mv_vk_, weights))
        return false;
    if (pending_claim)
    {
        pending_claim->point = point;
        pending_claim->evaluation = proof.h0_eval;
    }

    auto t5 = std::chrono::high_resolution_clock::now();
    timing.verify_sumcheck_us = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t3).count();

    timing.total_verify_us = std::chrono::duration_cast<std::chrono::microseconds>(t5 - total_start).count();
    return true;
}

// Embedding and absolute-position binding to the GKR input.
namespace
{
    int bits(std::size_t n)
    {
        if (n == 0 || n > (std::size_t(1) << 30))
            throw std::invalid_argument("APE dimension is outside the supported domain");
        return zkagent_integer::ceil_log2(n);
    }

    void validate_shape(const ApeShape &shape)
    {
        if (shape.padded_tokens == 0 || (shape.padded_tokens & (shape.padded_tokens - 1)) != 0)
        {
            throw std::invalid_argument("APE token capacity must be a power of two");
        }
        const int prefix = shape.tokenVariables() + shape.columnVariables();
        if (prefix > 30 || shape.input_variables < prefix || shape.input_variables > 30)
        {
            throw std::invalid_argument("APE prefix does not fit the GKR input domain");
        }
    }

    std::vector<Fr> position_row_weights(const ApeShape &shape, const std::vector<Fr> &point)
    {
        const int prefix_bits = shape.prefixVariables();
        if (point.size() != static_cast<std::size_t>(shape.input_variables))
            throw std::invalid_argument("APE opening point has the wrong domain");
        for (int bit = prefix_bits; bit < shape.input_variables; ++bit)
        {
            if (point[bit] != Fr(0))
                throw std::invalid_argument("APE opening is not an aligned prefix");
        }
        const int column_bits = shape.input_variables - shape.input_variables / 2;
        const int row_bits = std::max(0, prefix_bits - column_bits);
        return hyrax::equality_weights(row_bits ? point.data() + column_bits : nullptr, row_bits);
    }

    std::vector<Fr> fold_position_prefix(const ApeProverKey &key,
                                         const std::vector<Fr> &point, int threads)
    {
        const auto weights = position_row_weights(key.shape, point);
        if (key.position_prefix.size() != (std::size_t(1) << key.shape.prefixVariables()) || threads < 1)
            throw std::invalid_argument("APE witness layout or thread count mismatch");
        const int column_bits = key.shape.input_variables - key.shape.input_variables / 2;
        const std::size_t columns = std::size_t(1) << column_bits;
        std::vector<Fr> folded(columns, Fr(0));
        if (weights.size() == 1)
        {
            std::copy(key.position_prefix.begin(), key.position_prefix.end(), folded.begin());
            return folded;
        }
        // Disjoint column blocks preserve each column's row addition order.
        const std::size_t block_size = 256;
        const int blocks = static_cast<int>((columns + block_size - 1) / block_size);
        const int workers = std::min(threads, blocks);
#pragma omp parallel for schedule(static) num_threads(workers) if(workers > 1 && key.position_prefix.size() >= 4096)
        for (int block = 0; block < blocks; ++block)
        {
            const std::size_t begin = static_cast<std::size_t>(block) * block_size;
            const std::size_t end = std::min(columns, begin + block_size);
            for (std::size_t row = 0; row < weights.size(); ++row)
            {
                for (std::size_t col = begin; col < end; ++col)
                    folded[col] += weights[row] * key.position_prefix[row * columns + col];
            }
        }
        return folded;
    }

    bool verify_position_opening(const ApeVerifierKey &key, const ZkEmbedInputProof &proof,
                                 const std::vector<Fr> &point, const MVKZG_VerifierKey &vk,
                                 int threads)
    {
        const auto weights = position_row_weights(key.shape(), point);
        if (key.prefixCommitments().size() != weights.size())
            return false;
        G1 folded_commitment;
        G1::mulVecMT(folded_commitment, const_cast<G1 *>(key.prefixCommitments().data()),
                     weights.data(), weights.size(), std::max(1, threads));
        const int column_bits = key.shape().input_variables - key.shape().input_variables / 2;
        return mvkzg_eval_verify(point.data(), column_bits, folded_commitment,
                                proof.position_evaluation, proof.position_opening.eval_proof, vk);
    }

    void append_shape(FiatShamirTranscript &transcript, const ApeShape &shape)
    {
        transcript.append_u64(shape.padded_tokens);
        transcript.append_u64(shape.hidden);
        transcript.append_u64(shape.input_variables);
    }

    InputReductionLink finish_claim(const ApeShape &shape, const ZkEmbedOpeningClaim &h0,
                                    const ZkEmbedInputProof &proof,
                                    FiatShamirTranscript &transcript)
    {
        const std::size_t prefix_bits = static_cast<std::size_t>(shape.prefixVariables());
        if (h0.point.size() != static_cast<std::size_t>(shape.input_variables))
        {
            throw std::invalid_argument("embedding terminal claim has the wrong domain");
        }
        for (std::size_t i = prefix_bits; i < h0.point.size(); ++i)
        {
            if (h0.point[i] != Fr(0))
                throw std::invalid_argument("embedding claim is not an aligned prefix");
        }
        transcript.append_fr(proof.position_evaluation);
        transcript.append_g1_vector(proof.position_opening.eval_proof.q_comm,
                                    G1WireEncoding::Compressed);
        const Fr weight = transcript.nonzero_challenge();
        InputReductionLink claim;
        claim.input_domain_size = std::size_t(1) << shape.input_variables;
        claim.addPrefixEvaluation(
            std::vector<Fr>(h0.point.begin(), h0.point.begin() + prefix_bits),
            h0.evaluation + proof.position_evaluation, weight);
        return claim;
    }
}

int ApeShape::tokenVariables() const { return bits(padded_tokens); }
int ApeShape::columnVariables() const { return bits(hidden); }
int ApeShape::prefixVariables() const
{
    validate_shape(*this);
    return tokenVariables() + columnVariables();
}
std::size_t ApeShape::prefixRows() const
{
    return std::size_t(1) << std::max(0, prefixVariables() - (input_variables - input_variables / 2));
}
bool ApeShape::operator==(const ApeShape &other) const
{
    return padded_tokens == other.padded_tokens && hidden == other.hidden &&
           input_variables == other.input_variables;
}

ApeProverKey prepareApeProverKey(
    const ApeShape &shape, const std::vector<std::int32_t> &positions)
{
    validate_shape(shape);
    if (positions.size() % shape.hidden != 0 || positions.size() / shape.hidden < shape.padded_tokens)
    {
        throw std::invalid_argument("position table must cover the entire padded token capacity");
    }
    ApeProverKey result;
    result.shape = shape;
    result.position_prefix.assign(std::size_t(1) << shape.prefixVariables(), Fr(0));
    const std::size_t columns = std::size_t(1) << shape.columnVariables();
    for (std::size_t row = 0; row < shape.padded_tokens; ++row)
    {
        for (std::size_t col = 0; col < shape.hidden; ++col)
        {
            result.position_prefix[row * columns + col] = Fr(positions[row * shape.hidden + col]);
        }
    }
    return result;
}

ApeVerifierKey prepareApeVerifierKey(
    const ApeProverKey &source, const MVKZG_ProverKey &pk, int threads)
{
    const ScopedOmpThreads thread_scope(std::max(1, threads));
    validate_shape(source.shape);
    const std::size_t columns = std::size_t(1) << (source.shape.input_variables - source.shape.input_variables / 2);
    if (source.position_prefix.size() != (std::size_t(1) << source.shape.prefixVariables()) ||
        pk.srs_g1.size() < columns)
        throw std::invalid_argument("APE prover key layout mismatch");
    ApeVerifierKey key;
    key.shape_ = source.shape;
    key.prefix_commitments_.resize(source.shape.prefixRows());
    for (std::size_t row = 0; row < key.prefix_commitments_.size(); ++row)
    {
        const std::size_t start = row * columns;
        const std::size_t count = std::min(columns, source.position_prefix.size() - start);
        G1::mulVecMT(key.prefix_commitments_[row], const_cast<G1 *>(pk.srs_g1.data()),
                     source.position_prefix.data() + start, count, std::max(1, threads));
    }
    return key;
}

void ApeVerifierKey::saveToFile(const std::string &path) const
{
    validate_shape(shape_);
    if (prefix_commitments_.size() != shape_.prefixRows())
        throw std::invalid_argument("invalid APE key");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write("APEC", 4);
    cache_io::write_u64(out, shape_.padded_tokens);
    cache_io::write_u64(out, shape_.hidden);
    cache_io::write_u64(out, shape_.input_variables);
    cache_io::write_u64(out, prefix_commitments_.size());
    for (const auto &point : prefix_commitments_)
        cache_io::write_point<G1, 96>(out, point);
    if (!out)
        throw std::runtime_error("could not save APE verifier key: " + path);
}

ApeVerifierKey ApeVerifierKey::loadFromFile(const std::string &path, const ApeShape &expected)
{
    validate_shape(expected);
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != "APEC")
        throw std::runtime_error("invalid APE key header");
    const auto tokens = cache_io::read_u64(in), hidden = cache_io::read_u64(in), input_vars = cache_io::read_u64(in);
    if (tokens != expected.padded_tokens || hidden != expected.hidden || input_vars != static_cast<unsigned>(expected.input_variables))
    {
        throw std::runtime_error("APE key does not match token capacity/dimension/input layout");
    }
    ApeVerifierKey key;
    key.shape_ = expected;
    if (cache_io::read_u64(in) != expected.prefixRows())
        throw std::runtime_error("invalid APE row count");
    key.prefix_commitments_.resize(expected.prefixRows());
    for (auto &point : key.prefix_commitments_)
        point = cache_io::read_point<G1, 96>(in);
    if (in.peek() != EOF)
        throw std::runtime_error("trailing APE key data");
    return key;
}

std::size_t ZkEmbedInputProof::cryptographicPayloadBytes() const
{
    return embedding.cryptographicPayloadBytes() + 32 +
           position_opening.eval_proof.q_comm.size() * G1::getSerializedByteSize();
}

ZkEmbedInputProof proveZkEmbedInputLink(
    ZkEmbed &embedding, const std::vector<ll> &queries, const std::vector<Fr> &h0_values,
    const std::vector<G1> &input_rows, const ApeProverKey &ape_pk, const ApeVerifierKey &ape_vk,
    const MVKZG_ProverKey &pk, FiatShamirTranscript &transcript, InputReductionLink &pending)
{
    const ScopedOmpThreads thread_scope(embedding.threadCount());
    if (!(ape_pk.shape == ape_vk.shape()) || (std::size_t(1) << bits(queries.size())) != ape_vk.shape().padded_tokens ||
        static_cast<std::size_t>(embedding.embeddingDimension()) != ape_vk.shape().hidden)
    {
        throw std::invalid_argument("embedding/APE statement mismatch");
    }
    const auto &shape = ape_vk.shape();
    append_shape(transcript, shape);
    ZkEmbedStatement statement{queries, ZkEmbedOutputCommitment::inputPrefix(
                                            shape.input_variables, input_rows, ape_vk.prefixCommitments())};
    ZkEmbedDebugTrace trace;
    ZkEmbedTiming timing;
    ZkEmbedInputProof proof;
    proof.embedding = embedding.proveToInput(statement, h0_values, transcript, timing, &trace);
    ZkEmbedOpeningClaim h0;
    h0.point = trace.inner_sumcheck_challenges;
    h0.point.insert(h0.point.end(), trace.outer_sumcheck_challenges.begin(), trace.outer_sumcheck_challenges.end());
    h0.point.resize(shape.input_variables, Fr(0));
    h0.evaluation = proof.embedding.h0_eval;
    const int column_bits = shape.input_variables - shape.input_variables / 2;
    const auto folded = fold_position_prefix(ape_pk, h0.point, embedding.threadCount());
    proof.position_evaluation = hyrax::evaluate(
        folded.data(), h0.point.data(), std::min(shape.prefixVariables(), column_bits));
    proof.position_opening = hyrax::open_folded(
        folded, h0.point.data(), shape.input_variables, pk, hyrax::Layout::ContiguousRows);
    pending = finish_claim(shape, h0, proof, transcript);
    return proof;
}

std::unique_ptr<InputReductionLink> verifyZkEmbedInputLink(
    ZkEmbed &embedding, const std::vector<ll> &queries, const std::vector<G1> &input_rows,
    const ApeVerifierKey &ape_vk, const ZkEmbedInputProof &proof,
    const MVKZG_VerifierKey &vk, FiatShamirTranscript &transcript)
{
    const ScopedOmpThreads thread_scope(embedding.threadCount());
    try
    {
        const auto &shape = ape_vk.shape();
        validate_shape(shape);
        if ((std::size_t(1) << bits(queries.size())) != shape.padded_tokens ||
            static_cast<std::size_t>(embedding.embeddingDimension()) != shape.hidden ||
            ape_vk.prefixCommitments().size() != shape.prefixRows())
            return nullptr;
        append_shape(transcript, shape);
        ZkEmbedStatement statement{queries, ZkEmbedOutputCommitment::inputPrefix(
                                                shape.input_variables, input_rows, ape_vk.prefixCommitments())};
        ZkEmbedTiming timing;
        auto h0 = embedding.reduceToInput(statement, proof.embedding, transcript, timing);
        if (!h0 || !verify_position_opening(ape_vk, proof, h0->point, vk, embedding.threadCount()))
            return nullptr;
        return std::unique_ptr<InputReductionLink>(new InputReductionLink(finish_claim(shape, *h0, proof, transcript)));
    }
    catch (const std::exception &)
    {
        return nullptr;
    }
}
