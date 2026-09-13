#include "pcs/hyrax.hpp"
#include "common/field_integer.hpp"
#include "common/mle.hpp"
#include "pcs/mvkzg.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

#include <omp.h>
using namespace std;

using namespace mcl::bn;

namespace hyrax
{

    namespace
    {
        using zkagent_integer::fr_from_i128;
    }

    namespace
    {

        G1 linear_commit(const G1 *generators, const Fr *scalars, int count)
        {
            G1 ret;
            G1::mulVecMT(ret, const_cast<G1 *>(generators),
                         const_cast<Fr *>(scalars), count);
            return ret;
        }

        template <class CommitRow>
        void commit_independent_rows(std::size_t rows, std::size_t columns,
                                     G1 *generators, int requested_threads,
                                     const CommitRow &commit_row)
        {
            const int threads = std::max(1, requested_threads);
            if (threads > 1 && rows >= static_cast<std::size_t>(threads))
            {
#pragma omp parallel num_threads(threads)
                {
                    std::vector<G1> bases(generators, generators + columns);
#pragma omp for schedule(static)
                    for (std::size_t row = 0; row < rows; ++row)
                        commit_row(row, bases.data(), 1);
                }
            }
            else
            {
                for (std::size_t row = 0; row < rows; ++row)
                    commit_row(row, generators, threads);
            }
        }

    }

    G1 load_generators(G1 *g, const std::vector<G1> &srs_g1, int n)
    {
        if (g == nullptr || n <= 0 ||
            srs_g1.size() < static_cast<std::size_t>(n))
        {
            cerr << "Error: KZG SRS too small for Hyrax generators (need "
                 << n << ", have " << srs_g1.size() << ")" << endl;
            return G1();
        }
        std::copy_n(srs_g1.data(), n, g);
        return srs_g1[0];
    }

    G1 *commit(
        const ll *w, G1 *g, int l, int thread_n, Layout layout)
    {
        int halfl = l / 2;
        int rownum = (1 << halfl);
        int colnum = (1 << (l - halfl));
        G1 *Tk = new G1[rownum];
        const auto commit_row = [&](std::size_t row, G1 *bases, int workers)
        {
            Fr *row_scalars = new Fr[colnum];
            if (layout == Layout::InterleavedLowBits)
            {
                for (int column = 0; column < colnum; ++column)
                {
                    const std::size_t index =
                        static_cast<std::size_t>(row) +
                        static_cast<std::size_t>(column) * rownum;
                    row_scalars[column] = fr_from_i128(w[index]);
                }
            }
            else
            {
                const ll *row_values =
                    w + static_cast<std::size_t>(row) * colnum;
                for (int column = 0; column < colnum; ++column)
                {
                    row_scalars[column] = fr_from_i128(row_values[column]);
                }
            }

            if (workers > 1)
            {
                G1::mulVecMT(Tk[row], bases, row_scalars, colnum, workers);
            }
            else
            {
                G1::mulVec(Tk[row], bases, row_scalars, colnum);
            }

            delete[] row_scalars;
        };
        commit_independent_rows(rownum, colnum, g, thread_n, commit_row);

        return Tk;
    }

    CommitmentPair commit_pair(
        const ll *first, const ll *second, G1 *generators, int variables,
        int requested_threads, Layout layout)
    {
        if (first == nullptr || second == nullptr || generators == nullptr ||
            variables < 0 || variables > 30)
        {
            throw std::invalid_argument("invalid Hyrax commitment pair");
        }
        const int row_variables = variables / 2;
        const std::size_t rows = std::size_t(1) << row_variables;
        const std::size_t columns =
            std::size_t(1) << (variables - row_variables);
        const int threads = std::max(1, requested_threads);
        CommitmentPair result;
        result.first.resize(rows);
        result.second.resize(rows);
        std::vector<std::vector<Fr>> first_scalars(
            static_cast<std::size_t>(threads));
        std::vector<std::vector<Fr>> second_scalars(
            static_cast<std::size_t>(threads));
        const auto commit_row = [&](std::size_t row, G1 *bases, int workers)
        {
            const int thread = omp_in_parallel() ? omp_get_thread_num() : 0;
            std::vector<Fr> &first_row =
                first_scalars[static_cast<std::size_t>(thread)];
            std::vector<Fr> &second_row =
                second_scalars[static_cast<std::size_t>(thread)];
            first_row.resize(columns);
            second_row.resize(columns);
            for (std::size_t column = 0; column < columns; ++column)
            {
                const std::size_t index = layout == Layout::InterleavedLowBits
                                              ? row + column * rows
                                              : row * columns + column;
                first_row[column] = fr_from_i128(first[index]);
                second_row[column] = fr_from_i128(second[index]);
            }
            if (workers > 1)
            {
                G1::mulVecMT(
                    result.first[row], bases, first_row.data(), columns, workers);
                G1::mulVecMT(
                    result.second[row], bases, second_row.data(), columns, workers);
            }
            else
            {
                G1::mulVec(
                    result.first[row], bases, first_row.data(), columns);
                G1::mulVec(
                    result.second[row], bases, second_row.data(), columns);
            }
        };
        commit_independent_rows(
            rows, columns, generators, threads, commit_row);
        return result;
    }

    std::vector<G1> commit_rows(const Fr *values, std::size_t value_count,
                                G1 *generators, int column_variables,
                                std::size_t first_row, std::size_t row_count,
                                int threads)
    {
        if (column_variables < 0 || column_variables > 30 ||
            (value_count && !values) || !generators)
            throw std::invalid_argument("invalid Hyrax row interval");
        const auto max_rows = std::numeric_limits<std::size_t>::max() >> column_variables;
        if (row_count > max_rows || first_row > max_rows - row_count)
            throw std::invalid_argument("Hyrax row interval overflows");
        const std::size_t columns = std::size_t(1) << column_variables;
        std::vector<G1> result(row_count);
        const auto commit_row = [&](std::size_t row, G1 *bases, int workers)
        {
            const std::size_t offset = (first_row + row) * columns;
            if (offset >= value_count)
            {
                result[row].clear();
                return;
            }
            const std::size_t count = std::min(columns, value_count - offset);
            if (workers > 1)
                G1::mulVecMT(result[row], bases, values + offset, count, workers);
            else
                G1::mulVec(result[row], bases, values + offset, count);
        };
        commit_independent_rows(row_count, columns, generators, threads, commit_row);
        return result;
    }

    G1 *commit(const Fr *w, G1 *g, int l, int thread_n, Layout layout)
    {
        int halfl = l / 2;
        int rownum = (1 << halfl);
        int colnum = (1 << (l - halfl));
        G1 *Tk = new G1[rownum];
        if (layout == Layout::ContiguousRows)
        {
            const auto rows = commit_rows(w, std::size_t(1) << l, g, l - halfl, 0, rownum, thread_n);
            std::copy(rows.begin(), rows.end(), Tk);
            return Tk;
        }
        const int threads = std::max(1, thread_n);
        const std::size_t tile_rows = std::min<std::size_t>(
            16, std::max<std::size_t>(1, static_cast<std::size_t>(rownum) / threads));
        const std::size_t tiles = (static_cast<std::size_t>(rownum) + tile_rows - 1) / tile_rows;
        const bool owns_team = threads > 1 && tiles >= static_cast<std::size_t>(threads);
        std::vector<std::vector<Fr>> packed(static_cast<std::size_t>(threads));
        const auto commit_tile = [&](std::size_t tile, G1 *bases, int workers)
        {
            const std::size_t first_row = tile * tile_rows;
            const std::size_t rows = std::min<std::size_t>(tile_rows, rownum - first_row);
            auto &values = packed[owns_team ? static_cast<std::size_t>(omp_get_thread_num()) : 0];
            values.resize(rows * colnum);
            for (std::size_t column = 0; column < static_cast<std::size_t>(colnum); ++column)
            {
                const Fr *source = w + first_row + column * rownum;
                for (std::size_t row = 0; row < rows; ++row)
                    values[row * colnum + column] = source[row];
            }
            for (std::size_t row = 0; row < rows; ++row)
            {
                const Fr *scalars = values.data() + row * colnum;
                if (workers > 1)
                {
                    G1::mulVecMT(Tk[first_row + row], bases, scalars, colnum, workers);
                }
                else
                {
                    G1::mulVec(Tk[first_row + row], bases, scalars, colnum);
                }
            }
        };
        commit_independent_rows(tiles, colnum, g, threads, commit_tile);

        return Tk;
    }

    std::vector<Fr> equality_weights(const Fr *point, int variables)
    {
        if (variables < 0 ||
            variables >= static_cast<int>(std::numeric_limits<std::size_t>::digits) ||
            (variables != 0 && point == nullptr))
        {
            return {};
        }
        return mle::equality_weights(point, static_cast<std::size_t>(variables));
    }

    Fr evaluate(const Fr *values, const Fr *point, int variables)
    {
        if (values == nullptr || variables < 0 ||
            variables >= static_cast<int>(
                             std::numeric_limits<std::size_t>::digits) ||
            (variables != 0 && point == nullptr))
        {
            return Fr(0);
        }
        return mle::evaluate(
            std::vector<Fr>(values, values + (std::size_t(1) << variables)),
            point, static_cast<std::size_t>(variables), omp_get_max_threads());
    }

    bool OpeningBatchVerifier::append(
        const OpeningProof &proof,
        const Fr *point,
        const Fr &evaluation,
        const G1 *row_commitments,
        int variables,
        Layout layout,
        const Fr *precomputed_row_weights)
    {
        if (variables < 0 ||
            variables >= static_cast<int>(
                             std::numeric_limits<std::size_t>::digits) ||
            (variables != 0 && point == nullptr))
        {
            return false;
        }
        const int row_variables = variables / 2;
        const int column_variables = variables - row_variables;
        const std::size_t row_count = std::size_t(1) << row_variables;
        if (row_commitments == nullptr ||
            row_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            proof.eval_proof.q_comm.size() !=
                static_cast<std::size_t>(column_variables))
        {
            return false;
        }

        const Fr *row_point = variables == 0
                                  ? nullptr
                                  : (layout == Layout::InterleavedLowBits
                                         ? point
                                         : point + column_variables);
        std::vector<Fr> owned_row_weights;
        const Fr *row_weights = precomputed_row_weights;
        if (row_weights == nullptr)
        {
            owned_row_weights = equality_weights(row_point, row_variables);
            if (owned_row_weights.size() != row_count)
                return false;
            row_weights = owned_row_weights.data();
        }
        G1 folded_commitment = linear_commit(
            row_commitments, row_weights,
            static_cast<int>(row_count));

        append_folded(proof, point, evaluation, folded_commitment, variables, layout);
        return true;
    }

    bool OpeningBatchVerifier::append_combined(
        const OpeningProof &proof, const Fr *point, const Fr &evaluation,
        const std::vector<RowCommitmentTerm> &terms, int variables,
        int threads, Layout layout)
    {
        if (variables < 0 ||
            variables >= static_cast<int>(std::numeric_limits<std::size_t>::digits) ||
            (variables != 0 && point == nullptr) || terms.empty() || threads <= 0)
            return false;
        const int row_variables = variables / 2;
        const int column_variables = variables - row_variables;
        const std::size_t rows = std::size_t(1) << row_variables;
        if (proof.eval_proof.q_comm.size() != static_cast<std::size_t>(column_variables) ||
            rows > std::numeric_limits<std::size_t>::max() / terms.size())
            return false;
        for (const auto &term : terms)
            if (term.rows == nullptr || term.rows->size() != rows)
                return false;

        const Fr *row_point = variables == 0 ? nullptr
                                             : (layout == Layout::InterleavedLowBits ? point : point + column_variables);
        const auto row_weights = equality_weights(row_point, row_variables);
        const std::size_t count = rows * terms.size();
        std::vector<G1> bases(count);
        std::vector<Fr> scalars(count);
        for (std::size_t i = 0; i < terms.size(); ++i)
        {
            std::copy(terms[i].rows->begin(), terms[i].rows->end(), bases.begin() + i * rows);
            for (std::size_t row = 0; row < rows; ++row)
                scalars[i * rows + row] = terms[i].coefficient * row_weights[row];
        }
        G1 folded_commitment;
        G1::mulVecMT(folded_commitment, bases.data(), scalars.data(), count, threads);
        append_folded(proof, point, evaluation, folded_commitment, variables, layout);
        return true;
    }

    void OpeningBatchVerifier::append_folded(
        const OpeningProof &proof, const Fr *point, const Fr &evaluation,
        const G1 &commitment, int variables, Layout layout)
    {
        const int row_variables = variables / 2;
        const int column_variables = variables - row_variables;
        MVKZGEvalBatchItem item;
        const Fr *column_point = variables == 0
                                     ? nullptr
                                     : (layout == Layout::InterleavedLowBits
                                            ? point + row_variables
                                            : point);
        if (column_variables != 0)
        {
            item.point.assign(column_point, column_point + column_variables);
        }
        item.commitment = commitment;
        item.evaluation = evaluation;
        item.proof = proof.eval_proof;
        items_.push_back(std::move(item));
    }

    bool OpeningBatchVerifier::verify(
        const MVKZG_VerifierKey &vk, double *verify_time_sec,
        FiatShamirTranscript *shared_transcript) const
    {
        return mvkzg_eval_batch_verify(
            items_, vk, verify_time_sec, nullptr, shared_transcript);
    }

    bool OpeningBatchVerifier::verify_with_weights(
        const MVKZG_VerifierKey &vk, const std::vector<Fr> &weights,
        double *verify_time_sec) const
    {
        if (weights.size() != items_.size())
            return false;
        return mvkzg_eval_batch_verify(
            items_, vk, verify_time_sec, &weights);
    }

    void OpeningBatchVerifier::derive_weights(
        FiatShamirTranscript &transcript, std::vector<Fr> &weights) const
    {
        mvkzg_eval_derive_batch_weights(items_, transcript, weights);
    }

    OpeningProof open_folded(const std::vector<Fr> &folded, const Fr *point,
                             int variables, const MVKZG_ProverKey &pk, Layout layout)
    {
        if (variables < 0 || variables > 30 || (variables != 0 && point == nullptr))
        {
            throw std::invalid_argument("invalid folded PCS opening dimension");
        }
        const int row_variables = variables / 2;
        const int column_variables = variables - row_variables;
        if (folded.size() != (std::size_t(1) << column_variables))
        {
            throw std::invalid_argument("folded PCS column vector has the wrong size");
        }
        const Fr *column_point = variables == 0 ? nullptr
                                                : (layout == Layout::ContiguousRows ? point : point + row_variables);
        OpeningProof proof;
        if (!mvkzg_eval_prove(folded.data(), column_variables, column_point, pk, proof.eval_proof))
        {
            throw std::runtime_error("folded MVKZG evaluation proof failed");
        }
        return proof;
    }

    namespace
    {
        const Fr &field_value(const Fr &value) { return value; }
        Fr field_value(ll value) { return fr_from_i128(value); }
        bool nonzero(const Fr &value) { return !value.isZero(); }
        bool nonzero(ll value) { return value != 0; }

        template <class Value>
        OpeningProof open_impl(const Value *w, const Fr *r, int l,
                          const MVKZG_ProverKey &pk, Layout layout,
                          const Fr *precomputed_folded)
        {
            OpeningProof proof;
            if (l < 0 || w == nullptr ||
                (l != 0 && r == nullptr))
                return proof;
            const int row_variables = l / 2;
            const int column_variables = l - row_variables;
            const int row_count = 1 << row_variables;
            const int column_count = 1 << column_variables;
            const Fr *row_point = l == 0
                                      ? nullptr
                                      : (layout == Layout::InterleavedLowBits
                                             ? r
                                             : r + column_variables);
            const Fr *column_point = l == 0
                                         ? nullptr
                                         : (layout == Layout::InterleavedLowBits
                                                ? r + row_variables
                                                : r);
            std::vector<Fr> folded_storage;
            const Fr *folded = precomputed_folded;
            if (folded == nullptr)
            {
                const std::vector<Fr> row_weights =
                    equality_weights(row_point, row_variables);
                folded_storage.assign(static_cast<std::size_t>(column_count), Fr(0));
                if (layout == Layout::ContiguousRows)
                {
                    const int thread_count = std::max(
                        1, std::min(omp_get_max_threads(), column_count));
#pragma omp parallel num_threads(thread_count) if (thread_count > 1)
                    {
                        const int thread = omp_get_thread_num();
                        const int team_size = omp_get_num_threads();
                        const int first_column = column_count * thread / team_size;
                        const int last_column =
                            column_count * (thread + 1) / team_size;
                        for (int row = 0; row < row_count; ++row)
                        {
                            const Fr row_weight =
                                row_weights[static_cast<std::size_t>(row)];
                            const Value *row_values =
                                w + static_cast<std::size_t>(row) * column_count;
                            for (int column = first_column;
                                 column < last_column; ++column)
                            {
                                if (nonzero(row_values[column]))
                                {
                                    folded_storage[static_cast<std::size_t>(column)] +=
                                        row_weight * field_value(row_values[column]);
                                }
                            }
                        }
                    }
                }
                else
                {
#pragma omp parallel for
                    for (int column = 0; column < column_count; ++column)
                    {
                        Fr acc = Fr(0);
                        for (int row = 0; row < row_count; ++row)
                        {
                            const std::size_t index =
                                static_cast<std::size_t>(row) +
                                static_cast<std::size_t>(column) * row_count;
                            acc += row_weights[static_cast<std::size_t>(row)] * field_value(w[index]);
                        }
                        folded_storage[static_cast<std::size_t>(column)] = acc;
                    }
                }
                folded = folded_storage.data();
            }

            if (!mvkzg_eval_prove(
                    folded, column_variables, column_point, pk, proof.eval_proof))
            {
                cerr << "[open] MVKZG evaluation proof failed" << endl;
            }

            return proof;
        }
    }

    OpeningProof open(const ll *w, const Fr *r, int l,
                      const MVKZG_ProverKey &pk, Layout layout,
                      const Fr *precomputed_folded)
    {
        return open_impl(w, r, l, pk, layout, precomputed_folded);
    }

    OpeningProof open(const Fr *w, const Fr *r, int l,
                      const MVKZG_ProverKey &pk, Layout layout,
                      const Fr *precomputed_folded)
    {
        return open_impl(w, r, l, pk, layout, precomputed_folded);
    }

    OpeningProof open_onehot(const int *indices, int T, int K, const Fr *r,
                             const MVKZG_ProverKey &pk, int thread_n,
                             OneHotOpeningWorkspace *workspace)
    {
        int log_T = 0;
        int temp_T = T;
        while (temp_T > 1)
        {
            log_T++;
            temp_T >>= 1;
        }

        int log_K = 0;
        int temp_K = K;
        while (temp_K > 1)
        {
            log_K++;
            temp_K >>= 1;
        }

        int l = log_K + log_T;
        int halfl = l / 2;
        int rownum = (1 << halfl);
        int colnum = (1 << (l - halfl));

        std::vector<Fr> L = equality_weights(r, halfl);

        OpeningProof proof;
        std::vector<Fr> LT(colnum, Fr(0));

        if (T <= 0 || K <= 0)
        {
            return proof;
        }

        int row_mask = rownum - 1;
        int num_threads = thread_n;
        if (num_threads < 1)
        {
            num_threads = 1;
        }
        if (num_threads > T)
        {
            num_threads = T;
        }
        if (num_threads > colnum)
        {
            num_threads = colnum;
        }

        int avg_updates_per_col = colnum > 0 ? (T / colnum) : T;
        bool use_bucket = num_threads > 1 && avg_updates_per_col >= 32;
        if (!use_bucket)
        {
            num_threads = 1;
        }

        if (num_threads <= 1)
        {
            for (int y = 0; y < T; y++)
            {
                int x = indices[y];
                if (x < 0 || x >= K)
                {
                    continue;
                }
                long long idx = static_cast<long long>(x) + static_cast<long long>(y) * K;
                int row = static_cast<int>(idx & row_mask);
                int col = static_cast<int>(idx >> halfl);
                LT[col] += L[row];
            }
        }
        else
        {
            const bool reuse_buckets =
                workspace != nullptr && workspace->buckets_ready &&
                workspace->query_count == T && workspace->table_size == K &&
                workspace->col_offsets.size() ==
                    static_cast<std::size_t>(colnum + 1) &&
                !workspace->col_offsets.empty() &&
                workspace->col_offsets.front() == 0 &&
                workspace->col_offsets.back() ==
                    static_cast<int>(workspace->bucket_rows.size());

            std::vector<int> owned_col_offsets;
            std::vector<int> owned_bucket_rows;
            std::vector<int> &col_offsets = workspace != nullptr
                                                ? workspace->col_offsets
                                                : owned_col_offsets;
            std::vector<int> &bucket_rows = workspace != nullptr
                                               ? workspace->bucket_rows
                                               : owned_bucket_rows;

            if (!reuse_buckets)
            {
                std::vector<std::vector<int>> local_counts(
                    num_threads, std::vector<int>(colnum, 0));
                col_offsets.assign(static_cast<std::size_t>(colnum + 1), 0);
                bucket_rows.clear();
                bool has_entries = true;
#pragma omp parallel num_threads(num_threads)
                {
                    int tid = omp_get_thread_num();
                    const int team_size = omp_get_num_threads();
                    int start = static_cast<int>(
                        (static_cast<long long>(tid) * T) / team_size);
                    int end = static_cast<int>(
                        (static_cast<long long>(tid + 1) * T) / team_size);
                    std::vector<int> &counts = local_counts[tid];

                    for (int y = start; y < end; y++)
                    {
                        int x = indices[y];
                        if (x < 0 || x >= K)
                        {
                            continue;
                        }
                        long long idx = static_cast<long long>(x) +
                                        static_cast<long long>(y) * K;
                        int col = static_cast<int>(idx >> halfl);
                        counts[col] += 1;
                    }

#pragma omp barrier
#pragma omp single
                    {
                        for (int col = 0; col < colnum; col++)
                        {
                            int total = 0;
                            for (int t = 0; t < team_size; t++)
                            {
                                total += local_counts[t][col];
                            }
                            col_offsets[col + 1] =
                                col_offsets[col] + total;
                        }

                        int total_entries = col_offsets[colnum];
                        if (total_entries == 0)
                        {
                            has_entries = false;
                        }
                        else
                        {
                            bucket_rows.resize(total_entries);
                        }

                        for (int col = 0; col < colnum; col++)
                        {
                            int base = col_offsets[col];
                            for (int t = 0; t < team_size; t++)
                            {
                                int c = local_counts[t][col];
                                local_counts[t][col] = base;
                                base += c;
                            }
                        }
                    }
#pragma omp barrier

                    if (has_entries)
                    {
                        std::vector<int> &cursor = local_counts[tid];
                        for (int y = start; y < end; y++)
                        {
                            int x = indices[y];
                            if (x < 0 || x >= K)
                            {
                                continue;
                            }
                            long long idx = static_cast<long long>(x) + static_cast<long long>(y) * K;
                            int row = static_cast<int>(idx & row_mask);
                            int col = static_cast<int>(idx >> halfl);
                            int pos = cursor[col]++;
                            bucket_rows[pos] = row;
                        }
                    }

                    if (has_entries)
                    {
#pragma omp barrier
#pragma omp for schedule(static)
                        for (int col = 0; col < colnum; col++)
                        {
                            Fr acc = Fr(0);
                            for (int i = col_offsets[col]; i < col_offsets[col + 1]; i++)
                            {
                                acc += L[bucket_rows[i]];
                            }
                            LT[col] = acc;
                        }
                    }
                }

                if (workspace != nullptr)
                {
                    workspace->query_count = T;
                    workspace->table_size = K;
                    workspace->buckets_ready = true;
                }
            }
            else
            {
#pragma omp parallel for num_threads(num_threads) schedule(static)
                for (int col = 0; col < colnum; col++)
                {
                    Fr acc = Fr(0);
                    for (int i = col_offsets[col]; i < col_offsets[col + 1]; i++)
                    {
                        acc += L[bucket_rows[i]];
                    }
                    LT[col] = acc;
                }
            }
        }

        if (!mvkzg_eval_prove(
                LT.data(), l - halfl, r + halfl, pk, proof.eval_proof))
        {
            cerr << "[open_onehot] MVKZG evaluation proof failed" << endl;
        }

        return proof;
    }

    bool verify(const OpeningProof &proof, const Fr *r,
                const Fr &eval, const G1 *comm, int l,
                const MVKZG_VerifierKey &vk, Layout layout,
                const Fr *precomputed_row_weights,
                VerifyTiming *timing)
    {
        if (timing != nullptr)
        {
            *timing = VerifyTiming{};
        }
        if (l < 0 || comm == nullptr || (l != 0 && r == nullptr))
            return false;
        const int row_variables = l / 2;
        const int column_variables = l - row_variables;
        const int row_count = 1 << row_variables;
        const Fr *row_point = l == 0
                                  ? nullptr
                                  : (layout == Layout::InterleavedLowBits
                                         ? r
                                         : r + column_variables);
        const Fr *column_point = l == 0
                                     ? nullptr
                                     : (layout == Layout::InterleavedLowBits
                                            ? r + row_variables
                                            : r);
        std::vector<Fr> owned_row_weights;
        const Fr *row_weights = precomputed_row_weights;
        if (row_weights == nullptr)
        {
            owned_row_weights = equality_weights(row_point, row_variables);
            if (owned_row_weights.size() != static_cast<std::size_t>(row_count))
            {
                return false;
            }
            row_weights = owned_row_weights.data();
        }
        const auto row_commitment_start =
            std::chrono::high_resolution_clock::now();
        const G1 folded_commitment = linear_commit(comm, row_weights, row_count);
        if (timing != nullptr)
        {
            timing->row_commitment_seconds =
                std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::high_resolution_clock::now() -
                    row_commitment_start)
                    .count();
        }
        return mvkzg_eval_verify(
            column_point, column_variables, folded_commitment, eval,
            proof.eval_proof, vk,
            timing == nullptr ? nullptr : &timing->pcs_pairing_seconds);
    }

    G1 *commit_onehot(const int *indices, int T, int K, G1 *g, int thread_n)
    {
        int log_T = 0;
        int temp_T = T;
        while (temp_T > 1)
        {
            log_T++;
            temp_T >>= 1;
        }

        int log_K = 0;
        int temp_K = K;
        while (temp_K > 1)
        {
            log_K++;
            temp_K >>= 1;
        }

        int l = log_K + log_T;
        int halfl = l / 2;
        int rownum = (1 << halfl);
        int colnum = (1 << (l - halfl));
        (void)colnum;

        G1 *Tk = new G1[rownum];
        for (int i = 0; i < rownum; i++)
        {
            Tk[i].clear();
        }

        if (T <= 0)
        {
            return Tk;
        }

        int row_mask = rownum - 1;

        if (thread_n <= 1)
        {
            for (int y = 0; y < T; y++)
            {
                int x = indices[y];
                if (x < 0 || x >= K)
                {
                    continue;
                }
                long long idx = static_cast<long long>(x) + static_cast<long long>(y) * K;
                int row = static_cast<int>(idx & row_mask);
                int col = static_cast<int>(idx >> halfl);
                Tk[row] += g[col];
            }
            return Tk;
        }

        int num_threads = thread_n;
        std::vector<std::vector<int>> local_counts(num_threads, std::vector<int>(rownum, 0));
        std::vector<int> row_offsets(rownum + 1, 0);
        std::vector<int> bucket_cols;
        bool has_entries = true;

#pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            const int team_size = omp_get_num_threads();
            int start = static_cast<int>((static_cast<long long>(tid) * T) / team_size);
            int end = static_cast<int>((static_cast<long long>(tid + 1) * T) / team_size);
            std::vector<int> &counts = local_counts[tid];

            for (int y = start; y < end; y++)
            {
                int x = indices[y];
                if (x < 0 || x >= K)
                {
                    continue;
                }
                long long idx = static_cast<long long>(x) + static_cast<long long>(y) * K;
                int row = static_cast<int>(idx & row_mask);
                counts[row] += 1;
            }

#pragma omp barrier
#pragma omp single
            {
                for (int row = 0; row < rownum; row++)
                {
                    int total = 0;
                    for (int t = 0; t < team_size; t++)
                    {
                        total += local_counts[t][row];
                    }
                    row_offsets[row + 1] = row_offsets[row] + total;
                }

                int total_entries = row_offsets[rownum];
                if (total_entries == 0)
                {
                    has_entries = false;
                }
                else
                {
                    bucket_cols.resize(total_entries);
                }

                for (int row = 0; row < rownum; row++)
                {
                    int base = row_offsets[row];
                    for (int t = 0; t < team_size; t++)
                    {
                        int c = local_counts[t][row];
                        local_counts[t][row] = base;
                        base += c;
                    }
                }
            }
#pragma omp barrier

            if (has_entries)
            {
                std::vector<int> &cursor = local_counts[tid];
                for (int y = start; y < end; y++)
                {
                    int x = indices[y];
                    if (x < 0 || x >= K)
                    {
                        continue;
                    }
                    long long idx = static_cast<long long>(x) + static_cast<long long>(y) * K;
                    int row = static_cast<int>(idx & row_mask);
                    int col = static_cast<int>(idx >> halfl);
                    int pos = cursor[row]++;
                    bucket_cols[pos] = col;
                }
            }

            if (has_entries)
            {
#pragma omp barrier
#pragma omp for schedule(static)
                for (int row = 0; row < rownum; row++)
                {
                    G1 acc;
                    acc.clear();
                    for (int i = row_offsets[row]; i < row_offsets[row + 1]; i++)
                    {
                        acc += g[bucket_cols[i]];
                    }
                    Tk[row] = acc;
                }
            }
        }

        return Tk;
    }

}
