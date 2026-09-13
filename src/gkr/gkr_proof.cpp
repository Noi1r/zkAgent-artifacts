#include "gkr/gkr_proof.hpp"
#include "common/cache_io.hpp"
#include "common/bit_utils.hpp"
#include "gkr/prover.hpp"
#include "pcs/srs.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unistd.h>

const F *GkrFconnProverParameters::matrixData() const
{
    if (!prepared_input_values)
        return matrix_values.data();
    if (prepared_offset > prepared_input_values->size() ||
        prepared_count > prepared_input_values->size() - prepared_offset)
        throw std::runtime_error("prepared GKR model span is no longer available");
    return prepared_input_values->data() + prepared_offset;
}

std::size_t GkrFconnProverParameters::matrixSize() const
{
    return prepared_input_values ? prepared_count : matrix_values.size();
}

namespace
{

    constexpr char GkrKeyType[] = "zkagent.gkr.fconn-key\n";

    void checked_add(std::size_t &total, std::size_t value)
    {
        if (value > std::numeric_limits<std::size_t>::max() - total)
        {
            throw std::length_error("GKR proof-size accounting overflow");
        }
        total += value;
    }

    std::size_t opening_bytes(const hyrax::OpeningProof &proof)
    {
        return proof.eval_proof.q_comm.size() * G1::getSerializedByteSize();
    }

    int exact_log2(int value)
    {
        return zkagent_integer::log2_power_of_two(
            value, "FCONN public dimensions must be powers of two");
    }

    Fr signed_int_to_field(int value)
    {
        if (value >= 0)
            return Fr(static_cast<long long>(value));
        return -Fr(static_cast<long long>(
            -static_cast<std::int64_t>(value)));
    }

}

bool GkrFconnVerifierParameters::matchesInputRouting(
    const layer &circuit, std::size_t input_values, bool check_indices) const
{
    using zkagent_integer::is_power_of_two;
    if (!is_power_of_two(rows) || !is_power_of_two(columns) ||
        !is_power_of_two(input_rows) || input_offset > input_values)
        return false;
    const std::size_t count = static_cast<std::size_t>(rows) * input_rows;
    const int input_bits = exact_log2(rows) + exact_log2(input_rows);
    if (count > input_values - input_offset ||
        static_cast<std::size_t>(input_rows) * columns != circuit.size ||
        circuit.ty != layerType::FCONN ||
        !circuit.uni_gates.empty() || !circuit.bin_gates.empty() ||
        !circuit.bin_gates_v.empty() ||
        circuit.size_u[0] != count || circuit.ori_id_u.size() != count ||
        circuit.bit_length_u[0] != input_bits || circuit.max_bl_u != input_bits ||
        circuit.size_u[1] != 0 || circuit.bit_length_u[1] != -1 ||
        circuit.size_v[0] != 0 || circuit.bit_length_v[0] != -1 ||
        circuit.size_v[1] != 0 || circuit.bit_length_v[1] != -1 ||
        !circuit.ori_id_v.empty() ||
        circuit.ori_id_u.front() != input_offset ||
        circuit.ori_id_u.back() != input_offset + count - 1)
        return false;
    if (check_indices)
        for (std::size_t i = 0; i < count; ++i)
            if (circuit.ori_id_u[i] != input_offset + i)
                return false;
    return true;
}

GkrVerifierKey makeGkrBenchmarkVerifierKey(const prover &source)
{
    GkrVerifierKey key;
    std::vector<int> fconn_layers;
    for (int layer_id = 1; layer_id < source.C.size; ++layer_id)
    {
        if (source.C.circuit[layer_id].ty == layerType::FCONN)
            fconn_layers.push_back(layer_id);
    }
    const int count = static_cast<int>(fconn_layers.size());
    key.fconn.resize(static_cast<std::size_t>(count));
    for (int id = 0; id < count; ++id)
    {
        GkrFconnVerifierParameters &params =
            key.fconn[static_cast<std::size_t>(id)];
        params.rows = source.fc_row[id];
        params.columns = source.fc_col[id];
        params.input_rows = source.fc_input_row[id];
        params.matrix_variables =
            exact_log2(params.rows) + exact_log2(params.columns);
        if (source.fc_input_id[id] < 0 || source.fc_input_col[id] != params.rows)
            throw std::invalid_argument("invalid FCONN input tensor view");
        params.input_offset = static_cast<std::size_t>(source.fc_input_id[id]);
        if (!params.matchesInputRouting(source.C.circuit[fconn_layers[id]],
                                        source.C.circuit[0].size, true))
            throw std::invalid_argument("FCONN input is not routed to the committed tensor");
        if (source.fc_bias_id[id] >= 0 && params.columns > 0)
        {
            const int offset = source.fc_bias_id[id];
            if (static_cast<std::size_t>(offset) +
                        static_cast<std::size_t>(params.columns) >
                    source.val[0].size())
            {
                throw std::invalid_argument(
                    "FCONN bias range is outside the public model data");
            }
            params.bias.assign(
                source.val[0].begin() + offset,
                source.val[0].begin() + offset + params.columns);
        }
    }
    return key;
}

namespace
{

    GkrKeyBundle prepare_gkr_keys(
        const prover &source,
        const MVKZG_ProverKey &commitment_key,
        int threads,
        GkrVerifierKey verifier_key)
    {
        if (threads <= 0)
        {
            throw std::invalid_argument("GKR key preparation needs positive threads");
        }
        GkrKeyBundle keys;
        keys.verifier = std::move(verifier_key);
        const int count = static_cast<int>(keys.verifier.fconn.size());
        keys.prover.fconn.resize(static_cast<std::size_t>(count));
        for (int id = 0; id < count; ++id)
        {
            GkrFconnVerifierParameters &params =
                keys.verifier.fconn[static_cast<std::size_t>(id)];
            GkrFconnProverParameters &prover_params =
                keys.prover.fconn[static_cast<std::size_t>(id)];
            params.rows = source.fc_row[id];
            params.columns = source.fc_col[id];
            params.input_rows = source.fc_input_row[id];
            if (source.mat_val == nullptr || source.mat_val[id] == nullptr)
            {
                throw std::invalid_argument("FCONN public matrix is unavailable");
            }
            const int row_bits = exact_log2(params.rows);
            const int column_bits = exact_log2(params.columns);
            params.matrix_variables = row_bits + column_bits;
            prover_params.matrix_variables = params.matrix_variables;
            const std::size_t matrix_size =
                static_cast<std::size_t>(params.rows) * static_cast<std::size_t>(params.columns);
            if (source.prepared_model_values)
            {
                const int offset = source.fc_start_id[id];
                if (source.val.empty() || offset < 0 ||
                    static_cast<std::size_t>(offset) > source.val[0].size() ||
                    matrix_size > source.val[0].size() - static_cast<std::size_t>(offset))
                    throw std::invalid_argument("prepared FCONN model span is outside input storage");
                prover_params.prepared_input_values = &source.val[0];
                prover_params.prepared_offset = static_cast<std::size_t>(offset);
                prover_params.prepared_count = matrix_size;
            }
            else
            {
                prover_params.matrix_values.resize(matrix_size);
#pragma omp parallel for num_threads(threads) schedule(static) if (matrix_size >= static_cast<std::size_t>(threads) * 1024)
                for (std::size_t i = 0; i < matrix_size; ++i)
                {
                    prover_params.matrix_values[i] =
                        signed_int_to_field(source.mat_val[id][i]);
                }
            }
            const int column_variables =
                params.matrix_variables - params.matrix_variables / 2;
            const std::size_t required_generators =
                std::size_t(1) << column_variables;
            if (commitment_key.srs_g1.size() < required_generators)
            {
                throw std::invalid_argument(
                    "MVKZG key is too small for an FCONN matrix commitment");
            }
            const std::size_t row_count =
                std::size_t(1) << (params.matrix_variables / 2);
            if (params.matrix_commitments.empty())
            {
                G1 *commitment = hyrax::commit(
                    prover_params.matrixData(),
                    const_cast<G1 *>(commitment_key.srs_g1.data()),
                    params.matrix_variables, threads,
                    hyrax::Layout::InterleavedLowBits);
                params.matrix_commitments.assign(commitment, commitment + row_count);
                delete[] commitment;
            }
            else if (params.matrix_commitments.size() != row_count)
            {
                throw std::invalid_argument("cached FCONN commitment shape mismatch");
            }
        }
        return keys;
    }

    void key_expect(std::istream &in, std::uint64_t expected)
    {
        if (cache_io::read_u64(in) != expected)
            throw std::runtime_error("GKR key cache metadata mismatch");
    }

    int key_column_variables(const GkrVerifierKey &key)
    {
        int result = 0;
        for (const auto &params : key.fconn)
            result = std::max(result, params.matrix_variables - params.matrix_variables / 2);
        return result;
    }

    void save_gkr_key(const std::string &path,
                      const GkrVerifierKey &key, const MVKZG_VerifierKey &vk)
    {
        const std::string temporary = path + ".tmp-" + std::to_string(getpid());
        try
        {
            std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
            out.write(GkrKeyType, sizeof(GkrKeyType) - 1);
            const int columns = key_column_variables(key);
            cache_io::write_point(out, vk.g1_generator);
            cache_io::write_point(out, vk.g2_generator);
            cache_io::write_u64(out, columns);
            for (int i = 0; i < columns; ++i)
                cache_io::write_point(out, vk.g2_variables[i]);
            cache_io::write_u64(out, key.fconn.size());
            for (const auto &params : key.fconn)
            {
                cache_io::write_u64(out, params.rows);
                cache_io::write_u64(out, params.columns);
                cache_io::write_u64(out, params.input_rows);
                cache_io::write_u64(out, params.matrix_variables);
                cache_io::write_u64(out, params.input_offset);
                cache_io::write_u64(out, params.matrix_commitments.size());
                for (const auto &point : params.matrix_commitments)
                    cache_io::write_point(out, point);
            }
            out.close();
            if (!out || std::rename(temporary.c_str(), path.c_str()) != 0)
                throw std::runtime_error("could not save GKR key cache: " + path);
        }
        catch (...)
        {
            std::remove(temporary.c_str());
            throw;
        }
    }

    bool load_gkr_key(std::istream &in,
                      GkrVerifierKey &key, const MVKZG_VerifierKey &vk)
    {
        std::array<char, sizeof(GkrKeyType) - 1> key_type{};
        in.read(key_type.data(), key_type.size());
        if (!in || std::string(key_type.data(), key_type.size()) != GkrKeyType)
            return false;
        if (cache_io::read_point<G1>(in) != vk.g1_generator ||
            cache_io::read_point<G2>(in) != vk.g2_generator)
            throw std::runtime_error("GKR key cache SRS mismatch");
        const int columns = key_column_variables(key);
        key_expect(in, columns);
        for (int i = 0; i < columns; ++i)
            if (cache_io::read_point<G2>(in) != vk.g2_variables[i])
                throw std::runtime_error("GKR key cache SRS variable mismatch");
        key_expect(in, key.fconn.size());
        for (auto &params : key.fconn)
        {
            key_expect(in, params.rows);
            key_expect(in, params.columns);
            key_expect(in, params.input_rows);
            key_expect(in, params.matrix_variables);
            key_expect(in, params.input_offset);
            const std::size_t count = std::size_t(1) << (params.matrix_variables / 2);
            key_expect(in, count);
            params.matrix_commitments.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
                params.matrix_commitments.push_back(cache_io::read_point<G1>(in));
        }
        if (in.peek() != EOF)
            throw std::runtime_error("trailing GKR key cache bytes");
        return true;
    }

}

GkrKeyBundle prepareGkrKeys(const prover &source,
                            const MVKZG_ProverKey &commitment_key, int threads)
{
    return prepare_gkr_keys(source, commitment_key, threads,
                            makeGkrBenchmarkVerifierKey(source));
}

GkrKeyBundle prepareGkrKeysCached(
    const prover &source, const MVKZG_ProverKey &commitment_key,
    const MVKZG_VerifierKey &verifier_key,
    const std::string &cache_path, int threads, bool refresh, bool *cache_loaded)
{
    if (cache_loaded)
        *cache_loaded = false;
    if (cache_path.empty() || threads <= 0)
        throw std::invalid_argument("GKR key cache requires a path and positive threads");
    auto key = makeGkrBenchmarkVerifierKey(source);
    const int columns = key_column_variables(key);
    if (verifier_key.vars < columns ||
        verifier_key.g2_variables.size() < static_cast<std::size_t>(columns) ||
        commitment_key.srs_g1.empty() ||
        commitment_key.srs_g1[0] != verifier_key.g1_generator)
        throw std::invalid_argument("GKR key cache SRS dimensions mismatch");
    bool loaded = false;
    if (!refresh)
    {
        std::ifstream in(cache_path, std::ios::binary);
        if (in)
        {
            loaded = load_gkr_key(in, key, verifier_key);
        }
    }
    auto result = prepare_gkr_keys(source, commitment_key, threads, std::move(key));
    if (!loaded)
        save_gkr_key(cache_path, result.verifier, verifier_key);
    if (cache_loaded)
        *cache_loaded = loaded;
    return result;
}

GkrInputCommitment makeGkrInputCommitment(const prover &source)
{
    GkrInputCommitment result;
    if (source.cc.comm == nullptr || source.cc.l <= 0)
        return result;
    result.variables = source.cc.l;
    const std::size_t row_count =
        std::size_t(1) << (result.variables / 2);
    result.rows.assign(source.cc.comm, source.cc.comm + row_count);
    return result;
}

void appendGkrPublicContext(
    FiatShamirTranscript &transcript,
    const layeredCircuit &circuit,
    const GkrVerifierKey &verifier_key,
    const GkrInputCommitment &input_commitment,
    const GkrPublicContext &context)
{
    transcript.append_u64(context.sequence_length);
    transcript.append_u64(context.layers);
    transcript.append_u64(context.heads);
    transcript.append_u64(context.head_dimension);
    transcript.append_u64(context.hidden_dimension);
    transcript.append_u64(context.ffn_dimension);
    transcript.append_u64(context.vocabulary_size);
    transcript.append_u64(context.merged ? 1 : 0);
    transcript.append_u64(circuit.size);
    for (int layer_id = 0; layer_id < circuit.size; ++layer_id)
    {
        const layer &current = circuit.circuit[layer_id];
        transcript.append_u64(layer_id);
        transcript.append_u64(static_cast<std::uint64_t>(current.ty));
        transcript.append_u64(current.size);
        transcript.append_u64(current.zero_start_id);
        transcript.append_u64(requiresFreshZeroRoot(circuit, layer_id) ? 1 : 0);
        transcript.append_u64(current.bit_length);
        transcript.append_u64(current.max_bl_u);
        transcript.append_u64(current.max_bl_v);
        transcript.append_u64(current.need_phase2 ? 1 : 0);
        transcript.append_u64(current.uni_gates.size());
        transcript.append_u64(current.bin_gates.size());
    }
    transcript.append_u64(verifier_key.fconn.size());
    for (const GkrFconnVerifierParameters &params : verifier_key.fconn)
    {
        transcript.append_u64(params.rows);
        transcript.append_u64(params.columns);
        transcript.append_u64(params.input_rows);
        transcript.append_u64(params.matrix_variables);
        transcript.append_u64(params.input_offset);
        transcript.append_g1_vector(params.matrix_commitments, G1WireEncoding::Compressed);
        transcript.append_fr_vector(params.bias);
    }
    transcript.append_u64(0);
    transcript.append_u64(input_commitment.variables);
    transcript.append_g1_vector(input_commitment.rows,
        G1WireEncoding::Compressed);
}

std::size_t GkrProof::cryptographicPayloadBytes() const
{
    std::size_t bytes = 32;
    for (const GkrLayerProof &layer : layers)
    {
        if (!layer.provided)
            continue;
        if (layer.is_fconn)
        {
            checked_add(bytes, layer.fconn.rounds.size() * 3ULL * 32ULL);
            checked_add(bytes, 2ULL * 32ULL);
            checked_add(bytes, opening_bytes(layer.fconn.matrix_opening));
        }
        else
        {
            checked_add(bytes, layer.phase1_rounds.size() * 3ULL * 32ULL);
            checked_add(bytes, 2ULL * 32ULL);
            if (layer.has_phase2)
            {
                checked_add(bytes,
                            layer.phase2_rounds.size() * 3ULL * 32ULL);
                checked_add(bytes, 2ULL * 32ULL);
            }
        }
    }
    checked_add(bytes,
                input_reduction_rounds.size() * 3ULL * 32ULL);
    checked_add(bytes, 32ULL);
    if (input_opening.provided)
    {
        checked_add(bytes, opening_bytes(input_opening.opening));
    }
    return bytes;
}
