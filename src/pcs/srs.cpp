#include "pcs/srs.hpp"
#include "common/bit_utils.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>

#include <omp.h>

using namespace std;
using namespace mcl::bn;

static std::ostream &ml_log = std::cerr;
static std::ostream *progress_output = &std::cerr;

void setSrsProgressOutput(std::ostream &output) { progress_output = &output; }
static std::ostream &ml_info() { return *progress_output; }

int MVKZG_SRS::min_vars_for_size(size_t min_size)
{
    if (min_size > (std::size_t(1) << (8 * sizeof(std::size_t) - 1)))
        throw std::runtime_error("MVKZG_SRS::min_vars_for_size: size too large");
    return zkagent_integer::ceil_log2(min_size);
}

namespace
{

    constexpr std::size_t G1SerializedSize = 96;
    constexpr std::size_t G2SerializedSize = 192;
    constexpr std::size_t SrsIoChunkSize = 10000000;
    constexpr char MvkzgCacheType[] = "zkagent.mvkzg.srs\n";
    constexpr char KzgCacheType[] = "zkagent.kzg.srs\n";
    constexpr const char *MvkzgSetupSeed = "zkAgent_mvkzg_tau_v1_mv";
    constexpr const char *KzgSetupSeed = "zkAgent_trusted_setup_seed_v1_uni";

    void initialize_mvkzg_keys(MVKZG_ProverKey &pk,
                               MVKZG_VerifierKey &vk,
                               int vars,
                               const std::string &base_seed)
    {
        pk.vars = vars;
        pk.table_size = vars == 0 ? 1 : (std::size_t(1) << vars);
        pk.srs_g1.resize(pk.table_size);
        vk.vars = vars;
        vk.g2_variables.resize(static_cast<std::size_t>(vars));

        std::vector<Fr> taus(static_cast<std::size_t>(vars));
        for (int i = 0; i < vars; ++i)
        {
            const std::string seed_i = base_seed + "_" + std::to_string(i);
            taus[static_cast<std::size_t>(i)].setHashOf(
                seed_i.c_str(), seed_i.size());
        }

        hashAndMapToG1(vk.g1_generator, "g1_generator", 12);
        hashAndMapToG2(vk.g2_generator, "g2_generator", 12);
        pk.srs_g1[0] = vk.g1_generator;
        for (int i = 0; i < vars; ++i)
        {
            G2::mul(vk.g2_variables[static_cast<std::size_t>(i)],
                    vk.g2_generator, taus[static_cast<std::size_t>(i)]);
        }

        std::size_t block = 1;
        for (int i = 0; i < vars; ++i)
        {
#pragma omp parallel for schedule(static)
            for (std::size_t j = 0; j < block; ++j)
            {
                G1::mul(pk.srs_g1[j + block], pk.srs_g1[j],
                        taus[static_cast<std::size_t>(i)]);
            }
            block <<= 1;
        }
    }

    bool read_exact(std::istream &input, char *data, std::size_t size)
    {
        input.read(data, static_cast<std::streamsize>(size));
        return input.good() || input.gcount() == static_cast<std::streamsize>(size);
    }

    template <std::size_t N>
    bool read_cache_type(std::istream &input, const char (&expected)[N])
    {
        char actual[N - 1] = {};
        return read_exact(input, actual, sizeof(actual)) &&
               std::memcmp(actual, expected, sizeof(actual)) == 0;
    }

}

void MVKZG_SRS::setup(int vars)
{
    if (vars < 0 || vars >= static_cast<int>(8 * sizeof(std::size_t)))
    {
        throw std::runtime_error("MVKZG_SRS::setup: invalid vars");
    }
    initialize_mvkzg_keys(pk, vk, vars, MvkzgSetupSeed);
}

namespace
{

    std::string mvkzg_cache_filename(const std::string &cache_dir, int vars)
    {
        return cache_dir + "/mvkzg_m" + std::to_string(vars) + ".bin";
    }

    bool save_mvkzg_cache(const std::string &filename,
                          const MVKZG_ProverKey &pk,
                          const MVKZG_VerifierKey &vk)
    {
        if (pk.vars < 0 || vk.vars != pk.vars || pk.srs_g1.empty() ||
            pk.srs_g1.size() != pk.table_size ||
            vk.g2_variables.size() != static_cast<std::size_t>(pk.vars))
        {
            return false;
        }
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            ml_log << "save_mvkzg_cache: failed to open "
                   << filename << std::endl;
            return false;
        }

        const std::size_t buffer_size = 64 * 1024 * 1024;
        std::vector<char> buffer(buffer_size);
        file.rdbuf()->pubsetbuf(buffer.data(), buffer_size);

        const std::uint32_t vars_u = static_cast<std::uint32_t>(pk.vars);
        const std::uint64_t table_u = static_cast<std::uint64_t>(pk.table_size);
        file.write(MvkzgCacheType, sizeof(MvkzgCacheType) - 1);
        file.write(reinterpret_cast<const char *>(&vars_u), sizeof(vars_u));
        file.write(reinterpret_cast<const char *>(&table_u), sizeof(table_u));

        ml_info() << "save_mvkzg_cache: saving G1 prover key..."
                  << std::endl;
        for (std::size_t chunk_start = 0; chunk_start < pk.table_size;
             chunk_start += SrsIoChunkSize)
        {
            const std::size_t chunk_end = std::min(
                chunk_start + SrsIoChunkSize, pk.table_size);
            const std::size_t chunk_len = chunk_end - chunk_start;
            std::vector<char> chunk_buf(chunk_len * G1SerializedSize);
#pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < chunk_len; ++i)
            {
                pk.srs_g1[chunk_start + i].serialize(
                    chunk_buf.data() + i * G1SerializedSize,
                    G1SerializedSize, mcl::IoEcAffineSerialize);
            }
            file.write(chunk_buf.data(),
                       static_cast<std::streamsize>(chunk_buf.size()));
        }

        ml_info() << "save_mvkzg_cache: saving compact verifier key..."
                  << std::endl;
        std::vector<char> verifier_buf(
            (static_cast<std::size_t>(pk.vars) + 1) * G2SerializedSize);
        vk.g2_generator.serialize(
            verifier_buf.data(), G2SerializedSize, mcl::IoEcAffineSerialize);
        for (int i = 0; i < pk.vars; ++i)
        {
            vk.g2_variables[static_cast<std::size_t>(i)].serialize(
                verifier_buf.data() +
                    (static_cast<std::size_t>(i) + 1) * G2SerializedSize,
                G2SerializedSize, mcl::IoEcAffineSerialize);
        }
        file.write(verifier_buf.data(),
                   static_cast<std::streamsize>(verifier_buf.size()));
        file.close();
        ml_info() << "save_mvkzg_cache: saved SRS (m=" << pk.vars
                  << ", prover G1=" << pk.table_size
                  << ", verifier G2=" << (pk.vars + 1) << ") to "
                  << filename << std::endl;
        return true;
    }

    bool load_mvkzg_cache(const std::string &filename,
                          MVKZG_ProverKey &pk,
                          MVKZG_VerifierKey &vk)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            ml_info() << "load_mvkzg_cache: cache not found: "
                      << filename << std::endl;
            return false;
        }

        if (!read_cache_type(file, MvkzgCacheType))
            return false;
        std::uint32_t vars_u = 0;
        std::uint64_t table_u = 0;
        file.read(reinterpret_cast<char *>(&vars_u), sizeof(vars_u));
        file.read(reinterpret_cast<char *>(&table_u), sizeof(table_u));
        if (!file)
        {
            ml_log << "load_mvkzg_cache: invalid cache header"
                   << std::endl;
            return false;
        }

        pk.vars = static_cast<int>(vars_u);
        pk.table_size = static_cast<std::size_t>(table_u);
        if (pk.vars < 0 || pk.vars >= static_cast<int>(8 * sizeof(std::size_t)) ||
            pk.table_size != (pk.vars == 0
                                  ? std::size_t(1)
                                  : (std::size_t(1) << pk.vars)))
        {
            return false;
        }
        vk.vars = pk.vars;
        pk.srs_g1.resize(pk.table_size);
        for (std::size_t chunk_start = 0; chunk_start < pk.table_size;
             chunk_start += SrsIoChunkSize)
        {
            const std::size_t chunk_end = std::min(
                chunk_start + SrsIoChunkSize, pk.table_size);
            const std::size_t chunk_len = chunk_end - chunk_start;
            std::vector<char> chunk_buf(chunk_len * G1SerializedSize);
            if (!read_exact(file, chunk_buf.data(), chunk_buf.size()))
                return false;
#pragma omp parallel for schedule(dynamic, 10000)
            for (std::size_t i = 0; i < chunk_len; ++i)
            {
                pk.srs_g1[chunk_start + i].deserialize(
                    chunk_buf.data() + i * G1SerializedSize,
                    G1SerializedSize, mcl::IoEcAffineSerialize);
            }
        }

        vk.g1_generator = pk.srs_g1[0];
        vk.g2_variables.resize(static_cast<std::size_t>(pk.vars));
        std::vector<char> verifier_buf(
            (static_cast<std::size_t>(pk.vars) + 1) * G2SerializedSize);
        if (!read_exact(file, verifier_buf.data(), verifier_buf.size()))
        {
            return false;
        }
        vk.g2_generator.deserialize(
            verifier_buf.data(), G2SerializedSize,
            mcl::IoEcAffineSerialize);
        for (int i = 0; i < pk.vars; ++i)
        {
            vk.g2_variables[static_cast<std::size_t>(i)].deserialize(
                verifier_buf.data() +
                    (static_cast<std::size_t>(i) + 1) * G2SerializedSize,
                G2SerializedSize, mcl::IoEcAffineSerialize);
        }
        return true;
    }

    bool load_mvkzg_verifier_key(
        const std::string &filename,
        MVKZG_VerifierKey &vk)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
            return false;

        if (!read_cache_type(file, MvkzgCacheType))
            return false;
        std::uint32_t vars_u = 0;
        std::uint64_t table_u = 0;
        file.read(reinterpret_cast<char *>(&vars_u), sizeof(vars_u));
        file.read(reinterpret_cast<char *>(&table_u), sizeof(table_u));
        if (!file)
        {
            return false;
        }
        const int vars = static_cast<int>(vars_u);
        const std::size_t table_size = static_cast<std::size_t>(table_u);
        if (vars < 0 || vars >= static_cast<int>(8 * sizeof(std::size_t)) ||
            table_size != (vars == 0
                               ? std::size_t(1)
                               : (std::size_t(1) << vars)))
        {
            return false;
        }

        const std::streamoff header_size = static_cast<std::streamoff>(
            sizeof(MvkzgCacheType) - 1 + sizeof(std::uint32_t) + sizeof(std::uint64_t));
        std::vector<char> encoded_g1(G1SerializedSize);
        file.seekg(header_size, std::ios::beg);
        if (!read_exact(file, encoded_g1.data(), encoded_g1.size()))
            return false;
        vk.g1_generator.deserialize(
            encoded_g1.data(), G1SerializedSize, mcl::IoEcAffineSerialize);

        const std::streamoff g2_offset = header_size +
                                         static_cast<std::streamoff>(table_size * G1SerializedSize);
        std::vector<char> encoded_g2(G2SerializedSize);
        file.seekg(g2_offset, std::ios::beg);
        if (!read_exact(file, encoded_g2.data(), encoded_g2.size()))
            return false;
        vk.g2_generator.deserialize(
            encoded_g2.data(), G2SerializedSize, mcl::IoEcAffineSerialize);
        vk.vars = vars;
        vk.g2_variables.resize(static_cast<std::size_t>(vars));
        for (int i = 0; i < vars; ++i)
        {
            const std::size_t stored_index = static_cast<std::size_t>(i + 1);
            file.seekg(g2_offset + static_cast<std::streamoff>(
                                       stored_index * G2SerializedSize),
                       std::ios::beg);
            if (!read_exact(file, encoded_g2.data(), encoded_g2.size()))
            {
                return false;
            }
            vk.g2_variables[static_cast<std::size_t>(i)].deserialize(
                encoded_g2.data(), G2SerializedSize,
                mcl::IoEcAffineSerialize);
        }
        return true;
    }

}

void MVKZG_SRS::setup_cached(int vars, const std::string &cache_dir)
{
    if (vars < 0 || vars >= static_cast<int>(8 * sizeof(std::size_t)))
    {
        throw std::runtime_error("MVKZG_SRS::setup_cached: invalid vars");
    }
    struct stat st;
    if (stat(cache_dir.c_str(), &st) == -1)
    {
        mkdir(cache_dir.c_str(), 0755);
    }
    const std::string cache_filename =
        mvkzg_cache_filename(cache_dir, vars);

    MVKZG_ProverKey loaded_pk;
    MVKZG_VerifierKey loaded_vk;
    if (load_mvkzg_cache(cache_filename, loaded_pk, loaded_vk))
    {
        const std::size_t expected_size = vars == 0
                                              ? std::size_t(1)
                                              : (std::size_t(1) << vars);
        if (loaded_pk.vars == vars &&
            loaded_pk.table_size == expected_size &&
            loaded_vk.vars == vars &&
            loaded_vk.g2_variables.size() ==
                static_cast<std::size_t>(vars))
        {
            pk = std::move(loaded_pk);
            vk = std::move(loaded_vk);
            return;
        }
    }

    ml_info() << "MVKZG_SRS::setup_cached: Generating compact SRS (m="
              << vars << ")..." << std::endl;
    initialize_mvkzg_keys(pk, vk, vars, MvkzgSetupSeed);
    save_mvkzg_cache(cache_filename, pk, vk);
}

void KZG_SRS::setup(int n)
{
    if (n <= 0)
    {
        throw std::invalid_argument("KZG_SRS::setup: size must be positive");
    }
    pk.N = n;
    pk.srs_g1.resize(n);
    pk.srs_g2.resize(n + 1);

    Fr tau;
    tau.setHashOf(KzgSetupSeed, std::char_traits<char>::length(KzgSetupSeed));

    G1 g1_gen;
    G2 g2_gen;

    hashAndMapToG1(g1_gen, "g1_generator", 12);
    hashAndMapToG2(g2_gen, "g2_generator", 12);

    Fr tau_pow = Fr(1);
    for (int i = 0; i < n; i++)
    {
        G1::mul(pk.srs_g1[i], g1_gen, tau_pow);
        G2::mul(pk.srs_g2[i], g2_gen, tau_pow);
        Fr::mul(tau_pow, tau_pow, tau);
    }
    G2::mul(pk.srs_g2[n], g2_gen, tau_pow);
    vk.g1_generator = g1_gen;
    vk.g2_generator = g2_gen;
    vk.g2_tau = pk.srs_g2[1];
    vk.srs_size = n;
}

G1 KZG_ProverKey::commit_g1(const Fr *poly_coeff, int len) const
{
    G1 result;
    result.clear();

    if (len > 0 && len <= (int)srs_g1.size())
    {
        G1::mulVecMT(result, const_cast<G1 *>(srs_g1.data()),
                     poly_coeff, len, 0);
    }
    return result;
}

G2 KZG_ProverKey::commit_g2(const Fr *poly_coeff, int len) const
{
    G2 result;
    result.clear();

    if (len > 0 && len <= (int)srs_g2.size())
    {
        G2::mulVecMT(result, const_cast<G2 *>(srs_g2.data()),
                     poly_coeff, len, 0);
    }
    return result;
}

namespace
{

    std::string kzg_cache_filename(const std::string &cache_dir, int size)
    {
        return cache_dir + "/kzg_N" + std::to_string(size) + ".bin";
    }

    bool save_kzg_cache(const std::string &filename,
                        const std::vector<G1> &srs_g1,
                        const std::vector<G2> &srs_g2,
                        int N)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            ml_log << "save_kzg_cache: failed to open " << filename << std::endl;
            return false;
        }

        const size_t BUFFER_SIZE = 64 * 1024 * 1024;
        std::vector<char> buffer(BUFFER_SIZE);
        file.rdbuf()->pubsetbuf(buffer.data(), BUFFER_SIZE);

        const size_t G1_SZ = 96;
        const size_t G2_SZ = 192;

        file.write(KzgCacheType, sizeof(KzgCacheType) - 1);
        file.write(reinterpret_cast<const char *>(&N), sizeof(N));

        const int CHUNK_SIZE = 10000000;
        ml_info() << "save_kzg_cache: saving G1 prover key..." << std::endl;

        for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE)
        {
            int chunk_end = std::min(chunk_start + CHUNK_SIZE, N);
            int chunk_len = chunk_end - chunk_start;

            std::vector<char> chunk_buf(chunk_len * G1_SZ);
#pragma omp parallel for
            for (int i = 0; i < chunk_len; i++)
            {
                srs_g1[chunk_start + i].serialize(chunk_buf.data() + i * G1_SZ, G1_SZ, mcl::IoEcAffineSerialize);
            }

            file.write(chunk_buf.data(), chunk_buf.size());

            ml_info() << "  Saved " << chunk_end / 1000000 << "M/" << N / 1000000 << "M G1" << std::endl;
        }

        ml_info() << "save_kzg_cache: saving G2 prover key..." << std::endl;

        for (int chunk_start = 0; chunk_start <= N; chunk_start += CHUNK_SIZE)
        {
            int chunk_end = std::min(chunk_start + CHUNK_SIZE, N + 1);
            int chunk_len = chunk_end - chunk_start;

            std::vector<char> chunk_buf(chunk_len * G2_SZ);
#pragma omp parallel for
            for (int i = 0; i < chunk_len; i++)
            {
                srs_g2[chunk_start + i].serialize(chunk_buf.data() + i * G2_SZ, G2_SZ, mcl::IoEcAffineSerialize);
            }

            file.write(chunk_buf.data(), chunk_buf.size());

            ml_info() << "  Saved " << std::min(chunk_end, N + 1) / 1000000 << "M/" << (N + 1) / 1000000 << "M G2" << std::endl;
        }

        file.close();
        ml_info() << "save_kzg_cache: saved SRS (N=" << N << ") to "
                  << filename << std::endl;
        return true;
    }

    bool load_kzg_cache(const std::string &filename,
                        std::vector<G1> &srs_g1,
                        std::vector<G2> &srs_g2,
                        int &N)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            ml_info() << "load_kzg_cache: cache not found: " << filename << std::endl;
            return false;
        }

        if (!read_cache_type(file, KzgCacheType))
            return false;
        N = 0;
        file.read(reinterpret_cast<char *>(&N), sizeof(N));
        if (!file || N <= 0)
            return false;

        const size_t G1_SZ = 96;
        const size_t G2_SZ = 192;

        const int CHUNK_SIZE = 10000000;

        srs_g1.resize(N);

        for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE)
        {
            int chunk_end = std::min(chunk_start + CHUNK_SIZE, N);
            int chunk_len = chunk_end - chunk_start;
            std::vector<char> chunk_buf(chunk_len * G1_SZ);
            if (!read_exact(file, chunk_buf.data(), chunk_buf.size()))
                return false;
#pragma omp parallel for schedule(dynamic, 10000)
            for (int i = 0; i < chunk_len; i++)
            {
                srs_g1[chunk_start + i].deserialize(
                    chunk_buf.data() + i * G1_SZ, G1_SZ,
                    mcl::IoEcAffineSerialize);
            }
        }

        srs_g2.resize(N + 1);
        for (int chunk_start = 0; chunk_start <= N; chunk_start += CHUNK_SIZE)
        {
            int chunk_end = std::min(chunk_start + CHUNK_SIZE, N + 1);
            int chunk_len = chunk_end - chunk_start;
            std::vector<char> chunk_buf(chunk_len * G2_SZ);
            if (!read_exact(file, chunk_buf.data(), chunk_buf.size()))
                return false;
#pragma omp parallel for schedule(dynamic, 10000)
            for (int i = 0; i < chunk_len; i++)
            {
                srs_g2[chunk_start + i].deserialize(
                    chunk_buf.data() + i * G2_SZ, G2_SZ,
                    mcl::IoEcAffineSerialize);
            }
        }

        return true;
    }

    bool load_kzg_verifier_key(
        const std::string &filename,
        KZG_VerifierKey &vk)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
            return false;
        if (!read_cache_type(file, KzgCacheType))
            return false;
        int size = 0;
        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (!file || size <= 0)
        {
            return false;
        }
        constexpr std::streamoff header_size =
            sizeof(KzgCacheType) - 1 + sizeof(int);
        std::vector<char> encoded_g1(G1SerializedSize);
        file.seekg(header_size, std::ios::beg);
        if (!read_exact(file, encoded_g1.data(), encoded_g1.size()))
            return false;
        vk.g1_generator.deserialize(
            encoded_g1.data(), G1SerializedSize, mcl::IoEcAffineSerialize);

        const std::streamoff g2_offset = header_size +
                                         static_cast<std::streamoff>(size) * G1SerializedSize;
        std::vector<char> encoded_g2(2 * G2SerializedSize);
        file.seekg(g2_offset, std::ios::beg);
        if (!read_exact(file, encoded_g2.data(), encoded_g2.size()))
            return false;
        vk.g2_generator.deserialize(
            encoded_g2.data(), G2SerializedSize, mcl::IoEcAffineSerialize);
        vk.g2_tau.deserialize(
            encoded_g2.data() + G2SerializedSize,
            G2SerializedSize, mcl::IoEcAffineSerialize);
        vk.srs_size = size;
        return true;
    }

}

void KZG_SRS::setup_cached(int n, const std::string &cache_dir)
{
    if (n <= 0)
    {
        throw std::invalid_argument(
            "KZG_SRS::setup_cached: size must be positive");
    }

    struct stat st;
    if (stat(cache_dir.c_str(), &st) == -1)
    {
        mkdir(cache_dir.c_str(), 0755);
    }

    int cache_size = 1;
    while (cache_size < n)
        cache_size <<= 1;

    std::vector<int> try_sizes = {cache_size, cache_size * 2, cache_size * 4, 65536, 131072};

    for (int try_size : try_sizes)
    {
        if (try_size < n)
            continue;

        const std::string cache_filename =
            kzg_cache_filename(cache_dir, try_size);
        int loaded_N;

        if (load_kzg_cache(
                cache_filename, pk.srs_g1, pk.srs_g2, loaded_N))
        {
            if (loaded_N == try_size)
            {
                pk.N = n;

                if ((int)pk.srs_g1.size() > n)
                {
                    pk.srs_g1.resize(n);
                }
                if ((int)pk.srs_g2.size() > n + 1)
                {
                    pk.srs_g2.resize(n + 1);
                }
                vk.g1_generator = pk.srs_g1[0];
                vk.g2_generator = pk.srs_g2[0];
                vk.g2_tau = pk.srs_g2[1];
                vk.srs_size = n;

                return;
            }
        }
    }

    ml_info() << "KZG_SRS::setup_cached: Generating SRS (N="
              << cache_size << ")..." << std::endl;

    pk.N = cache_size;
    pk.srs_g1.resize(cache_size);
    pk.srs_g2.resize(cache_size + 1);

    Fr tau;
    tau.setHashOf(KzgSetupSeed,
                  std::char_traits<char>::length(KzgSetupSeed));

    G1 g1_gen;
    G2 g2_gen;
    hashAndMapToG1(g1_gen, "g1_generator", 12);
    hashAndMapToG2(g2_gen, "g2_generator", 12);

    std::vector<Fr> tau_powers(cache_size + 1);
    tau_powers[0] = Fr(1);

    for (int i = 1; i <= cache_size; i++)
    {
        Fr::mul(tau_powers[i], tau_powers[i - 1], tau);
    }
    ml_info() << "  Tau powers computed" << std::endl;

    int num_threads = omp_get_max_threads();
    ml_info() << "  Using " << num_threads << " threads for parallel SRS generation" << std::endl;

    std::atomic<int> progress(0);
    int report_interval = std::max(1, cache_size / 20);

#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 1000)
        for (int i = 0; i < cache_size; i++)
        {
            G1::mul(pk.srs_g1[i], g1_gen, tau_powers[i]);

            int current = ++progress;
            if (current % report_interval == 0)
            {
#pragma omp critical
                {
                    ml_info() << "  Generated " << current << "/" << cache_size
                              << " G1 elements (" << (100 * current / cache_size) << "%)" << std::endl;
                }
            }
        }
    }
    ml_info() << "  G1 elements done" << std::endl;

    progress = 0;
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 1000)
        for (int i = 0; i <= cache_size; i++)
        {
            G2::mul(pk.srs_g2[i], g2_gen, tau_powers[i]);

            int current = ++progress;
            if (current % report_interval == 0)
            {
#pragma omp critical
                {
                    ml_info() << "  Generated " << current << "/" << (cache_size + 1)
                              << " G2 elements (" << (100 * current / (cache_size + 1)) << "%)" << std::endl;
                }
            }
        }
    }
    ml_info() << "  G2 elements done" << std::endl;

    const std::string cache_filename =
        kzg_cache_filename(cache_dir, cache_size);
    save_kzg_cache(cache_filename, pk.srs_g1, pk.srs_g2, cache_size);

    pk.N = n;
    if (cache_size > n)
    {
        pk.srs_g1.resize(n);
        pk.srs_g2.resize(n + 1);
    }
    vk.g1_generator = pk.srs_g1[0];
    vk.g2_generator = pk.srs_g2[0];
    vk.g2_tau = pk.srs_g2[1];
    vk.srs_size = n;
}

void SRSBundle::setup_cached(const SRSPlan &plan,
                             const std::string &cache_dir)
{
    if (plan.mv_vars < -1 || plan.uni_size < 0)
    {
        throw std::invalid_argument("SRSBundle: invalid SRS plan");
    }
    if (plan.mv_vars >= 0)
    {
        mv.setup_cached(plan.mv_vars, cache_dir);
    }
    if (plan.uni_size > 0)
    {
        uni.setup_cached(plan.uni_size, cache_dir);
    }
}

bool SRSVerifierBundle::load_cached(const SRSPlan &plan,
                                    const std::string &cache_dir)
{
    if (plan.mv_vars < -1 || plan.uni_size < 0)
        return false;
    if (plan.mv_vars >= 0)
    {
        const std::string filename =
            mvkzg_cache_filename(cache_dir, plan.mv_vars);
        if (!load_mvkzg_verifier_key(filename, mv) ||
            mv.vars != plan.mv_vars)
        {
            return false;
        }
    }
    if (plan.uni_size > 0)
    {
        int cache_size = 1;
        while (cache_size < plan.uni_size)
            cache_size <<= 1;
        const std::vector<int> candidates = {
            cache_size, cache_size * 2, cache_size * 4, 65536, 131072};
        bool loaded = false;
        for (int candidate : candidates)
        {
            if (candidate < plan.uni_size)
                continue;
            KZG_VerifierKey candidate_vk;
            const std::string filename =
                kzg_cache_filename(cache_dir, candidate);
            if (load_kzg_verifier_key(filename, candidate_vk) &&
                candidate_vk.srs_size == candidate &&
                candidate_vk.srs_size >= plan.uni_size)
            {
                uni = std::move(candidate_vk);
                loaded = true;
                break;
            }
        }
        if (!loaded)
            return false;
    }
    return true;
}
