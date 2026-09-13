#ifndef ZKAGENT_SRS_HPP
#define ZKAGENT_SRS_HPP

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include <mcl/bn256.hpp>

using namespace mcl::bn;

void setSrsProgressOutput(std::ostream &output);

struct MVKZG_ProverKey
{
    std::vector<G1> srs_g1;
    int vars = 0;
    std::size_t table_size = 0;
};

struct MVKZG_VerifierKey
{
    G1 g1_generator;
    G2 g2_generator;
    std::vector<G2> g2_variables;
    int vars = 0;
};

struct MVKZG_SRS
{
    MVKZG_ProverKey pk;
    MVKZG_VerifierKey vk;

    void setup(int vars);
    static int min_vars_for_size(std::size_t min_size);

    void setup_cached(int vars, const std::string &cache_dir = "cache");
};

struct KZG_ProverKey
{
    std::vector<G1> srs_g1;
    std::vector<G2> srs_g2;
    int N = 0;

    G1 commit_g1(const Fr *poly_coeff, int len) const;
    G2 commit_g2(const Fr *poly_coeff, int len) const;
};

struct KZG_VerifierKey
{
    G1 g1_generator;
    G2 g2_generator;
    G2 g2_tau;
    int srs_size = 0;
};

struct KZG_SRS
{
    KZG_ProverKey pk;
    KZG_VerifierKey vk;

    void setup(int n);
    void setup_cached(int n, const std::string &cache_dir = "cache");
};

struct SRSPlan
{
    int mv_vars = -1;
    int uni_size = 0;
};

struct SRSBundle
{
    MVKZG_SRS mv;
    KZG_SRS uni;

    void setup_cached(const SRSPlan &plan,
                      const std::string &cache_dir = "cache");
};

struct SRSVerifierBundle
{
    MVKZG_VerifierKey mv;
    KZG_VerifierKey uni;

    bool load_cached(const SRSPlan &plan,
                     const std::string &cache_dir = "cache");
};

#endif
