#ifndef ZKAGENT_ZKDECODE_EXP_DIGITS_HPP
#define ZKAGENT_ZKDECODE_EXP_DIGITS_HPP

#include "common/typedef.hpp"

#include <array>
#include <string>
#include <vector>

constexpr double ZkDecodeTemperature = 1.0;
constexpr ll ZkDecodeProbabilityScale = static_cast<ll>(1) << 20;
constexpr std::array<int, 6> DecodeExpDigitBases{{1024, 1024, 8, 2048, 1024, 16}};

struct DecodeExpDigitsConfig
{
    std::array<int, 6> bases = DecodeExpDigitBases;
    int least_segments = 3;
    int nonlinear_segment = 3;
    int most_zero_segments = 2;
    double logit_scale = 7.952219286828541e-10;
    ll max_abs_logit = static_cast<ll>(84380505604096LL);
    ll expected_theta = ZkDecodeProbabilityScale;
    int max_active_count = 40;
};

struct DecodeExpDigitsWitness
{
    bool success = false;
    std::string message;
    DecodeExpDigitsConfig config;

    ll shift = 0;
    ll output_scale = 0;

    std::vector<std::vector<ll>> digits;
    std::vector<ll> nonlinear_outputs;
    std::vector<std::vector<ll>> high_zero_flags;
    std::vector<ll> unmasked_outputs;
};

struct DecodeExpDigitsPublicTables
{
    std::array<std::vector<ll>, 6> inputs;
    std::array<std::vector<ll>, 6> outputs;
};

const DecodeExpDigitsPublicTables &decode_exp_digits_public_tables(
    const DecodeExpDigitsConfig &config);

DecodeExpDigitsWitness build_decode_exp_digits_witness(
    const std::vector<ll> &deltas,
    const std::vector<int> &active,
    ll theta,
    const DecodeExpDigitsConfig &config);

ll decode_exp_digits_table_output(
    ll digit,
    ll theta,
    const DecodeExpDigitsConfig &config);

ll decode_exp_digits_shift_upper_bound(
    int active_count,
    const DecodeExpDigitsConfig &config);

ll decode_exp_digits_recommended_epsilon(
    ll theta,
    int active_count,
    const DecodeExpDigitsConfig &config);

int decode_exp_digit_bits(int base);

#endif
