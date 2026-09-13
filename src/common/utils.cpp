

#include "common/bit_utils.hpp"
#include <iostream>
#include <omp.h>

#include "common/utils.hpp"

using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::string;

i8 ceilPow2BitLength(u32 n)
{
    return n == 0 ? -1 : static_cast<i8>(zkagent_integer::ceil_log2(n));
}

void initHalfTable(vector<F> &beta_f, vector<F> &beta_s, const vector<F>::const_iterator &r, const F &init, u32 first_half, u32 second_half)
{
    beta_f[0] = init;
    beta_s[0] = F_ONE;
    for (u32 i = 0; i < first_half; ++i)
    {
        for (u32 j = 0; j < (1ULL << i); ++j)
        {
            auto tmp = beta_f[j] * r[i];
            beta_f[j | (1ULL << i)] = tmp;
            beta_f[j] = beta_f[j] - tmp;
        }
    }
    for (u32 i = 0; i < second_half; ++i)
    {
        for (u32 j = 0; j < (1ULL << i); ++j)
        {
            auto tmp = beta_s[j] * r[(i + first_half)];
            beta_s[j | (1ULL << i)] = tmp;
            beta_s[j] = beta_s[j] - tmp;
        }
    }
}

void initBetaTable(vector<F> &beta_g, u8 gLength, const vector<F>::const_iterator &r_0, const vector<F>::const_iterator &r_1,
                   const F &alpha, const F &beta, int thd)
{
    u8 first_half = gLength >> 1, second_half = gLength - first_half;
    u32 mask_fhalf = (1ULL << first_half) - 1;
    u64 total_size = 1ULL << gLength;
    const int thread_n = thd > 0 ? thd : 1;
    const bool parallel = thread_n > 1 &&
                          total_size >= static_cast<u64>(thread_n) * 256;

    static thread_local vector<F> beta_f, beta_s;
    beta_f.resize(1ULL << first_half);
    beta_s.resize(1ULL << second_half);
    F *beta_f_data = beta_f.data();
    F *beta_s_data = beta_s.data();
    if (!beta.isZero())
    {
        initHalfTable(beta_f, beta_s, r_1, beta, first_half, second_half);
#pragma omp parallel for schedule(static) if (parallel) num_threads(thread_n)
        for (u64 i = 0; i < total_size; ++i)
            beta_g[i] = beta_f_data[i & mask_fhalf] * beta_s_data[i >> first_half];
    }
    else
    {
#pragma omp parallel for schedule(static) if (parallel) num_threads(thread_n)
        for (u64 i = 0; i < total_size; ++i)
            beta_g[i].clear();
    }

    if (alpha.isZero())
        return;
    initHalfTable(beta_f, beta_s, r_0, alpha, first_half, second_half);
#pragma omp parallel for schedule(static) if (parallel) num_threads(thread_n)
    for (u64 i = 0; i < total_size; ++i)
        beta_g[i] += beta_f_data[i & mask_fhalf] * beta_s_data[i >> first_half];
}

void initBetaTable(vector<F> &beta_g, u8 gLength, const vector<F>::const_iterator &r, const F &init, int thd)
{
    if (gLength == (u8)-1)
        return;
    int first_half = gLength >> 1, second_half = gLength - first_half;
    u32 mask_fhalf = (1ULL << first_half) - 1;
    u64 total_size = 1ULL << gLength;
    const int thread_n = thd > 0 ? thd : 1;
    const bool parallel = thread_n > 1 &&
                          total_size >= static_cast<u64>(thread_n) * 256;
    static thread_local vector<F> beta_f, beta_s;
    beta_f.resize(1ULL << first_half);
    beta_s.resize(1ULL << second_half);
    F *beta_f_data = beta_f.data();
    F *beta_s_data = beta_s.data();
    if (!init.isZero())
    {
        initHalfTable(beta_f, beta_s, r, init, first_half, second_half);
#pragma omp parallel for schedule(static) if (parallel) num_threads(thread_n)
        for (u64 i = 0; i < total_size; ++i)
            beta_g[i] = beta_f_data[i & mask_fhalf] * beta_s_data[i >> first_half];
    }
    else
    {
#pragma omp parallel for schedule(static) if (parallel) num_threads(thread_n)
        for (u64 i = 0; i < total_size; ++i)
            beta_g[i].clear();
    }
}

void initLayer(layer &circuit, long size, layerType ty)
{
    circuit.size = circuit.zero_start_id = size;
    circuit.bit_length = ceilPow2BitLength(size);
    circuit.ty = ty;
}
