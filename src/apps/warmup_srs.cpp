
#include "pcs/srs.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <mcl/bn256.hpp>

using namespace mcl::bn;
using std::cout;
using std::endl;
using Clock = std::chrono::steady_clock;

static double secs_since(const Clock::time_point& t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

int main(int argc, char** argv) {
    int mv_max_vars   = (argc > 1) ? std::atoi(argv[1]) : 19;
    int uni_max_log_n = (argc > 2) ? std::atoi(argv[2]) : 16;
    std::string cache_dir = (argc > 3) ? argv[3] : "cache";

    initPairing(mcl::BN_SNARK1);

    cout << "warmup_srs: cache_dir=" << cache_dir
         << " mv_vars=13.." << mv_max_vars
         << " uni_log_n=7.." << uni_max_log_n << endl;

    auto t_start = Clock::now();

    for (int vars = 13; vars <= mv_max_vars; ++vars) {
        auto t0 = Clock::now();
        MVKZG_SRS srs;
        srs.setup_cached(vars, cache_dir);
        cout << "  mvkzg m=" << vars << "  size=" << (size_t(1) << vars)
             << "  " << secs_since(t0) << " s" << endl;
    }

    std::vector<int> uni_sizes = {128, 1024, 65536};
    for (int log_n = 7; log_n <= uni_max_log_n; ++log_n) {
        size_t n = size_t(1) << log_n;
        bool already = false;
        for (int s : uni_sizes) {
            if (s == (int)n) { already = true; break; }
        }
        if (!already) uni_sizes.push_back((int)n);
    }
    for (int n : uni_sizes) {
        auto t0 = Clock::now();
        KZG_SRS srs;
        srs.setup_cached(n, cache_dir);
        cout << "  uni   N=" << n
             << "  " << secs_since(t0) << " s" << endl;
    }

    cout << "warmup_srs: total " << secs_since(t_start) << " s" << endl;
    return 0;
}
