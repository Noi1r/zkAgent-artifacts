#ifndef ZKAGENT_OMP_THREADS_HPP
#define ZKAGENT_OMP_THREADS_HPP

#include <omp.h>
#include <stdexcept>

class ScopedOmpThreads {
public:
    explicit ScopedOmpThreads(int threads)
        : previous_threads_(omp_get_max_threads()),
          previous_active_levels_(omp_get_max_active_levels()) {
        if (threads <= 0) throw std::invalid_argument("thread count must be positive");
        omp_set_num_threads(threads);
        omp_set_max_active_levels(1);
    }

    ~ScopedOmpThreads() {
        omp_set_max_active_levels(previous_active_levels_);
        omp_set_num_threads(previous_threads_);
    }

    ScopedOmpThreads(const ScopedOmpThreads&) = delete;
    ScopedOmpThreads& operator=(const ScopedOmpThreads&) = delete;

private:
    int previous_threads_;
    int previous_active_levels_;
};

#endif
