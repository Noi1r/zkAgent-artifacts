#ifndef ZKAGENT_PARALLEL_BUILD_HPP
#define ZKAGENT_PARALLEL_BUILD_HPP

#include <cstddef>
#include <exception>
#include <vector>

#include <omp.h>

namespace compiled_mle {
namespace detail {

template <class Function>
void parallelBuildTasks(std::size_t count, int threads, Function function) {
    if (threads <= 1 || count <= 1) {
        for (std::size_t i = 0; i < count; ++i) function(i);
        return;
    }
    std::vector<std::exception_ptr> errors(count);
    auto invoke = [&](std::size_t i) {
        try {
            function(i);
        } catch (...) {
            errors[i] = std::current_exception();
        }
    };
    auto schedule = [&]() {
        #pragma omp taskgroup
        {
            for (std::size_t i = 0; i < count; ++i) {
                #pragma omp task firstprivate(i) shared(invoke)
                { invoke(i); }
            }
        }
    };
    if (omp_in_parallel()) {
        schedule();
    } else {
        #pragma omp parallel num_threads(threads)
        {
            #pragma omp single
            { schedule(); }
        }
    }
    for (const auto& error : errors) {
        if (error) std::rethrow_exception(error);
    }
}

}
}

#endif
