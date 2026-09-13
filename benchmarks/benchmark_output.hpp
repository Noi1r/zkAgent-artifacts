#pragma once

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace benchmark_output {

inline void createDirectory(const std::string& path) {
    if (::mkdir(path.c_str(), 0777) == 0) return;
    int error = errno;
    if (error == EEXIST) {
        struct stat info;
        if (::stat(path.c_str(), &info) == 0) {
            if (S_ISDIR(info.st_mode)) return;
            throw std::runtime_error("Benchmark output path is not a directory: " + path);
        }
        error = errno;
    }
    throw std::runtime_error("Cannot create benchmark output directory '" + path +
                             "': " + std::strerror(error));
}

inline void createDirectories(const std::string& path) {
    if (path.empty()) return;
    for (std::size_t end = path.find('/'); end != std::string::npos;
         end = path.find('/', end + 1)) {
        if (end != 0) createDirectory(path.substr(0, end));
    }
    createDirectory(path);
}

inline std::string reportPath(const std::string& table) {
    const char* configured = std::getenv("BENCHMARK_REPORT_PATH");
    std::string path;
    if (configured && *configured) {
        path = configured;
    } else {
        const auto now = std::chrono::system_clock::now();
        const auto epoch_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();
        const std::time_t seconds = static_cast<std::time_t>(epoch_us / 1000000);
        std::tm local_time;
        if (::localtime_r(&seconds, &local_time) == nullptr) {
            throw std::runtime_error("Cannot format benchmark output timestamp");
        }
        char timestamp[32];
        if (std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S",
                          &local_time) == 0) {
            throw std::runtime_error("Cannot format benchmark output timestamp");
        }
        std::ostringstream run_name;
        run_name << timestamp << '_' << std::setfill('0') << std::setw(6)
                 << (epoch_us % 1000000) << '_' << ::getpid();
        path = "benchmark_logs/" + table + '/' + run_name.str() + "/report.md";
    }
    const std::size_t separator = path.find_last_of('/');
    if (separator != std::string::npos) {
        createDirectories(separator == 0 ? "/" : path.substr(0, separator));
    }
    return path;
}

} // namespace benchmark_output
