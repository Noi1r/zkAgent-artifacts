#pragma once

#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <string>

namespace benchmark_report {

class QuietLibraryOutput {
    class TailBuffer : public std::streambuf {
        static constexpr std::size_t capacity = 16 * 1024;
        std::string tail_;
        std::mutex mutex_;
        std::streamsize xsputn(const char* data, std::streamsize count) override {
            if (count <= 0) return 0;
            const std::lock_guard<std::mutex> lock(mutex_);
            const auto size = static_cast<std::size_t>(count);
            if (size >= capacity) tail_.assign(data + size - capacity, capacity);
            else {
                if (tail_.size() + size > capacity)
                    tail_.erase(0, tail_.size() + size - capacity);
                tail_.append(data, size);
            }
            return count;
        }
        int_type overflow(int_type ch) override {
            if (traits_type::eq_int_type(ch, traits_type::eof()))
                return traits_type::not_eof(ch);
            const char value = traits_type::to_char_type(ch);
            xsputn(&value, 1);
            return ch;
        }
    public:
        const std::string& tail() const { return tail_; }
    } buffer_;
    std::streambuf* original_ = nullptr;
public:
    explicit QuietLibraryOutput(bool debug) {
        if (!debug) original_ = std::cout.rdbuf(&buffer_);
    }
    QuietLibraryOutput(const QuietLibraryOutput&) = delete;
    QuietLibraryOutput& operator=(const QuietLibraryOutput&) = delete;
    void finish(bool success) {
        if (!original_) return;
        std::cout.rdbuf(original_);
        original_ = nullptr;
        if (!success && !buffer_.tail().empty())
            std::cerr << "[Failure Diagnostics]\n" << buffer_.tail() << '\n';
    }
    ~QuietLibraryOutput() { finish(false); }
};

class Writer {
    std::ostream& out_;
    bool first_section_ = true;
public:
    explicit Writer(std::ostream& out = std::cout) : out_(out) {}
    void section(const char* name) {
        if (!first_section_) out_ << '\n';
        first_section_ = false;
        out_ << '[' << name << "]\n";
    }
    template<class T> void field(const char* name, const T& value) {
        out_ << std::left << std::setw(29) << (std::string(name) + ':')
             << value << '\n';
    }
    void seconds(const char* name, double value) {
        std::ostringstream text;
        text << std::fixed << std::setprecision(6) << value << " s";
        field(name, text.str());
    }
};

}
