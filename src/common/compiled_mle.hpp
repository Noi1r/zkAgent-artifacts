#ifndef ZKAGENT_COMPILED_MLE_HPP
#define ZKAGENT_COMPILED_MLE_HPP

#include "common/global_var.hpp"

#include <cstddef>
#include <array>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace compiled_mle
{

    using EncodedAssignment = unsigned __int128;

    struct Entry
    {
        EncodedAssignment point = 0;
        ll value = 0;
    };

    struct BuildTiming
    {
        std::size_t input_entries = 0;
        std::size_t retained_entries = 0;
        std::size_t nonterminal_nodes = 0;
        std::size_t terminal_values = 0;
        double sort_seconds = 0;
        double coalesce_seconds = 0;
        double terminal_seconds = 0;
        double memo_prepare_seconds = 0;
        double dag_seconds = 0;
        double cleanup_seconds = 0;
        std::size_t reuse_queries = 0;
        std::size_t reuse_hits = 0;
        std::size_t reused_entries = 0;
        std::size_t compared_entries = 0;
    };

    class Polynomial
    {
    public:
        Polynomial() = default;
        Polynomial(const Polynomial &) = delete;
        Polynomial &operator=(const Polynomial &) = delete;
        Polynomial(Polynomial &&) noexcept = default;
        Polynomial &operator=(Polynomial &&) noexcept = default;

        static Polynomial build(
            std::vector<Entry> entries,
            std::size_t variable_count,
            int threads = 1,
            BuildTiming *timing = nullptr,
            bool reuse_subproblems = false);

        Fr evaluate(
            const std::vector<Fr> &point,
            int threads,
            std::vector<Fr> &workspace) const;

        void writeTo(std::ostream &output) const;
        static Polynomial readFrom(
            std::istream &input);

        bool empty() const { return root_ == 0; }
        std::size_t variableCount() const { return variable_count_; }
        std::size_t nodeCount() const;
        std::size_t terminalCount() const { return terminals_.size(); }
        std::size_t workspaceElements() const { return value_count_; }
        std::size_t logicalStorageBytes() const;

    private:
        friend struct Builder;

        struct Node
        {
            std::uint32_t id = 0;
            std::uint32_t low = 0;
            std::uint32_t high = 0;
        };

        std::size_t variable_count_ = 0;
        std::uint32_t root_ = 0;
        std::size_t value_count_ = 1;
        std::vector<Fr> terminals_;
        std::vector<std::vector<Node>> nodes_by_level_;
    };

    class InterleavedIndexEncoder
    {
    public:
        InterleavedIndexEncoder(int first_bits, int second_bits, int third_bits = 0);
        EncodedAssignment encode(u32 first, u32 second, u32 third = 0) const;

    private:
        std::array<int, 3> widths_;
        std::array<std::vector<std::array<EncodedAssignment, 256>>, 3> tables_;
    };

    EncodedAssignment encodeInterleaved(
        u32 first, int first_bits,
        u32 second, int second_bits);

    EncodedAssignment encodeInterleaved(
        u32 first, int first_bits,
        u32 second, int second_bits,
        u32 third, int third_bits);

    std::vector<Fr> interleavePoint(
        const std::vector<Fr> &first, int first_bits,
        const std::vector<Fr> &second, int second_bits);

    std::vector<Fr> interleavePoint(
        const std::vector<Fr> &first, int first_bits,
        const std::vector<Fr> &second, int second_bits,
        const std::vector<Fr> &third, int third_bits);

}

#endif
