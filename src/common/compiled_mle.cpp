#include "common/compiled_mle.hpp"
#include "common/field_integer.hpp"
#include "common/parallel_build.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <istream>
#include <limits>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <unordered_map>

#include <omp.h>

namespace compiled_mle
{
    namespace
    {

        template <typename T>
        void writeScalar(std::ostream &output, const T &value)
        {
            output.write(reinterpret_cast<const char *>(&value), sizeof(value));
            if (!output)
            {
                throw std::runtime_error("failed to write compiled MLE plan");
            }
        }

        template <typename T>
        T readScalar(std::istream &input)
        {
            T value{};
            input.read(reinterpret_cast<char *>(&value), sizeof(value));
            if (!input)
            {
                throw std::runtime_error("truncated compiled MLE plan");
            }
            return value;
        }

        struct ScalarKey
        {
            std::uint64_t low = 0;
            std::uint64_t high = 0;

            bool operator==(const ScalarKey &other) const
            {
                return low == other.low && high == other.high;
            }
        };

        struct ScalarKeyHash
        {
            std::size_t operator()(const ScalarKey &value) const
            {
                std::uint64_t hash = value.low ^
                                     (value.high + 0x9e3779b97f4a7c15ULL +
                                      (value.low << 6) + (value.low >> 2));
                hash ^= hash >> 30;
                hash *= 0xbf58476d1ce4e5b9ULL;
                hash ^= hash >> 27;
                hash *= 0x94d049bb133111ebULL;
                return static_cast<std::size_t>(hash ^ (hash >> 31));
            }
        };

        class UniqueNodes
        {
        public:
            struct Slot
            {
                std::uint32_t low = 0;
                std::uint32_t high = 0;
                std::uint32_t id = 0;
            };

            Slot &locate(std::uint32_t low, std::uint32_t high)
            {
                if (slots_.empty())
                    grow(16);
                std::size_t position = find(slots_, low, high);
                if (slots_[position].id == 0 &&
                    size_ + 1 > slots_.size() - slots_.size() / 4)
                {
                    if (slots_.size() > std::numeric_limits<std::size_t>::max() / 2)
                        throw std::length_error("compiled MLE unique table is too large");
                    grow(slots_.size() * 2);
                    position = find(slots_, low, high);
                }
                return slots_[position];
            }

            void insert(Slot &slot, std::uint32_t low, std::uint32_t high,
                        std::uint32_t id)
            {
                slot = {low, high, id};
                ++size_;
            }

        private:
            static std::size_t hash(std::uint32_t low, std::uint32_t high)
            {
                std::uint64_t hash = (static_cast<std::uint64_t>(low) << 32) | high;
                hash ^= hash >> 30;
                hash *= 0xbf58476d1ce4e5b9ULL;
                hash ^= hash >> 27;
                hash *= 0x94d049bb133111ebULL;
                return static_cast<std::size_t>(hash ^ (hash >> 31));
            }

            static std::size_t find(const std::vector<Slot> &slots,
                                    std::uint32_t low, std::uint32_t high)
            {
                const std::size_t mask = slots.size() - 1;
                std::size_t position = hash(low, high) & mask;
                while (slots[position].id != 0 &&
                       (slots[position].low != low || slots[position].high != high))
                    position = (position + 1) & mask;
                return position;
            }

            void grow(std::size_t capacity)
            {
                std::vector<Slot> next(capacity);
                for (const Slot &slot : slots_)
                {
                    if (slot.id != 0)
                        next[find(next, slot.low, slot.high)] = slot;
                }
                slots_.swap(next);
            }

            std::vector<Slot> slots_;
            std::size_t size_ = 0;
        };

        void sortEntries(std::vector<Entry> &entries, int threads)
        {
            if (entries.size() < (std::size_t(1) << 18))
            {
                std::sort(entries.begin(), entries.end(),
                          [](const Entry &a, const Entry &b)
                          { return a.point < b.point; });
                return;
            }

            const std::size_t count = entries.size();
            const std::size_t chunks = std::min<std::size_t>(
                static_cast<std::size_t>(threads), 1 + (count - 1) / (1U << 16));
            std::vector<EncodedAssignment> variations(chunks, 0);
            const EncodedAssignment first = entries.front().point;
            detail::parallelBuildTasks(chunks, threads, [&](std::size_t chunk)
                                       {
        EncodedAssignment bits = 0;
        for (std::size_t i = count * chunk / chunks;
             i < count * (chunk + 1) / chunks; ++i)
            bits |= entries[i].point ^ first;
        variations[chunk] = bits; });
            EncodedAssignment varying = 0;
            for (EncodedAssignment bits : variations)
                varying |= bits;
            if (varying == 0)
                return;

            std::vector<Entry> buffer(count);
            std::vector<std::array<std::size_t, 256>> offsets(chunks);
            Entry *source = entries.data();
            Entry *target = buffer.data();
            for (unsigned shift = 0; shift < 128; shift += 8)
            {
                if (((varying >> shift) & 255U) == 0)
                    continue;
                detail::parallelBuildTasks(chunks, threads, [&](std::size_t chunk)
                                           {
            auto& histogram = offsets[chunk];
            histogram.fill(0);
            for (std::size_t i = count * chunk / chunks;
                 i < count * (chunk + 1) / chunks; ++i)
                ++histogram[static_cast<unsigned>((source[i].point >> shift) & 255U)]; });
                std::size_t next = 0;
                for (std::size_t digit = 0; digit < 256; ++digit)
                {
                    for (std::size_t chunk = 0; chunk < chunks; ++chunk)
                    {
                        const std::size_t size = offsets[chunk][digit];
                        offsets[chunk][digit] = next;
                        next += size;
                    }
                }
                detail::parallelBuildTasks(chunks, threads, [&](std::size_t chunk)
                                           {
            auto& positions = offsets[chunk];
            for (std::size_t i = count * chunk / chunks;
                 i < count * (chunk + 1) / chunks; ++i) {
                const unsigned digit = static_cast<unsigned>(
                    (source[i].point >> shift) & 255U);
                target[positions[digit]++] = source[i];
            } });
                std::swap(source, target);
            }
            if (source == buffer.data())
                entries.swap(buffer);
        }

        ScalarKey scalarKey(ll value)
        {
            static_assert(sizeof(ll) == 2 * sizeof(std::uint64_t),
                          "compiled MLE expects a 128-bit coefficient type");
            ScalarKey result;
            std::memcpy(&result.low, &value, sizeof(result.low));
            std::memcpy(
                &result.high,
                reinterpret_cast<const std::uint8_t *>(&value) + sizeof(result.low),
                sizeof(result.high));
            return result;
        }

        std::size_t checkedVariableCount(int first, int second, int third = 0)
        {
            if (first < 0 || second < 0 || third < 0)
            {
                throw std::invalid_argument(
                    "compiled MLE received a negative bit length");
            }
            const std::size_t result = static_cast<std::size_t>(
                first + second + third);
            if (result > 120)
            {
                throw std::invalid_argument(
                    "compiled MLE supports at most 120 variables");
            }
            return result;
        }

        bool indexFits(u32 index, int bits)
        {
            return bits >= 32 ||
                   index < (static_cast<std::uint32_t>(1) << bits);
        }

        void appendInterleaved(
            EncodedAssignment &output,
            u32 first, int first_bits,
            u32 second, int second_bits,
            u32 third, int third_bits)
        {
            if (!indexFits(first, first_bits) ||
                !indexFits(second, second_bits) ||
                !indexFits(third, third_bits))
            {
                throw std::invalid_argument(
                    "compiled MLE index exceeds its declared domain");
            }
            const int max_bits = std::max(first_bits, std::max(second_bits, third_bits));
            for (int bit = 0; bit < max_bits; ++bit)
            {
                if (bit < first_bits)
                {
                    output = (output << 1) | ((first >> bit) & 1U);
                }
                if (bit < second_bits)
                {
                    output = (output << 1) | ((second >> bit) & 1U);
                }
                if (bit < third_bits)
                {
                    output = (output << 1) | ((third >> bit) & 1U);
                }
            }
        }

        std::vector<Fr> interleavePointImpl(
            const std::vector<Fr> &first, int first_bits,
            const std::vector<Fr> &second, int second_bits,
            const std::vector<Fr> &third, int third_bits)
        {
            if (first_bits < 0 || second_bits < 0 || third_bits < 0 ||
                first.size() < static_cast<std::size_t>(first_bits) ||
                second.size() < static_cast<std::size_t>(second_bits) ||
                third.size() < static_cast<std::size_t>(third_bits))
            {
                throw std::invalid_argument(
                    "compiled MLE evaluation point has the wrong dimension");
            }
            std::vector<Fr> result;
            result.reserve(checkedVariableCount(first_bits, second_bits, third_bits));
            const int max_bits = std::max(first_bits, std::max(second_bits, third_bits));
            for (int bit = 0; bit < max_bits; ++bit)
            {
                if (bit < first_bits)
                {
                    result.push_back(first[static_cast<std::size_t>(bit)]);
                }
                if (bit < second_bits)
                {
                    result.push_back(second[static_cast<std::size_t>(bit)]);
                }
                if (bit < third_bits)
                {
                    result.push_back(third[static_cast<std::size_t>(bit)]);
                }
            }
            return result;
        }

        class SubproblemMemo
        {
        public:
            static constexpr std::size_t MinimumEntries = 256;
            struct Key
            {
                std::uint64_t fingerprint = 0;
                std::size_t count = 0;
                std::size_t level = 0;
                bool operator==(const Key &other) const
                {
                    return fingerprint == other.fingerprint && count == other.count &&
                           level == other.level;
                }
            };

            explicit SubproblemMemo(const std::vector<Entry> &entries, int threads)
                : entries_(entries), prefix_(new std::uint64_t[entries.size() + 1])
            {
                const std::size_t count = entries.size();
                const std::size_t chunks = std::min<std::size_t>(
                    static_cast<std::size_t>(threads), 1 + count / (1U << 16));
                std::vector<std::uint64_t> ends(chunks), offsets(chunks);
                prefix_[0] = 0;
                detail::parallelBuildTasks(chunks, threads, [&](std::size_t chunk)
                                           {
            std::uint64_t hash = 0;
            for (std::size_t i = count * chunk / chunks;
                 i < count * (chunk + 1) / chunks; ++i) {
                const EncodedAssignment delta = entries[i].point -
                    (i == 0 ? EncodedAssignment(0) : entries[i - 1].point);
                hash = hash * Base + token(delta, entries[i].value);
                prefix_[i + 1] = hash;
            }
            ends[chunk] = hash; });
                std::uint64_t hash = 0;
                for (std::size_t chunk = 0; chunk < chunks; ++chunk)
                {
                    offsets[chunk] = hash;
                    const std::size_t length = count * (chunk + 1) / chunks - count * chunk / chunks;
                    hash = hash * power(length) + ends[chunk];
                }
                detail::parallelBuildTasks(chunks, threads, [&](std::size_t chunk)
                                           {
            if (offsets[chunk] == 0) return;
            std::uint64_t scale = Base;
            for (std::size_t i = count * chunk / chunks;
                 i < count * (chunk + 1) / chunks; ++i) {
                prefix_[i + 1] += offsets[chunk] * scale;
                scale *= Base;
            } });
                records_.reserve(1024);
            }

            Key key(std::size_t begin, std::size_t end, std::size_t level,
                    std::size_t variables) const
            {
                const EncodedAssignment mask = (EncodedAssignment(1) << (variables - level)) - 1;
                const std::uint64_t suffix = prefix_[end] - prefix_[begin + 1] * power(end - begin - 1);
                return {suffix ^ token(entries_[begin].point & mask, entries_[begin].value),
                        end - begin, level};
            }

            bool lookup(const Key &key, std::size_t begin, std::size_t variables,
                        std::uint32_t &id)
            {
                ++queries;
                const auto found = records_.find(key);
                if (found == records_.end())
                    return false;
                const EncodedAssignment mask = (EncodedAssignment(1) << (variables - key.level)) - 1;
                std::size_t matched = 0;
                for (; matched < key.count; ++matched)
                {
                    const Entry &a = entries_[begin + matched];
                    const Entry &b = entries_[found->second.begin + matched];
                    if (((a.point ^ b.point) & mask) != 0 || a.value != b.value)
                        break;
                }
                compared_entries += matched + (matched < key.count ? 1 : 0);
                if (matched != key.count)
                    return false;
                ++hits;
                reused_entries += key.count;
                id = found->second.id;
                return true;
            }

            void remember(const Key &key, std::size_t begin, std::uint32_t id)
            {
                records_.emplace(key, Record{begin, id});
            }

            std::size_t queries = 0, hits = 0, reused_entries = 0, compared_entries = 0;

        private:
            static constexpr std::uint64_t Base = 0x9e3779b185ebca87ULL;
            static std::uint64_t mix(std::uint64_t value)
            {
                value ^= value >> 30;
                value *= 0xbf58476d1ce4e5b9ULL;
                value ^= value >> 27;
                value *= 0x94d049bb133111ebULL;
                return value ^ (value >> 31);
            }
            static std::uint64_t token(EncodedAssignment delta, ll value)
            {
                const auto coefficient = static_cast<EncodedAssignment>(value);
                std::uint64_t hash = mix(static_cast<std::uint64_t>(delta) + Base);
                hash = mix(hash ^ static_cast<std::uint64_t>(delta >> 64));
                hash = mix(hash + static_cast<std::uint64_t>(coefficient));
                return mix(hash ^ static_cast<std::uint64_t>(coefficient >> 64));
            }
            static std::uint64_t power(std::size_t exponent)
            {
                std::uint64_t result = 1, base = Base;
                while (exponent != 0)
                {
                    if (exponent & 1U)
                        result *= base;
                    base *= base;
                    exponent >>= 1;
                }
                return result;
            }
            struct KeyHash
            {
                std::size_t operator()(const Key &key) const
                {
                    return static_cast<std::size_t>(mix(key.fingerprint ^
                                                        (key.count * Base) ^ (static_cast<std::uint64_t>(key.level) << 48)));
                }
            };
            struct Record
            {
                std::size_t begin;
                std::uint32_t id;
            };
            const std::vector<Entry> &entries_;
            std::unique_ptr<std::uint64_t[]> prefix_;
            std::unordered_map<Key, Record, KeyHash> records_;
        };

    }

    InterleavedIndexEncoder::InterleavedIndexEncoder(
        int first_bits, int second_bits, int third_bits)
        : widths_{{first_bits, second_bits, third_bits}}
    {
        const auto variables = checkedVariableCount(first_bits, second_bits, third_bits);
        std::array<std::array<unsigned, 32>, 3> positions{};
        std::size_t level = 0;
        for (int bit = 0; bit < std::max({first_bits, second_bits, third_bits}); ++bit)
        {
            for (std::size_t role = 0; role < widths_.size(); ++role)
            {
                if (bit >= widths_[role])
                    continue;
                if (bit < 32)
                    positions[role][bit] = static_cast<unsigned>(variables - 1 - level);
                ++level;
            }
        }
        for (std::size_t role = 0; role < widths_.size(); ++role)
        {
            tables_[role].resize(static_cast<std::size_t>(std::min(4, (widths_[role] + 7) / 8)));
            for (std::size_t byte = 0; byte < tables_[role].size(); ++byte)
            {
                for (unsigned value = 0; value < 256; ++value)
                {
                    EncodedAssignment encoded = 0;
                    for (unsigned bit = 0; bit < 8; ++bit)
                    {
                        const auto source_bit = static_cast<unsigned>(byte * 8 + bit);
                        if (source_bit < static_cast<unsigned>(widths_[role]) &&
                            ((value >> bit) & 1U))
                            encoded |= EncodedAssignment(1) << positions[role][source_bit];
                    }
                    tables_[role][byte][value] = encoded;
                }
            }
        }
    }

    EncodedAssignment InterleavedIndexEncoder::encode(u32 first, u32 second, u32 third) const
    {
        const std::array<u32, 3> indices{{first, second, third}};
        EncodedAssignment result = 0;
        for (std::size_t role = 0; role < indices.size(); ++role)
        {
            if (!indexFits(indices[role], widths_[role]))
                throw std::invalid_argument("compiled MLE index exceeds its declared domain");
            for (std::size_t byte = 0; byte < tables_[role].size(); ++byte)
                result |= tables_[role][byte][(indices[role] >> (8 * byte)) & 255U];
        }
        return result;
    }

    struct Builder
    {
        using Node = Polynomial::Node;

        std::size_t variables = 0;
        std::uint32_t next_node_id = 1;
        std::unordered_map<ScalarKey, std::uint32_t, ScalarKeyHash> terminal_ids;
        std::vector<UniqueNodes> nodes;
        std::unique_ptr<SubproblemMemo> memo;
        Polynomial *output = nullptr;

        std::uint32_t intern(std::size_t level,
                             std::uint32_t low, std::uint32_t high)
        {
            if (low == high)
                return low;
            UniqueNodes &table = nodes[level];
            UniqueNodes::Slot &slot = table.locate(low, high);
            if (slot.id != 0)
                return slot.id;
            if (next_node_id == std::numeric_limits<std::uint32_t>::max())
            {
                throw std::length_error(
                    "compiled MLE decision DAG value IDs exceed 32 bits");
            }
            const std::uint32_t id = next_node_id++;
            table.insert(slot, low, high, id);
            output->nodes_by_level_[level].push_back({id, low, high});
            return id;
        }

        template <bool Reuse>
        std::uint32_t build(
            const std::vector<Entry> &entries,
            std::size_t begin,
            std::size_t end,
            std::size_t level)
        {
            if (begin == end)
                return 0;
            SubproblemMemo::Key key;
            const bool candidate = Reuse && end - begin >= SubproblemMemo::MinimumEntries;
            if (candidate)
            {
                key = memo->key(begin, end, level, variables);
                std::uint32_t existing = 0;
                if (memo->lookup(key, begin, variables, existing))
                    return existing;
            }
            if (end == begin + 1)
            {
                const Entry &entry = entries[begin];
                std::uint32_t id = terminal_ids.size() == 1
                                       ? 1
                                       : terminal_ids.at(scalarKey(entry.value));
                for (std::size_t depth = variables; depth > level; --depth)
                {
                    const unsigned shift = static_cast<unsigned>(variables - depth);
                    id = ((entry.point >> shift) & 1U) != 0
                             ? intern(depth - 1, 0, id)
                             : intern(depth - 1, id, 0);
                }
                return id;
            }
            if (level == variables)
            {
                throw std::logic_error("compiled MLE retained duplicate leaves");
            }

            const unsigned shift = static_cast<unsigned>(variables - 1 - level);
            const EncodedAssignment mask = EncodedAssignment(1) << shift;
            const auto first = entries.begin() + static_cast<std::ptrdiff_t>(begin);
            const auto last = entries.begin() + static_cast<std::ptrdiff_t>(end);
            const auto split = std::partition_point(
                first, last,
                [mask](const Entry &entry)
                {
                    return (entry.point & mask) == 0;
                });
            const std::size_t middle = static_cast<std::size_t>(
                split - entries.begin());
            const std::uint32_t low = build<Reuse>(entries, begin, middle, level + 1);
            const std::uint32_t high = build<Reuse>(entries, middle, end, level + 1);
            const std::uint32_t id = intern(level, low, high);
            if (candidate)
                memo->remember(key, begin, id);
            return id;
        }
    };

    Polynomial Polynomial::build(
        std::vector<Entry> entries,
        std::size_t variable_count,
        int threads,
        BuildTiming *timing,
        bool reuse_subproblems)
    {
        if (variable_count > 120 || threads <= 0)
        {
            throw std::invalid_argument(
                "compiled MLE received invalid construction parameters");
        }
        Polynomial result;
        result.variable_count_ = variable_count;
        result.nodes_by_level_.resize(variable_count);
        if (timing)
        {
            *timing = BuildTiming{};
            timing->input_entries = entries.size();
        }
        if (entries.empty())
            return result;

        using Clock = std::chrono::steady_clock;
        auto start = Clock::now();
        sortEntries(entries, threads);
        if (timing)
            timing->sort_seconds = std::chrono::duration<double>(
                                       Clock::now() - start)
                                       .count();
        start = Clock::now();
        std::size_t write = 0;
        for (const Entry &entry : entries)
        {
            if (write != 0 && entries[write - 1].point == entry.point)
            {
                ll sum = 0;
                if (__builtin_add_overflow(entries[write - 1].value,
                                           entry.value, &sum))
                {
                    throw std::overflow_error(
                        "compiled MLE duplicate coefficient overflow");
                }
                entries[write - 1].value = sum;
            }
            else
            {
                entries[write++] = entry;
            }
        }
        entries.resize(write);
        entries.erase(
            std::remove_if(
                entries.begin(), entries.end(),
                [](const Entry &entry)
                { return entry.value == 0; }),
            entries.end());
        if (timing)
        {
            timing->retained_entries = entries.size();
            timing->coalesce_seconds = std::chrono::duration<double>(
                                           Clock::now() - start)
                                           .count();
        }
        if (entries.empty())
            return result;

        {
            start = Clock::now();
            Builder builder;
            builder.variables = variable_count;
            builder.output = &result;
            builder.nodes.resize(variable_count);
            builder.terminal_ids.reserve(std::min<std::size_t>(entries.size(), 1024));
            ll previous = 0;
            bool have_previous = false;
            for (const Entry &entry : entries)
            {
                if (have_previous && previous == entry.value)
                    continue;
                previous = entry.value;
                have_previous = true;
                if (builder.terminal_ids.size() >=
                        std::numeric_limits<std::uint32_t>::max() - 1 &&
                    builder.terminal_ids.find(scalarKey(entry.value)) ==
                        builder.terminal_ids.end())
                {
                    throw std::length_error(
                        "compiled MLE terminal value IDs exceed 32 bits");
                }
                const auto inserted = builder.terminal_ids.emplace(
                    scalarKey(entry.value), static_cast<std::uint32_t>(
                                                builder.terminal_ids.size() + 1));
                if (inserted.second)
                    result.terminals_.push_back(zkagent_integer::fr_from_i128(entry.value));
            }
            builder.next_node_id = static_cast<std::uint32_t>(
                result.terminals_.size() + 1);
            if (timing)
                timing->terminal_seconds = std::chrono::duration<double>(
                                               Clock::now() - start)
                                               .count();
            start = Clock::now();
            if (reuse_subproblems && entries.size() >= 4096 &&
                (entries.back().point >> variable_count) == 0)
                builder.memo.reset(new SubproblemMemo(entries, threads));
            if (timing)
                timing->memo_prepare_seconds = std::chrono::duration<double>(
                                                   Clock::now() - start)
                                                   .count();
            start = Clock::now();
            result.root_ = builder.memo
                               ? builder.build<true>(entries, 0, entries.size(), 0)
                               : builder.build<false>(entries, 0, entries.size(), 0);
            result.value_count_ = builder.next_node_id;
            if (timing)
                timing->dag_seconds = std::chrono::duration<double>(
                                          Clock::now() - start)
                                          .count();
            if (timing && builder.memo)
            {
                timing->reuse_queries = builder.memo->queries;
                timing->reuse_hits = builder.memo->hits;
                timing->reused_entries = builder.memo->reused_entries;
                timing->compared_entries = builder.memo->compared_entries;
            }
            start = Clock::now();
        }
        if (timing)
            timing->cleanup_seconds = std::chrono::duration<double>(
                                          Clock::now() - start)
                                          .count();
        if (timing)
        {
            timing->nonterminal_nodes = result.nodeCount();
            timing->terminal_values = result.terminalCount();
        }
        return result;
    }

    Fr Polynomial::evaluate(
        const std::vector<Fr> &point,
        int threads,
        std::vector<Fr> &workspace) const
    {
        if (empty())
            return Fr(0);
        if (point.size() != variable_count_ || threads <= 0 ||
            workspace.size() < value_count_)
        {
            throw std::invalid_argument(
                "compiled MLE evaluation has an invalid point or workspace");
        }

        workspace[0] = Fr(0);
        for (std::size_t terminal = 0; terminal < terminals_.size(); ++terminal)
        {
            workspace[terminal + 1] = terminals_[terminal];
        }
        for (std::size_t reverse = variable_count_; reverse > 0; --reverse)
        {
            const std::size_t level = reverse - 1;
            const std::vector<Node> &nodes = nodes_by_level_[level];
            const Fr coordinate = point[level];
            const int actual_threads = std::max(
                1, std::min<int>(threads, static_cast<int>(nodes.size())));
#pragma omp parallel for schedule(static) num_threads(actual_threads) if (nodes.size() >= 1024 && actual_threads > 1)
            for (long long raw = 0;
                 raw < static_cast<long long>(nodes.size()); ++raw)
            {
                const Node &node = nodes[static_cast<std::size_t>(raw)];
                workspace[node.id] = workspace[node.low] + coordinate *
                                                               (workspace[node.high] - workspace[node.low]);
            }
        }
        return workspace[root_];
    }

    std::size_t Polynomial::nodeCount() const
    {
        std::size_t result = 0;
        for (const std::vector<Node> &level : nodes_by_level_)
        {
            result += level.size();
        }
        return result;
    }

    std::size_t Polynomial::logicalStorageBytes() const
    {
        return nodeCount() * sizeof(Node) + terminals_.size() * sizeof(Fr);
    }

    void Polynomial::writeTo(std::ostream &output) const
    {
        writeScalar(output, static_cast<std::uint32_t>(variable_count_));
        writeScalar(output, root_);
        writeScalar(output, static_cast<std::uint64_t>(terminals_.size()));
        std::array<std::uint8_t, 32> encoded{};
        for (const Fr &terminal : terminals_)
        {
            if (terminal.serialize(encoded.data(), encoded.size()) !=
                encoded.size())
            {
                throw std::runtime_error(
                    "failed to encode compiled MLE terminal");
            }
            output.write(
                reinterpret_cast<const char *>(encoded.data()), encoded.size());
            if (!output)
            {
                throw std::runtime_error("failed to write compiled MLE terminal");
            }
        }
        writeScalar(output, static_cast<std::uint64_t>(nodes_by_level_.size()));
        for (const std::vector<Node> &level : nodes_by_level_)
        {
            writeScalar(output, static_cast<std::uint64_t>(level.size()));
            for (const Node &node : level)
            {
                writeScalar(output, node.id);
                writeScalar(output, node.low);
                writeScalar(output, node.high);
            }
        }
    }

    Polynomial Polynomial::readFrom(
        std::istream &input)
    {
        Polynomial result;
        result.variable_count_ = readScalar<std::uint32_t>(input);
        result.root_ = readScalar<std::uint32_t>(input);
        if (result.variable_count_ > 120)
        {
            throw std::runtime_error(
                "compiled MLE plan has too many variables");
        }
        const std::uint64_t terminal_count = readScalar<std::uint64_t>(input);
        if (terminal_count >= std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error(
                "compiled MLE plan has too many terminals");
        }
        result.terminals_.resize(static_cast<std::size_t>(terminal_count));
        std::array<std::uint8_t, 32> encoded{};
        std::array<std::uint8_t, 32> canonical{};
        for (Fr &terminal : result.terminals_)
        {
            input.read(reinterpret_cast<char *>(encoded.data()), encoded.size());
            if (!input ||
                terminal.deserialize(encoded.data(), encoded.size()) !=
                    encoded.size() ||
                terminal.serialize(canonical.data(), canonical.size()) !=
                    canonical.size() ||
                canonical != encoded)
            {
                throw std::runtime_error(
                    "compiled MLE plan has an invalid field terminal");
            }
        }
        const std::uint64_t level_count = readScalar<std::uint64_t>(input);
        if (level_count != result.variable_count_)
        {
            throw std::runtime_error(
                "compiled MLE plan has inconsistent levels");
        }
        result.nodes_by_level_.resize(result.variable_count_);
        std::size_t total_nodes = 0;
        for (std::size_t level = 0; level < result.variable_count_; ++level)
        {
            const std::uint64_t count = readScalar<std::uint64_t>(input);
            if (count > std::numeric_limits<std::uint32_t>::max() -
                            1 - terminal_count - total_nodes)
            {
                throw std::runtime_error(
                    "compiled MLE plan value IDs exceed 32 bits");
            }
            total_nodes += static_cast<std::size_t>(count);
            std::vector<Node> &nodes = result.nodes_by_level_[level];
            nodes.resize(static_cast<std::size_t>(count));
            for (Node &node : nodes)
            {
                node.id = readScalar<std::uint32_t>(input);
                node.low = readScalar<std::uint32_t>(input);
                node.high = readScalar<std::uint32_t>(input);
            }
        }
        const std::size_t value_count =
            1 + result.terminals_.size() + total_nodes;
        if (value_count > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error(
                "compiled MLE plan value IDs exceed 32 bits");
        }
        result.value_count_ = value_count;

        std::vector<int> id_level(value_count, -2);
        id_level[0] = static_cast<int>(result.variable_count_);
        for (std::size_t terminal = 0;
             terminal < result.terminals_.size(); ++terminal)
        {
            id_level[terminal + 1] = static_cast<int>(result.variable_count_);
        }
        for (std::size_t level = 0; level < result.variable_count_; ++level)
        {
            for (const Node &node : result.nodes_by_level_[level])
            {
                if (node.id >= value_count || node.id <= result.terminals_.size() ||
                    id_level[node.id] != -2)
                {
                    throw std::runtime_error(
                        "compiled MLE plan has an invalid node ID");
                }
                id_level[node.id] = static_cast<int>(level);
            }
        }
        for (std::size_t level = 0; level < result.variable_count_; ++level)
        {
            for (const Node &node : result.nodes_by_level_[level])
            {
                if (node.low >= value_count || node.high >= value_count ||
                    id_level[node.low] <= static_cast<int>(level) ||
                    id_level[node.high] <= static_cast<int>(level))
                {
                    throw std::runtime_error(
                        "compiled MLE plan has an invalid child edge");
                }
            }
        }
        if (result.root_ >= value_count || id_level[result.root_] == -2 ||
            (result.root_ == 0 &&
             (!result.terminals_.empty() || total_nodes != 0)))
        {
            throw std::runtime_error(
                "compiled MLE plan has an invalid root");
        }
        return result;
    }

    EncodedAssignment encodeInterleaved(
        u32 first, int first_bits,
        u32 second, int second_bits)
    {
        (void)checkedVariableCount(first_bits, second_bits);
        EncodedAssignment result = 0;
        appendInterleaved(
            result, first, first_bits, second, second_bits, 0, 0);
        return result;
    }

    EncodedAssignment encodeInterleaved(
        u32 first, int first_bits,
        u32 second, int second_bits,
        u32 third, int third_bits)
    {
        (void)checkedVariableCount(first_bits, second_bits, third_bits);
        EncodedAssignment result = 0;
        appendInterleaved(
            result, first, first_bits, second, second_bits,
            third, third_bits);
        return result;
    }

    std::vector<Fr> interleavePoint(
        const std::vector<Fr> &first, int first_bits,
        const std::vector<Fr> &second, int second_bits)
    {
        const std::vector<Fr> empty;
        return interleavePointImpl(
            first, first_bits, second, second_bits, empty, 0);
    }

    std::vector<Fr> interleavePoint(
        const std::vector<Fr> &first, int first_bits,
        const std::vector<Fr> &second, int second_bits,
        const std::vector<Fr> &third, int third_bits)
    {
        return interleavePointImpl(
            first, first_bits, second, second_bits, third, third_bits);
    }

}
