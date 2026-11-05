/* Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the repository root. */
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <acl/acl.h>
#include "extent.h"
#include "tiling/platform/platform_ascendc.h"
#include "detail/static_multiset/kernels.h"

namespace aclco {
namespace detail::multiset {
inline void Check(aclError code, const char* operation)
{
    if (code != ACL_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": ACL " + std::to_string(code));
}
struct Buffer {
    void* ptr = nullptr;
    explicit Buffer(uint64_t bytes)
    {
        if (bytes)
            Check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "allocate");
    }
    ~Buffer()
    {
        if (ptr)
            (void)aclrtFree(ptr);
    }
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;
};
inline uint64_t Bytes(uint64_t n, uint64_t element)
{
    if (element == 0)
        throw std::invalid_argument("element size must be positive");
    if (n > std::numeric_limits<std::size_t>::max() / element)
        throw std::overflow_error("buffer size overflow");
    return n * element;
}
inline void Require(void const* p)
{
    if (!p)
        throw std::invalid_argument("null nonempty buffer");
}
} // namespace detail::multiset

template <class Key>
class StaticMultiset {
    static_assert(std::is_same_v<Key, int32_t> || std::is_same_v<Key, int64_t>, "I32/I64 keys only");
    using E = aclco::Extent<std::size_t>;
    using P = detail::multiset::Params<Key>;
    struct State {
        uint64_t capacity, slots, size = 0;
        Key empty;
        detail::multiset::Buffer table, ordered, occupied;
        State(uint64_t cap, uint64_t slotsIn, Key emptyIn)
            : capacity(cap),
              slots(slotsIn),
              empty(emptyIn),
              table(detail::multiset::Bytes(slots, sizeof(StaticMultisetEntry<Key>))),
              ordered(detail::multiset::Bytes(cap, sizeof(Key))),
              occupied(detail::multiset::Bytes((slots + 31) / 32, 4))
        {}
    };
    std::unique_ptr<State> state_;
    static uint64_t Slots(uint64_t capacity)
    {
        if (!capacity)
            throw std::invalid_argument("capacity must be positive");
        if (capacity > (uint64_t(1) << 60))
            throw std::overflow_error("capacity too large");
        uint64_t n = 2, required = capacity * 2;
        while (n < required)
            n *= 2;
        detail::multiset::Bytes(n, sizeof(uint64_t));
        return n;
    }
    State& Get() const
    {
        if (!state_)
            throw std::logic_error("destroyed or moved-from StaticMultiset");
        return *state_;
    }
    P Params(uint64_t n) const
    {
        auto& s = Get();
        P p{};
        p.table = (StaticMultisetEntry<Key>*)s.table.ptr;
        p.occupied = (uint32_t*)s.occupied.ptr;
        p.ordered = (Key*)s.ordered.ptr;
        p.slots = s.slots;
        p.empty = s.empty;
        p.n = n;
        return p;
    }
    template <int Op, class Stencil = uint32_t, class Pred = detail::multiset::Always, bool Conditional = false>
    void Launch(P p, aclrtStream stream) const
    {
        if (!p.n)
            return;
        auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (!platform)
            throw std::runtime_error("missing Ascend platform");
        uint32_t cores = platform->GetCoreNumAiv();
        if (!cores)
            throw std::runtime_error("no AIV cores");
        uint32_t blocks = static_cast<uint32_t>(
            std::min<uint64_t>(cores, (p.n + detail::multiset::threads - 1) / detail::multiset::threads));
        if (p.n <= uint64_t(UINT32_MAX) - uint64_t(blocks) * detail::multiset::threads)
            detail::multiset::Kernel<Key, Op, Stencil, Pred, Conditional, uint32_t><<<blocks, 0, stream>>>(p);
        else
            detail::multiset::Kernel<Key, Op, Stencil, Pred, Conditional, uint64_t><<<blocks, 0, stream>>>(p);
    }
    static void Sync(aclrtStream stream) { detail::multiset::Check(aclrtSynchronizeStream(stream), "synchronize"); }
    static void Zero(void* ptr, uint64_t bytes, aclrtStream stream)
    {
        detail::multiset::Check(aclrtMemsetAsync(ptr, bytes, 0, bytes, stream), "zero");
    }
    static void Read(uint64_t* output, void* device, std::size_t n, aclrtStream stream)
    {
        detail::multiset::Check(aclrtMemcpyAsync(output, n * 8, device, n * 8, ACL_MEMCPY_DEVICE_TO_HOST, stream),
                                "read counter");
        Sync(stream);
    }
    uint64_t Scan(uint64_t* data, uint64_t n, aclrtStream stream) const
    {
        if (!n)
            return 0;
        uint64_t groups = (n + detail::multiset::scanTile - 1) / detail::multiset::scanTile;
        detail::multiset::Buffer totals(detail::multiset::Bytes(groups, 8));
        P p = Params(groups);
        p.offsets = data;
        p.totals = (uint64_t*)totals.ptr;
        p.limit = n;
        Launch<detail::multiset::ScanTiles>(p, stream);
        uint64_t total;
        if (groups == 1)
            Read(&total, totals.ptr, 1, stream);
        else {
            total = Scan((uint64_t*)totals.ptr, groups, stream);
            p.n = n;
            Launch<detail::multiset::AddOffsets>(p, stream);
            Sync(stream);
        }
        return total;
    }
    template <class Stencil, class Pred, bool Conditional>
    uint64_t InsertImpl(void* keys, Stencil* stencil, uint64_t n, aclrtStream stream)
    {
        auto& s = Get();
        if (!n)
            return 0;
        if constexpr (!Conditional) {
            if (!keys)
                return n;
        }
        detail::multiset::Require(keys);
        if constexpr (Conditional)
            detail::multiset::Require(stencil);
        detail::multiset::Bytes(n, sizeof(Key));
        detail::multiset::Buffer total(16);
        Zero(total.ptr, 16, stream);
        P p = Params(n);
        p.input = (Key*)keys;
        p.stencil = stencil;
        p.totals = (uint64_t*)total.ptr;
        Launch<detail::multiset::Eligible, Stencil, Pred, Conditional>(p, stream);
        uint64_t counts[2];
        Read(counts, total.ptr, 2, stream);
        uint64_t accepted = std::min(counts[0], s.capacity - s.size);
        if (!accepted)
            return counts[1];
        p.base = s.size;
        p.limit = accepted;
        Zero(total.ptr, 16, stream);
        if (counts[0] == n)
            Launch<detail::multiset::Insert>(p, stream);
        else {
            detail::multiset::Buffer offsets(detail::multiset::Bytes(n, 8));
            p.offsets = (uint64_t*)offsets.ptr;
            Launch<detail::multiset::Flags, Stencil, Pred, Conditional>(p, stream);
            Scan(p.offsets, n, stream);
            Launch<detail::multiset::InsertCompact, Stencil, Pred, Conditional>(p, stream);
            Sync(stream);
        }
        uint64_t failures;
        Read(&failures, total.ptr, 1, stream);
        if (failures)
            throw std::runtime_error("internal multiset index capacity invariant violated");
        s.size += accepted;
        return counts[1] - accepted;
    }
    template <int Op, class Stencil = uint32_t, class Pred = detail::multiset::Always, bool Conditional = false>
    void Query(void* keys, Stencil* stencil, void* output, uint64_t n, aclrtStream stream) const
    {
        Get();
        if (!n)
            return;
        detail::multiset::Require(keys);
        detail::multiset::Require(output);
        if constexpr (Conditional)
            detail::multiset::Require(stencil);
        detail::multiset::Bytes(n, sizeof(Key));
        if constexpr (Op == detail::multiset::CountEach || Op == detail::multiset::CountOuter)
            detail::multiset::Bytes(n, sizeof(uint64_t));
        P p = Params(n);
        p.input = (Key*)keys;
        p.output = output;
        p.stencil = stencil;
        Launch<Op, Stencil, Pred, Conditional>(p, stream);
        Sync(stream);
    }

public:
    using KeyType = Key;
    using SizeType = uint64_t;
    StaticMultiset(E capacity, Key empty, aclrtStream stream = nullptr)
        : state_(std::make_unique<State>(static_cast<std::size_t>(capacity), Slots(static_cast<std::size_t>(capacity)),
                                         empty))
    {
        Clear(stream);
    }
    StaticMultiset(StaticMultiset const&) = delete;
    StaticMultiset& operator=(StaticMultiset const&) = delete;
    StaticMultiset(StaticMultiset&&) noexcept = default;
    StaticMultiset& operator=(StaticMultiset&&) noexcept = default;
    ~StaticMultiset() = default;
    static StaticMultiset Create(E capacity, Key empty, aclrtStream stream = nullptr)
    {
        return StaticMultiset(capacity, empty, stream);
    }
    void Destroy() noexcept { state_.reset(); }
    // The caller must keep this container alive and serialize mutations with ref queries.
    StaticMultisetRef<Key> Ref() const
    {
        auto& s = Get();
        return {(StaticMultisetEntry<Key>*)s.table.ptr, s.slots, s.empty, (uint32_t*)s.occupied.ptr};
    }
    uint64_t Capacity() const { return Get().capacity; }
    uint64_t Size(aclrtStream stream = nullptr) const
    {
        auto& s = Get();
        Sync(stream);
        return s.size;
    }
    void Clear(aclrtStream stream = nullptr)
    {
        auto& s = Get();
        Launch<detail::multiset::Init>(Params(s.slots), stream);
        Sync(stream);
        s.size = 0;
    }
    uint64_t Insert(void* keys, E n, aclrtStream stream)
    {
        return InsertImpl<uint32_t, detail::multiset::Always, false>(keys, nullptr, n, stream);
    }
    template <class Stencil, class Pred>
    uint64_t InsertIf(void* keys, Stencil* stencil, E n, aclrtStream stream)
    {
        return InsertImpl<Stencil, Pred, true>(keys, stencil, n, stream);
    }
    void Contains(void* keys, void* output, E n, aclrtStream stream) const
    {
        Query<detail::multiset::Contains>(keys, (uint32_t*)nullptr, output, n, stream);
    }
    void Find(void* keys, void* output, E n, aclrtStream stream) const
    {
        Query<detail::multiset::Find>(keys, (uint32_t*)nullptr, output, n, stream);
    }
    template <class Stencil, class Pred>
    void ContainsIf(void* keys, Stencil* stencil, void* output, E n, aclrtStream stream) const
    {
        Query<detail::multiset::Contains, Stencil, Pred, true>(keys, stencil, output, n, stream);
    }
    template <class Stencil, class Pred>
    void FindIf(void* keys, Stencil* stencil, void* output, E n, aclrtStream stream) const
    {
        Query<detail::multiset::Find, Stencil, Pred, true>(keys, stencil, output, n, stream);
    }
    void CountEach(void* keys, void* output, E n, aclrtStream stream) const
    {
        Query<detail::multiset::CountEach>(keys, (uint32_t*)nullptr, output, n, stream);
    }
    void CountEachOuter(void* keys, void* output, E n, aclrtStream stream) const
    {
        Query<detail::multiset::CountOuter>(keys, (uint32_t*)nullptr, output, n, stream);
    }
    uint64_t Count(void* keys, E n, aclrtStream stream) const
    {
        Get();
        if (!static_cast<std::size_t>(n))
            return 0;
        detail::multiset::Require(keys);
        detail::multiset::Bytes(static_cast<std::size_t>(n), sizeof(Key));
        if (Get().size && static_cast<std::size_t>(n) > std::numeric_limits<uint64_t>::max() / Get().size)
            throw std::overflow_error("potential Count overflow");
        detail::multiset::Buffer sum(8);
        Zero(sum.ptr, 8, stream);
        P p = Params(n);
        p.input = (Key*)keys;
        p.totals = (uint64_t*)sum.ptr;
        Launch<detail::multiset::Count>(p, stream);
        uint64_t result;
        Read(&result, sum.ptr, 1, stream);
        return result;
    }
    uint64_t Retrieve(void* keys, E n, void* probes, void* matches, E outputCapacity, aclrtStream stream) const
    {
        Get();
        uint64_t count = static_cast<std::size_t>(n);
        if (!count)
            return 0;
        detail::multiset::Require(keys);
        if (Get().size && count > std::numeric_limits<uint64_t>::max() / Get().size)
            throw std::overflow_error("potential Retrieve count overflow");
        detail::multiset::Buffer offsets(detail::multiset::Bytes(count, 8));
        CountEach(keys, offsets.ptr, n, stream);
        uint64_t total = Scan((uint64_t*)offsets.ptr, count, stream);
        if (total > static_cast<std::size_t>(outputCapacity))
            throw std::length_error("Retrieve output capacity too small");
        if (!total)
            return 0;
        detail::multiset::Bytes(total, sizeof(Key));
        detail::multiset::Require(probes);
        detail::multiset::Require(matches);
        P p = Params(count);
        p.input = (Key*)keys;
        p.output = probes;
        p.matches = (Key*)matches;
        p.offsets = (uint64_t*)offsets.ptr;
        p.limit = total;
        Launch<detail::multiset::Retrieve>(p, stream);
        Sync(stream);
        return total;
    }
    uint64_t RetrieveAll(void* output, E outputCapacity, aclrtStream stream) const
    {
        auto& s = Get();
        if (s.size > static_cast<std::size_t>(outputCapacity))
            throw std::length_error("RetrieveAll output capacity too small");
        if (!s.size)
            return 0;
        detail::multiset::Require(output);
        uint64_t bytes = detail::multiset::Bytes(s.size, sizeof(Key));
        detail::multiset::Check(
            aclrtMemcpyAsync(output, bytes, s.ordered.ptr, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream), "retrieve all");
        Sync(stream);
        return s.size;
    }
};
} // namespace aclco
