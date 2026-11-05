/* Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the repository root. */
#pragma once
#include "static_multiset_ref.h"
#include "simt_api/device_warp_functions.h"

namespace aclco::detail::multiset {
constexpr uint32_t threads = 1024;
constexpr uint64_t scanTile = 256;
struct Always {
    COLLECTION_SIMT_DEVICE bool operator()(uint32_t) const { return true; }
};
template <class Key>
struct Params {
    __gm__ StaticMultisetEntry<Key>* table;
    __gm__ uint32_t* occupied;
    __gm__ Key* ordered;
    __gm__ Key* input;
    __gm__ void* output;
    __gm__ Key* matches;
    __gm__ void* stencil;
    __gm__ uint64_t* offsets;
    __gm__ uint64_t* totals;
    uint64_t n;
    uint64_t slots;
    uint64_t base;
    uint64_t limit;
    Key empty;
};
enum Operation {
    Init,
    Eligible,
    Flags,
    Insert,
    InsertCompact,
    Contains,
    Find,
    CountEach,
    CountOuter,
    Count,
    Retrieve,
    ScanTiles,
    AddOffsets
};

// Every helper containing a warp collective is called by all lanes in that warp.
// Inactive tail lanes still participate; do not move these calls under active checks.
COLLECTION_SIMT_DEVICE uint64_t WarpSum(uint64_t value)
{
    for (uint32_t delta = 16; delta; delta /= 2) {
        value += asc_shfl_down(value, delta, 32);
    }
    return value;
}

COLLECTION_SIMT_DEVICE uint32_t WarpBits(uint32_t value)
{
    for (uint32_t delta = 16; delta; delta /= 2) {
        value |= asc_shfl_down(value, delta, 32);
    }
    return asc_shfl(value, 0, 32);
}

template <class Key, class Index>
COLLECTION_SIMT_DEVICE void ScanTileOffsets(Params<Key> p, Index tid, Index stride, uint32_t lane)
{
    for (uint64_t tile = tid / 32; tile < p.n; tile += stride / 32) {
        uint64_t carry = 0;
        for (uint32_t chunk = 0; chunk < scanTile / 32; ++chunk) {
            uint64_t index = tile * scanTile + chunk * 32 + lane;
            uint64_t value = index < p.limit ? p.offsets[index] : 0;
            uint64_t original = value;
            for (uint32_t delta = 1; delta < 32; delta *= 2) {
                uint64_t previous = asc_shfl_up(value, delta, 32);
                if (lane >= delta)
                    value += previous;
            }
            if (index < p.limit)
                p.offsets[index] = carry + value - original;
            carry += asc_shfl(value, 31, 32);
        }
        if (lane == 31)
            p.totals[tile] = carry;
    }
}

COLLECTION_SIMT_DEVICE bool ReserveSlot(__gm__ uint32_t* occupied, uint32_t slot, uint32_t lane, bool active)
{
    uint32_t bit = uint32_t(1) << (slot & 31);
    bool pending = active, success = false;
    // Coalesce at most two large groups; dispersed/duplicate bits use scalar atomics.
    for (uint32_t group = 0; group < 2; ++group) {
        uint32_t pendingMask = asc_ballot(pending);
        if (!pendingMask)
            break;
        uint32_t leader = __builtin_ctz(pendingMask);
        uint32_t bucket = asc_shfl(slot / 32, leader, 32);
        bool member = pending && slot / 32 == bucket;
        uint32_t memberMask = asc_ballot(member);
        if (__builtin_popcount(memberMask) < 4)
            break;
        uint32_t bits = WarpBits(member ? bit : 0);
        if (__builtin_popcount(bits) != __builtin_popcount(memberMask))
            break;
        uint32_t old = 0;
        if (lane == leader)
            old = AscendC::Simt::AtomicOr(occupied + bucket, bits);
        old = asc_shfl(old, leader, 32);
        if (member) {
            success = !(old & bit);
            pending = false;
        }
    }
    if (pending)
        success = !(AscendC::Simt::AtomicOr(occupied + slot / 32, bit) & bit);
    return success;
}

template <class Key>
COLLECTION_SIMT_DEVICE uint64_t InsertWarpKey(Params<Key> p, StaticMultisetRef<Key> ref, Key key, bool active,
                                              uint32_t lane)
{
    uint32_t slot = ref.template Home<uint32_t>(key), probes = 0;
    uint64_t failures = 0;
    while (asc_ballot(active)) {
        bool success = ReserveSlot(p.occupied, slot, lane, active);
        if (active) {
            if (success) {
                p.table[slot].key = key;
                active = false;
            } else {
                ++probes;
                if (probes == p.slots) {
                    ++failures;
                    active = false;
                } else
                    slot = ref.Next(slot, probes - 1);
            }
        }
    }
    return failures;
}

template <class Key, int Op, class Stencil, class Pred, bool Conditional, class Index>
COLLECTION_SIMT_DEVICE void InsertWarp(Params<Key> p, Index tid, Index stride, uint32_t lane)
{
    StaticMultisetRef<Key> ref{p.table, p.slots, p.empty, p.occupied};
    uint64_t sum = 0;
    for (uint64_t base = uint64_t(tid / 32) * 32; base < p.n; base += stride) {
        uint64_t i = base + lane;
        Key key = i < p.n ? p.input[i] : p.empty;
        bool active = i < p.n && key != p.empty;
        if constexpr (Conditional) {
            if (active)
                active = Pred{}(((__gm__ Stencil*)p.stencil)[i]);
        }
        uint64_t pos = i;
        if constexpr (Op == InsertCompact)
            pos = i < p.n ? p.offsets[i] : 0;
        active = active && pos < p.limit;
        if (active)
            p.ordered[p.base + pos] = key;
        sum += InsertWarpKey(p, ref, key, active, lane);
    }
    sum = WarpSum(sum);
    if (lane == 0 && sum)
        AscendC::Simt::AtomicAdd(p.totals, sum);
}

template <class Key, int Op>
COLLECTION_SIMT_DEVICE uint64_t QueryKey(Params<Key> p, StaticMultisetRef<Key> ref, uint64_t i, Key key, bool selected)
{
    if constexpr (Op == Contains) {
        ((__gm__ uint8_t*)p.output)[i] = selected && ref.Contains(key);
    } else if constexpr (Op == Find) {
        ((__gm__ Key*)p.output)[i] = selected ? ref.Find(key) : p.empty;
    } else if constexpr (Op == CountEach || Op == CountOuter) {
        uint64_t count = ref.Count(key);
        if constexpr (Op == CountOuter) {
            if (!count)
                count = 1;
        }
        ((__gm__ uint64_t*)p.output)[i] = count;
    } else if constexpr (Op == Count) {
        return ref.Count(key);
    } else if constexpr (Op == Retrieve) {
        uint64_t end = (i + 1 < p.n) ? p.offsets[i + 1] : p.limit;
        for (uint64_t j = p.offsets[i]; j < end; ++j) {
            ((__gm__ Key*)p.output)[j] = key;
            p.matches[j] = key;
        }
    }
    return 0;
}

template <class Key, int Op>
COLLECTION_SIMT_DEVICE uint64_t ProcessKey(Params<Key> p, StaticMultisetRef<Key> ref, uint64_t i, Key key,
                                           bool selected)
{
    if constexpr (Op == Eligible) {
        return selected && key != p.empty;
    } else if constexpr (Op == Flags) {
        p.offsets[i] = selected && key != p.empty;
    } else if constexpr (Op == Insert || Op == InsertCompact) {
        uint64_t pos = i;
        if constexpr (Op == InsertCompact)
            pos = p.offsets[i];
        if (selected && key != p.empty && pos < p.limit) {
            p.ordered[p.base + pos] = key;
            return !ref.AddToIndex(key);
        }
    } else {
        return QueryKey<Key, Op>(p, ref, i, key, selected);
    }
    return 0;
}

template <class Key, int Op>
COLLECTION_SIMT_DEVICE void ReduceCounters(Params<Key> p, uint64_t sum, uint64_t selectedSum, uint32_t lane)
{
    if constexpr (Op == Eligible || Op == Count || Op == Insert || Op == InsertCompact) {
        // Keep the two reductions interleaved exactly as in the validated kernel.
        for (uint32_t delta = 16; delta; delta /= 2) {
            sum += asc_shfl_down(sum, delta, 32);
            if constexpr (Op == Eligible)
                selectedSum += asc_shfl_down(selectedSum, delta, 32);
        }
        if (lane == 0) {
            if (sum)
                AscendC::Simt::AtomicAdd(p.totals, sum);
            if constexpr (Op == Eligible) {
                if (selectedSum)
                    AscendC::Simt::AtomicAdd(p.totals + 1, selectedSum);
            }
        }
    }
}

template <class Key, int Op, class Stencil, class Pred, bool Conditional, class Index>
COLLECTION_SIMT_DEVICE void ProcessElements(Params<Key> p, Index tid, Index stride, uint32_t lane)
{
    StaticMultisetRef<Key> ref{p.table, p.slots, p.empty, p.occupied};
    uint64_t sum = 0, selectedSum = 0;
    for (Index i = tid; i < static_cast<Index>(p.n); i += stride) {
        if constexpr (Op == Init) {
            p.table[i].key = p.empty;
            if (i < (p.slots + 31) / 32)
                p.occupied[i] = 0;
        } else if constexpr (Op == AddOffsets) {
            p.offsets[i] += p.totals[i / scanTile];
        } else {
            Key key = p.input[i];
            bool selected = true;
            if constexpr (Conditional)
                selected = Pred{}(((__gm__ Stencil*)p.stencil)[i]);
            if constexpr (Op == Eligible)
                selectedSum += selected;
            sum += ProcessKey<Key, Op>(p, ref, i, key, selected);
        }
    }
    ReduceCounters<Key, Op>(p, sum, selectedSum, lane);
}

template <class Key, int Op, class Stencil = uint32_t, class Pred = Always, bool Conditional = false,
          class Index = uint64_t>
COLLECTION_SIMT_VF LAUNCH_BOUND(threads) inline void Simt(Params<Key> p)
{
    Index tid = Index(AscendC::Simt::GetBlockIdx()) * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    Index stride = Index(AscendC::Simt::GetBlockNum()) * AscendC::Simt::GetThreadNum();
    uint32_t lane = AscendC::Simt::GetThreadIdx() % 32;
    if constexpr (Op == ScanTiles) {
        ScanTileOffsets(p, tid, stride, lane);
        return;
    }
    if constexpr (Op == Insert || Op == InsertCompact) {
        if (p.slots <= UINT32_MAX) {
            InsertWarp<Key, Op, Stencil, Pred, Conditional>(p, tid, stride, lane);
            return;
        }
    }
    ProcessElements<Key, Op, Stencil, Pred, Conditional>(p, tid, stride, lane);
}

template <class Key, int Op, class Stencil = uint32_t, class Pred = Always, bool Conditional = false,
          class Index = uint64_t>
COLLECTION_AIV_GLOBAL void Kernel(Params<Key> p)
{
    AscendC::Simt::VF_CALL<Simt<Key, Op, Stencil, Pred, Conditional, Index>>(AscendC::Simt::Dim3{threads}, p);
}
} // namespace aclco::detail::multiset
