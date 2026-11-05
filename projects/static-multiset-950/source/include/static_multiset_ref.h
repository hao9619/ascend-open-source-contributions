/* Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the repository root. */
#pragma once
#include <cstdint>
#include "kernel_operator.h"
#include "macros.h"
#include "hash_functions.h"
#include "utility/atomic_cas_wrap.h"

namespace aclco {
// Non-owning SIMT index reference. Query only after the insertion stream completes.
template <class Key>
struct StaticMultisetEntry {
    Key key;
};

template <class Key>
struct StaticMultisetRef {
    __gm__ StaticMultisetEntry<Key>* entries;
    uint64_t slots;
    Key empty;
    __gm__ uint32_t* occupied;

    // Preserve a warp-sized integer neighborhood while mixing all higher bits.
    // The probe permutation visits every slot, including other low-bit lanes
    // after a lane's subtable fills; skewed keys retain full logical capacity.
    template <class Index>
    COLLECTION_SIMT_DEVICE Index Home(Key key) const
    {
        if (slots < 32)
            return static_cast<Index>(xxhash_32<Key>{}(key)) & static_cast<Index>(slots - 1);
        uint64_t bits;
        if constexpr (sizeof(Key) == 4)
            bits = static_cast<uint32_t>(key);
        else
            bits = static_cast<uint64_t>(key);
        Key high = static_cast<Key>(bits >> 5);
        uint32_t hash = xxhash_32<Key>{}(high);
        return (static_cast<Index>(hash) & static_cast<Index>(slots / 32 - 1)) * Index(32) +
               static_cast<Index>((bits ^ (hash >> 27)) & 31);
    }
    template <class Index>
    COLLECTION_SIMT_DEVICE Index Next(Index slot, Index probe) const
    {
        if (slots < 32)
            return (slot + 1) & static_cast<Index>(slots - 1);
        slot = (slot + Index(32)) & static_cast<Index>(slots - 1);
        if (((probe + 1) & static_cast<Index>(slots / 32 - 1)) == 0)
            slot = (slot + 1) & static_cast<Index>(slots - 1);
        return slot;
    }

    template <class Index>
    COLLECTION_SIMT_DEVICE uint64_t FindSlotImpl(Key key) const
    {
        if (key == empty)
            return slots;
        Index slot = Home<Index>(key);
        for (Index probe = 0; probe < static_cast<Index>(slots); ++probe) {
            Key value = entries[slot].key;
            if (value == key)
                return slot;
            if (value == empty)
                return slots;
            slot = Next(slot, probe);
        }
        return slots;
    }
    template <class Index>
    COLLECTION_SIMT_DEVICE uint64_t CountImpl(Key key) const
    {
        if (key == empty)
            return 0;
        Index slot = Home<Index>(key);
        uint64_t result = 0;
        for (Index probe = 0; probe < static_cast<Index>(slots); ++probe) {
            Key value = entries[slot].key;
            if (value == empty)
                break;
            result += value == key;
            slot = Next(slot, probe);
        }
        return result;
    }
    COLLECTION_SIMT_DEVICE uint64_t FindSlot(Key key) const
    {
        if (slots <= uint64_t(UINT32_MAX))
            return FindSlotImpl<uint32_t>(key);
        return FindSlotImpl<uint64_t>(key);
    }
    COLLECTION_SIMT_DEVICE uint64_t Count(Key key) const
    {
        if (slots <= uint64_t(UINT32_MAX))
            return CountImpl<uint32_t>(key);
        return CountImpl<uint64_t>(key);
    }
    COLLECTION_SIMT_DEVICE bool AddToIndex(Key key) const
    {
        if (slots <= uint64_t(UINT32_MAX))
            return AddToIndexImpl<uint32_t>(key);
        return AddToIndexImpl<uint64_t>(key);
    }
    COLLECTION_SIMT_DEVICE bool Contains(Key key) const { return FindSlot(key) != slots; }
    COLLECTION_SIMT_DEVICE Key Find(Key key) const { return Contains(key) ? key : empty; }
    // Internal bulk-insertion primitive; ordered storage is managed by the host container.
    template <class Index>
    COLLECTION_SIMT_DEVICE bool AddToIndexImpl(Key key) const
    {
        if (key == empty)
            return false;
        Index slot = Home<Index>(key);
        for (Index probe = 0; probe < static_cast<Index>(slots); ++probe) {
            uint32_t bit = uint32_t(1) << (slot & 31);
            uint32_t old = AscendC::Simt::AtomicOr(occupied + slot / 32, bit);
            if (!(old & bit)) {
                entries[slot].key = key;
                return true;
            }
            // Equal keys occupy distinct slots: duplicates must never be deduplicated.
            slot = Next(slot, probe);
        }
        return false;
    }
};
} // namespace aclco
