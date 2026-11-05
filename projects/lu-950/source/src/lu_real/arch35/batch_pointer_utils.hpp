/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Included after BatchFence(). One lane stores one independent existing LU.
__aicore__ bool BatchMaskAll(LocalTensor<uint8_t> mask, int count)
{
    auto words = mask.ReinterpretCast<uint64_t>();
    for (int word = 0; word * 64 < count; ++word)
    {
        const int bits = count - word * 64 < 64 ? count - word * 64 : 64;
        const uint64_t expected = bits == 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1;
        if ((words.GetValue(word) & expected) != expected) return false;
    }
    return true;
}

__aicore__ bool BatchDensePointers(LocalTensor<uint32_t> raw, LocalTensor<int32_t> indices,
                                   LocalTensor<uint32_t> offsets, LocalTensor<int32_t> expected,
                                   LocalTensor<int32_t> actual, LocalTensor<uint8_t> mask, int group, int count,
                                   int bytes)
{
    const uint64_t first = raw.ReinterpretCast<uint64_t>().GetValue(0);
    if (!first ||
        static_cast<uint64_t>(static_cast<uint32_t>(first)) + static_cast<uint64_t>(count - 1) * bytes > 0xffffffffULL)
        return false;
    Muls(expected, indices, bytes, group);
    PipeBarrier<PIPE_V>();
    Adds(expected, expected, static_cast<int32_t>(first), group);
    Gather(actual, raw.ReinterpretCast<int32_t>(), offsets, 0, group);
    PipeBarrier<PIPE_V>();
    Compare(mask, actual, expected, CMPMODE::EQ, group);
    BatchFence<HardEvent::V_S>();
    if (!BatchMaskAll(mask, count)) return false;
    Gather(actual, raw.ReinterpretCast<int32_t>(), offsets, sizeof(uint32_t), group);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, actual, static_cast<int32_t>(first >> 32), CMPMODE::EQ, group);
    BatchFence<HardEvent::V_S>();
    return BatchMaskAll(mask, count);
}
