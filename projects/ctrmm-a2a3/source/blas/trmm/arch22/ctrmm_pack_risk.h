/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
namespace {
__aicore__ inline void StorePackRisk(PackBuffers& buffers, uint64_t address)
{
    if (!address)
        return;
    GlobalTensor<float> flags;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(address));
    auto data = buffers.stage.Get<float>();
#ifdef CTRMM_COMPACT_REDUCTION
    // CheckPackRange writes at most CHUNK lanes; the second half remains zero.
    if constexpr (CHUNK <= 4096) {
        WholeReduceMax(
            data[2 * CHUNK], buffers.range.Get<float>(), 64, CHUNK / 64, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(data[4 * CHUNK], data[2 * CHUNK], CHUNK / 64, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    } else {
        ReduceMax(data[4 * CHUNK], buffers.range.Get<float>(), data[2 * CHUNK], CHUNK, false);
    }
#else
    ReduceMax(data[4 * CHUNK], buffers.range.Get<float>(), data[2 * CHUNK], CHUNK, false);
#endif
    PipeBarrier<PIPE_ALL>();
    const bool risky = data.GetValue(4 * CHUNK) >= 131.0f;
    Duplicate(data, risky ? 1.0f : 0.0f, 8);
    PipeBarrier<PIPE_V>();
    VectorToStore();
    StoreVector(flags, data, GetBlockIdx() * 8, 8);
    cblas3::SyncAfterChunkStore();
}
} // namespace
