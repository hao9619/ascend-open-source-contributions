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
constexpr uint32_t CLASSIFY_CHUNK = 4096;

__aicore__ inline void ClassifyChunk(
    LocalTensor<float> data, GlobalTensor<float>& input, uint64_t offset, uint32_t count)
{
    if (count < CLASSIFY_CHUNK) {
        Duplicate(data, 0.0f, CLASSIFY_CHUNK);
        PipeBarrier<PIPE_ALL>();
    }
    LoadVector(data, input, offset, count);
    LoadToVector();
    ShiftLeft(data.ReinterpretCast<uint32_t>(), data.ReinterpretCast<uint32_t>(), 1U, CLASSIFY_CHUNK);
    PipeBarrier<PIPE_V>();
    ShiftRight(data.ReinterpretCast<uint32_t>(), data.ReinterpretCast<uint32_t>(), 24U, CLASSIFY_CHUNK);
    PipeBarrier<PIPE_V>();
    Cast(data, data.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, CLASSIFY_CHUNK);
    PipeBarrier<PIPE_V>();
    Max(data[CLASSIFY_CHUNK], data[CLASSIFY_CHUNK], data, CLASSIFY_CHUNK);
    PipeBarrier<PIPE_ALL>();
}
} // namespace

extern "C" __global__ __aicore__ void ctrmm_classify_kernel(GM_ADDR input, GM_ADDR flags, uint64_t count)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, (3 * CLASSIFY_CHUNK + 8) * sizeof(float));
    auto data = storage.Get<float>();
    auto result = data[3 * CLASSIFY_CHUNK];
    GlobalTensor<float> inGm, outGm;
    inGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(flags));
    Duplicate(data[CLASSIFY_CHUNK], 0.0f, CLASSIFY_CHUNK);
    PipeBarrier<PIPE_V>();
    const uint64_t stride = static_cast<uint64_t>(GetBlockNum()) * CLASSIFY_CHUNK;
    for (uint64_t offset = static_cast<uint64_t>(GetBlockIdx()) * CLASSIFY_CHUNK; offset < count; offset += stride) {
        const uint32_t len = count - offset < CLASSIFY_CHUNK ? count - offset : CLASSIFY_CHUNK;
        ClassifyChunk(data, inGm, offset, len);
    }
    ReduceMax(result, data[CLASSIFY_CHUNK], data[2 * CLASSIFY_CHUNK], CLASSIFY_CHUNK, false);
    PipeBarrier<PIPE_ALL>();
    // Exponent >= 143 includes |x| >= 2^16, Inf, and every NaN payload.
    const float exceptional = result.GetValue(0) >= 143.0f ? 1.0f : 0.0f;
    Duplicate(result, exceptional, 8);
    PipeBarrier<PIPE_V>();
    VectorToStore();
    StoreVector(outGm, result, static_cast<uint64_t>(GetBlockIdx()) * 8, 8);
    cblas3::SyncAfterChunkStore();
}

void ctrmm_classify_do(uint32_t blocks, void* stream, uint8_t* input, uint8_t* flags, uint64_t count)
{
    ctrmm_classify_kernel<<<blocks, nullptr, stream>>>(input, flags, count);
}
