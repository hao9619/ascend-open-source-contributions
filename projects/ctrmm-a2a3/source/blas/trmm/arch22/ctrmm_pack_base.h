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
#ifndef CTRMM_VEC_CHUNK
#define CTRMM_VEC_CHUNK 2304
#endif
constexpr uint32_t CHUNK = CTRMM_VEC_CHUNK;
static_assert(CHUNK >= 64 && CHUNK % 64 == 0, "vector chunk must be a positive multiple of 64");

__aicore__ inline uint32_t Clip(uint32_t remaining) { return remaining < CHUNK ? remaining : CHUNK; }

__aicore__ inline void VectorToStore()
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
}

__aicore__ inline void LoadToVector()
{
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}

__aicore__ inline void StoreVector(GlobalTensor<float>& dst, LocalTensor<float> src, uint64_t offset, uint32_t len)
{
    DataCopyExtParams copy{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(dst[offset], src, copy);
}

struct PackBuffers {
    TBuf<TPosition::VECCALC> stage, re, im, index, range;
};

__aicore__ inline void LoadComplexChunk(
    GlobalTensor<float>& src, LocalTensor<float> stage, LocalTensor<float> re, LocalTensor<float> im,
    LocalTensor<int32_t> index, const CtrmmPackTiling& t, uint32_t col, uint32_t row, uint32_t len)
{
    const uint64_t srcOffset = t.transpose ? (static_cast<uint64_t>(row) * t.sourceLd + col) * 2 :
                                             (static_cast<uint64_t>(col) * t.sourceLd + row) * 2;
    DataCopyExtParams copy{1, len * 8, 0, 0, 0};
    if (t.transpose) {
        copy.blockCount = static_cast<uint16_t>(len);
        copy.blockLen = 8;
        copy.srcStride = len == 1 ? 0 : (t.sourceLd - 1) * 8;
    }
    DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    DataCopyPad(stage, src[srcOffset], copy, pad);
    LoadToVector();
    CreateVecIndex(index, 0, len);
    PipeBarrier<PIPE_V>();
    Muls(index, index, static_cast<int32_t>(t.transpose ? 32 : 8), len);
    PipeBarrier<PIPE_V>();
    Gather(re, stage, index.ReinterpretCast<uint32_t>(), 0, len);
    PipeBarrier<PIPE_V>();
    Adds(index, index, static_cast<int32_t>(4), len);
    PipeBarrier<PIPE_V>();
    Gather(im, stage, index.ReinterpretCast<uint32_t>(), 0, len);
    PipeBarrier<PIPE_V>();
    if (t.conjugate) {
        Muls(im, im, -1.0f, len);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void LoadPackChunk(
    GlobalTensor<float>& src, PackBuffers& buffers, const CtrmmPackTiling& t, uint32_t col, uint32_t row, uint32_t len)
{
    LoadComplexChunk(
        src, buffers.stage.Get<float>(), buffers.re.Get<float>(), buffers.im.Get<float>(), buffers.index.Get<int32_t>(),
        t, col, row, len);
}

__aicore__ inline void CheckPackRange(PackBuffers& buffers, uint32_t len)
{
    auto scratch = buffers.stage.Get<float>();
    auto range = buffers.range.Get<float>();
    for (uint32_t part = 0; part < 2; ++part) {
        auto input = part ? buffers.im.Get<float>() : buffers.re.Get<float>();
        ShiftLeft(scratch.ReinterpretCast<uint32_t>(), input.ReinterpretCast<uint32_t>(), 1U, len);
        PipeBarrier<PIPE_V>();
        ShiftRight(scratch.ReinterpretCast<uint32_t>(), scratch.ReinterpretCast<uint32_t>(), 24U, len);
        PipeBarrier<PIPE_V>();
        Cast(scratch, scratch.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, len);
        PipeBarrier<PIPE_V>();
        Max(range, range, scratch, len);
        PipeBarrier<PIPE_V>();
    }
}

} // namespace
