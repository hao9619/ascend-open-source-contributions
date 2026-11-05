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
__aicore__ inline void CombineChunk(
    CombineBuffers& buffers, const CtrmmCombineTiling& t, uint32_t col, uint32_t row, uint32_t cap = CHUNK)
{
    auto data = buffers.data.Get<float>();
    auto pr = data;
    auto pi = data[CHUNK];
    auto x = data[2 * CHUNK];
    auto y = data[3 * CHUNK];
    auto pair = data[4 * CHUNK];
    auto stage = data[6 * CHUNK];
    auto index = buffers.index.Get<int32_t>();
    const uint32_t len = Clip(t.m - row) < cap ? Clip(t.m - row) : cap;
    const uint64_t offset = static_cast<uint64_t>(col) * t.tempLd + row;
    LoadVector(pr, buffers.rrGm, offset, len);
    LoadVector(pi, buffers.riGm, offset, len);
    LoadVector(x, buffers.iiGm, offset, len);
    if (t.fourProduct)
        LoadVector(y, buffers.irGm, offset, len);
    LoadToVector();
    if (t.fourProduct) {
        Add(pi, pi, y, len);
    } else {
        Sub(pi, pi, pr, len);
        PipeBarrier<PIPE_V>();
        Sub(pi, pi, x, len);
    }
    Sub(pr, pr, x, len);
    PipeBarrier<PIPE_V>();
    if (t.alphaReal == 1.0f && t.alphaImag == 0.0f) {
        pair = pr;
    } else {
        Muls(pair, pr, t.alphaReal, len);
        Muls(pair[CHUNK], pi, t.alphaReal, len);
        Muls(x, pi, t.alphaImag, len);
        Muls(y, pr, t.alphaImag, len);
        PipeBarrier<PIPE_V>();
        Sub(pair, pair, x, len);
        Add(pair[CHUNK], pair[CHUNK], y, len);
        PipeBarrier<PIPE_V>();
    }
    Gather(stage, pair, index.ReinterpretCast<uint32_t>(), 0, len * 2);
    PipeBarrier<PIPE_V>();
    VectorToStore();
    StoreVector(buffers.outGm, stage, (static_cast<uint64_t>(col) * t.ldc + row) * 2, len * 2);
    cblas3::SyncAfterChunkStore();
}
} // namespace
