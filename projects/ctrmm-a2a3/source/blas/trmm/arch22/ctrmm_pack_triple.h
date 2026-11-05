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
__aicore__ inline void StorePackedTriple(
    GlobalTensor<float>& real, GlobalTensor<float>& imag, GlobalTensor<float>& sum, PackBuffers& buffers,
    uint64_t offset, uint32_t count)
{
    auto x = buffers.re.Get<float>();
    auto y = buffers.im.Get<float>();
    auto z = buffers.stage.Get<float>();
    Add(z, x, y, count);
    PipeBarrier<PIPE_V>();
    VectorToStore();
    StoreVector(real, x, offset, count);
    StoreVector(imag, y, offset, count);
    StoreVector(sum, z, offset, count);
    cblas3::SyncAfterChunkStore();
}

__aicore__ inline void PackSmallTriangle(
    GlobalTensor<float>& input, GlobalTensor<float>& real, GlobalTensor<float>& imag, GlobalTensor<float>& sum,
    PackBuffers& buffers, const CtrmmPackTiling& t, uint32_t col)
{
    auto stage = buffers.stage.Get<float>();
    auto x = buffers.re.Get<float>();
    auto y = buffers.im.Get<float>();
    auto index = buffers.index.Get<int32_t>();
    Duplicate(stage, 0.0f, 2 * t.packedLd);
    PipeBarrier<PIPE_ALL>();
    const uint32_t begin = t.upper ? 0 : col + t.unit;
    const uint32_t end = t.upper ? col + 1 - t.unit : t.rows;
    if (begin < end) {
        const uint32_t left = begin % 4;
        const uint32_t right = (4 - (end - begin + left) % 4) % 4;
        DataCopyExtParams copy{1, (end - begin) * 8, 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, static_cast<uint8_t>(left * 2), static_cast<uint8_t>(right * 2), 0.0f};
        DataCopyPad(stage[(begin - left) * 2], input[(uint64_t(col) * t.sourceLd + begin) * 2], copy, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    if (t.unit)
        stage.SetValue(col * 2, 1.0f);
    PipeBarrier<PIPE_ALL>();
    CreateVecIndex(index, 0, t.packedLd);
    PipeBarrier<PIPE_V>();
    Muls(index, index, 8, t.packedLd);
    PipeBarrier<PIPE_V>();
    Gather(x, stage, index.ReinterpretCast<uint32_t>(), 0, t.packedLd);
    Adds(index, index, 4, t.packedLd);
    PipeBarrier<PIPE_V>();
    Gather(y, stage, index.ReinterpretCast<uint32_t>(), 0, t.packedLd);
    PipeBarrier<PIPE_V>();
    if (t.conjugate) {
        Muls(y, y, -1.0f, t.packedLd);
        PipeBarrier<PIPE_V>();
    }
    if (t.riskFlags)
        CheckPackRange(buffers, t.packedLd);
    StorePackedTriple(real, imag, sum, buffers, uint64_t(col) * t.packedLd, t.packedLd);
}

__aicore__ inline void PackDensePadded(
    GlobalTensor<float>& input, GlobalTensor<float>& real, GlobalTensor<float>& imag, GlobalTensor<float>& sum,
    PackBuffers& buffers, const CtrmmPackTiling& t, uint32_t col)
{
    auto x = buffers.re.Get<float>();
    auto y = buffers.im.Get<float>();
    for (uint32_t row = 0; row < t.packedLd; row += CHUNK) {
        const uint32_t count = Clip(t.packedLd - row);
        Duplicate(x, 0.0f, count);
        Duplicate(y, 0.0f, count);
        PipeBarrier<PIPE_ALL>();
        if (row < t.rows) {
            const uint32_t valid = t.rows - row < count ? t.rows - row : count;
            LoadPackChunk(input, buffers, t, col, row, valid);
            if (t.riskFlags)
                CheckPackRange(buffers, valid);
        }
        StorePackedTriple(real, imag, sum, buffers, uint64_t(col) * t.packedLd + row, count);
    }
}

__aicore__ inline void PackTripleColumn(
    GlobalTensor<float>& input, GlobalTensor<float>& real, GlobalTensor<float>& imag, GlobalTensor<float>& sum,
    PackBuffers& buffers, const CtrmmPackTiling& t, uint32_t col)
{
    auto x = buffers.re.Get<float>();
    auto y = buffers.im.Get<float>();
    const uint64_t base = uint64_t(col) * t.packedLd;
    Duplicate(x, 0.0f, CHUNK);
    Duplicate(y, 0.0f, CHUNK);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = t.triangular ? 0 : t.rows; row < t.packedLd; row += CHUNK) {
        StorePackedTriple(real, imag, sum, buffers, base + row, Clip(t.packedLd - row));
    }
    const bool upper = t.upper != t.transpose;
    const uint32_t begin = t.triangular && !upper ? col + t.unit : 0;
    const uint32_t end = t.triangular && upper ? col + 1 - t.unit : t.rows;
    const uint32_t step = t.transpose && t.sourceLd > UINT32_MAX / 8 ? 1 : CHUNK;
    for (uint32_t row = begin; row < end; row += step) {
        const uint32_t len = step < end - row ? step : end - row;
        LoadPackChunk(input, buffers, t, col, row, len);
        if (t.riskFlags)
            CheckPackRange(buffers, len);
        StorePackedTriple(real, imag, sum, buffers, base + row, len);
    }
    if (t.unit) {
        Duplicate(x, 1.0f, 8);
        Duplicate(y, 0.0f, 8);
        PipeBarrier<PIPE_V>();
        StorePackedTriple(real, imag, sum, buffers, base + col, 1);
    }
}

__aicore__ inline void PackColumnBatch(
    GlobalTensor<float>& input, GlobalTensor<float>& real, GlobalTensor<float>& imag, GlobalTensor<float>& sum,
    PackBuffers& buffers, const CtrmmPackTiling& t)
{
    auto stage = buffers.stage.Get<float>();
    auto x = buffers.re.Get<float>();
    auto y = buffers.im.Get<float>();
    auto index = buffers.index.Get<int32_t>();
    const uint32_t width = CHUNK / t.packedLd;
    for (uint32_t col = GetBlockIdx() * width; col < t.cols; col += GetBlockNum() * width) {
        const uint32_t count = t.cols - col < width ? t.cols - col : width;
        const uint32_t len = count * t.packedLd;
        DataCopyExtParams copy{static_cast<uint16_t>(count), t.rows * 8, 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>((t.packedLd - t.rows) * 2), 0.0f};
        DataCopyPad(stage, input[uint64_t(col) * t.sourceLd * 2], copy, pad);
        LoadToVector();
        CreateVecIndex(index, 0, len);
        PipeBarrier<PIPE_V>();
        Muls(index, index, 8, len);
        PipeBarrier<PIPE_V>();
        Gather(x, stage, index.ReinterpretCast<uint32_t>(), 0, len);
        Adds(index, index, 4, len);
        PipeBarrier<PIPE_V>();
        Gather(y, stage, index.ReinterpretCast<uint32_t>(), 0, len);
        PipeBarrier<PIPE_V>();
        if (t.riskFlags)
            CheckPackRange(buffers, len);
        StorePackedTriple(real, imag, sum, buffers, uint64_t(col) * t.packedLd, len);
    }
}

__aicore__ inline bool UsePackColumnBatch(const CtrmmPackTiling& t)
{
    return !t.triangular && !t.transpose && t.sourceLd == t.rows && t.packedLd <= CHUNK && t.packedLd > t.rows &&
           t.packedLd - t.rows < 4;
}

__aicore__ inline void PackAndSum(
    GM_ADDR src, GM_ADDR re, GM_ADDR im, GM_ADDR sum, PackBuffers& buffers, const CtrmmPackTiling& t)
{
    GlobalTensor<float> input, real, imag, output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
    real.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(re));
    imag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(im));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum));
    if (UsePackColumnBatch(t)) {
        PackColumnBatch(input, real, imag, output, buffers, t);
        return;
    }
    const bool flat = !t.triangular && !t.transpose && t.rows == t.sourceLd && t.rows == t.packedLd;
    if (flat && uint64_t(t.rows) * t.cols <= UINT32_MAX) {
        CtrmmPackTiling linear = t;
        linear.rows = t.rows * t.cols;
        for (uint32_t row = GetBlockIdx() * CHUNK; row < linear.rows; row += GetBlockNum() * CHUNK) {
            const uint32_t len = Clip(linear.rows - row);
            LoadPackChunk(input, buffers, linear, 0, row, len);
            if (t.riskFlags)
                CheckPackRange(buffers, len);
            StorePackedTriple(real, imag, output, buffers, row, len);
        }
        return;
    }
    for (uint32_t col = GetBlockIdx(); col < t.cols; col += GetBlockNum()) {
        if (t.triangular && !t.transpose && t.packedLd <= 128)
            PackSmallTriangle(input, real, imag, output, buffers, t, col);
        else if (!t.triangular && !t.transpose && t.rows < t.packedLd)
            PackDensePadded(input, real, imag, output, buffers, t, col);
        else
            PackTripleColumn(input, real, imag, output, buffers, t, col);
    }
}
} // namespace
