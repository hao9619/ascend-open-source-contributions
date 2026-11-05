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
constexpr uint32_t SMALL_CAP = 2048;

struct SmallVectorBuffers {
    TBuf<TPosition::VECCALC> dataBuffer, rawBuffer, indexBuffer, resultBuffer, bBuffer;
    LocalTensor<float> ar, ai, br, bi, pr, pi, work, raw, result, bc;
    LocalTensor<int32_t> index;
};

__aicore__ inline void InitSmallVector(TPipe& pipe, SmallVectorBuffers& b)
{
    pipe.InitBuffer(b.dataBuffer, 7 * SMALL_CAP * sizeof(float));
    pipe.InitBuffer(b.rawBuffer, 2 * SMALL_CAP * sizeof(float));
    pipe.InitBuffer(b.indexBuffer, (2 * SMALL_CAP + 128) * sizeof(int32_t));
    pipe.InitBuffer(b.resultBuffer, 256 * sizeof(float));
    pipe.InitBuffer(b.bBuffer, 512 * sizeof(float));
    b.ar = b.dataBuffer.Get<float>();
    b.ai = b.ar[SMALL_CAP];
    b.br = b.ar[2 * SMALL_CAP];
    b.bi = b.ar[3 * SMALL_CAP];
    b.pr = b.ar[4 * SMALL_CAP];
    b.pi = b.ar[5 * SMALL_CAP];
    b.work = b.ar[6 * SMALL_CAP];
    b.raw = b.rawBuffer.Get<float>();
    b.index = b.indexBuffer.Get<int32_t>();
    b.result = b.resultBuffer.Get<float>();
    b.bc = b.bBuffer.Get<float>();
}

__aicore__ inline bool SmallRisk(SmallVectorBuffers& b, LocalTensor<float> input, uint32_t count)
{
    ShiftLeft(b.br.ReinterpretCast<uint32_t>(), input.ReinterpretCast<uint32_t>(), 1U, count);
    PipeBarrier<PIPE_V>();
    // abs(bits)>>1 is finite, monotonic and maps |x|=16 to 0x1.8p-62f.
    ShiftRight(b.br.ReinterpretCast<uint32_t>(), b.br.ReinterpretCast<uint32_t>(), 2U, count);
    PipeBarrier<PIPE_V>();
    if (count <= 64) {
        WholeReduceMax(b.result, b.br, count, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    } else if ((count & (count - 1)) == 0) {
        const uint32_t repeats = count / 64;
        WholeReduceMax(b.pr, b.br, 64, repeats, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        if (repeats <= 64) {
            WholeReduceMax(b.result, b.pr, repeats, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        } else {
            WholeReduceMax(b.pi, b.pr, 64, repeats / 64, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(b.result, b.pi, repeats / 64, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        }
    } else {
        ReduceMax(b.result, b.br, b.pr, count, false);
    }
    PipeBarrier<PIPE_ALL>();
    return b.result.GetValue(0) >= 0x1.8p-62f;
}

__aicore__ inline void SmallLoadA(
    SmallVectorBuffers& b, GlobalTensor<float>& gm, const CtrmmOrderedTiling& t, uint32_t row, uint32_t rows)
{
    const uint32_t columns = t.trans ? rows : t.d;
    const uint32_t pitch = 2 * (t.trans ? t.d : rows);
    Duplicate(b.raw, 0.0f, columns * pitch);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t local = 0; local < columns; ++local) {
        const uint32_t col = t.trans ? row + local : local;
        const uint32_t triBegin = t.upper ? 0 : col + t.unit;
        const uint32_t triEnd = t.upper ? col + 1 - t.unit : t.d;
        const uint32_t begin = t.trans || triBegin > row ? triBegin : row;
        const uint32_t end = t.trans || triEnd < row + rows ? triEnd : row + rows;
        if (begin >= end)
            continue;
        const uint32_t localBegin = t.trans ? begin : begin - row;
        const uint32_t left = localBegin % 4;
        const uint32_t right = (4 - (left + end - begin) % 4) % 4;
        DataCopyExtParams copy{1, (end - begin) * 8, 0, 0, 0};
        DataCopyPadExtParams<float> pad{true, static_cast<uint8_t>(left * 2), static_cast<uint8_t>(right * 2), 0.0f};
        DataCopyPad(
            b.raw[local * pitch + (localBegin - left) * 2], gm[(uint64_t(col) * t.rawALd + begin) * 2], copy, pad);
    }
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    if (t.unit) {
        for (uint32_t local = 0; local < columns; ++local) {
            const uint32_t col = t.trans ? row + local : local;
            if (t.trans || (col >= row && col < row + rows)) {
                const uint32_t diagonal = t.trans ? col : col - row;
                b.raw.SetValue(local * pitch + diagonal * 2, 1.0f);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void SmallPrepareA(SmallVectorBuffers& b, const CtrmmOrderedTiling& t, uint32_t rows)
{
    const uint32_t count = rows * t.d;
    const uint32_t shift = t.d == 16 ? 4 : (t.d == 32 ? 5 : 6);
    CreateVecIndex(b.index, 0, count);
    PipeBarrier<PIPE_V>();
    if (!t.trans) {
        auto quotient = b.br.ReinterpretCast<int32_t>();
        auto product = b.bi.ReinterpretCast<int32_t>();
        ShiftRight(quotient.ReinterpretCast<uint32_t>(), b.index.ReinterpretCast<uint32_t>(), shift, count);
        PipeBarrier<PIPE_V>();
        Muls(product, quotient, static_cast<int32_t>(t.d), count);
        PipeBarrier<PIPE_V>();
        Sub(b.index, b.index, product, count);
        PipeBarrier<PIPE_V>();
        Muls(b.index, b.index, static_cast<int32_t>(rows), count);
        PipeBarrier<PIPE_V>();
        Add(b.index, b.index, quotient, count);
        PipeBarrier<PIPE_V>();
    }
    Muls(b.index, b.index, 8, count);
    PipeBarrier<PIPE_V>();
    Gather(b.ar, b.raw, b.index.ReinterpretCast<uint32_t>(), 0, count);
    Adds(b.index, b.index, 4, count);
    PipeBarrier<PIPE_V>();
    Gather(b.ai, b.raw, b.index.ReinterpretCast<uint32_t>(), 0, count);
    PipeBarrier<PIPE_V>();
    if (t.conjugate) {
        Muls(b.ai, b.ai, -1.0f, count);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void SmallPrepareIndices(SmallVectorBuffers& b, uint32_t rows, uint32_t k)
{
    const uint32_t count = rows * 4 * k;
    const uint32_t kShift = k == 16 ? 4 : (k == 32 ? 5 : 6);
    auto quotient = b.pr.ReinterpretCast<int32_t>();
    auto columns = b.pi.ReinterpretCast<int32_t>();
    CreateVecIndex(b.index, 0, count);
    PipeBarrier<PIPE_V>();
    ShiftRight(quotient.ReinterpretCast<uint32_t>(), b.index.ReinterpretCast<uint32_t>(), kShift, count);
    PipeBarrier<PIPE_V>();
    Muls(quotient, quotient, static_cast<int32_t>(k), count);
    PipeBarrier<PIPE_V>();
    Sub(quotient, b.index, quotient, count);
    PipeBarrier<PIPE_V>();
    Muls(quotient, quotient, 8, count);
    uint32_t shift = kShift;
    for (uint32_t value = rows; value > 1; value >>= 1)
        ++shift;
    ShiftRight(columns.ReinterpretCast<uint32_t>(), b.index.ReinterpretCast<uint32_t>(), shift, count);
    PipeBarrier<PIPE_V>();
    Muls(columns, columns, static_cast<int32_t>(k * 8), count);
    PipeBarrier<PIPE_V>();
    Add(b.index, quotient, columns, count);
    PipeBarrier<PIPE_V>();
    Adds(b.index[SMALL_CAP], b.index, 4, count);
    PipeBarrier<PIPE_V>();
    cblas3::BuildInterleaveIndex(
        b.index[2 * SMALL_CAP], b.work.ReinterpretCast<int32_t>(), b.pi.ReinterpretCast<int32_t>(), rows * 4, 64);
}

__aicore__ inline void SmallPrepareB(SmallVectorBuffers& b, uint32_t count)
{
    Gather(b.br, b.bc, b.index.ReinterpretCast<uint32_t>(), 0, count);
    Gather(b.bi, b.bc, b.index[SMALL_CAP].ReinterpretCast<uint32_t>(), 0, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void SmallProducts(SmallVectorBuffers& b, uint32_t rows, uint32_t k)
{
    const uint32_t count = rows * k;
    Mul(b.pr, b.ar, b.br, count);
    Mul(b.work, b.ai, b.bi, count);
    Mul(b.pi, b.ar, b.bi, count);
    PipeBarrier<PIPE_V>();
    Sub(b.pr, b.pr, b.work, count);
    PipeBarrier<PIPE_V>();
    Mul(b.work, b.ai, b.br, count);
    PipeBarrier<PIPE_V>();
    Add(b.pi, b.pi, b.work, count);
    PipeBarrier<PIPE_V>();
    WholeReduceSum(b.result, b.pr, k, rows, 1, 1, k / 8);
    WholeReduceSum(b.result[64], b.pi, k, rows, 1, 1, k / 8);
    PipeBarrier<PIPE_V>();
    Gather(b.result[128], b.result, b.index[2 * SMALL_CAP].ReinterpretCast<uint32_t>(), 0, rows * 2);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline uint64_t SmallFast(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, const CtrmmOrderedTiling& t, uint32_t row, uint32_t rows, uint32_t firstColumn,
    uint32_t columnStep)
{
    TPipe pipe;
    SmallVectorBuffers buffers;
    InitSmallVector(pipe, buffers);
    GlobalTensor<float> ag, bg, cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    SmallLoadA(buffers, ag, t, row, rows);
    const bool aRisk = SmallRisk(buffers, buffers.raw, rows * t.d * 2);
    SmallPrepareA(buffers, t, rows);
    for (uint32_t batch = 1; batch < 4; ++batch) {
        DataCopy(buffers.ar[batch * rows * t.d], buffers.ar, rows * t.d);
        DataCopy(buffers.ai[batch * rows * t.d], buffers.ai, rows * t.d);
    }
    PipeBarrier<PIPE_ALL>();
    SmallPrepareIndices(buffers, rows, t.d);
    uint64_t repair = 0;
    for (uint32_t col = firstColumn * 4; col < t.n; col += columnStep * 4) {
        const uint32_t width = t.n - col < 4 ? t.n - col : 4;
        Duplicate(buffers.bc, 0.0f, 8 * t.d);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copy{static_cast<uint16_t>(width), t.d * 8, (t.rawBLd - t.d) * 8, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(buffers.bc, bg[uint64_t(col) * t.rawBLd * 2], copy, pad);
        LoadToVector();
        if (aRisk || SmallRisk(buffers, buffers.bc, 8 * t.d)) {
            for (uint32_t batch = 0; batch < width; ++batch)
                repair |= 1ULL << (col + batch);
            continue;
        }
        SmallPrepareB(buffers, rows * 4 * t.d);
        SmallProducts(buffers, rows * 4, t.d);
        VectorToStore();
        DataCopyExtParams store{static_cast<uint16_t>(width), rows * 8, 0, (t.ldc - rows) * 8, 0};
        DataCopyPad(cg[(uint64_t(col) * t.ldc + row) * 2], buffers.result[128], store);
        cblas3::SyncAfterChunkStore();
    }
    return repair;
}

} // namespace
#include "ctrmm_ordered_element.h"
namespace {

__aicore__ inline void SmallRepair(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, CtrmmOrderedTiling t, uint32_t row, uint32_t rows, uint64_t mask)
{
    TPipe pipe;
    OrderedBuffers buffers;
    TBuf<TPosition::VECCALC> output;
    t.guardBlocks = 1;
    t.aLd = 128;
    t.bLd = 128;
    t.alphaReal = 1.0f;
    t.alphaImag = 0.0f;
    t.sideLeft = 1;
    InitOrdered(pipe, buffers, a, a, b, b, c, c, t);
    pipe.InitBuffer(output, 128 * sizeof(float));
    auto values = output.Get<float>();
    GlobalTensor<float> cg;
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    for (uint32_t col = 0; col < t.n; ++col) {
        if (!(mask & (1ULL << col)))
            continue;
        for (uint32_t index = 0; index < rows; ++index) {
            SmallOrderedElement(buffers, t, row + index, col);
            values.SetValue(index * 2, buffers.data.GetValue(2 * ORDER_CHUNK));
            values.SetValue(index * 2 + 1, buffers.data.GetValue(3 * ORDER_CHUNK));
        }
        PipeBarrier<PIPE_ALL>();
        StoreVector(cg, values, (uint64_t(col) * t.ldc + row) * 2, rows * 2);
        cblas3::SyncAfterChunkStore();
    }
}
} // namespace

extern "C" __global__ __aicore__ void ctrmm_small_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, CtrmmOrderedTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t tile = t.m <= 16 ? 4 : 8;
    const uint32_t groups = (t.m + tile - 1) / tile;
    const uint32_t row = GetBlockIdx() % groups * tile;
    const uint32_t rows = t.m - row < tile ? t.m - row : tile;
    const uint64_t repair = SmallFast(a, b, c, t, row, rows, GetBlockIdx() / groups, GetBlockNum() / groups);
    if (repair)
        SmallRepair(a, b, c, t, row, rows, repair);
}

void ctrmm_small_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, CtrmmOrderedTiling t)
{
    ctrmm_small_kernel<<<blocks, nullptr, stream>>>(a, b, c, t);
}
