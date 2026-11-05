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
constexpr uint32_t BULK_LANES = 256, BULK_K = 16, BULK_SCAN = 256;
constexpr uint32_t BULK_TERMS = BULK_LANES * BULK_K;
struct BulkRepair {
    TBuf<TPosition::VECCALC> buffer;
    LocalTensor<float> a, b, ar, ai, br, bi, pr, pi, scan, scratch, re, im, gr, gi;
    LocalTensor<int32_t> index;
    LocalTensor<uint32_t> points, offsets;
    GlobalTensor<float> ag, bg, cg, packedA, packedB;
    uint32_t count = 0, begin = 0;
    GlobalTensor<float> seed;
};

__aicore__ inline void InitBulk(
    TPipe& pipe, BulkRepair& v, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR scratch, GM_ADDR seed, uint32_t d)
{
    pipe.InitBuffer(v.buffer, (11 * BULK_TERMS + 4 * BULK_SCAN + 3 * BULK_LANES + 8) * sizeof(float));
    v.a = v.buffer.Get<float>();
    v.b = v.a[2 * BULK_TERMS];
    v.ar = v.a[4 * BULK_TERMS];
    v.ai = v.a[5 * BULK_TERMS];
    v.br = v.a[6 * BULK_TERMS];
    v.bi = v.a[7 * BULK_TERMS];
    v.pr = v.a[8 * BULK_TERMS];
    v.pi = v.a[9 * BULK_TERMS];
    v.index = v.a[10 * BULK_TERMS].ReinterpretCast<int32_t>();
    v.scan = v.a[11 * BULK_TERMS];
    v.scratch = v.scan[2 * BULK_SCAN];
    v.re = v.scratch[2 * BULK_SCAN];
    v.im = v.re[BULK_LANES];
    v.gr = v.re[2 * BULK_LANES];
    v.points = v.gr[8].ReinterpretCast<uint32_t>();
    v.ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    v.bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    v.cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    const uint64_t one = uint64_t(BULK_LANES) * ((d + BULK_K - 1) / BULK_K * BULK_K) * 2;
    auto* base = reinterpret_cast<__gm__ float*>(scratch) + uint64_t(GetBlockIdx()) * 2 * one;
    v.packedA.SetGlobalBuffer(base);
    v.packedB.SetGlobalBuffer(base + one);
    v.begin = (d * 75 / 100 / BULK_K) * BULK_K;
    v.seed.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(seed));
}

__aicore__ inline void InitProductOffsets(BulkRepair& v)
{
    static_assert(BULK_LANES == 256 && BULK_K == 16, "Index shift mapping requires 256 lanes and K=16");
    CreateVecIndex(v.index, 0, BULK_TERMS);
    PipeBarrier<PIPE_V>();
    ShiftRight(v.ar.ReinterpretCast<uint32_t>(), v.index.ReinterpretCast<uint32_t>(), 8U, BULK_TERMS);
    ShiftLeft(v.index.ReinterpretCast<uint32_t>(), v.index.ReinterpretCast<uint32_t>(), 24U, BULK_TERMS);
    PipeBarrier<PIPE_V>();
    ShiftLeft(v.ar.ReinterpretCast<uint32_t>(), v.ar.ReinterpretCast<uint32_t>(), 3U, BULK_TERMS);
    ShiftRight(v.index.ReinterpretCast<uint32_t>(), v.index.ReinterpretCast<uint32_t>(), 17U, BULK_TERMS);
    PipeBarrier<PIPE_V>();
    Add(v.index, v.index, v.ar.ReinterpretCast<int32_t>(), BULK_TERMS);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void BulkProducts(BulkRepair& v)
{
    LoadToVector();
    Gather(v.ar, v.a, v.index.ReinterpretCast<uint32_t>(), 0, BULK_TERMS);
    Gather(v.br, v.b, v.index.ReinterpretCast<uint32_t>(), 0, BULK_TERMS);
    Gather(v.ai, v.a, v.index.ReinterpretCast<uint32_t>(), 4, BULK_TERMS);
    Gather(v.bi, v.b, v.index.ReinterpretCast<uint32_t>(), 4, BULK_TERMS);
    PipeBarrier<PIPE_V>();
    Mul(v.pr, v.ai, v.bi, BULK_TERMS);
    Mul(v.pi, v.ai, v.br, BULK_TERMS);
    PipeBarrier<PIPE_V>();
    Muls(v.pi, v.pi, -1.0f, BULK_TERMS);
    PipeBarrier<PIPE_V>();
    MulAddDst(v.pr, v.ar, v.br, BULK_TERMS);
    MulAddDst(v.pi, v.ar, v.bi, BULK_TERMS);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void BulkLoad(LocalTensor<float> dst, GlobalTensor<float>& src, uint64_t offset, uint32_t count)
{
    DataCopyExtParams copy{1, count * uint32_t(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>((8 - count % 8) % 8), 0.0f};
    DataCopyPad(dst, src[offset], copy, pad);
}

__aicore__ inline void PrepackBulk(BulkRepair& v, uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t d)
{
    if (ldc == 0) {
        return;
    }
    const uint32_t aligned = (d - v.begin + BULK_K - 1) / BULK_K * BULK_K;
    DataCopyExtParams copy{static_cast<uint16_t>(aligned / BULK_K), BULK_K * 8, 0, (BULK_LANES - 1) * BULK_K * 8, 0};
    for (uint32_t lane = 0; lane < v.count; ++lane) {
        Duplicate(v.a, 0.0f, 2 * aligned);
        Duplicate(v.b, 0.0f, 2 * aligned);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        if (lane < v.count) {
            const uint32_t point = v.points.GetValue(lane);
            const uint32_t row = point % ldc, col = point / ldc;
            if (row > v.begin) {
                BulkLoad(v.a, v.ag, (uint64_t(row) * lda + v.begin) * 2, (row - v.begin) * 2);
                BulkLoad(v.b, v.bg, (uint64_t(col) * ldb + v.begin) * 2, (row - v.begin) * 2);
            }
        }
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(v.packedA[lane * BULK_K * 2], v.a, copy);
        DataCopyPad(v.packedB[lane * BULK_K * 2], v.b, copy);
        cblas3::SyncAfterChunkStore();
    }
}

__aicore__ inline uint32_t LoadBulk(
    BulkRepair& v, uint32_t start, bool diagonal, uint32_t lda, uint32_t ldb, uint32_t ldc)
{
    if (ldc == 0) {
        return 0;
    }
    if (!diagonal) {
        if (v.count < BULK_LANES) {
            Duplicate(v.a, 0.0f, 2 * BULK_TERMS);
            Duplicate(v.b, 0.0f, 2 * BULK_TERMS);
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        LoadVector(v.a, v.packedA, uint64_t(start - v.begin) * BULK_LANES * 2, 2 * v.count * BULK_K);
        LoadVector(v.b, v.packedB, uint64_t(start - v.begin) * BULK_LANES * 2, 2 * v.count * BULK_K);
        BulkProducts(v);
        return 0;
    }
    Duplicate(v.a, 0.0f, 2 * BULK_TERMS);
    Duplicate(v.b, 0.0f, 2 * BULK_TERMS);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    uint32_t last = 0;
    for (uint32_t lane = 0; lane < v.count; ++lane) {
        const uint32_t point = v.points.GetValue(lane);
        const uint32_t row = point % ldc, col = point / ldc;
        last = row > last ? row : last;
        if (diagonal || row > start) {
            const uint32_t k = diagonal ? row : start;
            const uint32_t len = diagonal ? 1 : (row - start < BULK_K ? row - start : BULK_K);
            BulkLoad(v.a[lane * 2 * BULK_K], v.ag, (uint64_t(row) * lda + k) * 2, len * 2);
            BulkLoad(v.b[lane * 2 * BULK_K], v.bg, (uint64_t(col) * ldb + k) * 2, len * 2);
        }
    }
    BulkProducts(v);
    return last;
}

__aicore__ inline void BulkSum(BulkRepair& v, uint32_t len)
{
    for (uint32_t k = 0; k < len; ++k) {
        Add(v.re, v.re, v.pr[k * BULK_LANES], BULK_LANES);
        Add(v.im, v.im, v.pi[k * BULK_LANES], BULK_LANES);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void SeedBulk(BulkRepair& v, uint32_t ldc)
{
    if (ldc == 0) {
        return;
    }
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (uint32_t lane = 0; lane < v.count; ++lane) {
        const uint32_t point = v.points.GetValue(lane);
        BulkLoad(v.ar[lane * 8], v.seed, uint64_t(point) * 2, 2);
        v.pr.SetValue(lane, point % ldc >= v.begin ? 1.0f : 0.0f);
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    LoadToVector();
    CreateVecIndex(v.ai.ReinterpretCast<int32_t>(), 0, BULK_LANES);
    PipeBarrier<PIPE_V>();
    Muls(v.ai.ReinterpretCast<int32_t>(), v.ai.ReinterpretCast<int32_t>(), 32, BULK_LANES);
    PipeBarrier<PIPE_V>();
    Gather(v.br, v.ar, v.ai.ReinterpretCast<uint32_t>(), 0, BULK_LANES);
    Gather(v.bi, v.ar, v.ai.ReinterpretCast<uint32_t>(), 4, BULK_LANES);
    Mul(v.re, v.re, v.pr, BULK_LANES);
    Mul(v.im, v.im, v.pr, BULK_LANES);
    PipeBarrier<PIPE_V>();
    Add(v.re, v.re, v.br, BULK_LANES);
    Add(v.im, v.im, v.bi, BULK_LANES);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void RunBulk(BulkRepair& v, uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t d)
{
    PrepackBulk(v, lda, ldb, ldc, d);
    const uint32_t last = LoadBulk(v, 0, true, lda, ldb, ldc);
    Adds(v.re, v.pr, 0.0f, BULK_LANES);
    Adds(v.im, v.pi, 0.0f, BULK_LANES);
    PipeBarrier<PIPE_V>();
    SeedBulk(v, ldc);
    for (uint32_t start = v.begin; start < last; start += BULK_K) {
        LoadBulk(v, start, false, lda, ldb, ldc);
        BulkSum(v, last - start < BULK_K ? last - start : BULK_K);
    }
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (uint32_t lane = 0; lane < v.count; ++lane) {
        v.gr.SetValue(0, v.re.GetValue(lane));
        v.gr.SetValue(1, v.im.GetValue(lane));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        StoreVector(v.cg, v.gr, uint64_t(v.points.GetValue(lane)) * 2, 2);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
    v.count = 0;
}

__aicore__ inline uint32_t SelectBulk(BulkRepair& v, uint64_t start, uint32_t len, float cut)
{
    Duplicate(v.scan, 0.0f, 2 * BULK_SCAN);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    BulkLoad(v.scan, v.cg, start * 2, len * 2);
    LoadToVector();
    Abs(v.scan, v.scan, BULK_SCAN * 2);
    PipeBarrier<PIPE_V>();
    uint64_t count = 0;
    GatherMask(v.ar, v.scan, 1, false, 0, {1, 8, 8, 0}, count);
    GatherMask(v.ai, v.scan, 2, false, 0, {1, 8, 8, 0}, count);
    PipeBarrier<PIPE_V>();
    Max(v.pr, v.ar, v.ai, BULK_SCAN);
    PipeBarrier<PIPE_V>();
    CompareScalar(v.scratch.ReinterpretCast<uint8_t>(), v.pr, cut, CMPMODE::GE, BULK_SCAN);
    PipeBarrier<PIPE_V>();
    CreateVecIndex(v.pr.ReinterpretCast<int32_t>(), 0, BULK_SCAN);
    PipeBarrier<PIPE_V>();
    GatherMask(
        v.scratch[128].ReinterpretCast<int32_t>(), v.pr.ReinterpretCast<int32_t>(),
        v.scratch.ReinterpretCast<uint32_t>(), true, BULK_SCAN, {1, 1, 8, 0}, count);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    return count;
}
} // namespace

extern "C" __global__ __aicore__ void ctrmm_repair_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR scratch, GM_ADDR seed, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb,
    uint32_t ldc, float cut)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockNum() == 0)
        return;
    TPipe pipe;
    BulkRepair v;
    InitBulk(pipe, v, a, b, c, scratch, seed, m);
    InitProductOffsets(v);
    const uint32_t completePrefix = v.begin / CTRMM_TILE_N * CTRMM_TILE_N;
    const uint32_t rowBlocks = (m - completePrefix + BULK_SCAN - 1) / BULK_SCAN;
    const uint64_t tiles = uint64_t(rowBlocks) * n;
    for (uint64_t tile = GetBlockIdx(); tile < tiles; tile += GetBlockNum()) {
        const uint32_t rowStart = completePrefix + uint32_t(tile / n) * BULK_SCAN;
        const uint64_t start = (tile % n) * ldc + rowStart;
        const uint32_t len = m - rowStart < BULK_SCAN ? m - rowStart : BULK_SCAN;
        const uint32_t selected = SelectBulk(v, start, len, cut);
        for (uint32_t i = 0; i < selected; ++i) {
            const uint32_t offset = v.scratch[128].ReinterpretCast<uint32_t>().GetValue(i);
            const uint32_t row = (start + offset) % ldc;
            if (row >= completePrefix && row < m) {
                v.points.SetValue(v.count++, start + offset);
                if (v.count == BULK_LANES)
                    RunBulk(v, lda, ldb, ldc, m);
            }
        }
    }
    if (v.count)
        RunBulk(v, lda, ldb, ldc, m);
}

void ctrmm_repair_do(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* scratch, uint8_t* seed, uint32_t m,
    uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float cut)
{
    ctrmm_repair_kernel<<<blocks, nullptr, stream>>>(a, b, c, scratch, seed, m, n, lda, ldb, ldc, cut);
}

uint64_t ctrmm_repair_scratch_per_core(uint32_t d)
{
    return 2ULL * BULK_LANES * ((uint64_t(d) + BULK_K - 1) / BULK_K * BULK_K) * 2 * sizeof(float);
}

uint32_t ctrmm_repair_prefix(uint32_t d) { return (d * 75 / 100 / BULK_K) * BULK_K; }
