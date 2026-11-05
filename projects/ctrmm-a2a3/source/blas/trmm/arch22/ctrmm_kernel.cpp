/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "common/helper/complex_blas3_arch22.h"
#include "ctrmm_kernel.h"

#include "ctrmm_device_imports.h"

#include "ctrmm_pack_base.h"

#include "ctrmm_pack_risk.h"
#include "ctrmm_pack_column.h"
extern "C" __global__ __aicore__ void ctrmm_pack_kernel(GM_ADDR src, GM_ADDR re, GM_ADDR im, CtrmmPackTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockNum() == 0) {
        return;
    }
    TPipe pipe;
    PackBuffers buffers;
    pipe.InitBuffer(buffers.stage, CHUNK * 32);
    pipe.InitBuffer(buffers.re, CHUNK * sizeof(float));
    pipe.InitBuffer(buffers.im, CHUNK * sizeof(float));
    pipe.InitBuffer(buffers.index, CHUNK * sizeof(int32_t));
    pipe.InitBuffer(buffers.range, 2 * CHUNK * sizeof(float));
    Duplicate(buffers.range.Get<float>(), 0.0f, 2 * CHUNK);
    PipeBarrier<PIPE_V>();
    GlobalTensor<float> srcGm, reGm, imGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src));
    reGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(re));
    imGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(im));
    for (uint32_t col = GetBlockIdx(); col < t.cols; col += GetBlockNum()) {
        PackColumn(srcGm, reGm, imGm, buffers, t, col);
    }
    StorePackRisk(buffers, t.riskFlags);
}

#include "ctrmm_pack_triple.h"
#include "ctrmm_pack_pair.h"

extern "C" __global__ __aicore__ void ctrmm_pack_pair_kernel(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, GM_ADDR as, GM_ADDR b, GM_ADDR br, GM_ADDR bi, GM_ADDR bs, CtrmmPackTiling pa,
    CtrmmPackTiling pb)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunPackPair(a, ar, ai, as, b, br, bi, bs, pa, pb);
}

void ctrmm_pack_pair_do(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* ar, uint8_t* ai, uint8_t* as, uint8_t* b, uint8_t* br,
    uint8_t* bi, uint8_t* bs, CtrmmPackTiling pa, CtrmmPackTiling pb)
{
    ctrmm_pack_pair_kernel<<<blocks, nullptr, stream>>>(a, ar, ai, as, b, br, bi, bs, pa, pb);
}

#ifndef CTRMM_TILE_M
#define CTRMM_TILE_M 128
#endif
#ifndef CTRMM_TILE_N
#define CTRMM_TILE_N 128
#endif
#ifndef CTRMM_BASE_K
#define CTRMM_BASE_K 64
#endif
namespace {
__aicore__ static constexpr MatmulShapeParams CtrmmShape()
{
    return {CTRMM_TILE_M, CTRMM_TILE_N, cblas3::SINGLE_K, CTRMM_TILE_M, CTRMM_TILE_N, CTRMM_BASE_K};
}
constexpr auto CTRMM_CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(CtrmmShape(), MatmulBiasParams{false});
constexpr auto CTRMM_STATIC =
    GetMatmulApiTiling<cblas3::MmPlain, cblas3::MmPlain, cblas3::MmPlain, cblas3::MmPlain>(CTRMM_CONFIG);
using CtrmmMatmul = MatmulImpl<cblas3::MmPlain, cblas3::MmPlain, cblas3::MmPlain, cblas3::MmPlain, CTRMM_STATIC>;

constexpr auto CTRMM_TRANS_STATIC =
    GetMatmulApiTiling<cblas3::MmTrans, cblas3::MmPlain, cblas3::MmPlain, cblas3::MmPlain>(CTRMM_CONFIG);
using CtrmmTransMatmul =
    MatmulImpl<cblas3::MmTrans, cblas3::MmPlain, cblas3::MmPlain, cblas3::MmPlain, CTRMM_TRANS_STATIC>;

template <bool TRANS, class MM>
__aicore__ inline void GemmFour(
    MM& mm, GM_ADDR lr, GM_ADDR li, GM_ADDR rr, GM_ADDR ri, GM_ADDR ls, GM_ADDR rs, GM_ADDR out0, GM_ADDR out1,
    GM_ADDR out2, GM_ADDR out3, const CtrmmGemmTiling& tiling, uint32_t row, uint32_t col)
{
    const auto& t = tiling.base;
    const uint32_t rows = row + CTRMM_TILE_M > t.m ? t.m - row : CTRMM_TILE_M;
    const uint32_t cols = col + CTRMM_TILE_N > t.n ? t.n - col : CTRMM_TILE_N;
    const uint32_t logicalStart = tiling.sideLeft ? (tiling.upper ? col : 0) : (tiling.upper ? 0 : row);
    const uint32_t logicalEnd = tiling.sideLeft ? (tiling.upper ? t.k : col + cols) : (tiling.upper ? row + rows : t.k);
    const uint32_t first = t.kBase > logicalStart ? t.kBase : logicalStart;
    const uint32_t last = t.kBase + t.kCount < logicalEnd ? t.kBase + t.kCount : logicalEnd;
    if (first >= last)
        return;
    GM_ADDR left[4] = {lr, li, tiling.fourProduct ? lr : ls, li};
    GM_ADDR right[4] = {rr, ri, tiling.fourProduct ? ri : rs, rr};
    GM_ADDR output[4] = {out0, out1, out2, out3};
    const uint32_t products = tiling.fourProduct ? 4 : 3;
    for (uint32_t product = 0; product < products; ++product) {
        GlobalTensor<float> a, b, c;
        const uint64_t aOffset = TRANS ? uint64_t(first) * t.ldA + row : uint64_t(row) * t.ldA + first;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(left[product]) + aOffset);
        b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(right[product]) + static_cast<uint64_t>(first) * t.ldB + col);
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output[product]) + static_cast<uint64_t>(row) * t.ldc + col);
        mm.SetSingleShape(rows, cols, last - first);
        mm.SetTensorA(a, TRANS);
        mm.SetTensorB(b);
        mm.IterateAll(c, static_cast<uint8_t>(first > logicalStart));
    }
}
} // namespace

template <bool TRANS, class MM>
__aicore__ inline void RunCtrmmGemm(
    GM_ADDR lr, GM_ADDR li, GM_ADDR rr, GM_ADDR ri, GM_ADDR ls, GM_ADDR rs, GM_ADDR out0, GM_ADDR out1, GM_ADDR out2,
    GM_ADDR out3, CtrmmGemmTiling tiling)
{
    const auto& t = tiling.base;
    if (GetBlockNum() == 0 || t.nBlocks == 0)
        return;
    TPipe pipe;
    MM mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    if constexpr (TRANS)
        mm.SetOrgShape(t.ldA, t.ldB, t.k, t.k, t.ldc);
    else
        mm.SetOrgShape(t.m, t.ldB, t.ldA, t.ldA, t.ldc);
    for (uint64_t tile = GetBlockIdx(); tile < static_cast<uint64_t>(t.mBlocks) * t.nBlocks; tile += GetBlockNum()) {
        const uint32_t row = (tile / t.nBlocks) * CTRMM_TILE_M;
        const uint32_t col = (tile % t.nBlocks) * CTRMM_TILE_N;
        GemmFour<TRANS>(mm, lr, li, rr, ri, ls, rs, out0, out1, out2, out3, tiling, row, col);
    }
    mm.End();
}

extern "C" __global__ __aicore__ void ctrmm_gemm_kernel(
    GM_ADDR lr, GM_ADDR li, GM_ADDR rr, GM_ADDR ri, GM_ADDR ls, GM_ADDR rs, GM_ADDR out0, GM_ADDR out1, GM_ADDR out2,
    GM_ADDR out3, CtrmmGemmTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (tiling.base.transMode == CBLAS3_GEMM_TRANS_LEFT)
        RunCtrmmGemm<true, CtrmmTransMatmul>(lr, li, rr, ri, ls, rs, out0, out1, out2, out3, tiling);
    else
        RunCtrmmGemm<false, CtrmmMatmul>(lr, li, rr, ri, ls, rs, out0, out1, out2, out3, tiling);
}

#include "ctrmm_combine_buffers.h"
#include "ctrmm_combine_chunk.h"

__aicore__ inline void RunCombine(GM_ADDR rr, GM_ADDR ii, GM_ADDR ri, GM_ADDR ir, GM_ADDR out, CtrmmCombineTiling t)
{
    if (GetBlockNum() == 0)
        return;
    TPipe pipe;
    CombineBuffers buffers;
    InitCombine(pipe, buffers, rr, ii, ri, ir, out);
    // Equal pitches let a vector transfer cross column boundaries safely.
    if (t.m == 128 && t.n <= 128 && t.tempLd == t.m && t.ldc == t.m) {
        const uint32_t total = t.m * t.n;
        const uint32_t block = ((total + GetBlockNum() - 1) / GetBlockNum() + 3) / 4 * 4;
        t.m = total;
        t.n = 1;
        t.tempLd = total;
        t.ldc = total;
        for (uint32_t row = GetBlockIdx() * block; row < total; row += GetBlockNum() * block) {
            CombineChunk(buffers, t, 0, row, block);
        }
        return;
    }
    if (t.ldc == t.m && uint64_t(t.m) * t.n <= UINT32_MAX) {
        const uint32_t total = t.m * t.n;
        for (uint32_t tile = GetBlockIdx() * CHUNK; tile < total; tile += GetBlockNum() * CHUNK) {
            uint32_t cursor = tile;
            const uint32_t stop = total - tile < CHUNK ? total : tile + CHUNK;
            while (cursor < stop) {
                const uint32_t col = cursor / t.m;
                const uint32_t row = cursor % t.m;
                const uint32_t count = t.m - row < stop - cursor ? t.m - row : stop - cursor;
                CombineChunk(buffers, t, col, row, count);
                cursor += count;
            }
        }
        return;
    }
    // Four complex columns start on a 32-byte boundary for any leading dimension.
    // Keep each such group on one core so adjacent partial DMA writes never race.
    const uint32_t columnsPerGroup = t.ldc % 4 == 0 ? 1 : 4;
    for (uint32_t start = GetBlockIdx() * columnsPerGroup; start < t.n; start += GetBlockNum() * columnsPerGroup) {
        for (uint32_t col = start; col < start + columnsPerGroup && col < t.n; ++col) {
            for (uint32_t row = 0; row < t.m; row += CHUNK) {
                CombineChunk(buffers, t, col, row);
            }
        }
    }
}

extern "C" __global__ __aicore__ void ctrmm_combine_kernel(
    GM_ADDR rr, GM_ADDR ii, GM_ADDR ri, GM_ADDR ir, GM_ADDR out, CtrmmCombineTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunCombine(rr, ii, ri, ir, out, t);
}

void ctrmm_pack_do(uint32_t blocks, void* stream, uint8_t* src, uint8_t* re, uint8_t* im, CtrmmPackTiling tiling)
{
    ctrmm_pack_kernel<<<blocks, nullptr, stream>>>(src, re, im, tiling);
}

void ctrmm_gemm_do(
    uint32_t blocks, void* stream, uint8_t* lr, uint8_t* li, uint8_t* rr, uint8_t* ri, uint8_t* ls, uint8_t* rs,
    uint8_t* out0, uint8_t* out1, uint8_t* out2, uint8_t* out3, CtrmmGemmTiling tiling)
{
    ctrmm_gemm_kernel<<<blocks, nullptr, stream>>>(lr, li, rr, ri, ls, rs, out0, out1, out2, out3, tiling);
}

uint32_t ctrmm_tile_m() { return CTRMM_TILE_M; }
uint32_t ctrmm_tile_n() { return CTRMM_TILE_N; }

void ctrmm_combine_do(
    uint32_t blocks, void* stream, uint8_t* rr, uint8_t* ii, uint8_t* ri, uint8_t* ir, uint8_t* out,
    CtrmmCombineTiling tiling)
{
    ctrmm_combine_kernel<<<blocks, nullptr, stream>>>(rr, ii, ri, ir, out, tiling);
}

uint32_t ctrmm_single_k() { return cblas3::SINGLE_K; }

uint32_t ctrmm_vector_ub_bytes() { return CHUNK * 56; }

#include "ctrmm_ordered.h"

namespace {
__aicore__ inline void NarrowOrderedOutput(GM_ADDR a, GM_ADDR b, GM_ADDR output, CtrmmOrderedTiling t)
{
    TPipe pipe;
    OrderedBuffers buffers;
    InitOrdered(pipe, buffers, a, a, b, b, output, output, t);
    GlobalTensor<float> out;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    for (uint32_t first = GetBlockIdx() * 4; first < t.n; first += GetBlockNum() * 4) {
        for (uint32_t col = first; col < t.n && col < first + 4; ++col) {
            for (uint32_t row = 0; row < t.m; row += ORDER_CHUNK) {
                const uint32_t len = t.m - row < ORDER_CHUNK ? t.m - row : ORDER_CHUNK;
                LoadOrderedCoefficients(buffers, t, col);
                OrderedInitial(buffers, t, col, row, len);
                const bool above = t.upper == t.trans;
                const uint32_t begin = above ? col + 1 : 0;
                const uint32_t end = above ? t.d : col;
                for (uint32_t index = begin; index < end; ++index) {
                    const uint32_t k = t.trans && !t.upper ? end - 1 - (index - begin) : index;
                    OrderedTerm(buffers, t, k, row, len);
                }
                auto index = buffers.data[6 * ORDER_CHUNK].ReinterpretCast<int32_t>();
                cblas3::BuildInterleaveIndex(
                    index, buffers.data[8 * ORDER_CHUNK].ReinterpretCast<int32_t>(),
                    buffers.data[10 * ORDER_CHUNK].ReinterpretCast<int32_t>(), len, ORDER_CHUNK);
                auto stage = buffers.data[4 * ORDER_CHUNK];
                Gather(stage, buffers.data[2 * ORDER_CHUNK], index.ReinterpretCast<uint32_t>(), 0, len * 2);
                PipeBarrier<PIPE_V>();
                VectorToStore();
                StoreVector(out, stage, (uint64_t(col) * t.ldc + row) * 2, len * 2);
                cblas3::SyncAfterChunkStore();
            }
        }
    }
}
} // namespace

extern "C" __global__ __aicore__ void ctrmm_guarded_combine_kernel(
    GM_ADDR rr, GM_ADDR ii, GM_ADDR ri, GM_ADDR ir, GM_ADDR out, GM_ADDR a, GM_ADDR b, GM_ADDR flags,
    CtrmmCombineTiling combine, CtrmmOrderedTiling ordered)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (OrderedCheckExceptional(flags, ordered.guardBlocks)) {
        NarrowOrderedOutput(a, b, out, ordered);
    } else {
        RunCombine(rr, ii, ri, ir, out, combine);
    }
}

void ctrmm_guarded_combine_do(
    uint32_t blocks, void* stream, uint8_t* rr, uint8_t* ii, uint8_t* ri, uint8_t* ir, uint8_t* out, uint8_t* a,
    uint8_t* b, uint8_t* flags, CtrmmCombineTiling combine, CtrmmOrderedTiling ordered)
{
    ctrmm_guarded_combine_kernel<<<blocks, nullptr, stream>>>(rr, ii, ri, ir, out, a, b, flags, combine, ordered);
}

#ifdef CTRMM_ORDERED_CLASSIFY
#include "ctrmm_classify.h"
#endif

#include "sum_kernel.h"

#include "repair_packed.h"

namespace {
__aicore__ inline void TinyTerm(
    GlobalTensor<float>& a, GlobalTensor<float>& b, const CtrmmOrderedTiling& t, uint32_t row, uint32_t col, uint32_t k,
    float& re, float& im, bool first)
{
    const uint32_t ai = (t.trans ? row * t.rawALd + k : k * t.rawALd + row) * 2;
    const uint32_t bi = (col * t.rawBLd + k) * 2;
    const float ar = t.unit && row == k ? 1.0f : a.GetValue(ai);
    const float av = t.unit && row == k ? 0.0f : a.GetValue(ai + 1);
    const float aj = t.conjugate ? -av : av;
    const float br = b.GetValue(bi);
    const float bj = b.GetValue(bi + 1);
    const float pr = ar * br - aj * bj;
    const float pi = ar * bj + aj * br;
    re = first ? pr : re + pr;
    im = first ? pi : im + pi;
}
} // namespace

extern "C" __global__ __aicore__ void ctrmm_tiny_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, CtrmmOrderedTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> output;
    pipe.InitBuffer(output, 32);
    auto local = output.Get<float>();
    GlobalTensor<float> ag, bg, cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    for (uint32_t start = GetBlockIdx() * 4; start < t.m * t.n; start += GetBlockNum() * 4) {
        const uint32_t count = t.m * t.n - start < 4 ? t.m * t.n - start : 4;
        for (uint32_t lane = 0; lane < count; ++lane) {
            const uint32_t row = (start + lane) % t.m;
            const uint32_t col = (start + lane) / t.m;
            float re = 0.0f, im = 0.0f;
            TinyTerm(ag, bg, t, row, col, row, re, im, true);
            const bool above = bool(t.upper) != bool(t.trans);
            const uint32_t begin = above ? row + 1 : 0;
            const uint32_t end = above ? t.d : row;
            for (uint32_t index = begin; index < end; ++index) {
                const uint32_t k = !t.trans && !t.upper ? end - 1 - (index - begin) : index;
                TinyTerm(ag, bg, t, row, col, k, re, im, false);
            }
            local.SetValue(lane * 2, re);
            local.SetValue(lane * 2 + 1, im);
        }
        PipeBarrier<PIPE_ALL>();
        StoreVector(cg, local, start * 2, count * 2);
        cblas3::SyncAfterChunkStore();
    }
}

void ctrmm_tiny_do(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, CtrmmOrderedTiling t)
{
    ctrmm_tiny_kernel<<<blocks, nullptr, stream>>>(a, b, c, t);
}

#include "ctrmm_small_vector.h"
