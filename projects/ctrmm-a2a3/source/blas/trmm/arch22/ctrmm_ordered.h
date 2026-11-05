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

#include "ctrmm_ordered_core.h"

#include "ctrmm_ordered_guard.h"

#ifdef CTRMM_ORDERED_PRESCALE
extern "C" __global__ __aicore__ void ctrmm_prescale_kernel(
    GM_ADDR br, GM_ADDR bi, GM_ADDR flags, uint32_t classifyBlocks, CtrmmOrderedTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (OrderedCheckExceptional(flags, classifyBlocks))
        return;
    TPipe pipe;
    OrderedBuffers b;
    InitOrdered(pipe, b, br, bi, br, bi, br, bi, t);
    const uint64_t count = static_cast<uint64_t>(t.d) * t.bLd;
    const uint64_t stride = static_cast<uint64_t>(GetBlockNum()) * ORDER_CHUNK;
    for (uint64_t offset = static_cast<uint64_t>(GetBlockIdx()) * ORDER_CHUNK; offset < count; offset += stride) {
        const uint32_t len = count - offset < ORDER_CHUNK ? count - offset : ORDER_CHUNK;
        LoadVector(b.data, b.br, offset, len);
        LoadVector(b.data[ORDER_CHUNK], b.bi, offset, len);
        LoadToVector();
        OrderedMultiply<false>(b, b.data, b.data[ORDER_CHUNK], t.alphaReal, t.alphaImag, len);
        VectorToStore();
        StoreVector(b.rr, b.data[4 * ORDER_CHUNK], offset, len);
        StoreVector(b.ri, b.data[5 * ORDER_CHUNK], offset, len);
        cblas3::SyncAfterChunkStore();
    }
}
#endif

extern "C" __global__ __aicore__ void ctrmm_ordered_kernel(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR rr, GM_ADDR ri, GM_ADDR flags, uint32_t classifyBlocks,
    CtrmmOrderedTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (t.guardBlocks && !OrderedCheckExceptional(flags, t.guardBlocks))
        return;
    TPipe pipe;
    OrderedBuffers buffers;
    InitOrdered(pipe, buffers, ar, ai, br, bi, rr, ri, t);
    buffers.checkExceptional = OrderedCheckExceptional(flags, classifyBlocks);
#ifdef CTRMM_ORDERED_PRESCALE
    buffers.alphaPrepared = t.sideLeft && !t.trans && !buffers.checkExceptional;
#endif
    const uint32_t minorSize = t.sideLeft ? t.n : t.m;
    const uint32_t chunks = (minorSize + ORDER_CHUNK - 1) / ORDER_CHUNK;
    for (uint64_t tile = GetBlockIdx(); tile < static_cast<uint64_t>(t.d) * chunks; tile += GetBlockNum()) {
        const uint32_t major = tile / chunks;
        const uint32_t minor = tile % chunks * ORDER_CHUNK;
        const uint32_t len = minorSize - minor < ORDER_CHUNK ? minorSize - minor : ORDER_CHUNK;
#ifdef CTRMM_ORDERED_STATIC_DISPATCH
        if (!buffers.checkExceptional) {
            OrderedTile<false>(buffers, t, major, minor, len);
            continue;
        }
#endif
        OrderedTile<true>(buffers, t, major, minor, len);
    }
}

namespace {
__aicore__ inline void OrderedReadOutput(
    LocalTensor<float> dst, LocalTensor<float> stage, LocalTensor<int32_t> index, GlobalTensor<float>& gm,
    const CtrmmOrderedTiling& t, uint32_t row, uint32_t col, uint32_t len)
{
    if (!t.sideLeft) {
        LoadVector(dst, gm, static_cast<uint64_t>(col) * t.bLd + row, len);
        LoadToVector();
        return;
    }
    DataCopyExtParams copy{static_cast<uint16_t>(len), 4, (t.bLd - 1) * 4, 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    DataCopyPad(stage, gm[static_cast<uint64_t>(row) * t.bLd + col], copy, pad);
    LoadToVector();
    CreateVecIndex(index, 0, ORDER_CHUNK);
    PipeBarrier<PIPE_V>();
    Muls(index, index, 32, ORDER_CHUNK);
    PipeBarrier<PIPE_V>();
    Gather(dst, stage, index.ReinterpretCast<uint32_t>(), 0, len);
    PipeBarrier<PIPE_ALL>();
}
} // namespace

extern "C" __global__ __aicore__ void ctrmm_ordered_output_kernel(
    GM_ADDR re, GM_ADDR im, GM_ADDR output, CtrmmOrderedTiling t, GM_ADDR flags)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (t.guardBlocks && !OrderedCheckExceptional(flags, t.guardBlocks))
        return;
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuffer, stageBuffer, indexBuffer;
    pipe.InitBuffer(dataBuffer, 2 * ORDER_CHUNK * sizeof(float));
    pipe.InitBuffer(stageBuffer, 8 * ORDER_CHUNK * sizeof(float));
    pipe.InitBuffer(indexBuffer, 6 * ORDER_CHUNK * sizeof(int32_t));
    auto data = dataBuffer.Get<float>();
    auto stage = stageBuffer.Get<float>();
    auto index = indexBuffer.Get<int32_t>();
    GlobalTensor<float> reGm, imGm, outGm;
    reGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(re));
    imGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(im));
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    for (uint32_t col = GetBlockIdx(); col < t.n; col += GetBlockNum()) {
        for (uint32_t row = 0; row < t.m; row += ORDER_CHUNK) {
            const uint32_t len = t.m - row < ORDER_CHUNK ? t.m - row : ORDER_CHUNK;
            OrderedReadOutput(data, stage, index, reGm, t, row, col, len);
            OrderedReadOutput(data[ORDER_CHUNK], stage, index, imGm, t, row, col, len);
            cblas3::BuildInterleaveIndex(
                index, index[2 * ORDER_CHUNK], index[4 * ORDER_CHUNK], ORDER_CHUNK, ORDER_CHUNK);
            Gather(stage, data, index.ReinterpretCast<uint32_t>(), 0, len * 2);
            PipeBarrier<PIPE_V>();
            VectorToStore();
            StoreVector(outGm, stage, (static_cast<uint64_t>(col) * t.ldc + row) * 2, len * 2);
            cblas3::SyncAfterChunkStore();
        }
    }
}

void ctrmm_ordered_do(
    uint32_t blocks, void* stream, uint8_t* ar, uint8_t* ai, uint8_t* br, uint8_t* bi, uint8_t* rr, uint8_t* ri,
    uint8_t* flags, uint32_t classifyBlocks, CtrmmOrderedTiling tiling)
{
#ifdef CTRMM_ORDERED_PRESCALE
    if (tiling.sideLeft && !tiling.trans)
        ctrmm_prescale_kernel<<<blocks, nullptr, stream>>>(br, bi, flags, classifyBlocks, tiling);
#endif
    ctrmm_ordered_kernel<<<blocks, nullptr, stream>>>(ar, ai, br, bi, rr, ri, flags, classifyBlocks, tiling);
}

void ctrmm_ordered_output_do(
    uint32_t blocks, void* stream, uint8_t* re, uint8_t* im, uint8_t* out, CtrmmOrderedTiling tiling, uint8_t* flags)
{
    ctrmm_ordered_output_kernel<<<blocks, nullptr, stream>>>(re, im, out, tiling, flags);
}

extern "C" __global__ __aicore__ void ctrmm_fallback_pack_kernel(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, GM_ADDR b, GM_ADDR br, GM_ADDR bi, GM_ADDR flags, uint32_t guardBlocks,
    CtrmmPackTiling pa, CtrmmPackTiling pb)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (guardBlocks && !OrderedCheckExceptional(flags, guardBlocks))
        return;
    TPipe pipe;
    PackBuffers buffers;
    pipe.InitBuffer(buffers.stage, CHUNK * 32);
    pipe.InitBuffer(buffers.re, CHUNK * sizeof(float));
    pipe.InitBuffer(buffers.im, CHUNK * sizeof(float));
    pipe.InitBuffer(buffers.index, CHUNK * sizeof(int32_t));
    GlobalTensor<float> srcGm, reGm, imGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    reGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ar));
    imGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ai));
    for (uint32_t col = GetBlockIdx(); col < pa.cols; col += GetBlockNum())
        PackColumn(srcGm, reGm, imGm, buffers, pa, col);
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    reGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(br));
    imGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bi));
    for (uint32_t col = GetBlockIdx(); col < pb.cols; col += GetBlockNum())
        PackColumn(srcGm, reGm, imGm, buffers, pb, col);
}

void ctrmm_fallback_pack_do(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* ar, uint8_t* ai, uint8_t* b, uint8_t* br, uint8_t* bi,
    uint8_t* flags, uint32_t guardBlocks, CtrmmPackTiling pa, CtrmmPackTiling pb)
{
    ctrmm_fallback_pack_kernel<<<blocks, nullptr, stream>>>(a, ar, ai, b, br, bi, flags, guardBlocks, pa, pb);
}
