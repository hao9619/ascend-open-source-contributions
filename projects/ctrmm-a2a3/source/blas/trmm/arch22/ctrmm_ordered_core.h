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
#include "ctrmm_ordered_math.h"
namespace {
__aicore__ inline void LoadRawOrdered(
    OrderedBuffers& b, GlobalTensor<float>& input, const CtrmmPackTiling& p, uint32_t col, uint32_t row, uint32_t len,
    LocalTensor<float> re, LocalTensor<float> im)
{
    // Reuse product scratch while no multiply is in flight; keep the live sums at slots 2/3 intact.
    static_assert(ORDER_CHUNK >= 2048, "Raw input loading requires the 2048-output workspace");
    auto stage = b.data[8 * ORDER_CHUNK];
    auto real = b.data[12 * ORDER_CHUNK];
    auto imag = b.data[13 * ORDER_CHUNK];
    auto index = b.data[14 * ORDER_CHUNK].ReinterpretCast<int32_t>();
    PipeBarrier<PIPE_ALL>();
    constexpr uint32_t rawChunk = CHUNK < 512 ? CHUNK : 512;
    for (uint32_t offset = 0; offset < len; offset += rawChunk) {
        const uint32_t count = len - offset < rawChunk ? len - offset : rawChunk;
        LoadComplexChunk(input, stage, real, imag, index, p, col, row + offset, count);
        Adds(re[offset], real, 0.0f, count);
        Adds(im[offset], imag, 0.0f, count);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void AlignRawCoefficients(OrderedBuffers& b, uint32_t begin, uint32_t end, uint32_t length)
{
    auto positions = b.data[8 * ORDER_CHUNK];
    auto indices = b.data[10 * ORDER_CHUNK].ReinterpretCast<int32_t>();
    auto mask = b.data[12 * ORDER_CHUNK].ReinterpretCast<uint8_t>();
    CreateVecIndex(indices, -static_cast<int32_t>(begin), length);
    PipeBarrier<PIPE_V>();
    Cast(positions, indices, RoundMode::CAST_NONE, length);
    PipeBarrier<PIPE_V>();
    Maxs(positions, positions, 0.0f, length);
    PipeBarrier<PIPE_V>();
    Mins(positions, positions, static_cast<float>(static_cast<int32_t>(end - begin - 1)), length);
    PipeBarrier<PIPE_V>();
    Cast(indices, positions, RoundMode::CAST_RINT, length);
    PipeBarrier<PIPE_V>();
    Muls(indices, indices, 4, length);
    PipeBarrier<PIPE_V>();
    Gather(b.aRe, b.data, indices.ReinterpretCast<uint32_t>(), 0, length);
    Gather(b.aIm, b.data[2 * ORDER_CHUNK], indices.ReinterpretCast<uint32_t>(), 0, length);
    PipeBarrier<PIPE_V>();
    CreateVecIndex(indices, 0, length);
    PipeBarrier<PIPE_V>();
    Cast(positions, indices, RoundMode::CAST_NONE, length);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, positions, static_cast<float>(static_cast<int32_t>(begin)), CMPMODE::GE, length);
    PipeBarrier<PIPE_V>();
    Select(b.aRe, mask, b.aRe, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
    Select(b.aIm, mask, b.aIm, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask, positions, static_cast<float>(static_cast<int32_t>(end)), CMPMODE::LT, length);
    PipeBarrier<PIPE_V>();
    Select(b.aRe, mask, b.aRe, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
    Select(b.aIm, mask, b.aIm, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
    PipeBarrier<PIPE_V>();
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void LoadRawCoefficients(OrderedBuffers& b, const CtrmmOrderedTiling& t, uint32_t major)
{
    const bool transpose = t.sideLeft ? !t.trans : t.trans;
    const bool upper = bool(t.upper) != transpose;
    const uint32_t begin = upper ? 0 : major + t.unit;
    const uint32_t end = upper ? major + 1 - t.unit : t.d;
    CtrmmPackTiling pack{t.d, t.d, t.rawALd, t.aLd, 1, t.upper, transpose, t.conjugate, t.unit};
    Duplicate(b.aRe, 0.0f, t.aLd);
    Duplicate(b.aIm, 0.0f, t.aLd);
    PipeBarrier<PIPE_ALL>();
    if (begin < end) {
        LoadRawOrdered(b, b.ar, pack, major, begin, end - begin, b.data, b.data[2 * ORDER_CHUNK]);
        AlignRawCoefficients(b, begin, end, t.aLd);
    }
    if (t.unit) {
        b.aRe.SetValue(major, 1.0f);
        PipeBarrier<PIPE_ALL>();
    }
}

template <bool CHECK = true>
__aicore__ inline void LoadOrderedCoefficients(OrderedBuffers& b, const CtrmmOrderedTiling& t, uint32_t major)
{
    if (t.guardBlocks) {
        LoadRawCoefficients(b, t, major);
    } else {
        LoadVector(b.aRe, b.ar, static_cast<uint64_t>(major) * t.aLd, t.aLd);
        LoadVector(b.aIm, b.ai, static_cast<uint64_t>(major) * t.aLd, t.aLd);
        LoadToVector();
    }
    if (!t.sideLeft) {
        for (uint32_t offset = 0; offset < t.d; offset += ORDER_CHUNK) {
            const uint32_t len = t.d - offset < ORDER_CHUNK ? t.d - offset : ORDER_CHUNK;
            OrderedMultiply<CHECK>(b, b.aRe[offset], b.aIm[offset], t.alphaReal, t.alphaImag, len);
            CopyOrderedProduct(b, b.aRe[offset], b.aIm[offset], len);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void LoadOrderedB(
    OrderedBuffers& b, const CtrmmOrderedTiling& t, uint32_t k, uint32_t minor, uint32_t len)
{
    if (t.guardBlocks) {
        CtrmmPackTiling pack{t.m, t.n, t.rawBLd, t.bLd, 0, 0, t.sideLeft, 0, 0};
        LoadRawOrdered(b, b.br, pack, k, minor, len, b.data, b.data[ORDER_CHUNK]);
        return;
    }
    const uint64_t offset = static_cast<uint64_t>(k) * t.bLd + minor;
    LoadVector(b.data, b.br, offset, len);
    LoadVector(b.data[ORDER_CHUNK], b.bi, offset, len);
    LoadToVector();
}

template <bool CHECK = true>
__aicore__ inline void OrderedInitial(
    OrderedBuffers& b, const CtrmmOrderedTiling& t, uint32_t major, uint32_t minor, uint32_t len)
{
    LoadOrderedB(b, t, major, minor, len);
    if (t.sideLeft && !t.trans && !b.alphaPrepared) {
        OrderedMultiply<CHECK>(b, b.data, b.data[ORDER_CHUNK], t.alphaReal, t.alphaImag, len);
        CopyOrderedProduct(b, b.data, b.data[ORDER_CHUNK], len);
    }
    const bool skip = t.sideLeft ? t.unit : (t.trans && t.unit && t.alphaReal == 1.0f && t.alphaImag == 0.0f);
    if (!skip) {
        OrderedMultiply<CHECK>(
            b, b.data, b.data[ORDER_CHUNK], b.aRe.GetValue(major), b.aIm.GetValue(major), len, t.sideLeft, true);
        CopyOrderedProduct(b, b.data, b.data[ORDER_CHUNK], len);
    }
    Adds(b.data[2 * ORDER_CHUNK], b.data, 0.0f, len);
    Adds(b.data[3 * ORDER_CHUNK], b.data[ORDER_CHUNK], 0.0f, len);
    PipeBarrier<PIPE_ALL>();
}

template <bool CHECK = true>
__aicore__ inline void OrderedTerm(
    OrderedBuffers& b, const CtrmmOrderedTiling& t, uint32_t k, uint32_t minor, uint32_t len)
{
    const float ar = b.aRe.GetValue(k);
    const float ai = b.aIm.GetValue(k);
    if (!t.sideLeft && ar == 0.0f && ai == 0.0f)
        return;
    LoadOrderedB(b, t, k, minor, len);
    if (t.sideLeft && !t.trans && !b.alphaPrepared) {
        OrderedMultiply<CHECK>(b, b.data, b.data[ORDER_CHUNK], t.alphaReal, t.alphaImag, len);
        CopyOrderedProduct(b, b.data, b.data[ORDER_CHUNK], len);
    }
    OrderedMultiply<CHECK>(b, b.data, b.data[ORDER_CHUNK], ar, ai, len, t.sideLeft, true);
    Add(b.data[2 * ORDER_CHUNK], b.data[2 * ORDER_CHUNK], b.data[4 * ORDER_CHUNK], len);
    Add(b.data[3 * ORDER_CHUNK], b.data[3 * ORDER_CHUNK], b.data[5 * ORDER_CHUNK], len);
    PipeBarrier<PIPE_ALL>();
}

template <bool CHECK = true>
__aicore__ inline void OrderedTile(
    OrderedBuffers& b, const CtrmmOrderedTiling& t, uint32_t major, uint32_t minor, uint32_t len)
{
    LoadOrderedCoefficients<CHECK>(b, t, major);
    OrderedInitial<CHECK>(b, t, major, minor, len);
    const bool descending = (t.sideLeft && !t.trans && !t.upper) || (!t.sideLeft && t.trans && !t.upper);
    const bool above = t.sideLeft ? t.upper != t.trans : t.upper == t.trans;
    const uint32_t begin = above ? major + 1 : 0;
    const uint32_t end = above ? t.d : major;
    for (uint32_t index = begin; index < end; ++index) {
        const uint32_t k = descending ? end - 1 - (index - begin) : index;
        OrderedTerm<CHECK>(b, t, k, minor, len);
    }
    if (t.sideLeft && t.trans) {
        OrderedMultiply<CHECK>(b, b.data[2 * ORDER_CHUNK], b.data[3 * ORDER_CHUNK], t.alphaReal, t.alphaImag, len);
        CopyOrderedProduct(b, b.data[2 * ORDER_CHUNK], b.data[3 * ORDER_CHUNK], len);
    }
    VectorToStore();
    const uint64_t offset = static_cast<uint64_t>(major) * t.bLd + minor;
    StoreVector(b.rr, b.data[2 * ORDER_CHUNK], offset, len);
    StoreVector(b.ri, b.data[3 * ORDER_CHUNK], offset, len);
    cblas3::SyncAfterChunkStore();
}
} // namespace
