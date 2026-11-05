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
constexpr uint32_t ORDER_CHUNK = CTRMM_ORDER_CHUNK;

struct OrderedBuffers {
    bool checkExceptional = true;
    bool alphaPrepared = false;
    TBuf<TPosition::VECCALC> coeff, vectors;
#ifdef CTRMM_ORDERED_EDGE
    TBuf<TPosition::VECCALC> edgeBuffer;
    LocalTensor<float> edge;
#endif
    GlobalTensor<float> ar, ai, br, bi, rr, ri;
    LocalTensor<float> data, aRe, aIm;
};

__aicore__ inline void InitOrdered(
    TPipe& pipe, OrderedBuffers& b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR rr, GM_ADDR ri,
    const CtrmmOrderedTiling& t)
{
    pipe.InitBuffer(b.coeff, 2 * t.aLd * sizeof(float));
    pipe.InitBuffer(b.vectors, 16 * ORDER_CHUNK * sizeof(float));
    b.data = b.vectors.Get<float>();
#ifdef CTRMM_ORDERED_EDGE
    pipe.InitBuffer(b.edgeBuffer, 16 * 256 * sizeof(float));
    b.edge = b.edgeBuffer.Get<float>();
#endif
    b.aRe = b.coeff.Get<float>();
    b.aIm = b.aRe[t.aLd];
    b.ar.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ar));
    b.ai.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ai));
    b.br.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(br));
    b.bi.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bi));
    b.rr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rr));
    b.ri.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ri));
}

__aicore__ inline void OrderedProductError(OrderedBuffers& b, LocalTensor<float> src, float factor, uint32_t len)
{
#ifdef CTRMM_ORDERED_PAIRED
    auto hi = b.data[6 * ORDER_CHUNK];
    auto lo = b.data[8 * ORDER_CHUNK];
    auto product = b.data[10 * ORDER_CHUNK];
    auto productError = b.data[12 * ORDER_CHUNK];
    auto scratch = b.data[14 * ORDER_CHUNK];
#else
    auto hi = b.data[8 * ORDER_CHUNK];
    auto lo = b.data[9 * ORDER_CHUNK];
    auto product = b.data[10 * ORDER_CHUNK];
    auto productError = b.data[11 * ORDER_CHUNK];
    auto scratch = b.data[12 * ORDER_CHUNK];
#endif
    union FloatBits {
        float value;
        uint32_t bits;
    } split{factor};
    split.bits &= 0xfffff000U;
    const float high = split.value;
    const float low = factor - high;
    ShiftRight(hi.ReinterpretCast<uint32_t>(), src.ReinterpretCast<uint32_t>(), 12U, len);
    PipeBarrier<PIPE_V>();
    ShiftLeft(hi.ReinterpretCast<uint32_t>(), hi.ReinterpretCast<uint32_t>(), 12U, len);
    PipeBarrier<PIPE_V>();
    Sub(lo, src, hi, len);
    Muls(product, src, factor, len);
    Muls(productError, hi, high, len);
    PipeBarrier<PIPE_V>();
    Sub(productError, productError, product, len);
    Muls(scratch, lo, high, len);
    PipeBarrier<PIPE_V>();
    Add(productError, productError, scratch, len);
    Muls(scratch, hi, low, len);
    PipeBarrier<PIPE_V>();
    Add(productError, productError, scratch, len);
    Muls(scratch, lo, low, len);
    PipeBarrier<PIPE_V>();
    Add(productError, productError, scratch, len);
    PipeBarrier<PIPE_V>();
}

// Approximate fused rounding for ordinary finite inputs with a two-part product.
// Overflow, subnormals and nonfinite values are not fully handled; see the FMA probe.
__aicore__ inline void OrderedSoftwareFma(
    OrderedBuffers& b, LocalTensor<float> src, float factor, LocalTensor<float> accumulator, uint32_t len)
{
#ifdef CTRMM_ORDERED_PAIRED
    // The high/low split is dead after OrderedProductError; reuse it for TwoSum.
    // Each scratch segment spans two output vectors, keeping total UB unchanged.
    auto product = b.data[10 * ORDER_CHUNK];
    auto productError = b.data[12 * ORDER_CHUNK];
    auto scratch = b.data[14 * ORDER_CHUNK];
    auto sum = b.data[6 * ORDER_CHUNK];
    auto residual = b.data[8 * ORDER_CHUNK];
    auto error = b.data[14 * ORDER_CHUNK];
#else
    auto product = b.data[10 * ORDER_CHUNK];
    auto productError = b.data[11 * ORDER_CHUNK];
    auto scratch = b.data[12 * ORDER_CHUNK];
    auto sum = b.data[13 * ORDER_CHUNK];
    auto residual = b.data[14 * ORDER_CHUNK];
    auto error = b.data[15 * ORDER_CHUNK];
#endif
    if (factor == 0.0f || factor == 1.0f || factor == -1.0f) {
        Duplicate(scratch, factor, len);
        PipeBarrier<PIPE_V>();
        MulAddDst(accumulator, scratch, src, len);
        PipeBarrier<PIPE_V>();
        return;
    }
    OrderedProductError(b, src, factor, len);
    Add(sum, product, accumulator, len);
    PipeBarrier<PIPE_V>();
    Sub(residual, sum, product, len);
    PipeBarrier<PIPE_V>();
    Sub(error, sum, residual, len);
    PipeBarrier<PIPE_V>();
    Sub(error, product, error, len);
    Sub(residual, accumulator, residual, len);
    PipeBarrier<PIPE_V>();
    Add(error, error, residual, len);
    PipeBarrier<PIPE_V>();
    Add(error, error, productError, len);
    PipeBarrier<PIPE_V>();
    Add(accumulator, sum, error, len);
    PipeBarrier<PIPE_V>();
}

// One complex coefficient multiplies a vector of independent output elements.
__aicore__ inline void OrderedMultiplyBase(
    OrderedBuffers& b, LocalTensor<float> re, LocalTensor<float> im, float sr, float si, uint32_t len,
    bool scalarFirst = true, bool contiguousPair = false)
{
    auto wr = b.data[4 * ORDER_CHUNK];
    auto wi = b.data[5 * ORDER_CHUNK];
#ifndef CTRMM_ORDERED_COMPENSATED
    auto realFactor = b.data[6 * ORDER_CHUNK];
    auto imagFactor = b.data[7 * ORDER_CHUNK];
#endif
    (void)contiguousPair;
    // Match the reference compiler: round the imaginary product before the real FMA.
    Muls(wr, im, -si, len);
    Muls(wi, scalarFirst ? re : im, scalarFirst ? si : sr, len);
#ifndef CTRMM_ORDERED_COMPENSATED
    Duplicate(realFactor, sr, len);
    Duplicate(imagFactor, scalarFirst ? sr : si, len);
#endif
    PipeBarrier<PIPE_V>();
#ifdef CTRMM_ORDERED_COMPENSATED
#ifdef CTRMM_ORDERED_PAIRED
    if (scalarFirst && contiguousPair && len == ORDER_CHUNK) {
        OrderedSoftwareFma(b, re, sr, wr, 2 * len);
        return;
    }
#endif
    OrderedSoftwareFma(b, re, sr, wr, len);
    OrderedSoftwareFma(b, scalarFirst ? im : re, scalarFirst ? sr : si, wi, len);
#elif defined(CTRMM_ORDERED_VMADD)
    FusedMulAdd(realFactor, re, wr, len);
    FusedMulAdd(imagFactor, scalarFirst ? im : re, wi, len);
    PipeBarrier<PIPE_V>();
    Adds(wr, realFactor, 0.0f, len);
    Adds(wi, imagFactor, 0.0f, len);
#else
    MulAddDst(wr, realFactor, re, len);
    MulAddDst(wi, imagFactor, scalarFirst ? im : re, len);
#endif
    PipeBarrier<PIPE_V>();
}

#ifdef CTRMM_ORDERED_EDGE
#include "ctrmm_ordered_edge.h"
#endif

template <bool CHECK = true>
__aicore__ inline void OrderedMultiply(
    OrderedBuffers& b, LocalTensor<float> re, LocalTensor<float> im, float sr, float si, uint32_t len,
    bool scalarFirst = true, bool contiguousPair = false)
{
    OrderedMultiplyBase(b, re, im, sr, si, len, scalarFirst, contiguousPair);
#ifdef CTRMM_ORDERED_EDGE
    if constexpr (CHECK) {
        if (b.checkExceptional && OrderedNeedsRepair(b, len)) {
            OrderedRepairMultiply(b, re, im, sr, si, len, scalarFirst);
        }
    }
#endif
}

__aicore__ inline void CopyOrderedProduct(OrderedBuffers& b, LocalTensor<float> re, LocalTensor<float> im, uint32_t len)
{
    Adds(re, b.data[4 * ORDER_CHUNK], 0.0f, len);
    Adds(im, b.data[5 * ORDER_CHUNK], 0.0f, len);
    PipeBarrier<PIPE_V>();
}

} // namespace
