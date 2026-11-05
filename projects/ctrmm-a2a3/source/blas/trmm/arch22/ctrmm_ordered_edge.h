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

constexpr uint32_t EDGE_CHUNK = 256;

__aicore__ inline float EdgeFloat(uint32_t bits)
{
    union Bits {
        uint32_t bits;
        float value;
    } value{bits};
    return value.value;
}

__aicore__ inline bool OrderedNeedsRepair(OrderedBuffers& b, uint32_t len)
{
    auto re = b.data[6 * ORDER_CHUNK];
    auto im = b.data[8 * ORDER_CHUNK];
    auto top = b.data[10 * ORDER_CHUNK];
    auto work = b.data[12 * ORDER_CHUNK];
    ShiftLeft(re.ReinterpretCast<uint32_t>(), b.data[4 * ORDER_CHUNK].ReinterpretCast<uint32_t>(), 1U, len);
    ShiftLeft(im.ReinterpretCast<uint32_t>(), b.data[5 * ORDER_CHUNK].ReinterpretCast<uint32_t>(), 1U, len);
    PipeBarrier<PIPE_V>();
    ShiftRight(re.ReinterpretCast<uint32_t>(), re.ReinterpretCast<uint32_t>(), 24U, len);
    ShiftRight(im.ReinterpretCast<uint32_t>(), im.ReinterpretCast<uint32_t>(), 24U, len);
    PipeBarrier<PIPE_V>();
    Cast(re, re.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, len);
    Cast(im, im.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, len);
    PipeBarrier<PIPE_V>();
    Max(re, re, im, len);
    PipeBarrier<PIPE_V>();
    ReduceMax(top, re, work, len, false);
    PipeBarrier<PIPE_ALL>();
    return top.GetValue(0) == 255.0f;
}

__aicore__ inline void EdgeSelect(
    LocalTensor<float> dst, LocalTensor<uint8_t> mask, LocalTensor<float> yes, LocalTensor<float> no, uint32_t len)
{
    Select(dst, mask, yes, no, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
    PipeBarrier<PIPE_V>();
}

// Classify the IEEE fields explicitly; vector comparisons involving NaNs are
// not a portable way to implement an unordered predicate on this device.
__aicore__ inline void EdgeIsNaN(OrderedBuffers& b, LocalTensor<uint8_t> mask, LocalTensor<float> src, uint32_t len)
{
    auto exponent = b.data[6 * ORDER_CHUNK];
    auto fraction = b.data[8 * ORDER_CHUNK];
    ShiftLeft(exponent.ReinterpretCast<uint32_t>(), src.ReinterpretCast<uint32_t>(), 1U, len);
    ShiftLeft(fraction.ReinterpretCast<uint32_t>(), src.ReinterpretCast<uint32_t>(), 9U, len);
    PipeBarrier<PIPE_V>();
    ShiftRight(exponent.ReinterpretCast<uint32_t>(), exponent.ReinterpretCast<uint32_t>(), 24U, len);
    ShiftRight(fraction.ReinterpretCast<uint32_t>(), fraction.ReinterpretCast<uint32_t>(), 9U, len);
    PipeBarrier<PIPE_V>();
    Cast(exponent, exponent.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, len);
    Cast(fraction, fraction.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, len);
    PipeBarrier<PIPE_V>();
    Compares(mask, exponent, 255.0f, CMPMODE::EQ, len);
    Duplicate(exponent, 1.0f, len);
    PipeBarrier<PIPE_V>();
    Select(exponent, mask, exponent, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
    Compares(mask, fraction, 0.0f, CMPMODE::GT, len);
    PipeBarrier<PIPE_V>();
    Select(exponent, mask, exponent, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
    PipeBarrier<PIPE_V>();
    Compares(mask, exponent, 1.0f, CMPMODE::EQ, len);
    PipeBarrier<PIPE_V>();
}

// Ordinary inputs retain the existing rounding path. Scale only the exceptional
// product, with exact powers of two, then restore before the ordered addition.
__aicore__ inline void EdgeScaledFma(
    OrderedBuffers& b, LocalTensor<float> src, float factor, LocalTensor<float> acc, LocalTensor<float> out,
    uint32_t len)
{
    auto x = b.edge[4 * EDGE_CHUNK];
    auto y = b.edge[5 * EDGE_CHUNK];
    auto scale = b.edge[6 * EDGE_CHUNK];
    auto temp = b.edge[7 * EDGE_CHUNK];
    auto magnitude = b.edge[8 * EDGE_CHUNK];
    auto mask = b.edge[9 * EDGE_CHUNK].ReinterpretCast<uint8_t>();
    Abs(magnitude, src, len);
    Abs(temp, acc, len);
    PipeBarrier<PIPE_V>();
    Max(magnitude, magnitude, temp, len);
    Duplicate(scale, 0x1p-64f, len);
    PipeBarrier<PIPE_V>();
    Compares(mask, magnitude, 0x1p64f, CMPMODE::GT, len);
    PipeBarrier<PIPE_V>();
    Select(scale, mask, scale, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
    PipeBarrier<PIPE_V>();
    const float factorScale = (factor > 0x1p64f || factor < -0x1p64f) ? 0x1p-64f : 1.0f;
    Mul(x, src, scale, len);
    Mul(y, acc, scale, len);
    PipeBarrier<PIPE_V>();
    Muls(y, y, factorScale, len);
    PipeBarrier<PIPE_V>();
    OrderedSoftwareFma(b, x, factor * factorScale, y, len);
    Div(y, y, scale, len);
    PipeBarrier<PIPE_V>();
    Muls(out, y, 1.0f / factorScale, len);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void EdgeFma(
    OrderedBuffers& b, LocalTensor<float> src, float factor, LocalTensor<float> acc, LocalTensor<float> out,
    uint32_t len)
{
    EdgeScaledFma(b, src, factor, acc, out, len);
    auto temp = b.edge[7 * EDGE_CHUNK];
    auto magnitude = b.edge[8 * EDGE_CHUNK];
    auto mask = b.edge[9 * EDGE_CHUNK].ReinterpretCast<uint8_t>();
    Muls(temp, src, factor, len);
    PipeBarrier<PIPE_V>();
    Add(temp, temp, acc, len);
    PipeBarrier<PIPE_V>();
    // With finite multiplicands, an infinite accumulator remains infinite even
    // when the separately rounded product overflows with the opposite sign.
    Abs(magnitude, acc, len);
    PipeBarrier<PIPE_V>();
    Compares(mask, magnitude, EdgeFloat(0x7f800000U), CMPMODE::EQ, len);
    PipeBarrier<PIPE_V>();
    EdgeSelect(out, mask, acc, out, len);
    Abs(magnitude, src, len);
    PipeBarrier<PIPE_V>();
    Compares(mask, magnitude, EdgeFloat(0x7f800000U), CMPMODE::GE, len);
    PipeBarrier<PIPE_V>();
    EdgeSelect(out, mask, temp, out, len);
    EdgeIsNaN(b, mask, src, len);
    PipeBarrier<PIPE_V>();
    EdgeSelect(out, mask, temp, out, len);
    if (!(factor <= EdgeFloat(0x7f7fffffU) && factor >= -EdgeFloat(0x7f7fffffU))) {
        Adds(out, temp, 0.0f, len);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void OrderedRepairMultiply(
    OrderedBuffers& b, LocalTensor<float> re, LocalTensor<float> im, float sr, float si, uint32_t len, bool scalarFirst)
{
    for (uint32_t offset = 0; offset < len; offset += EDGE_CHUNK) {
        const uint32_t count = len - offset < EDGE_CHUNK ? len - offset : EDGE_CHUNK;
        const uint32_t active = (count + 63U) / 64U * 64U;
        auto x = b.edge;
        auto y = b.edge[EDGE_CHUNK];
        auto wr = b.edge[2 * EDGE_CHUNK];
        auto wi = b.edge[3 * EDGE_CHUNK];
        Duplicate(x, 0.0f, active);
        Duplicate(y, 0.0f, active);
        PipeBarrier<PIPE_V>();
        Adds(x, re[offset], 0.0f, count);
        Adds(y, im[offset], 0.0f, count);
        PipeBarrier<PIPE_V>();
        Muls(wr, y, -si, active);
        Muls(wi, scalarFirst ? x : y, scalarFirst ? si : sr, active);
        PipeBarrier<PIPE_V>();
        EdgeFma(b, x, sr, wr, b.edge[12 * EDGE_CHUNK], active);
        EdgeFma(b, scalarFirst ? y : x, scalarFirst ? sr : si, wi, b.edge[13 * EDGE_CHUNK], active);
        // Preserve the reference Fortran NaN/Inf result without C-complex recovery.
        Adds(b.data[4 * ORDER_CHUNK + offset], b.edge[12 * EDGE_CHUNK], 0.0f, count);
        Adds(b.data[5 * ORDER_CHUNK + offset], b.edge[13 * EDGE_CHUNK], 0.0f, count);
        PipeBarrier<PIPE_V>();
    }
}
