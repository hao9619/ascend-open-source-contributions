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
__aicore__ inline void SmallOrderedElement(OrderedBuffers& b, const CtrmmOrderedTiling& t, uint32_t row, uint32_t col)
{
    LoadOrderedCoefficients(b, t, row);
    OrderedInitial(b, t, row, col, 1);
    const bool above = bool(t.upper) != bool(t.trans);
    const uint32_t begin = above ? row + 1 : 0;
    const uint32_t end = above ? t.d : row;
    for (uint32_t index = begin; index < end; ++index) {
        const uint32_t k = !t.trans && !t.upper ? end - 1 - (index - begin) : index;
        OrderedTerm(b, t, k, col, 1);
    }
    if (t.trans) {
        OrderedMultiply(b, b.data[2 * ORDER_CHUNK], b.data[3 * ORDER_CHUNK], 1.0f, 0.0f, 1);
        CopyOrderedProduct(b, b.data[2 * ORDER_CHUNK], b.data[3 * ORDER_CHUNK], 1);
    }
    PipeBarrier<PIPE_ALL>();
}
} // namespace
