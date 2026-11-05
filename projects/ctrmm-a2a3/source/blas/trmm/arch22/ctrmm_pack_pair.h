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
__aicore__ inline void RunPackPair(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, GM_ADDR as, GM_ADDR b, GM_ADDR br, GM_ADDR bi, GM_ADDR bs, CtrmmPackTiling pa,
    CtrmmPackTiling pb)
{
    if (GetBlockNum() == 0)
        return;
    TPipe pipe;
    PackBuffers buffers;
    pipe.InitBuffer(buffers.stage, CHUNK * 32);
    pipe.InitBuffer(buffers.re, CHUNK * sizeof(float));
    pipe.InitBuffer(buffers.im, CHUNK * sizeof(float));
    pipe.InitBuffer(buffers.index, CHUNK * sizeof(int32_t));
    pipe.InitBuffer(buffers.range, 2 * CHUNK * sizeof(float));
    Duplicate(buffers.range.Get<float>(), 0.0f, 2 * CHUNK);
    PipeBarrier<PIPE_V>();
    PackAndSum(a, ar, ai, as, buffers, pa);
    PackAndSum(b, br, bi, bs, buffers, pb);
    StorePackRisk(buffers, pa.riskFlags);
}
} // namespace
