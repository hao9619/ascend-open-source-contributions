/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#define CTRMM_COMPACT_REDUCTION
#include "kernel_operator.h"
#include "common/helper/complex_blas3_arch22.h"
#include "ctrmm_kernel.h"

#include "ctrmm_device_imports.h"

#include "ctrmm_pack_base.h"

#include "ctrmm_pack_risk.h"
#include "ctrmm_pack_column.h"
#include "ctrmm_pack_triple.h"
#include "ctrmm_pack_pair.h"

extern "C" __global__ __aicore__ void ctrmm_compact_pack_pair_kernel(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, GM_ADDR as, GM_ADDR b, GM_ADDR br, GM_ADDR bi, GM_ADDR bs, CtrmmPackTiling pa,
    CtrmmPackTiling pb)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunPackPair(a, ar, ai, as, b, br, bi, bs, pa, pb);
}

void ctrmm_compact_pack_pair_do(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* ar, uint8_t* ai, uint8_t* as, uint8_t* b, uint8_t* br,
    uint8_t* bi, uint8_t* bs, CtrmmPackTiling pa, CtrmmPackTiling pb)
{
    ctrmm_compact_pack_pair_kernel<<<blocks, nullptr, stream>>>(a, ar, ai, as, b, br, bi, bs, pa, pb);
}

#include "ctrmm_combine_buffers.h"
#include "ctrmm_combine_chunk.h"

__aicore__ inline void RunCompactCombine(
    GM_ADDR rr, GM_ADDR ii, GM_ADDR ri, GM_ADDR ir, GM_ADDR out, CtrmmCombineTiling t)
{
    if (GetBlockNum() == 0)
        return;
    const uint32_t total = t.m * t.n;
    const uint32_t block = Clip(((total + GetBlockNum() - 1) / GetBlockNum() + 3) / 4 * 4);
    TPipe pipe;
    CombineBuffers buffers;
    InitCombine<true>(pipe, buffers, rr, ii, ri, ir, out, block);
    t.m = total;
    t.n = 1;
    t.tempLd = total;
    t.ldc = total;
    for (uint32_t row = GetBlockIdx() * block; row < total; row += GetBlockNum() * block) {
        CombineChunk(buffers, t, 0, row, block);
    }
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0; see LICENSE.
 */

#include "ctrmm_ordered_core.h"

#include "ctrmm_ordered_guard.h"

namespace {} // namespace
#include "ctrmm_ordered_element.h"
namespace {
__aicore__ inline void SmallLeftOrderedOutput(GM_ADDR a, GM_ADDR b, GM_ADDR c, CtrmmOrderedTiling t)
{
    TPipe pipe;
    OrderedBuffers buffers;
    TBuf<TPosition::VECCALC> output;
    InitOrdered(pipe, buffers, a, a, b, b, c, c, t);
    pipe.InitBuffer(output, 8 * sizeof(float));
    auto values = output.Get<float>();
    GlobalTensor<float> cg;
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    const uint32_t groups = t.m / 4;
    for (uint32_t item = GetBlockIdx(); item < groups * t.n; item += GetBlockNum()) {
        const uint32_t row = (item % groups) * 4;
        const uint32_t col = item / groups;
        for (uint32_t i = 0; i < 4; ++i) {
            SmallOrderedElement(buffers, t, row + i, col);
            values.SetValue(2 * i, buffers.data.GetValue(2 * ORDER_CHUNK));
            values.SetValue(2 * i + 1, buffers.data.GetValue(3 * ORDER_CHUNK));
        }
        PipeBarrier<PIPE_ALL>();
        StoreVector(cg, values, (uint64_t(col) * t.ldc + row) * 2, 8);
        cblas3::SyncAfterChunkStore();
    }
}
} // namespace

extern "C" __global__ __aicore__ void ctrmm_left_guarded_combine_kernel(
    GM_ADDR rr, GM_ADDR ii, GM_ADDR ri, GM_ADDR ir, GM_ADDR out, GM_ADDR a, GM_ADDR b, GM_ADDR flags,
    CtrmmCombineTiling combine, CtrmmOrderedTiling ordered)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (OrderedCheckExceptional(flags, ordered.guardBlocks)) {
        SmallLeftOrderedOutput(a, b, out, ordered);
    } else {
        RunCompactCombine(rr, ii, ri, ir, out, combine);
    }
}

void ctrmm_left_guarded_combine_do(
    uint32_t blocks, void* stream, uint8_t* rr, uint8_t* ii, uint8_t* ri, uint8_t* ir, uint8_t* out, uint8_t* a,
    uint8_t* b, uint8_t* flags, CtrmmCombineTiling combine, CtrmmOrderedTiling ordered)
{
    ctrmm_left_guarded_combine_kernel<<<blocks, nullptr, stream>>>(rr, ii, ri, ir, out, a, b, flags, combine, ordered);
}
