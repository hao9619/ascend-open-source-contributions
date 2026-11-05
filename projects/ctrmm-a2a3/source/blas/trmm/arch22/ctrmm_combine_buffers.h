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
struct CombineBuffers {
    TBuf<TPosition::VECCALC> data, index, aux0, aux1;
    GlobalTensor<float> rrGm, iiGm, riGm, irGm, outGm;
};

#ifdef CTRMM_COMPACT_REDUCTION
template <bool COMPACT = false>
#endif
#ifdef CTRMM_COMPACT_REDUCTION
__aicore__ inline void InitCombine(
    TPipe& pipe, CombineBuffers& buffers, GM_ADDR rr, GM_ADDR ii, GM_ADDR ri, GM_ADDR ir, GM_ADDR out,
    uint32_t pairs = CHUNK)
#else
__aicore__ inline void InitCombine(
    TPipe& pipe, CombineBuffers& buffers, GM_ADDR rr, GM_ADDR ii, GM_ADDR ri, GM_ADDR ir, GM_ADDR out)
#endif
{
    pipe.InitBuffer(buffers.data, CHUNK * 8 * sizeof(float));
    pipe.InitBuffer(buffers.index, CHUNK * 2 * sizeof(int32_t));
    pipe.InitBuffer(buffers.aux0, CHUNK * 2 * sizeof(int32_t));
    pipe.InitBuffer(buffers.aux1, CHUNK * 2 * sizeof(int32_t));
    auto index = buffers.index.Get<int32_t>();
#ifdef CTRMM_COMPACT_REDUCTION
    cblas3::BuildInterleaveIndex(
        index, buffers.aux0.Get<int32_t>(), buffers.aux1.Get<int32_t>(), COMPACT ? pairs : CHUNK, CHUNK);
#else
    cblas3::BuildInterleaveIndex(index, buffers.aux0.Get<int32_t>(), buffers.aux1.Get<int32_t>(), CHUNK, CHUNK);
#endif
    buffers.rrGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rr));
    buffers.iiGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ii));
    buffers.riGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ri));
    buffers.irGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ir));
    buffers.outGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
}

} // namespace
