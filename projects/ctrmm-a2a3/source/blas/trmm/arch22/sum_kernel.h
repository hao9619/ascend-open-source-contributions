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

extern "C" __global__ __aicore__ void ctrmm_sum_kernel(GM_ADDR re, GM_ADDR im, GM_ADDR sum, uint64_t count)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockNum() == 0)
        return;
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 2 * CHUNK * sizeof(float));
    auto x = storage.Get<float>();
    auto y = x[CHUNK];
    GlobalTensor<float> real, imag, output;
    real.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(re));
    imag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(im));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum));
    for (uint64_t offset = uint64_t(GetBlockIdx()) * CHUNK; offset < count; offset += uint64_t(GetBlockNum()) * CHUNK) {
        const uint32_t len = count - offset < CHUNK ? count - offset : CHUNK;
        LoadVector(x, real, offset, len);
        LoadVector(y, imag, offset, len);
        LoadToVector();
        Add(x, x, y, len);
        PipeBarrier<PIPE_V>();
        VectorToStore();
        StoreVector(output, x, offset, len);
        cblas3::SyncAfterChunkStore();
    }
}

void ctrmm_sum_do(uint32_t blocks, void* stream, uint8_t* re, uint8_t* im, uint8_t* sum, uint64_t count)
{
    ctrmm_sum_kernel<<<blocks, nullptr, stream>>>(re, im, sum, count);
}
