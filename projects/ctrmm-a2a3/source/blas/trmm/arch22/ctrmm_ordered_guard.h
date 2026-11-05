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
__aicore__ inline bool OrderedCheckExceptional(GM_ADDR flags, uint32_t count)
{
    if (count == 0)
        return true;
    // Read the contiguous per-core flag slots with one DMA instead of scalar GM loads.
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    const uint32_t length = (count * 8 + 63) / 64 * 64;
    pipe.InitBuffer(storage, (2 * length + 8) * sizeof(float));
    auto data = storage.Get<float>();
    GlobalTensor<float> values;
    values.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(flags));
    Duplicate(data, 0.0f, length);
    PipeBarrier<PIPE_ALL>();
    LoadVector(data, values, 0, count * 8);
    LoadToVector();
#ifdef CTRMM_COMPACT_REDUCTION
    if (length <= 4096) {
        WholeReduceMax(data[length], data, 64, length / 64, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(data[2 * length], data[length], length / 64, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    } else {
        ReduceMax(data[2 * length], data, data[length], length, false);
    }
#else
    ReduceMax(data[2 * length], data, data[length], length, false);
#endif
    PipeBarrier<PIPE_ALL>();
    return data.GetValue(2 * length) != 0.0f;
}
