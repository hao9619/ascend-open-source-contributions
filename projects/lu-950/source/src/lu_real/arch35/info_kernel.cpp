/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "launch_internal.h"

using namespace AscendC;

__global__ __aicore__ void lu950_fill_info(GM_ADDR info, int count, int value)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECOUT> buffer;
    pipe.InitBuffer(buffer, 256 * sizeof(int));
    auto local = buffer.Get<int>();
    Duplicate(local, value, 256);
    const auto event = static_cast<int32_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    GlobalTensor<int> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    for (int start = 0; start < count; start += 256)
    {
        const int size = count - start < 256 ? count - start : 256;
        DataCopyPad(global[start], local, DataCopyExtParams{1, static_cast<uint32_t>(size * sizeof(int)), 0, 0, 0});
    }
}

void Lu950InfoLaunch(int *info, int count, int value, void *stream)
{
    lu950_fill_info<<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(info), count, value);
}
