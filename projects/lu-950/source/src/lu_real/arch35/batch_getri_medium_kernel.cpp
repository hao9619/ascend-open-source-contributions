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
namespace AscendC
{
namespace
{
template <HardEvent Event>
__aicore__ inline void BatchFence()
{
    const auto event = static_cast<int32_t>(GetTPipePtr()->FetchEventID(Event));
    SetFlag<Event>(event);
    WaitFlag<Event>(event);
}
#include "batch_getri_medium_tiled.hpp"
}  // namespace
}  // namespace AscendC

using namespace AscendC;

__global__ __aicore__ void lu950_getri_batched_medium_tiled(int n, GM_ADDR a, int lda, GM_ADDR pivots, GM_ADDR c,
                                                            int ldc, GM_ADDR info, int batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BatchGetriMediumTiled(n, a, lda, pivots, c, ldc, info, batch);
}

void Lu950BatchGetriMediumTiledLaunch(int n, uint8_t* a, int lda, uint8_t* pivots, uint8_t* c, int ldc, uint8_t* info,
                                      int batch, unsigned blocks, void* stream)
{
    lu950_getri_batched_medium_tiled<<<blocks, nullptr, stream>>>(n, a, lda, pivots, c, ldc, info, batch);
}
