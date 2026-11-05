/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define LU950_HIGH_SCHEDULE 1
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
#include "batch_pointer_utils.hpp"
#ifdef LU950_FUSED_BATCH
#include "batch_getri_fused.hpp"
#endif
#include "batch_getri_small.hpp"
}  // namespace
}  // namespace AscendC

using namespace AscendC;

// Isolate high-order scheduling from the small-order compiler compatibility flags.
__global__ __aicore__ void lu950_matinv_batched_high(int n, GM_ADDR a, int lda, GM_ADDR c, int ldc, GM_ADDR info,
                                                     int batch, int group)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (n < 18 || n > 32) return;
    BatchGetriSmall(n, a, lda, nullptr, c, ldc, info, batch, group, true);
}

void Lu950MatinvHighLaunch(int n, const float* const* a, int lda, float* const* c, int ldc, int* info, int batch,
                           int group, unsigned blocks, void* stream)
{
    lu950_matinv_batched_high<<<blocks, nullptr, stream>>>(n, reinterpret_cast<uint8_t*>(const_cast<float**>(a)), lda,
                                                           reinterpret_cast<uint8_t*>(const_cast<float**>(c)), ldc,
                                                           reinterpret_cast<uint8_t*>(info), batch, group);
}
