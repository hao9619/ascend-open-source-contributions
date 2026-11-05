/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "launch_internal.h"
#include "lu950_internal.h"
#include "matinv_contract.hpp"
#include "tiling/platform/platform_ascendc.h"

#ifdef LU950_RAW_N2
#endif

#ifdef LU950_RAW_BATCH
#endif

aclsolverStatus_t Lu950MatinvBatched(aclsolverHandle_t handle, int n, const float* const A[], int lda,
                                     float* const Ainv[], int lda_inv, int* info, int batchSize)
{
    if (handle == nullptr)
    {
        return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    }
    aclrtStream stream = nullptr;
    const auto status = aclsolverGetStream(handle, &stream);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return status;
    }
    const int error = lu950::ValidateMatinv(n, A, lda, Ainv, lda_inv, info, batchSize);
    if (batchSize > 0 && info != nullptr)
    {
        auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (platform == nullptr)
        {
            return ACLSOLVER_STATUS_NOT_INITIALIZED;
        }
        unsigned blocks = platform->GetCoreNumAiv();
        if (blocks == 0)
        {
            return ACLSOLVER_STATUS_ARCH_MISMATCH;
        }
        if (blocks > static_cast<unsigned>(batchSize))
        {
            blocks = static_cast<unsigned>(batchSize);
        }
#ifdef LU950_RAW_N2
        if (n == 2 && lda == 2 && lda_inv == 2 && batchSize > 0 && error == 0)
        {
            Lu950MatinvN2Launch(A, Ainv, info, batchSize, blocks, stream);
            return ACLSOLVER_STATUS_SUCCESS;
        }
#endif
#ifdef LU950_RAW_BATCH
        if (n >= 3 && n <= 32 && (batchSize >= 32 || (n >= 10 && n <= 16)) && error == 0)
        {
            Lu950MatinvBatchLaunch(n, A, lda, Ainv, lda_inv, info, batchSize, blocks, stream);
            return ACLSOLVER_STATUS_SUCCESS;
        }
#endif
        Lu950MatinvLaunch(n, A, lda, Ainv, lda_inv, info, batchSize, error, blocks, stream);
    }
    return error == 0 ? ACLSOLVER_STATUS_SUCCESS : ACLSOLVER_STATUS_INVALID_VALUE;
}
