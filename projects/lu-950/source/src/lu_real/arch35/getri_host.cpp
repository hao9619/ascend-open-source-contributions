/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cube_internal.h"
#include "getri_internal.h"
#include "launch_internal.h"
#include "tiling/platform/platform_ascendc.h"

size_t Lu950GetriWorkspaceBytes(int n)
{
    if (n < 1 || n > 4096) return 0;
    const size_t matrixBytes = static_cast<size_t>(n) * ((n + 7) / 8 * 8) * sizeof(float);
    return n <= 128 ? 2 * matrixBytes
                    : (4 * matrixBytes + 511) / 512 * 512 + ((n + 31) / 32 * 32 + 511) / 512 * 512 +
                          Lu950CubeWorkspaceBytes();
}

aclsolverStatus_t Lu950Getri(aclsolverHandle_t handle, int n, float *a, int lda, const int *pivots, int *info,
                             void *workspace, size_t workspaceBytes)
{
    if (!handle) return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    const size_t required = Lu950GetriWorkspaceBytes(n);
    if (!required || lda < n || !a || !info || !workspace || workspaceBytes < required)
        return ACLSOLVER_STATUS_INVALID_VALUE;
    aclrtStream stream = nullptr;
    const auto status = aclsolverGetStream(handle, &stream);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform) return ACLSOLVER_STATUS_NOT_INITIALIZED;
    unsigned blocks = platform->GetCoreNumAiv();
    if (!blocks) return ACLSOLVER_STATUS_ARCH_MISMATCH;
    if (blocks > static_cast<unsigned>(n)) blocks = n;
    if (n > 128 && pivots == nullptr)
    {
        Lu950NoPivotGetriLaunch(n, a, lda, info, static_cast<float *>(workspace), blocks, stream);
        return ACLSOLVER_STATUS_SUCCESS;
    }
    if (n > 128)
        return Lu950LargeGetriLaunch(n, a, lda, pivots, info, workspace, blocks, stream) == 0
                   ? ACLSOLVER_STATUS_SUCCESS
                   : ACLSOLVER_STATUS_INTERNAL_ERROR;
    Lu950GetriLaunch(n, a, lda, pivots, info, static_cast<float *>(workspace), blocks, stream);
    return ACLSOLVER_STATUS_SUCCESS;
}
