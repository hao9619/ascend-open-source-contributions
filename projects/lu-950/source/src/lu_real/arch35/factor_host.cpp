/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>

#include "cube_internal.h"
#include "factor_internal.h"
#include "launch_internal.h"
#include "tiling/platform/platform_ascendc.h"

aclsolverStatus_t Lu950Getrf(aclsolverHandle_t handle, int m, int n, float *a, int lda, int *pivots, int *info)
{
    if (!handle) return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    if (m < 1 || m > 4096 || n < 1 || lda < m || !a || !info) return ACLSOLVER_STATUS_INVALID_VALUE;
    aclrtStream stream = nullptr;
    const auto status = aclsolverGetStream(handle, &stream);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform) return ACLSOLVER_STATUS_NOT_INITIALIZED;
    const unsigned blocks = platform->GetCoreNumAiv();
    if (!blocks) return ACLSOLVER_STATUS_ARCH_MISMATCH;
    const int panel = m <= 2048 ? 16 : 8;
    Lu950FactorLaunch(m, n, a, lda, pivots, info, panel, blocks, stream);
    return ACLSOLVER_STATUS_SUCCESS;
}

size_t Lu950GetrfCubeWorkspaceBytes(int m, int n)
{
    if (m < 1 || m > 4096 || n < 1 || n > 8192) return 0;
    const int panel = m <= 2048 ? 16 : 8;
    return Lu950CubeWorkspaceBytes() + static_cast<size_t>(n) * panel * sizeof(float);
}

aclsolverStatus_t Lu950GetrfCube(aclsolverHandle_t handle, int m, int n, float *a, int lda, int *pivots, int *info,
                                 void *workspace, size_t workspaceBytes)
{
    if (!handle) return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    const size_t required = Lu950GetrfCubeWorkspaceBytes(m, n);
    if (!required || lda < m || !a || !info || !workspace || workspaceBytes < required)
        return ACLSOLVER_STATUS_INVALID_VALUE;
    aclrtStream stream = nullptr;
    const auto status = aclsolverGetStream(handle, &stream);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform) return ACLSOLVER_STATUS_NOT_INITIALIZED;
    const unsigned blocks = platform->GetCoreNumAiv();
    if (!blocks) return ACLSOLVER_STATUS_ARCH_MISMATCH;
    auto *negativeUpper = reinterpret_cast<float *>(static_cast<uint8_t *>(workspace) + Lu950CubeWorkspaceBytes());
    const int result =
        Lu950FactorCubeLaunch(m, n, a, lda, pivots, info, m <= 2048 ? 16 : 8, blocks, negativeUpper, workspace, stream);
    return result == 0 ? ACLSOLVER_STATUS_SUCCESS : ACLSOLVER_STATUS_INTERNAL_ERROR;
}
