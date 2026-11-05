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
#include <mutex>
#include <new>

#include "../../utils/aclsolver_handle_internal.h"
#include "getri_internal.h"
#include "launch_internal.h"
#include "lu950_internal.h"
#include "tiling/platform/platform_ascendc.h"

namespace
{
aclsolverStatus_t AcquireWorkspace(_aclsolver_handle &handle, size_t bytes, aclrtStream stream, size_t &slot)
{
    for (size_t index = 0; index < handle.workspaces.size(); ++index)
    {
        auto &workspace = handle.workspaces[index];
        // Keep scratch attached to its stream. This also avoids a cross-stream
        // reuse decision on a transient event query between mixed AIC/AIV work.
        if (workspace.bytes >= bytes && workspace.stream == stream)
        {
            workspace.stream = stream;
            slot = index;
            return ACLSOLVER_STATUS_SUCCESS;
        }
    }
    _aclsolver_workspace workspace;
    if (aclrtMalloc(&workspace.address, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
        return ACLSOLVER_STATUS_ALLOC_FAILED;
    if (aclrtCreateEventExWithFlag(&workspace.completion, ACL_EVENT_CAPTURE_STREAM_PROGRESS) != ACL_SUCCESS)
    {
        aclrtFree(workspace.address);
        return ACLSOLVER_STATUS_ALLOC_FAILED;
    }
    workspace.bytes = bytes;
    workspace.stream = stream;
    try
    {
        handle.workspaces.push_back(workspace);
    }
    catch (const std::bad_alloc &)
    {
        aclrtDestroyEvent(workspace.completion);
        aclrtFree(workspace.address);
        return ACLSOLVER_STATUS_ALLOC_FAILED;
    }
    slot = handle.workspaces.size() - 1;
    return ACLSOLVER_STATUS_SUCCESS;
}
}  // namespace

aclsolverStatus_t Lu950GetriBatched(aclsolverHandle_t handle, int n, const float *const *a, int lda, const int *pivots,
                                    float *const *c, int ldc, int *info, int batch)
{
    if (!handle) return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    if (n < 1 || n > 4096 || lda < n || ldc < n || batch < 0 || (batch > 0 && (!a || !c || !info)))
        return ACLSOLVER_STATUS_INVALID_VALUE;
    if (batch == 0) return ACLSOLVER_STATUS_SUCCESS;
    aclrtStream stream = nullptr;
    const auto status = aclsolverGetStream(handle, &stream);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform) return ACLSOLVER_STATUS_NOT_INITIALIZED;
    const unsigned blocks = platform->GetCoreNumAiv();
    if (!blocks) return ACLSOLVER_STATUS_ARCH_MISMATCH;
    if (n <= 128)
    {
        Lu950BatchGetriLaunch(n, a, lda, pivots, c, ldc, info, batch, blocks, stream);
        return ACLSOLVER_STATUS_SUCCESS;
    }
    auto &internal = *reinterpret_cast<_aclsolver_handle *>(handle);
    std::lock_guard<std::mutex> guard(internal.workspaceMutex);
    aclrtContext current = nullptr;
    if (aclrtGetCurrentContext(&current) != ACL_SUCCESS || current != internal.creationContext)
        return ACLSOLVER_STATUS_NOT_INITIALIZED;
    size_t slot = 0;
    constexpr size_t scratchBudget = 512ULL * 1024 * 1024;
    const int chunk = static_cast<int>(
        std::min(static_cast<size_t>(batch), std::max(size_t{1}, scratchBudget / Lu950LargeBatchMatrixBytes(n))));
    const auto allocation = AcquireWorkspace(internal, Lu950LargeBatchWorkspaceBytes(n, chunk), stream, slot);
    if (allocation != ACLSOLVER_STATUS_SUCCESS) return allocation;
    auto &workspace = internal.workspaces[slot];
    aclsolverStatus_t result = ACLSOLVER_STATUS_SUCCESS;
    for (int matrix = 0; matrix < batch; matrix += chunk)
        if (Lu950ParallelBatchGetriLaunch(n, a, lda, pivots, c, ldc, info, matrix, std::min(chunk, batch - matrix),
                                          workspace.address, blocks, stream) != 0)
        {
            result = ACLSOLVER_STATUS_INTERNAL_ERROR;
            break;
        }
    if (aclrtRecordEvent(workspace.completion, stream) != ACL_SUCCESS) return ACLSOLVER_STATUS_EXECUTION_FAILED;
    return result;
}
