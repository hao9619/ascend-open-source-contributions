/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cann_ops_solver.h"

namespace solver_test
{
class DeviceBuffers
{
   public:
    explicit DeviceBuffers(aclrtStream stream) : stream_(stream) {}
    ~DeviceBuffers()
    {
        aclrtSynchronizeStream(stream_);
        for (auto pointer : buffers_) aclrtFree(pointer);
    }
    void *Allocate(size_t bytes)
    {
        if (!bytes) return nullptr;
        void *pointer = nullptr;
        if (aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return nullptr;
        buffers_.push_back(pointer);
        return pointer;
    }

   private:
    aclrtStream stream_;
    std::vector<void *> buffers_;
};

// Preserve the existing file-based tests' row-major data format, while the
// public solver itself receives only Device column-major matrices.
inline aclError RunRealGetrf(aclsolverHandle_t handle, int m, int n, float *hostA, int *hostPivots, int *hostInfo,
                             bool inverse = false)
{
    if (m < 1 || n < 1 || !hostA || !hostInfo || (inverse && m != n)) return ACL_ERROR_INVALID_PARAM;
    aclrtStream stream = nullptr;
    if (aclsolverGetStream(handle, &stream) != ACLSOLVER_STATUS_SUCCESS) return ACL_ERROR_INVALID_PARAM;
    DeviceBuffers buffers(stream);
    const size_t matrixBytes = static_cast<size_t>(m) * n * sizeof(float);
    const size_t pivotBytes = static_cast<size_t>(std::min(m, n)) * sizeof(int);
    auto *matrix = static_cast<float *>(buffers.Allocate(matrixBytes));
    auto *pivots = static_cast<int *>(buffers.Allocate(pivotBytes));
    auto *info = static_cast<int *>(buffers.Allocate(sizeof(int)));
    if (!matrix || !pivots || !info) return ACL_ERROR_INVALID_PARAM;
    std::vector<float> columnMajor(static_cast<size_t>(m) * n);
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) columnMajor[static_cast<size_t>(col) * m + row] = hostA[row * n + col];
    auto error = aclrtMemcpy(matrix, matrixBytes, columnMajor.data(), matrixBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (error != ACL_SUCCESS) return error;
    int factorSize = 0, inverseSize = 0;
    if (aclsolverSgetrf_bufferSize(handle, m, n, matrix, m, &factorSize) != ACLSOLVER_STATUS_SUCCESS)
        return ACL_ERROR_INVALID_PARAM;
    if (inverse && aclsolverSgetri_bufferSize(handle, n, matrix, m, &inverseSize) != ACLSOLVER_STATUS_SUCCESS)
        return ACL_ERROR_INVALID_PARAM;
    const size_t workspaceBytes = static_cast<size_t>(std::max(factorSize, inverseSize)) * sizeof(float);
    auto *workspace = static_cast<float *>(buffers.Allocate(workspaceBytes));
    if (workspaceBytes && !workspace) return ACL_ERROR_INVALID_PARAM;
    if (aclsolverSgetrf(handle, m, n, matrix, m, workspace, pivots, info) != ACLSOLVER_STATUS_SUCCESS)
        return ACL_ERROR_INVALID_PARAM;
    error = aclrtSynchronizeStream(stream);
    if (error != ACL_SUCCESS) return error;
    error = aclrtMemcpy(hostInfo, sizeof(int), info, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
    if (error != ACL_SUCCESS) return error;
    if (inverse && *hostInfo == 0)
    {
        if (aclsolverSgetri(handle, n, matrix, m, pivots, workspace, inverseSize, info) != ACLSOLVER_STATUS_SUCCESS)
            return ACL_ERROR_INVALID_PARAM;
        error = aclrtSynchronizeStream(stream);
        if (error != ACL_SUCCESS) return error;
        error = aclrtMemcpy(hostInfo, sizeof(int), info, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
        if (error != ACL_SUCCESS) return error;
    }
    if (hostPivots)
    {
        error = aclrtMemcpy(hostPivots, pivotBytes, pivots, pivotBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (error != ACL_SUCCESS) return error;
    }
    error = aclrtMemcpy(columnMajor.data(), matrixBytes, matrix, matrixBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (error != ACL_SUCCESS) return error;
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) hostA[row * n + col] = columnMajor[static_cast<size_t>(col) * m + row];
    return ACL_SUCCESS;
}
}  // namespace solver_test
