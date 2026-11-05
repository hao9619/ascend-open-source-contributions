/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../lu_real/lu_real_host.hpp"

aclsolverStatus_t aclsolverSgetrf_bufferSize(aclsolverHandle_t handle, int m, int n, float* A, int lda, int* Lwork)
{
    aclrtStream stream = nullptr;
    const auto status = lu950::Stream(handle, stream);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    if (!Lwork || m < 0 || m > 4096 || n < 0 || n > 8192 || lda < std::max(1, m)) return ACLSOLVER_STATUS_INVALID_VALUE;
#ifdef ACLSOLVER_HAS_LU950
    return lu950::QuerySize(lu950::BlockedFactor(m, n) ? Lu950GetrfBlockedWorkspaceBytes(m, n) : 0, Lwork);
#else
    return ACLSOLVER_STATUS_NOT_SUPPORTED;
#endif
}

aclsolverStatus_t aclsolverSgetrf(aclsolverHandle_t handle, int m, int n, float* A, int lda, float* Workspace,
                                  int* devIpiv, int* devInfo)
{
    aclrtStream stream = nullptr;
    const auto status = lu950::Stream(handle, stream);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    int error = m < 0 || m > 4096 ? 1 : (n < 0 || n > 8192 ? 2 : 0);
    if (!error && m > 0 && n > 0 && !lu950::DeviceSpan(A, sizeof(float))) error = 3;
    if (!error && lda < std::max(1, m)) error = 4;
    if (!error && m > 0 && n > 0 && !lu950::DeviceSpan(A, lu950::MatrixBytes(m, n, lda))) error = 3;
    size_t workspaceBytes = 0;
#ifdef ACLSOLVER_HAS_LU950
    if (!error && lu950::BlockedFactor(m, n)) workspaceBytes = Lu950GetrfBlockedWorkspaceBytes(m, n);
#endif
    if (!error && workspaceBytes &&
        (!lu950::DeviceSpan(Workspace, workspaceBytes) ||
         lu950::Overlap(A, lu950::MatrixBytes(m, n, lda), Workspace, workspaceBytes)))
        error = 5;
    const size_t pivotBytes = static_cast<size_t>(std::max(0, std::min(m, n))) * sizeof(int);
    if (!error && devIpiv && pivotBytes &&
        (!lu950::DeviceSpan(devIpiv, pivotBytes) ||
         lu950::Overlap(A, lu950::MatrixBytes(m, n, lda), devIpiv, pivotBytes) ||
         lu950::Overlap(Workspace, workspaceBytes, devIpiv, pivotBytes)))
        error = 6;
    if (error) return lu950::Info(devInfo, 1, -error, stream);
    if (!lu950::DeviceSpan(devInfo, sizeof(int)) ||
        lu950::Overlap(devInfo, sizeof(int), A, lu950::MatrixBytes(m, n, lda)) ||
        lu950::Overlap(devInfo, sizeof(int), devIpiv, pivotBytes) ||
        lu950::Overlap(devInfo, sizeof(int), Workspace, workspaceBytes))
        return ACLSOLVER_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return lu950::Info(devInfo, 1, 0, stream);
#ifdef ACLSOLVER_HAS_LU950
    return workspaceBytes ? Lu950GetrfBlocked(handle, m, n, A, lda, devIpiv, devInfo, Workspace, workspaceBytes)
                          : Lu950Getrf(handle, m, n, A, lda, devIpiv, devInfo);
#else
    return ACLSOLVER_STATUS_NOT_SUPPORTED;
#endif
}
