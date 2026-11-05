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

aclsolverStatus_t aclsolverSgetri_bufferSize(aclsolverHandle_t handle, int n, float* A, int lda, int* Lwork)
{
    aclrtStream stream = nullptr;
    const auto status = lu950::Stream(handle, stream);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    if (!Lwork || n < 0 || n > 4096 || lda < std::max(1, n)) return ACLSOLVER_STATUS_INVALID_VALUE;
#ifdef ACLSOLVER_HAS_LU950
    return lu950::QuerySize(n > 0 ? Lu950GetriWorkspaceBytes(n) : 0, Lwork);
#else
    return ACLSOLVER_STATUS_NOT_SUPPORTED;
#endif
}

aclsolverStatus_t aclsolverSgetri(aclsolverHandle_t handle, int n, float* A, int lda, int* devIpiv, float* Workspace,
                                  int Lwork, int* devInfo)
{
    aclrtStream stream = nullptr;
    const auto status = lu950::Stream(handle, stream);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    int error = n < 0 || n > 4096 ? 1 : 0;
    if (!error && n > 0 && !lu950::DeviceSpan(A, sizeof(float))) error = 2;
    if (!error && lda < std::max(1, n)) error = 3;
    if (!error && n > 0 && !lu950::DeviceSpan(A, lu950::MatrixBytes(n, n, lda))) error = 2;
    const size_t pivotBytes = static_cast<size_t>(std::max(0, n)) * sizeof(int);
    if (!error && devIpiv && n > 0 &&
        (!lu950::DeviceSpan(devIpiv, pivotBytes) ||
         lu950::Overlap(A, lu950::MatrixBytes(n, n, lda), devIpiv, pivotBytes)))
        error = 4;
    size_t workspaceBytes = 0;
#ifdef ACLSOLVER_HAS_LU950
    if (!error && n > 0) workspaceBytes = Lu950GetriWorkspaceBytes(n);
#endif
    if (!error && workspaceBytes &&
        (!lu950::DeviceSpan(Workspace, workspaceBytes) ||
         lu950::Overlap(A, lu950::MatrixBytes(n, n, lda), Workspace, workspaceBytes) ||
         lu950::Overlap(devIpiv, pivotBytes, Workspace, workspaceBytes)))
        error = 5;
    if (!error && (Lwork < 0 || static_cast<size_t>(Lwork) < workspaceBytes / sizeof(float))) error = 6;
    if (error) return lu950::Info(devInfo, 1, -error, stream);
    if (!lu950::DeviceSpan(devInfo, sizeof(int)) ||
        lu950::Overlap(devInfo, sizeof(int), A, lu950::MatrixBytes(n, n, lda)) ||
        lu950::Overlap(devInfo, sizeof(int), devIpiv, pivotBytes) ||
        lu950::Overlap(devInfo, sizeof(int), Workspace, workspaceBytes))
        return ACLSOLVER_STATUS_INVALID_VALUE;
    if (n == 0) return lu950::Info(devInfo, 1, 0, stream);
#ifdef ACLSOLVER_HAS_LU950
    return Lu950Getri(handle, n, A, lda, devIpiv, devInfo, Workspace, workspaceBytes);
#else
    return ACLSOLVER_STATUS_NOT_SUPPORTED;
#endif
}
