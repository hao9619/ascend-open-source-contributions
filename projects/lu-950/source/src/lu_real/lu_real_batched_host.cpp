/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lu_real_host.hpp"

namespace
{
int BatchArguments(int n, const void* a, int lda, const int* pivots, const void* c, int ldc, const int* info, int batch,
                   bool matinv)
{
    if (n < 0 || n > (matinv ? 32 : 4096)) return 1;
    const int cArgument = matinv ? 4 : 5, infoArgument = matinv ? 6 : 7, batchArgument = matinv ? 7 : 8;
    if (n > 0 && batch > 0 && !lu950::DeviceSpan(a, static_cast<size_t>(batch) * sizeof(void*))) return 2;
    if (lda < std::max(1, n)) return 3;
    if (!matinv && n > 0 && batch > 0 && pivots &&
        !lu950::DeviceSpan(pivots, static_cast<size_t>(batch) * n * sizeof(int)))
        return 4;
    if (n > 0 && batch > 0 && !lu950::DeviceSpan(c, static_cast<size_t>(batch) * sizeof(void*))) return cArgument;
    if (ldc < std::max(1, n)) return cArgument + 1;
    if (batch > 0 && !lu950::DeviceSpan(info, static_cast<size_t>(batch) * sizeof(int))) return infoArgument;
    if (batch < 0) return batchArgument;
    if (n > 0 && batch > 0)
    {
        const size_t arrayBytes = static_cast<size_t>(batch) * sizeof(void*);
        const size_t infoBytes = static_cast<size_t>(batch) * sizeof(int);
        const size_t pivotBytes = static_cast<size_t>(batch) * n * sizeof(int);
        if (lu950::Overlap(c, arrayBytes, info, infoBytes) || lu950::Overlap(a, arrayBytes, info, infoBytes) ||
            (!matinv && lu950::Overlap(pivots, pivotBytes, info, infoBytes)))
            return infoArgument;
    }
    return 0;
}
}  // namespace

aclsolverStatus_t aclsolverSgetriBatched(aclsolverHandle_t handle, int n, const float* const Aarray[], int lda,
                                         const int* PivotArray, float* const Carray[], int ldc, int* infoArray,
                                         int batchSize)
{
    aclrtStream stream = nullptr;
    const auto status = lu950::Stream(handle, stream);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    const int error = BatchArguments(n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize, false);
    if (error == 7) return ACLSOLVER_STATUS_INVALID_VALUE;
    if (error) return lu950::Info(infoArray, std::max(0, batchSize), -error, stream);
    if (batchSize == 0) return ACLSOLVER_STATUS_SUCCESS;
    if (n == 0) return lu950::Info(infoArray, batchSize, 0, stream);
#ifdef ACLSOLVER_HAS_LU950
    return Lu950GetriBatched(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize);
#else
    return ACLSOLVER_STATUS_NOT_SUPPORTED;
#endif
}

aclsolverStatus_t aclsolverSmatinvBatched(aclsolverHandle_t handle, int n, const float* const A[], int lda,
                                          float* const Ainv[], int lda_inv, int* info, int batchSize)
{
    aclrtStream stream = nullptr;
    const auto status = lu950::Stream(handle, stream);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    const int error = BatchArguments(n, A, lda, nullptr, Ainv, lda_inv, info, batchSize, true);
    if (error == 6) return ACLSOLVER_STATUS_INVALID_VALUE;
    if (error) return lu950::Info(info, std::max(0, batchSize), -error, stream);
    if (batchSize == 0) return ACLSOLVER_STATUS_SUCCESS;
    if (n == 0) return lu950::Info(info, batchSize, 0, stream);
#ifdef ACLSOLVER_HAS_LU950
    return Lu950MatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
#else
    return ACLSOLVER_STATUS_NOT_SUPPORTED;
#endif
}
