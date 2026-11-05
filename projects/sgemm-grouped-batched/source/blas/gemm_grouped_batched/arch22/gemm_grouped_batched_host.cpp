/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "gemm_grouped_batched_tiling.h"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace {
bool ValidTranspose(aclblasOperation_t operation) { return operation == ACLBLAS_OP_N || operation == ACLBLAS_OP_T; }

bool ValidGroup(aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, int lda, int ldb, int ldc, int size)
{
    return ValidTranspose(ta) && ValidTranspose(tb) && m >= 0 && n >= 0 && k >= 0 && size >= 0 &&
           lda >= std::max(1, ta == ACLBLAS_OP_N ? m : k) && ldb >= std::max(1, tb == ACLBLAS_OP_N ? k : n) &&
           ldc >= std::max(1, m);
}

GemmGrouped22Batch BuildBatch(
    const GemmGrouped22Params& params, uint64_t start, uint32_t count, const float* const* Aarray,
    const float* const* Barray, float* const* Carray)
{
    GemmGrouped22Batch batch{};
    batch.group = params;
    batch.count = count;
    batch.groupCount = 1;
    batch.groupEnds[0] = count;
    batch.tileRows = 128;
    batch.tileCols = 1024;
    batch.outputLd = params.ldc;
    for (uint32_t i = 0; i < count; ++i) {
        batch.a[i] = reinterpret_cast<uint64_t>(Aarray[start + i]);
        batch.b[i] = reinterpret_cast<uint64_t>(Barray[start + i]);
        batch.c[i] = reinterpret_cast<uint64_t>(Carray[start + i]);
        batch.output[i] = batch.c[i];
    }
    return batch;
}

// Scalar stores must never assign the same output cache line to different
// cores.
bool IndependentScalarOutputs(const GemmGrouped22Batch& batch)
{
    const uint64_t bytes = ((static_cast<uint64_t>(batch.group.n) - 1) * batch.group.ldc + batch.group.m) * 4;
    for (uint32_t i = 0; i < batch.count; ++i) {
        if (batch.c[i] > std::numeric_limits<uint64_t>::max() - (bytes - 1)) {
            return false;
        }
        const uint64_t first = batch.c[i] / 64;
        const uint64_t last = (batch.c[i] + bytes - 1) / 64;
        for (uint32_t j = 0; j < i; ++j) {
            const uint64_t otherFirst = batch.c[j] / 64;
            const uint64_t otherLast = (batch.c[j] + bytes - 1) / 64;
            if (first <= otherLast && otherFirst <= last) {
                return false;
            }
        }
    }
    return true;
}

// Adjacent independent small groups can share one launch without changing API order.
bool SameSmallGroup(const GemmGrouped22Params& a, const GemmGrouped22Params& b)
{
    return a.m <= 16 && a.n <= 16 && a.k > 0 && a.k <= 64 && !a.transA && !a.transB && a.alpha == 1.0f &&
           a.beta == 0.0f && a.m == b.m && a.n == b.n && a.k == b.k && a.lda == b.lda && a.ldb == b.ldb &&
           a.ldc == b.ldc && !b.transA && !b.transB && b.alpha == 1.0f && b.beta == 0.0f;
}

bool ShareCacheLine(uint64_t a, uint64_t aBytes, uint64_t b, uint64_t bBytes)
{
    if (a > std::numeric_limits<uint64_t>::max() - (aBytes - 1) ||
        b > std::numeric_limits<uint64_t>::max() - (bBytes - 1)) {
        return true;
    }
    return a / 64 <= (b + bBytes - 1) / 64 && b / 64 <= (a + aBytes - 1) / 64;
}

bool IndependentSmallInputs(const GemmGrouped22Batch& batch)
{
    const auto& p = batch.group;
    const uint64_t aBytes = ((static_cast<uint64_t>(p.k) - 1) * p.lda + p.m) * 4;
    const uint64_t bBytes = ((static_cast<uint64_t>(p.n) - 1) * p.ldb + p.k) * 4;
    const uint64_t cBytes = ((static_cast<uint64_t>(p.n) - 1) * p.ldc + p.m) * 4;
    for (uint32_t i = 0; i < batch.count; ++i) {
        for (uint32_t j = 0; j < batch.count; ++j) {
            if (ShareCacheLine(batch.c[i], cBytes, batch.a[j], aBytes) ||
                ShareCacheLine(batch.c[i], cBytes, batch.b[j], bBytes)) {
                return false;
            }
        }
    }
    return true;
}

bool EligibleSmallMultiGroup(const GemmGrouped22Params& p)
{
    return p.m > 0 && p.n > 0 && (p.m > 16 || p.n > 16) && p.m <= 256 && p.n <= 256 && p.k > 0 && p.k <= 256 &&
           !p.transA && !p.transB && p.alpha == 1.0f && p.beta == 0.0f && p.lda == p.m && p.ldb == p.k && p.ldc == p.m;
}

bool IndependentSmallMultiInputs(const GemmGrouped22Batch& batch)
{
    uint64_t aBytes[GEMM_GROUPED22_BATCH_CAPACITY]{};
    uint64_t bBytes[GEMM_GROUPED22_BATCH_CAPACITY]{};
    uint64_t cBytes[GEMM_GROUPED22_BATCH_CAPACITY]{};
    uint32_t begin = 0;
    for (uint32_t g = 0; g < batch.groupCount; ++g) {
        const auto& p = g == 0 ? batch.group : batch.extraGroups[g - 1];
        for (uint32_t i = begin; i < batch.groupEnds[g]; ++i) {
            aBytes[i] = static_cast<uint64_t>(p.m) * p.k * 4;
            bBytes[i] = static_cast<uint64_t>(p.n) * p.k * 4;
            cBytes[i] = static_cast<uint64_t>(p.m) * p.n * 4;
        }
        begin = batch.groupEnds[g];
    }
    for (uint32_t i = 0; i < batch.count; ++i) {
        for (uint32_t j = 0; j < batch.count; ++j) {
            if ((i != j && ShareCacheLine(batch.c[i], cBytes[i], batch.c[j], cBytes[j])) ||
                ShareCacheLine(batch.c[i], cBytes[i], batch.a[j], aBytes[j]) ||
                ShareCacheLine(batch.c[i], cBytes[i], batch.b[j], bBytes[j])) {
                return false;
            }
        }
    }
    return true;
}

template <typename ParamsAt>
bool LaunchSmallMultiGroups(
    aclblasHandle_t handle, int first, int groupCount, const int* sizes, uint64_t offset, uint32_t vectors,
    const float* const* ap, const float* const* bp, float* const* cp, ParamsAt paramsAt, int& groupsUsed,
    uint32_t& matricesUsed)
{
    const uint32_t cubes = GetAicCoreCount();
    if (first + 1 >= groupCount || !EligibleSmallMultiGroup(paramsAt(first)) || vectors != 2 * cubes || vectors > 64 ||
        !GetEffectiveWorkspace(handle) || GetEffectiveWorkspaceSize(handle) < 32768 ||
        reinterpret_cast<uint64_t>(GetEffectiveWorkspace(handle)) % 64) {
        return false;
    }
    GemmGrouped22Batch batch{};
    for (int g = first; g < groupCount && batch.groupCount < 8; ++g) {
        const auto params = paramsAt(g);
        if (!EligibleSmallMultiGroup(params) || sizes[g] <= 0 ||
            static_cast<uint32_t>(sizes[g]) > GEMM_GROUPED22_BATCH_CAPACITY - batch.count) {
            break;
        }
        const uint32_t index = batch.groupCount;
        if (index == 0)
            batch.group = params;
        else
            batch.extraGroups[index - 1] = params;
        for (int i = 0; i < sizes[g]; ++i) {
            const uint64_t at = offset + batch.count;
            batch.a[batch.count] = reinterpret_cast<uint64_t>(ap[at]);
            batch.b[batch.count] = reinterpret_cast<uint64_t>(bp[at]);
            batch.c[batch.count] = reinterpret_cast<uint64_t>(cp[at]);
            batch.output[batch.count] = batch.c[batch.count];
            ++batch.count;
        }
        const uint64_t largest = static_cast<uint64_t>(std::max(params.m, params.n)) * params.k;
        batch.groupWorkers[index] = std::min<uint64_t>((largest + 4095) / 4096, std::max(1U, vectors / sizes[g]));
        batch.groupRows[index] = static_cast<uint32_t>(sizes[g]) < cubes ? 64 : 512;
        batch.groupCols[index] = static_cast<uint32_t>(sizes[g]) < cubes ? 64 : 1024;
        batch.groupEnds[index] = batch.count;
        ++batch.groupCount;
    }
    if (batch.groupCount < 2 || !IndependentSmallMultiInputs(batch))
        return false;
    batch.flags = reinterpret_cast<uint64_t>(GetEffectiveWorkspace(handle));
    batch.scanWorkers = batch.groupWorkers[0];
    groupsUsed = batch.groupCount;
    matricesUsed = batch.count;
    bool uniform = true;
    for (uint32_t g = 1; g < batch.groupCount; ++g) {
        const auto& p = batch.extraGroups[g - 1];
        uniform &= p.m == batch.group.m && p.n == batch.group.n && p.k == batch.group.k;
    }
    // All eligible groups already have identical NN/scalar/layout contracts.
    // Equal shapes can share one Matmul initialization on each Cube core.
    if (uniform) {
        batch.groupCount = 1;
        batch.groupEnds[0] = batch.count;
        const uint64_t largest = static_cast<uint64_t>(std::max(batch.group.m, batch.group.n)) * batch.group.k;
        batch.scanWorkers = std::min<uint64_t>((largest + 4095) / 4096, std::max(1U, vectors / batch.count));
        batch.outputLd = batch.group.ldc;
        batch.tileRows = 128;
        batch.tileCols = 128;
        if (batch.group.m <= 64 && batch.group.n <= 64 && batch.group.k <= 64)
            batch.scanWorkers = 0;
    }
    GemmGrouped22Cube(cubes, vectors, handle->stream, batch);
    return true;
}

aclblasStatus_t LaunchGroup(
    aclblasHandle_t handle, const GemmGrouped22Params& params, uint64_t offset, int groupSize, uint32_t vectors,
    const float* const* Aarray, const float* const* Barray, float* const* Carray)
{
    if (groupSize == 0)
        return ACLBLAS_STATUS_SUCCESS;
    const bool combine = params.alpha != 1.0f || params.beta != 0.0f;
    const uint64_t outputLd = combine ? (static_cast<uint64_t>(params.m) + 7) / 8 * 8 : params.ldc;
    const uint64_t plane = outputLd * params.n;
    const bool smallCube = params.m <= 64 && params.n <= 64 && params.k <= 64;
    const uint64_t capacity =
        combine ? std::min<uint64_t>(
                      GEMM_GROUPED22_BATCH_CAPACITY, GetEffectiveWorkspaceSize(handle) / sizeof(float) / plane) :
                  (smallCube ? GEMM_GROUPED22_POINTER_CAPACITY : GEMM_GROUPED22_BATCH_CAPACITY);
    const bool boundedAlpha = params.alpha > -65536.0f && params.alpha < 65536.0f;
    const bool boundedBeta = params.beta > -65536.0f && params.beta < 65536.0f;
    const bool skinny = params.m < 16 || params.n < 16;
    const bool smallVector = !params.transA && !params.transB && !combine && params.m <= 16 && params.n <= 16 &&
                             params.k <= 64 && groupSize > 1;
    const bool ordered = (combine || skinny) && params.k >= 512 && boundedAlpha && boundedBeta &&
                         params.alpha != 0.0f && GetEffectiveWorkspace(handle) &&
                         GetEffectiveWorkspaceSize(handle) >= 64;
    if (ordered) {
        const uint32_t capacity =
            std::min<uint64_t>(GEMM_GROUPED22_BATCH_CAPACITY, GetEffectiveWorkspaceSize(handle) / 64);
        for (int j = 0; j < groupSize;) {
            auto batch =
                BuildBatch(params, offset + j, std::min<uint32_t>(capacity, groupSize - j), Aarray, Barray, Carray);
            batch.flags = reinterpret_cast<uint64_t>(GetEffectiveWorkspace(handle));
            GemmGrouped22Ordered(vectors, handle->stream, batch);
            j += batch.count;
        }
    } else if (
        !smallVector && params.alpha != 0.0f && boundedAlpha && boundedBeta && params.k > 0 && params.m >= 16 &&
        params.n >= 16 && outputLd <= std::numeric_limits<int>::max() &&
        (!combine || (capacity > 0 && GetEffectiveWorkspace(handle)))) {
        const uint32_t cubes = GetAicCoreCount();
        if (cubes == 0) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        for (int j = 0; j < groupSize;) {
            auto batch =
                BuildBatch(params, offset + j, std::min<uint64_t>(capacity, groupSize - j), Aarray, Barray, Carray);
            batch.combine = combine;
            batch.outputLd = static_cast<int>(outputLd);
            if (combine) {
                for (uint32_t i = 0; i < batch.count; ++i) {
                    batch.output[i] =
                        reinterpret_cast<uint64_t>(GetEffectiveWorkspace(handle)) + i * plane * sizeof(float);
                }
            }
            if (!combine && vectors == 2 * cubes && params.lda == (params.transA ? params.k : params.m) &&
                params.ldb == (params.transB ? params.n : params.k) && GetEffectiveWorkspace(handle) &&
                GetEffectiveWorkspaceSize(handle) >= 4096 && vectors <= 64 &&
                reinterpret_cast<uint64_t>(GetEffectiveWorkspace(handle)) % 64 == 0) {
                const uint64_t largestInput = static_cast<uint64_t>(std::max(params.m, params.n)) * params.k;
                const uint32_t inputBlocks =
                    static_cast<uint32_t>(std::min<uint64_t>(64, (largestInput + 4095) / 4096));
                batch.scanWorkers = std::min(inputBlocks, std::max(1U, vectors / batch.count));
                batch.flags = reinterpret_cast<uint64_t>(GetEffectiveWorkspace(handle));
            }
            // Small matrix batches avoid the fixed full-core synchronization cost.
            if (params.m <= 64 && params.n <= 64 && params.k <= 64)
                batch.scanWorkers = 0;
            const uint64_t wideTasks = ((static_cast<uint64_t>(params.m) + 1023) / 1024) *
                                       ((static_cast<uint64_t>(params.n) + 511) / 512) * batch.count;
            batch.tileRows = wideTasks >= cubes ? 512 : 128;
            if (batch.count < cubes && params.m <= 256 && params.n <= 256) {
                batch.tileRows = 64;
                batch.tileCols = 64;
            }
            const uint64_t tasks = ((static_cast<uint64_t>(params.m) + batch.tileCols - 1) / batch.tileCols) *
                                   ((static_cast<uint64_t>(params.n) + batch.tileRows - 1) / batch.tileRows) *
                                   batch.count;
            GemmGrouped22Cube(
                batch.scanWorkers ? cubes : std::min<uint64_t>(cubes, tasks), vectors, handle->stream, batch);
            j += batch.count;
        }
    } else {
        // Scalar stores share cache lines: one owner per matrix on the fallback
        // path.
        for (int j = 0; j < groupSize;) {
            if (params.m <= 16 && params.n <= 16 && params.k <= 64 && (groupSize - j > 1 || smallVector)) {
                const auto batch = BuildBatch(
                    params, offset + j,
                    std::min<uint32_t>(
                        smallVector ? GEMM_GROUPED22_POINTER_CAPACITY : GEMM_GROUPED22_BATCH_CAPACITY, groupSize - j),
                    Aarray, Barray, Carray);
                if (IndependentScalarOutputs(batch)) {
                    GemmGrouped22ScalarBatch(std::min(vectors, batch.count), handle->stream, batch);
                    j += batch.count;
                    continue;
                }
            }
            GemmGrouped22Launch(
                1, handle->stream, reinterpret_cast<uint8_t*>(const_cast<float*>(Aarray[offset + j])),
                reinterpret_cast<uint8_t*>(const_cast<float*>(Barray[offset + j])),
                reinterpret_cast<uint8_t*>(Carray[offset + j]), params);
            ++j;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace

aclblasStatus_t aclblasSgemmGroupedBatched(
    aclblasHandle_t handle, int groupCount, const aclblasOperation_t* transaArray,
    const aclblasOperation_t* transbArray, const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray, const float* const* Aarray, const int* ldaArray, const float* const* Barray,
    const int* ldbArray, const float* betaArray, float* const* Carray, const int* ldcArray, const int* groupSizeArray)
{
    if (!handle) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (groupCount < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (groupCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (!transaArray || !transbArray || !mArray || !nArray || !kArray || !alphaArray || !betaArray || !ldaArray ||
        !ldbArray || !ldcArray || !groupSizeArray || !Aarray || !Barray || !Carray) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Check every group before inspecting flattened addresses or submitting any
    // work.
    uint64_t total = 0;
    bool hasWork = false;
    for (int g = 0; g < groupCount; ++g) {
        if (!ValidGroup(
                transaArray[g], transbArray[g], mArray[g], nArray[g], kArray[g], ldaArray[g], ldbArray[g], ldcArray[g],
                groupSizeArray[g])) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        total += static_cast<uint64_t>(groupSizeArray[g]);
        if (total > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        hasWork |= mArray[g] > 0 && nArray[g] > 0 && groupSizeArray[g] > 0;
    }
    // Inactive groups retain their slots in the caller's flattened pointer
    // arrays.
    for (uint64_t index = 0; index < total; ++index) {
        if (!Aarray[index] || !Barray[index] || !Carray[index]) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (!hasWork) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint32_t vectors = GetAivCoreCount();
    if (vectors == 0) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    auto paramsFor = [&](int g) {
        return GemmGrouped22Params{
            mArray[g],
            nArray[g],
            kArray[g],
            ldaArray[g],
            ldbArray[g],
            ldcArray[g],
            transaArray[g] == ACLBLAS_OP_T,
            transbArray[g] == ACLBLAS_OP_T,
            alphaArray[g],
            betaArray[g]};
    };
    uint64_t offset = 0;
    for (int g = 0; g < groupCount;) {
        int groupsUsed = 0;
        uint32_t matricesUsed = 0;
        if (LaunchSmallMultiGroups(
                handle, g, groupCount, groupSizeArray, offset, vectors, Aarray, Barray, Carray, paramsFor, groupsUsed,
                matricesUsed)) {
            offset += matricesUsed;
            g += groupsUsed;
            continue;
        }

        const auto params = paramsFor(g);
        int count = groupSizeArray[g];
        int next = g + 1;
        if (params.m > 0 && params.n > 0) {
            while (next < groupCount && count > 0 && count < static_cast<int>(GEMM_GROUPED22_BATCH_CAPACITY) &&
                   groupSizeArray[next] > 0 &&
                   groupSizeArray[next] <= static_cast<int>(GEMM_GROUPED22_BATCH_CAPACITY) - count &&
                   SameSmallGroup(params, paramsFor(next))) {
                const auto batch = BuildBatch(params, offset, count + groupSizeArray[next], Aarray, Barray, Carray);
                if (!IndependentScalarOutputs(batch) || !IndependentSmallInputs(batch))
                    break;
                count += groupSizeArray[next++];
            }
            const auto status = LaunchGroup(handle, params, offset, count, vectors, Aarray, Barray, Carray);
            if (status != ACLBLAS_STATUS_SUCCESS)
                return status;
        }
        offset += static_cast<uint64_t>(count);
        g = next;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
