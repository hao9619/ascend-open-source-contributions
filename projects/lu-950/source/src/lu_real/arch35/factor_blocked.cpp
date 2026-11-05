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
#include "factor_internal.h"
#include "kernel_operator.h"
#include "launch_internal.h"
#include "tiling/platform/platform_ascendc.h"
using namespace AscendC;
namespace
{
constexpr int panel = 64;
constexpr int lanes = 32;
constexpr int maxLanes = 64;
template <HardEvent Event>
__aicore__ inline void NestedFence()
{
    const auto event = static_cast<int32_t>(GetTPipePtr()->FetchEventID(Event));
    SetFlag<Event>(event);
    WaitFlag<Event>(event);
}
}  // namespace

__global__ __aicore__ void lu950_factor_outer(int m, int n, int lda, int start, int nb, GM_ADDR a, GM_ADDR pivots,
                                              GM_ADDR negativeUpper)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECIN> expandedBuffer, diagonalBuffer, pivotBuffer;
    TBuf<TPosition::VECCALC> compactBuffer, indexBuffer, offsetsBuffer, temporaryBuffer;
    TBuf<TPosition::VECOUT> packedBuffer;
    pipe.InitBuffer(expandedBuffer, 2 * panel * lanes * 8 * sizeof(float));
    pipe.InitBuffer(compactBuffer, 2 * panel * lanes * sizeof(float));
    pipe.InitBuffer(diagonalBuffer, panel * panel * sizeof(float));
    pipe.InitBuffer(pivotBuffer, panel * sizeof(int));
    pipe.InitBuffer(indexBuffer, 3 * panel * sizeof(int));
    pipe.InitBuffer(offsetsBuffer, (maxLanes + panel + maxLanes * 8) * sizeof(uint32_t));
    pipe.InitBuffer(temporaryBuffer, maxLanes * sizeof(float));
    pipe.InitBuffer(packedBuffer, panel * maxLanes * sizeof(float));
    auto expanded = expandedBuffer.Get<float>(), compact = compactBuffer.Get<float>();
    auto diagonal = diagonalBuffer.Get<float>(), packed = packedBuffer.Get<float>();
    auto temporary = temporaryBuffer.Get<float>();
    auto cachedPivots = pivotBuffer.Get<int>(), rows = indexBuffer.Get<int>();
    auto targets = rows[2 * panel];
    auto gatherOffsets = offsetsBuffer.Get<uint32_t>();
    auto packOffsets = gatherOffsets[maxLanes], expandOffsets = packOffsets[panel];
    GlobalTensor<float> matrix, negative;
    GlobalTensor<int> gpivots;
    matrix.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
    negative.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(negativeUpper));
    gpivots.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(pivots));
    if (pivots != nullptr)
        DataCopyPad(cachedPivots, gpivots[start],
                    DataCopyExtParams{1, static_cast<uint32_t>(nb * sizeof(int)), 0, 0, 0},
                    DataCopyPadExtParams<int>{true, 0, static_cast<uint8_t>((8 - nb % 8) % 8), 0});
    for (int col = 0; col < nb; ++col)
        DataCopyPad(diagonal[col * panel], matrix[static_cast<uint64_t>(start + col) * lda + start],
                    DataCopyExtParams{1, static_cast<uint32_t>(nb * sizeof(float)), 0, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - nb % 8) % 8), 0.f});
    NestedFence<HardEvent::MTE2_S>();
    const int after = start + nb;
    int count = nb;
    for (int k = 0; k < nb; ++k) rows.SetValue(k, start + k);
    for (int k = 0; k < nb; ++k)
    {
        const int pivot = pivots == nullptr ? start + k : cachedPivots.GetValue(k) - 1;
        int index = pivot - start;
        if (pivot >= after)
        {
            index = nb;
            while (index < count && rows.GetValue(index) != pivot) ++index;
            if (index == count) rows.SetValue(count++, pivot);
        }
        targets.SetValue(k, index);
    }
    // The fixed UB budget holds either 128 rows x 32 columns or 64 rows x 64 columns.
    const int width = count <= panel ? maxLanes : lanes;
    for (int lane = 0; lane < width; ++lane) gatherOffsets.SetValue(lane, lane * 8 * sizeof(float));
    for (int k = 0; k < panel; ++k) packOffsets.SetValue(k, k * width * sizeof(float));
    for (int k = 0; k < width * 8; ++k) expandOffsets.SetValue(k, k / 8 * sizeof(float));
    NestedFence<HardEvent::S_V>();
    const int oldTiles = (start + width - 1) / width;
    const int tasks = oldTiles + (n - after + width - 1) / width;
    for (int task = GetBlockIdx(); task < tasks; task += GetBlockNum())
    {
        const bool trailing = task >= oldTiles;
        const int col = trailing ? after + (task - oldTiles) * width : task * width;
        const int end = trailing ? n : start;
        const int columns = end - col < width ? end - col : width;
        const DataCopyExtParams gatherCopy{static_cast<uint16_t>(columns), sizeof(float),
                                           static_cast<uint32_t>((lda - 1) * sizeof(float)), 0, 0};
        for (int row = 0; row < count; ++row)
            DataCopyPad(expanded[row * width * 8], matrix[static_cast<uint64_t>(col) * lda + rows.GetValue(row)],
                        gatherCopy, DataCopyPadExtParams<float>{true, 0, 7, 0.f});
        NestedFence<HardEvent::MTE2_V>();
        for (int row = 0; row < count; ++row)
        {
            Gather(compact[row * width], expanded, gatherOffsets, row * width * 8 * sizeof(float), columns);
            PipeBarrier<PIPE_V>();
        }
        // Apply the complete outer panel's pivot sequence to vectors of RHS columns.
        for (int k = 0; k < nb; ++k)
        {
            const int target = targets.GetValue(k);
            if (target == k) continue;
            Muls(temporary, compact[k * width], 1.f, columns);
            PipeBarrier<PIPE_V>();
            Muls(compact[k * width], compact[target * width], 1.f, columns);
            PipeBarrier<PIPE_V>();
            Muls(compact[target * width], temporary, 1.f, columns);
            PipeBarrier<PIPE_V>();
        }
        if (trailing)
        {
            for (int row = 1; row < nb; ++row)
                for (int k = 0; k < row; ++k)
                {
                    Axpy(compact[row * width], compact[k * width], -diagonal.GetValue(k * panel + row), columns);
                    PipeBarrier<PIPE_V>();
                }
            Duplicate(packed, 0.f, columns * panel);
            PipeBarrier<PIPE_V>();
            for (int lane = 0; lane < columns; ++lane)
            {
                Gather(packed[lane * panel], compact, packOffsets, lane * sizeof(float), nb);
                PipeBarrier<PIPE_V>();
            }
            Muls(packed, packed, -1.f, columns * panel);
            NestedFence<HardEvent::V_MTE3>();
            DataCopyPad(negative[static_cast<uint64_t>(col - after) * panel], packed,
                        DataCopyExtParams{1, static_cast<uint32_t>(columns * panel * sizeof(float)), 0, 0, 0});
        }
        for (int row = 0; row < count; ++row)
        {
            Gather(expanded[row * width * 8], compact[row * width], expandOffsets, 0, columns * 8);
            PipeBarrier<PIPE_V>();
        }
        NestedFence<HardEvent::V_MTE3>();
        for (int row = 0; row < count; ++row)
            DataCopyPad(matrix[static_cast<uint64_t>(col) * lda + rows.GetValue(row)], expanded[row * width * 8],
                        DataCopyExtParams{static_cast<uint16_t>(columns), sizeof(float), 0,
                                          static_cast<uint32_t>((lda - 1) * sizeof(float)), 0});
        NestedFence<HardEvent::MTE3_MTE2>();
        NestedFence<HardEvent::MTE3_V>();
    }
}

size_t Lu950GetrfBlockedWorkspaceBytes(int m, int n)
{
    return m >= 1 && m <= 4096 && n >= 1 && n <= 8192
               ? Lu950CubeWorkspaceBytes() + static_cast<size_t>(n) * panel * sizeof(float)
               : 0;
}

aclsolverStatus_t Lu950GetrfBlocked(aclsolverHandle_t handle, int m, int n, float *a, int lda, int *pivots, int *info,
                                    void *workspace, size_t workspaceBytes)
{
    if (!handle) return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    const size_t required = Lu950GetrfBlockedWorkspaceBytes(m, n);
    if (!required || lda < m || !a || !info || !workspace || workspaceBytes < required)
        return ACLSOLVER_STATUS_INVALID_VALUE;
    aclrtStream stream = nullptr;
    const auto status = aclsolverGetStream(handle, &stream);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform) return ACLSOLVER_STATUS_NOT_INITIALIZED;
    const unsigned blocks = platform->GetCoreNumAiv();
    if (!blocks) return ACLSOLVER_STATUS_ARCH_MISMATCH;
    auto *negative = reinterpret_cast<float *>(static_cast<uint8_t *>(workspace) + Lu950CubeWorkspaceBytes());
    const int limit = m < n ? m : n;
    for (int start = 0; start < limit; start += panel)
    {
        const int nb = limit - start < panel ? limit - start : panel;
        const int after = start + nb;
        for (int inner = start; inner < after; inner += 8)
            Lu950FactorInnerLaunch(m, after, start, a, lda, inner, after - inner < 8 ? after - inner : 8, pivots, info,
                                   blocks, stream);
        lu950_factor_outer<<<blocks, nullptr, stream>>>(m, n, lda, start, nb, reinterpret_cast<uint8_t *>(a),
                                                        reinterpret_cast<uint8_t *>(pivots),
                                                        reinterpret_cast<uint8_t *>(negative));
        if (after < m && after < n)
            if (Lu950CubeMatmul(n - after, m - after, nb, negative, panel,
                                a + static_cast<uint64_t>(start) * lda + after, lda,
                                a + static_cast<uint64_t>(after) * lda + after, lda, workspace, stream, true) != 0)
                return ACLSOLVER_STATUS_INTERNAL_ERROR;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}
