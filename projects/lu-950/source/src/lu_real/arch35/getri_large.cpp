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
#include "kernel_operator.h"
#include "launch_internal.h"

namespace AscendC
{
namespace
{
constexpr int diagonalTile = 32;
template <HardEvent Event>
__aicore__ inline void LargeFence()
{
    const auto event = static_cast<int32_t>(GetTPipePtr()->FetchEventID(Event));
    SetFlag<Event>(event);
    WaitFlag<Event>(event);
}

__aicore__ float CompensatedSubtract(float value, float product, float &correction)
{
    const float term = -product - correction;
    const float updated = value + term;
    correction = (updated - value) - term;
    return updated;
}
}  // namespace

#ifdef LU950_VECTOR_DIAGONAL
#include "getri_diagonal_vector.hpp"
#endif
}  // namespace AscendC

using namespace AscendC;

__aicore__ void LargeInitialize(int n, int lda, GM_ADDR a, GM_ADDR negative, GM_ADDR lower, GM_ADDR upper,
                                GM_ADDR temporary, int workspaceLd, int indirect, int batchIndex, int firstTask,
                                int taskStride)
{
    if (workspaceLd <= 0 || workspaceLd > 4096) return;
    TPipe pipe;
    TBuf<TPosition::VECIN> inputBuffer;
    TBuf<TPosition::VECOUT> zeroBuffer;
    pipe.InitBuffer(inputBuffer, 4096 * sizeof(float));
    pipe.InitBuffer(zeroBuffer, 4096 * sizeof(float));
    auto input = inputBuffer.Get<float>(), zero = zeroBuffer.Get<float>();
    GM_ADDR inputAddress = a;
    if (indirect)
    {
        GlobalTensor<uint32_t> pointers;
        pointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(a));
        auto pointerScratch = input.ReinterpretCast<uint32_t>();
        DataCopyPad(pointerScratch, pointers[static_cast<uint64_t>(batchIndex) * 2], DataCopyExtParams{1, 8, 0, 0, 0},
                    DataCopyPadExtParams<uint32_t>{true, 0, 6, 0});
        LargeFence<HardEvent::MTE2_S>();
        inputAddress = reinterpret_cast<__gm__ uint8_t *>(pointerScratch.ReinterpretCast<uint64_t>().GetValue(0));
        LargeFence<HardEvent::S_MTE2>();
        LargeFence<HardEvent::S_V>();
    }
    GlobalTensor<float> ga, gn, gl, gu, gt;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputAddress));
    gn.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(negative));
    gl.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lower));
    gu.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(upper));
    gt.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(temporary));
    const int divisor = workspaceLd > 0 && workspaceLd <= 4096 ? workspaceLd : 4096;
    const int columnsPerTile = 4096 / divisor;
    const int tiles = (n + columnsPerTile - 1) / columnsPerTile;
    Duplicate(zero, 0.0f, 4096);
    LargeFence<HardEvent::V_MTE3>();
    for (int tile = firstTask; tile < tiles; tile += taskStride)
    {
        const int col = tile * columnsPerTile;
        const int columns = n - col < columnsPerTile ? n - col : columnsPerTile;
        const int elements = columns * workspaceLd;
        if (inputAddress != nullptr)
            DataCopyPad(input, ga[static_cast<uint64_t>(col) * lda],
                        DataCopyExtParams{static_cast<uint16_t>(columns), static_cast<uint32_t>(n * sizeof(float)),
                                          static_cast<uint32_t>((lda - n) * sizeof(float)), 0, 0},
                        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(workspaceLd - n), 0.f});
        else
            Duplicate(input, 0.0f, elements);
        LargeFence<HardEvent::MTE2_V>();
        Muls(input, input, -1.0f, elements);
        LargeFence<HardEvent::V_MTE3>();
        const DataCopyExtParams workspaceCopy{1, static_cast<uint32_t>(elements * sizeof(float)), 0, 0, 0};
        DataCopyPad(gn[static_cast<uint64_t>(col) * workspaceLd], input, workspaceCopy);
        DataCopyPad(gl[static_cast<uint64_t>(col) * workspaceLd], zero, workspaceCopy);
        DataCopyPad(gu[static_cast<uint64_t>(col) * workspaceLd], zero, workspaceCopy);
        DataCopyPad(gt[static_cast<uint64_t>(col) * workspaceLd], zero, workspaceCopy);
        LargeFence<HardEvent::MTE3_MTE2>();
        LargeFence<HardEvent::MTE3_V>();
    }
}

__global__ __aicore__ void lu950_getri_initialize(int n, int lda, GM_ADDR a, GM_ADDR negative, GM_ADDR lower,
                                                  GM_ADDR upper, GM_ADDR temporary, int workspaceLd, int indirect,
                                                  int batchIndex)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    LargeInitialize(n, lda, a, negative, lower, upper, temporary, workspaceLd, indirect, batchIndex, GetBlockIdx(),
                    GetBlockNum());
}

__aicore__ void LargeDiagonal(int n, GM_ADDR negative, GM_ADDR lower, GM_ADDR upper, GM_ADDR statuses, int workspaceLd,
                              int firstTask, int taskStride)
{
#ifdef LU950_VECTOR_DIAGONAL
    LargeDiagonalVector(n, negative, lower, upper, statuses, workspaceLd, firstTask, taskStride);
    return;
#endif
    TPipe pipe;
    TBuf<TPosition::VECIN> inputBuffer;
    TBuf<TPosition::VECOUT> resultBuffer, infoBuffer;
    pipe.InitBuffer(inputBuffer, diagonalTile * diagonalTile * sizeof(float));
    pipe.InitBuffer(resultBuffer, 2 * diagonalTile * sizeof(float));
    pipe.InitBuffer(infoBuffer, 32);
    auto input = inputBuffer.Get<float>(), lowerColumn = resultBuffer.Get<float>();
    auto upperColumn = lowerColumn[diagonalTile];
    auto localInfo = infoBuffer.Get<int>();
    GlobalTensor<float> gn, gl, gu;
    GlobalTensor<int> gs;
    gn.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(negative));
    gl.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lower));
    gu.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(upper));
    gs.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(statuses));
    for (int task = firstTask; task < n; task += taskStride)
    {
        const int start = task / diagonalTile * diagonalTile;
        const int col = task - start;
        const int nb = n - start < diagonalTile ? n - start : diagonalTile;
        const DataCopyExtParams copy{1, static_cast<uint32_t>(nb * sizeof(float)), 0, 0, 0};
        for (int k = 0; k < nb; ++k)
            DataCopyPad(input[k * diagonalTile], gn[static_cast<uint64_t>(start + k) * workspaceLd + start], copy,
                        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - nb % 8) % 8), 0.f});
        LargeFence<HardEvent::MTE2_V>();
        Muls(input, input, -1.0f, nb * diagonalTile);
        Duplicate(lowerColumn, 0.0f, 2 * diagonalTile);
        LargeFence<HardEvent::V_S>();
        int firstSingular = n + 1;
        for (int row = 0; row < nb; ++row)
            if (input.GetValue(row * diagonalTile + row) == 0.0f && firstSingular == n + 1)
                firstSingular = start + row + 1;
        // Each core solves complete columns of the two small triangular blocks.
        for (int row = col; row < nb; ++row)
        {
            float value = row == col ? 1.0f : 0.0f, correction = 0.0f;
            for (int k = col; k < row; ++k)
                value = CompensatedSubtract(value, input.GetValue(k * diagonalTile + row) * lowerColumn.GetValue(k),
                                            correction);
            lowerColumn.SetValue(row, value);
        }
        for (int row = col; row >= 0; --row)
        {
            float value = row == col ? 1.0f : 0.0f, correction = 0.0f;
            for (int k = row + 1; k <= col; ++k)
                value = CompensatedSubtract(value, input.GetValue(k * diagonalTile + row) * upperColumn.GetValue(k),
                                            correction);
            const float diagonal = input.GetValue(row * diagonalTile + row);
            upperColumn.SetValue(row, diagonal == 0.f ? 0.f : value / diagonal);
        }
        LargeFence<HardEvent::S_MTE3>();
        DataCopyPad(gl[static_cast<uint64_t>(task) * workspaceLd + start], lowerColumn, copy);
        DataCopyPad(gu[static_cast<uint64_t>(task) * workspaceLd + start], upperColumn, copy);
        if (col == 0)
        {
            for (int entry = 1; entry < 8; ++entry) localInfo.SetValue(entry, 0);
            localInfo.SetValue(0, firstSingular);
            LargeFence<HardEvent::S_MTE3>();
            DataCopyPad(gs[start / diagonalTile * 8], localInfo, DataCopyExtParams{1, 32, 0, 0, 0});
        }
        LargeFence<HardEvent::MTE3_MTE2>();
        LargeFence<HardEvent::MTE3_S>();
        LargeFence<HardEvent::S_V>();
    }
}

__global__ __aicore__ void lu950_getri_diagonal(int n, GM_ADDR negative, GM_ADDR lower, GM_ADDR upper, GM_ADDR statuses,
                                                int workspaceLd)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    LargeDiagonal(n, negative, lower, upper, statuses, workspaceLd, GetBlockIdx(), GetBlockNum());
}

__aicore__ void LargeFinish(int n, int lda, GM_ADDR a, GM_ADDR pivots, GM_ADDR info, GM_ADDR inverse, GM_ADDR statuses,
                            GM_ADDR original, int inputLd, int workspaceLd, int indirect, int batchIndex, int firstTask,
                            int taskStride)
{
    TPipe pipe;
    TBuf<TPosition::VECIN> inputBuffer, pivotBuffer, statusBuffer;
    TBuf<TPosition::VECCALC> permutationBuffer;
    TBuf<TPosition::VECOUT> infoBuffer;
    pipe.InitBuffer(inputBuffer, 4096 * sizeof(float));
    pipe.InitBuffer(pivotBuffer, 4096 * sizeof(int));
    pipe.InitBuffer(permutationBuffer, 2 * 4096 * sizeof(int));
    pipe.InitBuffer(statusBuffer, 128 * 8 * sizeof(int));
    pipe.InitBuffer(infoBuffer, 32);
    auto input = inputBuffer.Get<float>();
    auto localPivots = pivotBuffer.Get<int>(), permutation = permutationBuffer.Get<int>();
    auto inversePermutation = permutation[4096];
    auto localStatuses = statusBuffer.Get<int>(), localInfo = infoBuffer.Get<int>();
    GM_ADDR outputAddress = a;
    int pointerError = 0;
    if (indirect)
    {
        GlobalTensor<uint32_t> ap, cp;
        ap.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(original));
        cp.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(a));
        auto pointerScratch = input.ReinterpretCast<uint32_t>();
        const DataCopyExtParams pointerCopy{1, 8, 0, 0, 0};
        const DataCopyPadExtParams<uint32_t> pointerPad{true, 0, 6, 0};
        DataCopyPad(pointerScratch, ap[static_cast<uint64_t>(batchIndex) * 2], pointerCopy, pointerPad);
        DataCopyPad(pointerScratch[8], cp[static_cast<uint64_t>(batchIndex) * 2], pointerCopy, pointerPad);
        LargeFence<HardEvent::MTE2_S>();
        auto addresses = pointerScratch.ReinterpretCast<uint64_t>();
        const uint64_t av = addresses.GetValue(0), cv = addresses.GetValue(4);
        const uint64_t aBytes = (static_cast<uint64_t>(n - 1) * inputLd + n) * sizeof(float);
        const uint64_t cBytes = (static_cast<uint64_t>(n - 1) * lda + n) * sizeof(float);
        pointerError = av == 0 ? -2 : (cv == 0 || (av < cv + cBytes && cv < av + aBytes) ? -5 : 0);
        outputAddress = reinterpret_cast<__gm__ uint8_t *>(cv);
    }
    GlobalTensor<float> ga, gi;
    GlobalTensor<int> gp, gs, gf;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputAddress));
    gi.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inverse));
    gp.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(pivots));
    gs.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(statuses));
    gf.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    const int count = (n + diagonalTile - 1) / diagonalTile;
    DataCopyPad(localStatuses, gs, DataCopyExtParams{1, static_cast<uint32_t>(count * 8 * sizeof(int)), 0, 0, 0},
                DataCopyPadExtParams<int>{true, 0, static_cast<uint8_t>(0), 0});
    if (pivots != nullptr)
        DataCopyPad(localPivots, gp, DataCopyExtParams{1, static_cast<uint32_t>(n * sizeof(int)), 0, 0, 0},
                    DataCopyPadExtParams<int>{true, 0, static_cast<uint8_t>((8 - n % 8) % 8), 0});
    LargeFence<HardEvent::MTE2_S>();
    int code = n + 1;
    for (int k = 0; k < count; ++k)
        if (localStatuses.GetValue(k * 8) < code) code = localStatuses.GetValue(k * 8);
    if (code == n + 1) code = 0;
    if (pointerError != 0) code = pointerError;
    if (pivots != nullptr)
    {
        for (int k = 0; k < n; ++k) permutation.SetValue(k, k);
        for (int k = 0; k < n; ++k)
        {
            const int target = localPivots.GetValue(k) - 1;
            if (target < k || target >= n)
            {
                if (code != -2) code = -4;
                break;
            }
            const int saved = permutation.GetValue(k);
            permutation.SetValue(k, permutation.GetValue(target));
            permutation.SetValue(target, saved);
        }
        if (code == 0)
            for (int k = 0; k < n; ++k) inversePermutation.SetValue(permutation.GetValue(k), k);
    }
    if (firstTask == 0)
    {
        localInfo.SetValue(0, code);
        LargeFence<HardEvent::S_MTE3>();
        if (indirect) SetAtomicAdd<int>();
        DataCopyPad(gf, localInfo, DataCopyExtParams{1, sizeof(int), 0, 0, 0});
        if (indirect) SetAtomicNone();
        LargeFence<HardEvent::MTE3_S>();
    }
    if (code != 0) return;
    const DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
    // Unaligned user columns may share a 32-byte output block. Serialize their
    // final copies; internal matrices and Cube outputs use aligned columns.
    const bool serialOutput = lda % 8 || reinterpret_cast<uint64_t>(outputAddress) % 32;
    if (serialOutput && firstTask != 0) return;
    const int outputBlocks = serialOutput ? 1 : taskStride;
    for (int col = firstTask; col < n; col += outputBlocks)
    {
        const int source = pivots == nullptr ? col : inversePermutation.GetValue(col);
        DataCopyPad(input, gi[static_cast<uint64_t>(source) * workspaceLd], copy,
                    DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - n % 8) % 8), 0.f});
        LargeFence<HardEvent::MTE2_MTE3>();
        DataCopyPad(ga[static_cast<uint64_t>(col) * lda], input, copy);
        LargeFence<HardEvent::MTE3_MTE2>();
    }
}

__global__ __aicore__ void lu950_getri_finish(int n, int lda, GM_ADDR a, GM_ADDR pivots, GM_ADDR info, GM_ADDR inverse,
                                              GM_ADDR statuses, GM_ADDR original, int inputLd, int workspaceLd,
                                              int indirect, int batchIndex)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    LargeFinish(n, lda, a, pivots, info, inverse, statuses, original, inputLd, workspaceLd, indirect, batchIndex,
                GetBlockIdx(), GetBlockNum());
}

int Lu950LargeGetriDispatch(int n, void *a, int lda, const int *pivots, int *info, void *c, int ldc, bool indirect,
                            int batchIndex, void *workspace, unsigned blocks, aclrtStream stream)
{
    auto *negative = static_cast<float *>(workspace);
    const int workspaceLd = (n + 7) / 8 * 8;
    const size_t matrixElements = static_cast<size_t>(n) * workspaceLd;
    auto *lower = negative + matrixElements;
    auto *upper = lower + matrixElements;
    auto *temporary = upper + matrixElements;
    const size_t matrixBytesAligned = (4 * matrixElements * sizeof(float) + 511) / 512 * 512;
    auto *statuses = reinterpret_cast<int *>(static_cast<uint8_t *>(workspace) + matrixBytesAligned);
    const size_t statusBytes = ((n + diagonalTile - 1) / diagonalTile * 32 + 511) / 512 * 512;
    auto *system = reinterpret_cast<uint8_t *>(statuses) + statusBytes;
    lu950_getri_initialize<<<blocks, nullptr, stream>>>(
        n, lda, reinterpret_cast<uint8_t *>(a), reinterpret_cast<uint8_t *>(negative),
        reinterpret_cast<uint8_t *>(lower), reinterpret_cast<uint8_t *>(upper), reinterpret_cast<uint8_t *>(temporary),
        workspaceLd, indirect ? 1 : 0, batchIndex);
    lu950_getri_diagonal<<<blocks, nullptr, stream>>>(
        n, reinterpret_cast<uint8_t *>(negative), reinterpret_cast<uint8_t *>(lower),
        reinterpret_cast<uint8_t *>(upper), reinterpret_cast<uint8_t *>(statuses), workspaceLd);
    // Column-major C=A*B is row-major C^T=B^T*A^T. All workspace columns have 32-byte aligned starts.
    auto multiply = [&](int m, int columns, int k, const float *left, const float *right, float *out)
    {
        return Lu950CubeMatmul(columns, m, k, right, workspaceLd, left, workspaceLd, out, workspaceLd, system, stream,
                               false);
    };
    for (int width = diagonalTile; width < n; width *= 2)
        for (int start = 0; start + width < n; start += 2 * width)
        {
            const int left = width, right = n - start - width < width ? n - start - width : width;
            const size_t top = static_cast<size_t>(start) * workspaceLd + start;
            const size_t bottom = static_cast<size_t>(start + width) * workspaceLd + start + width;
            const size_t upperBlock = static_cast<size_t>(start + width) * workspaceLd + start;
            const size_t lowerBlock = static_cast<size_t>(start) * workspaceLd + start + width;
            if (multiply(left, right, right, negative + upperBlock, upper + bottom, temporary + upperBlock) != 0 ||
                multiply(left, right, left, upper + top, temporary + upperBlock, upper + upperBlock) != 0 ||
                multiply(right, left, left, negative + lowerBlock, lower + top, temporary + lowerBlock) != 0 ||
                multiply(right, left, right, lower + bottom, temporary + lowerBlock, lower + lowerBlock) != 0)
                return -1;
        }
    if (multiply(n, n, n, upper, lower, temporary) != 0) return -1;
    lu950_getri_finish<<<blocks, nullptr, stream>>>(
        n, ldc, reinterpret_cast<uint8_t *>(c), reinterpret_cast<uint8_t *>(const_cast<int *>(pivots)),
        reinterpret_cast<uint8_t *>(info), reinterpret_cast<uint8_t *>(temporary),
        reinterpret_cast<uint8_t *>(statuses), reinterpret_cast<uint8_t *>(a), lda, workspaceLd, indirect ? 1 : 0,
        batchIndex);
    return 0;
}

int Lu950LargeGetriLaunch(int n, float *a, int lda, const int *pivots, int *info, void *workspace, unsigned blocks,
                          aclrtStream stream)
{
    return Lu950LargeGetriDispatch(n, a, lda, pivots, info, a, lda, false, 0, workspace, blocks, stream);
}

size_t Lu950LargeBatchMatrixBytes(int n)
{
    const size_t matrixBytes = static_cast<size_t>(n) * ((n + 7) / 8 * 8) * sizeof(float);
    return (4 * matrixBytes + 511) / 512 * 512 + ((n + 31) / 32 * 32 + 511) / 512 * 512;
}

size_t Lu950LargeBatchWorkspaceBytes(int n, int batch)
{
    return Lu950LargeBatchMatrixBytes(n) * batch + Lu950CubeWorkspaceBytes();
}

__global__ __aicore__ void lu950_getri_batch_stage(int stage, int n, int lda, GM_ADDR a, GM_ADDR pivots, int ldc,
                                                   GM_ADDR c, GM_ADDR info, GM_ADDR workspace, uint64_t matrixBytes,
                                                   int firstMatrix, int batch, int cohort)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (cohort <= 0 || GetBlockNum() < cohort) return;
    const int divisor = cohort > 0 ? cohort : 1;
    const int lane = GetBlockIdx() % divisor;
    const int groups = GetBlockNum() / divisor;
    const int ld = (n + 7) / 8 * 8;
    const uint64_t matrixElements = static_cast<uint64_t>(n) * ld;
    for (int index = GetBlockIdx() / divisor; index < batch; index += groups)
    {
        GM_ADDR negative = workspace + index * matrixBytes;
        GM_ADDR lower = negative + matrixElements * sizeof(float);
        GM_ADDR upper = lower + matrixElements * sizeof(float);
        GM_ADDR temporary = upper + matrixElements * sizeof(float);
        GM_ADDR statuses = negative + (4 * matrixElements * sizeof(float) + 511) / 512 * 512;
        const int matrix = firstMatrix + index;
        if (stage == 0)
            LargeInitialize(n, lda, a, negative, lower, upper, temporary, ld, 1, matrix, lane, cohort);
        else if (stage == 1)
            LargeDiagonal(n, negative, lower, upper, statuses, ld, lane, cohort);
        else
            LargeFinish(n, ldc, c,
                        pivots == nullptr ? nullptr : pivots + static_cast<uint64_t>(matrix) * n * sizeof(int),
                        info + matrix * sizeof(int), temporary, statuses, a, lda, ld, 1, matrix, lane, cohort);
    }
}

int Lu950ParallelBatchGetriLaunch(int n, const float *const *a, int lda, const int *pivots, float *const *c, int ldc,
                                  int *info, int firstMatrix, int batch, void *workspace, unsigned blocks,
                                  aclrtStream stream)
{
    if (batch <= 0 || blocks == 0) return -1;
    const int ld = (n + 7) / 8 * 8;
    const size_t matrixBytes = Lu950LargeBatchMatrixBytes(n);
    const uint64_t matrixStride = matrixBytes / sizeof(float);
    const uint64_t elements = static_cast<uint64_t>(n) * ld;
    auto *negative = static_cast<float *>(workspace), *lower = negative + elements;
    auto *upper = lower + elements, *temporary = upper + elements;
    auto *system = static_cast<uint8_t *>(workspace) + matrixBytes * batch;
    const unsigned groups = batch < static_cast<int>(blocks) ? batch : blocks;
    const unsigned cohort = blocks / groups;
    blocks = groups * cohort;
    auto stage = [&](int operation)
    {
        lu950_getri_batch_stage<<<blocks, nullptr, stream>>>(
            operation, n, lda, reinterpret_cast<uint8_t *>(const_cast<float **>(a)),
            reinterpret_cast<uint8_t *>(const_cast<int *>(pivots)), ldc,
            reinterpret_cast<uint8_t *>(const_cast<float **>(c)), reinterpret_cast<uint8_t *>(info),
            reinterpret_cast<uint8_t *>(workspace), matrixBytes, firstMatrix, batch, cohort);
    };
    Lu950InfoLaunch(info + firstMatrix, batch, 0, stream);
    stage(0);
    stage(1);
    auto multiply = [&](int rows, int columns, int k, const float *left, const float *right, float *out)
    {
        return Lu950CubeMatmulBatch(columns, rows, k, right, ld, left, ld, out, ld, system, stream, false, batch,
                                    matrixStride);
    };
    for (int width = diagonalTile; width < n; width *= 2)
        for (int start = 0; start + width < n; start += 2 * width)
        {
            const int left = width, right = n - start - width < width ? n - start - width : width;
            const size_t top = static_cast<size_t>(start) * ld + start;
            const size_t bottom = static_cast<size_t>(start + width) * ld + start + width;
            const size_t upperBlock = static_cast<size_t>(start + width) * ld + start;
            const size_t lowerBlock = static_cast<size_t>(start) * ld + start + width;
            if (multiply(left, right, right, negative + upperBlock, upper + bottom, temporary + upperBlock) != 0 ||
                multiply(left, right, left, upper + top, temporary + upperBlock, upper + upperBlock) != 0 ||
                multiply(right, left, left, negative + lowerBlock, lower + top, temporary + lowerBlock) != 0 ||
                multiply(right, left, right, lower + bottom, temporary + lowerBlock, lower + lowerBlock) != 0)
                return -1;
        }
    if (multiply(n, n, n, upper, lower, temporary) != 0) return -1;
    stage(2);
    return 0;
}
