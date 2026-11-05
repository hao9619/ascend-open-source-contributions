/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "launch_internal.h"
namespace AscendC
{
namespace
{
template <HardEvent Event>
__aicore__ inline void BatchFence()
{
    const auto event = static_cast<int32_t>(GetTPipePtr()->FetchEventID(Event));
    SetFlag<Event>(event);
    WaitFlag<Event>(event);
}
#include "batch_pointer_utils.hpp"
#ifdef LU950_N2_DENSE
#include "batch_getri_n2.hpp"
#endif
#ifdef LU950_FUSED_BATCH
#include "batch_getri_fused.hpp"
#include "batch_getri_n4_dense.hpp"
#include "batch_getri_register_dense.hpp"
#endif
#include "batch_getri_small.hpp"

__aicore__ void BatchGetriMedium(int n, GM_ADDR a, int lda, GM_ADDR pivots, GM_ADDR c, int ldc, GM_ADDR info, int batch)
{
    constexpr int rhsWidth = 64;
    TPipe pipe;
    TBuf<TPosition::VECIN> factorBuffer, pivotBuffer;
    TBuf<TPosition::VECCALC> rhsBuffer, coefficientBuffer, productBuffer, offsetsBuffer, mapBuffer, diagonalBuffer;
    TBuf<TPosition::VECOUT> outputBuffer, infoBuffer;
    pipe.InitBuffer(factorBuffer, 128 * 128 * sizeof(float));
    pipe.InitBuffer(pivotBuffer, 128 * sizeof(int));
    pipe.InitBuffer(rhsBuffer, 128 * rhsWidth * sizeof(float));
    pipe.InitBuffer(coefficientBuffer, 128 * rhsWidth * sizeof(float));
    pipe.InitBuffer(productBuffer, 128 * rhsWidth * sizeof(float));
    pipe.InitBuffer(offsetsBuffer, (128 * rhsWidth + 128) * sizeof(uint32_t));
    pipe.InitBuffer(mapBuffer, 2 * 128 * sizeof(int));
    pipe.InitBuffer(diagonalBuffer, rhsWidth * sizeof(float));
    pipe.InitBuffer(outputBuffer, 128 * rhsWidth * sizeof(float));
    pipe.InitBuffer(infoBuffer, 32);
    auto factors = factorBuffer.Get<float>(), rhs = rhsBuffer.Get<float>();
    auto coefficients = coefficientBuffer.Get<float>(), products = productBuffer.Get<float>();
    auto diagonal = diagonalBuffer.Get<float>(), output = outputBuffer.Get<float>();
    auto offsets = offsetsBuffer.Get<uint32_t>(), outputOffsets = offsets[128 * rhsWidth];
    auto permutation = mapBuffer.Get<int>(), inversePermutation = permutation[128];
    auto localPivots = pivotBuffer.Get<int>(), localInfo = infoBuffer.Get<int>();
    const int padded = (n + 7) / 8 * 8;
    for (int row = 0; row < padded; ++row)
    {
        for (int col = 0; col < rhsWidth; ++col) offsets.SetValue(row * rhsWidth + col, row * sizeof(float));
        outputOffsets.SetValue(row, row * rhsWidth * sizeof(float));
    }
    BatchFence<HardEvent::S_V>();
    GlobalTensor<uint32_t> aPointers, cPointers;
    GlobalTensor<int> globalPivots, globalInfo;
    aPointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a));
    cPointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(c));
    globalPivots.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(pivots));
    globalInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(info));
    const BinaryRepeatParams broadcast{1, 1, 1, 8, 0, 8};
    for (int matrix = GetBlockIdx(); matrix < batch; matrix += GetBlockNum())
    {
        auto pointerScratch = factors.ReinterpretCast<uint32_t>();
        const DataCopyExtParams pointerCopy{1, 8, 0, 0, 0};
        const DataCopyPadExtParams<uint32_t> pointerPad{true, 0, 6, 0};
        DataCopyPad(pointerScratch, aPointers[static_cast<uint64_t>(matrix) * 2], pointerCopy, pointerPad);
        DataCopyPad(pointerScratch[8], cPointers[static_cast<uint64_t>(matrix) * 2], pointerCopy, pointerPad);
        BatchFence<HardEvent::MTE2_S>();
        auto addresses = pointerScratch.ReinterpretCast<uint64_t>();
        const uint64_t ap = addresses.GetValue(0), cp = addresses.GetValue(4);
        const uint64_t aBytes = (static_cast<uint64_t>(n - 1) * lda + n) * sizeof(float);
        const uint64_t cBytes = (static_cast<uint64_t>(n - 1) * ldc + n) * sizeof(float);
        int code = ap == 0 ? -2 : (cp == 0 || (ap < cp + cBytes && cp < ap + aBytes) ? -5 : 0);
        GlobalTensor<float> source, destination;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ap));
        destination.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cp));
        if (code == 0)
        {
            DataCopyPad(factors, source,
                        DataCopyExtParams{static_cast<uint16_t>(n), static_cast<uint32_t>(n * sizeof(float)),
                                          static_cast<uint32_t>((lda - n) * sizeof(float)), 0, 0},
                        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(padded - n), 0.f});
            if (pivots != nullptr)
                DataCopyPad(localPivots, globalPivots[static_cast<uint64_t>(matrix) * n],
                            DataCopyExtParams{1, static_cast<uint32_t>(n * sizeof(int)), 0, 0, 0},
                            DataCopyPadExtParams<int>{true, 0, static_cast<uint8_t>(padded - n), 0});
            BatchFence<HardEvent::MTE2_S>();
            for (int k = 0; k < n; ++k)
            {
                if (factors.GetValue(k * padded + k) == 0.f && code == 0) code = k + 1;
                permutation.SetValue(k, k);
            }
            if (pivots != nullptr)
                for (int k = 0; k < n; ++k)
                {
                    const int target = localPivots.GetValue(k) - 1;
                    if (target < k || target >= n)
                    {
                        code = -4;
                        break;
                    }
                    const int saved = permutation.GetValue(k);
                    permutation.SetValue(k, permutation.GetValue(target));
                    permutation.SetValue(target, saved);
                }
            if (code == 0)
                for (int k = 0; k < n; ++k) inversePermutation.SetValue(permutation.GetValue(k), k);
        }
        if (code == 0)
            for (int col = 0; col < n; col += rhsWidth)
            {
                const int columns = n - col < rhsWidth ? n - col : rhsWidth;
                Duplicate(rhs, 0.f, padded * rhsWidth);
                BatchFence<HardEvent::V_S>();
                for (int lane = 0; lane < columns; ++lane)
                    rhs.SetValue(inversePermutation.GetValue(col + lane) * rhsWidth + lane, 1.f);
                BatchFence<HardEvent::S_V>();
                for (int k = 0; k + 1 < n; ++k)
                {
                    const int rows = n - k - 1, offset = (k + 1) * rhsWidth;
                    Gather(coefficients[offset], factors, offsets[offset], k * padded * sizeof(float), rows * rhsWidth);
                    PipeBarrier<PIPE_V>();
                    Mul(products[offset], rhs[k * rhsWidth], coefficients[offset], 64, static_cast<uint8_t>(rows),
                        broadcast);
                    PipeBarrier<PIPE_V>();
                    Sub(rhs[offset], rhs[offset], products[offset], rows * rhsWidth);
                    PipeBarrier<PIPE_V>();
                }
                for (int k = n - 1; k >= 0; --k)
                {
                    Duplicate(diagonal, factors.GetValue(k * padded + k), rhsWidth);
                    PipeBarrier<PIPE_V>();
                    Div(rhs[k * rhsWidth], rhs[k * rhsWidth], diagonal, rhsWidth);
                    PipeBarrier<PIPE_V>();
                    if (k == 0) continue;
                    Gather(coefficients, factors, offsets, k * padded * sizeof(float), k * rhsWidth);
                    PipeBarrier<PIPE_V>();
                    Mul(products, rhs[k * rhsWidth], coefficients, 64, static_cast<uint8_t>(k), broadcast);
                    PipeBarrier<PIPE_V>();
                    Sub(rhs, rhs, products, k * rhsWidth);
                    PipeBarrier<PIPE_V>();
                }
                for (int lane = 0; lane < columns; ++lane)
                {
                    Gather(output[lane * padded], rhs, outputOffsets, lane * sizeof(float), n);
                    PipeBarrier<PIPE_V>();
                }
                BatchFence<HardEvent::V_MTE3>();
                DataCopyPad(destination[static_cast<uint64_t>(col) * ldc], output,
                            DataCopyExtParams{static_cast<uint16_t>(columns), static_cast<uint32_t>(n * sizeof(float)),
                                              0, static_cast<uint32_t>((ldc - n) * sizeof(float)), 0});
                BatchFence<HardEvent::MTE3_V>();
            }
        localInfo.SetValue(0, code);
        BatchFence<HardEvent::S_MTE3>();
        SetAtomicAdd<int>();
        DataCopyPad(globalInfo[matrix], localInfo, DataCopyExtParams{1, sizeof(int), 0, 0, 0});
        SetAtomicNone();
        BatchFence<HardEvent::MTE3_S>();
        BatchFence<HardEvent::MTE3_MTE2>();
    }
}
}  // namespace
}  // namespace AscendC

using namespace AscendC;

__global__ __aicore__ void lu950_getri_batched_small(int n, GM_ADDR a, int lda, GM_ADDR pivots, GM_ADDR c, int ldc,
                                                     GM_ADDR info, int batch, int group)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BatchGetriSmall(n, a, lda, pivots, c, ldc, info, batch, group);
}
__global__ __aicore__ void lu950_getri_batched_medium(int n, GM_ADDR a, int lda, GM_ADDR pivots, GM_ADDR c, int ldc,
                                                      GM_ADDR info, int batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BatchGetriMedium(n, a, lda, pivots, c, ldc, info, batch);
}

#ifdef LU950_N2_DENSE
__global__ __aicore__ void lu950_getri_batched_n2(GM_ADDR a, GM_ADDR pivots, GM_ADDR c, GM_ADDR info, int batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BatchGetriN2(a, pivots, c, info, batch);
}
#endif

#ifdef LU950_FUSED_BATCH
__global__ __aicore__ void lu950_getri_batched_n4_dense(GM_ADDR a, GM_ADDR pivots, GM_ADDR c, GM_ADDR info, int batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BatchGetriN4Dense(a, pivots, c, info, batch);
}
#endif

#ifdef LU950_FUSED_BATCH
__global__ __aicore__ void lu950_getri_batched_n8_dense(GM_ADDR a, GM_ADDR pivots, GM_ADDR c, GM_ADDR info, int batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BatchGetriRegisterDense<8, 768>(a, pivots, c, info, batch);
}
#endif

#ifdef LU950_FUSED_BATCH
#endif

#ifdef LU950_FUSED_BATCH
__global__ __aicore__ void lu950_getri_batched_n16_dense(GM_ADDR a, GM_ADDR pivots, GM_ADDR c, GM_ADDR info, int batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BatchGetriRegisterDense<16, 216>(a, pivots, c, info, batch);
}
#endif

#ifdef LU950_RAW_N2
__global__ __aicore__ void lu950_matinv_batched_n2(GM_ADDR a, GM_ADDR c, GM_ADDR info, int batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (batch < 65536)
        BatchGetriN2<256>(a, nullptr, c, info, batch, true);
    else
        BatchGetriN2(a, nullptr, c, info, batch, true);
}

void Lu950MatinvN2Launch(const float* const* a, float* const* c, int* info, int batch, unsigned blocks, void* stream)
{
    const unsigned group = batch < 65536 ? 256 : 4096;
    const unsigned tasks = (static_cast<unsigned>(batch) + group - 1) / group;
    if (blocks > tasks) blocks = tasks;
    lu950_matinv_batched_n2<<<blocks, nullptr, stream>>>(reinterpret_cast<uint8_t*>(const_cast<float**>(a)),
                                                         reinterpret_cast<uint8_t*>(const_cast<float**>(c)),
                                                         reinterpret_cast<uint8_t*>(info), batch);
}
#endif

#ifdef LU950_RAW_BATCH
__global__ __aicore__ void lu950_matinv_batched_small(int n, GM_ADDR a, int lda, GM_ADDR c, int ldc, GM_ADDR info,
                                                      int batch, int group)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    BatchGetriSmall(n, a, lda, nullptr, c, ldc, info, batch, group, true);
}

void Lu950MatinvBatchLaunch(int n, const float* const* a, int lda, float* const* c, int ldc, int* info, int batch,
                            unsigned blocks, void* stream)
{
    int group = batch < 4096 ? (n <= 16 ? 32 : 8)
                             : (n <= 4    ? 512
                                : n <= 8  ? 256
                                : n <= 16 ? 64
                                : n <= 24 ? 32
                                : n <= 28 ? 24
                                          : 16);
    if (n >= 17 && n <= 32 && batch >= 512 && batch < 4096) group = 16;
    if (n >= 17 && n <= 32 && batch >= 4096)
    {
        const bool compact = lda == n && ldc == n;
        group = compact ? (n <= 21 ? 64 : n <= 27 ? 40 : n <= 30 ? 32 : 16) : (n <= 19 ? 64 : n <= 28 ? 32 : 24);
        const unsigned oneWaveLimit = n == 17 ? 96 : 80;
        if (compact && n <= 18 && blocks != 0 && static_cast<unsigned>(batch) > blocks * 64 &&
            static_cast<unsigned>(batch) <= blocks * oneWaveLimit)
        {
            const unsigned perCore = (static_cast<unsigned>(batch) + blocks - 1) / blocks;
            group = static_cast<int>((perCore + 7) / 8 * 8);
        }
    }
    if (n >= 3 && n <= 5 && batch >= 4096 && batch <= 32768 && blocks != 0)
    {
        const int capacity = group;
        int bestGroup = group;
        unsigned waves = ((static_cast<unsigned>(batch) + group - 1) / group + blocks - 1) / blocks;
        unsigned work = waves * group;
        // Minimize the largest per-core matrix count, then reduce repeated tile setup.
        for (int candidate = 64; candidate <= capacity; candidate += 64)
        {
            const unsigned tasks = (static_cast<unsigned>(batch) + candidate - 1) / candidate;
            const unsigned candidateWaves = (tasks + blocks - 1) / blocks;
            const unsigned candidateWork = candidateWaves * candidate;
            if (candidateWork < work || (candidateWork == work && candidateWaves < waves))
            {
                bestGroup = candidate;
                waves = candidateWaves;
                work = candidateWork;
            }
        }
        group = bestGroup;
    }
    if (n >= 10 && n <= 13 && batch >= 4096 && blocks != 0 && static_cast<unsigned>(batch) > blocks * 64 &&
        static_cast<unsigned>(batch) <= blocks * 96)
    {
        // One wave avoids repeated tile setup; the maximum group remains within UB capacity.
        const unsigned perCore = (static_cast<unsigned>(batch) + blocks - 1) / blocks;
        group = static_cast<int>((perCore + 7) / 8 * 8);
    }
    const unsigned tasks = (static_cast<unsigned>(batch) + group - 1) / group;
    if (blocks > tasks) blocks = tasks;
    if (n >= 18 && n <= 32 && n != 19 && n != 20)
    {
        Lu950MatinvHighLaunch(n, a, lda, c, ldc, info, batch, group, blocks, stream);
        return;
    }
    lu950_matinv_batched_small<<<blocks, nullptr, stream>>>(n, reinterpret_cast<uint8_t*>(const_cast<float**>(a)), lda,
                                                            reinterpret_cast<uint8_t*>(const_cast<float**>(c)), ldc,
                                                            reinterpret_cast<uint8_t*>(info), batch, group);
}
#endif

void Lu950BatchGetriLaunch(int n, const float* const* a, int lda, const int* pivots, float* const* c, int ldc,
                           int* info, int batch, unsigned blocks, void* stream)
{
    auto ap = reinterpret_cast<uint8_t*>(const_cast<float**>(a));
    auto cp = reinterpret_cast<uint8_t*>(const_cast<float**>(c));
    auto pp = reinterpret_cast<uint8_t*>(const_cast<int*>(pivots));
    auto ip = reinterpret_cast<uint8_t*>(info);
#ifdef LU950_N2_DENSE
    if (n == 2 && lda == 2 && ldc == 2 && batch >= 65536)
    {
        const unsigned tasks = (static_cast<unsigned>(batch) + 4095) / 4096;
        if (blocks > tasks) blocks = tasks;
        lu950_getri_batched_n2<<<blocks, nullptr, stream>>>(ap, pp, cp, ip, batch);
        return;
    }
#endif
#ifdef LU950_FUSED_BATCH
    if (n == 4 && lda == 4 && ldc == 4 && batch >= 65536)
    {
        const unsigned tasks = (static_cast<unsigned>(batch) + 2047) / 2048;
        if (blocks > tasks) blocks = tasks;
        lu950_getri_batched_n4_dense<<<blocks, nullptr, stream>>>(ap, pp, cp, ip, batch);
        return;
    }
#endif
#ifdef LU950_FUSED_BATCH
    if (n == 8 && lda == 8 && ldc == 8 && batch >= 65536)
    {
        const unsigned tasks = (static_cast<unsigned>(batch) + 767) / 768;
        if (blocks > tasks) blocks = tasks;
        lu950_getri_batched_n8_dense<<<blocks, nullptr, stream>>>(ap, pp, cp, ip, batch);
        return;
    }
#endif

#ifdef LU950_FUSED_BATCH
    if (n == 16 && lda == n && ldc == n && batch >= 8192)
    {
        const unsigned tasks = (static_cast<unsigned>(batch) + 215) / 216;
        if (blocks > tasks) blocks = tasks;
        lu950_getri_batched_n16_dense<<<blocks, nullptr, stream>>>(ap, pp, cp, ip, batch);
        return;
    }
#endif
    if (n <= 32)
    {
        const int group = batch < 4096 ? (n <= 16 ? 32 : 8)
                                       : (n <= 2    ? 512
                                          : n <= 4  ? 256
                                          : n <= 8  ? 128
                                          : n <= 16 ? 64
                                                    : 16);
        const unsigned tasks = (static_cast<unsigned>(batch) + group - 1) / group;
        if (blocks > tasks) blocks = tasks;
        lu950_getri_batched_small<<<blocks, nullptr, stream>>>(n, ap, lda, pp, cp, ldc, ip, batch, group);
    }
    else
    {
        if (blocks > static_cast<unsigned>(batch)) blocks = batch;
        Lu950InfoLaunch(info, batch, 0, stream);
        if (batch >= 4096)
            Lu950BatchGetriMediumTiledLaunch(n, ap, lda, pp, cp, ldc, ip, batch, blocks, stream);
        else
            lu950_getri_batched_medium<<<blocks, nullptr, stream>>>(n, ap, lda, pp, cp, ldc, ip, batch);
    }
}
