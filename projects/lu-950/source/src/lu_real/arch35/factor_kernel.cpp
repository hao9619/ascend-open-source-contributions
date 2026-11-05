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
template <HardEvent Event>
__aicore__ inline void FactorFence()
{
    const auto event = static_cast<int32_t>(GetTPipePtr()->FetchEventID(Event));
    SetFlag<Event>(event);
    WaitFlag<Event>(event);
}

__aicore__ void Panel(int m, int lda, int start, int nb, GM_ADDR a, GM_ADDR pivots, GM_ADDR info)
{
    TPipe pipe;
    TBuf<TPosition::VECIN> panelBuffer;
    TBuf<TPosition::VECCALC> workBuffer, reduceBuffer, multiplierBuffer;
    TBuf<TPosition::VECOUT> pivotBuffer, infoBuffer;
    pipe.InitBuffer(panelBuffer, 32768 * sizeof(float));
    pipe.InitBuffer(workBuffer, 4096 * sizeof(float));
    pipe.InitBuffer(reduceBuffer, 4096 * sizeof(float));
    pipe.InitBuffer(multiplierBuffer, 4096 * sizeof(float));
    pipe.InitBuffer(pivotBuffer, 32 * sizeof(int));
    pipe.InitBuffer(infoBuffer, 32);
    auto panel = panelBuffer.Get<float>();
    auto work = workBuffer.Get<float>();
    auto reduction = reduceBuffer.Get<float>();
    auto multiplier = multiplierBuffer.Get<float>();
    auto localPivots = pivotBuffer.Get<int>();
    auto localInfo = infoBuffer.Get<int>();
    GlobalTensor<float> matrix;
    GlobalTensor<int> globalPivots, globalInfo;
    matrix.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    globalPivots.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(pivots));
    globalInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(info));
    int firstSingular = 0;
    if (start > 0)
    {
        DataCopyPad(localInfo, globalInfo, DataCopyExtParams{1, sizeof(int), 0, 0, 0},
                    DataCopyPadExtParams<int>{true, 0, 7, 0});
        FactorFence<HardEvent::MTE2_S>();
        firstSingular = localInfo.GetValue(0);
    }
    const int rows = m - start;
    const int stride = (rows + 7) / 8 * 8;
    const DataCopyExtParams copy{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
    for (int col = 0; col < nb; ++col)
        DataCopyPad(panel[col * stride], matrix[static_cast<uint64_t>(start + col) * lda + start], copy,
                    DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(stride - rows), 0.0f});
    FactorFence<HardEvent::MTE2_S>();
    FactorFence<HardEvent::MTE2_V>();
    for (int k = 0; k < nb; ++k)
    {
        int pivot = k;
        if (pivots != nullptr)
        {
            Abs(work, panel[k * stride], rows);
            PipeBarrier<PIPE_V>();
            if (k > 0) Duplicate(work, -1.0f, k);
            PipeBarrier<PIPE_V>();
            ReduceMax(reduction, work, multiplier, rows, true);
            FactorFence<HardEvent::V_S>();
            pivot = static_cast<int>(reduction.ReinterpretCast<uint32_t>().GetValue(1));
        }
        localPivots.SetValue(k, start + pivot + 1);
        if (pivot != k)
        {
            for (int col = 0; col < nb; ++col)
            {
                const float saved = panel.GetValue(col * stride + k);
                panel.SetValue(col * stride + k, panel.GetValue(col * stride + pivot));
                panel.SetValue(col * stride + pivot, saved);
            }
        }
        const float diagonal = panel.GetValue(k * stride + k);
        if (diagonal == 0.0f && firstSingular == 0) firstSingular = start + k + 1;
        FactorFence<HardEvent::S_V>();
        if (diagonal != 0.0f && diagonal > -0x1p-126f && diagonal < 0x1p-126f)
        {
            // Direct ratios avoid overflow of 1/diagonal for subnormal pivots.
            Muls(multiplier, panel[k * stride], 1.0f, stride);
            FactorFence<HardEvent::V_S>();
            for (int row = k + 1; row < rows; ++row)
                multiplier.SetValue(row, panel.GetValue(k * stride + row) / diagonal);
            FactorFence<HardEvent::S_V>();
        }
        else
            Muls(multiplier, panel[k * stride], diagonal == 0.0f ? 1.0f : 1.0f / diagonal, stride);
        FactorFence<HardEvent::V_S>();
        for (int row = 0; row <= k; ++row) multiplier.SetValue(row, panel.GetValue(k * stride + row));
        FactorFence<HardEvent::S_V>();
        Muls(panel[k * stride], multiplier, 1.0f, stride);
        PipeBarrier<PIPE_V>();
        Duplicate(multiplier, 0.0f, k + 1);
        PipeBarrier<PIPE_V>();
        for (int col = k + 1; col < nb; ++col)
        {
            const float upper = panel.GetValue(col * stride + k);
            Axpy(panel[col * stride], multiplier, -upper, stride);
            PipeBarrier<PIPE_V>();
        }
        FactorFence<HardEvent::V_S>();
    }
    FactorFence<HardEvent::V_MTE3>();
    for (int col = 0; col < nb; ++col)
        DataCopyPad(matrix[static_cast<uint64_t>(start + col) * lda + start], panel[col * stride], copy);
    localInfo.SetValue(0, firstSingular);
    FactorFence<HardEvent::S_MTE3>();
    if (pivots != nullptr)
        DataCopyPad(globalPivots[start], localPivots,
                    DataCopyExtParams{1, static_cast<uint32_t>(nb * sizeof(int)), 0, 0, 0});
    DataCopyPad(globalInfo, localInfo, DataCopyExtParams{1, sizeof(int), 0, 0, 0});
    FactorFence<HardEvent::MTE3_S>();
}

__aicore__ void SwapSolve(int m, int n, int lda, int start, int nb, GM_ADDR a, GM_ADDR pivots, GM_ADDR negativeUpper,
                          int firstColumn)
{
    TPipe pipe;
    TBuf<TPosition::VECIN> columnBuffer, diagonalBuffer, lowerBuffer;
    TBuf<TPosition::VECIN> pivotBuffer;
    pipe.InitBuffer(columnBuffer, 4096 * sizeof(float));
    pipe.InitBuffer(diagonalBuffer, 32 * 32 * sizeof(float));
    pipe.InitBuffer(pivotBuffer, 32 * sizeof(int));
    pipe.InitBuffer(lowerBuffer, 32768 * sizeof(float));
    auto column = columnBuffer.Get<float>();
    auto diagonal = diagonalBuffer.Get<float>();
    auto cachedPivots = pivotBuffer.Get<int>();
    auto lower = lowerBuffer.Get<float>();
    const int after = start + nb;
    const int lowerRows = m - after;
    const int lowerStride = (lowerRows + 7) / 8 * 8;
    GlobalTensor<float> matrix;
    GlobalTensor<int> globalPivots;
    matrix.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    globalPivots.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(pivots));
    if (pivots != nullptr)
        DataCopyPad(cachedPivots, globalPivots[start],
                    DataCopyExtParams{1, static_cast<uint32_t>(nb * sizeof(int)), 0, 0, 0},
                    DataCopyPadExtParams<int>{true, 0, static_cast<uint8_t>((8 - nb % 8) % 8), 0});
    for (int col = 0; col < nb; ++col)
        DataCopyPad(diagonal[col * 32], matrix[static_cast<uint64_t>(start + col) * lda + start],
                    DataCopyExtParams{1, static_cast<uint32_t>(nb * sizeof(float)), 0, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - nb % 8) % 8), 0.0f});
    if (negativeUpper == nullptr && lowerRows > 0 && n > after)
        for (int k = 0; k < nb; ++k)
            DataCopyPad(lower[k * lowerStride], matrix[static_cast<uint64_t>(start + k) * lda + after],
                        DataCopyExtParams{1, static_cast<uint32_t>(lowerRows * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(lowerStride - lowerRows), 0.0f});
    FactorFence<HardEvent::MTE2_V>();
    FactorFence<HardEvent::MTE2_S>();
    // Mask the upper triangle once, then solve using vector rank-one updates.
    for (int col = 0; col < nb; ++col)
        for (int row = 0; row <= col; ++row) diagonal.SetValue(col * 32 + row, 0.0f);
    FactorFence<HardEvent::S_V>();
    for (int col = firstColumn + GetBlockIdx(); col < n; col += GetBlockNum())
    {
        if (col >= start && col < start + nb) continue;
        DataCopyPad(column, matrix[static_cast<uint64_t>(col) * lda],
                    DataCopyExtParams{1, static_cast<uint32_t>(m * sizeof(float)), 0, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - m % 8) % 8), 0.0f});
        FactorFence<HardEvent::MTE2_S>();
        for (int k = 0; k < nb; ++k)
        {
            const int pivot = pivots != nullptr ? cachedPivots.GetValue(k) - 1 : start + k;
            if (pivot != start + k)
            {
                const float saved = column.GetValue(start + k);
                column.SetValue(start + k, column.GetValue(pivot));
                column.SetValue(pivot, saved);
            }
        }
        if (col >= start + nb)
        {
            FactorFence<HardEvent::S_V>();
            for (int k = 0; k < nb; ++k)
            {
                const float value = column.GetValue(start + k);
                Axpy(column[start], diagonal[k * 32], -value, (nb + 7) / 8 * 8);
                FactorFence<HardEvent::V_S>();
            }
        }
        if (negativeUpper == nullptr && col >= after && lowerRows > 0)
        {
            // Reuse the panel already loaded by this core for every assigned
            // trailing column, and keep C resident across TRSM and the update.
            for (int k = 0; k < nb; ++k)
            {
                const float upper = column.GetValue(start + k);
                Axpy(column[after], lower[k * lowerStride], -upper, lowerStride);
                PipeBarrier<PIPE_V>();
            }
            FactorFence<HardEvent::V_S>();
        }
        if (negativeUpper != nullptr && col >= after)
        {
            GlobalTensor<float> packed;
            packed.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(negativeUpper));
            for (int k = 0; k < nb; ++k) lower.SetValue(k, -column.GetValue(start + k));
            FactorFence<HardEvent::S_MTE3>();
            DataCopyPad(packed[static_cast<uint64_t>(col - after) * nb], lower,
                        DataCopyExtParams{1, static_cast<uint32_t>(nb * sizeof(float)), 0, 0, 0});
            FactorFence<HardEvent::MTE3_S>();
        }
        FactorFence<HardEvent::S_MTE3>();
        DataCopyPad(matrix[static_cast<uint64_t>(col) * lda], column,
                    DataCopyExtParams{1, static_cast<uint32_t>(m * sizeof(float)), 0, 0, 0});
        FactorFence<HardEvent::MTE3_MTE2>();
        FactorFence<HardEvent::MTE3_S>();
    }
}

__aicore__ void Update(int m, int n, int lda, int start, int nb, GM_ADDR a)
{
    constexpr int tile = 256;
    TPipe pipe;
    TBuf<TPosition::VECIN> lowerBuffer, columnBuffer;
    TBuf<TPosition::VECCALC> productBuffer;
    pipe.InitBuffer(lowerBuffer, 32 * tile * sizeof(float));
    pipe.InitBuffer(columnBuffer, tile * sizeof(float));
    pipe.InitBuffer(productBuffer, tile * sizeof(float));
    auto lower = lowerBuffer.Get<float>();
    auto column = columnBuffer.Get<float>();
    auto product = productBuffer.Get<float>();
    GlobalTensor<float> matrix;
    matrix.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    const int after = start + nb;
    const int rowTiles = (m - after + tile - 1) / tile;
    const int tasks = rowTiles * (n - after);
    for (int task = GetBlockIdx(); task < tasks; task += GetBlockNum())
    {
        const int col = after + task / rowTiles;
        const int row = after + task % rowTiles * tile;
        const int count = m - row < tile ? m - row : tile;
        const int padded = (count + 7) / 8 * 8;
        const DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(padded - count), 0.0f};
        for (int k = 0; k < nb; ++k)
            DataCopyPad(lower[k * tile], matrix[static_cast<uint64_t>(start + k) * lda + row], copy, pad);
        DataCopyPad(column, matrix[static_cast<uint64_t>(col) * lda + row], copy, pad);
        FactorFence<HardEvent::MTE2_V>();
        for (int k = 0; k < nb; ++k)
        {
            const float upper = matrix.GetValue(static_cast<uint64_t>(col) * lda + start + k);
            Muls(product, lower[k * tile], upper, padded);
            PipeBarrier<PIPE_V>();
            Sub(column, column, product, padded);
            PipeBarrier<PIPE_V>();
        }
        FactorFence<HardEvent::V_MTE3>();
        DataCopyPad(matrix[static_cast<uint64_t>(col) * lda + row], column, copy);
        FactorFence<HardEvent::MTE3_MTE2>();
        FactorFence<HardEvent::MTE3_V>();
    }
}
}  // namespace
// The wide kernels share the private synchronization helper above.
#include "factor_wide.hpp"
}  // namespace AscendC

using namespace AscendC;

__global__ __aicore__ void lu950_factor_panel(int m, int lda, int start, int nb, GM_ADDR a, GM_ADDR pivots,
                                              GM_ADDR info)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Panel(m, lda, start, nb, a, pivots, info);
}
__global__ __aicore__ void lu950_factor_swap_solve(int m, int n, int lda, int start, int nb, GM_ADDR a, GM_ADDR pivots)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SwapSolve(m, n, lda, start, nb, a, pivots, nullptr, 0);
}
__global__ __aicore__ void lu950_factor_update(int m, int n, int lda, int start, int nb, GM_ADDR a)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Update(m, n, lda, start, nb, a);
}

void Lu950FactorLaunch(int m, int n, float* a, int lda, int* pivots, int* info, int panel, unsigned blocks,
                       void* stream)
{
    auto address = reinterpret_cast<uint8_t*>(a);
    auto pivotAddress = reinterpret_cast<uint8_t*>(pivots);
    auto infoAddress = reinterpret_cast<uint8_t*>(info);
    const bool wide = m >= 1 && m <= 128 && n >= 256;
    const int columns = wide ? m : n;
    const int limit = m < n ? m : n;
    for (int start = 0; start < limit; start += panel)
    {
        const int nb = limit - start < panel ? limit - start : panel;
        lu950_factor_panel<<<1, nullptr, stream>>>(m, lda, start, nb, address, pivotAddress, infoAddress);
        lu950_factor_swap_solve<<<blocks, nullptr, stream>>>(m, columns, lda, start, nb, address, pivotAddress);
    }
    if (wide)
    {
        const unsigned tasks = static_cast<unsigned>(n - m + 63) / 64;
        if (blocks > tasks) blocks = tasks;
        if (m == 2)
            lu950_factor_wide<2><<<blocks, nullptr, stream>>>(n, lda, address, pivotAddress);
        else if (m == 4)
            lu950_factor_wide<4><<<blocks, nullptr, stream>>>(n, lda, address, pivotAddress);
        else if (m == 8)
            lu950_factor_wide<8><<<blocks, nullptr, stream>>>(n, lda, address, pivotAddress);
        else if (m == 16)
            lu950_factor_wide<16><<<blocks, nullptr, stream>>>(n, lda, address, pivotAddress);
        else if (m == 32)
            lu950_factor_wide<32><<<blocks, nullptr, stream>>>(n, lda, address, pivotAddress);
        else if (m == 64)
            lu950_factor_wide<64><<<blocks, nullptr, stream>>>(n, lda, address, pivotAddress);
        else
            lu950_factor_wide_basic<<<blocks, nullptr, stream>>>(m, n, lda, address, pivotAddress);
    }
}

__global__ __aicore__ void lu950_factor_cube_swap_solve(int m, int n, int lda, int start, int nb, GM_ADDR a,
                                                        GM_ADDR pivots, GM_ADDR negativeUpper)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SwapSolve(m, n, lda, start, nb, a, pivots, negativeUpper, 0);
}

int Lu950FactorCubeLaunch(int m, int n, float* a, int lda, int* pivots, int* info, int panel, unsigned blocks,
                          float* negativeUpper, void* workspace, aclrtStream stream)
{
    auto address = reinterpret_cast<uint8_t*>(a);
    auto pivotAddress = reinterpret_cast<uint8_t*>(pivots);
    auto infoAddress = reinterpret_cast<uint8_t*>(info);
    auto upperAddress = reinterpret_cast<uint8_t*>(negativeUpper);
    const int limit = m < n ? m : n;
    for (int start = 0; start < limit; start += panel)
    {
        const int nb = limit - start < panel ? limit - start : panel;
        const int after = start + nb;
        lu950_factor_panel<<<1, nullptr, stream>>>(m, lda, start, nb, address, pivotAddress, infoAddress);
        lu950_factor_cube_swap_solve<<<blocks, nullptr, stream>>>(m, n, lda, start, nb, address, pivotAddress,
                                                                  upperAddress);
        if (after < m && after < n)
        {
            const int result = Lu950CubeMatmul(
                n - after, m - after, nb, negativeUpper, nb, a + static_cast<uint64_t>(start) * lda + after, lda,
                a + static_cast<uint64_t>(after) * lda + after, lda, workspace, stream, true);
            if (result != 0) return result;
        }
    }
    return 0;
}

__global__ __aicore__ void lu950_factor_inner_range(int m, int end, int first, int lda, int start, int nb, GM_ADDR a,
                                                    GM_ADDR pivots)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SwapSolve(m, end, lda, start, nb, a, pivots, nullptr, first);
}

void Lu950FactorInnerLaunch(int m, int end, int first, float* a, int lda, int start, int nb, int* pivots, int* info,
                            unsigned blocks, aclrtStream stream)
{
    auto address = reinterpret_cast<uint8_t*>(a);
    auto pivotAddress = reinterpret_cast<uint8_t*>(pivots);
    lu950_factor_panel<<<1, nullptr, stream>>>(m, lda, start, nb, address, pivotAddress,
                                               reinterpret_cast<uint8_t*>(info));
    lu950_factor_inner_range<<<blocks, nullptr, stream>>>(m, end, first, lda, start, nb, address, pivotAddress);
}
