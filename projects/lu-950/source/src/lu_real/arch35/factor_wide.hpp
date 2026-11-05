/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// A = [A0 A1], with square A0: factor A0, then U1 = L^-1 P A1.
template <int Rows>
__simd_vf__ inline void FactorWidePrepareVF(__ubuf__ float* stage, __ubuf__ float* rhs, __ubuf__ int32_t* pivots,
                                            __ubuf__ uint32_t* seed, uint32_t columns, bool pivoting)
{
    constexpr int pitch = (Rows + 7) / 8 * 8;
    auto mask = Reg::UpdateMask<float>(columns);
    Reg::RegTensor<uint32_t> lanes, base, index, target;
    Reg::RegTensor<float> value, other;
    Reg::LoadAlign(lanes, seed);
    Reg::Muls(base, lanes, static_cast<uint32_t>(pitch), mask);
    for (int row = 0; row < Rows; ++row)
    {
        Reg::Adds(index, base, static_cast<uint32_t>(row), mask);
        Reg::Gather(value, stage, index, mask);
        Reg::StoreAlign(rhs + row * 64, value, mask);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    if (pivoting)
        for (int k = 0; k < Rows; ++k)
        {
            Reg::Duplicate(index, static_cast<uint32_t>(k), mask);
            Reg::Gather(target, reinterpret_cast<__ubuf__ uint32_t*>(pivots), index, mask);
            Reg::Adds(target, target, uint32_t{0xffffffff}, mask);
            Reg::Muls(target, target, uint32_t{64}, mask);
            Reg::Add(target, target, lanes, mask);
            Reg::LoadAlign(value, rhs + k * 64);
            Reg::Gather(other, rhs, target, mask);
            Reg::StoreAlign(rhs + k * 64, other, mask);
            Reg::Scatter(rhs, value, target, mask);
            Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
        }
}

template <int Rows>
__simd_vf__ inline void FactorWideSolveVF(__ubuf__ float* rhs, __ubuf__ float* factors, __ubuf__ uint32_t* seed,
                                          uint32_t columns, int firstRow, int endRow)
{
    constexpr int pitch = (Rows + 7) / 8 * 8;
    auto mask = Reg::UpdateMask<float>(columns);
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> lanes, base, index;
    Reg::RegTensor<float> value, other, rowFactors, coefficient;
    Reg::LoadAlign(lanes, seed);
    uint32_t factorLanes = Rows < 64 ? Rows : 64;
    auto factorMask = Reg::UpdateMask<float>(factorLanes);
    Reg::Muls(base, lanes, static_cast<uint32_t>(pitch), factorMask);
    for (int row = firstRow; row < endRow; ++row)
    {
        Reg::Adds(index, base, static_cast<uint32_t>(row), factorMask);
        Reg::Gather(rowFactors, factors, index, factorMask);
        Reg::LoadAlign(value, rhs + row * 64);
        for (int k = 0; k < row; ++k)
        {
            Reg::Duplicate(index, static_cast<uint32_t>(k), all);
            Reg::Gather(coefficient, rowFactors, index);
            Reg::Muls(coefficient, coefficient, -1.f, mask);
            Reg::LoadAlign(other, rhs + k * 64);
            Reg::MulAddDst(value, other, coefficient, mask);
        }
        Reg::StoreAlign(rhs + row * 64, value, mask);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

template <int Rows>
__simd_vf__ inline void FactorWidePackVF(__ubuf__ float* stage, __ubuf__ float* rhs, __ubuf__ uint32_t* seed,
                                         uint32_t columns)
{
    constexpr int pitch = (Rows + 7) / 8 * 8;
    auto mask = Reg::UpdateMask<float>(columns);
    Reg::RegTensor<uint32_t> lanes, base, index;
    Reg::RegTensor<float> value;
    Reg::LoadAlign(lanes, seed);
    Reg::Muls(base, lanes, static_cast<uint32_t>(pitch), mask);
    for (int row = 0; row < Rows; ++row)
    {
        Reg::LoadAlign(value, rhs + row * 64);
        Reg::Adds(index, base, static_cast<uint32_t>(row), mask);
        Reg::Scatter(stage, value, index, mask);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <int Rows>
__global__ __aicore__ void lu950_factor_wide(int n, int lda, GM_ADDR a, GM_ADDR pivots)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr int pitch = (Rows + 7) / 8 * 8;
    TPipe pipe;
    TBuf<TPosition::VECIN> stageBuffer, factorBuffer, pivotBuffer;
    TBuf<TPosition::VECCALC> rhsBuffer, seedBuffer;
    pipe.InitBuffer(stageBuffer, pitch * 64 * sizeof(float));
    pipe.InitBuffer(factorBuffer, pitch * Rows * sizeof(float));
    pipe.InitBuffer(rhsBuffer, Rows * 64 * sizeof(float));
    pipe.InitBuffer(pivotBuffer, pitch * sizeof(int));
    pipe.InitBuffer(seedBuffer, 64 * sizeof(uint32_t));
    auto stage = stageBuffer.Get<float>(), factors = factorBuffer.Get<float>(), rhs = rhsBuffer.Get<float>();
    auto localPivots = pivotBuffer.Get<int>();
    auto seed = seedBuffer.Get<uint32_t>();
    GlobalTensor<float> matrix;
    GlobalTensor<int> globalPivots;
    matrix.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    globalPivots.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(pivots));
    DataCopyPad(
        factors, matrix,
        DataCopyExtParams{Rows, Rows * sizeof(float), static_cast<uint32_t>((lda - Rows) * sizeof(float)), 0, 0},
        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(pitch - Rows), 0.f});
    if (pivots != nullptr)
        DataCopyPad(localPivots, globalPivots, DataCopyExtParams{1, Rows * sizeof(int), 0, 0, 0},
                    DataCopyPadExtParams<int>{true, 0, static_cast<uint8_t>(pitch - Rows), 0});
    for (int lane = 0; lane < 64; ++lane) seed.SetValue(lane, lane);
    FactorFence<HardEvent::S_V>();
    FactorFence<HardEvent::MTE2_V>();
    for (int col = Rows + GetBlockIdx() * 64; col < n; col += GetBlockNum() * 64)
    {
        const int count = n - col < 64 ? n - col : 64;
        DataCopyPad(stage, matrix[static_cast<uint64_t>(col) * lda],
                    DataCopyExtParams{static_cast<uint16_t>(count), Rows * sizeof(float),
                                      static_cast<uint32_t>((lda - Rows) * sizeof(float)), 0, 0},
                    DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(pitch - Rows), 0.f});
        FactorFence<HardEvent::MTE2_V>();
        auto* stageAddress = reinterpret_cast<__ubuf__ float*>(stage.GetPhyAddr());
        auto* rhsAddress = reinterpret_cast<__ubuf__ float*>(rhs.GetPhyAddr());
        auto* factorAddress = reinterpret_cast<__ubuf__ float*>(factors.GetPhyAddr());
        auto* pivotAddress = reinterpret_cast<__ubuf__ int32_t*>(localPivots.GetPhyAddr());
        auto* seedAddress = reinterpret_cast<__ubuf__ uint32_t*>(seed.GetPhyAddr());
        FactorWidePrepareVF<Rows>(stageAddress, rhsAddress, pivotAddress, seedAddress, static_cast<uint32_t>(count),
                                  pivots != nullptr);
        for (int first = 0; first < Rows; first += 32)
        {
            const int end = first + 32 < Rows ? first + 32 : Rows;
            FactorWideSolveVF<Rows>(rhsAddress, factorAddress, seedAddress, static_cast<uint32_t>(count), first, end);
        }
        FactorWidePackVF<Rows>(stageAddress, rhsAddress, seedAddress, static_cast<uint32_t>(count));
        FactorFence<HardEvent::V_MTE3>();
        DataCopyPad(matrix[static_cast<uint64_t>(col) * lda], stage,
                    DataCopyExtParams{static_cast<uint16_t>(count), Rows * sizeof(float), 0,
                                      static_cast<uint32_t>((lda - Rows) * sizeof(float)), 0});
        FactorFence<HardEvent::MTE3_MTE2>();
        FactorFence<HardEvent::MTE3_V>();
    }
}

// Keep the 128-row solve in basic vector APIs: one lane per RHS column.
__global__ __aicore__ void lu950_factor_wide_basic(int rows, int n, int lda, GM_ADDR a, GM_ADDR pivots)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr int maxRows = 128, width = 64;
    const int pitch = (rows + 7) / 8 * 8;
    TPipe pipe;
    TBuf<TPosition::VECIN> stageBuffer, factorBuffer, pivotBuffer;
    TBuf<TPosition::VECCALC> rhsBuffer, offsetBuffer, mapBuffer;
    pipe.InitBuffer(stageBuffer, maxRows * width * sizeof(float));
    pipe.InitBuffer(factorBuffer, maxRows * maxRows * sizeof(float));
    pipe.InitBuffer(rhsBuffer, maxRows * width * sizeof(float));
    pipe.InitBuffer(pivotBuffer, maxRows * sizeof(int));
    pipe.InitBuffer(offsetBuffer, (width + maxRows) * sizeof(uint32_t));
    pipe.InitBuffer(mapBuffer, maxRows * sizeof(int));
    auto stage = stageBuffer.Get<float>(), factors = factorBuffer.Get<float>(), rhs = rhsBuffer.Get<float>();
    auto localPivots = pivotBuffer.Get<int>(), permutation = mapBuffer.Get<int>();
    auto offsets = offsetBuffer.Get<uint32_t>(), packOffsets = offsets[width];
    GlobalTensor<float> matrix;
    GlobalTensor<int> globalPivots;
    matrix.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    globalPivots.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(pivots));
    DataCopyPad(factors, matrix,
                DataCopyExtParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(rows * sizeof(float)),
                                  static_cast<uint32_t>((lda - rows) * sizeof(float)), 0, 0},
                DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(pitch - rows), 0.f});
    if (pivots != nullptr)
        DataCopyPad(localPivots, globalPivots, DataCopyExtParams{1, static_cast<uint32_t>(rows * sizeof(int)), 0, 0, 0},
                    DataCopyPadExtParams<int>{true, 0, static_cast<uint8_t>(pitch - rows), 0});
    FactorFence<HardEvent::MTE2_S>();
    for (int row = 0; row < rows; ++row)
    {
        permutation.SetValue(row, row);
        packOffsets.SetValue(row, row * width * sizeof(float));
    }
    for (int col = 0; col < width; ++col) offsets.SetValue(col, col * pitch * sizeof(float));
    if (pivots != nullptr)
        for (int k = 0; k < rows; ++k)
        {
            const int target = localPivots.GetValue(k) - 1;
            const int saved = permutation.GetValue(k);
            permutation.SetValue(k, permutation.GetValue(target));
            permutation.SetValue(target, saved);
        }
    FactorFence<HardEvent::S_V>();
    for (int col = rows + GetBlockIdx() * width; col < n; col += GetBlockNum() * width)
    {
        const int count = n - col < width ? n - col : width;
        DataCopyPad(stage, matrix[static_cast<uint64_t>(col) * lda],
                    DataCopyExtParams{static_cast<uint16_t>(count), static_cast<uint32_t>(rows * sizeof(float)),
                                      static_cast<uint32_t>((lda - rows) * sizeof(float)), 0, 0},
                    DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(pitch - rows), 0.f});
        FactorFence<HardEvent::MTE2_V>();
        for (int row = 0; row < rows; ++row)
        {
            Gather(rhs[row * width], stage, offsets, permutation.GetValue(row) * sizeof(float), count);
            PipeBarrier<PIPE_V>();
        }
        for (int row = 1; row < rows; ++row)
            for (int k = 0; k < row; ++k)
            {
                const float coefficient = -factors.GetValue(k * pitch + row);
                Axpy(rhs[row * width], rhs[k * width], coefficient, count);
                PipeBarrier<PIPE_V>();
            }
        for (int lane = 0; lane < count; ++lane)
        {
            Gather(stage[lane * pitch], rhs, packOffsets, lane * sizeof(float), rows);
            PipeBarrier<PIPE_V>();
        }
        FactorFence<HardEvent::V_MTE3>();
        DataCopyPad(matrix[static_cast<uint64_t>(col) * lda], stage,
                    DataCopyExtParams{static_cast<uint16_t>(count), static_cast<uint32_t>(rows * sizeof(float)), 0,
                                      static_cast<uint32_t>((lda - rows) * sizeof(float)), 0});
        FactorFence<HardEvent::MTE3_MTE2>();
        FactorFence<HardEvent::MTE3_V>();
    }
}
