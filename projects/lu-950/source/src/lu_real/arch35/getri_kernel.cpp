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
#define LU950_INLINE __aicore__ inline
#include "lu_math.hpp"

using namespace AscendC;
namespace
{
template <HardEvent Event>
__aicore__ inline void GetriFence()
{
    const auto event = static_cast<int32_t>(GetTPipePtr()->FetchEventID(Event));
    SetFlag<Event>(event);
    WaitFlag<Event>(event);
}
__aicore__ int VectorDotInverse(LocalTensor<float> input, LocalTensor<float> output, LocalTensor<int> pivots,
                                LocalTensor<float> rhs, LocalTensor<float> work, int n, int stride, bool pivoting)
{
    auto rowValues = work;
    auto products = work[128];
    auto reduction = work[256];
    auto temporary = work[384];
    auto offsets = work[512].ReinterpretCast<uint32_t>();
    for (int row = 0; row < n; ++row)
    {
        if (input.GetValue(row * stride + row) == 0.0f) return row + 1;
        if (pivoting && (pivots.GetValue(row) < row + 1 || pivots.GetValue(row) > n)) return -1;
    }
    for (int k = 0; k < n; ++k) offsets.SetValue(k, k * stride * sizeof(float));
    GetriFence<HardEvent::S_V>();
    Duplicate(rowValues, 0.0f, stride);
    PipeBarrier<PIPE_V>();
    for (int col = GetBlockIdx(); col < n; col += GetBlockNum())
    {
        Duplicate(rhs, 0.0f, stride);
        GetriFence<HardEvent::V_S>();
        rhs.SetValue(col, 1.0f);
        if (pivoting)
            for (int k = 0; k < n; ++k)
            {
                const int pivot = pivots.GetValue(k) - 1;
                const float value = rhs.GetValue(k);
                rhs.SetValue(k, rhs.GetValue(pivot));
                rhs.SetValue(pivot, value);
            }
        for (int row = 1; row < n; ++row)
        {
            GetriFence<HardEvent::S_V>();
            Gather(rowValues, input, offsets, row * sizeof(float), n);
            PipeBarrier<PIPE_V>();
            Mul(products, rowValues, rhs, stride);
            PipeBarrier<PIPE_V>();
            ReduceSum(reduction, products, temporary, row);
            GetriFence<HardEvent::V_S>();
            rhs.SetValue(row, rhs.GetValue(row) - reduction.GetValue(0));
        }
        for (int row = n - 1; row >= 0; --row)
        {
            float sum = 0.0f;
            if (row < n - 1)
            {
                GetriFence<HardEvent::S_V>();
                Gather(rowValues, input, offsets, row * sizeof(float), n);
                PipeBarrier<PIPE_V>();
                Mul(products, rowValues, rhs, stride);
                PipeBarrier<PIPE_V>();
                Duplicate(products, 0.0f, row + 1);
                PipeBarrier<PIPE_V>();
                ReduceSum(reduction, products, temporary, n);
                GetriFence<HardEvent::V_S>();
                sum = reduction.GetValue(0);
            }
            rhs.SetValue(row, (rhs.GetValue(row) - sum) / input.GetValue(row * stride + row));
        }
        for (int row = 0; row < n; ++row) output.SetValue(col * stride + row, rhs.GetValue(row));
    }
    return 0;
}

struct Matrix
{
    LocalTensor<float> values;
    int stride;
    __aicore__ float Get(int row, int col) const { return values.GetValue(col * stride + row); }
    __aicore__ void Set(int row, int col, float value) { values.SetValue(col * stride + row, value); }
};
struct Pivots
{
    LocalTensor<int> values;
    __aicore__ int Get(int row) const { return values.GetValue(row); }
};
struct Vector
{
    LocalTensor<float> values;
    __aicore__ float Get(int row) const { return values.GetValue(row); }
    __aicore__ void Set(int row, float value) { values.SetValue(row, value); }
};
}  // namespace

__global__ __aicore__ void lu950_getri_copy(int n, int lda, GM_ADDR a, GM_ADDR workspace)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECIN> buffer;
    pipe.InitBuffer(buffer, 4096 * sizeof(float));
    auto local = buffer.Get<float>();
    GlobalTensor<float> input, packed;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
    packed.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    const DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
    for (int col = GetBlockIdx(); col < n; col += GetBlockNum())
    {
        DataCopyPad(local, input[static_cast<uint64_t>(col) * lda], copy,
                    DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - n % 8) % 8), 0.0f});
        GetriFence<HardEvent::MTE2_MTE3>();
        DataCopyPad(packed[col * ((n + 7) / 8 * 8)], local, copy);
        GetriFence<HardEvent::MTE3_MTE2>();
    }
}

__global__ __aicore__ void lu950_getri_solve(int n, int lda, GM_ADDR a, GM_ADDR pivots, GM_ADDR info, GM_ADDR workspace)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECIN> inputBuffer, pivotBuffer;
    TBuf<TPosition::VECOUT> outputBuffer, infoBuffer;
    TBuf<TPosition::VECCALC> rhsBuffer, workBuffer;
    pipe.InitBuffer(inputBuffer, 128 * 128 * sizeof(float));
    pipe.InitBuffer(outputBuffer, 128 * 128 * sizeof(float));
    pipe.InitBuffer(pivotBuffer, 128 * sizeof(int));
    pipe.InitBuffer(rhsBuffer, 128 * sizeof(float));
    pipe.InitBuffer(workBuffer, 640 * sizeof(float));
    pipe.InitBuffer(infoBuffer, 32);
    auto input = inputBuffer.Get<float>(), output = outputBuffer.Get<float>();
    auto localPivots = pivotBuffer.Get<int>();
    auto rhs = rhsBuffer.Get<float>();
    auto localInfo = infoBuffer.Get<int>();
    GlobalTensor<float> packed, result;
    GlobalTensor<int> gpivots, ginfo;
    packed.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    result.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
    gpivots.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(pivots));
    ginfo.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    const int stride = (n + 7) / 8 * 8;
    const DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
    for (int col = 0; col < n; ++col)
        DataCopyPad(input[col * stride], packed[col * ((n + 7) / 8 * 8)], copy,
                    DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(stride - n), 0.0f});
    if (pivots != nullptr)
        DataCopyPad(localPivots, gpivots, copy,
                    DataCopyPadExtParams<int>{true, 0, static_cast<uint8_t>(stride - n), 0});
    GetriFence<HardEvent::MTE2_S>();
    const int resultCode =
        n <= 4 ? lu950::InvertSmallLapack(Matrix{input, stride}, Matrix{output, stride}, Pivots{localPivots}, n,
                                          pivots != nullptr)
        : n > 32
            ? VectorDotInverse(input, output, localPivots, rhs, workBuffer.Get<float>(), n, stride, pivots != nullptr)
            : lu950::InvertFactors(Matrix{input, stride}, Matrix{output, stride}, Pivots{localPivots}, Vector{rhs}, n,
                                   pivots != nullptr, GetBlockIdx(), GetBlockNum());
    const int code = resultCode == -1 ? -4 : resultCode;
    if (GetBlockIdx() == 0)
    {
        localInfo.SetValue(0, code);
        GetriFence<HardEvent::S_MTE3>();
        DataCopyPad(ginfo, localInfo, DataCopyExtParams{1, sizeof(int), 0, 0, 0});
    }
    if (code == 0)
    {
        GetriFence<HardEvent::S_MTE3>();
        for (int col = GetBlockIdx(); col < n; col += GetBlockNum())
            DataCopyPad(result[static_cast<uint64_t>(col) * lda], output[col * stride], copy);
    }
    GetriFence<HardEvent::MTE3_S>();
}

// Serialize only the final unaligned stores; all triangular solves use aligned columns.
__global__ __aicore__ void lu950_getri_unpack(int n, int lda, GM_ADDR a, GM_ADDR packed, GM_ADDR info)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECIN> buffer;
    pipe.InitBuffer(buffer, 4096 * sizeof(float));
    auto local = buffer.Get<float>();
    GlobalTensor<float> source, destination;
    GlobalTensor<int> globalInfo;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(packed));
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
    globalInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    DataCopyPad(local.ReinterpretCast<int>(), globalInfo, DataCopyExtParams{1, sizeof(int), 0, 0, 0},
                DataCopyPadExtParams<int>{true, 0, 7, 0});
    GetriFence<HardEvent::MTE2_S>();
    if (local.ReinterpretCast<int>().GetValue(0) != 0) return;
    const int stride = (n + 7) / 8 * 8;
    const DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
    for (int col = 0; col < n; ++col)
    {
        DataCopyPad(local, source[col * stride], copy,
                    DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(stride - n), 0.f});
        GetriFence<HardEvent::MTE2_MTE3>();
        DataCopyPad(destination[static_cast<uint64_t>(col) * lda], local, copy);
        GetriFence<HardEvent::MTE3_MTE2>();
    }
}

// Preserve both product and addition rounding errors when unpivoted factors have large growth.
__simd_vf__ inline void NoPivotProductsVF(__ubuf__ float *coefficients, __ubuf__ float *rhs, __ubuf__ float *high,
                                          __ubuf__ float *low, __ubuf__ uint32_t *seed, int begin, int count)
{
    Reg::RegTensor<uint32_t> lanes, indices;
    Reg::RegTensor<float> coefficient, source, product, residual;
    Reg::LoadAlign(lanes, seed);
    for (int base = 0; base < count; base += 64)
    {
        uint32_t active = static_cast<uint32_t>(count - base < 64 ? count - base : 64);
        auto mask = Reg::UpdateMask<float>(active);
        Reg::Muls(indices, lanes, uint32_t{8}, mask);
        Reg::Gather(coefficient, coefficients + base * 8, indices, mask);
        Reg::Adds(indices, lanes, static_cast<uint32_t>(begin + base), mask);
        Reg::Gather(source, rhs, indices, mask);
        Reg::Muls(coefficient, coefficient, -1.f, mask);
        Reg::Mul(product, source, coefficient, mask);
        Reg::Muls(residual, product, -1.f, mask);
        Reg::MulAddDst(residual, source, coefficient, mask);
        Reg::StoreAlign(high + base, product, mask);
        Reg::StoreAlign(low + base, residual, mask);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__aicore__ float NoPivotRowValue(GlobalTensor<float> factors, LocalTensor<float> coefficients, LocalTensor<float> rhs,
                                 LocalTensor<float> high, LocalTensor<float> low, LocalTensor<uint32_t> seed, int row,
                                 int begin, int count, int stride)
{
    float value = rhs.GetValue(row);
    if (count == 0) return value;
    DataCopyPad(coefficients, factors[static_cast<uint64_t>(begin) * stride + row],
                DataCopyExtParams{static_cast<uint16_t>(count), sizeof(float),
                                  static_cast<uint32_t>((stride - 1) * sizeof(float)), 0, 0},
                DataCopyPadExtParams<float>{true, 0, 7, 0.f});
    GetriFence<HardEvent::MTE2_V>();
    GetriFence<HardEvent::S_V>();
    NoPivotProductsVF(reinterpret_cast<__ubuf__ float *>(coefficients.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float *>(rhs.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float *>(high.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float *>(low.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(seed.GetPhyAddr()), begin, count);
    GetriFence<HardEvent::V_S>();
    float correction = 0.f;
    for (int k = 0; k < count; ++k)
    {
        const float product = high.GetValue(k);
        const float updated = value + product;
        const float added = updated - value;
        correction += ((value - (updated - added)) + (product - added)) + low.GetValue(k);
        value = updated;
    }
    return value + correction;
}

__global__ __aicore__ void lu950_getri_nopivot_compensated(int n, GM_ADDR factors, GM_ADDR output, GM_ADDR info)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECIN> coefficientBuffer;
    TBuf<TPosition::VECCALC> rhsBuffer, highBuffer, lowBuffer, diagonalBuffer, seedBuffer;
    TBuf<TPosition::VECOUT> infoBuffer;
    pipe.InitBuffer(coefficientBuffer, 4096 * 8 * sizeof(float));
    pipe.InitBuffer(rhsBuffer, 4096 * sizeof(float));
    pipe.InitBuffer(highBuffer, 4096 * sizeof(float));
    pipe.InitBuffer(lowBuffer, 4096 * sizeof(float));
    pipe.InitBuffer(diagonalBuffer, 4096 * sizeof(float));
    pipe.InitBuffer(seedBuffer, 64 * sizeof(uint32_t));
    pipe.InitBuffer(infoBuffer, 32);
    auto coefficients = coefficientBuffer.Get<float>(), rhs = rhsBuffer.Get<float>();
    auto high = highBuffer.Get<float>(), low = lowBuffer.Get<float>(), diagonal = diagonalBuffer.Get<float>();
    auto seed = seedBuffer.Get<uint32_t>();
    auto localInfo = infoBuffer.Get<int>();
    GlobalTensor<float> input, result;
    GlobalTensor<int> globalInfo;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(factors));
    result.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
    globalInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    const int stride = (n + 7) / 8 * 8;
    // DataCopyPad permits at most 4095 blocks; chunk the 4096-entry diagonal as well.
    for (int base = 0; base < n; base += 2048)
    {
        const int count = n - base < 2048 ? n - base : 2048;
        DataCopyPad(coefficients[base * 8], input[static_cast<uint64_t>(base) * (stride + 1)],
                    DataCopyExtParams{static_cast<uint16_t>(count), sizeof(float),
                                      static_cast<uint32_t>(stride * sizeof(float)), 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 7, 0.f});
    }
    GetriFence<HardEvent::MTE2_S>();
    int code = 0;
    for (int row = 0; row < n; ++row)
    {
        const float value = coefficients.GetValue(row * 8);
        diagonal.SetValue(row, value);
        if (value == 0.f && code == 0) code = row + 1;
    }
    for (int lane = 0; lane < 64; ++lane) seed.SetValue(lane, lane);
    if (GetBlockIdx() == 0)
    {
        localInfo.SetValue(0, code);
        GetriFence<HardEvent::S_MTE3>();
        DataCopyPad(globalInfo, localInfo, DataCopyExtParams{1, sizeof(int), 0, 0, 0});
        GetriFence<HardEvent::MTE3_S>();
    }
    if (code != 0) return;
    for (int col = GetBlockIdx(); col < n; col += GetBlockNum())
    {
        for (int row = 0; row < n; ++row) rhs.SetValue(row, row == col ? 1.f : 0.f);
        for (int row = 0; row < n; ++row)
            rhs.SetValue(row, NoPivotRowValue(input, coefficients, rhs, high, low, seed, row, 0, row, stride));
        for (int row = n - 1; row >= 0; --row)
        {
            const float value =
                NoPivotRowValue(input, coefficients, rhs, high, low, seed, row, row + 1, n - row - 1, stride);
            rhs.SetValue(row, value / diagonal.GetValue(row));
        }
        GetriFence<HardEvent::S_MTE3>();
        DataCopyPad(result[static_cast<uint64_t>(col) * stride], rhs,
                    DataCopyExtParams{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0});
        GetriFence<HardEvent::MTE3_S>();
    }
}

void Lu950NoPivotGetriLaunch(int n, float *a, int lda, int *info, float *workspace, unsigned blocks, void *stream)
{
    const int stride = (n + 7) / 8 * 8;
    auto factors = reinterpret_cast<uint8_t *>(workspace);
    auto output = reinterpret_cast<uint8_t *>(workspace + n * stride);
    lu950_getri_copy<<<blocks, nullptr, stream>>>(n, lda, reinterpret_cast<uint8_t *>(a), factors);
    lu950_getri_nopivot_compensated<<<blocks, nullptr, stream>>>(n, factors, output, reinterpret_cast<uint8_t *>(info));
    lu950_getri_unpack<<<1, nullptr, stream>>>(n, lda, reinterpret_cast<uint8_t *>(a), output,
                                               reinterpret_cast<uint8_t *>(info));
}

// Pointer arrays occupy the otherwise unused output half of the caller's two-matrix workspace.
__global__ __aicore__ void lu950_getri_batch_pointers(GM_ADDR scratch, GM_ADDR factors, GM_ADDR result)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECOUT> buffer;
    pipe.InitBuffer(buffer, 64);
    auto local = buffer.Get<uint32_t>();
    auto addresses = local.ReinterpretCast<uint64_t>();
    addresses.SetValue(0, reinterpret_cast<uint64_t>(factors));
    addresses.SetValue(4, reinterpret_cast<uint64_t>(result));
    GlobalTensor<uint32_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(scratch));
    GetriFence<HardEvent::S_MTE3>();
    DataCopyPad(output, local, DataCopyExtParams{1, 8, 0, 0, 0});
    DataCopyPad(output[8], local[8], DataCopyExtParams{1, 8, 0, 0, 0});
    GetriFence<HardEvent::MTE3_S>();
}

void Lu950GetriLaunch(int n, float *a, int lda, const int *pivots, int *info, float *workspace, unsigned blocks,
                      void *stream)
{
    auto address = reinterpret_cast<uint8_t *>(a);
    auto factors = reinterpret_cast<uint8_t *>(workspace);
    lu950_getri_copy<<<blocks, nullptr, stream>>>(n, lda, address, factors);
    if (n >= 5 && n <= 32)
    {
        const int stride = (n + 7) / 8 * 8;
        auto *pointerStorage = workspace + n * stride;
        lu950_getri_batch_pointers<<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(pointerStorage), factors,
                                                           address);
        Lu950BatchGetriLaunch(n, reinterpret_cast<const float *const *>(pointerStorage), stride, pivots,
                              reinterpret_cast<float *const *>(pointerStorage + 8), lda, info, 1, blocks, stream);
        return;
    }
    if (lda % 8 || reinterpret_cast<uintptr_t>(a) % 32)
    {
        const int stride = (n + 7) / 8 * 8;
        auto output = reinterpret_cast<uint8_t *>(workspace + n * stride);
        lu950_getri_solve<<<blocks, nullptr, stream>>>(n, stride, output,
                                                       reinterpret_cast<uint8_t *>(const_cast<int *>(pivots)),
                                                       reinterpret_cast<uint8_t *>(info), factors);
        lu950_getri_unpack<<<1, nullptr, stream>>>(n, lda, address, output, reinterpret_cast<uint8_t *>(info));
        return;
    }
    lu950_getri_solve<<<blocks, nullptr, stream>>>(n, lda, address,
                                                   reinterpret_cast<uint8_t *>(const_cast<int *>(pivots)),
                                                   reinterpret_cast<uint8_t *>(info), factors);
}
