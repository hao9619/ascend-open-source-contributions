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
__aicore__ inline void Fence()
{
    const auto event = static_cast<int32_t>(GetTPipePtr()->FetchEventID(Event));
    SetFlag<Event>(event);
    WaitFlag<Event>(event);
}

__aicore__ inline void SwapRows(LocalTensor<float> matrix, LocalTensor<float> scratch, int first, int second,
                                int stride)
{
    Muls(scratch, matrix[first * stride], 1.0f, stride);
    PipeBarrier<PIPE_V>();
    Muls(matrix[first * stride], matrix[second * stride], 1.0f, stride);
    PipeBarrier<PIPE_V>();
    Muls(matrix[second * stride], scratch, 1.0f, stride);
    PipeBarrier<PIPE_V>();
}

__aicore__ int EliminateAugmented(LocalTensor<float> augmented, LocalTensor<float> products,
                                  LocalTensor<float> coefficients, LocalTensor<float> broadcast,
                                  LocalTensor<float> scratch, int n, int stride)
{
    const int width = 2 * stride;
    for (int k = 0; k < n; ++k)
    {
        int pivot = k;
        float largest = 0.0f;
        for (int row = k; row < n; ++row)
        {
            const float value = augmented.GetValue(row * width + k);
            const float magnitude = value < 0.0f ? -value : value;
            if (magnitude > largest)
            {
                largest = magnitude;
                pivot = row;
            }
        }
        if (largest == 0.0f) return k + 1;
        if (pivot != k)
        {
            SwapRows(augmented, scratch, k, pivot, width);
            Fence<HardEvent::V_S>();
        }
        const float reciprocal = 1.0f / augmented.GetValue(k * width + k);
        Muls(augmented[k * width], augmented[k * width], reciprocal, width);
        PipeBarrier<PIPE_V>();
        for (int row = 0; row < stride; ++row)
        {
            coefficients.SetValue(row, row < n && row != k ? augmented.GetValue(row * width + k) : 0.0f);
        }
        Fence<HardEvent::S_V>();
        // Eliminate all rows in one outer-product update of [A | I]. Each
        // coefficient occupies one repeated 32-byte block; the pivot row is
        // reused with a zero repeat stride. Its own coefficient is zero.
        Brcb(broadcast, coefficients, static_cast<uint8_t>(stride / 8), BrcbRepeatParams{1, 8});
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams repeat{1, 0, 1, static_cast<uint8_t>(width / 8), 1, 0};
        Mul(products, broadcast, augmented[k * width], static_cast<uint64_t>(width), static_cast<uint8_t>(n), repeat);
        PipeBarrier<PIPE_V>();
        Sub(augmented, augmented, products, n * width);
        Fence<HardEvent::V_S>();
    }
    return 0;
}

__aicore__ int RankOneInverse(LocalTensor<float> columns, LocalTensor<float> augmented, LocalTensor<float> inverse,
                              LocalTensor<float> products, LocalTensor<float> coefficients,
                              LocalTensor<float> broadcast, LocalTensor<float> scratch, int n, int stride)
{
    const int width = 2 * stride;
    for (int row = 0; row < n; ++row)
    {
        for (int col = 0; col < stride; ++col)
        {
            augmented.SetValue(row * width + col, col < n ? columns.GetValue(col * stride + row) : 0.0f);
            augmented.SetValue(row * width + stride + col, row == col ? 1.0f : 0.0f);
        }
    }
    Fence<HardEvent::S_V>();
    const int result = EliminateAugmented(augmented, products, coefficients, broadcast, scratch, n, stride);
    if (result != 0) return result;
    // Extract the inverse half directly into padded column-major storage.
    for (int row = 0; row < n; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            inverse.SetValue(col * stride + row, augmented.GetValue(row * width + stride + col));
        }
    }
    return 0;
}

__aicore__ int TransposeInverse(LocalTensor<float> columns, LocalTensor<float> augmented, LocalTensor<float> inverse,
                                LocalTensor<float> products, LocalTensor<float> coefficients,
                                LocalTensor<float> broadcast, LocalTensor<float> scratch, int n, int stride)
{
    const int width = 2 * stride;
    Duplicate(augmented, 0.0f, n * width);
    PipeBarrier<PIPE_V>();
    // Column-major A is already row-major A^T in the staging buffer.
    for (int row = 0; row < n; ++row)
    {
        Muls(augmented[row * width], columns[row * stride], 1.0f, stride);
    }
    Fence<HardEvent::V_S>();
    for (int row = 0; row < n; ++row)
    {
        augmented.SetValue(row * width + stride + row, 1.0f);
    }
    Fence<HardEvent::S_V>();
    const int result = EliminateAugmented(augmented, products, coefficients, broadcast, scratch, n, stride);
    if (result != 0) return result;
    // Row-major inv(A^T) is column-major inv(A), so no scalar transpose is needed.
    for (int row = 0; row < n; ++row)
    {
        Muls(inverse[row * stride], augmented[row * width + stride], 1.0f, stride);
    }
    Fence<HardEvent::V_S>();
    return 0;
}

#include "matinv_batch.hpp"

__aicore__ void Matinv(int n, GM_ADDR a, int lda, GM_ADDR c, int ldc, GM_ADDR info, int batch, int error)
{
    TPipe pipe;
    TBuf<TPosition::VECIN> aBuffer;
    TBuf<TPosition::VECOUT> cBuffer;
    TBuf<TPosition::VECCALC> rowBuffer;
    TBuf<TPosition::VECCALC> scratchBuffer;
    TBuf<TPosition::VECCALC> productBuffer;
    TBuf<TPosition::VECCALC> coefficientBuffer;
    TBuf<TPosition::VECCALC> broadcastBuffer;
    TBuf<TPosition::VECOUT> infoBuffer;
    pipe.InitBuffer(aBuffer, 32 * 32 * sizeof(float));
    pipe.InitBuffer(cBuffer, 32 * 32 * sizeof(float));
    pipe.InitBuffer(rowBuffer, 32 * 64 * sizeof(float));
    pipe.InitBuffer(scratchBuffer, 64 * sizeof(float));
    pipe.InitBuffer(productBuffer, 32 * 64 * sizeof(float));
    pipe.InitBuffer(coefficientBuffer, 32 * sizeof(float));
    pipe.InitBuffer(broadcastBuffer, 32 * 8 * sizeof(float));
    pipe.InitBuffer(infoBuffer, 32);
    auto localInfo = infoBuffer.Get<int>();
    GlobalTensor<int> globalInfo;
    globalInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    GlobalTensor<uint32_t> aPointers;
    GlobalTensor<uint32_t> cPointers;
    aPointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(a));
    cPointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(c));
    const int stride = (n + 7) / 8 * 8;
    auto matrix = aBuffer.Get<float>();
    auto inverse = cBuffer.Get<float>();
    for (int b = GetBlockIdx(); b < batch; b += GetBlockNum())
    {
        int result = -error;
        if (error == 0 && n > 0)
        {
            auto pointerScratch = matrix.ReinterpretCast<uint32_t>();
            const DataCopyExtParams pointerCopy{1, 8, 0, 0, 0};
            const DataCopyPadExtParams<uint32_t> pointerPad{true, 0, 6, 0};
            DataCopyPad(pointerScratch, aPointers[static_cast<uint64_t>(b) * 2], pointerCopy, pointerPad);
            DataCopyPad(pointerScratch[8], cPointers[static_cast<uint64_t>(b) * 2], pointerCopy, pointerPad);
            Fence<HardEvent::MTE2_S>();
            auto addresses = pointerScratch.ReinterpretCast<uint64_t>();
            const uint64_t input = addresses.GetValue(0), output = addresses.GetValue(4);
            const uint64_t inputBytes = (static_cast<uint64_t>(n - 1) * lda + n) * sizeof(float);
            const uint64_t outputBytes = (static_cast<uint64_t>(n - 1) * ldc + n) * sizeof(float);
            if (input == 0)
            {
                result = -2;
            }
            else if (output == 0 || (input < output + outputBytes && output < input + inputBytes))
            {
                result = -4;
            }
            else
            {
                GlobalTensor<float> globalA;
                GlobalTensor<float> globalC;
                globalA.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
                globalC.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
                const DataCopyExtParams params{1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
                const DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(stride - n), 0.0f};
                for (int col = 0; col < n; ++col)
                {
                    DataCopyPad(matrix[col * stride], globalA[static_cast<uint64_t>(col) * lda], params, pad);
                }
                Fence<HardEvent::MTE2_S>();
                Fence<HardEvent::MTE2_V>();
                result = TransposeInverse(matrix, rowBuffer.Get<float>(), inverse, productBuffer.Get<float>(),
                                          coefficientBuffer.Get<float>(), broadcastBuffer.Get<float>(),
                                          scratchBuffer.Get<float>(), n, stride);
                if (result > 0)
                {
                    // Singular positions of A and A^T may differ. Replay the
                    // original orientation to preserve the required info index.
                    result = RankOneInverse(matrix, rowBuffer.Get<float>(), inverse, productBuffer.Get<float>(),
                                            coefficientBuffer.Get<float>(), broadcastBuffer.Get<float>(),
                                            scratchBuffer.Get<float>(), n, stride);
                }
                if (result == 0)
                {
                    Fence<HardEvent::S_MTE3>();
                    for (int col = 0; col < n; ++col)
                    {
                        DataCopyPad(globalC[static_cast<uint64_t>(col) * ldc], inverse[col * stride], params);
                    }
                }
            }
        }
        localInfo.SetValue(0, result);
        Fence<HardEvent::S_MTE3>();
        SetAtomicAdd<int>();
        DataCopyPad(globalInfo[b], localInfo, DataCopyExtParams{1, sizeof(int), 0, 0, 0});
        SetAtomicNone();
        // TBuf storage is reused for the next matrix only after DMA completes.
        Fence<HardEvent::MTE3_S>();
        Fence<HardEvent::MTE3_MTE2>();
    }
}
}  // namespace
}  // namespace AscendC

using namespace AscendC;

__global__ __aicore__ void lu950_matinv_kernel(int n, GM_ADDR a, int lda, GM_ADDR c, int ldc, GM_ADDR info, int batch,
                                               int error)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (n >= 2 && n <= 8 && error == 0 && batch >= 32)
    {
        BatchMatinvSmall(n, a, lda, c, ldc, info, batch);
        return;
    }
    Matinv(n, a, lda, c, ldc, info, batch, error);
}

void Lu950MatinvLaunch(int n, const float *const *a, int lda, float *const *c, int ldc, int *info, int batch, int error,
                       unsigned blocks, void *stream)
{
    if (!(n >= 2 && n <= 8 && error == 0 && batch >= 32)) Lu950InfoLaunch(info, batch, 0, stream);
    lu950_matinv_kernel<<<blocks, nullptr, stream>>>(n, reinterpret_cast<uint8_t *>(const_cast<const float **>(a)), lda,
                                                     reinterpret_cast<uint8_t *>(const_cast<float **>(c)), ldc,
                                                     reinterpret_cast<uint8_t *>(info), batch, error);
}
