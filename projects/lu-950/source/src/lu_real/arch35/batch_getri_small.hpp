/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

__aicore__ void BatchGetriSmall(int n, GM_ADDR a, int lda, GM_ADDR pivots, GM_ADDR c, int ldc, GM_ADDR info, int batch,
                                int group, bool raw = false)
{
    if (group <= 0) return;
    const int padded = (n + 7) / 8 * 8;
    const int width = n * group;
#ifdef LU950_FUSED_BATCH
    const bool reuseInverse = raw && n >= 17;
#else
    const bool reuseInverse = false;
#endif
    const bool compactStage = reuseInverse && lda == n && ldc == n;
    const int stageSlot = compactStage ? (n * n + 7) / 8 * 8 : n * padded;
    TPipe pipe;
    TBuf<TPosition::VECIN> stageBuffer, pointerBuffer, pivotBuffer;
    TBuf<TPosition::VECCALC> factorBuffer, inverseBuffer, rowBuffer, indexBuffer, outputOffsetsBuffer;
    TBuf<TPosition::VECCALC> valuesBuffer, maskBuffer;
    TBuf<TPosition::VECOUT> infoBuffer;
    pipe.InitBuffer(stageBuffer, group * stageSlot * sizeof(float));
    pipe.InitBuffer(factorBuffer, (reuseInverse ? stageSlot * group : n * width) * sizeof(float));
    if (!reuseInverse)
    {
        pipe.InitBuffer(inverseBuffer, n * width * sizeof(float));
        pipe.InitBuffer(rowBuffer, 2 * width * sizeof(float));
    }
    pipe.InitBuffer(pointerBuffer, 4 * group * sizeof(uint32_t));
    pipe.InitBuffer(pivotBuffer, group * n * sizeof(int));
    pipe.InitBuffer(indexBuffer, 6 * group * sizeof(uint32_t));
    pipe.InitBuffer(outputOffsetsBuffer, (stageSlot > 128 ? stageSlot : 128) * sizeof(uint32_t));
    pipe.InitBuffer(valuesBuffer, 8 * group * sizeof(float));
    pipe.InitBuffer(maskBuffer, 2 * ((group + 255) / 256 * 32));
    pipe.InitBuffer(infoBuffer, group * sizeof(int));
    auto stage = stageBuffer.Get<float>(), factors = factorBuffer.Get<float>();
    // Original staged input is dead after the fused transpose; LU storage is dead before output packing.
    auto inverse = reuseInverse ? stage : inverseBuffer.Get<float>();
    auto product = reuseInverse ? stage : rowBuffer.Get<float>();
    auto saved = product[width];
    auto packedStage = reuseInverse ? factors : stage;
    auto rawA = pointerBuffer.Get<uint32_t>(), rawC = rawA[2 * group];
    auto rawPivots = pivotBuffer.Get<int>();
    auto indices = indexBuffer.Get<int32_t>();
    auto offsets = indices[group].ReinterpretCast<uint32_t>();
    auto expected = indices[2 * group], actual = indices[3 * group];
    auto matrixOffsets = indices[4 * group].ReinterpretCast<uint32_t>();
    auto pivotOffsets = indices[5 * group].ReinterpretCast<uint32_t>();
    auto outputOffsets = outputOffsetsBuffer.Get<uint32_t>();
    auto ones = valuesBuffer.Get<float>(), infoFloat = ones[group], candidate = ones[2 * group];
    auto nextInfo = ones[3 * group], safeDiagonal = ones[4 * group], pivotFloat = ones[5 * group];
    auto pivotIntegers = ones[6 * group].ReinterpretCast<int32_t>();
    auto invalidPivot = ones[7 * group];
    auto mask = maskBuffer.Get<uint8_t>(), freeMask = mask[(group + 255) / 256 * 32];
    auto localInfo = infoBuffer.Get<int>();
    for (int lane = 0; lane < group; ++lane)
    {
        indices.SetValue(lane, lane);
        offsets.SetValue(lane, lane * sizeof(uint64_t));
        pivotOffsets.SetValue(lane, lane * n * sizeof(int));
    }
    const bool cachedRawSeeds = raw && n >= 18 && n <= 32 && group <= 64;
    if (cachedRawSeeds)
    {
        const int divisor = group > 0 ? group : 1;
        for (int lane = 0; lane < 64; ++lane)
        {
            outputOffsets.SetValue(lane, lane % divisor);
            outputOffsets.SetValue(64 + lane, (lane % divisor) * stageSlot + lane / divisor);
        }
    }
    BatchFence<HardEvent::S_V>();
    Duplicate(ones, 1.f, group);
    Duplicate(invalidPivot, -4.f, group);
    PipeBarrier<PIPE_V>();
    GlobalTensor<uint32_t> aPointers, cPointers;
    GlobalTensor<int> globalPivots, globalInfo;
    aPointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a));
    cPointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(c));
    globalPivots.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(pivots));
    globalInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(info));
    const int chunk = group < 64 ? group : 64;
    const BinaryRepeatParams rows{
        1, 1, 1, static_cast<uint8_t>(group / 8), static_cast<uint8_t>(group / 8), static_cast<uint8_t>(group / 8)};
    const BinaryRepeatParams broadcast{1, 1, 1, static_cast<uint8_t>(group / 8), static_cast<uint8_t>(group / 8), 0};
    for (int start = GetBlockIdx() * group; start < batch; start += GetBlockNum() * group)
    {
        const int count = batch - start < group ? batch - start : group;
        const uint32_t pointerBytes = count * sizeof(uint64_t);
        const auto pointerPad =
            DataCopyPadExtParams<uint32_t>{true, 0, static_cast<uint8_t>((8 - count * 2 % 8) % 8), 0};
        DataCopyPad(rawA, aPointers[start * 2], DataCopyExtParams{1, pointerBytes, 0, 0, 0}, pointerPad);
        DataCopyPad(rawC, cPointers[start * 2], DataCopyExtParams{1, pointerBytes, 0, 0, 0}, pointerPad);
        if (pivots != nullptr)
            DataCopyPad(rawPivots, globalPivots[static_cast<uint64_t>(start) * n],
                        DataCopyExtParams{1, static_cast<uint32_t>(count * n * sizeof(int)), 0, 0, 0},
                        DataCopyPadExtParams<int>{true, 0, static_cast<uint8_t>((8 - count * n % 8) % 8), 0});
        Duplicate(localInfo, 0, group);
        BatchFence<HardEvent::MTE2_S>();
        BatchFence<HardEvent::MTE2_V>();
        auto ap = rawA.ReinterpretCast<uint64_t>(), cp = rawC.ReinterpretCast<uint64_t>();
        const uint64_t firstA = ap.GetValue(0), firstC = cp.GetValue(0);
        const uint64_t span = static_cast<uint64_t>(count) * n * n * sizeof(float);
        bool dense = lda == n && ldc == n && !(firstA < firstC + span && firstC < firstA + span);
        if (dense)
            dense = BatchDensePointers(rawA, indices, offsets, expected, actual, mask, group, count, n * n * 4) &&
                    BatchDensePointers(rawC, indices, offsets, expected, actual, mask, group, count, n * n * 4);
        if (dense)
        {
            GlobalTensor<float> source;
            source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(firstA));
            DataCopyPad(stage, source,
                        DataCopyExtParams{1, static_cast<uint32_t>(count * n * n * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - count * n * n % 8) % 8), 0.f});
        }
        else
        {
            Duplicate(stage, 0.f, group * stageSlot);
            BatchFence<HardEvent::V_MTE2>();
            BatchFence<HardEvent::V_S>();
            for (int lane = 0; lane < count; ++lane)
            {
                const uint64_t av = ap.GetValue(lane), cv = cp.GetValue(lane);
                const uint64_t aBytes = (static_cast<uint64_t>(n - 1) * lda + n) * sizeof(float);
                const uint64_t cBytes = (static_cast<uint64_t>(n - 1) * ldc + n) * sizeof(float);
                const int error =
                    av == 0 ? -2 : (cv == 0 || (av < cv + cBytes && cv < av + aBytes) ? (raw ? -4 : -5) : 0);
                localInfo.SetValue(lane, error);
                if (error != 0) continue;
                GlobalTensor<float> source;
                source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(av));
                if (compactStage)
                    DataCopyPad(stage[lane * stageSlot], source,
                                DataCopyExtParams{1, static_cast<uint32_t>(n * n * sizeof(float)), 0, 0, 0},
                                DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(stageSlot - n * n), 0.f});
                else
                    DataCopyPad(stage[lane * stageSlot], source,
                                DataCopyExtParams{static_cast<uint16_t>(n), static_cast<uint32_t>(n * sizeof(float)),
                                                  static_cast<uint32_t>((lda - n) * sizeof(float)), 0, 0},
                                DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(padded - n), 0.f});
            }
        }
        BatchFence<HardEvent::MTE2_V>();
        BatchFence<HardEvent::S_V>();
        bool prepared = false;
#ifdef LU950_FUSED_BATCH
        prepared =
            BatchFusedPrepare(n, inverse, factors, stage, rawPivots, localInfo, saved, indices, group,
                              compactStage ? n : padded, dense, pivots != nullptr, raw, compactStage ? stageSlot : 0);
        PipeBarrier<PIPE_V>();
#endif
        if (!prepared)
        {
            Cast(infoFloat, localInfo, RoundMode::CAST_NONE, group);
            Muls(matrixOffsets.ReinterpretCast<int32_t>(), indices, (dense ? n * n : stageSlot) * sizeof(float), group);
            Duplicate(inverse, 0.f, n * width);
            PipeBarrier<PIPE_V>();
            for (int row = 0; row < n; ++row)
            {
                Duplicate(inverse[(row * n + row) * group], 1.f, group);
                for (int col = 0; col < n; ++col)
                {
                    Adds(actual, matrixOffsets.ReinterpretCast<int32_t>(),
                         (col * (dense || compactStage ? n : padded) + row) * sizeof(float), group);
                    PipeBarrier<PIPE_V>();
                    Gather(factors[(row * n + col) * group], stage, actual.ReinterpretCast<uint32_t>(), 0, group);
                    PipeBarrier<PIPE_V>();
                }
            }
            Muls(factors, factors, -1.f, n * width);
            PipeBarrier<PIPE_V>();
            for (int k = 0; k < n; ++k)
            {
                Muls(factors[(k * n + k) * group], factors[(k * n + k) * group], -1.f, group);
                PipeBarrier<PIPE_V>();
            }
            if (pivots != nullptr)
                for (int k = 0; k < n; ++k)
                {
                    Gather(pivotIntegers, rawPivots, pivotOffsets, k * sizeof(int), group);
                    PipeBarrier<PIPE_V>();
                    Cast(pivotFloat, pivotIntegers, RoundMode::CAST_NONE, group);
                    PipeBarrier<PIPE_V>();
                    for (int bound = 0; bound < 2; ++bound)
                    {
                        CompareScalar(mask, pivotFloat, static_cast<float>(bound == 0 ? k + 1 : n),
                                      bound == 0 ? CMPMODE::LT : CMPMODE::GT, group);
                        CompareScalar(freeMask, infoFloat, 0.f, CMPMODE::EQ, group);
                        PipeBarrier<PIPE_V>();
                        Select(nextInfo, mask, invalidPivot, infoFloat, SELMODE::VSEL_TENSOR_TENSOR_MODE, group);
                        PipeBarrier<PIPE_V>();
                        Select(infoFloat, freeMask, nextInfo, infoFloat, SELMODE::VSEL_TENSOR_TENSOR_MODE, group);
                        PipeBarrier<PIPE_V>();
                    }
                    for (int target = k + 1; target < n; ++target)
                    {
                        CompareScalar(mask, pivotFloat, static_cast<float>(target + 1), CMPMODE::EQ, group);
                        PipeBarrier<PIPE_V>();
                        if (group > 64) BatchFence<HardEvent::V_S>();
                        for (int lane = 0; lane < group; lane += chunk)
                        {
                            if (group > 64)
                            {
                                freeMask.ReinterpretCast<uint64_t>().SetValue(
                                    0, mask.ReinterpretCast<uint64_t>().GetValue(lane / 64));
                                BatchFence<HardEvent::S_V>();
                            }
                            auto alignedMask = group > 64 ? freeMask : mask;
                            Select(saved[lane], alignedMask, inverse[k * width + lane], inverse[target * width + lane],
                                   SELMODE::VSEL_CMPMASK_SPR, static_cast<uint64_t>(chunk), n, rows);
                            Select(inverse[k * width + lane], alignedMask, inverse[target * width + lane],
                                   inverse[k * width + lane], SELMODE::VSEL_CMPMASK_SPR, static_cast<uint64_t>(chunk),
                                   n, rows);
                            PipeBarrier<PIPE_V>();
                            if (group > 64) BatchFence<HardEvent::V_S>();
                        }
                        Muls(inverse[target * width], saved, 1.f, width);
                        PipeBarrier<PIPE_V>();
                    }
                }
        }
        else
        {
#ifdef LU950_RAW_BATCH
            if (raw)
            {
                BatchFusedFactor(n, factors, rawPivots, localInfo, outputOffsets, group, cachedRawSeeds);
                PipeBarrier<PIPE_V>();
            }
#endif
            if (!(raw && n >= 18))
            {
                Cast(infoFloat, localInfo, RoundMode::CAST_NONE, group);
                PipeBarrier<PIPE_V>();
            }
        }
        bool fused = false;
#ifdef LU950_FUSED_BATCH
        fused = BatchFusedTriangular(n, inverse, factors, rawPivots, outputOffsets, group, pivots != nullptr || raw,
                                     raw && prepared && n >= 18 && group <= 64);
        PipeBarrier<PIPE_V>();
#endif
        if (n != 2 && !fused)
        {
            // Forward unit-lower triangular solve, followed by the upper solve.
            for (int k = 0; k < n; ++k)
                for (int row = k + 1; row < n; ++row)
                {
                    for (int lane = 0; lane < group; lane += chunk)
                        MulAddDst(inverse[row * width + lane], inverse[k * width + lane],
                                  factors[(row * n + k) * group + lane], static_cast<uint64_t>(chunk), n, broadcast);
                    PipeBarrier<PIPE_V>();
                }
        }
        // Fused raw LU has already selected the first zero pivot into localInfo.
        if (!(raw && prepared && n >= 18))
        {
            for (int k = 0; k < n; ++k)
            {
                CompareScalar(mask, factors[(k * n + k) * group], 0.f, CMPMODE::EQ, group);
                CompareScalar(freeMask, infoFloat, 0.f, CMPMODE::EQ, group);
                Duplicate(candidate, static_cast<float>(k + 1), group);
                PipeBarrier<PIPE_V>();
                Select(nextInfo, mask, candidate, infoFloat, SELMODE::VSEL_TENSOR_TENSOR_MODE, group);
                PipeBarrier<PIPE_V>();
                Select(infoFloat, freeMask, nextInfo, infoFloat, SELMODE::VSEL_TENSOR_TENSOR_MODE, group);
                PipeBarrier<PIPE_V>();
            }
        }
        if (n == 2)
        {
            // Invert U first, then remove L, matching the FP32 LAPACK order.
            // A rounded reciprocal is shared across the two columns.
            for (int k = 0; k < 2; ++k)
            {
                CompareScalar(mask, factors[(k * 2 + k) * group], 0.f, CMPMODE::EQ, group);
                PipeBarrier<PIPE_V>();
                Select(safeDiagonal, mask, ones, factors[(k * 2 + k) * group], SELMODE::VSEL_TENSOR_TENSOR_MODE, group);
                PipeBarrier<PIPE_V>();
                auto diagonal = inverse[(k * 2 + k) * group];
                Div(diagonal, ones, safeDiagonal, group);
                Muls(candidate, safeDiagonal, -1.f, group);
                Muls(product, ones, 1.f, group);
                PipeBarrier<PIPE_V>();
                MulAddDst(product, diagonal, candidate, group);
                PipeBarrier<PIPE_V>();
                Div(nextInfo, product, safeDiagonal, group);
                PipeBarrier<PIPE_V>();
                Add(diagonal, diagonal, nextInfo, group);
                PipeBarrier<PIPE_V>();
            }
            Mul(inverse[group], inverse, factors[group], group);
            PipeBarrier<PIPE_V>();
            Mul(inverse[group], inverse[group], inverse[3 * group], group);
            Mul(inverse[2 * group], inverse[3 * group], factors[2 * group], group);
            PipeBarrier<PIPE_V>();
            Mul(product, inverse[group], factors[2 * group], group);
            PipeBarrier<PIPE_V>();
            Add(inverse, inverse, product, group);
            PipeBarrier<PIPE_V>();
            if (pivots != nullptr)
            {
                Gather(pivotIntegers, rawPivots, pivotOffsets, 0, group);
                PipeBarrier<PIPE_V>();
                CompareScalar(mask, pivotIntegers, 2, CMPMODE::EQ, group);
                PipeBarrier<PIPE_V>();
                for (int row = 0; row < 2; ++row)
                {
                    Muls(saved, inverse[row * width], 1.f, group);
                    PipeBarrier<PIPE_V>();
                    Select(inverse[row * width], mask, inverse[row * width + group], saved,
                           SELMODE::VSEL_TENSOR_TENSOR_MODE, group);
                    PipeBarrier<PIPE_V>();
                    Select(inverse[row * width + group], mask, saved, inverse[row * width + group],
                           SELMODE::VSEL_TENSOR_TENSOR_MODE, group);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
        else if (!fused)
        {
            for (int k = n - 1; k >= 0; --k)
            {
                CompareScalar(mask, factors[(k * n + k) * group], 0.f, CMPMODE::EQ, group);
                PipeBarrier<PIPE_V>();
                Select(safeDiagonal, mask, ones, factors[(k * n + k) * group], SELMODE::VSEL_TENSOR_TENSOR_MODE, group);
                PipeBarrier<PIPE_V>();
                Muls(saved, inverse[k * width], 1.f, width);
                PipeBarrier<PIPE_V>();
                for (int lane = 0; lane < group; lane += chunk)
                    Div(inverse[k * width + lane], inverse[k * width + lane], safeDiagonal[lane],
                        static_cast<uint64_t>(chunk), n, broadcast);
                PipeBarrier<PIPE_V>();
                // Correct the approximate vector quotient using the fused residual.
                Muls(candidate, safeDiagonal, -1.f, group);
                PipeBarrier<PIPE_V>();
                for (int lane = 0; lane < group; lane += chunk)
                    MulAddDst(saved[lane], inverse[k * width + lane], candidate[lane], static_cast<uint64_t>(chunk), n,
                              broadcast);
                PipeBarrier<PIPE_V>();
                for (int lane = 0; lane < group; lane += chunk)
                    Div(product[lane], saved[lane], safeDiagonal[lane], static_cast<uint64_t>(chunk), n, broadcast);
                PipeBarrier<PIPE_V>();
                Add(inverse[k * width], inverse[k * width], product, width);
                PipeBarrier<PIPE_V>();
                for (int row = 0; row < k; ++row)
                {
                    for (int lane = 0; lane < group; lane += chunk)
                        MulAddDst(inverse[row * width + lane], inverse[k * width + lane],
                                  factors[(row * n + k) * group + lane], static_cast<uint64_t>(chunk), n, broadcast);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
#ifdef LU950_FUSED_BATCH
        if (raw && n >= 3 && n <= 5 && fused)
            BatchRefineRawInverse(n, inverse, factors, stage, indices, group, padded, dense);
#endif
        if (raw && prepared && n >= 18)
            CompareScalar(mask, localInfo, 0, CMPMODE::EQ, group);
        else
        {
            Cast(localInfo, infoFloat, RoundMode::CAST_RINT, group);
            CompareScalar(mask, infoFloat, 0.f, CMPMODE::EQ, group);
        }
        BatchFence<HardEvent::V_S>();
        const bool outputAll = dense && BatchMaskAll(mask, count);
        const int outputStride = dense || compactStage ? n : padded;
        const int outputSlot = (n * outputStride + 7) / 8 * 8;
        // Fused raw packing constructs its own lane indices and never reads this fallback table.
        if (!(raw && fused && n >= 18))
        {
            for (int index = 0; index < outputSlot; ++index)
            {
                const int col = index / outputStride, row = index % outputStride;
                outputOffsets.SetValue(
                    index, row < n && col < n ? ((fused ? col * n + row : row * n + col) * group) * sizeof(float) : 0);
            }
            BatchFence<HardEvent::S_V>();
        }
        bool packed = false;
#ifdef LU950_FUSED_BATCH
        if (fused)
            packed = BatchFusedPack(n, packedStage, inverse, cachedRawSeeds ? outputOffsets[64] : outputOffsets, group,
                                    outputSlot, outputStride, cachedRawSeeds);
        PipeBarrier<PIPE_V>();
#endif
        if (!packed)
            for (int lane = 0; lane < count; ++lane)
            {
                Gather(packedStage[lane * outputSlot], inverse, outputOffsets, lane * sizeof(float), outputSlot);
                PipeBarrier<PIPE_V>();
            }
        BatchFence<HardEvent::V_MTE3>();
        if (outputAll)
        {
            GlobalTensor<float> destination;
            destination.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(firstC));
            DataCopyPad(
                destination, packedStage,
                DataCopyExtParams{static_cast<uint16_t>(count), static_cast<uint32_t>(n * n * sizeof(float)), 0, 0, 0});
        }
        else
            for (int lane = 0; lane < count; ++lane)
            {
                if (localInfo.GetValue(lane) != 0) continue;
                GlobalTensor<float> destination;
                destination.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cp.GetValue(lane)));
                if (dense || compactStage)
                    DataCopyPad(destination, packedStage[lane * outputSlot],
                                DataCopyExtParams{1, static_cast<uint32_t>(n * n * sizeof(float)), 0, 0, 0});
                else
                    DataCopyPad(destination, packedStage[lane * outputSlot],
                                DataCopyExtParams{static_cast<uint16_t>(n), static_cast<uint32_t>(n * sizeof(float)), 0,
                                                  static_cast<uint32_t>((ldc - n) * sizeof(float)), 0});
            }
        BatchFence<HardEvent::V_MTE3>();
        DataCopyPad(globalInfo[start], localInfo,
                    DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(int)), 0, 0, 0});
        BatchFence<HardEvent::MTE3_MTE2>();
        BatchFence<HardEvent::MTE3_V>();
        BatchFence<HardEvent::MTE3_S>();
    }
}
