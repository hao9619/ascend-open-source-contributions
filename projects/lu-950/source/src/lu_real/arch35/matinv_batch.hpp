/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Included inside the experimental kernel namespace after Fence().
// One SIMD lane carries one independent matrix with n<=8.
__aicore__ void BatchMatinvSmall(int n, GM_ADDR a, int lda, GM_ADDR c, int ldc, GM_ADDR info, int batch)
{
    constexpr int group = 32;
    constexpr int width = 16 * group;
    TPipe pipe;
    TBuf<TPosition::VECIN> inputBuffer, pointerBuffer;
    TBuf<TPosition::VECOUT> outputBuffer;
    TBuf<TPosition::VECOUT> infoBuffer;
    TBuf<TPosition::VECCALC> augBuffer, productBuffer, scratchBuffer, offsetBuffer, outputOffsetBuffer;
    TBuf<TPosition::VECCALC> temporaryBuffer, maskBuffer;
    pipe.InitBuffer(inputBuffer, group * 64 * sizeof(float));
    pipe.InitBuffer(pointerBuffer, 4 * group * sizeof(uint32_t));
    pipe.InitBuffer(outputBuffer, group * 64 * sizeof(float));
    pipe.InitBuffer(infoBuffer, group * sizeof(int));
    pipe.InitBuffer(augBuffer, 8 * width * sizeof(float));
    pipe.InitBuffer(productBuffer, 8 * width * sizeof(float));
    pipe.InitBuffer(scratchBuffer, width * sizeof(float));
    pipe.InitBuffer(offsetBuffer, 64 * group * sizeof(uint32_t));
    pipe.InitBuffer(outputOffsetBuffer, 64 * sizeof(uint32_t));
    pipe.InitBuffer(temporaryBuffer, 12 * group * sizeof(float));
    pipe.InitBuffer(maskBuffer, 64);
    auto input = inputBuffer.Get<float>();
    auto output = outputBuffer.Get<float>();
    auto localInfo = infoBuffer.Get<int>();
    auto aug = augBuffer.Get<float>();
    auto products = productBuffer.Get<float>();
    auto scratch = scratchBuffer.Get<float>();
    auto offsets = offsetBuffer.Get<uint32_t>();
    auto outputOffsets = outputOffsetBuffer.Get<uint32_t>();
    auto temp = temporaryBuffer.Get<float>();
    auto lanes = temp.ReinterpretCast<int32_t>();
    auto absPivot = temp[group];
    auto absCandidate = temp[2 * group];
    auto ones = temp[3 * group];
    auto reciprocal = temp[4 * group];
    auto safePivot = temp[5 * group];
    auto infoFloat = temp[6 * group];
    auto candidateInfo = temp[7 * group];
    auto nextInfo = temp[8 * group];
    auto selectMask = maskBuffer.Get<uint8_t>();
    auto infoMask = selectMask[32];
    for (int lane = 0; lane < group; ++lane) lanes.SetValue(lane, lane * 64 * sizeof(float));
    for (int col = 0; col < 8; ++col)
    {
        for (int row = 0; row < 8; ++row)
        {
            outputOffsets.SetValue(col * 8 + row, (row * width + (8 + col) * group) * sizeof(float));
        }
    }
    Fence<HardEvent::S_V>();
    for (int row = 0; row < n; ++row)
    {
        for (int col = 0; col < 8; ++col)
            Adds(offsets.ReinterpretCast<int32_t>()[(row * 8 + col) * group], lanes,
                 static_cast<int32_t>((col * 8 + row) * sizeof(float)), group);
    }
    Duplicate(ones, 1.0f, group);
    PipeBarrier<PIPE_V>();
    GlobalTensor<uint32_t> aPointers, cPointers;
    GlobalTensor<int> globalInfo;
    aPointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(a));
    cPointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(c));
    globalInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    const BinaryRepeatParams rows{1, 1, 1, group / 8, group / 8, group / 8};
    const BinaryRepeatParams broadcast{1, 1, 1, group / 8, group / 8, 0};
    for (int start = GetBlockIdx() * group; start < batch; start += GetBlockNum() * group)
    {
        const int count = batch - start < group ? batch - start : group;
        auto rawA = pointerBuffer.Get<uint32_t>(), rawC = rawA[2 * group];
        const DataCopyExtParams pointerCopy{1, static_cast<uint32_t>(count * sizeof(uint64_t)), 0, 0, 0};
        const DataCopyPadExtParams<uint32_t> pointerPad{true, 0, static_cast<uint8_t>((8 - count * 2 % 8) % 8), 0};
        DataCopyPad(rawA, aPointers[start * 2], pointerCopy, pointerPad);
        DataCopyPad(rawC, cPointers[start * 2], pointerCopy, pointerPad);
        Fence<HardEvent::MTE2_S>();
        auto pointerA = rawA.ReinterpretCast<uint64_t>(), pointerC = rawC.ReinterpretCast<uint64_t>();
        uint64_t outputs[group];
        int errors[group];
        Duplicate(input, 0.0f, group * 64);
        Fence<HardEvent::V_MTE2>();
        for (int lane = 0; lane < count; ++lane)
        {
            const uint64_t ap = pointerA.GetValue(lane);
            const uint64_t cp = pointerC.GetValue(lane);
            outputs[lane] = cp;
            const uint64_t aBytes = (static_cast<uint64_t>(n - 1) * lda + n) * sizeof(float);
            const uint64_t cBytes = (static_cast<uint64_t>(n - 1) * ldc + n) * sizeof(float);
            errors[lane] = ap == 0 ? -2 : (cp == 0 || (ap < cp + cBytes && cp < ap + aBytes) ? -4 : 0);
            if (errors[lane] == 0)
            {
                GlobalTensor<float> source;
                source.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ap));
                DataCopyPad(input[lane * 64], source,
                            DataCopyExtParams{static_cast<uint16_t>(n), static_cast<uint32_t>(n * sizeof(float)),
                                              static_cast<uint32_t>((lda - n) * sizeof(float)), 0, 0},
                            DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(8 - n), 0.0f});
            }
        }
        Fence<HardEvent::MTE2_V>();
        Duplicate(aug, 0.0f, 8 * width);
        Duplicate(infoFloat, 0.0f, group);
        PipeBarrier<PIPE_V>();
        for (int row = 0; row < n; ++row)
        {
            Gather(aug[row * width], input, offsets[row * 8 * group], 0, 8 * group);
            Duplicate(aug[row * width + (8 + row) * group], 1.0f, group);
        }
        PipeBarrier<PIPE_V>();
        for (int k = 0; k < n; ++k)
        {
            // Conditional row exchanges find the largest pivot independently per lane.
            for (int row = k + 1; row < n; ++row)
            {
                Abs(absPivot, aug[k * width + k * group], group);
                Abs(absCandidate, aug[row * width + k * group], group);
                PipeBarrier<PIPE_V>();
                Compare(selectMask, absCandidate, absPivot, CMPMODE::GT, group);
                PipeBarrier<PIPE_V>();
                Select(scratch, selectMask, aug[k * width], aug[row * width], SELMODE::VSEL_CMPMASK_SPR,
                       static_cast<uint64_t>(group), 16, rows);
                Select(aug[k * width], selectMask, aug[row * width], aug[k * width], SELMODE::VSEL_CMPMASK_SPR,
                       static_cast<uint64_t>(group), 16, rows);
                PipeBarrier<PIPE_V>();
                Muls(aug[row * width], scratch, 1.0f, width);
                PipeBarrier<PIPE_V>();
            }
            CompareScalar(selectMask, aug[k * width + k * group], 0.0f, CMPMODE::EQ, group);
            CompareScalar(infoMask, infoFloat, 0.0f, CMPMODE::EQ, group);
            Duplicate(candidateInfo, static_cast<float>(k + 1), group);
            PipeBarrier<PIPE_V>();
            Select(nextInfo, selectMask, candidateInfo, infoFloat, SELMODE::VSEL_CMPMASK_SPR, group);
            Select(safePivot, selectMask, ones, aug[k * width + k * group], SELMODE::VSEL_CMPMASK_SPR, group);
            PipeBarrier<PIPE_V>();
            Select(infoFloat, infoMask, nextInfo, infoFloat, SELMODE::VSEL_CMPMASK_SPR, group);
            Div(reciprocal, ones, safePivot, group);
            PipeBarrier<PIPE_V>();
            Mul(aug[k * width], aug[k * width], reciprocal, static_cast<uint64_t>(group), 16, broadcast);
            Duplicate(products[k * width], 0.0f, width);
            PipeBarrier<PIPE_V>();
            for (int row = 0; row < n; ++row)
            {
                if (row != k)
                    Mul(products[row * width], aug[k * width], aug[row * width + k * group],
                        static_cast<uint64_t>(group), 16, broadcast);
            }
            PipeBarrier<PIPE_V>();
            Sub(aug, aug, products, n * width);
            PipeBarrier<PIPE_V>();
        }
        for (int lane = 0; lane < count; ++lane)
            Gather(output[lane * 64], aug, outputOffsets, lane * sizeof(float), 64);
        Fence<HardEvent::V_S>();
        Fence<HardEvent::V_MTE3>();
        for (int lane = 0; lane < count; ++lane)
        {
            const int result = errors[lane] != 0 ? errors[lane] : static_cast<int>(infoFloat.GetValue(lane));
            localInfo.SetValue(lane, result);
            if (result == 0)
            {
                GlobalTensor<float> destination;
                destination.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputs[lane]));
                DataCopyPad(destination, output[lane * 64],
                            DataCopyExtParams{static_cast<uint16_t>(n), static_cast<uint32_t>(n * sizeof(float)), 0,
                                              static_cast<uint32_t>((ldc - n) * sizeof(float)), 0});
            }
        }
        Fence<HardEvent::S_MTE3>();
        DataCopyPad(globalInfo[start], localInfo,
                    DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(int)), 0, 0, 0});
        Fence<HardEvent::MTE3_S>();
        Fence<HardEvent::MTE3_MTE2>();
        Fence<HardEvent::MTE3_V>();
    }
}
