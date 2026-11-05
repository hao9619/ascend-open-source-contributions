/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batch_getri_n16_tiled.hpp"
#include "batch_getri_n8_packed.hpp"

template <int N>
__aicore__ inline void GetriPackedDispatch(__ubuf__ float* values, __ubuf__ int32_t* pivots, __ubuf__ int32_t* infos,
                                           __ubuf__ uint32_t* seed, int count, bool pivoting, __ubuf__ float* tile)
{
    static_assert(N == 8 || N == 16);
    if constexpr (N == 8)
        GetriN8PackedVF(values, pivots, infos, seed, count, pivoting);
    else
        GetriN16TiledVF(values, pivots, infos, seed, count, pivoting, tile);
}

template <int N, int group>
__aicore__ void BatchGetriRegisterDense(GM_ADDR a, GM_ADDR pivots, GM_ADDR c, GM_ADDR info, int batch)
{
    TPipe pipe;
    TBuf<TPosition::VECIN> pointerBuffer, valueBuffer, pivotBuffer;
    TBuf<TPosition::VECCALC> indexBuffer, maskBuffer, seedBuffer, tileBuffer;
    TBuf<TPosition::VECOUT> infoBuffer;
    pipe.InitBuffer(pointerBuffer, 4 * group * sizeof(uint32_t));
    pipe.InitBuffer(valueBuffer, N * N * group * sizeof(float));
    pipe.InitBuffer(pivotBuffer, N * group * sizeof(int));
    pipe.InitBuffer(indexBuffer, 4 * group * sizeof(uint32_t));
    pipe.InitBuffer(maskBuffer, (group + 255) / 256 * 32);
    pipe.InitBuffer(seedBuffer, 128 * sizeof(uint32_t));
    pipe.InitBuffer(infoBuffer, group * sizeof(int));
    if constexpr (N == 16) pipe.InitBuffer(tileBuffer, 16 * 64 * sizeof(float));
    auto rawA = pointerBuffer.Get<uint32_t>(), rawC = rawA[2 * group];
    auto values = valueBuffer.Get<float>();
    auto localPivots = pivotBuffer.Get<int>(), localInfo = infoBuffer.Get<int>();
    auto indices = indexBuffer.Get<int32_t>(), expected = indices[2 * group], actual = indices[3 * group];
    auto offsets = indices[group].ReinterpretCast<uint32_t>();
    auto seed = seedBuffer.Get<uint32_t>();
    auto mask = maskBuffer.Get<uint8_t>();
    __ubuf__ float* tile = nullptr;
    if constexpr (N == 16) tile = reinterpret_cast<__ubuf__ float*>(tileBuffer.Get<float>().GetPhyAddr());
    for (int lane = 0; lane < group; ++lane)
    {
        indices.SetValue(lane, lane);
        offsets.SetValue(lane, lane * sizeof(uint64_t));
        if (lane == 0)
            for (int index = 0; index < 64; ++index)
            {
                seed.SetValue(index, index % (64 / N));
                seed.SetValue(index + 64, index / (64 / N));
            }
    }
    BatchFence<HardEvent::S_V>();
    GlobalTensor<uint32_t> aPointers, cPointers;
    GlobalTensor<int> globalPivots, globalInfo;
    aPointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a));
    cPointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(c));
    globalPivots.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(pivots));
    globalInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(info));
    for (int start = GetBlockIdx() * group; start < batch; start += GetBlockNum() * group)
    {
        const int count = batch - start < group ? batch - start : group;
        const DataCopyExtParams copyPointers{1, static_cast<uint32_t>(count * sizeof(uint64_t)), 0, 0, 0};
        const DataCopyPadExtParams<uint32_t> pointerPad{true, 0, static_cast<uint8_t>((8 - count * 2 % 8) % 8), 0};
        DataCopyPad(rawA, aPointers[start * 2], copyPointers, pointerPad);
        DataCopyPad(rawC, cPointers[start * 2], copyPointers, pointerPad);
        if (pivots != nullptr)
            DataCopyPad(localPivots, globalPivots[start * N],
                        DataCopyExtParams{1, static_cast<uint32_t>(count * N * sizeof(int)), 0, 0, 0},
                        DataCopyPadExtParams<int>{true, 0, static_cast<uint8_t>((8 - count * N * N % 8) % 8), 0});
        BatchFence<HardEvent::MTE2_S>();
        BatchFence<HardEvent::MTE2_V>();
        auto ap = rawA.ReinterpretCast<uint64_t>(), cp = rawC.ReinterpretCast<uint64_t>();
        const uint64_t firstA = ap.GetValue(0), firstC = cp.GetValue(0),
                       bytes = static_cast<uint64_t>(count) * N * N * sizeof(float);
        const bool dense =
            !(firstA < firstC + bytes && firstC < firstA + bytes) &&
            BatchDensePointers(rawA, indices, offsets, expected, actual, mask, group, count, N * N * sizeof(float)) &&
            BatchDensePointers(rawC, indices, offsets, expected, actual, mask, group, count, N * N * sizeof(float));
        if (dense)
        {
            GlobalTensor<float> source, destination;
            source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(firstA));
            destination.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(firstC));
            DataCopyPad(values, source, DataCopyExtParams{1, static_cast<uint32_t>(bytes), 0, 0, 0},
                        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - count * N * N % 8) % 8), 0.f});
            BatchFence<HardEvent::MTE2_V>();
            GetriPackedDispatch<N>(reinterpret_cast<__ubuf__ float*>(values.GetPhyAddr()),
                                   reinterpret_cast<__ubuf__ int32_t*>(localPivots.GetPhyAddr()),
                                   reinterpret_cast<__ubuf__ int32_t*>(localInfo.GetPhyAddr()),
                                   reinterpret_cast<__ubuf__ uint32_t*>(seed.GetPhyAddr()), count, pivots != nullptr,
                                   tile);
            PipeBarrier<PIPE_V>();
            CompareScalar(mask, localInfo, 0, CMPMODE::EQ, group);
            BatchFence<HardEvent::V_S>();
            if (BatchMaskAll(mask, count))
            {
                BatchFence<HardEvent::V_MTE3>();
                DataCopyPad(destination, values, DataCopyExtParams{1, static_cast<uint32_t>(bytes), 0, 0, 0});
            }
            else
                for (int lane = 0; lane < count; ++lane)
                {
                    if (localInfo.GetValue(lane) != 0) continue;
                    auto scratch = expected.ReinterpretCast<float>();
                    for (int entry = 0; entry < N * N; ++entry)
                        scratch.SetValue(entry, values.GetValue(lane * N * N + entry));
                    BatchFence<HardEvent::S_MTE3>();
                    DataCopyPad(destination[lane * N * N], scratch,
                                DataCopyExtParams{1, N * N * sizeof(float), 0, 0, 0});
                    BatchFence<HardEvent::MTE3_S>();
                }
        }
        else
            for (int lane = 0; lane < count; ++lane)
            {
                const uint64_t av = ap.GetValue(lane), cv = cp.GetValue(lane);
                int code =
                    av == 0
                        ? -2
                        : (cv == 0 || (av < cv + N * N * sizeof(float) && cv < av + N * N * sizeof(float)) ? -5 : 0);
                if (code == 0)
                {
                    GlobalTensor<float> source, destination;
                    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(av));
                    destination.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cv));
                    for (int k = 0; k < N; ++k)
                        localPivots.SetValue(k, pivots == nullptr ? k + 1 : localPivots.GetValue(lane * N + k));
                    DataCopyPad(values, source, DataCopyExtParams{1, N * N * sizeof(float), 0, 0, 0},
                                DataCopyPadExtParams<float>{true, 0, 0, 0.f});
                    BatchFence<HardEvent::MTE2_V>();
                    BatchFence<HardEvent::S_V>();
                    GetriPackedDispatch<N>(reinterpret_cast<__ubuf__ float*>(values.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ int32_t*>(localPivots.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ int32_t*>(actual.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ uint32_t*>(seed.GetPhyAddr()), 1,
                                           pivots != nullptr, tile);
                    BatchFence<HardEvent::V_S>();
                    code = actual.GetValue(0);
                    if (code == 0)
                    {
                        BatchFence<HardEvent::V_MTE3>();
                        DataCopyPad(destination, values, DataCopyExtParams{1, N * N * sizeof(float), 0, 0, 0});
                    }
                    BatchFence<HardEvent::MTE3_V>();
                    BatchFence<HardEvent::MTE3_MTE2>();
                }
                localInfo.SetValue(lane, code);
            }
        BatchFence<HardEvent::S_MTE3>();
        BatchFence<HardEvent::V_MTE3>();
        DataCopyPad(globalInfo[start], localInfo,
                    DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(int)), 0, 0, 0});
        BatchFence<HardEvent::MTE3_MTE2>();
        BatchFence<HardEvent::MTE3_V>();
        BatchFence<HardEvent::MTE3_S>();
    }
}
