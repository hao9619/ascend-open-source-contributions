/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

__simd_callee__ inline void N2Reciprocal(Reg::RegTensor<float>& result, Reg::RegTensor<float>& diagonal,
                                         Reg::RegTensor<float>& ones, Reg::RegTensor<float>& negative,
                                         Reg::RegTensor<float>& residual, Reg::RegTensor<float>& correction,
                                         Reg::MaskReg& mask)
{
    Reg::Div(result, ones, diagonal, mask);
    Reg::Muls(negative, diagonal, -1.f, mask);
    Reg::Muls(residual, ones, 1.f, mask);
    Reg::MulAddDst(residual, result, negative, mask);
    Reg::Div(correction, residual, diagonal, mask);
    Reg::Add(result, result, correction, mask);
}

__simd_vf__ inline void GetriN2VF(__ubuf__ float* values, __ubuf__ int32_t* pivots, __ubuf__ int32_t* infos,
                                  __ubuf__ uint32_t* seed, int count, bool pivoting, bool raw)
{
    Reg::RegTensor<uint32_t> lanes, index, shifted, pivotIndex;
    Reg::RegTensor<float> d0, d1, lower, upper, r0, r1, x01, x10, x00;
    Reg::RegTensor<float> ones, negative, residual, correction, saved;
    Reg::RegTensor<int32_t> code, zero, one, two, invalid, p0, p1;
    Reg::MaskReg condition, swapped;
    uint32_t remaining = count;
    auto full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::LoadAlign(lanes, seed);
    Reg::Muls(index, lanes, uint32_t{4}, full);
    Reg::Muls(pivotIndex, lanes, uint32_t{2}, full);
    for (int start = 0; start < count; start += 64)
    {
        auto mask = Reg::UpdateMask<float>(remaining);
        auto* data = values + start * 4;
        Reg::Gather(d0, data, index, mask);
        Reg::Adds(shifted, index, uint32_t{1}, mask);
        Reg::Gather(lower, data, shifted, mask);
        Reg::Adds(shifted, index, uint32_t{2}, mask);
        Reg::Gather(upper, data, shifted, mask);
        Reg::Adds(shifted, index, uint32_t{3}, mask);
        Reg::Gather(d1, data, shifted, mask);
        Reg::Duplicate(ones, 1.f, mask);
        if (raw)
        {
            Reg::Abs(r0, d0, mask);
            Reg::Abs(r1, lower, mask);
            Reg::Compare<float, CMPMODE::GT>(swapped, r1, r0, mask);
            Reg::Muls(saved, d0, 1.f, mask);
            Reg::Select(d0, lower, d0, swapped);
            Reg::Select(lower, saved, lower, swapped);
            Reg::Muls(saved, upper, 1.f, mask);
            Reg::Select(upper, d1, upper, swapped);
            Reg::Select(d1, saved, d1, swapped);
            N2Reciprocal(r0, d0, ones, negative, residual, correction, mask);
            Reg::Mul(lower, lower, r0, mask);
            // Match the actual FP32 LAPACK factorization order: round the
            // product before subtraction. FMA changes near-singular pivots.
            Reg::Mul(saved, lower, upper, mask);
            Reg::Sub(d1, d1, saved, mask);
        }
        Reg::Duplicate(zero, 0, mask);
        Reg::Duplicate(one, 1, mask);
        Reg::Duplicate(two, 2, mask);
        Reg::Duplicate(invalid, -4, mask);
        Reg::Compares<float, CMPMODE::EQ>(condition, d1, 0.f, mask);
        Reg::Select(code, two, zero, condition);
        Reg::Compares<float, CMPMODE::EQ>(condition, d0, 0.f, mask);
        Reg::Select(code, one, code, condition);
        N2Reciprocal(r0, d0, ones, negative, residual, correction, mask);
        N2Reciprocal(r1, d1, ones, negative, residual, correction, mask);
        Reg::Muls(upper, upper, -1.f, mask);
        Reg::Muls(lower, lower, -1.f, mask);
        Reg::Mul(x01, r0, upper, mask);
        Reg::Mul(x01, x01, r1, mask);
        Reg::Mul(x10, r1, lower, mask);
        Reg::Mul(x00, x01, lower, mask);
        Reg::Add(x00, r0, x00, mask);
        if (pivoting)
        {
            Reg::Gather(p0, pivots + start * 2, pivotIndex, mask);
            Reg::Adds(shifted, pivotIndex, uint32_t{1}, mask);
            Reg::Gather(p1, pivots + start * 2, shifted, mask);
            Reg::Compares<int32_t, CMPMODE::LT>(condition, p0, 1, mask);
            Reg::Select(code, invalid, code, condition);
            Reg::Compares<int32_t, CMPMODE::GT>(condition, p0, 2, mask);
            Reg::Select(code, invalid, code, condition);
            Reg::Compares<int32_t, CMPMODE::NE>(condition, p1, 2, mask);
            Reg::Select(code, invalid, code, condition);
            Reg::Compares<int32_t, CMPMODE::EQ>(swapped, p0, 2, mask);
        }
        if (pivoting || raw)
        {
            Reg::Muls(saved, x00, 1.f, mask);
            Reg::Select(x00, x01, x00, swapped);
            Reg::Select(x01, saved, x01, swapped);
            Reg::Muls(saved, x10, 1.f, mask);
            Reg::Select(x10, r1, x10, swapped);
            Reg::Select(r1, saved, r1, swapped);
        }
        Reg::Scatter(data, x00, index, mask);
        Reg::Adds(shifted, index, uint32_t{1}, mask);
        Reg::Scatter(data, x10, shifted, mask);
        Reg::Adds(shifted, index, uint32_t{2}, mask);
        Reg::Scatter(data, x01, shifted, mask);
        Reg::Adds(shifted, index, uint32_t{3}, mask);
        Reg::Scatter(data, r1, shifted, mask);
        Reg::StoreAlign(infos + start, code, mask);
    }
}

template <int group = 4096>
__aicore__ void BatchGetriN2(GM_ADDR a, GM_ADDR pivots, GM_ADDR c, GM_ADDR info, int batch, bool raw = false)
{
    TPipe pipe;
    TBuf<TPosition::VECIN> pointerBuffer, valueBuffer, pivotBuffer;
    TBuf<TPosition::VECCALC> indexBuffer, maskBuffer, seedBuffer;
    TBuf<TPosition::VECOUT> infoBuffer;
    pipe.InitBuffer(pointerBuffer, 4 * group * sizeof(uint32_t));
    pipe.InitBuffer(valueBuffer, 4 * group * sizeof(float));
    pipe.InitBuffer(pivotBuffer, 2 * group * sizeof(int));
    pipe.InitBuffer(indexBuffer, 4 * group * sizeof(uint32_t));
    pipe.InitBuffer(maskBuffer, group / 8);
    pipe.InitBuffer(seedBuffer, 64 * sizeof(uint32_t));
    pipe.InitBuffer(infoBuffer, group * sizeof(int));
    auto rawA = pointerBuffer.Get<uint32_t>(), rawC = rawA[2 * group];
    auto values = valueBuffer.Get<float>();
    auto localPivots = pivotBuffer.Get<int>(), localInfo = infoBuffer.Get<int>();
    auto indices = indexBuffer.Get<int32_t>(), expected = indices[2 * group], actual = indices[3 * group];
    auto offsets = indices[group].ReinterpretCast<uint32_t>();
    auto seed = seedBuffer.Get<uint32_t>();
    auto mask = maskBuffer.Get<uint8_t>();
    for (int lane = 0; lane < group; ++lane)
    {
        indices.SetValue(lane, lane);
        offsets.SetValue(lane, lane * sizeof(uint64_t));
        if (lane < 64) seed.SetValue(lane, lane);
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
            DataCopyPad(localPivots, globalPivots[start * 2],
                        DataCopyExtParams{1, static_cast<uint32_t>(count * 2 * sizeof(int)), 0, 0, 0},
                        DataCopyPadExtParams<int>{true, 0, static_cast<uint8_t>((8 - count * 2 % 8) % 8), 0});
        BatchFence<HardEvent::MTE2_S>();
        BatchFence<HardEvent::MTE2_V>();
        auto ap = rawA.ReinterpretCast<uint64_t>(), cp = rawC.ReinterpretCast<uint64_t>();
        const uint64_t firstA = ap.GetValue(0), firstC = cp.GetValue(0), bytes = static_cast<uint64_t>(count) * 16;
        const bool dense = !(firstA < firstC + bytes && firstC < firstA + bytes) &&
                           BatchDensePointers(rawA, indices, offsets, expected, actual, mask, group, count, 16) &&
                           BatchDensePointers(rawC, indices, offsets, expected, actual, mask, group, count, 16);
        if (dense)
        {
            GlobalTensor<float> source, destination;
            source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(firstA));
            destination.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(firstC));
            DataCopyPad(values, source, DataCopyExtParams{1, static_cast<uint32_t>(bytes), 0, 0, 0},
                        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - count * 4 % 8) % 8), 0.f});
            BatchFence<HardEvent::MTE2_V>();
            GetriN2VF(reinterpret_cast<__ubuf__ float*>(values.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ int32_t*>(localPivots.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ int32_t*>(localInfo.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t*>(seed.GetPhyAddr()), count, pivots != nullptr, raw);
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
                    for (int entry = 0; entry < 4; ++entry) scratch.SetValue(entry, values.GetValue(lane * 4 + entry));
                    BatchFence<HardEvent::S_MTE3>();
                    DataCopyPad(destination[lane * 4], scratch, DataCopyExtParams{1, 16, 0, 0, 0});
                    BatchFence<HardEvent::MTE3_S>();
                }
        }
        else
            for (int lane = 0; lane < count; ++lane)
            {
                const uint64_t av = ap.GetValue(lane), cv = cp.GetValue(lane);
                int code = av == 0 ? -2 : (cv == 0 || (av < cv + 16 && cv < av + 16) ? (raw ? -4 : -5) : 0);
                if (code == 0)
                {
                    GlobalTensor<float> source, destination;
                    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(av));
                    destination.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cv));
                    const int p0 = pivots == nullptr ? 1 : localPivots.GetValue(lane * 2);
                    const int p1 = pivots == nullptr ? 2 : localPivots.GetValue(lane * 2 + 1);
                    localPivots.SetValue(0, p0);
                    localPivots.SetValue(1, p1);
                    DataCopyPad(values, source, DataCopyExtParams{1, 16, 0, 0, 0},
                                DataCopyPadExtParams<float>{true, 0, 4, 0.f});
                    BatchFence<HardEvent::MTE2_V>();
                    BatchFence<HardEvent::S_V>();
                    GetriN2VF(reinterpret_cast<__ubuf__ float*>(values.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ int32_t*>(localPivots.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ int32_t*>(actual.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ uint32_t*>(seed.GetPhyAddr()), 1, pivots != nullptr, raw);
                    BatchFence<HardEvent::V_S>();
                    code = actual.GetValue(0);
                    if (code == 0)
                    {
                        BatchFence<HardEvent::V_MTE3>();
                        DataCopyPad(destination, values, DataCopyExtParams{1, 16, 0, 0, 0});
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
