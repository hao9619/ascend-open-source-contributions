/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Four-by-four LAPACK inversion with all matrix entries kept in vector registers.
__simd_vf__ inline void GetriN4DenseVF(__ubuf__ float* values, __ubuf__ int32_t* pivots, __ubuf__ int32_t* infos,
                                       __ubuf__ uint32_t* seed, int count, bool pivoting)
{
    uint32_t remaining = count;
    for (int start = 0; start < count; start += 64)
    {
        auto mask = Reg::UpdateMask<float>(remaining);
        auto* data = values + start * 16;
        Reg::RegTensor<float> ones, negative, residual, correction, reciprocal, sum;
        Reg::RegTensor<float> t0, t1, t2, t3;
        Reg::RegTensor<uint32_t> lanes, index, pivot, base;
        Reg::RegTensor<int32_t> code, number, invalid;
        Reg::MaskReg bad;
        Reg::MaskReg swapped;
        Reg::RegTensor<float> a00, a01, a02, a03;
        Reg::RegTensor<float> a10, a11, a12, a13;
        Reg::RegTensor<float> a20, a21, a22, a23;
        Reg::RegTensor<float> a30, a31, a32, a33;
        Reg::LoadAlign(lanes, seed);
        Reg::Duplicate(ones, 1.f, mask);
        Reg::Duplicate(code, 0, mask);
        Reg::Duplicate(invalid, -4, mask);
        Reg::Muls(base, lanes, uint32_t{16}, mask);
        Reg::Adds(index, base, uint32_t{0}, mask);
        Reg::Gather(a00, data, index, mask);
        Reg::Adds(index, base, uint32_t{4}, mask);
        Reg::Gather(a01, data, index, mask);
        Reg::Muls(a01, a01, -1.f, mask);
        Reg::Adds(index, base, uint32_t{8}, mask);
        Reg::Gather(a02, data, index, mask);
        Reg::Muls(a02, a02, -1.f, mask);
        Reg::Adds(index, base, uint32_t{12}, mask);
        Reg::Gather(a03, data, index, mask);
        Reg::Muls(a03, a03, -1.f, mask);
        Reg::Adds(index, base, uint32_t{1}, mask);
        Reg::Gather(a10, data, index, mask);
        Reg::Muls(a10, a10, -1.f, mask);
        Reg::Adds(index, base, uint32_t{5}, mask);
        Reg::Gather(a11, data, index, mask);
        Reg::Adds(index, base, uint32_t{9}, mask);
        Reg::Gather(a12, data, index, mask);
        Reg::Muls(a12, a12, -1.f, mask);
        Reg::Adds(index, base, uint32_t{13}, mask);
        Reg::Gather(a13, data, index, mask);
        Reg::Muls(a13, a13, -1.f, mask);
        Reg::Adds(index, base, uint32_t{2}, mask);
        Reg::Gather(a20, data, index, mask);
        Reg::Muls(a20, a20, -1.f, mask);
        Reg::Adds(index, base, uint32_t{6}, mask);
        Reg::Gather(a21, data, index, mask);
        Reg::Muls(a21, a21, -1.f, mask);
        Reg::Adds(index, base, uint32_t{10}, mask);
        Reg::Gather(a22, data, index, mask);
        Reg::Adds(index, base, uint32_t{14}, mask);
        Reg::Gather(a23, data, index, mask);
        Reg::Muls(a23, a23, -1.f, mask);
        Reg::Adds(index, base, uint32_t{3}, mask);
        Reg::Gather(a30, data, index, mask);
        Reg::Muls(a30, a30, -1.f, mask);
        Reg::Adds(index, base, uint32_t{7}, mask);
        Reg::Gather(a31, data, index, mask);
        Reg::Muls(a31, a31, -1.f, mask);
        Reg::Adds(index, base, uint32_t{11}, mask);
        Reg::Gather(a32, data, index, mask);
        Reg::Muls(a32, a32, -1.f, mask);
        Reg::Adds(index, base, uint32_t{15}, mask);
        Reg::Gather(a33, data, index, mask);
        Reg::Compares<float, CMPMODE::EQ>(bad, a33, 0.f, mask);
        Reg::Duplicate(number, 4, mask);
        Reg::Select(code, number, code, bad);
        Reg::Compares<float, CMPMODE::EQ>(bad, a22, 0.f, mask);
        Reg::Duplicate(number, 3, mask);
        Reg::Select(code, number, code, bad);
        Reg::Compares<float, CMPMODE::EQ>(bad, a11, 0.f, mask);
        Reg::Duplicate(number, 2, mask);
        Reg::Select(code, number, code, bad);
        Reg::Compares<float, CMPMODE::EQ>(bad, a00, 0.f, mask);
        Reg::Duplicate(number, 1, mask);
        Reg::Select(code, number, code, bad);
        {
            Reg::Div(reciprocal, ones, a00, mask);
            Reg::Muls(negative, a00, -1.f, mask);
            Reg::Muls(residual, ones, 1.f, mask);
            Reg::MulAddDst(residual, reciprocal, negative, mask);
            Reg::Div(correction, residual, a00, mask);
            Reg::Add(reciprocal, reciprocal, correction, mask);
            Reg::Muls(a00, reciprocal, 1.f, mask);
        }
        {
            Reg::Div(reciprocal, ones, a11, mask);
            Reg::Muls(negative, a11, -1.f, mask);
            Reg::Muls(residual, ones, 1.f, mask);
            Reg::MulAddDst(residual, reciprocal, negative, mask);
            Reg::Div(correction, residual, a11, mask);
            Reg::Add(reciprocal, reciprocal, correction, mask);
            Reg::Duplicate(sum, 0.f, mask);
            Reg::MulAddDst(sum, a00, a01, mask);
            Reg::Mul(a01, sum, reciprocal, mask);
            Reg::Muls(a11, reciprocal, 1.f, mask);
        }
        {
            Reg::Div(reciprocal, ones, a22, mask);
            Reg::Muls(negative, a22, -1.f, mask);
            Reg::Muls(residual, ones, 1.f, mask);
            Reg::MulAddDst(residual, reciprocal, negative, mask);
            Reg::Div(correction, residual, a22, mask);
            Reg::Add(reciprocal, reciprocal, correction, mask);
            Reg::Duplicate(sum, 0.f, mask);
            Reg::MulAddDst(sum, a00, a02, mask);
            Reg::MulAddDst(sum, a01, a12, mask);
            Reg::Mul(a02, sum, reciprocal, mask);
            Reg::Duplicate(sum, 0.f, mask);
            Reg::MulAddDst(sum, a11, a12, mask);
            Reg::Mul(a12, sum, reciprocal, mask);
            Reg::Muls(a22, reciprocal, 1.f, mask);
        }
        {
            Reg::Div(reciprocal, ones, a33, mask);
            Reg::Muls(negative, a33, -1.f, mask);
            Reg::Muls(residual, ones, 1.f, mask);
            Reg::MulAddDst(residual, reciprocal, negative, mask);
            Reg::Div(correction, residual, a33, mask);
            Reg::Add(reciprocal, reciprocal, correction, mask);
            Reg::Duplicate(sum, 0.f, mask);
            Reg::MulAddDst(sum, a00, a03, mask);
            Reg::MulAddDst(sum, a01, a13, mask);
            Reg::MulAddDst(sum, a02, a23, mask);
            Reg::Mul(a03, sum, reciprocal, mask);
            Reg::Duplicate(sum, 0.f, mask);
            Reg::MulAddDst(sum, a11, a13, mask);
            Reg::MulAddDst(sum, a12, a23, mask);
            Reg::Mul(a13, sum, reciprocal, mask);
            Reg::Duplicate(sum, 0.f, mask);
            Reg::MulAddDst(sum, a22, a23, mask);
            Reg::Mul(a23, sum, reciprocal, mask);
            Reg::Muls(a33, reciprocal, 1.f, mask);
        }
        {
            // Remove column 3 of L; delay writes until every coefficient has been consumed.
            Reg::Duplicate(t0, 0.f, mask);
            Reg::Add(t0, a03, t0, mask);
            Reg::Duplicate(t1, 0.f, mask);
            Reg::Add(t1, a13, t1, mask);
            Reg::Duplicate(t2, 0.f, mask);
            Reg::Add(t2, a23, t2, mask);
            Reg::Duplicate(t3, 0.f, mask);
            Reg::Add(t3, a33, t3, mask);
            Reg::Muls(a03, t0, 1.f, mask);
            Reg::Muls(a13, t1, 1.f, mask);
            Reg::Muls(a23, t2, 1.f, mask);
            Reg::Muls(a33, t3, 1.f, mask);
        }
        {
            // Remove column 2 of L; delay writes until every coefficient has been consumed.
            Reg::Duplicate(t0, 0.f, mask);
            Reg::MulAddDst(t0, a03, a32, mask);
            Reg::Add(t0, a02, t0, mask);
            Reg::Duplicate(t1, 0.f, mask);
            Reg::MulAddDst(t1, a13, a32, mask);
            Reg::Add(t1, a12, t1, mask);
            Reg::Duplicate(t2, 0.f, mask);
            Reg::MulAddDst(t2, a23, a32, mask);
            Reg::Add(t2, a22, t2, mask);
            Reg::Duplicate(t3, 0.f, mask);
            Reg::MulAddDst(t3, a33, a32, mask);
            Reg::Muls(a02, t0, 1.f, mask);
            Reg::Muls(a12, t1, 1.f, mask);
            Reg::Muls(a22, t2, 1.f, mask);
            Reg::Muls(a32, t3, 1.f, mask);
        }
        {
            // Remove column 1 of L; delay writes until every coefficient has been consumed.
            Reg::Duplicate(t0, 0.f, mask);
            Reg::MulAddDst(t0, a02, a21, mask);
            Reg::MulAddDst(t0, a03, a31, mask);
            Reg::Add(t0, a01, t0, mask);
            Reg::Duplicate(t1, 0.f, mask);
            Reg::MulAddDst(t1, a12, a21, mask);
            Reg::MulAddDst(t1, a13, a31, mask);
            Reg::Add(t1, a11, t1, mask);
            Reg::Duplicate(t2, 0.f, mask);
            Reg::MulAddDst(t2, a22, a21, mask);
            Reg::MulAddDst(t2, a23, a31, mask);
            Reg::Duplicate(t3, 0.f, mask);
            Reg::MulAddDst(t3, a32, a21, mask);
            Reg::MulAddDst(t3, a33, a31, mask);
            Reg::Muls(a01, t0, 1.f, mask);
            Reg::Muls(a11, t1, 1.f, mask);
            Reg::Muls(a21, t2, 1.f, mask);
            Reg::Muls(a31, t3, 1.f, mask);
        }
        {
            // Remove column 0 of L; delay writes until every coefficient has been consumed.
            Reg::Duplicate(t0, 0.f, mask);
            Reg::MulAddDst(t0, a01, a10, mask);
            Reg::MulAddDst(t0, a02, a20, mask);
            Reg::MulAddDst(t0, a03, a30, mask);
            Reg::Add(t0, a00, t0, mask);
            Reg::Duplicate(t1, 0.f, mask);
            Reg::MulAddDst(t1, a11, a10, mask);
            Reg::MulAddDst(t1, a12, a20, mask);
            Reg::MulAddDst(t1, a13, a30, mask);
            Reg::Duplicate(t2, 0.f, mask);
            Reg::MulAddDst(t2, a21, a10, mask);
            Reg::MulAddDst(t2, a22, a20, mask);
            Reg::MulAddDst(t2, a23, a30, mask);
            Reg::Duplicate(t3, 0.f, mask);
            Reg::MulAddDst(t3, a31, a10, mask);
            Reg::MulAddDst(t3, a32, a20, mask);
            Reg::MulAddDst(t3, a33, a30, mask);
            Reg::Muls(a00, t0, 1.f, mask);
            Reg::Muls(a10, t1, 1.f, mask);
            Reg::Muls(a20, t2, 1.f, mask);
            Reg::Muls(a30, t3, 1.f, mask);
        }
        Reg::RegTensor<uint32_t> m0, m1, m2, m3, savedMap;
        Reg::Duplicate(m0, uint32_t{0}, mask);
        Reg::Duplicate(m1, uint32_t{4}, mask);
        Reg::Duplicate(m2, uint32_t{8}, mask);
        Reg::Duplicate(m3, uint32_t{12}, mask);
        if (pivoting)
        {
            Reg::Muls(index, lanes, uint32_t{4}, mask);
            Reg::Adds(index, index, uint32_t{0}, mask);
            Reg::Gather(pivot, reinterpret_cast<__ubuf__ uint32_t*>(pivots + start * 4), index, mask);
            Reg::Compares<uint32_t, CMPMODE::LT>(bad, pivot, uint32_t{1}, mask);
            Reg::Select(code, invalid, code, bad);
            Reg::Compares<uint32_t, CMPMODE::GT>(bad, pivot, uint32_t{4}, mask);
            Reg::Select(code, invalid, code, bad);
            Reg::Compares<uint32_t, CMPMODE::EQ>(swapped, pivot, uint32_t{2}, mask);
            Reg::Adds(savedMap, m0, uint32_t{0}, mask);
            Reg::Select(m0, m1, m0, swapped);
            Reg::Select(m1, savedMap, m1, swapped);
            Reg::Compares<uint32_t, CMPMODE::EQ>(swapped, pivot, uint32_t{3}, mask);
            Reg::Adds(savedMap, m0, uint32_t{0}, mask);
            Reg::Select(m0, m2, m0, swapped);
            Reg::Select(m2, savedMap, m2, swapped);
            Reg::Compares<uint32_t, CMPMODE::EQ>(swapped, pivot, uint32_t{4}, mask);
            Reg::Adds(savedMap, m0, uint32_t{0}, mask);
            Reg::Select(m0, m3, m0, swapped);
            Reg::Select(m3, savedMap, m3, swapped);
            Reg::Muls(index, lanes, uint32_t{4}, mask);
            Reg::Adds(index, index, uint32_t{1}, mask);
            Reg::Gather(pivot, reinterpret_cast<__ubuf__ uint32_t*>(pivots + start * 4), index, mask);
            Reg::Compares<uint32_t, CMPMODE::LT>(bad, pivot, uint32_t{2}, mask);
            Reg::Select(code, invalid, code, bad);
            Reg::Compares<uint32_t, CMPMODE::GT>(bad, pivot, uint32_t{4}, mask);
            Reg::Select(code, invalid, code, bad);
            Reg::Compares<uint32_t, CMPMODE::EQ>(swapped, pivot, uint32_t{3}, mask);
            Reg::Adds(savedMap, m1, uint32_t{0}, mask);
            Reg::Select(m1, m2, m1, swapped);
            Reg::Select(m2, savedMap, m2, swapped);
            Reg::Compares<uint32_t, CMPMODE::EQ>(swapped, pivot, uint32_t{4}, mask);
            Reg::Adds(savedMap, m1, uint32_t{0}, mask);
            Reg::Select(m1, m3, m1, swapped);
            Reg::Select(m3, savedMap, m3, swapped);
            Reg::Muls(index, lanes, uint32_t{4}, mask);
            Reg::Adds(index, index, uint32_t{2}, mask);
            Reg::Gather(pivot, reinterpret_cast<__ubuf__ uint32_t*>(pivots + start * 4), index, mask);
            Reg::Compares<uint32_t, CMPMODE::LT>(bad, pivot, uint32_t{3}, mask);
            Reg::Select(code, invalid, code, bad);
            Reg::Compares<uint32_t, CMPMODE::GT>(bad, pivot, uint32_t{4}, mask);
            Reg::Select(code, invalid, code, bad);
            Reg::Compares<uint32_t, CMPMODE::EQ>(swapped, pivot, uint32_t{4}, mask);
            Reg::Adds(savedMap, m2, uint32_t{0}, mask);
            Reg::Select(m2, m3, m2, swapped);
            Reg::Select(m3, savedMap, m3, swapped);
            Reg::Muls(index, lanes, uint32_t{4}, mask);
            Reg::Adds(index, index, uint32_t{3}, mask);
            Reg::Gather(pivot, reinterpret_cast<__ubuf__ uint32_t*>(pivots + start * 4), index, mask);
            Reg::Compares<uint32_t, CMPMODE::LT>(bad, pivot, uint32_t{4}, mask);
            Reg::Select(code, invalid, code, bad);
            Reg::Compares<uint32_t, CMPMODE::GT>(bad, pivot, uint32_t{4}, mask);
            Reg::Select(code, invalid, code, bad);
        }
        Reg::Add(index, base, m0, mask);
        Reg::Adds(index, index, uint32_t{0}, mask);
        Reg::Scatter(data, a00, index, mask);
        Reg::Add(index, base, m0, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(data, a10, index, mask);
        Reg::Add(index, base, m0, mask);
        Reg::Adds(index, index, uint32_t{2}, mask);
        Reg::Scatter(data, a20, index, mask);
        Reg::Add(index, base, m0, mask);
        Reg::Adds(index, index, uint32_t{3}, mask);
        Reg::Scatter(data, a30, index, mask);
        Reg::Add(index, base, m1, mask);
        Reg::Adds(index, index, uint32_t{0}, mask);
        Reg::Scatter(data, a01, index, mask);
        Reg::Add(index, base, m1, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(data, a11, index, mask);
        Reg::Add(index, base, m1, mask);
        Reg::Adds(index, index, uint32_t{2}, mask);
        Reg::Scatter(data, a21, index, mask);
        Reg::Add(index, base, m1, mask);
        Reg::Adds(index, index, uint32_t{3}, mask);
        Reg::Scatter(data, a31, index, mask);
        Reg::Add(index, base, m2, mask);
        Reg::Adds(index, index, uint32_t{0}, mask);
        Reg::Scatter(data, a02, index, mask);
        Reg::Add(index, base, m2, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(data, a12, index, mask);
        Reg::Add(index, base, m2, mask);
        Reg::Adds(index, index, uint32_t{2}, mask);
        Reg::Scatter(data, a22, index, mask);
        Reg::Add(index, base, m2, mask);
        Reg::Adds(index, index, uint32_t{3}, mask);
        Reg::Scatter(data, a32, index, mask);
        Reg::Add(index, base, m3, mask);
        Reg::Adds(index, index, uint32_t{0}, mask);
        Reg::Scatter(data, a03, index, mask);
        Reg::Add(index, base, m3, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Scatter(data, a13, index, mask);
        Reg::Add(index, base, m3, mask);
        Reg::Adds(index, index, uint32_t{2}, mask);
        Reg::Scatter(data, a23, index, mask);
        Reg::Add(index, base, m3, mask);
        Reg::Adds(index, index, uint32_t{3}, mask);
        Reg::Scatter(data, a33, index, mask);
        Reg::StoreAlign(infos + start, code, mask);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <int group = 2048>
__aicore__ void BatchGetriN4Dense(GM_ADDR a, GM_ADDR pivots, GM_ADDR c, GM_ADDR info, int batch)
{
    TPipe pipe;
    TBuf<TPosition::VECIN> pointerBuffer, valueBuffer, pivotBuffer;
    TBuf<TPosition::VECCALC> indexBuffer, maskBuffer, seedBuffer;
    TBuf<TPosition::VECOUT> infoBuffer;
    pipe.InitBuffer(pointerBuffer, 4 * group * sizeof(uint32_t));
    pipe.InitBuffer(valueBuffer, 16 * group * sizeof(float));
    pipe.InitBuffer(pivotBuffer, 4 * group * sizeof(int));
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
            DataCopyPad(localPivots, globalPivots[start * 4],
                        DataCopyExtParams{1, static_cast<uint32_t>(count * 4 * sizeof(int)), 0, 0, 0},
                        DataCopyPadExtParams<int>{true, 0, static_cast<uint8_t>((8 - count * 16 % 8) % 8), 0});
        BatchFence<HardEvent::MTE2_S>();
        BatchFence<HardEvent::MTE2_V>();
        auto ap = rawA.ReinterpretCast<uint64_t>(), cp = rawC.ReinterpretCast<uint64_t>();
        const uint64_t firstA = ap.GetValue(0), firstC = cp.GetValue(0), bytes = static_cast<uint64_t>(count) * 64;
        const bool dense = !(firstA < firstC + bytes && firstC < firstA + bytes) &&
                           BatchDensePointers(rawA, indices, offsets, expected, actual, mask, group, count, 64) &&
                           BatchDensePointers(rawC, indices, offsets, expected, actual, mask, group, count, 64);
        if (dense)
        {
            GlobalTensor<float> source, destination;
            source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(firstA));
            destination.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(firstC));
            DataCopyPad(values, source, DataCopyExtParams{1, static_cast<uint32_t>(bytes), 0, 0, 0},
                        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>((8 - count * 16 % 8) % 8), 0.f});
            BatchFence<HardEvent::MTE2_V>();
            GetriN4DenseVF(reinterpret_cast<__ubuf__ float*>(values.GetPhyAddr()),
                           reinterpret_cast<__ubuf__ int32_t*>(localPivots.GetPhyAddr()),
                           reinterpret_cast<__ubuf__ int32_t*>(localInfo.GetPhyAddr()),
                           reinterpret_cast<__ubuf__ uint32_t*>(seed.GetPhyAddr()), count, pivots != nullptr);
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
                    for (int entry = 0; entry < 16; ++entry)
                        scratch.SetValue(entry, values.GetValue(lane * 16 + entry));
                    BatchFence<HardEvent::S_MTE3>();
                    DataCopyPad(destination[lane * 16], scratch, DataCopyExtParams{1, 64, 0, 0, 0});
                    BatchFence<HardEvent::MTE3_S>();
                }
        }
        else
            for (int lane = 0; lane < count; ++lane)
            {
                const uint64_t av = ap.GetValue(lane), cv = cp.GetValue(lane);
                int code = av == 0 ? -2 : (cv == 0 || (av < cv + 64 && cv < av + 64) ? -5 : 0);
                if (code == 0)
                {
                    GlobalTensor<float> source, destination;
                    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(av));
                    destination.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cv));
                    for (int k = 0; k < 4; ++k)
                        localPivots.SetValue(k, pivots == nullptr ? k + 1 : localPivots.GetValue(lane * 4 + k));
                    DataCopyPad(values, source, DataCopyExtParams{1, 64, 0, 0, 0},
                                DataCopyPadExtParams<float>{true, 0, 0, 0.f});
                    BatchFence<HardEvent::MTE2_V>();
                    BatchFence<HardEvent::S_V>();
                    GetriN4DenseVF(reinterpret_cast<__ubuf__ float*>(values.GetPhyAddr()),
                                   reinterpret_cast<__ubuf__ int32_t*>(localPivots.GetPhyAddr()),
                                   reinterpret_cast<__ubuf__ int32_t*>(actual.GetPhyAddr()),
                                   reinterpret_cast<__ubuf__ uint32_t*>(seed.GetPhyAddr()), 1, pivots != nullptr);
                    BatchFence<HardEvent::V_S>();
                    code = actual.GetValue(0);
                    if (code == 0)
                    {
                        BatchFence<HardEvent::V_MTE3>();
                        DataCopyPad(destination, values, DataCopyExtParams{1, 64, 0, 0, 0});
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
