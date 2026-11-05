/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Row-vector triangular inversion for one medium matrix.
template <bool TwoRows>
__simd_vf__ inline void GetriMediumTiledVF(__ubuf__ float* factors, __ubuf__ int32_t* pivots, __ubuf__ int32_t* info,
                                           __ubuf__ uint32_t* seed, __ubuf__ float* tile, int n, bool pivoting)
{
    constexpr int stride = TwoRows ? 128 : 64;
    auto full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> rows, rowsHigh, index, pivot, fallback, base;
    Reg::RegTensor<int32_t> code, number, invalid;
    Reg::RegTensor<float> scratch, diagonal, reciprocal, negative, residual, correction, ones;
    Reg::RegTensor<float> sum, sumHigh, coefficient, value, valueHigh, other, otherHigh;
    Reg::MaskReg mask, maskHigh, zero, available, diagonalRow, diagonalHigh, bad;
    Reg::LoadAlign(rows, seed);
    Reg::Compares<uint32_t, CMPMODE::LT>(mask, rows, static_cast<uint32_t>(n), full);
    if constexpr (TwoRows)
    {
        Reg::Adds(rowsHigh, rows, uint32_t{64}, full);
        Reg::Compares<uint32_t, CMPMODE::LT>(maskHigh, rowsHigh, static_cast<uint32_t>(n), full);
    }
    for (int col = 0; col < n; ++col)
    {
        Reg::LoadAlign(scratch, factors + col * stride);
        Reg::Muls(scratch, scratch, -1.f, full);
        Reg::StoreAlign(factors + col * stride, scratch, full);
        if constexpr (TwoRows)
        {
            Reg::LoadAlign(scratch, factors + col * stride + 64);
            Reg::Muls(scratch, scratch, -1.f, full);
            Reg::StoreAlign(factors + col * stride + 64, scratch, full);
        }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
    Reg::Duplicate(ones, 1.f, mask);
    Reg::Duplicate(code, 0, mask);
    Reg::Duplicate(invalid, -4, mask);
    for (int col = 0; col < n; ++col)
    {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(diagonal, factors + col * stride + col);
        Reg::Muls(diagonal, diagonal, -1.f, mask);
        Reg::Compares<float, CMPMODE::EQ>(zero, diagonal, 0.f, mask);
        Reg::Compares<int32_t, CMPMODE::EQ>(available, code, 0, mask);
        Reg::Duplicate(number, col + 1, mask);
        Reg::Select(number, number, code, zero);
        Reg::Select(code, number, code, available);
        Reg::Div(reciprocal, ones, diagonal, mask);
        Reg::Muls(negative, diagonal, -1.f, mask);
        Reg::Muls(residual, ones, 1.f, mask);
        Reg::MulAddDst(residual, reciprocal, negative, mask);
        Reg::Div(correction, residual, diagonal, mask);
        Reg::Add(reciprocal, reciprocal, correction, mask);
        Reg::Duplicate(sum, 0.f, mask);
        if constexpr (TwoRows) Reg::Duplicate(sumHigh, 0.f, maskHigh);
        const int top = TwoRows && col > 64 ? 64 : col;
        for (int k = 0; k < top; ++k)
        {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(coefficient, factors + col * stride + k);
            Reg::LoadAlign(other, tile + k * stride);
            Reg::MulAddDst(sum, other, coefficient, mask);
        }
        if constexpr (TwoRows)
            for (int k = 64; k < col; ++k)
            {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(coefficient, factors + col * stride + k);
                Reg::LoadAlign(other, tile + k * stride);
                Reg::LoadAlign(otherHigh, tile + k * stride + 64);
                Reg::MulAddDst(sum, other, coefficient, mask);
                Reg::MulAddDst(sumHigh, otherHigh, coefficient, maskHigh);
            }
        Reg::Mul(value, sum, reciprocal, mask);
        Reg::Compares<uint32_t, CMPMODE::EQ>(diagonalRow, rows, static_cast<uint32_t>(col), mask);
        Reg::Select(value, reciprocal, value, diagonalRow);
        Reg::StoreAlign(tile + col * stride, value, mask);
        if constexpr (TwoRows)
        {
            Reg::Mul(valueHigh, sumHigh, reciprocal, maskHigh);
            Reg::Compares<uint32_t, CMPMODE::EQ>(diagonalHigh, rowsHigh, static_cast<uint32_t>(col), maskHigh);
            Reg::Select(valueHigh, reciprocal, valueHigh, diagonalHigh);
            Reg::StoreAlign(tile + col * stride + 64, valueHigh, maskHigh);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
    }
    for (int col = n - 1; col >= 0; --col)
    {
        Reg::Duplicate(sum, 0.f, mask);
        if constexpr (TwoRows) Reg::Duplicate(sumHigh, 0.f, maskHigh);
        for (int k = col + 1; k < n; ++k)
        {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(coefficient, factors + col * stride + k);
            Reg::LoadAlign(other, tile + k * stride);
            if constexpr (TwoRows) Reg::LoadAlign(otherHigh, tile + k * stride + 64);
            Reg::MulAddDst(sum, other, coefficient, mask);
            if constexpr (TwoRows) Reg::MulAddDst(sumHigh, otherHigh, coefficient, maskHigh);
        }
        Reg::LoadAlign(value, tile + col * stride);
        Reg::Add(value, value, sum, mask);
        Reg::StoreAlign(tile + col * stride, value, mask);
        if constexpr (TwoRows)
        {
            Reg::LoadAlign(valueHigh, tile + col * stride + 64);
            Reg::Add(valueHigh, valueHigh, sumHigh, maskHigh);
            Reg::StoreAlign(tile + col * stride + 64, valueHigh, maskHigh);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
    }
    if (pivoting)
        for (int col = n - 1; col >= 0; --col)
        {
            Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(pivot,
                                                                  reinterpret_cast<__ubuf__ uint32_t*>(pivots + col));
            Reg::Duplicate(fallback, static_cast<uint32_t>(col + 1), mask);
            Reg::Compares<uint32_t, CMPMODE::LT>(bad, pivot, static_cast<uint32_t>(col + 1), mask);
            Reg::Select(code, invalid, code, bad);
            Reg::Select(pivot, fallback, pivot, bad);
            Reg::Compares<uint32_t, CMPMODE::GT>(bad, pivot, static_cast<uint32_t>(n), mask);
            Reg::Select(code, invalid, code, bad);
            Reg::Select(pivot, fallback, pivot, bad);
            Reg::Adds(pivot, pivot, uint32_t{0xffffffff}, mask);
            Reg::Muls(base, pivot, static_cast<uint32_t>(stride), mask);
            Reg::Add(index, base, rows, mask);
            Reg::LoadAlign(value, tile + col * stride);
            Reg::Gather(other, tile, index, mask);
            Reg::Scatter(tile, value, index, mask);
            Reg::StoreAlign(tile + col * stride, other, mask);
            if constexpr (TwoRows)
            {
                Reg::Add(index, base, rowsHigh, maskHigh);
                Reg::LoadAlign(valueHigh, tile + col * stride + 64);
                Reg::Gather(otherHigh, tile, index, maskHigh);
                Reg::Scatter(tile, valueHigh, index, maskHigh);
                Reg::StoreAlign(tile + col * stride + 64, otherHigh, maskHigh);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
        }
    Reg::StoreAlign(info, code, mask);
    Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
}

__aicore__ void BatchGetriMediumTiled(int n, GM_ADDR a, int lda, GM_ADDR pivots, GM_ADDR c, int ldc, GM_ADDR info,
                                      int batch)
{
    TPipe pipe;
    TBuf<TPosition::VECIN> factorBuffer, pivotBuffer, pointerBuffer;
    TBuf<TPosition::VECCALC> tileBuffer, seedBuffer;
    TBuf<TPosition::VECOUT> infoBuffer;
    pipe.InitBuffer(factorBuffer, 128 * 128 * sizeof(float));
    pipe.InitBuffer(tileBuffer, 128 * 128 * sizeof(float));
    pipe.InitBuffer(pivotBuffer, 128 * sizeof(int));
    pipe.InitBuffer(pointerBuffer, 64);
    pipe.InitBuffer(seedBuffer, 64 * sizeof(uint32_t));
    pipe.InitBuffer(infoBuffer, 64 * sizeof(int));
    auto factors = factorBuffer.Get<float>(), tile = tileBuffer.Get<float>();
    auto localPivots = pivotBuffer.Get<int>(), localInfo = infoBuffer.Get<int>();
    auto pointers = pointerBuffer.Get<uint32_t>(), seed = seedBuffer.Get<uint32_t>();
    for (int k = 0; k < 64; ++k) seed.SetValue(k, k);
    const int stride = (n + 63) / 64 * 64, padded = (n + 7) / 8 * 8;
    GlobalTensor<uint32_t> ap, cp;
    GlobalTensor<int> pp, ip;
    ap.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(a));
    cp.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(c));
    pp.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(pivots));
    ip.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(info));
    for (int matrix = GetBlockIdx(); matrix < batch; matrix += GetBlockNum())
    {
        DataCopyPad(pointers, ap[matrix * 2], DataCopyExtParams{1, 8, 0, 0, 0},
                    DataCopyPadExtParams<uint32_t>{true, 0, 6, 0});
        DataCopyPad(pointers[8], cp[matrix * 2], DataCopyExtParams{1, 8, 0, 0, 0},
                    DataCopyPadExtParams<uint32_t>{true, 0, 6, 0});
        BatchFence<HardEvent::MTE2_S>();
        const auto addresses = pointers.ReinterpretCast<uint64_t>();
        const uint64_t av = addresses.GetValue(0), cv = addresses.GetValue(4);
        const uint64_t ab = (static_cast<uint64_t>(n - 1) * lda + n) * sizeof(float);
        const uint64_t cb = (static_cast<uint64_t>(n - 1) * ldc + n) * sizeof(float);
        int code = av == 0 ? -2 : (cv == 0 || (av < cv + cb && cv < av + ab) ? -5 : 0);
        if (code == 0)
        {
            GlobalTensor<float> source, destination;
            source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(av));
            destination.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cv));
            DataCopyPad(factors, source,
                        DataCopyExtParams{static_cast<uint16_t>(n), static_cast<uint32_t>(n * sizeof(float)),
                                          static_cast<uint32_t>((lda - n) * sizeof(float)),
                                          static_cast<uint32_t>((stride - padded) / 8), 0},
                        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(padded - n), 0.f});
            if (pivots != nullptr)
                DataCopyPad(localPivots, pp[static_cast<uint64_t>(matrix) * n],
                            DataCopyExtParams{1, static_cast<uint32_t>(n * sizeof(int)), 0, 0, 0},
                            DataCopyPadExtParams<int>{true, 0, static_cast<uint8_t>(padded - n), 0});
            BatchFence<HardEvent::MTE2_V>();
            BatchFence<HardEvent::S_V>();
            if (n <= 64)
                GetriMediumTiledVF<false>(reinterpret_cast<__ubuf__ float*>(factors.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ int32_t*>(localPivots.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ int32_t*>(localInfo.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ uint32_t*>(seed.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float*>(tile.GetPhyAddr()), n, pivots != nullptr);
            else
                GetriMediumTiledVF<true>(reinterpret_cast<__ubuf__ float*>(factors.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ int32_t*>(localPivots.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ int32_t*>(localInfo.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ uint32_t*>(seed.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ float*>(tile.GetPhyAddr()), n, pivots != nullptr);
            BatchFence<HardEvent::V_S>();
            code = localInfo.GetValue(0);
            if (code == 0)
            {
                BatchFence<HardEvent::V_MTE3>();
                DataCopyPad(destination, tile,
                            DataCopyExtParams{static_cast<uint16_t>(n), static_cast<uint32_t>(n * sizeof(float)),
                                              static_cast<uint32_t>((stride - padded) / 8),
                                              static_cast<uint32_t>((ldc - n) * sizeof(float)), 0});
                BatchFence<HardEvent::MTE3_V>();
            }
        }
        localInfo.SetValue(0, code);
        BatchFence<HardEvent::S_MTE3>();
        SetAtomicAdd<int>();
        DataCopyPad(ip[matrix], localInfo, DataCopyExtParams{1, sizeof(int), 0, 0, 0});
        SetAtomicNone();
        BatchFence<HardEvent::MTE3_S>();
        BatchFence<HardEvent::MTE3_MTE2>();
    }
}
