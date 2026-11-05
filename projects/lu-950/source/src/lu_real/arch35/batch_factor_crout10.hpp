/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Column-oriented partial-pivot LU. Pair sums retain their low part before the final FP32 upper update.
template <int N>
__simd_vf__ inline void BatchFactorCroutVF(__ubuf__ float* factors, __ubuf__ int32_t* pivots, __ubuf__ int32_t* infos,
                                           __ubuf__ uint32_t* seed, int group, int offset, uint32_t active)
{
    constexpr int n = N;
    auto mask = Reg::UpdateMask<float>(active);
    Reg::RegTensor<float> value, other, sum, low, pair, product, next, v, u, error;
    Reg::RegTensor<float> largest, magnitude, diagonal, reciprocal, ones, negative, residual, correction;
    Reg::RegTensor<uint32_t> lanes, index, target, pivot, rowNumber;
    Reg::RegTensor<int32_t> code, singular, nextCode;
    Reg::MaskReg condition, available;
    Reg::LoadAlign(lanes, seed);
    Reg::LoadAlign(code, infos + offset);
    Reg::Duplicate(ones, 1.f, mask);
    for (int col = 0; col < n; ++col)
    {
        for (int row = 1; row < col; ++row)
        {
            Reg::Duplicate(sum, 0.f, mask);
            Reg::Duplicate(low, 0.f, mask);
            if constexpr (N >= 18)
            {
                for (int k = 0; k < row; ++k)
                {
                    Reg::LoadAlign(value, factors + (row * n + k) * group + offset);
                    Reg::LoadAlign(other, factors + (k * n + col) * group + offset);
                    Reg::MulAddDst(sum, value, other, mask);
                }
            }
            else
            {
                for (int k = 0; k < row; k += 2)
                {
                    Reg::LoadAlign(value, factors + (row * n + k) * group + offset);
                    Reg::LoadAlign(other, factors + (k * n + col) * group + offset);
                    Reg::Mul(pair, value, other, mask);
                    if (k + 1 < row)
                    {
                        Reg::LoadAlign(value, factors + (row * n + k + 1) * group + offset);
                        Reg::LoadAlign(other, factors + ((k + 1) * n + col) * group + offset);
                        Reg::Mul(product, value, other, mask);
                        Reg::Add(pair, pair, product, mask);
                    }
                    Reg::Add(next, sum, pair, mask);
                    Reg::Sub(v, next, sum, mask);
                    Reg::Sub(u, next, v, mask);
                    Reg::Sub(error, sum, u, mask);
                    Reg::Sub(u, pair, v, mask);
                    Reg::Add(error, error, u, mask);
                    Reg::Add(low, low, error, mask);
                    Reg::Muls(sum, next, 1.f, mask);
                }
                Reg::Add(sum, sum, low, mask);
            }

            Reg::LoadAlign(value, factors + (row * n + col) * group + offset);
            Reg::Sub(value, value, sum, mask);
            Reg::StoreAlign(factors + (row * n + col) * group + offset, value, mask);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }
        for (int row = col; row < n; ++row)
        {
            Reg::Duplicate(sum, 0.f, mask);
            if constexpr (N >= 18)
            {
                for (int k = 0; k < col; ++k)
                {
                    Reg::LoadAlign(value, factors + (row * n + k) * group + offset);
                    Reg::LoadAlign(other, factors + (k * n + col) * group + offset);
                    Reg::MulAddDst(sum, value, other, mask);
                }
            }
            else
            {
                for (int k = 0; k < col; ++k)
                {
                    Reg::LoadAlign(value, factors + (row * n + k) * group + offset);
                    Reg::LoadAlign(other, factors + (k * n + col) * group + offset);
                    Reg::MulAddDst(sum, value, other, mask);
                }
            }
            Reg::LoadAlign(value, factors + (row * n + col) * group + offset);
            Reg::Sub(value, value, sum, mask);
            Reg::StoreAlign(factors + (row * n + col) * group + offset, value, mask);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::LoadAlign(value, factors + (col * n + col) * group + offset);
        Reg::Abs(largest, value, mask);
        Reg::Duplicate(pivot, static_cast<uint32_t>(col), mask);
        for (int row = col + 1; row < n; ++row)
        {
            Reg::LoadAlign(value, factors + (row * n + col) * group + offset);
            Reg::Abs(magnitude, value, mask);
            Reg::Compare<float, CMPMODE::GT>(condition, magnitude, largest, mask);
            Reg::Select(largest, magnitude, largest, condition);
            Reg::Duplicate(rowNumber, static_cast<uint32_t>(row), mask);
            Reg::Select(pivot, rowNumber, pivot, condition);
        }
        Reg::Compares<float, CMPMODE::EQ>(condition, largest, 0.f, mask);
        Reg::Compares<int32_t, CMPMODE::EQ>(available, code, 0, mask);
        Reg::Duplicate(singular, col + 1, mask);
        Reg::Select(nextCode, singular, code, condition);
        Reg::Select(code, nextCode, code, available);
        Reg::Muls(target, pivot, static_cast<uint32_t>(n * group), mask);
        Reg::Add(target, target, lanes, mask);
        for (int k = 0; k < n; ++k)
        {
            Reg::Adds(index, target, static_cast<uint32_t>(k * group), mask);
            Reg::LoadAlign(value, factors + (col * n + k) * group + offset);
            Reg::Gather(other, factors, index, mask);
            Reg::StoreAlign(factors + (col * n + k) * group + offset, other, mask);
            Reg::Scatter(factors, value, index, mask);
        }
        Reg::Adds(pivot, pivot, uint32_t{1}, mask);
        Reg::Muls(index, lanes, static_cast<uint32_t>(n), mask);
        Reg::Adds(index, index, static_cast<uint32_t>(col), mask);
        Reg::Scatter(reinterpret_cast<__ubuf__ uint32_t*>(pivots), pivot, index, mask);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::LoadAlign(diagonal, factors + (col * n + col) * group + offset);
        Reg::Compares<float, CMPMODE::EQ>(condition, diagonal, 0.f, mask);
        Reg::Select(diagonal, ones, diagonal, condition);
        Reg::Div(reciprocal, ones, diagonal, mask);
        Reg::Muls(negative, diagonal, -1.f, mask);
        Reg::Muls(residual, ones, 1.f, mask);
        Reg::MulAddDst(residual, reciprocal, negative, mask);
        if constexpr (N >= 18)
            Reg::MulAddDst(reciprocal, residual, reciprocal, mask);
        else
        {
            Reg::Div(correction, residual, diagonal, mask);
            Reg::Add(reciprocal, reciprocal, correction, mask);
        }
        for (int row = col + 1; row < n; ++row)
        {
            Reg::LoadAlign(value, factors + (row * n + col) * group + offset);
            Reg::Mul(value, value, reciprocal, mask);
            Reg::StoreAlign(factors + (row * n + col) * group + offset, value, mask);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
    Reg::StoreAlign(infos + offset, code, mask);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__aicore__ inline void BatchFactorCroutDispatch(int n, __ubuf__ float* factors, __ubuf__ int32_t* pivots,
                                                __ubuf__ int32_t* infos, __ubuf__ uint32_t* seed, int group, int offset,
                                                uint32_t active)
{
    switch (n)
    {
        case 10:
            BatchFactorCroutVF<10>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 11:
            BatchFactorCroutVF<11>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 12:
            BatchFactorCroutVF<12>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 13:
            BatchFactorCroutVF<13>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 14:
            BatchFactorCroutVF<14>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 15:
            BatchFactorCroutVF<15>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 16:
            BatchFactorCroutVF<16>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 17:
            BatchFactorCroutVF<17>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 18:
            BatchFactorCroutVF<18>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 19:
            BatchFactorCroutVF<19>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 20:
            BatchFactorCroutVF<20>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 21:
            BatchFactorCroutVF<21>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 22:
            BatchFactorCroutVF<22>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 23:
            BatchFactorCroutVF<23>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 24:
            BatchFactorCroutVF<24>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 25:
            BatchFactorCroutVF<25>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 26:
            BatchFactorCroutVF<26>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 27:
            BatchFactorCroutVF<27>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 28:
            BatchFactorCroutVF<28>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 29:
            BatchFactorCroutVF<29>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 30:
            BatchFactorCroutVF<30>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 31:
            BatchFactorCroutVF<31>(factors, pivots, infos, seed, group, offset, active);
            break;
        case 32:
            BatchFactorCroutVF<32>(factors, pivots, infos, seed, group, offset, active);
            break;
    }
}
