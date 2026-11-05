/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Four matrices share all sixteen SIMD rows. One 4 KiB tile holds intermediate columns.
__simd_vf__ inline void GetriN16TiledVF(__ubuf__ float* values, __ubuf__ int32_t* pivots, __ubuf__ int32_t* infos,
                                        __ubuf__ uint32_t* seed, int count, bool pivoting, __ubuf__ float* tile)
{
    auto full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> matrixLane, rows, base, index, packedLane, pivot, fallback, target;
    Reg::RegTensor<int32_t> code, number, invalid;
    Reg::RegTensor<float> factors, diagonal, reciprocal, negative, residual, correction;
    Reg::RegTensor<float> sum, coefficient, value, other, ones;
    Reg::MaskReg mask, zero, available, diagonalRow, bad, swapped;
    Reg::LoadAlign(matrixLane, seed);
    Reg::LoadAlign(rows, seed + 64);
    Reg::Muls(base, matrixLane, uint32_t{256}, full);
    Reg::Add(base, base, rows, full);
    Reg::Muls(packedLane, rows, uint32_t{4}, full);
    Reg::Add(packedLane, packedLane, matrixLane, full);
    for (int start = 0; start < count; start += 4)
    {
        uint32_t active = static_cast<uint32_t>(count - start < 4 ? count - start : 4);
        Reg::Compares<uint32_t, CMPMODE::LT>(mask, matrixLane, active, full);
        Reg::Duplicate(ones, 1.f, mask);
        Reg::Duplicate(code, 0, mask);
        Reg::Duplicate(invalid, -4, mask);
        auto* data = values + start * 256;
        for (int col = 0; col < 16; ++col)
        {
            Reg::Adds(index, base, static_cast<uint32_t>(col * 16), mask);
            Reg::Gather(factors, data, index, mask);
            Reg::Adds(index, matrixLane, static_cast<uint32_t>(col * 4), mask);
            Reg::Gather(diagonal, factors, index);
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
            for (int k = 0; k < col; ++k)
            {
                Reg::Adds(index, matrixLane, static_cast<uint32_t>(k * 4), mask);
                Reg::Gather(coefficient, factors, index);
                Reg::Muls(coefficient, coefficient, -1.f, mask);
                Reg::LoadAlign(other, tile + k * 64);
                Reg::MulAddDst(sum, other, coefficient, mask);
            }
            Reg::Mul(value, sum, reciprocal, mask);
            Reg::Compares<uint32_t, CMPMODE::EQ>(diagonalRow, rows, static_cast<uint32_t>(col), mask);
            Reg::Select(value, reciprocal, value, diagonalRow);
            Reg::StoreAlign(tile + col * 64, value, mask);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }
        for (int col = 15; col >= 0; --col)
        {
            Reg::Adds(index, base, static_cast<uint32_t>(col * 16), mask);
            Reg::Gather(factors, data, index, mask);
            Reg::Duplicate(sum, 0.f, mask);
            for (int k = col + 1; k < 16; ++k)
            {
                Reg::Adds(index, matrixLane, static_cast<uint32_t>(k * 4), mask);
                Reg::Gather(coefficient, factors, index);
                Reg::Muls(coefficient, coefficient, -1.f, mask);
                Reg::LoadAlign(other, tile + k * 64);
                Reg::MulAddDst(sum, other, coefficient, mask);
            }
            Reg::LoadAlign(value, tile + col * 64);
            Reg::Add(value, value, sum, mask);
            Reg::StoreAlign(tile + col * 64, value, mask);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }
        if (pivoting)
            for (int col = 15; col >= 0; --col)
            {
                Reg::Muls(index, matrixLane, uint32_t{16}, mask);
                Reg::Adds(index, index, static_cast<uint32_t>(col), mask);
                Reg::Gather(pivot, reinterpret_cast<__ubuf__ uint32_t*>(pivots + start * 16), index, mask);
                Reg::Duplicate(fallback, static_cast<uint32_t>(col + 1), mask);
                Reg::Compares<uint32_t, CMPMODE::LT>(bad, pivot, static_cast<uint32_t>(col + 1), mask);
                Reg::Select(code, invalid, code, bad);
                Reg::Select(pivot, fallback, pivot, bad);
                Reg::Compares<uint32_t, CMPMODE::GT>(bad, pivot, uint32_t{16}, mask);
                Reg::Select(code, invalid, code, bad);
                Reg::Select(pivot, fallback, pivot, bad);
                Reg::Compares<uint32_t, CMPMODE::NE>(swapped, pivot, static_cast<uint32_t>(col + 1), mask);
                Reg::Adds(target, pivot, uint32_t{0xffffffff}, mask);
                Reg::Muls(target, target, uint32_t{64}, mask);
                Reg::Add(target, target, packedLane, mask);
                Reg::LoadAlign(value, tile + col * 64);
                Reg::Gather(other, tile, target, swapped);
                Reg::StoreAlign(tile + col * 64, other, swapped);
                Reg::Scatter(tile, value, target, swapped);
                Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
            }
        for (int col = 0; col < 16; ++col)
        {
            Reg::LoadAlign(value, tile + col * 64);
            Reg::Adds(index, base, static_cast<uint32_t>(col * 16), mask);
            Reg::Scatter(data, value, index, mask);
        }
        auto infoMask = Reg::UpdateMask<int32_t>(active);
        Reg::Adds(index, matrixLane, static_cast<uint32_t>(start), infoMask);
        Reg::Scatter(infos, code, index, infoMask);
        Reg::LocalMemBar<Reg::MemType::VEC_ALL, Reg::MemType::VEC_ALL>();
    }
}
