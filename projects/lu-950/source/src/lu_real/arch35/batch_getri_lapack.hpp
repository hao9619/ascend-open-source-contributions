/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batch_register_rows.hpp"

// Choose a register tile without partially empty row groups where a small divisor exists.
template <int Rows, int Candidate = (Rows < 16 ? Rows : 16)>
struct BatchInverseRowTile
{
    static constexpr int value = Rows % Candidate == 0 ? Candidate : BatchInverseRowTile<Rows, Candidate - 1>::value;
};

template <int Rows>
struct BatchInverseRowTile<Rows, 1>
{
    static constexpr int value = Rows < 8 ? Rows : 8;
};

// Invert U, remove L, then undo pivots in LAPACK order. SIMD rows share each
// lower coefficient, packing multiple RHS rows when the matrix group is small.
template <int MaxN, int RowsPerVector, int FixedN = 0>
__simd_vf__ inline void BatchLapackVF(__ubuf__ float* inverse, __ubuf__ float* factors, __ubuf__ int32_t* pivots,
                                      __ubuf__ uint32_t* seed, int inputN, int group, int offset, uint32_t active,
                                      bool pivoting)
{
    const int n = FixedN == 0 ? inputN : FixedN;
    uint32_t remaining = active;
    auto mask = Reg::UpdateMask<float>(remaining);
    Reg::RegTensor<float> value, coefficient, source, diagonal, reciprocal;
    Reg::RegTensor<float> ones, negative, residual, correction, saved;
    Reg::RegTensor<uint32_t> lanes, index, pivot, fallback;
    Reg::MaskReg bad, swapped;
    Reg::LoadAlign(lanes, seed);
    Reg::Duplicate(value, 0.f, mask);
    Reg::Duplicate(ones, 1.f, mask);
    for (int col = 0; col < n; ++col)
    {
#ifdef LU950_HIGH_SCHEDULE
        // Every upper entry is produced before it is read; only the strict lower triangle needs zero fill.
        const int firstZeroRow = FixedN >= 18 ? col + 1 : 0;
#else
        const int firstZeroRow = 0;
#endif
        for (int row = firstZeroRow; row < n; ++row)
            Reg::StoreAlign(inverse + (col * n + row) * group + offset, value, mask);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (int col = 0; col < n; ++col)
    {
        Reg::LoadAlign(diagonal, factors + (col * n + col) * group + offset);
        Reg::Div(reciprocal, ones, diagonal, mask);
        Reg::Muls(negative, diagonal, -1.f, mask);
        Reg::Muls(residual, ones, 1.f, mask);
        Reg::MulAddDst(residual, reciprocal, negative, mask);
        if constexpr (FixedN >= 18)
            Reg::MulAddDst(reciprocal, residual, reciprocal, mask);
        else
        {
            Reg::Div(correction, residual, diagonal, mask);
            Reg::Add(reciprocal, reciprocal, correction, mask);
        }
        if constexpr (FixedN >= 18 && RowsPerVector > 1)
        {
            Reg::StoreAlign(inverse + (col * n + col) * group + offset, reciprocal, mask);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            for (int row = 0; row < col; row += RowsPerVector)
            {
                uint32_t upperActive = (col - row < RowsPerVector ? col - row : RowsPerVector) * group;
                auto upperMask = Reg::UpdateMask<float>(upperActive);
                Reg::Duplicate(value, 0.f, upperMask);
                for (int k = row; k < col; ++k)
                {
                    Reg::LoadAlign(source, inverse + (k * n + row) * group + offset);
                    Reg::Gather(coefficient, factors + (k * n + col) * group, lanes, upperMask);
                    Reg::MulAddDst(value, source, coefficient, upperMask);
                }
                Reg::Gather(saved, inverse + (col * n + col) * group, lanes, upperMask);
                Reg::Mul(value, value, saved, upperMask);
                Reg::StoreAlign(inverse + (col * n + row) * group + offset, value, upperMask);
            }
        }
        else
        {
            for (int row = 0; row < col; ++row)
            {
                Reg::Duplicate(value, 0.f, mask);
                for (int k = row; k < col; ++k)
                {
                    Reg::LoadAlign(source, inverse + (k * n + row) * group + offset);
                    Reg::LoadAlign(coefficient, factors + (k * n + col) * group + offset);
                    Reg::MulAddDst(value, source, coefficient, mask);
                }
                Reg::Mul(value, value, reciprocal, mask);
                Reg::StoreAlign(inverse + (col * n + row) * group + offset, value, mask);
            }
        }
        Reg::StoreAlign(inverse + (col * n + col) * group + offset, reciprocal, mask);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
    constexpr int rowsPerVector = RowsPerVector;
    constexpr int completeRows = (MaxN + RowsPerVector - 1) / RowsPerVector;
#ifdef LU950_HIGH_SCHEDULE
    constexpr int rowRegisters = FixedN >= 18 ? BatchInverseRowTile<completeRows>::value : completeRows;
#else
    constexpr int rowRegisters = completeRows;
#endif
    BatchRowRegisters sums;
    uint32_t packedActive = group < 64 ? group * rowsPerVector : active;
    auto packedMask = Reg::UpdateMask<float>(packedActive);
    for (int col = n - 1; col >= 0; --col)
    {
        for (int rowBase = 0; rowBase < n; rowBase += rowRegisters * rowsPerVector)
        {
            BatchClearRows<rowRegisters>(sums, packedMask);
            for (int k = col + 1; k < n; ++k)
            {
                Reg::Gather(coefficient, factors + (k * n + col) * group, lanes, packedMask);
                BatchAccumulateRows<rowRegisters>(sums, inverse, source, coefficient, packedMask, n, group, offset,
                                                  rowsPerVector, k, rowBase);
            }
            // SGEMV accumulates the products before adding the existing U inverse.
            BatchStoreRows<rowRegisters>(sums, inverse, source, n, group, offset, rowsPerVector, active, col, rowBase);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }
    }
    if (pivoting)
        for (int col = n - 1; col >= 0; --col)
        {
            Reg::Muls(index, lanes, static_cast<uint32_t>(n), mask);
            Reg::Adds(index, index, static_cast<uint32_t>(col), mask);
            Reg::Gather(pivot, reinterpret_cast<__ubuf__ uint32_t*>(pivots), index, mask);
            Reg::Duplicate(fallback, static_cast<uint32_t>(col + 1), mask);
            Reg::Compares<uint32_t, CMPMODE::LT>(bad, pivot, static_cast<uint32_t>(col + 1), mask);
            Reg::Select(pivot, fallback, pivot, bad);
            Reg::Compares<uint32_t, CMPMODE::GT>(bad, pivot, static_cast<uint32_t>(n), mask);
            Reg::Select(pivot, fallback, pivot, bad);
            Reg::Compares<uint32_t, CMPMODE::NE>(swapped, pivot, static_cast<uint32_t>(col + 1), mask);
            Reg::Adds(pivot, pivot, uint32_t{0xffffffff}, mask);
            Reg::Muls(pivot, pivot, static_cast<uint32_t>(n * group), mask);
            Reg::Add(pivot, pivot, lanes, mask);
            for (int row = 0; row < n; ++row)
            {
                Reg::Adds(index, pivot, static_cast<uint32_t>(row * group), mask);
                Reg::LoadAlign(value, inverse + (col * n + row) * group + offset);
                Reg::Gather(saved, inverse, index, swapped);
                Reg::StoreAlign(inverse + (col * n + row) * group + offset, saved, swapped);
                Reg::Scatter(inverse, value, index, swapped);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }
}
