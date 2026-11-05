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
__simd_vf__ inline void BatchLapackN4VF(__ubuf__ float* inverse, __ubuf__ float* factors, __ubuf__ int32_t* pivots,
                                        __ubuf__ uint32_t* seed, int group, int offset, uint32_t active, bool pivoting)
{
    auto mask = Reg::UpdateMask<float>(active);
    Reg::RegTensor<float> ones, negative, residual, correction, reciprocal, sum, saved;
    Reg::RegTensor<float> t0, t1, t2, t3;
    Reg::RegTensor<uint32_t> lanes, index, pivot;
    Reg::MaskReg swapped;
    Reg::RegTensor<float> a00, a01, a02, a03;
    Reg::RegTensor<float> a10, a11, a12, a13;
    Reg::RegTensor<float> a20, a21, a22, a23;
    Reg::RegTensor<float> a30, a31, a32, a33;
    Reg::LoadAlign(lanes, seed);
    Reg::Duplicate(ones, 1.f, mask);
    Reg::LoadAlign(a00, factors + 0 * group + offset);
    Reg::LoadAlign(a01, factors + 1 * group + offset);
    Reg::LoadAlign(a02, factors + 2 * group + offset);
    Reg::LoadAlign(a03, factors + 3 * group + offset);
    Reg::LoadAlign(a10, factors + 4 * group + offset);
    Reg::LoadAlign(a11, factors + 5 * group + offset);
    Reg::LoadAlign(a12, factors + 6 * group + offset);
    Reg::LoadAlign(a13, factors + 7 * group + offset);
    Reg::LoadAlign(a20, factors + 8 * group + offset);
    Reg::LoadAlign(a21, factors + 9 * group + offset);
    Reg::LoadAlign(a22, factors + 10 * group + offset);
    Reg::LoadAlign(a23, factors + 11 * group + offset);
    Reg::LoadAlign(a30, factors + 12 * group + offset);
    Reg::LoadAlign(a31, factors + 13 * group + offset);
    Reg::LoadAlign(a32, factors + 14 * group + offset);
    Reg::LoadAlign(a33, factors + 15 * group + offset);
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
    if (pivoting)
    {
        Reg::Muls(index, lanes, uint32_t{4}, mask);
        Reg::Adds(index, index, uint32_t{2}, mask);
        Reg::Gather(pivot, reinterpret_cast<__ubuf__ uint32_t*>(pivots), index, mask);
        Reg::Compares<uint32_t, CMPMODE::EQ>(swapped, pivot, uint32_t{4}, mask);
        Reg::Muls(saved, a02, 1.f, mask);
        Reg::Select(a02, a03, a02, swapped);
        Reg::Select(a03, saved, a03, swapped);
        Reg::Muls(saved, a12, 1.f, mask);
        Reg::Select(a12, a13, a12, swapped);
        Reg::Select(a13, saved, a13, swapped);
        Reg::Muls(saved, a22, 1.f, mask);
        Reg::Select(a22, a23, a22, swapped);
        Reg::Select(a23, saved, a23, swapped);
        Reg::Muls(saved, a32, 1.f, mask);
        Reg::Select(a32, a33, a32, swapped);
        Reg::Select(a33, saved, a33, swapped);
        Reg::Muls(index, lanes, uint32_t{4}, mask);
        Reg::Adds(index, index, uint32_t{1}, mask);
        Reg::Gather(pivot, reinterpret_cast<__ubuf__ uint32_t*>(pivots), index, mask);
        Reg::Compares<uint32_t, CMPMODE::EQ>(swapped, pivot, uint32_t{3}, mask);
        Reg::Muls(saved, a01, 1.f, mask);
        Reg::Select(a01, a02, a01, swapped);
        Reg::Select(a02, saved, a02, swapped);
        Reg::Muls(saved, a11, 1.f, mask);
        Reg::Select(a11, a12, a11, swapped);
        Reg::Select(a12, saved, a12, swapped);
        Reg::Muls(saved, a21, 1.f, mask);
        Reg::Select(a21, a22, a21, swapped);
        Reg::Select(a22, saved, a22, swapped);
        Reg::Muls(saved, a31, 1.f, mask);
        Reg::Select(a31, a32, a31, swapped);
        Reg::Select(a32, saved, a32, swapped);
        Reg::Compares<uint32_t, CMPMODE::EQ>(swapped, pivot, uint32_t{4}, mask);
        Reg::Muls(saved, a01, 1.f, mask);
        Reg::Select(a01, a03, a01, swapped);
        Reg::Select(a03, saved, a03, swapped);
        Reg::Muls(saved, a11, 1.f, mask);
        Reg::Select(a11, a13, a11, swapped);
        Reg::Select(a13, saved, a13, swapped);
        Reg::Muls(saved, a21, 1.f, mask);
        Reg::Select(a21, a23, a21, swapped);
        Reg::Select(a23, saved, a23, swapped);
        Reg::Muls(saved, a31, 1.f, mask);
        Reg::Select(a31, a33, a31, swapped);
        Reg::Select(a33, saved, a33, swapped);
        Reg::Muls(index, lanes, uint32_t{4}, mask);
        Reg::Adds(index, index, uint32_t{0}, mask);
        Reg::Gather(pivot, reinterpret_cast<__ubuf__ uint32_t*>(pivots), index, mask);
        Reg::Compares<uint32_t, CMPMODE::EQ>(swapped, pivot, uint32_t{2}, mask);
        Reg::Muls(saved, a00, 1.f, mask);
        Reg::Select(a00, a01, a00, swapped);
        Reg::Select(a01, saved, a01, swapped);
        Reg::Muls(saved, a10, 1.f, mask);
        Reg::Select(a10, a11, a10, swapped);
        Reg::Select(a11, saved, a11, swapped);
        Reg::Muls(saved, a20, 1.f, mask);
        Reg::Select(a20, a21, a20, swapped);
        Reg::Select(a21, saved, a21, swapped);
        Reg::Muls(saved, a30, 1.f, mask);
        Reg::Select(a30, a31, a30, swapped);
        Reg::Select(a31, saved, a31, swapped);
        Reg::Compares<uint32_t, CMPMODE::EQ>(swapped, pivot, uint32_t{3}, mask);
        Reg::Muls(saved, a00, 1.f, mask);
        Reg::Select(a00, a02, a00, swapped);
        Reg::Select(a02, saved, a02, swapped);
        Reg::Muls(saved, a10, 1.f, mask);
        Reg::Select(a10, a12, a10, swapped);
        Reg::Select(a12, saved, a12, swapped);
        Reg::Muls(saved, a20, 1.f, mask);
        Reg::Select(a20, a22, a20, swapped);
        Reg::Select(a22, saved, a22, swapped);
        Reg::Muls(saved, a30, 1.f, mask);
        Reg::Select(a30, a32, a30, swapped);
        Reg::Select(a32, saved, a32, swapped);
        Reg::Compares<uint32_t, CMPMODE::EQ>(swapped, pivot, uint32_t{4}, mask);
        Reg::Muls(saved, a00, 1.f, mask);
        Reg::Select(a00, a03, a00, swapped);
        Reg::Select(a03, saved, a03, swapped);
        Reg::Muls(saved, a10, 1.f, mask);
        Reg::Select(a10, a13, a10, swapped);
        Reg::Select(a13, saved, a13, swapped);
        Reg::Muls(saved, a20, 1.f, mask);
        Reg::Select(a20, a23, a20, swapped);
        Reg::Select(a23, saved, a23, swapped);
        Reg::Muls(saved, a30, 1.f, mask);
        Reg::Select(a30, a33, a30, swapped);
        Reg::Select(a33, saved, a33, swapped);
    }
    Reg::StoreAlign(inverse + 0 * group + offset, a00, mask);
    Reg::StoreAlign(inverse + 1 * group + offset, a10, mask);
    Reg::StoreAlign(inverse + 2 * group + offset, a20, mask);
    Reg::StoreAlign(inverse + 3 * group + offset, a30, mask);
    Reg::StoreAlign(inverse + 4 * group + offset, a01, mask);
    Reg::StoreAlign(inverse + 5 * group + offset, a11, mask);
    Reg::StoreAlign(inverse + 6 * group + offset, a21, mask);
    Reg::StoreAlign(inverse + 7 * group + offset, a31, mask);
    Reg::StoreAlign(inverse + 8 * group + offset, a02, mask);
    Reg::StoreAlign(inverse + 9 * group + offset, a12, mask);
    Reg::StoreAlign(inverse + 10 * group + offset, a22, mask);
    Reg::StoreAlign(inverse + 11 * group + offset, a32, mask);
    Reg::StoreAlign(inverse + 12 * group + offset, a03, mask);
    Reg::StoreAlign(inverse + 13 * group + offset, a13, mask);
    Reg::StoreAlign(inverse + 14 * group + offset, a23, mask);
    Reg::StoreAlign(inverse + 15 * group + offset, a33, mask);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}
