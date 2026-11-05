/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Named vector registers keep RHS row sums live without dynamic register indexing.
struct BatchRowRegisters
{
    Reg::RegTensor<float> r0;
    Reg::RegTensor<float> r1;
    Reg::RegTensor<float> r2;
    Reg::RegTensor<float> r3;
    Reg::RegTensor<float> r4;
    Reg::RegTensor<float> r5;
    Reg::RegTensor<float> r6;
    Reg::RegTensor<float> r7;
    Reg::RegTensor<float> r8;
    Reg::RegTensor<float> r9;
    Reg::RegTensor<float> r10;
    Reg::RegTensor<float> r11;
    Reg::RegTensor<float> r12;
    Reg::RegTensor<float> r13;
    Reg::RegTensor<float> r14;
    Reg::RegTensor<float> r15;
    Reg::RegTensor<float> r16;
    Reg::RegTensor<float> r17;
    Reg::RegTensor<float> r18;
    Reg::RegTensor<float> r19;
    Reg::RegTensor<float> r20;
    Reg::RegTensor<float> r21;
    Reg::RegTensor<float> r22;
    Reg::RegTensor<float> r23;
    Reg::RegTensor<float> r24;
    Reg::RegTensor<float> r25;
    Reg::RegTensor<float> r26;
    Reg::RegTensor<float> r27;
    Reg::RegTensor<float> r28;
    Reg::RegTensor<float> r29;
    Reg::RegTensor<float> r30;
    Reg::RegTensor<float> r31;
};

template <int Row>
__simd_callee__ inline Reg::RegTensor<float>& BatchRow(BatchRowRegisters& rows)
{
    static_assert(Row >= 0 && Row < 32);
    if constexpr (Row == 0)
        return rows.r0;
    else if constexpr (Row == 1)
        return rows.r1;
    else if constexpr (Row == 2)
        return rows.r2;
    else if constexpr (Row == 3)
        return rows.r3;
    else if constexpr (Row == 4)
        return rows.r4;
    else if constexpr (Row == 5)
        return rows.r5;
    else if constexpr (Row == 6)
        return rows.r6;
    else if constexpr (Row == 7)
        return rows.r7;
    else if constexpr (Row == 8)
        return rows.r8;
    else if constexpr (Row == 9)
        return rows.r9;
    else if constexpr (Row == 10)
        return rows.r10;
    else if constexpr (Row == 11)
        return rows.r11;
    else if constexpr (Row == 12)
        return rows.r12;
    else if constexpr (Row == 13)
        return rows.r13;
    else if constexpr (Row == 14)
        return rows.r14;
    else if constexpr (Row == 15)
        return rows.r15;
    else if constexpr (Row == 16)
        return rows.r16;
    else if constexpr (Row == 17)
        return rows.r17;
    else if constexpr (Row == 18)
        return rows.r18;
    else if constexpr (Row == 19)
        return rows.r19;
    else if constexpr (Row == 20)
        return rows.r20;
    else if constexpr (Row == 21)
        return rows.r21;
    else if constexpr (Row == 22)
        return rows.r22;
    else if constexpr (Row == 23)
        return rows.r23;
    else if constexpr (Row == 24)
        return rows.r24;
    else if constexpr (Row == 25)
        return rows.r25;
    else if constexpr (Row == 26)
        return rows.r26;
    else if constexpr (Row == 27)
        return rows.r27;
    else if constexpr (Row == 28)
        return rows.r28;
    else if constexpr (Row == 29)
        return rows.r29;
    else if constexpr (Row == 30)
        return rows.r30;
    else
        return rows.r31;
}

template <int Limit, int Row = 0>
__simd_callee__ inline void BatchClearRows(BatchRowRegisters& rows, Reg::MaskReg& mask)
{
    Reg::Duplicate(BatchRow<Row>(rows), 0.f, mask);
    if constexpr (Row + 1 < Limit) BatchClearRows<Limit, Row + 1>(rows, mask);
}

template <int Limit, int Row = 0>
__simd_callee__ inline void BatchAccumulateRows(BatchRowRegisters& rows, __ubuf__ float* inverse,
                                                Reg::RegTensor<float>& source, Reg::RegTensor<float>& coefficient,
                                                Reg::MaskReg& mask, int n, int group, int offset, int columnsPerVector,
                                                int k, int rowBase = 0)
{
    const int first = rowBase + Row * columnsPerVector;
    if (first < n)
    {
        Reg::LoadAlign(source, inverse + (k * n + first) * group + offset);
        Reg::MulAddDst(BatchRow<Row>(rows), source, coefficient, mask);
    }
    if constexpr (Row + 1 < Limit)
        BatchAccumulateRows<Limit, Row + 1>(rows, inverse, source, coefficient, mask, n, group, offset,
                                            columnsPerVector, k, rowBase);
}

template <int Limit, int Row = 0>
__simd_callee__ inline void BatchStoreRows(BatchRowRegisters& rows, __ubuf__ float* inverse,
                                           Reg::RegTensor<float>& source, int n, int group, int offset,
                                           int rowsPerVector, uint32_t active, int col, int rowBase = 0)
{
    const int first = rowBase + Row * rowsPerVector;
    if (first < n)
    {
        uint32_t count = static_cast<uint32_t>((n - first < rowsPerVector ? n - first : rowsPerVector) *
                                               (group < 64 ? group : active));
        auto mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(source, inverse + (col * n + first) * group + offset);
        Reg::Add(BatchRow<Row>(rows), BatchRow<Row>(rows), source, mask);
        Reg::StoreAlign(inverse + (col * n + first) * group + offset, BatchRow<Row>(rows), mask);
    }
    if constexpr (Row + 1 < Limit)
        BatchStoreRows<Limit, Row + 1>(rows, inverse, source, n, group, offset, rowsPerVector, active, col, rowBase);
}
