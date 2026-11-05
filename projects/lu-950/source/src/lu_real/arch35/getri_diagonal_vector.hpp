/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

__simd_vf__ inline void DiagonalInverseVF(__ubuf__ float *input, __ubuf__ float *rows, __ubuf__ float *columns,
                                          __ubuf__ uint32_t *seed, int n)
{
    uint32_t active = n;
    auto mask = Reg::UpdateMask<float>(active);
    Reg::RegTensor<float> source, destination, coefficient, diagonal, negative, residual, correction, ones;
    Reg::RegTensor<uint32_t> lanes, offsets;
    Reg::MaskReg singular;
    Reg::Duplicate(ones, 1.f, mask);
    // Forward solve for all columns of L^-1 at once.
    for (int k = 0; k < n; ++k)
    {
        Reg::LoadAlign(source, rows + k * 32);
        for (int row = k + 1; row < n; ++row)
        {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(coefficient, input + k * 32 + row);
            Reg::Muls(coefficient, coefficient, -1.f, mask);
            Reg::LoadAlign(destination, rows + row * 32);
            Reg::MulAddDst(destination, source, coefficient, mask);
            Reg::StoreAlign(rows + row * 32, destination, mask);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
    auto *upper = rows + 32 * 32;
    for (int k = n - 1; k >= 0; --k)
    {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(diagonal, input + k * 32 + k);
        Reg::Compares<float, CMPMODE::EQ>(singular, diagonal, 0.f, mask);
        Reg::Select(diagonal, ones, diagonal, singular);
        Reg::LoadAlign(destination, upper + k * 32);
        Reg::Div(source, destination, diagonal, mask);
        Reg::Muls(negative, diagonal, -1.f, mask);
        Reg::Muls(residual, destination, 1.f, mask);
        Reg::MulAddDst(residual, source, negative, mask);
        Reg::Div(correction, residual, diagonal, mask);
        Reg::Add(source, source, correction, mask);
        Reg::StoreAlign(upper + k * 32, source, mask);
        for (int row = 0; row < k; ++row)
        {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(coefficient, input + k * 32 + row);
            Reg::Muls(coefficient, coefficient, -1.f, mask);
            Reg::LoadAlign(destination, upper + row * 32);
            Reg::MulAddDst(destination, source, coefficient, mask);
            Reg::StoreAlign(upper + row * 32, destination, mask);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
    Reg::LoadAlign(lanes, seed);
    Reg::Muls(lanes, lanes, uint32_t{32}, mask);
    for (int row = 0; row < n; ++row)
    {
        Reg::Adds(offsets, lanes, static_cast<uint32_t>(row), mask);
        Reg::LoadAlign(source, rows + row * 32);
        Reg::Scatter(columns, source, offsets, mask);
        Reg::LoadAlign(source, upper + row * 32);
        Reg::Scatter(columns + 32 * 32, source, offsets, mask);
    }
}

__aicore__ inline void LargeDiagonalVector(int n, GM_ADDR negative, GM_ADDR lower, GM_ADDR upper, GM_ADDR statuses,
                                           int ld, int firstTask, int taskStride)
{
    TPipe pipe;
    TBuf<TPosition::VECIN> inputBuffer;
    TBuf<TPosition::VECCALC> rowBuffer, seedBuffer;
    TBuf<TPosition::VECOUT> columnBuffer, infoBuffer;
    pipe.InitBuffer(inputBuffer, 32 * 32 * sizeof(float));
    pipe.InitBuffer(rowBuffer, 2 * 32 * 32 * sizeof(float));
    pipe.InitBuffer(columnBuffer, 2 * 32 * 32 * sizeof(float));
    pipe.InitBuffer(seedBuffer, 64 * sizeof(uint32_t));
    pipe.InitBuffer(infoBuffer, 32);
    auto input = inputBuffer.Get<float>(), rows = rowBuffer.Get<float>(), columns = columnBuffer.Get<float>();
    auto seed = seedBuffer.Get<uint32_t>();
    auto localInfo = infoBuffer.Get<int>();
    for (int lane = 0; lane < 64; ++lane) seed.SetValue(lane, lane);
    GlobalTensor<float> gn, gl, gu;
    GlobalTensor<int> gs;
    gn.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(negative));
    gl.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(lower));
    gu.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(upper));
    gs.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(statuses));
    for (int block = firstTask; block < (n + 31) / 32; block += taskStride)
    {
        const int start = block * 32, nb = n - start < 32 ? n - start : 32;
        const int padded = (nb + 7) / 8 * 8;
        const DataCopyExtParams copy{1, static_cast<uint32_t>(nb * sizeof(float)), 0, 0, 0};
        for (int col = 0; col < nb; ++col)
            DataCopyPad(input[col * 32], gn[static_cast<uint64_t>(start + col) * ld + start], copy,
                        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(padded - nb), 0.f});
        LargeFence<HardEvent::MTE2_V>();
        Muls(input, input, -1.f, nb * 32);
        Duplicate(rows, 0.f, 2 * 32 * 32);
        LargeFence<HardEvent::V_S>();
        int code = n + 1;
        for (int row = 0; row < nb; ++row)
        {
            if (input.GetValue(row * 32 + row) == 0.f && code == n + 1) code = start + row + 1;
            rows.SetValue(row * 32 + row, 1.f);
            rows.SetValue(32 * 32 + row * 32 + row, 1.f);
        }
        LargeFence<HardEvent::S_V>();
        DiagonalInverseVF(reinterpret_cast<__ubuf__ float *>(input.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ float *>(rows.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ float *>(columns.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ uint32_t *>(seed.GetPhyAddr()), nb);
        LargeFence<HardEvent::V_MTE3>();
        const DataCopyExtParams output{static_cast<uint16_t>(nb), static_cast<uint32_t>(nb * sizeof(float)),
                                       static_cast<uint32_t>((32 - padded) / 8),
                                       static_cast<uint32_t>((ld - nb) * sizeof(float)), 0};
        DataCopyPad(gl[static_cast<uint64_t>(start) * ld + start], columns, output);
        DataCopyPad(gu[static_cast<uint64_t>(start) * ld + start], columns[32 * 32], output);
        for (int entry = 0; entry < 8; ++entry) localInfo.SetValue(entry, entry == 0 ? code : 0);
        LargeFence<HardEvent::S_MTE3>();
        DataCopyPad(gs[block * 8], localInfo, DataCopyExtParams{1, 32, 0, 0, 0});
        LargeFence<HardEvent::MTE3_MTE2>();
        LargeFence<HardEvent::MTE3_V>();
        LargeFence<HardEvent::MTE3_S>();
    }
}
