/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Independent partial-pivot LU per SIMD lane; packed trailing columns use
// otherwise idle vector lanes when fewer than 64 matrices fit in UB.
__simd_vf__ inline void BatchFactorStepVF(__ubuf__ float* factors, __ubuf__ int32_t* pivots, __ubuf__ int32_t* infos,
                                          __ubuf__ uint32_t* seed, int n, int group, int offset, uint32_t active, int k)
{
    uint32_t remaining = active;
    auto mask = Reg::UpdateMask<float>(remaining);
    Reg::RegTensor<float> value, other, largest, magnitude, diagonal, reciprocal;
    Reg::RegTensor<float> ones, negative, residual, correction, coefficient;
    Reg::RegTensor<uint32_t> lanes, index, target, pivot, rowNumber;
    Reg::RegTensor<int32_t> code, singular, next;
    Reg::MaskReg condition, available;
    Reg::LoadAlign(lanes, seed);
    Reg::LoadAlign(code, infos + offset);
    Reg::Duplicate(ones, 1.f, mask);
    const int columnsPerVector = group < 64 ? 64 / group : 1;
    {
        Reg::LoadAlign(value, factors + (k * n + k) * group + offset);
        Reg::Abs(largest, value, mask);
        Reg::Duplicate(pivot, static_cast<uint32_t>(k), mask);
        for (int row = k + 1; row < n; ++row)
        {
            Reg::LoadAlign(value, factors + (row * n + k) * group + offset);
            Reg::Abs(magnitude, value, mask);
            Reg::Compare<float, CMPMODE::GT>(condition, magnitude, largest, mask);
            Reg::Select(largest, magnitude, largest, condition);
            Reg::Duplicate(rowNumber, static_cast<uint32_t>(row), mask);
            Reg::Select(pivot, rowNumber, pivot, condition);
        }
        Reg::Compares<float, CMPMODE::EQ>(condition, largest, 0.f, mask);
        Reg::Compares<int32_t, CMPMODE::EQ>(available, code, 0, mask);
        Reg::Duplicate(singular, k + 1, mask);
        Reg::Select(next, singular, code, condition);
        Reg::Select(code, next, code, available);
        Reg::Muls(target, pivot, static_cast<uint32_t>(n * group), mask);
        Reg::Add(target, target, lanes, mask);
        for (int col = 0; col < n; ++col)
        {
            Reg::Adds(index, target, static_cast<uint32_t>(col * group), mask);
            Reg::LoadAlign(value, factors + (k * n + col) * group + offset);
            Reg::Gather(other, factors, index, mask);
            Reg::StoreAlign(factors + (k * n + col) * group + offset, other, mask);
            Reg::Scatter(factors, value, index, mask);
        }
        Reg::Adds(pivot, pivot, uint32_t{1}, mask);
        Reg::Muls(index, lanes, static_cast<uint32_t>(n), mask);
        Reg::Adds(index, index, static_cast<uint32_t>(k), mask);
        Reg::Scatter(reinterpret_cast<__ubuf__ uint32_t*>(pivots), pivot, index, mask);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::LoadAlign(diagonal, factors + (k * n + k) * group + offset);
        Reg::Compares<float, CMPMODE::EQ>(condition, diagonal, 0.f, mask);
        Reg::Select(diagonal, ones, diagonal, condition);
        Reg::Div(reciprocal, ones, diagonal, mask);
        Reg::Muls(negative, diagonal, -1.f, mask);
        Reg::Muls(residual, ones, 1.f, mask);
        Reg::MulAddDst(residual, reciprocal, negative, mask);
        Reg::Div(correction, residual, diagonal, mask);
        Reg::Add(reciprocal, reciprocal, correction, mask);
        for (int row = k + 1; row < n; ++row)
        {
            Reg::LoadAlign(value, factors + (row * n + k) * group + offset);
            Reg::Mul(value, value, reciprocal, mask);
            Reg::StoreAlign(factors + (row * n + k) * group + offset, value, mask);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        for (int row = k + 1; row < n; ++row)
            for (int col = k + 1; col < n; col += columnsPerVector)
            {
                uint32_t columnsActive = static_cast<uint32_t>(
                    (n - col < columnsPerVector ? n - col : columnsPerVector) * (group < 64 ? group : active));
                auto columnsMask = Reg::UpdateMask<float>(columnsActive);
                Reg::Adds(index, lanes, static_cast<uint32_t>((row * n + k) * group), columnsMask);
                Reg::Gather(coefficient, factors, index, columnsMask);
                Reg::Muls(coefficient, coefficient, -1.f, columnsMask);
                Reg::LoadAlign(other, factors + (k * n + col) * group + offset);
                Reg::LoadAlign(value, factors + (row * n + col) * group + offset);
                if (n <= 4)
                {
                    // Small LAPACK SGER rounds the product before adding it.
                    Reg::Mul(other, other, coefficient, columnsMask);
                    Reg::Add(value, value, other, columnsMask);
                }
                else
                    Reg::MulAddDst(value, other, coefficient, columnsMask);
                Reg::StoreAlign(factors + (row * n + col) * group + offset, value, columnsMask);
            }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
    Reg::StoreAlign(infos + offset, code, mask);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

#include "batch_factor_crout10.hpp"

// Convert the factor signs with one VF rather than a separate Vector API call per diagonal.
__simd_vf__ inline void BatchFactorSignVF(__ubuf__ float* factors, int n, int group)
{
    Reg::RegTensor<float> value;
    const int elements = n * n * group;
    uint32_t remaining = static_cast<uint32_t>(elements);
    for (int base = 0; base < elements; base += 64)
    {
        auto mask = Reg::UpdateMask<float>(remaining);
        Reg::LoadAlign(value, factors + base);
        Reg::Muls(value, value, -1.f, mask);
        Reg::StoreAlign(factors + base, value, mask);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (int k = 0; k < n; ++k)
        for (int offset = 0; offset < group; offset += 64)
        {
            uint32_t active = static_cast<uint32_t>(group - offset < 64 ? group - offset : 64);
            auto mask = Reg::UpdateMask<float>(active);
            Reg::LoadAlign(value, factors + (k * n + k) * group + offset);
            Reg::Muls(value, value, -1.f, mask);
            Reg::StoreAlign(factors + (k * n + k) * group + offset, value, mask);
        }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__aicore__ inline void BatchFusedFactor(int n, LocalTensor<float> factors, LocalTensor<int> pivots,
                                        LocalTensor<int> infos, LocalTensor<uint32_t> seed, int group,
                                        bool seedReady = false)
{
    if (n >= 10 && n <= 32)
    {
        for (int offset = 0; offset < group; offset += 64)
        {
            if (!seedReady)
                for (int lane = 0; lane < 64; ++lane) seed.SetValue(lane, group < 64 ? lane % group : lane + offset);
            BatchFence<HardEvent::S_V>();
            BatchFactorCroutDispatch(n, reinterpret_cast<__ubuf__ float*>(factors.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ int32_t*>(pivots.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ int32_t*>(infos.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ uint32_t*>(seed.GetPhyAddr()), group, offset,
                                     static_cast<uint32_t>(group - offset < 64 ? group - offset : 64));
            BatchFence<HardEvent::V_S>();
        }
    }
    else
        for (int k = 0; k < n; ++k)
            for (int offset = 0; offset < group; offset += 64)
            {
                for (int lane = 0; lane < 64; ++lane) seed.SetValue(lane, group < 64 ? lane % group : lane + offset);
                BatchFence<HardEvent::S_V>();
                BatchFactorStepVF(reinterpret_cast<__ubuf__ float*>(factors.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ int32_t*>(pivots.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ int32_t*>(infos.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ uint32_t*>(seed.GetPhyAddr()), n, group, offset,
                                  static_cast<uint32_t>(group - offset < 64 ? group - offset : 64), k);
                BatchFence<HardEvent::V_S>();
            }
    if (n >= 18)
    {
        BatchFactorSignVF(reinterpret_cast<__ubuf__ float*>(factors.GetPhyAddr()), n, group);
        return;
    }
    Muls(factors, factors, -1.f, n * n * group);
    PipeBarrier<PIPE_V>();
    for (int k = 0; k < n; ++k)
    {
        Muls(factors[(k * n + k) * group], factors[(k * n + k) * group], -1.f, group);
        PipeBarrier<PIPE_V>();
    }
}
