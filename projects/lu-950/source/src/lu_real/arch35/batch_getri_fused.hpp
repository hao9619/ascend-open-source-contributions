/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Validate pivot arrays and transpose matrix groups into SIMD lane layout.
__simd_vf__ inline void BatchPrepareVF(__ubuf__ float* inverse, __ubuf__ float* factors, __ubuf__ float* stage,
                                       __ubuf__ int32_t* pivots, __ubuf__ int32_t* infos, __ubuf__ int32_t* permutation,
                                       __ubuf__ uint32_t* seed, int n, int group, int stride, int slot, int offset,
                                       uint32_t active, bool pivoting, bool raw)
{
    auto mask = Reg::UpdateMask<float>(active);
    Reg::RegTensor<uint32_t> lanes, index, base, pivot, fallback, safe;
    Reg::RegTensor<int32_t> target, code, invalid;
    Reg::RegTensor<float> value;
    Reg::MaskReg available, bad;
    Reg::LoadAlign(lanes, seed + offset);
    Reg::LoadAlign(code, infos + offset);
    Reg::Duplicate(invalid, -4, mask);
    if (pivoting)
        for (int k = 0; k < n; ++k)
        {
            Reg::Muls(index, lanes, static_cast<uint32_t>(n), mask);
            Reg::Adds(index, index, static_cast<uint32_t>(k), mask);
            Reg::Gather(pivot, reinterpret_cast<__ubuf__ uint32_t*>(pivots), index, mask);
            Reg::Duplicate(fallback, static_cast<uint32_t>(k + 1), mask);
            Reg::Compares<uint32_t, CMPMODE::LT>(bad, pivot, static_cast<uint32_t>(k + 1), mask);
            Reg::Select(safe, fallback, pivot, bad);
            Reg::Compares<int32_t, CMPMODE::EQ>(available, code, 0, mask);
            Reg::Select(target, invalid, code, bad);
            Reg::Select(code, target, code, available);
            Reg::Compares<uint32_t, CMPMODE::GT>(bad, pivot, static_cast<uint32_t>(n), mask);
            Reg::Select(safe, fallback, safe, bad);
            Reg::Compares<int32_t, CMPMODE::EQ>(available, code, 0, mask);
            Reg::Select(target, invalid, code, bad);
            Reg::Select(code, target, code, available);
        }
    Reg::StoreAlign(infos + offset, code, mask);
    Reg::Muls(base, lanes, static_cast<uint32_t>(slot), mask);
    for (int row = 0; row < n; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            Reg::Adds(index, base, static_cast<uint32_t>(col * stride + row), mask);
            Reg::Gather(value, stage, index, mask);
            if (row != col && !raw) Reg::Muls(value, value, -1.f, mask);
            Reg::StoreAlign(factors + (row * n + col) * group + offset, value, mask);
        }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__aicore__ inline bool BatchFusedPrepare(int n, LocalTensor<float> inverse, LocalTensor<float> factors,
                                         LocalTensor<float> stage, LocalTensor<int> pivots, LocalTensor<int> infos,
                                         LocalTensor<float> scratch, LocalTensor<int32_t> indices, int group,
                                         int padded, bool dense, bool pivoting, bool raw, int stagingSlot = 0)
{
    if (n == 2) return false;
    for (int offset = 0; offset < group; offset += 64)
        BatchPrepareVF(reinterpret_cast<__ubuf__ float*>(inverse.GetPhyAddr()),
                       reinterpret_cast<__ubuf__ float*>(factors.GetPhyAddr()),
                       reinterpret_cast<__ubuf__ float*>(stage.GetPhyAddr()),
                       reinterpret_cast<__ubuf__ int32_t*>(pivots.GetPhyAddr()),
                       reinterpret_cast<__ubuf__ int32_t*>(infos.GetPhyAddr()),
                       reinterpret_cast<__ubuf__ int32_t*>(scratch.GetPhyAddr()),
                       reinterpret_cast<__ubuf__ uint32_t*>(indices.GetPhyAddr()), n, group, dense ? n : padded,
                       dense ? n * n : (stagingSlot != 0 ? stagingSlot : n * padded), offset,
                       static_cast<uint32_t>(group - offset < 64 ? group - offset : 64), pivoting, raw);
    return true;
}

#include "batch_getri_lapack.hpp"
#include "batch_getri_n4.hpp"
#include "batch_inverse_refine.hpp"
#ifdef LU950_RAW_BATCH
#include "batch_factor_vector.hpp"
#endif

// Pack adjacent matrix entries into otherwise unused SIMD lanes.
__simd_vf__ inline void BatchPackOutputVF(__ubuf__ float* stage, __ubuf__ float* inverse, __ubuf__ uint32_t* seed,
                                          int n, int group, int stride, int offset, uint32_t active)
{
    Reg::RegTensor<uint32_t> base, index;
    Reg::RegTensor<float> value;
    Reg::LoadAlign(base, seed);
    const int packedRows = group < 64 ? 64 / group : 1;
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; row += packedRows)
        {
            uint32_t remaining =
                static_cast<uint32_t>(group < 64 ? (n - row < packedRows ? n - row : packedRows) * group : active);
            auto mask = Reg::UpdateMask<float>(remaining);
            Reg::LoadAlign(value, inverse + (col * n + row) * group + offset);
            Reg::Adds(index, base, static_cast<uint32_t>(col * stride + row), mask);
            Reg::Scatter(stage, value, index, mask);
        }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__aicore__ inline bool BatchFusedPack(int n, LocalTensor<float> stage, LocalTensor<float> inverse,
                                      LocalTensor<uint32_t> seed, int group, int slot, int stride,
                                      bool seedReady = false)
{
    for (int offset = 0; offset < group; offset += 64)
    {
        if (!seedReady)
            for (int lane = 0; lane < 64; ++lane)
                seed.SetValue(lane, group < 64 ? (lane % group) * slot + lane / group : (lane + offset) * slot);
        BatchFence<HardEvent::S_V>();
        BatchPackOutputVF(reinterpret_cast<__ubuf__ float*>(stage.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ float*>(inverse.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ uint32_t*>(seed.GetPhyAddr()), n, group, stride, offset,
                          static_cast<uint32_t>(group - offset < 64 ? group - offset : 64));
        BatchFence<HardEvent::V_S>();
    }
    return true;
}

// Exact row count reduces live row sums for partially occupied SIMD row groups.
template <int N>
__aicore__ inline void BatchLapackExactPacked(__ubuf__ float* inverse, __ubuf__ float* factors,
                                              __ubuf__ int32_t* pivots, __ubuf__ uint32_t* seed, int group, int offset,
                                              uint32_t active, bool pivoting)
{
    if (group > 32)
        BatchLapackVF<N, 1, N>(inverse, factors, pivots, seed, N, group, offset, active, pivoting);
    else if (group == 8)
        BatchLapackVF<N, 8, N>(inverse, factors, pivots, seed, N, group, offset, active, pivoting);
    else if (group == 16)
        BatchLapackVF<N, 4, N>(inverse, factors, pivots, seed, N, group, offset, active, pivoting);
    else
        BatchLapackVF<N, 2, N>(inverse, factors, pivots, seed, N, group, offset, active, pivoting);
}

template <int MaxN>
__aicore__ inline void BatchLapackDispatch(__ubuf__ float* inverse, __ubuf__ float* factors, __ubuf__ int32_t* pivots,
                                           __ubuf__ uint32_t* seed, int n, int group, int offset, uint32_t active,
                                           bool pivoting)
{
    if constexpr (MaxN == 8 || MaxN == 16)
    {
        if (n == MaxN && group >= 64)
        {
            BatchLapackVF<MaxN, 1, MaxN>(inverse, factors, pivots, seed, n, group, offset, active, pivoting);
            return;
        }
    }
    if constexpr (MaxN == 16)
    {
        if (group >= 64)
        {
            switch (n)
            {
                case 11:
                    BatchLapackVF<11, 1, 11>(inverse, factors, pivots, seed, n, group, offset, active, pivoting);
                    return;
                case 12:
                    BatchLapackVF<12, 1, 12>(inverse, factors, pivots, seed, n, group, offset, active, pivoting);
                    return;
                case 13:
                    BatchLapackVF<13, 1, 13>(inverse, factors, pivots, seed, n, group, offset, active, pivoting);
                    return;
                case 14:
                    BatchLapackVF<14, 1, 14>(inverse, factors, pivots, seed, n, group, offset, active, pivoting);
                    return;
                case 15:
                    BatchLapackVF<15, 1, 15>(inverse, factors, pivots, seed, n, group, offset, active, pivoting);
                    return;
            }
        }
    }
    if constexpr (MaxN == 32)
    {
        if (n == 32 && group == 16)
        {
            BatchLapackVF<32, 4, 32>(inverse, factors, pivots, seed, n, group, offset, active, pivoting);
            return;
        }
    }
    if constexpr (MaxN == 32)
    {
        switch (n)
        {
            case 17:
                BatchLapackExactPacked<17>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 18:
                BatchLapackExactPacked<18>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 19:
                BatchLapackExactPacked<19>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 20:
                BatchLapackExactPacked<20>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 21:
                BatchLapackExactPacked<21>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 22:
                BatchLapackExactPacked<22>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 23:
                BatchLapackExactPacked<23>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 24:
                BatchLapackExactPacked<24>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 25:
                BatchLapackExactPacked<25>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 26:
                BatchLapackExactPacked<26>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 27:
                BatchLapackExactPacked<27>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 28:
                BatchLapackExactPacked<28>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 29:
                BatchLapackExactPacked<29>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 30:
                BatchLapackExactPacked<30>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 31:
                BatchLapackExactPacked<31>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
            case 32:
                BatchLapackExactPacked<32>(inverse, factors, pivots, seed, group, offset, active, pivoting);
                return;
        }
    }
    if constexpr (MaxN <= 16)
    {
        if (group < 64)
            BatchLapackVF<MaxN, 2>(inverse, factors, pivots, seed, n, group, offset, active, pivoting);
        else
            BatchLapackVF<MaxN, 1>(inverse, factors, pivots, seed, n, group, offset, active, pivoting);
    }
    else
    {
        if (group == 8)
            BatchLapackVF<MaxN, 8>(inverse, factors, pivots, seed, n, group, offset, active, pivoting);
        else if (group == 16)
            BatchLapackVF<MaxN, 4>(inverse, factors, pivots, seed, n, group, offset, active, pivoting);
        else
            BatchLapackVF<MaxN, 2>(inverse, factors, pivots, seed, n, group, offset, active, pivoting);
    }
}

__aicore__ inline bool BatchFusedTriangular(int n, LocalTensor<float> inverse, LocalTensor<float> factors,
                                            LocalTensor<int> pivots, LocalTensor<uint32_t> seed, int group,
                                            bool pivoting, bool seedReady = false)
{
    if (n == 2) return false;
    for (int offset = 0; offset < group; offset += 64)
    {
        // Raw Crout leaves the identical lane seed intact for groups of at most one vector.
        if (!seedReady)
            for (int lane = 0; lane < 64; ++lane) seed.SetValue(lane, group < 64 ? lane % group : lane + offset);
        BatchFence<HardEvent::S_V>();
        auto* out = reinterpret_cast<__ubuf__ float*>(inverse.GetPhyAddr());
        auto* lu = reinterpret_cast<__ubuf__ float*>(factors.GetPhyAddr());
        auto* ip = reinterpret_cast<__ubuf__ int32_t*>(pivots.GetPhyAddr());
        auto* indices = reinterpret_cast<__ubuf__ uint32_t*>(seed.GetPhyAddr());
        const uint32_t active = static_cast<uint32_t>(group - offset < 64 ? group - offset : 64);
        if (n == 4)
            BatchLapackN4VF(out, lu, ip, indices, group, offset, active, pivoting);
        else if (n <= 4)
            BatchLapackDispatch<4>(out, lu, ip, indices, n, group, offset, active, pivoting);
        else if (n <= 8)
            BatchLapackDispatch<8>(out, lu, ip, indices, n, group, offset, active, pivoting);
        else if (n <= 16)
            BatchLapackDispatch<16>(out, lu, ip, indices, n, group, offset, active, pivoting);
        else
            BatchLapackDispatch<32>(out, lu, ip, indices, n, group, offset, active, pivoting);
        BatchFence<HardEvent::V_S>();
    }
    return true;
}
