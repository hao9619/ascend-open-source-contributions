/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Error-free products and TwoSum retain the low part of each FP32 dot product.
__simd_vf__ inline void BatchInverseResidualVF(__ubuf__ float* residual, __ubuf__ float* inverse, __ubuf__ float* input,
                                               __ubuf__ uint32_t* seed, int n, int group, int slot, int stride,
                                               int offset, uint32_t active)
{
    auto mask = Reg::UpdateMask<float>(active);
    Reg::RegTensor<uint32_t> lanes, base, index;
    Reg::RegTensor<float> x, a, product, error, sum, low, next, v, u, e, value;
    Reg::LoadAlign(lanes, seed + offset);
    Reg::Muls(base, lanes, static_cast<uint32_t>(slot), mask);
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col)
        {
            Reg::Duplicate(sum, 0.f, mask);
            Reg::Duplicate(low, 0.f, mask);
            for (int k = 0; k < n; ++k)
            {
                Reg::LoadAlign(x, inverse + (k * n + row) * group + offset);
                Reg::Adds(index, base, static_cast<uint32_t>(col * stride + k), mask);
                Reg::Gather(a, input, index, mask);
                Reg::Mul(product, x, a, mask);
                Reg::Muls(error, product, -1.f, mask);
                Reg::MulAddDst(error, x, a, mask);
                Reg::Add(next, sum, product, mask);
                Reg::Sub(v, next, sum, mask);
                Reg::Sub(u, next, v, mask);
                Reg::Sub(e, sum, u, mask);
                Reg::Sub(u, product, v, mask);
                Reg::Add(e, e, u, mask);
                Reg::Add(e, e, error, mask);
                Reg::Add(low, low, e, mask);
                Reg::Muls(sum, next, 1.f, mask);
            }
            Reg::Muls(value, sum, -1.f, mask);
            if (row == col) Reg::Adds(value, value, 1.f, mask);
            Reg::Sub(value, value, low, mask);
            Reg::StoreAlign(residual + (col * n + row) * group + offset, value, mask);
        }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__simd_vf__ inline void BatchInverseCorrectionVF(__ubuf__ float* output, __ubuf__ float* residual,
                                                 __ubuf__ float* inverse, int n, int group, int offset, uint32_t active)
{
    auto mask = Reg::UpdateMask<float>(active);
    Reg::RegTensor<float> sum, r, x;
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col)
        {
            Reg::Duplicate(sum, 0.f, mask);
            for (int k = 0; k < n; ++k)
            {
                Reg::LoadAlign(r, residual + (k * n + row) * group + offset);
                Reg::LoadAlign(x, inverse + (col * n + k) * group + offset);
                Reg::MulAddDst(sum, r, x, mask);
            }
            Reg::LoadAlign(x, inverse + (col * n + row) * group + offset);
            Reg::Add(sum, x, sum, mask);
            Reg::StoreAlign(output + (col * n + row) * group + offset, sum, mask);
        }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__aicore__ inline void BatchRefineRawInverse(int n, LocalTensor<float> inverse, LocalTensor<float> residual,
                                             LocalTensor<float> stage, LocalTensor<int32_t> indices, int group,
                                             int padded, bool dense)
{
    for (int offset = 0; offset < group; offset += 64)
        BatchInverseResidualVF(reinterpret_cast<__ubuf__ float*>(residual.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ float*>(inverse.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ float*>(stage.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ uint32_t*>(indices.GetPhyAddr()), n, group,
                               dense ? n * n : n * padded, dense ? n : padded, offset,
                               static_cast<uint32_t>(group - offset < 64 ? group - offset : 64));
    PipeBarrier<PIPE_V>();
    // Finish every input read before reusing the AoS stage as SoA output.
    for (int offset = 0; offset < group; offset += 64)
        BatchInverseCorrectionVF(reinterpret_cast<__ubuf__ float*>(stage.GetPhyAddr()),
                                 reinterpret_cast<__ubuf__ float*>(residual.GetPhyAddr()),
                                 reinterpret_cast<__ubuf__ float*>(inverse.GetPhyAddr()), n, group, offset,
                                 static_cast<uint32_t>(group - offset < 64 ? group - offset : 64));
    PipeBarrier<PIPE_V>();
    Muls(inverse, stage, 1.f, n * n * group);
    PipeBarrier<PIPE_V>();
}
