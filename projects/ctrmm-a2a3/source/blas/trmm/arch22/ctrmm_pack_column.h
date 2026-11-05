/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
namespace {
__aicore__ inline void PackColumn(
    GlobalTensor<float>& src, GlobalTensor<float>& reGm, GlobalTensor<float>& imGm, PackBuffers& buffers,
    const CtrmmPackTiling& t, uint32_t col)
{
    auto re = buffers.re.Get<float>();
    auto im = buffers.im.Get<float>();
    const uint64_t outBase = static_cast<uint64_t>(col) * t.packedLd;
    Duplicate(re, 0.0f, CHUNK);
    PipeBarrier<PIPE_V>();
    VectorToStore();
    for (uint32_t row = 0; row < t.packedLd; row += CHUNK) {
        StoreVector(reGm, re, outBase + row, Clip(t.packedLd - row));
        StoreVector(imGm, re, outBase + row, Clip(t.packedLd - row));
    }
    cblas3::SyncAfterChunkStore();
    const bool upper = t.upper != t.transpose;
    const uint32_t begin = t.triangular && !upper ? col + t.unit : 0;
    const uint32_t end = t.triangular && upper ? col + 1 - t.unit : t.rows;
    const uint32_t step = t.transpose && t.sourceLd > UINT32_MAX / 8 ? 1 : CHUNK;
    for (uint32_t row = begin; row < end; row += step) {
        const uint32_t len = step < end - row ? step : end - row;
        LoadPackChunk(src, buffers, t, col, row, len);
        if (t.riskFlags)
            CheckPackRange(buffers, len);
        VectorToStore();
        StoreVector(reGm, re, outBase + row, len);
        StoreVector(imGm, im, outBase + row, len);
        cblas3::SyncAfterChunkStore();
    }
    if (t.unit) {
        Duplicate(re, 1.0f, 8);
        PipeBarrier<PIPE_V>();
        VectorToStore();
        StoreVector(reGm, re, outBase + col, 1);
        cblas3::SyncAfterChunkStore();
    }
}

__aicore__ inline void LoadVector(LocalTensor<float> dst, GlobalTensor<float>& src, uint64_t offset, uint32_t len)
{
    DataCopyExtParams copy{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    DataCopyPad(dst, src[offset], copy, pad);
}
} // namespace
