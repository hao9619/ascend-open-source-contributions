/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define ASCENDC_CUBE_ONLY
#include "adv_api/matmul/matmul_intf.h"
#include "cube_internal.h"
#include "kernel_operator.h"
#include "tiling/matrix/matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"

using namespace AscendC;
using CubeOperand = MatmulType<TPosition::GM, CubeFormat::ND, float>;

__global__ __aicore__ void lu950_cube_matmul(int m, int n, int k, int lda, int ldb, int ldc, GM_ADDR a, GM_ADDR b,
                                             GM_ADDR c, GM_ADDR workspace, AscendC::tiling::TCubeTiling tiling,
                                             int accumulate, int batch, uint64_t aStride, uint64_t bStride,
                                             uint64_t cStride)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    TPipe pipe;
    Matmul<CubeOperand, CubeOperand, CubeOperand, CubeOperand, CFG_NORM> mm;
    SetSysWorkspace(workspace);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    mm.SetHF32(false);
    GlobalTensor<float> ga, gb, gc;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
    gb.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
    gc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c));
    constexpr int tile = 64;
    const int columns = (n + tile - 1) / tile;
    const int tasksPerMatrix = (m + tile - 1) / tile * columns;
    const int tasks = tasksPerMatrix * batch;
    for (int task = GetBlockIdx(); task < tasks; task += GetBlockNum())
    {
        const int matrix = task / tasksPerMatrix, localTask = task % tasksPerMatrix;
        const int row = localTask / columns * tile;
        const int col = localTask % columns * tile;
        mm.SetOrgShape(m, ldb, lda, k, ldc);
        mm.SetSingleShape(m - row < tile ? m - row : tile, n - col < tile ? n - col : tile, k);
        mm.SetTensorA(ga[static_cast<uint64_t>(matrix) * aStride + static_cast<uint64_t>(row) * lda]);
        mm.SetTensorB(gb[static_cast<uint64_t>(matrix) * bStride + col]);
        mm.IterateAll(gc[static_cast<uint64_t>(matrix) * cStride + static_cast<uint64_t>(row) * ldc + col],
                      accumulate ? 1 : 0);
        mm.End();
    }
}

size_t Lu950CubeWorkspaceBytes()
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    return platform ? platform->GetLibApiWorkSpaceSize() : 0;
}

int Lu950CubeMatmulBatch(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc,
                         void *workspace, aclrtStream stream, bool accumulate, int batch, uint64_t matrixStride)
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform || !workspace || m < 1 || n < 1 || k < 1 || lda < k || ldb < n || ldc < n) return -1;
    matmul_tiling::MatmulApiTiling tiler(*platform);
    tiler.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
                   false);
    tiler.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
                   false);
    tiler.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tiler.SetShape(64, 64, k);
    tiler.SetOrgShape(m, ldb, lda, k);
    tiler.SetFixSplit(64, 64, -1);
    tiler.SetBufferSpace(-1, -1, -1);
    AscendC::tiling::TCubeTiling tiling{};
    if (tiler.GetTiling(tiling) < 0) return -2;
    const int tasks = (m + 63) / 64 * ((n + 63) / 64) * batch;
    unsigned blocks = platform->GetCoreNumAic();
    if (blocks > static_cast<unsigned>(tasks)) blocks = tasks;
    lu950_cube_matmul<<<blocks, nullptr, stream>>>(
        m, n, k, lda, ldb, ldc, reinterpret_cast<uint8_t *>(const_cast<float *>(a)),
        reinterpret_cast<uint8_t *>(const_cast<float *>(b)), reinterpret_cast<uint8_t *>(c),
        reinterpret_cast<uint8_t *>(workspace), tiling, accumulate ? 1 : 0, batch, matrixStride, matrixStride,
        matrixStride);
    return 0;
}

int Lu950CubeMatmul(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc,
                    void *workspace, aclrtStream stream, bool accumulate)
{
    return Lu950CubeMatmulBatch(m, n, k, a, lda, b, ldb, c, ldc, workspace, stream, accumulate, 1, 0);
}
