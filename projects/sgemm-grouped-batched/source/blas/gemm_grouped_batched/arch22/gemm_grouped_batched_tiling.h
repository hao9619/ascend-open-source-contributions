/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#pragma once
#include <cstdint>

// Passed by value at launch: no temporary metadata allocation survives the API
// call.
struct GemmGrouped22Params {
    int m, n, k, lda, ldb, ldc;
    int transA, transB;
    float alpha, beta;
};

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

void GemmGrouped22Launch(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmGrouped22Params params);

constexpr uint32_t GEMM_GROUPED22_BATCH_CAPACITY = 64;
constexpr uint32_t GEMM_GROUPED22_POINTER_CAPACITY = 96;
struct GemmGrouped22Batch {
    GemmGrouped22Params group;
    uint32_t count;
    int outputLd;
    bool combine;
    uint64_t flags;
    uint32_t scanWorkers;
    uint32_t tileRows;
    uint32_t tileCols;
    uint32_t matrixBase;
    uint32_t cubeTaskOffset;
    uint32_t groupCount;
    GemmGrouped22Params extraGroups[7];
    uint32_t groupEnds[8];
    uint32_t groupWorkers[8];
    uint32_t groupRows[8];
    uint32_t groupCols[8];
    uint64_t a[GEMM_GROUPED22_POINTER_CAPACITY];
    uint64_t b[GEMM_GROUPED22_POINTER_CAPACITY];
    uint64_t c[GEMM_GROUPED22_POINTER_CAPACITY];
    uint64_t output[GEMM_GROUPED22_POINTER_CAPACITY];
};
static_assert(sizeof(GemmGrouped22Batch) < 4096, "Keep launch arguments bounded");
void GemmGrouped22Cube(uint32_t blocks, uint32_t vectors, void* stream, GemmGrouped22Batch batch);

void GemmGrouped22Ordered(uint32_t vectors, void* stream, GemmGrouped22Batch batch);

void GemmGrouped22ScalarBatch(uint32_t blocks, void* stream, GemmGrouped22Batch batch);
