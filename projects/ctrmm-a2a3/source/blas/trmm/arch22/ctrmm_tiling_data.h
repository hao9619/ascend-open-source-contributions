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

#ifndef CTRMM_ORDER_CHUNK
#define CTRMM_ORDER_CHUNK 2048
#endif
#include <cstdint>
#define CTRMM_ORDERED 1
#define CTRMM_ORDERED_COMPENSATED 1
#define CTRMM_ORDERED_PAIRED 1
#define CTRMM_ORDERED_EDGE 1

struct CtrmmPackTiling {
    uint32_t rows, cols, sourceLd, packedLd;
    uint32_t triangular, upper, transpose, conjugate, unit;
    uint64_t riskFlags = 0;
};

struct CtrmmCombineTiling {
    uint32_t m, n, tempLd, ldc;
    float alphaReal, alphaImag;
    uint32_t fourProduct = 0;
};

#include "common/helper/complex_blas3_tiling_data.h"

struct CtrmmGemmTiling {
    CBlas3GemmTilingData base;
    uint32_t sideLeft;
    uint32_t upper;
    uint32_t fourProduct = 0;
};

struct CtrmmOrderedTiling {
    uint32_t m, n, d, aLd, bLd, ldc;
    uint32_t sideLeft, upper, trans, unit;
    float alphaReal, alphaImag;
    uint32_t guardBlocks = 0;
    uint32_t rawALd = 0, rawBLd = 0, conjugate = 0;
};
