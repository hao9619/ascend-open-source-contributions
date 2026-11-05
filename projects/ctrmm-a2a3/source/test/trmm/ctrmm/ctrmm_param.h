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
#include <algorithm>
#include "csv_loader.h"
#include "fill.h"

struct CtrmmParam : BlasTestParamBase {
    aclblasSideMode_t side;
    aclblasFillMode_t uplo;
    aclblasOperation_t trans;
    aclblasDiagType_t diag;
    int m, n, lda, ldb, ldc;
    aclblasComplex alpha;
    bool nullAlpha, nullA, nullB, nullC, inPlace;
    std::string aFill, bFill;

    explicit CtrmmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        side = parseSideMode(ReadMap(map, "side", "LEFT"));
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        diag = parseDiagType(ReadMap(map, "diag", "NON_UNIT"));
        m = parseInt(ReadMap(map, "m"));
        n = parseInt(ReadMap(map, "n"));
        const int d = side == ACLBLAS_SIDE_LEFT ? m : n;
        lda = parseInt(ReadMap(map, "lda"), std::max(1, d));
        ldb = parseInt(ReadMap(map, "ldb"), std::max(1, m));
        ldc = parseInt(ReadMap(map, "ldc"), std::max(1, m));
        const auto real = ReadMap(map, "alpha_real", "1");
        nullAlpha = real == "null" || real == "nullptr";
        alpha = {parseFloat(real, 1), parseFloat(ReadMap(map, "alpha_imag", "0"))};
        nullA = ReadMap(map, "nullA") == "1";
        nullB = ReadMap(map, "nullB") == "1";
        nullC = ReadMap(map, "nullC") == "1";
        inPlace = ReadMap(map, "in_place") == "1";
        aFill = ReadMap(map, "a_fill", "RANDOM_NORM_5_5");
        bFill = ReadMap(map, "b_fill", "RANDOM_NORM_5_5");
    }
};
