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
#include <vector>
#include "cblas_compat.h"
#include "ctrmm_param.h"

inline std::vector<aclblasComplex> CtrmmGolden(
    const CtrmmParam& p, const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& b)
{
    std::vector<aclblasComplex> c(static_cast<size_t>(p.ldc) * p.n, {42.0f, -42.0f});
    const bool zero = p.alpha.real == 0 && p.alpha.imag == 0;
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.m; ++row) {
            c[static_cast<size_t>(col) * p.ldc + row] =
                zero ? aclblasComplex{0, 0} : b[static_cast<size_t>(col) * p.ldb + row];
        }
    }
    if (!zero) {
        cblas_ctrmm(
            CblasColMajor, p.side == ACLBLAS_SIDE_LEFT ? CblasLeft : CblasRight, ToCblasUplo(p.uplo),
            ToCblasOp(p.trans), ToCblasDiag(p.diag), p.m, p.n, &p.alpha, a.data(), p.lda, c.data(), p.ldc);
    }
    return c;
}
