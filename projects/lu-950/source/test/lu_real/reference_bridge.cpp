/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Test-only bridge to the caller-selected FP32 LAPACK; never linked into the solver.
#include <algorithm>
#include <cstddef>
#include <vector>

using FactorFunction = void (*)(const int *, const int *, float *, const int *, int *, int *);
using InverseFunction = void (*)(const int *, float *, const int *, const int *, float *, const int *, int *);

extern "C" int lu950_reference_factor(int n, int batch, float *matrices, int ld, int *pivots, int *info, void *function)
{
    if (n < 1 || batch < 1 || ld < n || !matrices || !pivots || !info || !function) return 1;
    const auto factor = reinterpret_cast<FactorFunction>(function);
    for (int index = 0; index < batch; ++index)
        factor(&n, &n, matrices + static_cast<size_t>(index) * ld * n, &ld, pivots + static_cast<size_t>(index) * n,
               info + index);
    return 0;
}

extern "C" int lu950_reference_inverse(int n, int batch, const float *factors, int lda, const int *pivots,
                                       float *inverse, int ldc, int *info, void *function)
{
    if (n < 1 || batch < 1 || lda < n || ldc < n || !factors || !pivots || !inverse || !info || !function) return 1;
    const auto invert = reinterpret_cast<InverseFunction>(function);
    const int lwork = std::max(1, n * 64);
    std::vector<float> work(lwork);
    for (int index = 0; index < batch; ++index)
    {
        const auto *source = factors + static_cast<size_t>(index) * lda * n;
        auto *target = inverse + static_cast<size_t>(index) * ldc * n;
        for (int column = 0; column < n; ++column)
            std::copy_n(source + static_cast<size_t>(column) * lda, n, target + static_cast<size_t>(column) * ldc);
        invert(&n, target, &ldc, pivots + static_cast<size_t>(index) * n, work.data(), &lwork, info + index);
    }
    return 0;
}
