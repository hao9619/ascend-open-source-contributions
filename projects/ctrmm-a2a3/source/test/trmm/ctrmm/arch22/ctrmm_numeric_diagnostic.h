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
#include <array>
#include <iomanip>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include "ctrmm_param.h"

namespace ctrmm_diagnostic {
using Wide = std::complex<long double>;

inline bool ReadTerm(
    const CtrmmParam& p, const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& b, int row, int col,
    int k, Wide& av, Wide& bv)
{
    int ar = p.side == ACLBLAS_SIDE_LEFT ? row : k;
    int ac = p.side == ACLBLAS_SIDE_LEFT ? k : col;
    if (p.trans != ACLBLAS_OP_N)
        std::swap(ar, ac);
    if (p.uplo == ACLBLAS_UPPER ? ar > ac : ar < ac)
        return false;
    av = Wide(1, 0);
    if (p.diag != ACLBLAS_UNIT || ar != ac) {
        const auto value = a[static_cast<size_t>(ac) * p.lda + ar];
        av = Wide(value.real, value.imag);
        if (p.trans == ACLBLAS_OP_C)
            av = std::conj(av);
    }
    const size_t bi =
        p.side == ACLBLAS_SIDE_LEFT ? static_cast<size_t>(col) * p.ldb + k : static_cast<size_t>(k) * p.ldb + row;
    bv = Wide(b[bi].real, b[bi].imag);
    return true;
}

// Host numerical model only: not a claim about Cube internal reduction order.
// Four FP32 FMA dot products per block, followed by FP64 block accumulation.
inline Wide BlockModel(const std::vector<Wide>& av, const std::vector<Wide>& bv, int block)
{
    double real = 0, imag = 0;
    for (size_t start = 0; start < av.size(); start += block) {
        std::array<float, 4> sums{};
        const size_t end = std::min(av.size(), start + block);
        for (size_t k = start; k < end; ++k) {
            const float ar = av[k].real(), ai = av[k].imag();
            const float br = bv[k].real(), bi = bv[k].imag();
            sums[0] = std::fma(ar, br, sums[0]);
            sums[1] = std::fma(ai, bi, sums[1]);
            sums[2] = std::fma(ar, bi, sums[2]);
            sums[3] = std::fma(ai, br, sums[3]);
        }
        real += static_cast<double>(sums[0]) - sums[1];
        imag += static_cast<double>(sums[2]) + sums[3];
    }
    return Wide(real, imag);
}

inline void PrintPoint(
    const CtrmmParam& p, const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& b,
    const std::vector<aclblasComplex>& expected, size_t index)
{
    const int row = index % p.ldc, col = index / p.ldc;
    const int d = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
    std::vector<Wide> av(d), bv(d);
    Wide exact(0, 0);
    for (int k = 0; k < d; ++k) {
        if (ReadTerm(p, a, b, row, col, k, av[k], bv[k]))
            exact += av[k] * bv[k];
    }
    const Wide alpha(p.alpha.real, p.alpha.imag);
    exact *= alpha;
    const auto golden = expected[index];
    const auto emit = [&](const char* model, int block, Wide value) {
        // The operator output is FP32 even when the internal reference is wider.
        const float re = value.real(), im = value.imag();
        std::cout << std::setprecision(17) << "NUMERIC_MODEL " << p.caseName << " row=" << row << " col=" << col
                  << " model=" << model << " block=" << block
                  << " real_error=" << std::abs(static_cast<double>(re) - golden.real)
                  << " imag_error=" << std::abs(static_cast<double>(im) - golden.imag)
                  << " real_exact_error=" << static_cast<double>(std::abs(value.real() - exact.real()))
                  << " imag_exact_error=" << static_cast<double>(std::abs(value.imag() - exact.imag())) << '\n';
    };
    emit("wide", 0, exact);
    for (const int block : {16, 32, 64, 128, 256, 512})
        emit("block_fma", block, BlockModel(av, bv, block) * alpha);
}

inline void Run(
    const CtrmmParam& p, const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& b,
    const std::vector<aclblasComplex>& expected, const size_t* worst)
{
    if (std::getenv("CTRMM_NUMERIC_MODEL") == nullptr)
        return;
    std::cout << "NUMERIC_MODEL_FORMAT long_double_digits=" << std::numeric_limits<long double>::digits << '\n';
    std::vector<size_t> points{worst[0], worst[1]};
    std::mt19937 engine(p.randomSeed);
    for (int i = 0; i < 64; ++i) {
        const size_t row = engine() % p.m, col = engine() % p.n;
        points.push_back(col * p.ldc + row);
    }
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    for (const size_t index : points)
        PrintPoint(p, a, b, expected, index);
}
} // namespace ctrmm_diagnostic
