/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include "blas_test.h"
#include "ctrmm_param.h"
#include "ctrmm_golden.h"
#include "ctrmm_npu_wrapper.h"
#include "ctrmm_numeric_diagnostic.h"

class CtrmmArch22Test : public BlasTest<CtrmmParam> {};

static std::vector<aclblasComplex> MakeInput(
    int rows, int cols, int ld, const std::string& fill, uint32_t seed, bool normal)
{
    auto data = makeBlasComplexMatrix(rows, cols, ld, parseFill(fill), seed);
    if (normal && fill == "RANDOM_NORM_5_5") {
        std::mt19937 engine(seed);
        std::normal_distribution<float> distribution(0.0f, 1.0f);
        for (auto& value : data)
            value = {distribution(engine), distribution(engine)};
    }
    return data;
}

static void DoubleDiagnostic(
    const CtrmmParam& p, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& expected,
    const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& b, int part, size_t index)
{
    const int row = index % p.ldc;
    const int col = index / p.ldc;
    const int d = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
    std::complex<double> reference(0, 0);
    for (int k = 0; k < d; ++k) {
        int ar = p.side == ACLBLAS_SIDE_LEFT ? row : k;
        int ac = p.side == ACLBLAS_SIDE_LEFT ? k : col;
        if (p.trans != ACLBLAS_OP_N)
            std::swap(ar, ac);
        if (p.uplo == ACLBLAS_UPPER ? ar > ac : ar < ac)
            continue;
        std::complex<double> av(1, 0);
        if (p.diag != ACLBLAS_UNIT || ar != ac) {
            const auto value = a[static_cast<size_t>(ac) * p.lda + ar];
            av = {value.real, value.imag};
            if (p.trans == ACLBLAS_OP_C)
                av = std::conj(av);
        }
        const size_t bi =
            p.side == ACLBLAS_SIDE_LEFT ? static_cast<size_t>(col) * p.ldb + k : static_cast<size_t>(k) * p.ldb + row;
        reference += av * std::complex<double>(b[bi].real, b[bi].imag);
    }
    reference *= std::complex<double>(p.alpha.real, p.alpha.imag);
    const auto npu = actual[index];
    const auto golden = expected[index];
    const double exact = part ? reference.imag() : reference.real();
    std::cout << "DOUBLE_DIAGNOSTIC " << p.caseName << " row=" << row << " col=" << col << " part=" << part
              << " npu_error=" << std::abs((part ? npu.imag : npu.real) - exact)
              << " netlib_error=" << std::abs((part ? golden.imag : golden.real) - exact) << '\n';
}

static void VerifyOutput(
    const CtrmmParam& p, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& expected,
    const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& b)
{
    size_t matched[2] = {0, 0};
    double maxError[2] = {0, 0};
    size_t nonfiniteMismatch = 0;
    size_t maxIndex[2] = {0, 0};
    double maxReference[2] = {0, 0};
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.ldc; ++row) {
            const size_t index = static_cast<size_t>(col) * p.ldc + row;
            const float av[2] = {actual[index].real, actual[index].imag};
            const float ev[2] = {expected[index].real, expected[index].imag};
            for (int part = 0; part < 2; ++part) {
                if (row >= p.m) {
                    EXPECT_EQ(av[part], ev[part]) << "padding changed at " << index;
                    continue;
                }
                if (!std::isfinite(av[part]) || !std::isfinite(ev[part])) {
                    const bool same = (std::isnan(av[part]) && std::isnan(ev[part])) || av[part] == ev[part];
                    matched[part] += same;
                    nonfiniteMismatch += !same;
                    continue;
                }
                const double error = std::abs(static_cast<double>(av[part]) - ev[part]);
                matched[part] += error <= std::ldexp(1.0, -13) * (1.0 + std::abs(ev[part]));
                if (error > maxError[part]) {
                    maxError[part] = error;
                    maxIndex[part] = index;
                }
                maxReference[part] = std::max(maxReference[part], std::abs(static_cast<double>(ev[part])));
            }
        }
    }
    ctrmm_diagnostic::Run(p, a, b, expected, maxIndex);
    EXPECT_EQ(nonfiniteMismatch, 0);
    for (int part = 0; part < 2; ++part) {
        const double ratio = static_cast<double>(matched[part]) / (static_cast<double>(p.m) * p.n);
        std::cout << "PRECISION " << p.caseName << " part=" << part << " matched=" << ratio
                  << " max_abs=" << maxError[part] << '\n';
        const float magnitude = static_cast<float>(maxReference[part]);
        const double ulp = std::nextafter(magnitude, std::numeric_limits<float>::infinity()) - magnitude;
        std::cout << "ULP_DIAGNOSTIC " << p.caseName << " part=" << part << " tensor_max_32ulp=" << 32 * ulp << '\n';
        if (maxError[part] > 0.01 && std::getenv("CTRMM_DIAGNOSTIC") != nullptr) {
            DoubleDiagnostic(p, actual, expected, a, b, part, maxIndex[part]);
        }
        EXPECT_GE(ratio, 0.99);
        EXPECT_LE(maxError[part], 0.01);
    }
}

static void RunCase(aclblasHandle_t handle, const CtrmmParam& p)
{
    const int d = p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n;
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.m == 0 || p.n == 0) {
        std::vector<aclblasComplex> dummy(1, {0, 0});
        auto c = dummy;
        EXPECT_EQ(CtrmmNpu(handle, p, dummy, dummy, c), p.expectResult);
        return;
    }
    const bool normal = (p.randomSeed % 2) != 0 && p.caseName.find("TC_PF") != 0;
    auto a = MakeInput(d, d, p.lda, p.aFill, p.randomSeed, normal);
    auto b = MakeInput(p.m, p.n, p.ldb, p.bFill, p.randomSeed + 1, normal);
    const float poison = std::numeric_limits<float>::quiet_NaN();
    for (int col = 0; col < d; ++col) {
        for (int row = 0; row < d; ++row) {
            const bool unused = p.uplo == ACLBLAS_UPPER ? row > col : row < col;
            if (unused || (p.diag == ACLBLAS_UNIT && row == col)) {
                a[static_cast<size_t>(col) * p.lda + row] = {poison, poison};
            }
        }
    }
    std::vector<aclblasComplex> c(static_cast<size_t>(p.ldc) * p.n, {42.0f, -42.0f});
    auto expected = CtrmmGolden(p, a, b);
    if (p.inPlace) {
        for (int col = 0; col < p.n; ++col) {
            for (int row = p.m; row < p.ldc; ++row) {
                expected[static_cast<size_t>(col) * p.ldc + row] = b[static_cast<size_t>(col) * p.ldb + row];
            }
        }
    }
    ASSERT_EQ(CtrmmNpu(handle, p, a, b, c), ACLBLAS_STATUS_SUCCESS);
    VerifyOutput(p, c, expected, a, b);
}

TEST_P(CtrmmArch22Test, CsvDriven) { RunCase(handle_, GetParam()); }

TEST_F(CtrmmArch22Test, InPlaceAndIndependentPadding)
{
    for (const auto side : {"LEFT", "RIGHT"}) {
        for (const auto uplo : {"UPPER", "LOWER"}) {
            for (const auto trans : {"N", "T", "C"}) {
                for (const auto diag : {"NON_UNIT", "UNIT"}) {
                    csv_map map{
                        {"case_name", "inplace"},
                        {"side", side},
                        {"uplo", uplo},
                        {"trans", trans},
                        {"diag", diag},
                        {"m", "17"},
                        {"n", "31"},
                        {"lda", "37"},
                        {"ldb", "23"},
                        {"ldc", "23"},
                        {"in_place", "1"},
                        {"alpha_real", "0.5"},
                        {"alpha_imag", "-0.5"},
                        {"random_seed", "20260001"}};
                    RunCase(handle_, CtrmmParam(map));
                    map["in_place"] = "0";
                    map["ldc"] = "29";
                    RunCase(handle_, CtrmmParam(map));
                }
            }
        }
    }
}

TEST_F(CtrmmArch22Test, GuardedFallbackLayouts)
{
    for (const auto side : {"LEFT", "RIGHT"}) {
        for (const auto uplo : {"UPPER", "LOWER"}) {
            for (const auto trans : {"N", "T", "C"}) {
                for (const auto diag : {"NON_UNIT", "UNIT"}) {
                    csv_map map{
                        {"case_name", "guarded_layout"},
                        {"side", side},
                        {"uplo", uplo},
                        {"trans", trans},
                        {"diag", diag},
                        {"m", "17"},
                        {"n", "31"},
                        {"lda", "37"},
                        {"ldb", "23"},
                        {"ldc", "29"},
                        {"alpha_real", "1"},
                        {"alpha_imag", "0"},
                        {"a_fill", "RANDOM_NORM_16_17"},
                        {"b_fill", "RANDOM_NORM_0_1"},
                        {"random_seed", "20261008"}};
                    RunCase(handle_, CtrmmParam(map));
                }
            }
        }
    }
}

TEST_F(CtrmmArch22Test, NonfiniteLayouts)
{
    for (int mode = 0; mode < 24; ++mode) {
        const bool left = mode < 12;
        const char* trans[] = {"N", "T", "C"};
        for (int pattern = 0; pattern < 4; ++pattern) {
            const char* fill = pattern % 2 == 0 ? "VALUE_NORM_INF" : "VALUE_NORM_NAN";
            for (const bool complexAlpha : {false, true}) {
                const std::string name = "nonfinite_" + std::to_string(mode) + "_" + std::to_string(pattern) +
                                         (complexAlpha ? "_complex_alpha" : "_unit_alpha");
                SCOPED_TRACE(name);
                csv_map map{
                    {"case_name", name},
                    {"side", left ? "LEFT" : "RIGHT"},
                    {"uplo", mode / 6 % 2 == 0 ? "UPPER" : "LOWER"},
                    {"trans", trans[mode / 2 % 3]},
                    {"diag", mode % 2 == 0 ? "NON_UNIT" : "UNIT"},
                    {"m", "7"},
                    {"n", "9"},
                    {"lda", left ? "10" : "12"},
                    {"ldb", "9"},
                    {"ldc", "11"},
                    {"alpha_real", complexAlpha ? "0.5" : "1"},
                    {"alpha_imag", complexAlpha ? "-0.5" : "0"},
                    {"a_fill", pattern < 2 ? fill : "RANDOM_NORM_5_5"},
                    {"b_fill", pattern < 2 ? "RANDOM_NORM_5_5" : fill},
                    {"random_seed", "20261015"}};
                RunCase(handle_, CtrmmParam(map));
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Ctrmm, CtrmmArch22Test, ::testing::ValuesIn(GetCasesFromCsv<CtrmmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtrmmParam>);
