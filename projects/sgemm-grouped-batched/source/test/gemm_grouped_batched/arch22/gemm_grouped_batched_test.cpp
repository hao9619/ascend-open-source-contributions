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
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "gemm_grouped_batched_param.h"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {
struct Grouped22TestParam : GemmGroupedBatchedParam {
    std::string aNull, bNull, cNull;
    explicit Grouped22TestParam(const csv_map& map)
        : GemmGroupedBatchedParam(map),
          aNull(ReadMap(map, "Aarray_null", "")),
          bNull(ReadMap(map, "Barray_null", "")),
          cNull(ReadMap(map, "Carray_null", ""))
    {}
};

struct Matrix {
    std::vector<float> host;
    void* device = nullptr;
    Matrix() = default;
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&& other) noexcept : host(std::move(other.host)), device(other.device) { other.device = nullptr; }
    ~Matrix()
    {
        if (device) {
            aclrtFree(device);
        }
    }
};

int Value(const std::vector<int>& array, int g, int fallback = 1)
{
    return static_cast<size_t>(g) < array.size() ? array[g] : fallback;
}

bool NullElement(const std::string& marker, int group, int batch)
{
    if (marker.empty() || marker == "NULLPTR") {
        return false;
    }
    const auto modes = parseIntArray(marker);
    return static_cast<size_t>(group) < modes.size() && modes[group] != 0 && batch == 0;
}

bool Upload(Matrix& matrix, int rows, int cols, const BlasFillMode& fill, unsigned seed)
{
    if (fill.method == BlasFillMode::M_GAUSS) {
        std::mt19937 random(seed);
        std::uniform_real_distribution<float> mean(-5.0f, 5.0f), sigma(0.1f, 2.0f);
        const float mu = mean(random), sd = sigma(random);
        std::normal_distribution<float> normal(mu, sd);
        matrix.host.resize(static_cast<size_t>(rows) * cols);
        for (float& value : matrix.host)
            value = normal(random);
    } else {
        matrix.host = makeBlasMatrix(rows, cols, rows, fill, seed);
    }
    const size_t bytes = matrix.host.size() * sizeof(float);
    if (aclrtMalloc(&matrix.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return false;
    }
    return aclrtMemcpy(matrix.device, bytes, matrix.host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

void ComputeGolden(
    const Grouped22TestParam& p, int g, const Matrix& a, const Matrix& b, float alpha, float beta,
    std::vector<float>& golden)
{
    const int m = p.mArray[g], n = p.nArray[g], k = p.kArray[g];
    const auto ta = p.transaArray[g] == ACLBLAS_OP_N ? CblasNoTrans : CblasTrans;
    auto tb = p.transbArray[g] == ACLBLAS_OP_N ? CblasNoTrans : CblasTrans;
    const float* referenceB = b.host.data();
    int ldb = p.ldbArray[g];
    std::vector<float> packed;
    // Match the upstream strided-batched golden optimization; bitwise equivalence
    // verified separately.
    if (tb == CblasTrans && m >= 128 && n >= 128 && k >= 128 && alpha != 0.0f) {
        packed.resize(static_cast<size_t>(k) * n);
        for (int col = 0; col < n; ++col) {
            for (int inner = 0; inner < k; ++inner) {
                packed[static_cast<size_t>(col) * k + inner] = b.host[static_cast<size_t>(inner) * ldb + col];
            }
        }
        referenceB = packed.data();
        ldb = k;
        tb = CblasNoTrans;
    }
    cblas_sgemm(
        CblasColMajor, ta, tb, m, n, k, alpha, a.host.data(), p.ldaArray[g], referenceB, ldb, beta, golden.data(),
        p.ldcArray[g]);
}

double MaxAbsoluteLimit(float golden)
{
    const double magnitude = std::abs(static_cast<double>(golden));
    if (magnitude < std::numeric_limits<float>::min()) {
        return 1.0e-2;
    }
    int exponent = 0;
    std::frexp(magnitude, &exponent);
    return std::max(1.0e-2, 32.0 * std::ldexp(1.0, exponent - 24));
}

bool Match(float actual, float golden)
{
    if (std::isnan(golden)) {
        return std::isnan(actual);
    }
    if (std::isinf(golden)) {
        return actual == golden;
    }
    return std::isfinite(actual) && std::abs(static_cast<double>(actual) - golden) <=
                                        (1.0 / 8192.0) * (1.0 + std::abs(static_cast<double>(golden)));
}
} // namespace

class GemmGroupedBatchedArch22Test : public BlasTest<Grouped22TestParam> {
protected:
    static void SetUpTestSuite()
    {
        BlasTest<Grouped22TestParam>::SetUpTestSuite();
        if (!HasFatalFailure()) {
            // The shared guard may have been cleaned by the previous parameterized
            // suite.
            blas_test_detail::globalAcl().cleaned = false;
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    GemmGroupedBatched, GemmGroupedBatchedArch22Test,
    ::testing::ValuesIn(GetCasesFromCsv<Grouped22TestParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Grouped22TestParam>);

INSTANTIATE_TEST_SUITE_P(
    GemmGroupedNormal, GemmGroupedBatchedArch22Test,
    ::testing::ValuesIn(GetCasesFromCsv<Grouped22TestParam>(
        std::string(__FILE__).substr(0, std::string(__FILE__).find_last_of('/')) + "/gemm_grouped_batched_normal.csv")),
    PrintCaseInfoString<Grouped22TestParam>);

TEST_P(GemmGroupedBatchedArch22Test, CsvDriven)
{
    const auto& p = GetParam();
    ASSERT_EQ(p.dtype, 0);
    size_t total = 0;
    for (int g = 0; g < p.groupCount; ++g) {
        total += static_cast<size_t>(std::max(0, Value(p.groupSizeArray, g)));
    }
    std::vector<Matrix> a(total), b(total), c(total);
    std::vector<const float*> ap(std::max<size_t>(1, total), nullptr), bp(ap.size(), nullptr);
    std::vector<float*> cp(ap.size(), nullptr);
    size_t offset = 0;
    for (int g = 0; g < p.groupCount; ++g) {
        const int m = std::max(1, Value(p.mArray, g));
        const int n = std::max(1, Value(p.nArray, g));
        const int k = std::max(1, Value(p.kArray, g));
        const bool ta = !p.transaArrayNull && p.transaArray[g] == ACLBLAS_OP_T;
        const bool tb = !p.transbArrayNull && p.transbArray[g] == ACLBLAS_OP_T;
        const int lda = std::max(ta ? k : m, Value(p.ldaArray, g));
        const int ldb = std::max(tb ? n : k, Value(p.ldbArray, g));
        const int ldc = std::max(m, Value(p.ldcArray, g));
        for (int j = 0; j < std::max(0, Value(p.groupSizeArray, g)); ++j, ++offset) {
            ASSERT_TRUE(Upload(a[offset], lda, ta ? m : k, p.A_fill, p.randomSeed + offset * 3));
            ASSERT_TRUE(Upload(b[offset], ldb, tb ? k : n, p.B_fill, p.randomSeed + offset * 3 + 1));
            ASSERT_TRUE(Upload(c[offset], ldc, n, p.C_fill, p.randomSeed + offset * 3 + 2));
            ap[offset] = NullElement(p.aNull, g, j) ? nullptr : static_cast<float*>(a[offset].device);
            bp[offset] = NullElement(p.bNull, g, j) ? nullptr : static_cast<float*>(b[offset].device);
            cp[offset] = NullElement(p.cNull, g, j) ? nullptr : static_cast<float*>(c[offset].device);
        }
    }
    const auto alpha = p.alphaArrayNull ? std::vector<float>{} : parseFloatArray(p.alphaArrayStr);
    const auto beta = p.betaArrayNull ? std::vector<float>{} : parseFloatArray(p.betaArrayStr);
    // Negative cases call the real API; the test does not synthesize the expected
    // status.
    const auto ret = aclblasSgemmGroupedBatched(
        p.description == "null_handle" ? nullptr : handle_, p.groupCount,
        p.transaArrayNull ? nullptr : p.transaArray.data(), p.transbArrayNull ? nullptr : p.transbArray.data(),
        p.mArrayNull ? nullptr : p.mArray.data(), p.nArrayNull ? nullptr : p.nArray.data(),
        p.kArrayNull ? nullptr : p.kArray.data(), p.alphaArrayNull ? nullptr : alpha.data(),
        p.aNull == "NULLPTR" ? nullptr : ap.data(), p.ldaArrayNull ? nullptr : p.ldaArray.data(),
        p.bNull == "NULLPTR" ? nullptr : bp.data(), p.ldbArrayNull ? nullptr : p.ldbArray.data(),
        p.betaArrayNull ? nullptr : beta.data(), p.cNull == "NULLPTR" ? nullptr : cp.data(),
        p.ldcArrayNull ? nullptr : p.ldcArray.data(), p.groupSizeArrayNull ? nullptr : p.groupSizeArray.data());
    // Complete stream work before RAII releases device storage, including
    // assertion failures.
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    ASSERT_EQ(ret, p.expectResult);
    if (ret != ACLBLAS_STATUS_SUCCESS || p.groupCount == 0) {
        return;
    }
    offset = 0;
    size_t matched = 0, count = 0;
    double maxError = 0.0;
    float maxErrorGolden = 0.0f;
    size_t maxErrorBatch = 0;
    int maxErrorRow = 0, maxErrorCol = 0;
    for (int g = 0; g < p.groupCount; ++g) {
        for (int j = 0; j < p.groupSizeArray[g]; ++j, ++offset) {
            const int m = p.mArray[g], n = p.nArray[g], ldc = p.ldcArray[g];
            std::vector<float> golden = c[offset].host;
            if (m > 0 && n > 0) {
                ComputeGolden(p, g, a[offset], b[offset], alpha[g], beta[g], golden);
            }
            std::vector<float> actual(golden.size());
            const size_t bytes = actual.size() * sizeof(float);
            ASSERT_EQ(
                aclrtMemcpy(actual.data(), bytes, c[offset].device, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
            const int storedRows = std::max(std::max(1, m), ldc);
            const int storedCols = std::max(1, n);
            for (int col = 0; col < storedCols; ++col) {
                for (int row = 0; row < storedRows; ++row) {
                    const size_t index = static_cast<size_t>(col) * storedRows + row;
                    if (row >= m || col >= n) {
                        ASSERT_EQ(std::memcmp(&actual[index], &c[offset].host[index], sizeof(float)), 0)
                            << "padding/no-op modified: batch=" << offset << " row=" << row << " col=" << col;
                        continue;
                    }
                    const bool match = Match(actual[index], golden[index]);
                    ++count;
                    matched += match;
                    if (!std::isfinite(golden[index]) || !std::isfinite(actual[index])) {
                        ASSERT_TRUE(match) << "nonfinite mismatch: batch=" << offset << " index=" << index
                                           << " actual=" << actual[index] << " golden=" << golden[index];
                    } else {
                        const double error = std::abs(static_cast<double>(actual[index]) - golden[index]);
                        if (error > maxError ||
                            (error == maxError && MaxAbsoluteLimit(golden[index]) < MaxAbsoluteLimit(maxErrorGolden))) {
                            maxError = error;
                            maxErrorGolden = golden[index];
                            maxErrorBatch = offset;
                            maxErrorRow = row;
                            maxErrorCol = col;
                        }
                    }
                }
            }
        }
    }
    EXPECT_TRUE(count == 0 || static_cast<double>(matched) / count >= 0.99)
        << "matched=" << matched << " total=" << count;
    const double maxLimit = MaxAbsoluteLimit(maxErrorGolden);
    std::cout << std::setprecision(12) << "[ ACCURACY ] case=" << p.caseName << " elements=" << count
              << " matched=" << matched << " max_abs_error=" << maxError << " max_abs_limit=" << maxLimit
              << " anchor_golden=" << maxErrorGolden << " batch=" << maxErrorBatch << " row=" << maxErrorRow
              << " col=" << maxErrorCol << std::endl;
    EXPECT_LE(maxError, maxLimit) << "maximum absolute error=" << maxError << " golden=" << maxErrorGolden;
}

TEST_F(GemmGroupedBatchedArch22Test, RepeatedAsyncCalls)
{
    constexpr int DIM = 32;
    Matrix a, b, c0, c1;
    ASSERT_TRUE(Upload(a, DIM, DIM, parseFill("VALUE_1"), 1));
    ASSERT_TRUE(Upload(b, DIM, DIM, parseFill("VALUE_2"), 2));
    ASSERT_TRUE(Upload(c0, DIM, DIM, parseFill("VALUE_3"), 3));
    ASSERT_TRUE(Upload(c1, DIM, DIM, parseFill("VALUE_3"), 4));
    const float* ap[]{static_cast<float*>(a.device), static_cast<float*>(a.device)};
    const float* bp[]{static_cast<float*>(b.device), static_cast<float*>(b.device)};
    float* cp[]{static_cast<float*>(c0.device), static_cast<float*>(c1.device)};
    int m = DIM, n = DIM, k = DIM, ld = DIM, size = 2;
    aclblasOperation_t ta = ACLBLAS_OP_N, tb = ACLBLAS_OP_N;
    float alpha = 0.5f, beta = 0.5f;
    auto call = [&]() {
        return aclblasSgemmGroupedBatched(
            handle_, 1, &ta, &tb, &m, &n, &k, &alpha, ap, &ld, bp, &ld, &beta, cp, &ld, &size);
    };
    const auto first = call();
    alpha = -1.0f;
    beta = 0.25f;
    ta = ACLBLAS_OP_T;
    tb = ACLBLAS_OP_T;
    const auto second = call();
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    ASSERT_EQ(first, ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(second, ACLBLAS_STATUS_SUCCESS);
    for (auto* matrix : {&c0, &c1}) {
        std::vector<float> actual(DIM * DIM);
        ASSERT_EQ(
            aclrtMemcpy(
                actual.data(), actual.size() * sizeof(float), matrix->device, actual.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
        for (float value : actual) {
            EXPECT_FLOAT_EQ(value, -55.625f);
        }
    }
}

TEST_F(GemmGroupedBatchedArch22Test, OrderedRepeatedAsyncCalls)
{
    constexpr int DIM = 32, INNER = 512;
    Matrix a, b, c0, c1;
    ASSERT_TRUE(Upload(a, DIM, INNER, parseFill("VALUE_1"), 1));
    ASSERT_TRUE(Upload(b, INNER, DIM, parseFill("VALUE_2"), 2));
    ASSERT_TRUE(Upload(c0, DIM, DIM, parseFill("VALUE_3"), 3));
    ASSERT_TRUE(Upload(c1, DIM, DIM, parseFill("VALUE_3"), 4));
    const float* ap[]{static_cast<float*>(a.device), static_cast<float*>(a.device)};
    const float* bp[]{static_cast<float*>(b.device), static_cast<float*>(b.device)};
    float* cp[]{static_cast<float*>(c0.device), static_cast<float*>(c1.device)};
    int m = DIM, n = DIM, k = INNER, lda = DIM, ldb = INNER, ldc = DIM, size = 2;
    aclblasOperation_t op = ACLBLAS_OP_N;
    float alpha = 0.5f, beta = 0.5f;
    auto call = [&]() {
        return aclblasSgemmGroupedBatched(
            handle_, 1, &op, &op, &m, &n, &k, &alpha, ap, &lda, bp, &ldb, &beta, cp, &ldc, &size);
    };
    const auto first = call();
    alpha = -1.0f;
    beta = 0.25f;
    const auto second = call();
    alpha = beta = std::numeric_limits<float>::quiet_NaN();
    m = n = k = 0;
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    ASSERT_EQ(first, ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(second, ACLBLAS_STATUS_SUCCESS);
    for (auto* matrix : {&c0, &c1}) {
        std::vector<float> actual(DIM * DIM);
        ASSERT_EQ(
            aclrtMemcpy(
                actual.data(), actual.size() * sizeof(float), matrix->device, actual.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
        for (float value : actual)
            EXPECT_FLOAT_EQ(value, -895.625f);
    }
}

TEST_F(GemmGroupedBatchedArch22Test, NonFiniteScalars)
{
    constexpr int DIM = 16, INNER = 512;
    Matrix a, b, c;
    ASSERT_TRUE(Upload(a, DIM, INNER, parseFill("VALUE_1"), 1));
    ASSERT_TRUE(Upload(b, INNER, DIM, parseFill("VALUE_1"), 2));
    ASSERT_TRUE(Upload(c, DIM, DIM, parseFill("VALUE_3"), 3));
    const float* ap[]{static_cast<float*>(a.device)};
    const float* bp[]{static_cast<float*>(b.device)};
    float* cp[]{static_cast<float*>(c.device)};
    int m = DIM, n = DIM, k = INNER, lda = DIM, ldb = INNER, ldc = DIM, size = 1;
    aclblasOperation_t op = ACLBLAS_OP_N;
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (const auto& scales : {std::pair<float, float>{1, inf}, {1, -inf}, {1, nan}, {inf, 0}, {nan, 0}, {0, 1}}) {
        ASSERT_EQ(
            aclrtMemcpy(
                c.device, c.host.size() * sizeof(float), c.host.data(), c.host.size() * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_SUCCESS);
        const auto ret = aclblasSgemmGroupedBatched(
            handle_, 1, &op, &op, &m, &n, &k, &scales.first, ap, &lda, bp, &ldb, &scales.second, cp, &ldc, &size);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
        auto golden = c.host;
        cblas_sgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans, DIM, DIM, INNER, scales.first, a.host.data(), lda, b.host.data(),
            ldb, scales.second, golden.data(), ldc);
        std::vector<float> actual(DIM * DIM);
        ASSERT_EQ(
            aclrtMemcpy(
                actual.data(), actual.size() * sizeof(float), c.device, actual.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
        for (size_t index = 0; index < actual.size(); ++index)
            EXPECT_TRUE(Match(actual[index], golden[index]));
    }
}

TEST_F(GemmGroupedBatchedArch22Test, SmallVectorPaddingTailAndNonFinite)
{
    for (int mode = 0; mode < 12; ++mode) {
        SCOPED_TRACE(mode);
        const int scenario = mode % 6;
        int m = mode >= 6 ? 16 : (scenario == 3 ? 15 : 5);
        int n = mode >= 6 ? 16 : 3, k = scenario == 3 ? 64 : 7;
        int lda = m + 3, ldb = k + 2, ldc = m + 5, size = 65;
        Matrix a, b;
        ASSERT_TRUE(Upload(a, lda, k, parseFill("VALUE_2"), 1));
        ASSERT_TRUE(Upload(b, ldb, n, parseFill("VALUE_1"), 2));
        if (scenario < 2 || scenario >= 4) {
            const float special =
                scenario % 2 == 0 ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
            std::fill(a.host.begin(), a.host.end(), special);
            std::fill(b.host.begin(), b.host.end(), scenario >= 4 ? -0.0f : 0.0f);
        } else {
            for (size_t i = 0; i < a.host.size(); ++i)
                a.host[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
            for (size_t i = 0; i < b.host.size(); ++i)
                b.host[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
        }
        ASSERT_EQ(
            aclrtMemcpy(a.device, a.host.size() * 4, a.host.data(), a.host.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMemcpy(b.device, b.host.size() * 4, b.host.data(), b.host.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_SUCCESS);
        std::vector<Matrix> outputs(size);
        std::vector<const float*> ap(size, static_cast<float*>(a.device));
        std::vector<const float*> bp(size, static_cast<float*>(b.device));
        std::vector<float*> cp(size);
        for (int i = 0; i < size; ++i) {
            ASSERT_TRUE(Upload(outputs[i], ldc, n, parseFill("VALUE_3"), 3));
            cp[i] = static_cast<float*>(outputs[i].device);
        }
        auto golden = outputs[0].host;
        cblas_sgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a.host.data(), lda, b.host.data(), ldb, 0.0f,
            golden.data(), ldc);
        float alpha = 1.0f, beta = 0.0f;
        aclblasOperation_t op = ACLBLAS_OP_N;
        for (int repeat = 0; repeat < 2; ++repeat) {
            ASSERT_EQ(
                aclblasSgemmGroupedBatched(
                    handle_, 1, &op, &op, &m, &n, &k, &alpha, ap.data(), &lda, bp.data(), &ldb, &beta, cp.data(), &ldc,
                    &size),
                ACLBLAS_STATUS_SUCCESS);
        }
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        for (int i = 0; i < size; ++i) {
            std::vector<float> actual(golden.size());
            ASSERT_EQ(
                aclrtMemcpy(
                    actual.data(), actual.size() * 4, outputs[i].device, actual.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_SUCCESS);
            for (int col = 0; col < n; ++col) {
                for (int row = 0; row < ldc; ++row) {
                    const int index = col * ldc + row;
                    if (row < m)
                        EXPECT_TRUE(Match(actual[index], golden[index])) << mode << ":" << i << ":" << index;
                    else
                        EXPECT_EQ(std::memcmp(&actual[index], &golden[index], sizeof(float)), 0);
                }
            }
        }
    }
}

TEST_F(GemmGroupedBatchedArch22Test, ExtremeScanBlockAndTail)
{
    // Cover both scan halves, multiple workers, padding fallback and the final 16-element tail.
    for (int scanK : {257, 4097}) {
        for (bool padded : {false, true}) {
            for (int position : {0, 2047, 2048, 4095, 4096, 16 * scanK - 1}) {
                for (float extreme :
                     {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN(), 1.0e35f, -1.0e35f}) {
                    SCOPED_TRACE(padded);
                    SCOPED_TRACE(position);
                    SCOPED_TRACE(extreme);
                    int m = 16, n = 16, k = scanK, lda = padded ? 19 : 16;
                    int ldb = k + (padded ? 3 : 0), ldc = m + 3, size = scanK == 4097 ? 16 : 2;
                    Matrix a, b;
                    std::vector<Matrix> outputs(size);
                    ASSERT_TRUE(Upload(a, lda, k, parseFill("VALUE_1"), 1));
                    ASSERT_TRUE(Upload(b, ldb, n, parseFill("VALUE_0"), 2));
                    for (auto& output : outputs)
                        ASSERT_TRUE(Upload(output, ldc, n, parseFill("VALUE_3"), 3));
                    a.host[(position / m) * lda + position % m] = extreme;
                    ASSERT_EQ(
                        aclrtMemcpy(
                            a.device, a.host.size() * sizeof(float), a.host.data(), a.host.size() * sizeof(float),
                            ACL_MEMCPY_HOST_TO_DEVICE),
                        ACL_SUCCESS);
                    std::vector<const float*> ap(size, static_cast<float*>(a.device));
                    std::vector<const float*> bp(size, static_cast<float*>(b.device));
                    std::vector<float*> cp(size);
                    for (int i = 0; i < size; ++i)
                        cp[i] = static_cast<float*>(outputs[i].device);
                    aclblasOperation_t op = ACLBLAS_OP_N;
                    float alpha = 1.0f, beta = 0.0f;
                    auto golden = outputs[0].host;
                    cblas_sgemm(
                        CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a.host.data(), lda, b.host.data(),
                        ldb, beta, golden.data(), ldc);
                    ASSERT_EQ(
                        aclblasSgemmGroupedBatched(
                            handle_, 1, &op, &op, &m, &n, &k, &alpha, ap.data(), &lda, bp.data(), &ldb, &beta,
                            cp.data(), &ldc, &size),
                        ACLBLAS_STATUS_SUCCESS);
                    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
                    for (auto& output : outputs) {
                        std::vector<float> actual(golden.size());
                        ASSERT_EQ(
                            aclrtMemcpy(
                                actual.data(), actual.size() * sizeof(float), output.device,
                                actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
                            ACL_SUCCESS);
                        for (int col = 0; col < n; ++col) {
                            for (int row = 0; row < ldc; ++row) {
                                const int index = col * ldc + row;
                                if (row < m)
                                    EXPECT_TRUE(Match(actual[index], golden[index])) << index;
                                else
                                    EXPECT_EQ(std::memcmp(&actual[index], &golden[index], sizeof(float)), 0);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_F(GemmGroupedBatchedArch22Test, SharedCacheLineAndOverlappingOutputs)
{
    // Preserve sequential fallback when distinct matrices share a cache line or the same address.
    for (bool sameAddress : {false, true}) {
        SCOPED_TRACE(sameAddress);
        int m = 1, n = 1, k = 1, ld = 1, size = 65;
        Matrix a, b, c;
        ASSERT_TRUE(Upload(a, 1, 1, parseFill("VALUE_2"), 1));
        ASSERT_TRUE(Upload(b, 1, 1, parseFill("VALUE_1"), 2));
        ASSERT_TRUE(Upload(c, size + 1, 1, parseFill("VALUE_3"), 3));
        std::vector<const float*> ap(size, static_cast<float*>(a.device));
        std::vector<const float*> bp(size, static_cast<float*>(b.device));
        std::vector<float*> cp(size);
        for (int i = 0; i < size; ++i)
            cp[i] = static_cast<float*>(c.device) + (sameAddress ? 0 : i);
        aclblasOperation_t op = ACLBLAS_OP_N;
        float alpha = 1.0f, beta = 1.0f;
        for (int repeat = 0; repeat < 2; ++repeat) {
            ASSERT_EQ(
                aclblasSgemmGroupedBatched(
                    handle_, 1, &op, &op, &m, &n, &k, &alpha, ap.data(), &ld, bp.data(), &ld, &beta, cp.data(), &ld,
                    &size),
                ACLBLAS_STATUS_SUCCESS);
        }
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        std::vector<float> actual(c.host.size());
        ASSERT_EQ(
            aclrtMemcpy(
                actual.data(), actual.size() * sizeof(float), c.device, actual.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
        for (int i = 0; i <= size; ++i) {
            const float expected = sameAddress ? (i == 0 ? 3.0f + 4.0f * size : 3.0f) : (i < size ? 7.0f : 3.0f);
            EXPECT_FLOAT_EQ(actual[i], expected) << i;
        }
    }
}

TEST_F(GemmGroupedBatchedArch22Test, AdjacentSmallGroupsAndInputDependencies)
{
    // Four independent groups may share one launch; an input/output dependency must retain order.
    for (int scenario = 0; scenario < 4; ++scenario) {
        const bool dependent = scenario % 2 != 0;
        SCOPED_TRACE(dependent);
        const int groups = dependent ? 2 : 4;
        const int dim = scenario >= 2 ? 32 : (dependent ? 2 : 16);
        const int perGroup = dependent ? 1 : 16;
        const int total = groups * perGroup;
        Matrix a, b;
        ASSERT_TRUE(Upload(a, dim, dim, parseFill("VALUE_1"), 1));
        ASSERT_TRUE(Upload(b, dim, dim, parseFill("VALUE_1"), 2));
        std::vector<Matrix> outputs(total);
        std::vector<const float*> ap(total, static_cast<float*>(a.device));
        std::vector<const float*> bp(total, static_cast<float*>(b.device));
        std::vector<float*> cp(total);
        for (int i = 0; i < total; ++i) {
            ASSERT_TRUE(Upload(outputs[i], dim, dim, parseFill("VALUE_3"), i + 3));
            cp[i] = static_cast<float*>(outputs[i].device);
        }
        if (dependent)
            ap[1] = cp[0];
        std::vector<aclblasOperation_t> op(groups, ACLBLAS_OP_N);
        std::vector<int> dims(groups, dim), sizes(groups, perGroup);
        std::vector<float> alpha(groups, 1.0f), beta(groups, 0.0f);
        ASSERT_EQ(
            aclblasSgemmGroupedBatched(
                handle_, groups, op.data(), op.data(), dims.data(), dims.data(), dims.data(), alpha.data(), ap.data(),
                dims.data(), bp.data(), dims.data(), beta.data(), cp.data(), dims.data(), sizes.data()),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        for (int i = 0; i < total; ++i) {
            std::vector<float> actual(dim * dim);
            ASSERT_EQ(
                aclrtMemcpy(
                    actual.data(), actual.size() * sizeof(float), outputs[i].device, actual.size() * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_SUCCESS);
            const float expected = dependent && i == 1 ? dim * dim : dim;
            for (float value : actual)
                EXPECT_FLOAT_EQ(value, expected);
        }
    }
}

TEST_F(GemmGroupedBatchedArch22Test, ScanFlagsAcrossSixtyFourMatrices)
{
    int m = 16, n = 16, k = 257, lda = m, ldb = k, ldc = m, size = 64;
    Matrix a, b;
    ASSERT_TRUE(Upload(a, lda, k, parseFill("VALUE_1"), 1));
    ASSERT_TRUE(Upload(b, ldb, n, parseFill("VALUE_0"), 2));
    std::vector<Matrix> outputs(size);
    std::vector<const float*> ap(size, static_cast<float*>(a.device)), bp(size, static_cast<float*>(b.device));
    std::vector<float*> cp(size);
    for (int i = 0; i < size; ++i) {
        ASSERT_TRUE(Upload(outputs[i], ldc, n, parseFill("VALUE_3"), i + 3));
        cp[i] = static_cast<float*>(outputs[i].device);
    }
    aclblasOperation_t op = ACLBLAS_OP_N;
    float alpha = 1.0f, beta = 0.0f;
    for (float fill : {std::numeric_limits<float>::infinity(), 1.0f}) {
        std::fill(a.host.begin(), a.host.end(), fill);
        ASSERT_EQ(
            aclrtMemcpy(
                a.device, a.host.size() * sizeof(float), a.host.data(), a.host.size() * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE),
            ACL_SUCCESS);
        ASSERT_EQ(
            aclblasSgemmGroupedBatched(
                handle_, 1, &op, &op, &m, &n, &k, &alpha, ap.data(), &lda, bp.data(), &ldb, &beta, cp.data(), &ldc,
                &size),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        for (int i = 0; i < size; ++i) {
            std::vector<float> actual(m * n);
            ASSERT_EQ(
                aclrtMemcpy(
                    actual.data(), actual.size() * sizeof(float), outputs[i].device, actual.size() * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_SUCCESS);
            for (float value : actual) {
                if (std::isinf(fill))
                    EXPECT_TRUE(std::isnan(value)) << i;
                else
                    EXPECT_FLOAT_EQ(value, 0.0f) << i;
            }
        }
    }
}
