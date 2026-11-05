/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cstdint>
#include "cann_ops_blas.h"
#include "log/log.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/complex_blas3_host_utils.h"
#include "common/helper/host_utils.h"
#include "ctrmm_kernel.h"

void ctrmm_guarded_combine_do(
    uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, CtrmmCombineTiling,
    CtrmmOrderedTiling);

void ctrmm_left_guarded_combine_do(
    uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, CtrmmCombineTiling,
    CtrmmOrderedTiling);

void ctrmm_compact_pack_pair_do(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* ar, uint8_t* ai, uint8_t* as, uint8_t* b, uint8_t* br,
    uint8_t* bi, uint8_t* bs, CtrmmPackTiling pa, CtrmmPackTiling pb);

void ctrmm_small_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, CtrmmOrderedTiling);
void ctrmm_tiny_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, CtrmmOrderedTiling);

namespace {
struct CtrmmWorkspace {
    uint8_t *ar, *ai, *br, *bi, *rr, *ii, *ri, *ir, *as, *bs, *seed, *flags;
    uint32_t aLd, bLd;
};

static aclblasStatus_t ValidateCtrmmParams(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n)
{
    const bool enumsValid = (side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT) &&
                            (uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER) &&
                            (trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C) &&
                            (diag == ACLBLAS_UNIT || diag == ACLBLAS_NON_UNIT);
    if (!enumsValid || m < 0 || n < 0) {
        OP_LOGE("aclblasCtrmm", "invalid enumeration or negative dimension");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static bool MatrixSpan(const void* pointer, int rows, int cols, int ld, uint64_t& first, uint64_t& last)
{
    first = reinterpret_cast<uintptr_t>(pointer);
    uint64_t elements = static_cast<uint64_t>(cols - 1) * ld + rows;
    uint64_t bytes = 0;
    return CBlas3TryMulU64(elements, sizeof(aclblasComplex), &bytes) && CBlas3TryAddU64(first, bytes, &last);
}

static bool ValidCtrmmOverlap(
    const aclblasComplex* a, const aclblasComplex* b, aclblasComplex* c, int d, int m, int n, int lda, int ldb, int ldc)
{
    uint64_t a0, a1, b0, b1, c0, c1;
    if (!MatrixSpan(a, d, d, lda, a0, a1) || !MatrixSpan(b, m, n, ldb, b0, b1) || !MatrixSpan(c, m, n, ldc, c0, c1))
        return false;
    if (c0 < a1 && a0 < c1)
        return false;
    return b == c ? ldb == ldc : !(c0 < b1 && b0 < c1);
}

static aclblasStatus_t PrepareCtrmmWorkspace(
    aclblasHandle_t handle, uint32_t m, uint32_t n, uint32_t d, CtrmmWorkspace& ws)
{
    ws.aLd = CeilAlign<uint32_t>(d, 128);
    ws.bLd = CeilAlign<uint32_t>(m, 128);
    const uint64_t aElements = static_cast<uint64_t>(ws.aLd) * d;
    const uint64_t bElements =
        std::max(static_cast<uint64_t>(ws.bLd) * n, static_cast<uint64_t>(CeilAlign<uint32_t>(n, 128)) * m);
    const uint64_t cap = ACLBLAS_MAX_WORKSPACE_SIZE;
    if (aElements > cap / 12 || bElements > cap / 28) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const size_t aBytes = aElements * sizeof(float);
    const size_t bBytes = bElements * sizeof(float);
    const size_t seedOffset =
        std::max<size_t>(3 * aBytes + 7 * bBytes, GetAivCoreCount() * ctrmm_repair_scratch_per_core(d));
    const size_t bytes = seedOffset + 2 * bBytes + GetAivCoreCount() * 64;
    if (bytes > cap)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    const auto status = EnsureDefaultWorkspace(handle, bytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCtrmm", "workspace allocation failed: %zu", bytes);
        return status;
    }
    auto* base = static_cast<uint8_t*>(GetEffectiveWorkspace(handle));
    ws.ar = base;
    ws.seed = base + seedOffset;
    ws.flags = ws.seed + 2 * bBytes;
    ws.ai = ws.ar + aBytes;
    ws.br = ws.ai + aBytes;
    ws.bi = ws.br + bBytes;
    ws.rr = ws.bi + bBytes;
    ws.ii = ws.rr + bBytes;
    ws.ri = ws.ii + bBytes;
    ws.ir = ws.ri + bBytes;
    ws.as = ws.ir + bBytes;
    ws.bs = ws.as + aBytes;
    return ACLBLAS_STATUS_SUCCESS;
}

static void LaunchCtrmmGemm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, int m, int n,
    const CtrmmWorkspace& ws, uint32_t aic, bool fourProduct, uint32_t endK = 0, uint32_t startK = 0)
{
    const uint32_t d = side == ACLBLAS_SIDE_LEFT ? m : n;
    CBlas3GemmTilingData gemm{};
    gemm.m = n;
    gemm.n = m;
    gemm.k = d;
    gemm.ldc = ws.bLd;
    gemm.mBlocks = CeilDiv<uint32_t>(n, ctrmm_tile_m());
    gemm.nBlocks = CeilDiv<uint32_t>(m, ctrmm_tile_n());
    gemm.uploMode = CBLAS3_UPLO_NONE;
    gemm.transMode = side == ACLBLAS_SIDE_RIGHT && n <= 128 && trans != ACLBLAS_OP_N ? CBLAS3_GEMM_TRANS_LEFT :
                                                                                       CBLAS3_GEMM_TRANS_NONE;
    const bool left = side == ACLBLAS_SIDE_LEFT;
    gemm.ldA = left ? ws.bLd : ws.aLd;
    gemm.ldB = left ? ws.aLd : ws.bLd;
    uint8_t* lr = left ? ws.br : ws.ar;
    uint8_t* li = left ? ws.bi : ws.ai;
    uint8_t* rr = left ? ws.ar : ws.br;
    uint8_t* ri = left ? ws.ai : ws.bi;
    const uint32_t blocks = std::min<uint64_t>(aic, static_cast<uint64_t>(gemm.mBlocks) * gemm.nBlocks);
    // Bound FP32 dot-product accumulation depth for long reductions.
    const uint32_t singleK = fourProduct ? 512 : ctrmm_single_k();
    const bool effectiveUpper = (uplo == ACLBLAS_UPPER) != (trans != ACLBLAS_OP_N);
    CtrmmGemmTiling full{gemm, left, effectiveUpper, fourProduct};
    const uint32_t limit = endK ? endK : d;
    for (uint32_t offset = startK; offset < limit; offset += singleK) {
        full.base.kBase = offset;
        full.base.kCount = std::min(singleK, limit - offset);
        ctrmm_gemm_do(
            blocks, handle->stream, lr, li, rr, ri, left ? ws.bs : ws.as, left ? ws.as : ws.bs, ws.rr, ws.ii, ws.ri,
            ws.ir, full);
    }
}

#ifdef CTRMM_ORDERED
static void LaunchOrderedPacks(
    aclblasHandle_t handle, bool left, uint32_t d, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb,
    const CtrmmWorkspace& ws, uint32_t aiv, uint32_t guardBlocks)
{
    CtrmmPackTiling pa{
        d,
        d,
        static_cast<uint32_t>(lda),
        ws.aLd,
        1,
        uplo == ACLBLAS_UPPER,
        left ? trans == ACLBLAS_OP_N : trans != ACLBLAS_OP_N,
        trans == ACLBLAS_OP_C,
        diag == ACLBLAS_UNIT};
    CtrmmPackTiling pb{
        static_cast<uint32_t>(left ? n : m),
        static_cast<uint32_t>(left ? m : n),
        static_cast<uint32_t>(ldb),
        ws.bLd,
        0,
        0,
        left,
        0,
        0};
    ctrmm_fallback_pack_do(
        aiv, handle->stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a)), ws.ar, ws.ai,
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(b)), ws.br, ws.bi, ws.flags, guardBlocks, pa, pb);
}

static uint64_t CtrmmUbCapacity()
{
    uint64_t bytes = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, bytes);
    return bytes;
}

static aclblasStatus_t LaunchOrderedCtrmm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex& alpha, const aclblasComplex* a, int lda,
    const aclblasComplex* b, int ldb, aclblasComplex* c, int ldc, uint32_t guardBlocks = 0)
{
    const bool left = side == ACLBLAS_SIDE_LEFT;
    const uint32_t d = left ? m : n;
    const uint32_t aiv = GetAivCoreCount();
    if (aiv == 0)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    CtrmmWorkspace ws{};
    auto status = PrepareCtrmmWorkspace(handle, left ? n : m, left ? m : n, d, ws);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    uint64_t orderedUb = 8ULL * ws.aLd + 64ULL * CTRMM_ORDER_CHUNK;
#ifdef CTRMM_ORDERED_EDGE
    // Select modes 1/2 require an additional 8 KiB API temporary buffer on A2/A3.
    orderedUb += 16ULL * 256 * sizeof(float) + 8192ULL;
#endif
    if (orderedUb > CtrmmUbCapacity())
        return ACLBLAS_STATUS_ALLOC_FAILED;
    if (!guardBlocks)
        LaunchOrderedPacks(handle, left, d, uplo, trans, diag, m, n, a, lda, b, ldb, ws, aiv, 0);
    CtrmmOrderedTiling tiling{
        static_cast<uint32_t>(m),
        static_cast<uint32_t>(n),
        d,
        ws.aLd,
        ws.bLd,
        static_cast<uint32_t>(ldc),
        left,
        uplo == ACLBLAS_UPPER,
        trans != ACLBLAS_OP_N,
        diag == ACLBLAS_UNIT,
        alpha.real,
        alpha.imag};
    tiling.guardBlocks = guardBlocks;
    tiling.rawALd = lda;
    tiling.rawBLd = ldb;
    tiling.conjugate = trans == ACLBLAS_OP_C;
    auto* rawA = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a));
    auto* rawB = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(b));
    ctrmm_ordered_do(
        aiv, handle->stream, guardBlocks ? rawA : ws.ar, ws.ai, guardBlocks ? rawB : ws.br, ws.bi, ws.rr, ws.ri,
        ws.flags, 0, tiling);
    ctrmm_ordered_output_do(
        ldc % 4 == 0 ? std::min<uint32_t>(aiv, n) : 1, handle->stream, ws.rr, ws.ri, reinterpret_cast<uint8_t*>(c),
        tiling, ws.flags);
    return ACLBLAS_STATUS_SUCCESS;
}
#endif

static uint32_t LaunchFastPacks(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    uint32_t d, const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, const CtrmmWorkspace& ws,
    uint32_t aiv, bool transposeOnCube, bool compactLeft)
{
    CtrmmPackTiling pa{
        d,
        d,
        static_cast<uint32_t>(lda),
        ws.aLd,
        1,
        uplo == ACLBLAS_UPPER,
        !transposeOnCube && trans != ACLBLAS_OP_N,
        trans == ACLBLAS_OP_C,
        diag == ACLBLAS_UNIT};
    CtrmmPackTiling pb{
        static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(ldb), ws.bLd, 0, 0, 0, 0, 0};
    auto* srcA = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a));
    auto* srcB = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(b));
    const uint32_t aBlocks = std::min(aiv, d);
    const uint32_t bBlocks = std::min(aiv, static_cast<uint32_t>(n));
    const bool fused = m <= 256 || n <= 256;
    const uint32_t guardBlocks = fused ? std::max(aBlocks, bBlocks) : aBlocks + bBlocks;
    pa.riskFlags = reinterpret_cast<uint64_t>(ws.flags);
    pb.riskFlags = reinterpret_cast<uint64_t>(ws.flags + aBlocks * 32);
    if (m <= 256 || n <= 256) {
        const auto launch = compactLeft ? ctrmm_compact_pack_pair_do : ctrmm_pack_pair_do;
        launch(
            std::min(aiv, std::max(d, uint32_t(n))), handle->stream, srcA, ws.ar, ws.ai, ws.as, srcB, ws.br, ws.bi,
            ws.bs, pa, pb);
    } else {
        ctrmm_pack_do(std::min(aiv, d), handle->stream, srcA, ws.ar, ws.ai, pa);
        ctrmm_pack_do(std::min(aiv, static_cast<uint32_t>(n)), handle->stream, srcB, ws.br, ws.bi, pb);
        ctrmm_sum_do(aiv, handle->stream, ws.ar, ws.ai, ws.as, uint64_t(ws.aLd) * d);
        ctrmm_sum_do(aiv, handle->stream, ws.br, ws.bi, ws.bs, uint64_t(ws.bLd) * n);
    }
    return guardBlocks;
}

static bool CtrmmCubeResourcesAvailable(uint32_t aiv, uint32_t aic)
{
    if (aiv == 0 || aic == 0) {
        OP_LOGE("aclblasCtrmm", "device core query failed");
        return false;
    }
    if (ctrmm_vector_ub_bytes() > CtrmmUbCapacity()) {
        OP_LOGE("aclblasCtrmm", "vector buffers exceed platform UB capacity");
        return false;
    }
    return true;
}

static uint32_t CtrmmOutputBlocks(uint32_t aiv, uint32_t m, uint32_t n, uint32_t ldc)
{
    if (ldc == m)
        return std::min<uint64_t>(aiv, uint64_t(m) * n);
    const uint32_t columnsPerGroup = ldc % 4 == 0 ? 1 : 4;
    return std::min(aiv, (n + columnsPerGroup - 1) / columnsPerGroup);
}

static CtrmmOrderedTiling MakeRawOrderedTiling(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, int d, int lda, int ldb,
    int ldc)
{
    CtrmmOrderedTiling t{};
    t.m = m;
    t.n = n;
    t.d = d;
    t.ldc = ldc;
    t.rawALd = lda;
    t.rawBLd = ldb;
    t.upper = uplo == ACLBLAS_UPPER;
    t.trans = trans != ACLBLAS_OP_N;
    t.conjugate = trans == ACLBLAS_OP_C;
    t.unit = diag == ACLBLAS_UNIT;
    return t;
}

static aclblasStatus_t LaunchCtrmmOutput(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex& alpha, const aclblasComplex* a, int lda,
    const aclblasComplex* b, int ldb, aclblasComplex* c, int ldc, const CtrmmWorkspace& ws, uint32_t aiv, uint32_t aic,
    uint32_t d, uint32_t guardBlocks, uint32_t outBlocks, bool repair, const CtrmmCombineTiling& combine)
{
    constexpr float repairCut = 1800.0f;
    auto* srcA = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a));
    auto* srcB = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(b));
    const bool compactLeft = side == ACLBLAS_SIDE_LEFT && m == 128 && n <= 128 && ldc == m && !repair;
    if ((side == ACLBLAS_SIDE_RIGHT && n <= 128) || compactLeft) {
        auto ordered = MakeRawOrderedTiling(uplo, trans, diag, m, n, d, lda, ldb, ldc);
        ordered.aLd = ws.aLd;
        ordered.bLd = ws.bLd;
        ordered.sideLeft = side == ACLBLAS_SIDE_LEFT;
        ordered.alphaReal = alpha.real;
        ordered.alphaImag = alpha.imag;
        ordered.guardBlocks = guardBlocks;
        const auto launch = compactLeft ? ctrmm_left_guarded_combine_do : ctrmm_guarded_combine_do;
        launch(
            outBlocks, handle->stream, ws.rr, ws.ii, ws.ri, ws.ir, reinterpret_cast<uint8_t*>(c), srcA, srcB, ws.flags,
            combine, ordered);
        return ACLBLAS_STATUS_SUCCESS;
    }
    ctrmm_combine_do(outBlocks, handle->stream, ws.rr, ws.ii, ws.ri, ws.ir, reinterpret_cast<uint8_t*>(c), combine);
    if (repair) {
        ctrmm_repair_do(
            aiv, handle->stream, srcA, srcB, reinterpret_cast<uint8_t*>(c), ws.ar, ws.seed, m, n, lda, ldb, ldc,
            repairCut);
    }
    const auto fallbackStatus =
        LaunchOrderedCtrmm(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, c, ldc, guardBlocks);
    if (fallbackStatus != ACLBLAS_STATUS_SUCCESS)
        return fallbackStatus;
    OP_LOGD("aclblasCtrmm", "m=%d n=%d aiv=%u aic=%u", m, n, aiv, aic);
    OP_LOGI("aclblasCtrmm", "launched pack, GEMM, combine pipeline");
    return ACLBLAS_STATUS_SUCCESS;
}
static aclblasStatus_t LaunchCtrmmKernel(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex& alpha, const aclblasComplex* a, int lda,
    const aclblasComplex* b, int ldb, aclblasComplex* c, int ldc)
{
    const uint32_t aiv = GetAivCoreCount();
    const uint32_t aic = GetAicCoreCount();
    if (!CtrmmCubeResourcesAvailable(aiv, aic)) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const uint32_t d = side == ACLBLAS_SIDE_LEFT ? m : n;
    CtrmmWorkspace ws{};
    auto status = PrepareCtrmmWorkspace(handle, m, n, d, ws);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    const bool transposeOnCube = side == ACLBLAS_SIDE_RIGHT && n <= 128 && trans != ACLBLAS_OP_N;
    const uint32_t guardBlocks = LaunchFastPacks(
        handle, uplo, trans, diag, m, n, d, a, lda, b, ldb, ws, aiv, transposeOnCube,
        side == ACLBLAS_SIDE_LEFT && m == 128 && n <= 128 && ldc == m);
    // This function is only entered with alpha=1, non-aliasing B/C, and d<=4096.
    const bool repair = uint64_t(ldc) * n <= UINT32_MAX && side == ACLBLAS_SIDE_LEFT && uplo == ACLBLAS_UPPER &&
                        trans == ACLBLAS_OP_C && diag == ACLBLAS_NON_UNIT && ldc == m && ldc % 4 == 0;
    const uint32_t prefix = repair ? ctrmm_repair_prefix(d) : 0;
    // Keep the coupled prefix/repair algorithm on its original arithmetic path.
    const bool fourProduct = d >= 2048 && !repair;
    CtrmmCombineTiling combine{
        static_cast<uint32_t>(m),
        static_cast<uint32_t>(n),
        ws.bLd,
        static_cast<uint32_t>(ldc),
        alpha.real,
        alpha.imag,
        fourProduct};
    const uint32_t outBlocks = CtrmmOutputBlocks(aiv, m, n, ldc);
    if (prefix) {
        LaunchCtrmmGemm(handle, side, uplo, trans, m, n, ws, aic, fourProduct, prefix);
        ctrmm_combine_do(outBlocks, handle->stream, ws.rr, ws.ii, ws.ri, ws.ir, ws.seed, combine);
    } else if (repair) {
        const size_t zeroBytes = 2ULL * ws.bLd * n * sizeof(float);
        if (aclrtMemsetAsync(ws.seed, zeroBytes, 0, zeroBytes, handle->stream))
            return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    LaunchCtrmmGemm(handle, side, uplo, trans, m, n, ws, aic, fourProduct, 0, prefix);
    return LaunchCtrmmOutput(
        handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, c, ldc, ws, aiv, aic, d, guardBlocks, outBlocks,
        repair, combine);
}

static aclblasStatus_t ZeroCtrmm(aclblasHandle_t handle, aclblasComplex* c, int m, int n, int ldc)
{
    for (int col = 0; col < n; ++col) {
        const size_t bytes = static_cast<size_t>(m) * sizeof(aclblasComplex);
        if (aclrtMemsetAsync(c + static_cast<size_t>(col) * ldc, bytes, 0, bytes, handle->stream)) {
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}
static bool UseSmallCtrmm(
    aclblasSideMode_t side, int m, int n, int ldc, const aclblasComplex* b, const aclblasComplex* c,
    const aclblasComplex& alpha)
{
    return side == ACLBLAS_SIDE_LEFT && m > 8 && m <= 64 && (m & (m - 1)) == 0 && n <= 64 && ldc == m && b != c &&
           alpha.real == 1.0f && alpha.imag == 0.0f;
}

static bool UseTinyCtrmm(
    aclblasSideMode_t side, int m, int n, int ldc, const aclblasComplex* b, const aclblasComplex* c,
    const aclblasComplex& alpha)
{
    return side == ACLBLAS_SIDE_LEFT && m <= 8 && n <= 8 && ldc == m && b != c && alpha.real == 1.0f &&
           alpha.imag == 0.0f;
}

static aclblasStatus_t LaunchSmallCtrmm(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, aclblasComplex* c, int ldc, bool tiny)
{
    auto t = MakeRawOrderedTiling(uplo, trans, diag, m, n, m, lda, ldb, ldc);
    auto* rawA = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a));
    auto* rawB = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(b));
    if (tiny) {
        ctrmm_tiny_do(
            std::min<uint32_t>(GetAivCoreCount(), (m * n + 3) / 4), handle->stream, rawA, rawB,
            reinterpret_cast<uint8_t*>(c), t);
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint32_t tile = m <= 16 ? 4 : 8;
    const uint32_t groups = (m + tile - 1) / tile;
    const uint32_t workers = std::min<uint32_t>((n + 3) / 4, GetAivCoreCount() / groups);
    if (workers == 0)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    ctrmm_small_do(workers * groups, handle->stream, rawA, rawB, reinterpret_cast<uint8_t*>(c), t);
    return ACLBLAS_STATUS_SUCCESS;
}
static bool ValidCtrmmStorage(
    const aclblasComplex* alpha, const aclblasComplex* c, int d, int m, int lda, int ldb, int ldc)
{
    return alpha != nullptr && c != nullptr && lda >= d && ldb >= m && ldc >= m;
}
} // namespace

aclblasStatus_t aclblasCtrmm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* a, int lda,
    const aclblasComplex* b, int ldb, aclblasComplex* c, int ldc)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    const auto status = ValidateCtrmmParams(side, uplo, trans, diag, m, n);
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0) {
        return status;
    }
    const int d = side == ACLBLAS_SIDE_LEFT ? m : n;
    if (!ValidCtrmmStorage(alpha, c, d, m, lda, ldb, ldc)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha->real == 0.0f && alpha->imag == 0.0f) {
        return ZeroCtrmm(handle, c, m, n, ldc);
    }
    if (a == nullptr || b == nullptr || !ValidCtrmmOverlap(a, b, c, d, m, n, lda, ldb, ldc)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (UseSmallCtrmm(side, m, n, ldc, b, c, *alpha)) {
        return LaunchSmallCtrmm(handle, uplo, trans, diag, m, n, a, lda, b, ldb, c, ldc, false);
    }
    if (UseTinyCtrmm(side, m, n, ldc, b, c, *alpha)) {
        return LaunchSmallCtrmm(handle, uplo, trans, diag, m, n, a, lda, b, ldb, c, ldc, true);
    }
    if (alpha->real != 1.0f || alpha->imag != 0.0f || b == c || d > 4096) {
        return LaunchOrderedCtrmm(handle, side, uplo, trans, diag, m, n, *alpha, a, lda, b, ldb, c, ldc);
    }
    return LaunchCtrmmKernel(handle, side, uplo, trans, diag, m, n, *alpha, a, lda, b, ldb, c, ldc);
}
