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
#include "gemm_grouped_batched_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ inline void GroupedScalarCompute(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmGrouped22Params& t, uint32_t first, uint32_t step)
{
    GlobalTensor<float> ag, bg, cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    const uint64_t total = static_cast<uint64_t>(t.m) * t.n;
    for (uint64_t index = first; index < total; index += step) {
        const uint64_t row = index % t.m;
        const uint64_t col = index / t.m;
        const uint64_t out = col * t.ldc + row;
        const float old = t.beta == 0.0f ? 0.0f : t.beta * cg.GetValue(out);
        float result = old;
        if (t.alpha != 0.0f && t.k != 0) {
            if (!t.transA) {
                // Netlib NN/NT scales each B element before updating C.
                for (int p = 0; p < t.k; ++p) {
                    const float bv = bg.GetValue(t.transB ? static_cast<uint64_t>(p) * t.ldb + col : col * t.ldb + p);
                    // Netlib 3.10 evaluates zero products too, preserving NaN/Inf propagation.
                    result += (t.alpha * bv) * ag.GetValue(static_cast<uint64_t>(p) * t.lda + row);
                }
            } else {
                float sum = 0.0f;
                for (int p = 0; p < t.k; ++p) {
                    const float av = ag.GetValue(row * t.lda + p);
                    const float bv = bg.GetValue(t.transB ? static_cast<uint64_t>(p) * t.ldb + col : col * t.ldb + p);
                    sum += av * bv;
                }
                result = t.alpha * sum + old;
            }
        }
        cg.SetValue(out, result);
    }
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(cg);
}

extern "C" __global__ __aicore__ void gemm_grouped_batched_22(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmGrouped22Params t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GroupedScalarCompute(a, b, c, t, GetBlockIdx(), GetBlockNum());
}

// Check input magnitudes before the fast vector path. Extreme products must
// preserve the scalar reference's overflow and nonfinite accumulation behavior.
__aicore__ inline bool SmallExtremeInput(
    LocalTensor<float> value, LocalTensor<int32_t> mask, LocalTensor<int32_t> magnitude, LocalTensor<float> converted,
    LocalTensor<float> reduced, uint32_t count)
{
    And(magnitude.ReinterpretCast<uint16_t>(), value.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(),
        count * 2);
    PipeBarrier<PIPE_V>();
    Cast(converted, magnitude, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    ReduceMax(reduced, converted, magnitude.ReinterpretCast<float>(), count, false);
    PipeBarrier<PIPE_ALL>();
    return reduced.GetValue(0) >= static_cast<float>(0x71000000);
}

// Outer-product updates keep K order while reusing each matrix from UB.
// This specialization only handles NN, alpha=1, beta=0; other contracts keep
// the scalar path.
__aicore__ inline void GroupedSmallVectorCompute(const GemmGrouped22Batch& batch)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 8192 * sizeof(float));
    auto all = storage.Get<float>();
    auto aBlock = all;
    auto bBlock = all[1024];
    auto accum = all[2048];
    auto aValues = all[2304];
    auto bValues = all[2560];
    auto aIndices = all[3328].ReinterpretCast<uint32_t>();
    auto bIndices = all[3584].ReinterpretCast<uint32_t>();
    auto extremeMask = all[4096].ReinterpretCast<int32_t>();
    auto magnitude = all[5120].ReinterpretCast<int32_t>();
    auto converted = all[6144];
    auto reduced = all[7168];
    Duplicate(extremeMask, 0x7fffffff, 1024);
    const auto& t = batch.group;
    const uint32_t packedM = (t.m + 7U) / 8U * 8U;
    const uint32_t packedK = (t.k + 7U) / 8U * 8U;
    const uint32_t count = packedM * t.n;
    auto indices = all[2816].ReinterpretCast<int32_t>();
    auto indexMask = all[3072].ReinterpretCast<int32_t>();
    CreateVecIndex(indices, static_cast<int32_t>(0), count);
    Duplicate(indexMask, static_cast<int32_t>(packedM - 1), count);
    PipeBarrier<PIPE_V>();
    And(aIndices.ReinterpretCast<uint16_t>(), indices.ReinterpretCast<uint16_t>(),
        indexMask.ReinterpretCast<uint16_t>(), count * 2);
    ShiftRight(bIndices, indices.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(packedM == 16 ? 4 : 3), count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(aIndices, aIndices, static_cast<uint32_t>(2), count);
    Muls(
        bIndices.ReinterpretCast<int32_t>(), bIndices.ReinterpretCast<int32_t>(), static_cast<int32_t>(packedK * 4),
        count);
    const DataCopyPadExtParams<float> pad{true, 0, 0, 0};
    const DataCopyExtParams copyA{
        static_cast<uint16_t>(t.k), static_cast<uint32_t>(t.m) * 4U, static_cast<uint32_t>(t.lda - t.m) * 4U, 0, 0};
    const DataCopyExtParams copyB{
        static_cast<uint16_t>(t.n), static_cast<uint32_t>(t.k) * 4U, static_cast<uint32_t>(t.ldb - t.k) * 4U, 0, 0};
    const DataCopyExtParams copyC{
        static_cast<uint16_t>(t.n), static_cast<uint32_t>(t.m) * 4U, 0, static_cast<uint32_t>(t.ldc - t.m) * 4U, 0};
    PipeBarrier<PIPE_ALL>();
    for (uint32_t matrix = GetBlockIdx(); matrix < batch.count; matrix += GetBlockNum()) {
        GlobalTensor<float> ag, bg, cg;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batch.a[matrix + batch.matrixBase]));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batch.b[matrix + batch.matrixBase]));
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batch.c[matrix + batch.matrixBase]));
        DataCopyPad(aBlock, ag, copyA, pad);
        DataCopyPad(bBlock, bg, copyB, pad);
        Duplicate(accum, 0.0f, count);
        PipeBarrier<PIPE_ALL>();
        if (SmallExtremeInput(aBlock, extremeMask, magnitude, converted, reduced, packedM * t.k) ||
            SmallExtremeInput(bBlock, extremeMask, magnitude, converted, reduced, packedK * t.n)) {
            GroupedScalarCompute(
                reinterpret_cast<GM_ADDR>(batch.a[matrix + batch.matrixBase]),
                reinterpret_cast<GM_ADDR>(batch.b[matrix + batch.matrixBase]),
                reinterpret_cast<GM_ADDR>(batch.c[matrix + batch.matrixBase]), t, 0, 1);
            continue;
        }
        for (int inner = 0; inner < t.k; ++inner) {
            Gather(aValues, aBlock, aIndices, inner * packedM * 4U, count);
            Gather(bValues, bBlock, bIndices, inner * 4U, count);
            PipeBarrier<PIPE_V>();
            // Evaluate every product, including zero times NaN/Inf, like Netlib 3.10.
            MulAddDst(accum, aValues, bValues, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(cg, accum, copyC);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" __global__ __aicore__ void gemm_grouped_batched_scalar_batch22(GemmGrouped22Batch batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!batch.group.transA && !batch.group.transB && batch.group.alpha == 1.0f && batch.group.beta == 0.0f &&
        batch.group.k > 0) {
        GroupedSmallVectorCompute(batch);
        return;
    }
    for (uint32_t i = GetBlockIdx(); i < batch.count; i += GetBlockNum()) {
        GroupedScalarCompute(
            reinterpret_cast<GM_ADDR>(batch.a[i]), reinterpret_cast<GM_ADDR>(batch.b[i]),
            reinterpret_cast<GM_ADDR>(batch.c[i]), batch.group, 0, 1);
    }
}

void GemmGrouped22Launch(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmGrouped22Params params)
{
    gemm_grouped_batched_22<<<blocks, nullptr, stream>>>(a, b, c, params);
}

void GemmGrouped22ScalarBatch(uint32_t blocks, void* stream, GemmGrouped22Batch batch)
{
    gemm_grouped_batched_scalar_batch22<<<blocks, nullptr, stream>>>(batch);
}

#include "lib/matmul_intf.h"

namespace {
using namespace matmul;
template <bool SMALL>
constexpr MatmulShapeParams GROUPED_SHAPE =
    SMALL ? MatmulShapeParams{128, 128, 4096, 128, 128, 64} : MatmulShapeParams{512, 1024, 4096, 128, 256, 64};
constexpr MatmulFuncParams GROUPED_FUNC{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                                        true,  true};
constexpr MatmulBiasParams GROUPED_BIAS{false};
template <bool SMALL>
constexpr MatmulConfig GROUPED_CONFIG =
    GetMMConfig<MatmulConfigMode::CONFIG_MDL>(GROUPED_SHAPE<SMALL>, GROUPED_FUNC, GROUPED_BIAS);
template <bool TRANS>
using GroupedInput = MatmulType<TPosition::GM, CubeFormat::ND, float, TRANS>;
using GroupedOutput = MatmulType<TPosition::GM, CubeFormat::ND, float>;
template <bool TA, bool TB, bool SMALL>
constexpr MatmulApiStaticTiling GROUPED_TILING =
    GetMatmulApiTiling<GroupedInput<TB>, GroupedInput<TA>, GroupedOutput, GroupedOutput>(GROUPED_CONFIG<SMALL>);

template <bool TA, bool TB, bool SMALL>
__aicore__ inline void GroupedCubeCompute(const GemmGrouped22Batch& batch, TPipe& pipe)
{
    const auto& t = batch.group;
    const uint64_t rows = (static_cast<uint64_t>(t.n) + batch.tileRows - 1) / batch.tileRows;
    const uint64_t cols = (static_cast<uint64_t>(t.m) + batch.tileCols - 1) / batch.tileCols;
    const uint32_t firstTask = (GetBlockIdx() + GetBlockNum() - batch.cubeTaskOffset % GetBlockNum()) % GetBlockNum();
    if (firstTask >= rows * cols * batch.count)
        return;
    MatmulImpl<GroupedInput<TB>, GroupedInput<TA>, GroupedOutput, GroupedOutput, GROUPED_TILING<TA, TB, SMALL>> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);

    // Parameters are launch values; host arrays may be released immediately after
    // API return. Schedule output tiles across all matrices in this chunk, rather
    // than assigning whole matrices.
    for (uint64_t task = firstTask; task < rows * cols * batch.count; task += GetBlockNum()) {
        const uint32_t matrix = task / (rows * cols);
        const uint64_t row = (task / cols % rows) * batch.tileRows;
        const uint64_t col = (task % cols) * batch.tileCols;
        GlobalTensor<float> ag, bg, cg;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batch.a[matrix + batch.matrixBase]));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batch.b[matrix + batch.matrixBase]));
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batch.output[matrix + batch.matrixBase]));
        // Column-major C is row-major C^T = op(B)^T * op(A)^T.
        mm.SetOrgShape(TB ? t.ldb : t.n, TA ? t.m : t.lda, TB ? t.k : t.ldb, TA ? t.lda : t.k, batch.outputLd);
        for (int64_t startK = 0; startK < t.k; startK += 4096) {
            mm.SetSingleShape(
                t.n - row < batch.tileRows ? t.n - row : batch.tileRows,
                t.m - col < batch.tileCols ? t.m - col : batch.tileCols, t.k - startK < 4096 ? t.k - startK : 4096);
            mm.SetTensorA(bg[TB ? startK * t.ldb + row : row * t.ldb + startK], TB);
            mm.SetTensorB(ag[TA ? col * t.lda + startK : startK * t.lda + col], TA);
            mm.IterateAll(cg[row * batch.outputLd + col], startK != 0);
            mm.End();
        }
    }
}
} // namespace

template <bool SMALL>
__aicore__ inline void GroupedCubeSelect(const GemmGrouped22Batch& batch, TPipe& pipe)
{
    if (batch.group.transA) {
        if (batch.group.transB) {
            GroupedCubeCompute<true, true, SMALL>(batch, pipe);
        } else {
            GroupedCubeCompute<true, false, SMALL>(batch, pipe);
        }
    } else {
        if (batch.group.transB) {
            GroupedCubeCompute<false, true, SMALL>(batch, pipe);
        } else {
            GroupedCubeCompute<false, false, SMALL>(batch, pipe);
        }
    }
}

__aicore__ inline void GroupedCubeDispatch(const GemmGrouped22Batch& batch, TPipe& pipe)
{
    // Mixed kernels keep one Matmul configuration; small standalone launches
    // select their separate configuration at the Cube-only entry.
    GroupedCubeSelect<false>(batch, pipe);
}

extern "C" __global__ __aicore__ __cube__ void gemm_grouped_batched_cube22(GemmGrouped22Batch batch)
{
    TPipe pipe;
    if (batch.group.m <= 64 && batch.group.n <= 64 && batch.group.k <= 64)
        GroupedCubeSelect<true>(batch, pipe);
    else
        GroupedCubeSelect<false>(batch, pipe);
}

namespace {
// Compare magnitudes as positive IEEE-754 bit patterns; this also catches every
// NaN and Inf.
__aicore__ inline bool HasExtremeInput(
    GM_ADDR address, int rows, int cols, int ld, LocalTensor<int32_t> input, LocalTensor<int32_t> mask,
    LocalTensor<int32_t> magnitude, LocalTensor<float> converted, LocalTensor<float> reduced, uint32_t worker = 0,
    uint32_t workers = 1)
{
    GlobalTensor<int32_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(address));
    const uint64_t span = rows == ld ? static_cast<uint64_t>(rows) * cols : rows;
    const int segments = rows == ld ? 1 : cols;
    for (int col = 0; col < segments; ++col) {
        for (uint64_t row = static_cast<uint64_t>(worker) * 4096; row < span; row += 4096ULL * workers) {
            const uint32_t count = span - row < 4096 ? span - row : 4096;
            const DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
            const DataCopyPadExtParams<int32_t> pad{true, 0, 0, 0};
            DataCopyPad(input, gm[static_cast<uint64_t>(col) * ld + row], copy, pad);
            PipeBarrier<PIPE_ALL>();
            // And counts 16-bit lanes; cover both halves of each int32 value.
            And(magnitude.ReinterpretCast<uint16_t>(), input.ReinterpretCast<uint16_t>(),
                mask.ReinterpretCast<uint16_t>(), count * 2);
            PipeBarrier<PIPE_ALL>();
            Cast(converted, magnitude, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_ALL>();
            ReduceMax(reduced, converted, input.ReinterpretCast<float>(), count, false);
            PipeBarrier<PIPE_ALL>();
            if (reduced.GetValue(0) >= static_cast<float>(0x71000000)) { // Conservative magnitude threshold near 6e29
                return true;
            }
        }
    }
    return false;
}
} // namespace

__aicore__ inline void GroupedScanBlocks(const GemmGrouped22Batch& batch, TPipe& pipe)
{
    if (GetBlockIdx() >= batch.scanWorkers * batch.count)
        return;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 5 * 4096 * sizeof(int32_t));
    auto all = storage.Get<int32_t>();
    auto input = all;
    auto mask = all[4096];
    auto magnitude = all[8192];
    auto converted = all[12288].ReinterpretCast<float>();
    auto reduced = all[16384].ReinterpretCast<float>();
    Duplicate(mask, 0x7fffffff, 4096);
    PipeBarrier<PIPE_ALL>();
    // In a 1:2 mixed kernel, AIV indices span [0, 2*GetBlockNum()).
    for (uint32_t task = GetBlockIdx(); task < batch.scanWorkers * batch.count; task += 2 * GetBlockNum()) {
        const uint32_t matrix = task / batch.scanWorkers;
        const uint32_t worker = task % batch.scanWorkers;
        const auto& t = batch.group;
        const bool extreme =
            HasExtremeInput(
                reinterpret_cast<GM_ADDR>(batch.a[matrix + batch.matrixBase]), t.transA ? t.k : t.m,
                t.transA ? t.m : t.k, t.lda, input, mask, magnitude, converted, reduced, worker, batch.scanWorkers) ||
            HasExtremeInput(
                reinterpret_cast<GM_ADDR>(batch.b[matrix + batch.matrixBase]), t.transB ? t.n : t.k,
                t.transB ? t.k : t.n, t.ldb, input, mask, magnitude, converted, reduced, worker, batch.scanWorkers);
        GlobalTensor<uint32_t> flags;
        flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(batch.flags));
        flags.SetValue(task * 16, extreme);
        DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE>(flags);
    }
}

extern "C" __global__ __aicore__ __schedmode__(1) void gemm_grouped_batched_cube_scan_repair22(GemmGrouped22Batch batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    if ASCEND_IS_AIC {
        GroupedCubeDispatch(batch, pipe);
    } else {
        GroupedScanBlocks(batch, pipe);
    }
    // Every Cube and vector core reaches this barrier, including scan workers with no input.
    // Batch scheduling reserves the full launch to avoid multi-stream synchronization deadlocks.
    SyncAll<false>();
    if ASCEND_IS_AIV {
        GlobalTensor<uint32_t> flags;
        flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(batch.flags));
        DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE>(flags);
        for (uint32_t matrix = GetBlockIdx(); matrix < batch.count; matrix += 2 * GetBlockNum()) {
            bool extreme = false;
            for (uint32_t worker = 0; worker < batch.scanWorkers; ++worker)
                extreme |= flags.GetValue((matrix * batch.scanWorkers + worker) * 16) != 0;
            if (extreme) {
                GroupedScalarCompute(
                    reinterpret_cast<GM_ADDR>(batch.a[matrix]), reinterpret_cast<GM_ADDR>(batch.b[matrix]),
                    reinterpret_cast<GM_ADDR>(batch.c[matrix]), batch.group, 0, 1);
            }
        }
    }
}

extern "C" __global__ __aicore__ __schedmode__(1) void gemm_grouped_batched_multi_small22(GemmGrouped22Batch batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const uint64_t flagBase = batch.flags;
    const GemmGrouped22Params firstGroup = batch.group;
    if ASCEND_IS_AIC {
        uint32_t matrixBase = 0, taskBase = 0;
        for (uint32_t g = 0; g < batch.groupCount; ++g) {
            batch.group = g == 0 ? firstGroup : batch.extraGroups[g - 1];
            batch.count = batch.groupEnds[g] - matrixBase;
            batch.matrixBase = matrixBase;
            batch.outputLd = batch.group.ldc;
            batch.tileRows = batch.groupRows[g];
            batch.tileCols = batch.groupCols[g];
            batch.cubeTaskOffset = taskBase;
            const uint32_t tasks = ((batch.group.m + batch.tileCols - 1) / batch.tileCols) *
                                   ((batch.group.n + batch.tileRows - 1) / batch.tileRows) * batch.count;
            const uint32_t first = (GetBlockIdx() + GetBlockNum() - taskBase % GetBlockNum()) % GetBlockNum();
            if (first < tasks) {
                TPipe pipe;
                GroupedCubeDispatch(batch, pipe);
            }
            taskBase += tasks;
            matrixBase = batch.groupEnds[g];
        }
    } else {
        TPipe pipe;
        TBuf<TPosition::VECCALC> storage;
        pipe.InitBuffer(storage, 5 * 4096 * sizeof(int32_t));
        auto all = storage.Get<int32_t>();
        auto input = all;
        auto mask = all[4096];
        auto magnitude = all[8192];
        auto converted = all[12288].ReinterpretCast<float>();
        auto reduced = all[16384].ReinterpretCast<float>();
        Duplicate(mask, 0x7fffffff, 4096);
        PipeBarrier<PIPE_ALL>();
        uint32_t matrixBase = 0, taskBase = 0;
        const uint32_t blocks = 2 * GetBlockNum();
        for (uint32_t g = 0; g < batch.groupCount; ++g) {
            const auto& t = g == 0 ? firstGroup : batch.extraGroups[g - 1];
            const uint32_t workers = batch.groupWorkers[g];
            const uint32_t tasks = workers * (batch.groupEnds[g] - matrixBase);
            const uint32_t first = (GetBlockIdx() + blocks - taskBase % blocks) % blocks;
            GlobalTensor<uint32_t> flags;
            flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(flagBase + g * 4096));
            for (uint32_t task = first; task < tasks; task += blocks) {
                const uint32_t matrix = matrixBase + task / workers;
                const uint32_t worker = task % workers;
                const bool extreme = HasExtremeInput(
                                         reinterpret_cast<GM_ADDR>(batch.a[matrix]), t.m, t.k, t.lda, input, mask,
                                         magnitude, converted, reduced, worker, workers) ||
                                     HasExtremeInput(
                                         reinterpret_cast<GM_ADDR>(batch.b[matrix]), t.k, t.n, t.ldb, input, mask,
                                         magnitude, converted, reduced, worker, workers);
                flags.SetValue(task * 16, extreme);
            }
            DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE>(flags);
            taskBase += tasks;
            matrixBase = batch.groupEnds[g];
        }
    }
    // Independent groups need only one barrier, after all Cube and scan tasks.
    SyncAll<false>();
    if ASCEND_IS_AIV {
        const uint32_t total = batch.groupEnds[batch.groupCount - 1];
        for (uint32_t matrix = GetBlockIdx(); matrix < total; matrix += 2 * GetBlockNum()) {
            uint32_t g = 0;
            while (matrix >= batch.groupEnds[g])
                ++g;
            const uint32_t localMatrix = matrix - (g == 0 ? 0 : batch.groupEnds[g - 1]);
            const auto& t = g == 0 ? firstGroup : batch.extraGroups[g - 1];
            GlobalTensor<uint32_t> flags;
            flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(flagBase + g * 4096));
            DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE>(flags);
            bool extreme = false;
            for (uint32_t worker = 0; worker < batch.groupWorkers[g]; ++worker)
                extreme |= flags.GetValue((localMatrix * batch.groupWorkers[g] + worker) * 16) != 0;
            if (extreme)
                GroupedScalarCompute(
                    reinterpret_cast<GM_ADDR>(batch.a[matrix]), reinterpret_cast<GM_ADDR>(batch.b[matrix]),
                    reinterpret_cast<GM_ADDR>(batch.c[matrix]), t, 0, 1);
        }
    }
}

__aicore__ inline void GroupedCombine(
    GM_ADDR temp, GM_ADDR c, const GemmGrouped22Batch& batch, LocalTensor<float> value, LocalTensor<float> previous,
    LocalTensor<float> scaled)
{
    const auto& t = batch.group;
    GlobalTensor<float> tempGm, outputGm;
    tempGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(temp));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    for (int col = 0; col < t.n; ++col) {
        for (int row = 0; row < t.m; row += 256) {
            const uint32_t count = t.m - row < 256 ? t.m - row : 256;
            const DataCopyExtParams copy{1, count * 4U, 0, 0, 0};
            const DataCopyPadExtParams<float> pad{true, 0, 0, 0};
            DataCopyPad(value, tempGm[static_cast<uint64_t>(col) * batch.outputLd + row], copy, pad);
            if (t.beta != 0.0f) {
                DataCopyPad(previous, outputGm[static_cast<uint64_t>(col) * t.ldc + row], copy, pad);
            }
            PipeBarrier<PIPE_ALL>();
            Muls(value, value, t.alpha, count);
            if (t.beta != 0.0f) {
                Muls(scaled, previous, t.beta, count);
                PipeBarrier<PIPE_ALL>();
                Add(value, value, scaled, count);
            }
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(outputGm[static_cast<uint64_t>(col) * t.ldc + row], value, copy);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

// Cube overflow/NaN order can differ from Netlib. Recompute only affected
// matrices in stream order.
extern "C" __global__ __aicore__ __vector__ void gemm_grouped_batched_repair22(GemmGrouped22Batch batch)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 5 * 4096 * sizeof(int32_t));
    auto all = storage.Get<int32_t>();
    auto input = all;
    auto mask = all[4096];
    auto magnitude = all[8192];
    auto converted = all[12288].ReinterpretCast<float>();
    auto reduced = all[16384].ReinterpretCast<float>();
    Duplicate(mask, 0x7fffffff, 4096);
    PipeBarrier<PIPE_ALL>();
    const auto& t = batch.group;
    for (uint32_t matrix = GetBlockIdx(); matrix < batch.count; matrix += GetBlockNum()) {
        GM_ADDR a = reinterpret_cast<GM_ADDR>(batch.a[matrix + batch.matrixBase]);
        GM_ADDR b = reinterpret_cast<GM_ADDR>(batch.b[matrix + batch.matrixBase]);
        GM_ADDR c = reinterpret_cast<GM_ADDR>(batch.c[matrix + batch.matrixBase]);
        bool extreme = false;
        if (batch.scanWorkers > 0) {
            GlobalTensor<uint32_t> flags;
            flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(batch.flags));
            DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE>(flags);
            for (uint32_t worker = 0; worker < batch.scanWorkers; ++worker) {
                extreme |= flags.GetValue((matrix * batch.scanWorkers + worker) * 16) != 0;
            }
        } else {
            extreme =
                HasExtremeInput(
                    a, t.transA ? t.k : t.m, t.transA ? t.m : t.k, t.lda, input, mask, magnitude, converted, reduced) ||
                HasExtremeInput(
                    b, t.transB ? t.n : t.k, t.transB ? t.k : t.n, t.ldb, input, mask, magnitude, converted, reduced);
        }
        if (extreme) {
            GroupedScalarCompute(a, b, c, t, 0, 1);
        } else if (batch.combine) {
            GroupedCombine(
                reinterpret_cast<GM_ADDR>(batch.output[matrix + batch.matrixBase]), c, batch,
                input.ReinterpretCast<float>(), mask.ReinterpretCast<float>(), magnitude.ReinterpretCast<float>());
            Duplicate(mask, 0x7fffffff, 4096);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

void GemmGrouped22Cube(uint32_t blocks, uint32_t vectors, void* stream, GemmGrouped22Batch batch)
{
    if (batch.scanWorkers > 0) {
        if (batch.groupCount > 1)
            gemm_grouped_batched_multi_small22<<<blocks, nullptr, stream>>>(batch);
        else
            gemm_grouped_batched_cube_scan_repair22<<<blocks, nullptr, stream>>>(batch);
    } else {
        gemm_grouped_batched_cube22<<<blocks, nullptr, stream>>>(batch);
        const uint32_t repairBlocks = vectors < batch.count ? vectors : batch.count;
        gemm_grouped_batched_repair22<<<repairBlocks, nullptr, stream>>>(batch);
    }
}

extern "C" __global__ __aicore__ __vector__ void gemm_grouped_batched_ordered_probe22(GemmGrouped22Batch batch)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 5 * 4096 * sizeof(int32_t));
    auto all = storage.Get<int32_t>();
    auto input = all;
    auto mask = all[4096];
    auto magnitude = all[8192];
    auto converted = all[12288].ReinterpretCast<float>();
    auto reduced = all[16384].ReinterpretCast<float>();
    Duplicate(mask, 0x7fffffff, 4096);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<uint32_t> flags;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(batch.flags));
    const auto& t = batch.group;
    for (uint32_t matrix = GetBlockIdx(); matrix < batch.count; matrix += GetBlockNum()) {
        GM_ADDR a = reinterpret_cast<GM_ADDR>(batch.a[matrix + batch.matrixBase]);
        GM_ADDR b = reinterpret_cast<GM_ADDR>(batch.b[matrix + batch.matrixBase]);
        GM_ADDR c = reinterpret_cast<GM_ADDR>(batch.c[matrix + batch.matrixBase]);
        const bool extreme =
            HasExtremeInput(
                a, t.transA ? t.k : t.m, t.transA ? t.m : t.k, t.lda, input, mask, magnitude, converted, reduced) ||
            HasExtremeInput(
                b, t.transB ? t.n : t.k, t.transB ? t.k : t.n, t.ldb, input, mask, magnitude, converted, reduced);
        // One 64-byte slot per matrix keeps scalar writers on distinct cache lines.
        flags.SetValue(matrix * 16, extreme);
        if (extreme)
            GroupedScalarCompute(a, b, c, t, 0, 1);
    }
    DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE>(flags);
}

extern "C" __global__ __aicore__ __vector__ void gemm_grouped_batched_ordered22(GemmGrouped22Batch batch)
{
    constexpr uint32_t ROWS = 64;
    constexpr uint32_t INNER = 256;
    TPipe pipe;
    TBuf<TPosition::VECCALC> aStorage, bStorage, workStorage;
    pipe.InitBuffer(aStorage, ROWS * INNER * sizeof(float));
    pipe.InitBuffer(bStorage, INNER * 8 * sizeof(float));
    pipe.InitBuffer(workStorage, 5 * ROWS * sizeof(float));
    auto aBlock = aStorage.Get<float>();
    auto bBlock = bStorage.Get<float>();
    auto work = workStorage.Get<float>();
    auto input = work;
    auto accum = work[ROWS];
    auto previous = work[2 * ROWS];
    auto multiplier = work[3 * ROWS];
    auto indices = work[4 * ROWS].ReinterpretCast<uint32_t>();
    for (uint32_t r = 0; r < ROWS; ++r)
        indices.SetValue(r, r * INNER * sizeof(float));
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<uint32_t> flags;
    flags.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(batch.flags));
    const auto& t = batch.group;
    const uint64_t rowBlocks = (static_cast<uint64_t>(t.m) + ROWS - 1) / ROWS;
    const uint64_t perMatrix = rowBlocks * t.n;
    const DataCopyPadExtParams<float> pad{true, 0, 0, 0};
    for (uint64_t task = GetBlockIdx(); task < perMatrix * batch.count; task += GetBlockNum()) {
        const uint32_t matrix = task / perMatrix;
        if (flags.GetValue(matrix * 16))
            continue;
        const uint64_t row = (task % rowBlocks) * ROWS;
        const uint64_t col = (task / rowBlocks) % t.n;
        const uint32_t count = t.m - row < ROWS ? t.m - row : ROWS;
        GlobalTensor<float> ag, bg, cg;
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batch.a[matrix + batch.matrixBase]));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batch.b[matrix + batch.matrixBase]));
        cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batch.c[matrix + batch.matrixBase]));
        const uint64_t out = col * t.ldc + row;
        const DataCopyExtParams outputCopy{1, count * 4U, 0, 0, 0};
        if (!t.transA && t.beta != 0.0f) {
            DataCopyPad(previous, cg[out], outputCopy, pad);
            PipeBarrier<PIPE_ALL>();
            Muls(accum, previous, t.beta, count);
        } else {
            Duplicate(accum, 0.0f, count);
        }
        PipeBarrier<PIPE_ALL>();
        for (int64_t startK = 0; startK < t.k; startK += INNER) {
            const uint32_t innerCount = t.k - startK < INNER ? t.k - startK : INNER;
            if (t.transA) {
                const uint32_t alignedK = (innerCount + 7U) / 8 * 8;
                const DataCopyExtParams copyA{
                    static_cast<uint16_t>(count), innerCount * 4U, static_cast<uint32_t>(t.lda - innerCount) * 4U,
                    (INNER - alignedK) / 8U, 0};
                DataCopyPad(aBlock, ag[row * t.lda + startK], copyA, pad);
            }
            if (t.transB) {
                const DataCopyExtParams copyB{
                    static_cast<uint16_t>(innerCount), 4, static_cast<uint32_t>(t.ldb - 1) * 4U, 0, 0};
                DataCopyPad(bBlock, bg[static_cast<uint64_t>(startK) * t.ldb + col], copyB, pad);
            } else {
                const DataCopyExtParams copyB{1, innerCount * 4U, 0, 0, 0};
                DataCopyPad(bBlock, bg[col * t.ldb + startK], copyB, pad);
            }
            PipeBarrier<PIPE_ALL>();
            // Keep the same FP32 reduction order as Netlib, vectorizing independent
            // output rows.
            for (uint32_t inner = 0; inner < innerCount; ++inner) {
                const float bv = bBlock.GetValue(t.transB ? inner * 8 : inner);
                if (t.transA) {
                    Gather(input, aBlock, indices, inner * 4U, count);
                } else {
                    DataCopyPad(input, ag[(startK + inner) * t.lda + row], outputCopy, pad);
                }
                Duplicate(multiplier, t.transA ? bv : t.alpha * bv, count);
                PipeBarrier<PIPE_ALL>();
                MulAddDst(accum, input, multiplier, static_cast<int32_t>(count));
                PipeBarrier<PIPE_ALL>();
            }
        }
        if (t.transA) {
            if (t.beta != 0.0f) {
                DataCopyPad(previous, cg[out], outputCopy, pad);
                PipeBarrier<PIPE_ALL>();
                Muls(previous, previous, t.beta, count);
                Duplicate(multiplier, t.alpha, count);
                PipeBarrier<PIPE_ALL>();
                MulAddDst(previous, accum, multiplier, static_cast<int32_t>(count));
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(cg[out], previous, outputCopy);
            } else {
                Muls(accum, accum, t.alpha, count);
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(cg[out], accum, outputCopy);
            }
        } else {
            DataCopyPad(cg[out], accum, outputCopy);
        }
        PipeBarrier<PIPE_ALL>();
    }
}

void GemmGrouped22Ordered(uint32_t vectors, void* stream, GemmGrouped22Batch batch)
{
    gemm_grouped_batched_ordered_probe22<<<(vectors < batch.count ? vectors : batch.count), nullptr, stream>>>(batch);
    gemm_grouped_batched_ordered22<<<vectors, nullptr, stream>>>(batch);
}
