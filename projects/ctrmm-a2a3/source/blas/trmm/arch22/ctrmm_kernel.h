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
#include <cstdint>
#include "ctrmm_tiling_data.h"
#include "common/helper/complex_blas3_tiling_data.h"

void ctrmm_pack_do(uint32_t blocks, void* stream, uint8_t* src, uint8_t* re, uint8_t* im, CtrmmPackTiling tiling);
void ctrmm_gemm_do(
    uint32_t blocks, void* stream, uint8_t* lr, uint8_t* li, uint8_t* rr, uint8_t* ri, uint8_t* ls, uint8_t* rs,
    uint8_t* out0, uint8_t* out1, uint8_t* out2, uint8_t* out3, CtrmmGemmTiling tiling);
uint32_t ctrmm_tile_m();
uint32_t ctrmm_tile_n();
void ctrmm_combine_do(
    uint32_t blocks, void* stream, uint8_t* rr, uint8_t* ii, uint8_t* ri, uint8_t* ir, uint8_t* out,
    CtrmmCombineTiling tiling);
uint32_t ctrmm_single_k();

uint32_t ctrmm_vector_ub_bytes();

void ctrmm_ordered_do(
    uint32_t blocks, void* stream, uint8_t* ar, uint8_t* ai, uint8_t* br, uint8_t* bi, uint8_t* rr, uint8_t* ri,
    uint8_t* flags, uint32_t classifyBlocks, CtrmmOrderedTiling tiling);
void ctrmm_ordered_output_do(
    uint32_t blocks, void* stream, uint8_t* re, uint8_t* im, uint8_t* out, CtrmmOrderedTiling tiling,
    uint8_t* flags = nullptr);
void ctrmm_classify_do(uint32_t blocks, void* stream, uint8_t* input, uint8_t* flags, uint64_t count);

void ctrmm_sum_do(uint32_t blocks, void* stream, uint8_t* re, uint8_t* im, uint8_t* sum, uint64_t count);

void ctrmm_repair_do(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* scratch, uint8_t* seed, uint32_t m,
    uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float cut);

uint64_t ctrmm_repair_scratch_per_core(uint32_t d);

uint32_t ctrmm_repair_prefix(uint32_t d);

void ctrmm_fallback_pack_do(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* ar, uint8_t* ai, uint8_t* b, uint8_t* br, uint8_t* bi,
    uint8_t* flags, uint32_t guardBlocks, CtrmmPackTiling pa, CtrmmPackTiling pb);

void ctrmm_pack_pair_do(
    uint32_t blocks, void* stream, uint8_t* a, uint8_t* ar, uint8_t* ai, uint8_t* as, uint8_t* b, uint8_t* br,
    uint8_t* bi, uint8_t* bs, CtrmmPackTiling pa, CtrmmPackTiling pb);
