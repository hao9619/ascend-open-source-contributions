/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "cann_ops_solver.h"

// Experimental target only. Device pointer arrays, column-major matrices,
// per-matrix Device info. See README.md for unverified requirements.
aclsolverStatus_t Lu950MatinvBatched(aclsolverHandle_t handle, int n, const float *const A[], int lda,
                                     float *const Ainv[], int lda_inv, int *info, int batchSize);

aclsolverStatus_t Lu950GetriBatched(aclsolverHandle_t handle, int n, const float *const *a, int lda, const int *pivots,
                                    float *const *c, int ldc, int *info, int batch);
