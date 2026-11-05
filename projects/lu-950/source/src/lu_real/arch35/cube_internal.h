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
#include <cstddef>
#include <cstdint>

#include "lu950_internal.h"

size_t Lu950CubeWorkspaceBytes();
int Lu950CubeMatmul(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc,
                    void *workspace, aclrtStream stream, bool accumulate);

int Lu950CubeMatmulBatch(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc,
                         void *workspace, aclrtStream stream, bool accumulate, int batch, uint64_t matrixStride);
