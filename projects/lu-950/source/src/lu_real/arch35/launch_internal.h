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

// Internal launch and workspace interfaces shared by host and kernel translation units.
void Lu950InfoLaunch(int *info, int count, int value, void *stream);

void Lu950FactorLaunch(int m, int n, float *a, int lda, int *pivots, int *info, int panel, unsigned blocks,
                       void *stream);
void Lu950FactorInnerLaunch(int m, int end, int first, float *a, int lda, int start, int nb, int *pivots, int *info,
                            unsigned blocks, void *stream);
int Lu950FactorCubeLaunch(int m, int n, float *a, int lda, int *pivots, int *info, int panel, unsigned blocks,
                          float *negativeUpper, void *workspace, void *stream);

void Lu950GetriLaunch(int n, float *a, int lda, const int *pivots, int *info, float *workspace, unsigned blocks,
                      void *stream);
void Lu950NoPivotGetriLaunch(int n, float *a, int lda, int *info, float *workspace, unsigned blocks, void *stream);
int Lu950LargeGetriLaunch(int n, float *a, int lda, const int *pivots, int *info, void *workspace, unsigned blocks,
                          void *stream);

void Lu950BatchGetriLaunch(int n, const float *const *a, int lda, const int *pivots, float *const *c, int ldc,
                           int *info, int batch, unsigned blocks, void *stream);
void Lu950BatchGetriMediumTiledLaunch(int n, uint8_t *a, int lda, uint8_t *pivots, uint8_t *c, int ldc, uint8_t *info,
                                      int batch, unsigned blocks, void *stream);
size_t Lu950LargeBatchMatrixBytes(int n);
size_t Lu950LargeBatchWorkspaceBytes(int n, int batch);
int Lu950ParallelBatchGetriLaunch(int n, const float *const *a, int lda, const int *pivots, float *const *c, int ldc,
                                  int *info, int firstMatrix, int batch, void *workspace, unsigned blocks,
                                  void *stream);

void Lu950MatinvLaunch(int n, const float *const *a, int lda, float *const *c, int ldc, int *info, int batch, int error,
                       unsigned blocks, void *stream);
void Lu950MatinvN2Launch(const float *const *a, float *const *c, int *info, int batch, unsigned blocks, void *stream);
void Lu950MatinvBatchLaunch(int n, const float *const *a, int lda, float *const *c, int ldc, int *info, int batch,
                            unsigned blocks, void *stream);
void Lu950MatinvHighLaunch(int n, const float *const *a, int lda, float *const *c, int ldc, int *info, int batch,
                           int group, unsigned blocks, void *stream);
