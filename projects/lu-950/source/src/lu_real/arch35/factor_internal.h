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
#include "lu950_internal.h"

// Experimental entry used to validate blocked LU before public API integration.
aclsolverStatus_t Lu950Getrf(aclsolverHandle_t handle, int m, int n, float *a, int lda, int *pivots, int *info);

// Cube candidate: workspace is allocated by the caller, never inside the API.
size_t Lu950GetrfCubeWorkspaceBytes(int m, int n);
aclsolverStatus_t Lu950GetrfCube(aclsolverHandle_t handle, int m, int n, float *a, int lda, int *pivots, int *info,
                                 void *workspace, size_t workspaceBytes);

size_t Lu950GetrfBlockedWorkspaceBytes(int m, int n);
aclsolverStatus_t Lu950GetrfBlocked(aclsolverHandle_t handle, int m, int n, float *a, int lda, int *pivots, int *info,
                                    void *workspace, size_t workspaceBytes);
