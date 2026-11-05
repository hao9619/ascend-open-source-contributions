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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../utils/aclsolver_handle_internal.h"
#include "cann_ops_solver.h"

#ifdef ACLSOLVER_HAS_LU950
#include "arch35/factor_internal.h"
#include "arch35/getri_internal.h"
#include "arch35/lu950_internal.h"

void Lu950InfoLaunch(int *info, int count, int value, aclrtStream stream);
#endif

namespace lu950
{
inline bool BlockedFactor(int m, int n)
{
    return std::min(m, n) > 256 || (m > 64 && static_cast<int64_t>(n) >= 4LL * m);
}

inline aclsolverStatus_t Stream(aclsolverHandle_t handle, aclrtStream &stream)
{
    if (!handle) return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    const auto &internal = *reinterpret_cast<_aclsolver_handle *>(handle);
    aclrtContext current = nullptr;
    if (aclrtGetCurrentContext(&current) != ACL_SUCCESS || current != internal.creationContext)
        return ACLSOLVER_STATUS_NOT_INITIALIZED;
    return aclsolverGetStream(handle, &stream);
}

inline bool DeviceSpan(const void *pointer, size_t bytes)
{
    if (!pointer || reinterpret_cast<uintptr_t>(pointer) % alignof(float)) return false;
    aclrtPtrAttributes attributes{};
    int32_t device = -1;
    if (aclrtPointerGetAttributes(pointer, &attributes) != ACL_SUCCESS ||
        attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE || aclrtGetDevice(&device) != ACL_SUCCESS ||
        attributes.location.id != static_cast<uint32_t>(device))
        return false;
    void *base = nullptr;
    size_t capacity = 0;
    if (aclrtMemGetAddressRange(const_cast<void *>(pointer), &base, &capacity) != ACL_SUCCESS) return false;
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer), start = reinterpret_cast<uintptr_t>(base);
    return address >= start && address - start <= capacity && bytes <= capacity - (address - start);
}

inline size_t MatrixBytes(int rows, int columns, int ld)
{
    return columns > 0 && rows > 0 ? (static_cast<size_t>(columns - 1) * ld + rows) * sizeof(float) : 0;
}

inline bool Overlap(const void *a, size_t aBytes, const void *b, size_t bBytes)
{
    if (!a || !b || !aBytes || !bBytes) return false;
    const uintptr_t av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
    return av <= bv ? bv - av < aBytes : av - bv < bBytes;
}

inline aclsolverStatus_t Info(int *info, int count, int value, aclrtStream stream)
{
    if (count <= 0) return value == 0 ? ACLSOLVER_STATUS_SUCCESS : ACLSOLVER_STATUS_INVALID_VALUE;
    if (!DeviceSpan(info, static_cast<size_t>(count) * sizeof(int))) return ACLSOLVER_STATUS_INVALID_VALUE;
#ifdef ACLSOLVER_HAS_LU950
    Lu950InfoLaunch(info, count, value, stream);
    return value == 0 ? ACLSOLVER_STATUS_SUCCESS : ACLSOLVER_STATUS_INVALID_VALUE;
#else
    return ACLSOLVER_STATUS_NOT_SUPPORTED;
#endif
}

inline aclsolverStatus_t QuerySize(size_t bytes, int *elements)
{
    if (!elements || bytes % sizeof(float) || bytes / sizeof(float) > static_cast<size_t>(INT32_MAX))
        return ACLSOLVER_STATUS_INVALID_VALUE;
    *elements = static_cast<int>(bytes / sizeof(float));
    return ACLSOLVER_STATUS_SUCCESS;
}
}  // namespace lu950
