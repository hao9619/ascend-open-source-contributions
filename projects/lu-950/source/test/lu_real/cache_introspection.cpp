/**
 * Copyright (c) 2026 xlvrNlfxKe All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Test-only inspection of the opaque handle; not part of the installed API.
#include <algorithm>
#include <cstdint>
#include <mutex>

#include "../../src/utils/aclsolver_handle_internal.h"

extern "C" int lu950_test_workspace_state(void *handle, uint64_t *records, int capacity)
{
    if (!handle || !records || capacity < 0) return -1;
    auto &internal = *static_cast<_aclsolver_handle *>(handle);
    std::lock_guard<std::mutex> guard(internal.workspaceMutex);
    const int count = std::min(capacity, static_cast<int>(internal.workspaces.size()));
    for (int index = 0; index < count; ++index)
    {
        const auto &workspace = internal.workspaces[index];
        aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        const auto result = aclrtQueryEventStatus(workspace.completion, &status);
        records[index * 5] = reinterpret_cast<uintptr_t>(workspace.address);
        records[index * 5 + 1] = workspace.bytes;
        records[index * 5 + 2] = reinterpret_cast<uintptr_t>(workspace.stream);
        records[index * 5 + 3] = status;
        records[index * 5 + 4] = result;
    }
    return count;
}
