/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclsolver_handle_internal.h
 * \brief aclsolver handle internal structure (not exposed externally).
 */

#pragma once

#include <acl/acl.h>

#include <cstddef>
#include <mutex>
#include <vector>

#include "cann_ops_solver_common.h"

/**
 * @brief aclsolver handle internal structure
 *
 * Contains only the minimal fields required for kernel execution:
 *   - stream: current stream for asynchronous kernel execution
 *
 * This struct definition is only visible in implementation files and is completely
 * opaque to external users. The external interface uses void* / void** as handle types.
 */
struct _aclsolver_workspace
{
    void *address = nullptr;
    size_t bytes = 0;
    aclrtStream stream = nullptr;
    aclrtEvent completion = nullptr;
};

struct _aclsolver_handle
{
    aclrtStream stream = nullptr;
    aclrtContext creationContext = nullptr;
    std::mutex workspaceMutex;
    std::vector<_aclsolver_workspace> workspaces;
};
