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

namespace lu950
{
// Argument numbering excludes the handle, following the supplied task sheet.
inline int ValidateMatinv(int n, const void *a, int lda, const void *c, int ldc, const void *info, int batch)
{
    if (n < 0 || n > 32)
    {
        return 1;
    }
    if (n > 0 && batch > 0 && a == nullptr)
    {
        return 2;
    }
    if (lda < (n > 0 ? n : 1))
    {
        return 3;
    }
    if (n > 0 && batch > 0 && c == nullptr)
    {
        return 4;
    }
    if (ldc < (n > 0 ? n : 1))
    {
        return 5;
    }
    if (batch > 0 && info == nullptr)
    {
        return 6;
    }
    if (batch < 0)
    {
        return 7;
    }
    return 0;
}
}  // namespace lu950
