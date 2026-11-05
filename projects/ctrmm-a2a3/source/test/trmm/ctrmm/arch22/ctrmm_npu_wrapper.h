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
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include "device.h"
#include "ctrmm_param.h"

inline aclblasStatus_t CtrmmNpu(
    aclblasHandle_t handle, const CtrmmParam& p, const std::vector<aclblasComplex>& a,
    const std::vector<aclblasComplex>& b, std::vector<aclblasComplex>& c)
{
    DeviceBuffer da(std::max<size_t>(8, a.size() * 8));
    DeviceBuffer db(std::max<size_t>(8, b.size() * 8));
    DeviceBuffer dc(std::max<size_t>(8, c.size() * 8));
    if (!a.empty())
        da.copyFromHost(a.data(), a.size() * 8);
    if (!b.empty())
        db.copyFromHost(b.data(), b.size() * 8);
    if (!c.empty())
        dc.copyFromHost(c.data(), c.size() * 8);
    auto* output = static_cast<aclblasComplex*>(p.inPlace ? db.ptr() : dc.ptr());
    const auto* av = p.nullA ? nullptr : static_cast<aclblasComplex*>(da.ptr());
    const auto* bv = p.nullB ? nullptr : static_cast<aclblasComplex*>(db.ptr());
    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("aclblasGetStream failed");
    }
    const bool profile = std::getenv("CTRMM_PROFILE") != nullptr;
    const int calls = profile ? 70 : 1;
    aclblasStatus_t status = ACLBLAS_STATUS_SUCCESS;
    for (int iteration = 0; iteration < calls; ++iteration) {
        status = aclblasCtrmm(
            p.description == "null_handle" ? nullptr : handle, p.side, p.uplo, p.trans, p.diag, p.m, p.n,
            p.nullAlpha ? nullptr : &p.alpha, av, p.lda, bv, p.ldb, p.nullC ? nullptr : output, p.ldc);
        if (status != ACLBLAS_STATUS_SUCCESS)
            return status;
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            throw std::runtime_error("CTRMM stream synchronization failed");
        }
    }
    if (!p.nullC && !c.empty()) {
        if (p.inPlace)
            db.copyToHost(c.data(), c.size() * 8);
        else
            dc.copyToHost(c.data(), c.size() * 8);
    }
    return status;
}
