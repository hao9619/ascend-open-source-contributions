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
 * \file sgetrf_test.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>

#include "../utils/real_solver_test.h"
#include "../utils/test_utils.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

int main(int argc, char **argv)
{
    int deviceId, M, N;
    deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    M = (argc > 2) ? std::atoi(argv[2]) : 32;
    N = (argc > 3) ? std::atoi(argv[3]) : 32;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclsolverHandle_t handle = nullptr;
    CHECK_ACL(aclsolverCreate(&handle));
    CHECK_ACL(aclsolverSetStream(handle, stream));

    size_t aMatrixFileSize = M * N * sizeof(float);

    float *A = nullptr;
    int32_t *ipiv = nullptr;
    auto cleanup = [&]() -> aclError
    {
        if (ipiv != nullptr)
        {
            delete[] ipiv;
            ipiv = nullptr;
        }
        if (A != nullptr)
        {
            CHECK_ACL(aclrtFreeHost(A));
        }
        CHECK_ACL(aclrtDestroyStream(stream));
        if (handle != nullptr)
        {
            CHECK_ACL(aclsolverDestroy(handle));
        }
        CHECK_ACL(aclrtResetDevice(deviceId));
        CHECK_ACL(aclFinalize());
        return ACL_SUCCESS;
    };

    CHECK_ACL(aclrtMallocHost((void **)(&A), aMatrixFileSize));
    CHECK_RET(ReadFile("./test/sgetrf/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize),
              LOG_PRINT("ReadFile A_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);

    std::cout << "[Input] A:" << std::endl;
    PrintPartOfMatrix<float>((uint8_t *)A, M, N, 8, 8);

    ipiv = new int32_t[std::min(M, N)];
    int32_t infoVal = 0;
    int32_t *info = &infoVal;

    auto ret = solver_test::RunRealGetrf(handle, M, N, A, ipiv, info);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverSgetrf failed. ERROR: %d\n", ret); cleanup(); return ret);

    std::cout << "[Output] A (LU):" << std::endl;
    PrintPartOfMatrix<float>((uint8_t *)A, M, N, 8, 8);

    WriteFile("./test/sgetrf/data/output/A_gm.bin", A, aMatrixFileSize);

    // Write pivot information
    std::vector<int32_t> ipivData(ipiv, ipiv + std::min(M, N));
    WriteFile("./test/sgetrf/data/output/W_gm.bin", ipivData.data(), ipivData.size() * sizeof(int32_t));

    auto cleanupRet = cleanup();
    CHECK_RET(cleanupRet == ACL_SUCCESS, return cleanupRet);

    return 0;
}
