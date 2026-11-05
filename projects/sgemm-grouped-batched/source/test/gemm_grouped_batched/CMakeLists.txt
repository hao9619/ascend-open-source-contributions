# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

ops_blas_add_gtest_tests(${OPS_BLAS} gemm_grouped_batched_test)

# The shared test framework already links REFBLAS_LIB. Preserve the existing arch35
# dependency, while arch22 uses the explicitly selected reference backend.
if("arch35" IN_LIST SOC_ARCH_DIRS)
    # Link OpenBLAS for golden reference (cblas_sgemm)
    find_library(OPENBLAS_LIB openblas PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
    if(OPENBLAS_LIB)
        target_link_libraries(gemm_grouped_batched_test PRIVATE ${OPENBLAS_LIB})
    else()
        message(WARNING "OpenBLAS not found; golden reference linking may fail at runtime")
    endif()
endif()
