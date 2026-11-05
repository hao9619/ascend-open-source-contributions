# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

add_executable(sgetrf_test
    sgetrf_test.cpp
)

target_include_directories(sgetrf_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

target_link_libraries(sgetrf_test PRIVATE
    ${OPS_SOLVER}
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendcl.so
)

add_executable(sgetrf_subnormal_pivot_test
    sgetrf_subnormal_pivot_test.cpp
)

target_include_directories(sgetrf_subnormal_pivot_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

target_link_libraries(sgetrf_subnormal_pivot_test PRIVATE
    ${OPS_SOLVER}
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendcl.so
)
