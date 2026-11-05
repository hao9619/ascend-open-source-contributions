# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)

project(ops_solver VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========= 基本配置 =========
if(NOT DEFINED SOC_VERSION OR SOC_VERSION STREQUAL "")
  set(SOC_VERSION ascend910b)
endif()
string(TOLOWER "${SOC_VERSION}" SOC_VERSION_LOWER)
# SOC_VERSION -> NPU_ARCH 映射（ascend910b->dav-2201, ascend950->dav-3510）
if(SOC_VERSION_LOWER MATCHES "^ascend910b")
  set(NPU_ARCH "dav-2201")
elseif(SOC_VERSION_LOWER MATCHES "^ascend910_93")
  set(NPU_ARCH "dav-2201")
elseif(SOC_VERSION_LOWER MATCHES "^ascend950")
  set(NPU_ARCH "dav-3510")
elseif(SOC_VERSION_LOWER MATCHES "^ascend310p")
  set(NPU_ARCH "dav-2002")
else()
  message(FATAL_ERROR "Unsupported SOC_VERSION: ${SOC_VERSION}. Supported: ascend910b*, ascend910_93*, ascend950*, ascend310p*")
endif()
message(STATUS "SOC_VERSION=${SOC_VERSION}, NPU_ARCH=${NPU_ARCH}")

# ========= SOC 架构目录配置 =========
# 根据 SOC 返回需要编译的架构目录列表
function(get_soc_arch_dirs soc_version arch_dirs)
  string(TOLOWER "${soc_version}" soc_lower)
  set(dirs "")

  if(soc_lower MATCHES "^ascend950")
    list(APPEND dirs "arch35")
  elseif(soc_lower MATCHES "^ascend310p")
    list(APPEND dirs "arch20")
  endif()
  # 其他 SOC 可以根据需要添加

  set(${arch_dirs} ${dirs} PARENT_SCOPE)
endfunction()

# 获取当前 SOC 对应的架构目录
get_soc_arch_dirs(${SOC_VERSION} SOC_ARCH_DIRS)
message(STATUS "SOC_ARCH_DIRS=${SOC_ARCH_DIRS}")

# 架构特定目录列表（全局定义）
set(ARCH_SPECIFIC_DIRS arch35 arch20 CACHE INTERNAL "Architecture specific directories")

set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/build_out" CACHE STRING "install path" FORCE)

# 外部传参
option(ENABLE_PACKAGE "Enable build package" OFF)

# ========= 构建动态库 =========
find_package(ASC REQUIRED)

# solver
set(ALL_SOLVER_SRC_FILES "")
set(OPS_SOLVER ops_solver)

add_subdirectory(src)

project(${OPS_SOLVER} LANGUAGES ASC CXX)

add_library(${OPS_SOLVER} SHARED ${ALL_SOLVER_SRC_FILES})

set_source_files_properties(
    ${ALL_SOLVER_SRC_FILES}
    PROPERTIES LANGUAGE ASC
)

target_include_directories(${OPS_SOLVER} BEFORE PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
)

if(SOC_VERSION_LOWER MATCHES "^ascend950")
    target_compile_definitions(${OPS_SOLVER} PRIVATE
        ACLSOLVER_HAS_LU950=1 LU950_N2_DENSE=1 LU950_VECTOR_DIAGONAL=1
        LU950_FUSED_BATCH=1 LU950_RAW_N2=1 LU950_RAW_BATCH=1)
    set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/src/lu_real/arch35/batch_getri_kernel.cpp"
        APPEND PROPERTY COMPILE_OPTIONS
        "--cce-long-scbz=true" "-mllvm" "-cce-aicore-vec-misched=0" "--cce-simd-vf-fusion=false")
    set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/src/lu_real/arch35/batch_getri_medium_kernel.cpp"
        APPEND PROPERTY COMPILE_OPTIONS
        "--cce-long-scbz=true" "--cce-simd-vf-fusion=false" "--cce-res-usage")
    # Scope high-order scheduling separately from the small-order compiler compatibility flags.
    set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/src/lu_real/arch35/batch_getri_high_kernel.cpp"
        APPEND PROPERTY COMPILE_OPTIONS
        "--cce-long-scbz=true" "--cce-simd-vf-fusion=false" "--cce-res-usage")
    target_compile_options(${OPS_SOLVER} PRIVATE -fno-fast-math -ffp-contract=off)
endif()

target_compile_options(${OPS_SOLVER} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
    $<$<COMPILE_LANGUAGE:ASC>:-iquote${CMAKE_CURRENT_LIST_DIR}/include>
)

target_link_libraries(${OPS_SOLVER} PRIVATE
    ascendcl
    register
    tiling_api
    platform
)

# ========= 安装规则（CPack 打包时放入 lib64 和 include） =========
install(TARGETS ${OPS_SOLVER}
    LIBRARY DESTINATION lib64
)
install(FILES
    include/cann_ops_solver.h
    include/cann_ops_solver_common.h
    DESTINATION include
)

# ========= 构建测试程序 =========
option(BUILD_TEST "Build test programs" OFF)
if(BUILD_TEST)
    add_subdirectory(test)
endif()

if(ENABLE_PACKAGE)
    include(cmake/package.cmake)
    add_dependencies(${OPS_SOLVER} gen_ops_solver_version_info)
    pack(${SOC_VERSION})
endif()
