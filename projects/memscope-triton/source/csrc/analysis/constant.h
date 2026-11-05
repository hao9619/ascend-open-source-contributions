/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#ifndef CONSTANT_H
#define CONSTANT_H

#include <string>
#include <unordered_map>

#include "event.h"

namespace MemScope
{

constexpr uint64_t BYTE_TO_MB = 1024 * 1024;
constexpr double PERCENT_SCALE_FACTOR = 100.0;

const std::unordered_map<PoolType, std::string> PoolTypeMap = {
    {PoolType::PTA_CACHING, "PTA"}, {PoolType::PTA_WORKSPACE, "PTA_WORKSPACE"},
    {PoolType::ATB, "ATB"},         {PoolType::MINDSPORE, "MINDSPORE"},
    {PoolType::HAL, "HAL"},         {PoolType::HOST, "HOST"},
};

// NPU池展示名（泄漏/Gap统计日志用），原定义于stepinner_analyzer.h，统一收敛至此处
const std::unordered_map<PoolType, std::string> PoolTypeToString = {
    {PoolType::ATB, "ATB memory pool"},
    {PoolType::PTA_CACHING, "Pytorch Caching memory pool"},
    {PoolType::MINDSPORE, "Mindspore memory pool"}};

// Module id
const std::unordered_map<int, std::string> MODULE_HASH_TABLE = {
    {0, "SLOG"},           /**< Slog */
    {1, "IDEDD"},          /**< IDE daemon device */
    {2, "IDEDH"},          /**< IDE daemon host */
    {3, "HCCL"},           /**< HCCL */
    {4, "FMK"},            /**< Adapter */
    {5, "HIAIENGINE"},     /**< Matrix */
    {6, "DVPP"},           /**< DVPP */
    {7, "RUNTIME"},        /**< Runtime */
    {8, "CCE"},            /**< CCE */
    {9, "HDC"},            /**< HDC */
    {10, "DRV"},           /**< Driver */
    {11, "MDCFUSION"},     /**< Mdc fusion */
    {12, "MDCLOCATION"},   /**< Mdc location */
    {13, "MDCPERCEPTION"}, /**< Mdc perception */
    {14, "MDCFSM"},
    {15, "MDCCOMMON"},
    {16, "MDCMONITOR"},
    {17, "MDCBSWP"},    /**< MDC base software platform */
    {18, "MDCDEFAULT"}, /**< MDC undefine */
    {19, "MDCSC"},      /**< MDC spatial cognition */
    {20, "MDCPNC"},
    {21, "MLL"},      /**< abandon */
    {22, "DEVMM"},    /**< Dlog memory managent */
    {23, "KERNEL"},   /**< Kernel */
    {24, "LIBMEDIA"}, /**< Libmedia */
    {25, "CCECPU"},   /**< aicpu shedule */
    {26, "ASCENDDK"}, /**< AscendDK */
    {27, "ROS"},      /**< ROS */
    {28, "HCCP"},
    {29, "ROCE"},
    {30, "TEFUSION"},
    {31, "PROFILING"}, /**< Profiling */
    {32, "DP"},        /**< Data Preprocess */
    {33, "APP"},       /**< User Application */
    {34, "TS"},        /**< TS module */
    {35, "TSDUMP"},    /**< TSDUMP module */
    {36, "AICPU"},     /**< AICPU module */
    {37, "LP"},        /**< LP module */
    {38, "TDT"},       /**< tsdaemon or aicpu shedule */
    {39, "FE"},
    {40, "MD"},
    {41, "MB"},
    {42, "ME"},
    {43, "IMU"},
    {44, "IMP"},
    {45, "GE"}, /**< Fmk */
    {46, "MDCFUSA"},
    {47, "CAMERA"},
    {48, "ASCENDCL"},
    {49, "TEEOS"},
    {50, "ISP"},
    {51, "SIS"},
    {52, "HSM"},
    {53, "DSS"},
    {54, "PROCMGR"},  // Process Manager, Base Platform
    {55, "BBOX"},
    {56, "AIVECTOR"},
    {57, "TBE"},
    {58, "FV"},
    {59, "MDCMAP"},
    {60, "TUNE"},
    {61, "HSS"}, /**< helper */
    {62, "FFTS"},
    {63, "OP"},
    {64, "UDF"},
    {65, "HICAID"},
    {66, "TSYNC"},
    {67, "AUDIO"},
    {68, "TPRT"},
    {69, "ASCENDCKERNEL"},
    {70, "ASYS"},
    {71, "ATRACE"},
    {72, "RTC"},
    {73, "SYSMONITOR"},
    {74, "AML"},
    {75, "INVLID_MOUDLE_ID"}  // add new module before INVLID_MOUDLE_ID
};

const std::unordered_map<EventBaseType, std::string> EVENT_BASE_TYPE_MAP = {
    {EventBaseType::MALLOC, "MALLOC"},       {EventBaseType::ACCESS, "ACCESS"},
    {EventBaseType::FREE, "FREE"},           {EventBaseType::MSTX, "MSTX"},
    {EventBaseType::OP_LAUNCH, "OP_LAUNCH"}, {EventBaseType::KERNEL_LAUNCH, "KERNEL_LAUNCH"},
    {EventBaseType::SYSTEM, "SYSTEM"},       {EventBaseType::CLEAN_UP, "CLEAN_UP"},
    {EventBaseType::SNAPSHOT, "SNAPSHOT"},   {EventBaseType::OOM_DETAIL, "OOM_DETAIL"},
};

const std::unordered_map<EventSubType, std::string> EVENT_SUB_TYPE_MAP = {
    {EventSubType::PTA_CACHING, "PTA"},  // 兼容性考虑，对外展示保持不变
    {EventSubType::PTA_WORKSPACE, "PTA"},
    {EventSubType::ATB, "ATB"},
    {EventSubType::MINDSPORE, "MINDSPORE"},
    {EventSubType::HAL, "HAL"},
    {EventSubType::HOST, "HOST"},
    {EventSubType::HOST_PINNED, "HOST_PINNED"},
    {EventSubType::ATB_READ, "READ"},
    {EventSubType::ATB_WRITE, "WRITE"},
    {EventSubType::ATB_READ_OR_WRITE, "UNKNOWN"},
    {EventSubType::ATEN_READ, "READ"},
    {EventSubType::ATEN_WRITE, "WRITE"},
    {EventSubType::ATEN_READ_OR_WRITE, "UNKNOWN"},
    {EventSubType::ATB_START, "ATB_START"},
    {EventSubType::ATB_END, "ATB_END"},
    {EventSubType::ATEN_START, "ATEN_START"},
    {EventSubType::ATEN_END, "ATEN_END"},
    {EventSubType::TRITON_START, "TRITON_START"},
    {EventSubType::TRITON_END, "TRITON_END"},
    {EventSubType::TRITON_READ, "READ"},
    {EventSubType::TRITON_WRITE, "WRITE"},
    {EventSubType::TRITON_UNKNOWN, "UNKNOWN"},
    {EventSubType::KERNEL_LAUNCH, "KERNEL_LAUNCH"},
    {EventSubType::KERNEL_EXECUTE_START, "KERNEL_EXECUTE_START"},
    {EventSubType::KERNEL_EXECUTE_END, "KERNEL_EXECUTE_END"},
    {EventSubType::ATB_KERNEL_START, "KERNEL_START"},
    {EventSubType::ATB_KERNEL_END, "KERNEL_END"},
    {EventSubType::ACL_INIT, "ACL_INIT"},
    {EventSubType::ACL_FINI, "ACL_FINI"},
    {EventSubType::TRACE_START, "START_TRACE"},
    {EventSubType::TRACE_STOP, "STOP_TRACE"},
    {EventSubType::MSTX_MARK, "Mark"},
    {EventSubType::MSTX_RANGE_START, "Range_start"},
    {EventSubType::MSTX_RANGE_END, "Range_end"},
    {EventSubType::RESIDUAL_BLOCK, "RESIDUAL_BLOCK"},
    {EventSubType::PROC_EXIT, "PROC_EXIT"},
    {EventSubType::STEP, "STEP"},
    {EventSubType::SNAPSHOT, "SNAPSHOT"},
    {EventSubType::OOM_TRIGGER, "OOM_TRIGGER"},
    {EventSubType::OOM_RECENT_ALLOC, "OOM_RECENT_ALLOC"},
    {EventSubType::OOM_TOP_ALLOC, "OOM_TOP_ALLOC"},
    {EventSubType::HOST_LEAK_STAGE_START, "HOST_LEAK_STAGE_START"},
    {EventSubType::HOST_LEAK_STAGE_END, "HOST_LEAK_STAGE_END"},
};

const std::vector<std::pair<std::string, std::string>> DUMP_RECORD_TABLE_SQL = {{"ID", "INTEGER"},
                                                                                {"Event", "TEXT"},
                                                                                {"Event Type", "TEXT"},
                                                                                {"Name", "TEXT"},
                                                                                {"Timestamp(ns)", "INTEGER"},
                                                                                {"Process Id", "INTEGER"},
                                                                                {"Thread Id", "INTEGER"},
                                                                                {"Device Id", "TEXT"},
                                                                                {"Ptr", "TEXT"},
                                                                                {"Attr", "TEXT"}};

const std::vector<std::pair<std::string, std::string>> PYTHON_TRACE_TABLE_SQL = {{"FuncInfo", "TEXT"},
                                                                                 {"StartTime(ns)", "TEXT"},
                                                                                 {"EndTime(ns)", "TEXT"},
                                                                                 {"Thread Id", "INTEGER"},
                                                                                 {"Process Id", "INTEGER"}};

const std::string DUMP_RECORD_TABLE = "memscope_dump";
const std::string PYTHON_TRACE_TABLE = "python_trace_";

}  // namespace MemScope

#endif
