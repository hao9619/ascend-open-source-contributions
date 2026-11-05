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

#ifndef DRIVER_PROF_API_H
#define DRIVER_PROF_API_H

#include <cstdint>
#include <string>

#include "vallina_symbol.h"

namespace MemScope
{

enum class DeviceSocType
{
    SOC_A2_A3,  // default / unknown
    SOC_A5,     // 950-series
    SOC_A6,     // 960-series
};
DeviceSocType GetDeviceSocType();

constexpr uint64_t MAX_BUFFER_SIZE = 1024 * 1024 * 2;
constexpr uint32_t SECTONSEC = 1000000000UL;
constexpr uint32_t MSTONS = 1000000;

typedef enum ProfChannelType
{
    PROF_CHANNEL_TYPE_TS,
    PROF_CHANNEL_TYPE_PERIPHERAL,
    PROF_CHANNEL_TYPE_MAX,
} PROF_CHANNEL_TYPE;

typedef struct TagStarsSocLogConfig
{
    uint32_t acsq_task;          // 1-enable,2-disable
    uint32_t acc_pmu;            // 1-enable,2-disable
    uint32_t cdqm_reg;           // 1-enable,2-disable
    uint32_t dvpp_vpc_block;     // 1-enable,2-disable
    uint32_t dvpp_jpegd_block;   // 1-enable,2-disable
    uint32_t dvpp_jpede_block;   // 1-enable,2-disable
    uint32_t ffts_thread_task;   // 1-enable,2-disable
    uint32_t ffts_block;         // 1-enable,2-disable
    uint32_t sdma_dmu;           // 1-enable,2-disable
    uint32_t tag;                // A5/A6 only bit==0-enable immediately, bit0==1-enable delay
    uint32_t block_shrink_flag;  // A5/A6 only 1-enable,2-disable
} StarsSocLogConfigT;

typedef enum TAG_TS_PROFILE_COMMAND_TYPE
{
    TS_PROFILE_COMMAND_TYPE_ACK = 0,
    TS_PROFILE_COMMAND_TYPE_PROFILING_ENABLE = 1,
    TS_PROFILE_COMMAND_TYPE_PROFILING_DISABLE = 2,
    TS_PROFILE_COMMAND_TYPE_BUFFERFULL = 3,
    TS_PROFILE_COMMAND_TASK_BASE_ENABLE = 4,     // task base profiling enable
    TS_PROFILE_COMMAND_TASK_BASE_DISENABLE = 5,  // task base profiling disenable
    TS_PROFILE_COMMAND_TS_FW_ENABLE = 6,         // TS fw data enable
    TS_PROFILE_COMMAND_TS_FW_DISENABLE = 7,      // TS fw data disenable
} TS_PROFILE_COMMAND_TYPE_T;

typedef struct ProfStartPara
{
    PROF_CHANNEL_TYPE channelType;
    unsigned int samplePeriod;
    unsigned int realTime;
    void *userData;
    unsigned int userDataSize;
} ProfStartParaT;

enum AI_DRV_CHANNEL
{
    PROF_CHANNEL_UNKNOWN = 0,
    PROF_CHANNEL_TS_FW = 44,
    PROF_CHANNEL_STARS_SOC_LOG = 50,
    PROF_CHANNEL_MAX = 160,
};

struct StarsSocLog
{
    uint32_t funcType : 6;
    uint32_t cnt : 4;
    uint32_t sqeType : 6;
    uint32_t magic : 16;
    uint32_t streamId : 16;
    uint32_t taskId : 16;
    uint32_t sysCntL : 32;
    uint32_t sysCntH : 32;
    uint32_t res0 : 16;
    uint32_t accId : 6;
    uint32_t acsqId : 10;
    uint32_t res1[11];
};

struct StarsSocLogA5A6
{
    uint32_t funcType : 6;
    uint32_t cnt : 4;
    uint32_t sqeType : 6;
    uint32_t magic : 16;
    uint32_t taskId : 32;
    uint64_t timestamp : 64;
    uint32_t res0 : 16;
    uint32_t accId : 6;
    uint32_t acsqId : 10;
    uint32_t res1[3];
};

struct DevTimeInfo
{
    uint64_t freq;
    uint64_t startRealTime;
    uint64_t startSysCnt;
};
enum class PlatformType
{
    CHIP_910B = 5,
    CHIP_310B = 7,
    END_TYPE
};
typedef enum
{
    DRV_INFO_TYPE_ENV = 0,
    DRV_INFO_TYPE_VERSION,
    DRV_INFO_TYPE_MASTERID,
    DRV_INFO_TYPE_CORE_NUM,
    DRV_INFO_TYPE_FREQUE,
    DRV_INFO_TYPE_OS_SCHED,
    DRV_INFO_TYPE_IN_USED,
    DRV_INFO_TYPE_ERROR_MAP,
    DRV_INFO_TYPE_OCCUPY,
    DRV_INFO_TYPE_ID,
    DRV_INFO_TYPE_IP,
    DRV_INFO_TYPE_ENDIAN,
    DRV_INFO_TYPE_P2P_CAPABILITY,
    DRV_INFO_TYPE_SYS_COUNT,
    DRV_INFO_TYPE_MONOTONIC_RAW,
    DRV_INFO_TYPE_CORE_NUM_LEVEL,
    DRV_INFO_TYPE_FREQUE_LEVEL,
    DRV_INFO_TYPE_FFTS_TYPE,
    DRV_INFO_TYPE_PHY_CHIP_ID,
    DRV_INFO_TYPE_PHY_DIE_ID,
    DRV_INFO_TYPE_PF_CORE_NUM,
    DRV_INFO_TYPE_PF_OCCUPY,
    DRV_INFO_TYPE_WORK_MODE,
    DRV_INFO_TYPE_UTILIZATION,
    DRV_INFO_TYPE_HOST_OSC_FREQUE,
    DRV_INFO_TYPE_DEV_OSC_FREQUE,
    DRV_INFO_TYPE_SDID,
    DRV_INFO_TYPE_SERVER_ID,
    DRV_INFO_TYPE_SCALE_TYPE,
    DRV_INFO_TYPE_SUPER_POD_ID,
    DRV_INFO_TYPE_ADDR_MODE,
    DRV_INFO_TYPE_RUN_MACH,
    DRV_INFO_TYPE_CURRENT_FREQ,
    DRV_INFO_TYPE_CONFIG,
} DrvInfoType;

typedef enum
{
    DRV_MODULE_TYPE_SYSTEM = 0,
    DRV_MODULE_TYPE_AICPU,
    DRV_MODULE_TYPE_CCPU,               /**< ccpu_info*/
    DRV_MODULE_TYPE_DCPU,               /**< dcpu info*/
    DRV_MODULE_TYPE_AICORE,             /**< AI CORE info*/
    DRV_MODULE_TYPE_TSCPU,              /**< tscpu info*/
    DRV_MODULE_TYPE_PCIE,               /**< PCIE info*/
    DRV_MODULE_TYPE_VECTOR_CORE,        /**< VECTOR CORE info*/
    DRV_MODULE_TYPE_HOST_AICPU,         /* Host Aicpu info */
    DRV_MODULE_TYPE_QOS,                /**<qos info> */
    DRV_MODULE_TYPE_COMPUTING = 0x8000, /* computing power info */
} DrvModuleType;

enum StarsFuncType
{
    STARS_FUNC_TYPE_BEGIN = 0,
    STARS_FUNC_TYPE_END = 1,
};

bool StartDriverKernelInfoTrace(int32_t devId);
void EndDriverKernelInfoTrace();
uint64_t GetRealTimeFromSysCnt(uint32_t deviceId, uint64_t sysCnt);

}  // namespace MemScope
#endif
