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

#include "driver_prof_api.h"

#include <atomic>
#include <mutex>

#include "ascend_hal.h"
#include "event_report.h"
#include "kernel_event_trace.h"
#include "runtime_prof_api.h"
#include "securec.h"

namespace MemScope
{

static DeviceSocType g_deviceSocType = DeviceSocType::SOC_A2_A3;
static std::once_flag g_socDetectFlag;
static std::atomic<int32_t> g_profileDevice{GD_INVALID_NUM};

static void DetectSocType()
{
    using AclrtGetSocNameFunc = const char* (*)();
    auto func = reinterpret_cast<AclrtGetSocNameFunc>(GetSymbol("aclrtGetSocName"));
    if (func == nullptr)
    {
        return;  // keep default A2/A3
    }
    const char* name = func();
    if (name == nullptr)
    {
        return;
    }
    std::string socName(name);
    if (socName.find("950") != std::string::npos)
    {
        g_deviceSocType = DeviceSocType::SOC_A5;
    }
    else if (socName.find("960") != std::string::npos)
    {
        g_deviceSocType = DeviceSocType::SOC_A6;
    }
}

DeviceSocType GetDeviceSocType()
{
    std::call_once(g_socDetectFlag, DetectSocType);
    return g_deviceSocType;
}

static tagDrvError HalGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    using HalGetDeviceInfoFunc = tagDrvError (*)(uint32_t, int32_t, int32_t, int64_t*);
    static auto vallina = reinterpret_cast<HalGetDeviceInfoFunc>(GetSymbol("halGetDeviceInfo"));
    if (vallina == nullptr)
    {
        LOG_ERROR("halGetDeviceInfo api get failed");
        return DRV_ERROR_NOT_SUPPORT;
    }

    return vallina(deviceId, moduleType, infoType, value);
}

static int64_t GetDrvVersion(uint32_t deviceId)
{
    static int64_t errorVersion = -1;
    int64_t version = 0;
    tagDrvError ret = HalGetDeviceInfo(deviceId, DRV_MODULE_TYPE_SYSTEM, DRV_INFO_TYPE_VERSION, &version);
    return (ret == DRV_ERROR_NONE) ? version : errorVersion;
}

static PlatformType GetChipTypeImpl(uint32_t deviceId)
{
    int64_t versionInfo = GetDrvVersion(deviceId);
    if (versionInfo < 0)
    {
        LOG_ERROR("Call GetDrvVersion failed");
        return PlatformType::END_TYPE;
    }
    uint32_t chipId = ((static_cast<uint64_t>(versionInfo) >> 8) & 0xff);
    if (chipId >= static_cast<uint32_t>(PlatformType::END_TYPE))
    {
        LOG_ERROR("Get Chip Type failed");
        return PlatformType::END_TYPE;
    }
    return static_cast<PlatformType>(chipId);
}

static uint64_t GetDevFreq(uint32_t device)
{
    static uint64_t freqDefault = 50;
    static const std::unordered_map<PlatformType, uint64_t> FREQ_MAP = {
        {PlatformType::CHIP_910B, 50},
        {PlatformType::CHIP_310B, 50},
    };
    int64_t freq = 0;
    tagDrvError ret = HalGetDeviceInfo(device, DRV_MODULE_TYPE_SYSTEM, DRV_INFO_TYPE_DEV_OSC_FREQUE, &freq);
    if (ret != DRV_ERROR_NONE)
    {
        auto platform = GetChipTypeImpl(device);
        auto iter = FREQ_MAP.find(platform);
        uint64_t defaultFreq = (iter == FREQ_MAP.end()) ? freqDefault : iter->second;
        return defaultFreq;
    }
    return freq;
}

static uint64_t GetClockRealTimeNs()
{
    struct timespec ts;
    if (memset_s(&ts, sizeof(timespec), 0, sizeof(timespec)) != EOK)
    {
        return 0;
    }
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * SECTONSEC + static_cast<uint64_t>(ts.tv_nsec);
}

static uint64_t GetDevStartSysCnt(uint32_t device)
{
    constexpr uint64_t errorSystemCnt = 0;
    int64_t syscnt = 0;
    tagDrvError ret = HalGetDeviceInfo(device, DRV_MODULE_TYPE_SYSTEM, DRV_INFO_TYPE_SYS_COUNT, &syscnt);
    return (ret == DRV_ERROR_NONE) ? static_cast<uint64_t>(syscnt) : errorSystemCnt;
}

DevTimeInfo g_devTimeInfo = {};

static void InitDevTimeInfo(uint32_t deviceId)
{
    static constexpr uint32_t aveNum = 2;

    g_devTimeInfo.freq = GetDevFreq(deviceId);
    auto t1 = GetClockRealTimeNs();
    g_devTimeInfo.startSysCnt = GetDevStartSysCnt(deviceId);
    auto t2 = GetClockRealTimeNs();
    g_devTimeInfo.startRealTime = (t2 + t1) / aveNum;
    return;
}

uint64_t GetRealTimeFromSysCnt(uint32_t deviceId, uint64_t sysCnt)
{
    if (g_devTimeInfo.freq == 0)
    {
        LOG_ERROR("g_devTimeInfo.freq is 0, please check!");
        return 0;
    }
    uint64_t realTime =
        MSTONS * (sysCnt - g_devTimeInfo.startSysCnt) / g_devTimeInfo.freq + g_devTimeInfo.startRealTime;
    return realTime;
}

bool StartDriverKernelInfoTrace(int32_t devId)
{
    InitDevTimeInfo(devId);
    SetProfCommand(devId);
    StarsSocLogConfigT configP;
    if (memset_s(&configP, sizeof(StarsSocLogConfigT), 0, sizeof(StarsSocLogConfigT)) != EOK)
    {
        LOG_ERROR("memset StarsSocLogConfigT failed");
        return false;
    }
    configP.acsq_task = TS_PROFILE_COMMAND_TYPE_PROFILING_ENABLE;
    configP.ffts_thread_task = TS_PROFILE_COMMAND_TYPE_PROFILING_ENABLE;
    ProfStartParaT profStartPara;
    profStartPara.channelType = PROF_CHANNEL_TYPE_TS;
    static const uint32_t SAMPLE_PERIOD = 20;
    profStartPara.samplePeriod = SAMPLE_PERIOD;
    profStartPara.realTime = 1;
    profStartPara.userData = &configP;
    DeviceSocType socType = GetDeviceSocType();
    if (socType == DeviceSocType::SOC_A5 || socType == DeviceSocType::SOC_A6)
    {
        profStartPara.userDataSize = static_cast<unsigned int>(sizeof(StarsSocLogConfigT));
    }
    else
    {
        // A2/A3: exclude tag and block_shrink_flag fields
        profStartPara.userDataSize = static_cast<unsigned int>(sizeof(StarsSocLogConfigT) - 2 * sizeof(uint32_t));
    }

    using DriverProfStartFunc = int (*)(unsigned int, unsigned int, struct ProfStartPara*);
    static auto vallina = reinterpret_cast<DriverProfStartFunc>(GetSymbol("prof_drv_start"));
    if (vallina == nullptr)
    {
        LOG_ERROR("DriverProfStartFunc is nullptr");
        return false;
    }
    int ret = vallina(static_cast<uint32_t>(devId), PROF_CHANNEL_STARS_SOC_LOG, &profStartPara);
    if (ret != 0)
    {
        LOG_ERROR("driver prof start failed.");
        return false;
    }
    g_profileDevice.store(devId);
    RuntimeKernelLinker::GetInstance();
    atexit(EndDriverKernelInfoTrace);
    return true;
}

void EndDriverKernelInfoTrace()
{
    const int32_t devId = g_profileDevice.exchange(GD_INVALID_NUM);
    if (devId == GD_INVALID_NUM)
    {
        return;
    }
    using DriverProfEndFunc = int (*)(unsigned int, unsigned int);
    static auto vallina = reinterpret_cast<DriverProfEndFunc>(GetSymbol("prof_stop"));
    if (vallina == nullptr)
    {
        LOG_ERROR("EndDriverKernelInfoTrace is nullptr");
        return;
    }

    int ret = vallina(static_cast<uint32_t>(devId), PROF_CHANNEL_STARS_SOC_LOG);
    if (ret != 0)
    {
        LOG_ERROR("driver prof end failed.");
    }
    return;
}
}  // namespace MemScope
