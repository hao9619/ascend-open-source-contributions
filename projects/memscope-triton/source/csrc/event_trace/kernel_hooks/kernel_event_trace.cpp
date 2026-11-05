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

#include "kernel_event_trace.h"

#include "bit_field.h"
#include "driver_prof_api.h"
#include "event_report.h"
#include "memory_watch/memory_watch.h"
#include "securec.h"
#include "runtime_prof_api.h"
#include "stars_common.h"
#include "utils.h"

namespace MemScope
{

void KernelEventTrace::KernelLaunch(const AclnnKernelMapInfo& kernelLaunchInfo)
{
    if (!EventTraceManager::Instance().IsTracingEnabled())
    {
        return;
    }
    if (!EventReport::Instance(MemScopeCommType::SHARED_MEMORY).ReportKernelLaunch(kernelLaunchInfo))
    {
        LOG_ERROR("KernelLaunch launch report failed");
    }

    return;
}

void KernelEventTrace::KernelStartExcute(TaskKey& key, uint64_t time)
{
    auto kernelName = RuntimeKernelLinker::GetInstance().GetKernelName(key, RecordSubType::KERNEL_START);
    if (!kernelName.empty())
    {
        if (!EventTraceManager::Instance().IsTracingEnabled())
        {
            return;
        }
        if (!EventReport::Instance(MemScopeCommType::SHARED_MEMORY)
                 .ReportKernelExcute(key, kernelName, time, RecordSubType::KERNEL_START))
        {
            LOG_ERROR("Kernel excute start report failed");
        }
    }
    return;
}

void KernelEventTrace::KernelEndExcute(TaskKey& key, uint64_t time)
{
    auto kernelName = RuntimeKernelLinker::GetInstance().GetKernelName(key, RecordSubType::KERNEL_END);
    if (!kernelName.empty())
    {
        if (!EventTraceManager::Instance().IsTracingEnabled())
        {
            return;
        }
        if (!EventReport::Instance(MemScopeCommType::SHARED_MEMORY)
                 .ReportKernelExcute(key, kernelName, time, RecordSubType::KERNEL_END))
        {
            LOG_ERROR("Kernel excute end report failed");
        }
    }
    return;
}

static void ReportStarsSocLog(uint32_t deviceId, const StarsSocLog* socLog)
{
    if (!socLog)
    {
        return;
    }
    constexpr int32_t bitOffset = 32;
    uint16_t streamId =
        StarsCommon::GetStreamId(static_cast<uint16_t>(socLog->streamId), static_cast<uint16_t>(socLog->taskId),
                                 static_cast<uint16_t>(socLog->sqeType));
    uint16_t taskId =
        StarsCommon::GetTaskId(static_cast<uint16_t>(socLog->streamId), static_cast<uint16_t>(socLog->taskId),
                               static_cast<uint16_t>(socLog->sqeType));
    auto taskKey = std::make_tuple(static_cast<uint16_t>(deviceId), streamId, taskId);
    if (socLog->funcType == STARS_FUNC_TYPE_BEGIN)
    {
        auto start = static_cast<uint64_t>(socLog->sysCntH) << bitOffset | socLog->sysCntL;
        start = GetRealTimeFromSysCnt(deviceId, start);
        KernelEventTrace::GetInstance().KernelStartExcute(taskKey, start);
    }
    else if (socLog->funcType == STARS_FUNC_TYPE_END)
    {
        auto end = static_cast<uint64_t>(socLog->sysCntH) << bitOffset | socLog->sysCntL;
        end = GetRealTimeFromSysCnt(deviceId, end);
        KernelEventTrace::GetInstance().KernelEndExcute(taskKey, end);
    }

    return;
}

static void ReportStarsSocLogA5A6(uint32_t deviceId, const StarsSocLogA5A6* socLog)
{
    if (!socLog)
    {
        return;
    }
    uint16_t streamId = 0;
    uint16_t taskId = static_cast<uint16_t>(socLog->taskId);
    auto taskKey = std::make_tuple(static_cast<uint16_t>(deviceId), streamId, taskId);
    if (socLog->funcType == STARS_FUNC_TYPE_BEGIN)
    {
        auto start = socLog->timestamp;
        start = GetRealTimeFromSysCnt(deviceId, start);
        KernelEventTrace::GetInstance().KernelStartExcute(taskKey, start);
    }
    else if (socLog->funcType == STARS_FUNC_TYPE_END)
    {
        auto end = socLog->timestamp;
        end = GetRealTimeFromSysCnt(deviceId, end);
        KernelEventTrace::GetInstance().KernelEndExcute(taskKey, end);
    }

    return;
}

static size_t TransStarsLog(char buffer[], size_t validSize, uint32_t deviceId)
{
    size_t pos = 0;
    DeviceSocType socType = GetDeviceSocType();
    bool isNewSoc = (socType == DeviceSocType::SOC_A5 || socType == DeviceSocType::SOC_A6);
    size_t size = isNewSoc ? sizeof(StarsSocLogA5A6) : sizeof(StarsSocLog);
    while (validSize - pos >= size)
    {
        if (isNewSoc)
        {
            const StarsSocLogA5A6* data = reinterpret_cast<StarsSocLogA5A6*>(buffer + pos);
            ReportStarsSocLogA5A6(deviceId, data);
        }
        else
        {
            const StarsSocLog* data = reinterpret_cast<StarsSocLog*>(buffer + pos);
            ReportStarsSocLog(deviceId, data);
        }
        pos += size;
    }
    return pos;
}

static size_t TransDataToActivityBuffer(char buffer[], size_t validSize, uint32_t deviceId, AI_DRV_CHANNEL channelId)
{
    switch (channelId)
    {
        case PROF_CHANNEL_STARS_SOC_LOG:
            return TransStarsLog(buffer, validSize, deviceId);
        default:
            return 0;
    }
}

void KernelEventTrace::CreateReadDataChannel(uint32_t devId)
{
    readTh_ = std::thread(
        [devId, this]() mutable
        {
            std::vector<char> buf(MAX_BUFFER_SIZE);
            size_t curPos = 0;
            int currLen = 0;
            using ReadFunc = int (*)(unsigned int, unsigned int, char*, unsigned int);
            while (started_)
            {
                static auto vallina = reinterpret_cast<ReadFunc>(GetSymbol("prof_channel_read"));
                if (vallina == nullptr)
                {
                    LOG_ERROR("ReadFunc is null");
                    return;
                }
                currLen = vallina(devId, PROF_CHANNEL_STARS_SOC_LOG, buf.data() + curPos, MAX_BUFFER_SIZE - curPos);
                if (currLen <= 0)
                {
                    continue;
                }
                auto uintCurrLen = static_cast<size_t>(currLen);
                if (uintCurrLen >= (MAX_BUFFER_SIZE - curPos))
                {
                    LOG_ERROR("Read invalid data len from driver");
                    continue;
                }
                size_t lastPos =
                    TransDataToActivityBuffer(buf.data(), curPos + uintCurrLen, devId, PROF_CHANNEL_STARS_SOC_LOG);
                if (lastPos < curPos + uintCurrLen)
                {
                    if (memcpy_s(buf.data(), MAX_BUFFER_SIZE, buf.data() + lastPos, curPos + uintCurrLen - lastPos) !=
                        EOK)
                    {
                        continue;
                    }
                }
                curPos = curPos + uintCurrLen - lastPos;
            }
        });
}

void KernelEventTrace::StartKernelEventTrace()
{
    GetDeviceSocType();  // trigger SoC type detection early
    int32_t devId = GD_INVALID_NUM;
    if (!GetDeviceInfo::Instance().GetDeviceId(devId) || devId == GD_INVALID_NUM)
    {
        LOG_ERROR("get device id failed");
    }

    RegisterRtProfileCallback();
    if (StartDriverKernelInfoTrace(devId))
    {
        CreateReadDataChannel(static_cast<uint32_t>(devId));
    }
}

void KernelEventTrace::EndKernelEventTrace() const { return EndDriverKernelInfoTrace(); }

KernelEventTrace::~KernelEventTrace()
{
    started_.store(false);
    if (readTh_.joinable())
    {
        readTh_.join();
    }
}

std::string RuntimeKernelLinker::GetLastKernelName(uint64_t tid)
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentNameMap_[tid];
}

void RuntimeKernelLinker::RuntimeTaskInfoLaunch(const TaskKey& key, uint64_t hashId)
{
    if (hashId == 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = kernelNameMp_.find(Utility::GetTid());
    if (iter == kernelNameMp_.end())
    {
        return;
    }
    auto& vec = iter->second;
    if (!vec.empty())
    {
        vec.back().taskKey = key;
        vec.back().kernelName = GetHashInfo(hashId);
        std::string name =
            std::to_string(std::get<0>(key)) + "_" + std::to_string(iter->first) + "/" + vec.back().kernelName;
        currentNameMap_[iter->first] = name;
        bool isKernelLevel = BitPresent(GetConfig().levelType, static_cast<size_t>(LevelType::LEVEL_KERNEL));
        if ((GetConfig().watchConfig.isWatched || TensorMonitor::GetInstance().IsInMonitoring()) && isKernelLevel)
        {
            MemoryWatch::GetInstance().KernelExcuteBegin(nullptr, name);
        }
        KernelEventTrace::GetInstance().KernelLaunch(vec.back());
    }

    return;
}

void RuntimeKernelLinker::KernelLaunch()
{
    uint64_t timestamp = Utility::GetTimeNanoseconds();
    std::lock_guard<std::mutex> lock(mutex_);

    AclnnKernelMapInfo value = {timestamp, std::make_tuple(-1, -1, -1), ""};
    auto iter = kernelNameMp_.find(Utility::GetTid());
    if (iter == kernelNameMp_.end())
    {
        std::vector<AclnnKernelMapInfo> vec{};
        vec.push_back(value);
        kernelNameMp_.insert({Utility::GetTid(), vec});
    }
    else
    {
        auto& vec = iter->second;
        vec.push_back(value);
    }

    return;
}

std::string RuntimeKernelLinker::GetKernelName(TaskKey& key, RecordSubType type)
{
    std::lock_guard<std::mutex> lock(mutex_);
    DeviceSocType socType = GetDeviceSocType();
    bool isNewSoc = (socType == DeviceSocType::SOC_A5 || socType == DeviceSocType::SOC_A6);

    if (isNewSoc)
    {
        // A5/A6: only compare deviceId and taskId (streamId is always 0 in key)
        for (auto& pair : kernelNameMp_)
        {
            auto& vec = pair.second;
            for (auto it = vec.begin(); it != vec.end(); ++it)
            {
                if (std::get<0>(it->taskKey) != std::get<0>(key) || std::get<2>(it->taskKey) != std::get<2>(key))
                {
                    continue;
                }
                std::string name = it->kernelName;
                // backfill streamId into key for subsequent data reporting
                std::get<1>(key) = std::get<1>(it->taskKey);
                if (type == RecordSubType::KERNEL_END)
                {
                    vec.erase(it);  // 使用完删除
                }
                return name;
            }
        }
    }
    else
    {
        for (auto& pair : kernelNameMp_)
        {
            auto& vec = pair.second;
            for (auto it = vec.begin(); it != vec.end(); ++it)
            {
                if (it->taskKey != key)
                {
                    continue;
                }
                std::string name = it->kernelName;
                if (type == RecordSubType::KERNEL_END)
                {
                    vec.erase(it);  // 使用完删除
                }
                return name;
            }
        }
    }
    return "";
}

void RuntimeKernelLinker::SetHashInfo(uint64_t hashId, const std::string& hashInfo)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = hashInfo_map_.find(hashId);
    if (iter == hashInfo_map_.end())
    {
        hashInfo_map_.insert({hashId, hashInfo});
    }
}

std::string RuntimeKernelLinker::GetHashInfo(uint64_t hashId)
{
    const auto iter = hashInfo_map_.find(hashId);
    if (iter != hashInfo_map_.end())
    {
        return iter->second;
    }
    return "";
}
}  // namespace MemScope
