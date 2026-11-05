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

#include "event_trace_manager.h"

#include "bit_field.h"
#include "client_parser.h"
#include "cpython.h"
#include "dump.h"
#include "event_report.h"
#include "json_manager.h"
#include "memory_state_manager.h"

namespace MemScope
{

namespace
{
// 事件上报抑制深度计数（thread_local）：仪器自身调用真实运行时接口（如dcmi_get_device_hbm_info）期间
// 置位，运行时内部的内存申请会被hook捕获并尝试上报，导致递归上报/幻影事件。
// 嵌套置位时递增计数，最外层离开时归零，各线程互不影响
thread_local int g_suppressEventReportDepth = 0;
}  // namespace

bool IsEventReportSuppressed() { return g_suppressEventReportDepth > 0; }

EventReportSuppressor::EventReportSuppressor() { ++g_suppressEventReportDepth; }
EventReportSuppressor::~EventReportSuppressor() { --g_suppressEventReportDepth; }

bool ConfigManager::Inited = false;

const std::unordered_map<std::string, std::function<void(const std::string &, Config &, bool &)>> parserConfigTable = {
    {"call_stack", ParseCallstack},
    {"level", ParseDataLevel},
    {"events", ParseEventTraceType},
    {"device", ParseDevice},
    {"format", ParseDataFormat},
    {"output_path", ParseOutputPath},
    // 兼容旧参数名（仅保留解析能力，不体现于用户手册/帮助信息）
    {"data_format", ParseDataFormat},
    {"output", ParseOutputPath},
    {"analysis", ParseAnalysis},
    {"watch", ParseWatchConfig},
    // host内存泄漏检测:config新增键,与CLI同名参数共用解析
    {"host_leak_mode", ParseHostLeakMode},
    {"block_size_threshold", ParseBlockSizeThreshold},
    // 显式采样率倒数(默认1=不采样),与CLI --sample-rate共用解析
    {"sample_rate", ParseSampleRate},
};

// 只允许设置一次的config参数；旧参数名与标准名归入同一组，视为同一个参数，不可重复设置
// （旧参数名data_format/output仅作兼容保留，不体现于用户手册/帮助信息）
const std::unordered_map<std::string, std::string> configPolicyTable = {
    {"format", "format"},      {"data_format", "format"}, {"output_path", "output_path"},
    {"output", "output_path"}, {"watch", "watch"},
};

ConfigManager::ConfigManager() { InitConfig(); }

void ConfigManager::InitConfig()
{
    Config config;
    // 命令行与python接口并存
    if (firstConfig && Utility::JsonConfig::GetInstance().ReadJsonConfig(config))
    {
        firstConfig = false;
        config_ = config;
    }
    else
    {
        // 单独python接口
        ClientParser parser;
        parser.InitialConfig(config);
        SetConfigImpl(config);
    }
    ConfigManager::Inited = true;
}

// 在python config接口时，将需要继承和不准修改的参数保留；不准修改的针对命令行传入的；
void ConfigManager::GetConfigAfterInit(Config &config)
{
    ClientParser parser;
    parser.InitialConfig(config);
    config.collectMode = static_cast<uint8_t>(CollectMode::DEFERRED);
    config.isEffective = config_.isEffective;
    config.dataFormat = config_.dataFormat;
    config.watchConfig = config_.watchConfig;

    if (strncpy_s(config.outputDir, sizeof(config.outputDir), config_.outputDir, sizeof(config.outputDir) - 1) != EOK)
    {
        std::cout << "[msmemscope] Error: strncpy dirpath FAILED" << std::endl;
        return;
    }
    config.outputDir[sizeof(config.outputDir) - 1] = '\0';
}

void ConfigManager::InitStartConfig()
{
    EventTraceManager::Instance().SetTraceStatus(EventTraceStatus::IN_TRACING);
    Utility::JsonConfig::GetInstance().SaveConfigToJson(config_);
}

const Config &ConfigManager::GetConfig()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

void ConfigManager::SetConfigImpl(const Config &config)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);

        config_ = config;
        SetEventDefaultConfig(config_);
        SetAnalysisDefaultConfig(config_);
    }
}

void ConfigManager::SetConfig(const Config &config)
{
    SetConfigImpl(config);
    SetEffectiveConfig(config_);
    Utility::SetLogLevel(static_cast<LogLv>(config_.logLevel));
    Utility::JsonConfig::GetInstance().SaveConfigToJson(config_);
    // HandleWithATenCollect 会初始化EventReport，需要在save之后
    // EventTraceManager的构造函数中会调用InitTraceStatus，会调用GetConfig，如果在InitConfig函数中调用SetConfig，可能
    // 会造成Instance()间接调用自身，导致单例的自引用，初始化死锁现象。
    EventTraceManager::Instance().HandleWithATenCollect();
    EventTraceManager::Instance().HandleWithDecompose();
    EventTraceManager::Instance().HandleWithCpuTensorCollect();
}

bool ConfigManager::SetConfig(const std::unordered_map<std::string, std::string> &pythonConfig)
{
    Config config;
    GetConfigAfterInit(config);

    // 记录本次调用中已设置过的"只允许设置一次"参数组（兼容旧名与标准名视为同一参数）
    std::vector<std::string> onceSetGroups;

    for (auto &p : pythonConfig)
    {
        const std::string &key = p.first;
        const std::string &value = p.second;
        auto itr = parserConfigTable.find(key);
        if (itr == parserConfigTable.end())
        {
            return false;
        }

        auto policyItr = configPolicyTable.find(key);
        if (policyItr != configPolicyTable.end())
        {
            const std::string &policyGroup = policyItr->second;
            // 兼容旧参数名仅保留解析能力，使用时提示迁移到标准名（不体现于用户手册/帮助信息）
            if (key != policyGroup)
            {
                std::cerr << "[msmemscope] Deprecation: '" << key << "' is deprecated, use '" << policyGroup
                          << "' instead." << std::endl;
            }
            // 只允许设置一次：配置已生效时不可再改，或本次调用中同组参数（含兼容旧名）已设置过
            if (config.isEffective ||
                std::find(onceSetGroups.begin(), onceSetGroups.end(), policyGroup) != onceSetGroups.end())
            {
                std::cout << "[msmemscope] Warn: Config:\"format\",\"output_path\",\"watch\" can only be set once."
                          << std::endl;
                continue;
            }
            onceSetGroups.emplace_back(policyGroup);
        }
        bool parseFail = false;
        itr->second(value, config, parseFail);
        if (parseFail)
        {
            return false;
        }
    }
    SetConfigImpl(config);
    SetEffectiveConfig(config_);
    Utility::SetLogLevel(static_cast<LogLv>(config_.logLevel));
    Utility::JsonConfig::GetInstance().SaveConfigToJson(config_);
    EventTraceManager::Instance().HandleWithATenCollect();
    EventTraceManager::Instance().HandleWithDecompose();
    EventTraceManager::Instance().HandleWithCpuTensorCollect();
    EventTraceManager::Instance().InitJudgeFuncTable();
    // 更新analysis参数
    EventReport::Instance(MemScopeCommType::SHARED_MEMORY).UpdateAnalysisType();

    return true;
}

bool IsNeedTraceOp() { return BitPresent(GetConfig().levelType, static_cast<size_t>(LevelType::LEVEL_OP)); }
bool IsNeedTraceKernel() { return BitPresent(GetConfig().levelType, static_cast<size_t>(LevelType::LEVEL_KERNEL)); }
bool IsNeedTraceAccess() { return BitPresent(GetConfig().eventType, static_cast<size_t>(EventType::ACCESS_EVENT)); }
bool IsNeedTraceLaunch() { return BitPresent(GetConfig().eventType, static_cast<size_t>(EventType::LAUNCH_EVENT)); }
bool IsNeedTraceAlloc() { return BitPresent(GetConfig().eventType, static_cast<size_t>(EventType::ALLOC_EVENT)); }
bool IsNeedTraceFree() { return BitPresent(GetConfig().eventType, static_cast<size_t>(EventType::FREE_EVENT)); }
bool IsNeedTraceKernelLaunch() { return IsNeedTraceKernel() && IsNeedTraceLaunch(); }
bool IsNeedTraceOpLaunch() { return IsNeedTraceOp() && IsNeedTraceLaunch(); }
bool IsNeedTraceMemory() { return IsNeedTraceAlloc() && IsNeedTraceFree(); }

TraceMode DetermineTraceMode()
{
    // 抑制窗口内（仪器自身调用真实运行时接口期间）触发的事件为运行时内部行为，
    // 直接跳过上报，防止递归上报与幻影事件
    if (IsEventReportSuppressed())
    {
        return TraceMode::SKIP;
    }

    bool needNormalTrace = EventTraceManager::Instance().IsNeedTrace(EventBaseType::MALLOC) ||
                           EventTraceManager::Instance().IsNeedTrace(EventBaseType::FREE);
    bool needShadowTrace = EventTraceManager::Instance().ShouldCollectShadowEvents();

    if (needNormalTrace)
    {
        return TraceMode::NORMAL;
    }
    if (needShadowTrace)
    {
        return TraceMode::SHADOW;
    }
    return TraceMode::SKIP;
}

void EventTraceManager::InitJudgeFuncTable()
{
    judgeFuncTable_ = {
        {EventBaseType::KERNEL_LAUNCH, []() { return IsNeedTraceKernelLaunch(); }},
        {EventBaseType::MALLOC, []() { return IsNeedTraceMemory(); }},
        {EventBaseType::FREE, []() { return IsNeedTraceMemory(); }},
        {EventBaseType::OP_LAUNCH, []() { return IsNeedTraceOpLaunch(); }},
        {EventBaseType::ACCESS, []() { return IsNeedTraceAccess(); }},
    };
};

// 1、判断是否处在采集范围
// 2、判断当前的采集项是否需要采集
bool EventTraceManager::IsNeedTrace(EventBaseType type)
{
    if (status_ != EventTraceStatus::IN_TRACING)
    {
        return false;
    }

    // 单例类析构之后不再访问其成员变量
    if (destroyed_.load())
    {
        return false;
    }

    auto itr = judgeFuncTable_.find(type);
    if (itr == judgeFuncTable_.end())
    {
        return true;
    }

    return itr->second();
}

bool EventTraceManager::IsTracingEnabled()
{
    if (status_ != EventTraceStatus::IN_TRACING)
    {
        return false;
    }

    // 单例类析构之后不再访问其成员变量
    if (destroyed_.load())
    {
        return false;
    }

    return true;
}

bool EventTraceManager::ShouldCollectShadowEvents()
{
    if (status_ != EventTraceStatus::NOT_IN_TRACING)
    {
        return false;
    }
    if (!IsNeedTraceAlloc() && !IsNeedTraceFree())
    {
        return false;
    }
    return true;
}

void EventTraceManager::PromoteHistoricalStates()
{
    auto &dump = Dump::GetInstance();
    auto dumpFunc = [&dump](MemoryState *state) { dump.DumpHistoricalState(state); };
    // 委托给MemoryStateManager内部加锁处理，避免多线程竞态
    MemoryStateManager::GetInstance().PromoteShadowStates(dumpFunc);
}

void EventTraceManager::InitTraceStatus()
{
    auto status = (GetConfig().collectMode == static_cast<uint8_t>(CollectMode::IMMEDIATE)) && GetConfig().isEffective
                      ? EventTraceStatus::IN_TRACING
                      : EventTraceStatus::NOT_IN_TRACING;
    status_ = status;
    return;
}

void EventTraceManager::SetTraceStatus(const EventTraceStatus status)
{
    if (status == EventTraceStatus::IN_TRACING)
    {
        // 先执行历史转正（在status_切换之前，确保转正过程使用当前正确的MemoryState快照）
        PromoteHistoricalStates();
        std::cout << "[msmemscope] Info: Start tracing.\n";
    }
    else if (status == EventTraceStatus::NOT_IN_TRACING)
    {
        std::cout << "[msmemscope] Info: Stop tracing.\n";
    }
    if (!EventReport::Instance(MemScopeCommType::SHARED_MEMORY).ReportTraceStatus(status))
    {
        std::cout << "[msmemscope] Error: Report trace status failed.\n";
    }

    std::lock_guard<std::mutex> lock(mutex_);

    status_ = status;

    HandleWithATenCollect();
    HandleWithDecompose();
    HandleWithCpuTensorCollect();
    return;
}

void EventTraceManager::HandleWithATenCollect()
{
    if ((status_ == EventTraceStatus::IN_TRACING) && IsNeedTraceOp() && aclInit_)
    {
        Utility::MemScopePythonCall("msmemscope.aten_collection", "enable_aten_collector");
        Utility::MemScopePythonCall("msmemscope.triton_collection", "enable_triton_collector");
        return;
    }

    Utility::MemScopePythonCall("msmemscope.triton_collection", "disable_triton_collector");
    Utility::MemScopePythonCall("msmemscope.aten_collection", "disable_aten_collector");

    return;
}

void EventTraceManager::HandleWithDecompose()
{
    if ((status_ == EventTraceStatus::IN_TRACING) &&
        BitPresent(GetConfig().analysisType, static_cast<size_t>(AnalysisType::DECOMPOSE_ANALYSIS)) && aclInit_)
    {
        Utility::MemScopePythonCall("msmemscope.optimizer_step_hook", "enable_optimizer_step_hook");
        Utility::MemScopePythonCall("msmemscope.hijacker.hijack_manager", "enable_decompose_hooks");
        return;
    }

    Utility::MemScopePythonCall("msmemscope.optimizer_step_hook", "disable_optimizer_step_hook");
    Utility::MemScopePythonCall("msmemscope.hijacker.hijack_manager", "disable_decompose_hooks");

    return;
}

void EventTraceManager::HandleWithCpuTensorCollect()
{
    if ((status_ == EventTraceStatus::IN_TRACING) && GetConfig().collectCpu)
    {
        Utility::MemScopePythonCall("msmemscope.cpu_tensor_collection", "enable_cpu_tensor_collect");
        return;
    }

    Utility::MemScopePythonCall("msmemscope.cpu_tensor_collection", "disable_cpu_tensor_collect");

    return;
}

void EventTraceManager::SetAclInitStatus(bool isInit)
{
    aclInit_ = isInit;

    HandleWithATenCollect();
    HandleWithDecompose();
    HandleWithCpuTensorCollect();
}

void EventTraceManager::SetDeviceReadyStatus(bool isReady)
{
    if (!isReady)
    {
        return;
    }

    bool expected = false;
    if (!deviceReady_.compare_exchange_strong(expected, true))
    {
        return;  // device readiness already signaled once
    }

    if ((status_ == EventTraceStatus::IN_TRACING) && GetConfig().collectCpu)
    {
        Utility::MemScopePythonCall("msmemscope.cpu_tensor_collection", "on_device_ready");
    }
}

bool EventTraceManager::IsDeviceReady() { return deviceReady_.load(); }

void EventTraceManager::CleanUpEventTraceManager()
{
    // 这里可以添加其他的CleanUp操作,最好把析构函数中的抽象出来,放到stop实现
    Dump::GetInstance().WritePublicEventToFile();
    Dump::GetInstance().FflushEventToFile();
}
}  // namespace MemScope
