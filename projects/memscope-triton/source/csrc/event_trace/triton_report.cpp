/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under Mulan PSL v2: http://license.coscl.org.cn/MulanPSL2
 */
#include <cstring>
#include <limits>
#include <stdexcept>

#include "call_stack.h"
#include "event_report.h"
#include "log.h"
#include "nlohmann/json.hpp"

namespace MemScope
{
namespace
{
using Json = nlohmann::json;
constexpr size_t MAX_TRITON_MESSAGE = 1024 * 1024;
constexpr size_t MAX_TRITON_PARAMETERS = 1024;

std::string Identifier(const Json& value)
{
    const auto text = value.get<std::string>();
    if (text.empty() || text.size() > 4096 ||
        text.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$") !=
            std::string::npos)
    {
        throw std::invalid_argument("Invalid Triton identifier");
    }
    return text;
}

uint64_t Unsigned(const Json& value)
{
    if (!value.is_number_unsigned())
    {
        throw std::invalid_argument("Expected unsigned Triton metadata");
    }
    return value.get<uint64_t>();
}

void ValidateAccess(const Json& access)
{
    Identifier(access.at("parameter"));
    const auto address = Unsigned(access.at("ptr"));
    const auto size = Unsigned(access.at("size"));
    if (address == 0 || size == 0 || size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        address > std::numeric_limits<uint64_t>::max() - size)
    {
        throw std::invalid_argument("Invalid Triton parameter extent");
    }
    for (const auto* field : {"read", "write", "unknown"})
    {
        if (!access.at(field).is_boolean())
        {
            throw std::invalid_argument("Invalid Triton memory effect");
        }
    }
}

std::string LaunchAttributes(const Json& message)
{
    return "collector:triton,call_id:" + std::to_string(Unsigned(message.at("call_id"))) +
           ",stream_handle:" + std::to_string(Unsigned(message.at("stream_handle"))) +
           ",extent_kind:parameter_storage_span,effect_kind:static_possible";
}

void SendAccess(const Json& access, const std::string& name, const std::string& attributes, int32_t device,
                const std::string& stack, EventSubType subtype)
{
    auto event = std::make_shared<MemoryEvent>();
    event->eventType = EventBaseType::ACCESS;
    event->eventSubType = subtype;
    event->poolType = PoolType::PTA_CACHING;
    event->device = device;
    event->name = name;
    event->addr = access.at("ptr").get<uint64_t>();
    event->size = access.at("size").get<int64_t>();
    event->attr = attributes + ",parameter:" + access.at("parameter").get<std::string>();
    event->pyCallStack = stack;
    Process::GetInstance().SendEvent(event);
}
}  // namespace

bool EventReport::ReportTritonMessage(const char* message)
{
    if (message == nullptr || strnlen(message, MAX_TRITON_MESSAGE + 1) > MAX_TRITON_MESSAGE)
    {
        return false;
    }
    try
    {
        const auto payload = Json::parse(message);
        const auto name = Identifier(payload.at("name"));
        const auto phase = payload.at("phase").get<std::string>();
        const auto logicalDevice = Unsigned(payload.at("device"));
        if ((phase != "begin" && phase != "end") || logicalDevice > std::numeric_limits<int32_t>::max() ||
            payload.at("extent_kind") != "parameter_storage_span" || payload.at("effect_kind") != "static_possible")
        {
            return false;
        }
        const auto attributes = LaunchAttributes(payload);
        const auto emptyAccesses = Json::array();
        const auto& accesses = phase == "begin" ? payload.at("accesses") : emptyAccesses;
        if (!accesses.is_array() || accesses.size() > MAX_TRITON_PARAMETERS)
        {
            return false;
        }
        // Validate the entire message before emitting any event.
        for (const auto& access : accesses)
        {
            ValidateAccess(access);
        }
        int32_t device = static_cast<int32_t>(logicalDevice);
        if (!GetDeviceInfo::Instance().TransDeviceId(device) || IsNeedSkip(device))
        {
            return true;
        }
        std::string stack;
        if (GetConfig().enablePyStack)
        {
            Utility::GetPythonCallstack(GetConfig().pyStackDepth, stack);
        }
        auto& trace = EventTraceManager::Instance();
        if (trace.IsNeedTrace(EventBaseType::OP_LAUNCH))
        {
            auto event = std::make_shared<OpLaunchEvent>();
            event->eventType = EventBaseType::OP_LAUNCH;
            event->eventSubType = phase == "begin" ? EventSubType::TRITON_START : EventSubType::TRITON_END;
            event->device = device;
            event->name = name;
            event->attr = attributes;
            event->pyCallStack = stack;
            Process::GetInstance().SendEvent(event);
        }
        if (!trace.IsNeedTrace(EventBaseType::ACCESS))
        {
            return true;
        }
        for (const auto& access : accesses)
        {
            if (access.at("read").get<bool>())
            {
                SendAccess(access, name, attributes, device, stack, EventSubType::TRITON_READ);
            }
            if (access.at("write").get<bool>())
            {
                SendAccess(access, name, attributes, device, stack, EventSubType::TRITON_WRITE);
            }
            if (access.at("unknown").get<bool>())
            {
                SendAccess(access, name, attributes, device, stack, EventSubType::TRITON_UNKNOWN);
            }
        }
        return true;
    }
    catch (const std::exception& error)
    {
        LOG_WARN("Invalid Triton collection message: %s", error.what());
        return false;
    }
}
}  // namespace MemScope
