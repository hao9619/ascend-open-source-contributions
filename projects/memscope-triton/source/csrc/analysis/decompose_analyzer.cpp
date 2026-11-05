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

#include "decompose_analyzer.h"

#include <string>
#include <vector>

#include "constant.h"
#include "describe_trace.h"
#include "event_dispatcher.h"

namespace MemScope
{

const std::string DecomposeAnalyzer::cannStr = "CANN";
const std::string DecomposeAnalyzer::ptaStr = "PTA";
const std::string DecomposeAnalyzer::ptaWorkspaceStr = "PTA_WORKSPACE";
const std::string DecomposeAnalyzer::atbStr = "ATB";
const std::string DecomposeAnalyzer::mindsporeStr = "MINDSPORE";

// 算子访问标记: 存放于细化分类2(DETAIL_2), 不带前导@(由GetOwnerStr按级别拼接)
const std::string DecomposeAnalyzer::opsStr = "ops";

DecomposeAnalyzer& DecomposeAnalyzer::GetInstance()
{
    static DecomposeAnalyzer analyzer{};
    return analyzer;
}

DecomposeAnalyzer::DecomposeAnalyzer() { DecomposeAnalyzer::Subscribe(); }

void DecomposeAnalyzer::EventHandle(std::shared_ptr<EventBase>& event, MemoryState* state)
{
    // Skip shadow/historical events
    if (auto memEvent = std::dynamic_pointer_cast<MemoryEvent>(event))
    {
        if (memEvent->isShadowEvent)
        {
            return;
        }
    }

    if (event->eventType == EventBaseType::MALLOC)
    {
        auto memEvent = std::dynamic_pointer_cast<MemoryEvent>(event);
        if (memEvent && memEvent->poolType == PoolType::HOST)
        {
            return;  // CPU memory events are not analyzed
        }
        if (memEvent != nullptr && state != nullptr)
        {
            InitOwner(memEvent, state);
        }
    }
    else if (event->eventType == EventBaseType::ACCESS)
    {
        auto memEvent = std::dynamic_pointer_cast<MemoryEvent>(event);
        if (memEvent != nullptr && state != nullptr)
        {
            UpdateOwnerByAtenAccess(memEvent, state);
        }
    }
    else if (event->eventType == EventBaseType::MEMORY_OWNER)
    {
        auto memOwnerEvent = std::dynamic_pointer_cast<MemoryOwnerEvent>(event);
        if (memOwnerEvent != nullptr && state != nullptr)
        {
            UpdateOwner(memOwnerEvent, state);
        }
    }
}

void DecomposeAnalyzer::InitOwner(std::shared_ptr<MemoryEvent>& event, MemoryState* state)
{
    // 框架级标签: 由分配器来源(事件subtype)填充, 不来自describe标签
    std::string framework;
    switch (event->eventSubType)
    {
        case EventSubType::HAL:
        {
            auto it = MODULE_HASH_TABLE.find(event->moduleId);
            if (it != MODULE_HASH_TABLE.end())
            {
                framework = cannStr + "@" + it->second;
            }
            else
            {
                framework = cannStr + "@UNKNOWN";
            }
            break;
        }
        case EventSubType::PTA_CACHING:
            framework = ptaStr;
            break;
        case EventSubType::PTA_WORKSPACE:
            framework = ptaWorkspaceStr;
            break;
        case EventSubType::MINDSPORE:
            framework = mindsporeStr;
            break;
        case EventSubType::ATB:
            framework = atbStr;
            break;
        default:
            break;
    }
    if (!framework.empty())
    {
        state->owner.AddLabel(OwnerLevel::FRAMEWORK, framework);
    }

    // 其余级别: 分析时直接从DescribeTrace读取(采集与分析同线程同步路由,
    // 线程局部标签栈即为申请时刻的状态, 无需随事件携带; FRAMEWORK槽由分配器来源占用)
    std::vector<std::string> labels = DescribeTrace::GetInstance().GetDescribe();
    for (uint8_t level = static_cast<uint8_t>(OwnerLevel::COMPONENT);
         level <= static_cast<uint8_t>(OwnerLevel::DETAIL_2); ++level)
    {
        if (!labels[level].empty())
        {
            state->owner.AddLabel(static_cast<OwnerLevel>(level), labels[level]);
        }
    }
    for (uint8_t i = 0; i < 3; ++i)
    {
        uint8_t level = static_cast<uint8_t>(OwnerLevel::USER_DEFINED_1) + i;
        if (!labels[level].empty())
        {
            state->owner.AddLabel(static_cast<OwnerLevel>(level), labels[level]);
        }
    }
}

void DecomposeAnalyzer::UpdateOwnerByAtenAccess(std::shared_ptr<MemoryEvent>& event, MemoryState* state)
{
    if (event->eventSubType != EventSubType::ATEN_READ && event->eventSubType != EventSubType::ATEN_WRITE &&
        event->eventSubType != EventSubType::ATEN_READ_OR_WRITE &&
        event->eventSubType != EventSubType::TRITON_READ && event->eventSubType != EventSubType::TRITON_WRITE &&
        event->eventSubType != EventSubType::TRITON_UNKNOWN)
    {
        return;
    }

    // ATEN 访问为弱标记: 细化分类2(DETAIL_2)为空时才写入, 不覆盖已有细化标签
    if (state->owner.GetLabel(OwnerLevel::DETAIL_2).empty())
    {
        state->owner.AddLabel(OwnerLevel::DETAIL_2, opsStr);
    }
}

void DecomposeAnalyzer::UpdateOwner(std::shared_ptr<MemoryOwnerEvent>& event, MemoryState* state)
{
    // 地址直标: 逐级更新块owner, 同级别重复时以地址标签为准(AddLabel覆盖语义)
    for (const auto& item : event->ownerLabels)
    {
        if (item.second.empty())
        {
            continue;
        }
        state->owner.AddLabel(item.first, item.second);
    }
}

DecomposeAnalyzer::~DecomposeAnalyzer() { UnSubscribe(); }

const char* DecomposeAnalyzer::GetName() const { return "decompose"; }

void DecomposeAnalyzer::Subscribe()
{
    auto func = std::bind(&DecomposeAnalyzer::EventHandle, this, std::placeholders::_1, std::placeholders::_2);
    std::vector<EventBaseType> eventList{EventBaseType::MALLOC, EventBaseType::ACCESS, EventBaseType::MEMORY_OWNER};
    EventDispatcher::GetInstance().Subscribe(SubscriberId::DECOMPOSE_ANALYZER, eventList,
                                             EventDispatcher::Priority::High, func, GetName());
}

void DecomposeAnalyzer::UnSubscribe() const
{
    EventDispatcher::GetInstance().UnSubscribe(SubscriberId::DECOMPOSE_ANALYZER);
}

}  // namespace MemScope
