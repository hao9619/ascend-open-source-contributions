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

#include "inefficient_analyzer.h"

#include <string>
#include <tuple>

#include "event_dispatcher.h"
namespace MemScope
{

InefficientAnalyzer& InefficientAnalyzer::GetInstance()
{
    static InefficientAnalyzer analyzer{};
    return analyzer;
}

InefficientAnalyzer::InefficientAnalyzer() { InefficientAnalyzer::Subscribe(); }

// op_Launch 开始和结束大概率state都会为空，因为free之后会清掉buffer
void InefficientAnalyzer::EventHandle(std::shared_ptr<EventBase>& event, MemoryState* state)
{
    // Skip shadow/historical events
    if (auto memEvent = std::dynamic_pointer_cast<MemoryEvent>(event))
    {
        if (memEvent->isShadowEvent)
        {
            return;
        }
    }

    const auto eventType = event->eventType;
    const auto eventSubType = event->eventSubType;

    // 处理OP_LAUNCH类型事件
    if (eventType == EventBaseType::OP_LAUNCH)
    {
        std::unordered_map<uint64_t, PidState>& pidStatesMap =
            (eventSubType == EventSubType::ATEN_START || eventSubType == EventSubType::ATEN_END ||
             eventSubType == EventSubType::TRITON_START || eventSubType == EventSubType::TRITON_END)
                ? PTAPidStatesMap
                : ATBPidStatesMap;
        InitNewPidState(event->pid, pidStatesMap);
        HandleOpLaunchEvent(event, pidStatesMap);
        return;
    }

    // 处理MALLOC FREE ACCESS事件
    if (eventType == EventBaseType::MALLOC || eventType == EventBaseType::FREE || eventType == EventBaseType::ACCESS)
    {
        if ((event->poolType != PoolType::PTA_CACHING && event->poolType != PoolType::ATB))
        {
            return;
        }
        std::unordered_map<uint64_t, PidState>& pidStatesMap =
            (event->poolType == PoolType::PTA_CACHING) ? PTAPidStatesMap : ATBPidStatesMap;
        InitNewPidState(event->pid, pidStatesMap);
        HandleMemoryEvent(event, state, pidStatesMap);
    }
}

void InefficientAnalyzer::InitNewPidState(const uint64_t pid, std::unordered_map<uint64_t, PidState>& pidStatesMap)
{
    auto initResult =
        pidStatesMap.emplace(std::piecewise_construct, std::forward_as_tuple(pid), std::forward_as_tuple());
    if (initResult.second)
    {
        Init(initResult.first->second);
    }
}

void InefficientAnalyzer::Init(PidState& pidState)
{
    pidState.apiId = 0;
    pidState.mallocApiTmpId = MAX_UNIT64;
    pidState.freeApiTmpId = MAX_UNIT64;
    pidState.isOpStart = false;
}

void InefficientAnalyzer::HandleOpLaunchEvent(std::shared_ptr<EventBase>& event,
                                              std::unordered_map<uint64_t, PidState>& pidStatesMap)
{
    const auto eventSubType = event->eventSubType;
    const uint64_t pid = event->pid;
    auto& pidState = pidStatesMap[pid];

    // 遇到start，api id 加1
    if (eventSubType == EventSubType::ATB_START || eventSubType == EventSubType::ATEN_START ||
        eventSubType == EventSubType::TRITON_START)
    {
        pidState.isOpStart = true;
        UpdateApiId(pidState);
        return;
    }
    // 遇到end，判断tmp api属于什么类型
    if (eventSubType == EventSubType::ATB_END || eventSubType == EventSubType::ATEN_END ||
        eventSubType == EventSubType::TRITON_END)
    {
        pidState.isOpStart = false;
        ClassifyEventsTmp(pidState);
        return;
    }
}

void InefficientAnalyzer::HandleMemoryEvent(std::shared_ptr<EventBase>& event, MemoryState* state,
                                            std::unordered_map<uint64_t, PidState>& pidStatesMap)
{
    // 一次查找后沿调用链复用，避免每事件对同一pid反复哈希
    auto& pidState = pidStatesMap[event->pid];
    auto memEvent = std::dynamic_pointer_cast<MemoryEvent>(event);
    if (!memEvent || !state)
    {
        return;
    }
    // 当前面没有START事件时，即表明当前事件为独立事件，M/A/F均API ID加1，并判断此时事件属于malloc还是free api
    if (!pidState.isOpStart)
    {
        UpdateApiId(pidState);
        AddEventToTmps(memEvent, pidState);
        AddApiIdToState(memEvent, state, pidState);
        ClassifyEventsTmp(pidState);
    }
    else
    {
        AddEventToTmps(memEvent, pidState);
        AddApiIdToState(memEvent, state, pidState);
    }

    InefficientAnalysis(memEvent, state, pidState);
}

void InefficientAnalyzer::ClassifyEventsTmp(PidState& pidState)
{
    if (pidState.apiTmp.empty())
    {
        return;
    }
    std::vector<std::shared_ptr<MemoryEvent>> local_events;
    local_events = std::move(pidState.apiTmp);
    bool hasMalloc = false;
    bool hasFree = false;

    for (const auto& event : local_events)
    {
        if (event->eventType == EventBaseType::MALLOC)
        {
            hasMalloc = true;
        }
        if (event->eventType == EventBaseType::FREE)
        {
            hasFree = true;
        }
    }

    if (hasMalloc)
    {
        pidState.mallocApiTmpId = pidState.apiId;
    }
    if (hasFree)
    {
        pidState.freeApiTmpId = pidState.apiId;
    }
}

void InefficientAnalyzer::UpdateApiId(PidState& pidState) { pidState.apiId.fetch_add(1, std::memory_order_relaxed); }

void InefficientAnalyzer::AddEventToTmps(const std::shared_ptr<MemoryEvent>& event, PidState& pidState)
{
    pidState.apiTmp.push_back(event);
}

void InefficientAnalyzer::AddApiIdToState(std::shared_ptr<MemoryEvent>& event, MemoryState* state, PidState& pidState)
{
    state->apiId.push_back(pidState.apiId);
}

void InefficientAnalyzer::InefficientAnalysis(std::shared_ptr<MemoryEvent>& event, MemoryState* state,
                                              PidState& pidState)
{
    // 存在部分state中的events开头不是MALLOC事件。和部分events的长度与apiId长度不相等的情况
    if (state->events.at(0)->eventType != EventBaseType::MALLOC || state->events.size() != state->apiId.size())
    {
        return;
    }

    if (event->eventType == EventBaseType::ACCESS)
    {
        TemporaryIdleness(event, state);

        if (pidState.freeApiTmpId == MAX_UNIT64)
        {
            return;
        }

        EarlyAllocation(event, state, pidState);
    }
    if (event->eventType == EventBaseType::FREE && pidState.mallocApiTmpId != MAX_UNIT64)
    {
        LateDeallocation(event, state, pidState);
    }
}

void InefficientAnalyzer::EarlyAllocation(std::shared_ptr<MemoryEvent>& event, MemoryState* state, PidState& pidState)
{
    // 过早申请判断：1.先找到FIRST ACCESS(FA)。2.判断MALLOC和FA之间有无FREE API
    if (state->inefficientType.find("early_allocation") != std::string::npos)
    {
        return;
    }

    uint64_t eventsLen = state->events.size();
    uint64_t firstAccessApiId = MAX_UNIT64;
    uint64_t mallocApiId = 0;
    // 1.找到MALLOC的API值。2.找到第一个API值不等于MALLOC的API的ACCESS，找到即结束
    for (uint64_t i = 0; i < eventsLen; i++)
    {
        if (firstAccessApiId < MAX_UNIT64)
        {
            break;
        }
        if (state->events[i]->eventType == EventBaseType::MALLOC)
        {
            mallocApiId = state->apiId[i];
        }
        else if (state->events[i]->eventType == EventBaseType::ACCESS && state->apiId[i] != mallocApiId)
        {
            firstAccessApiId = state->apiId[i];
        }
    }
    // 1.如果没找到第一个ACCESS值，说明此时ACCESS API与MALLOC的相等
    // 2.如果MALLOC和最近的FREE事件在同一个API中，则无法交换。
    // ACCESS所在API此时未结束，还没有将此时API更新至最近MALLOC或者FREE的ID，所以FA 和 FREE事件API值不可能相等
    if (firstAccessApiId == MAX_UNIT64 || mallocApiId == pidState.freeApiTmpId)
    {
        return;
    }
    // 如果FREE的API Id在FA与MALLOC之间，则判断MALLOC内存块为过早申请
    if (firstAccessApiId > pidState.freeApiTmpId && mallocApiId < pidState.freeApiTmpId)
    {
        if (!state->inefficientType.empty())
        {
            state->inefficientType += ",";
        }
        state->inefficientType += "early_allocation";
    }
}

void InefficientAnalyzer::LateDeallocation(std::shared_ptr<MemoryEvent>& event, MemoryState* state, PidState& pidState)
{
    // 1.找到LAST ACCESS。2.判断FREE API与LA API之间有无MALLOC API
    uint64_t eventsLen = state->events.size();
    uint64_t lastAccessApiId = MAX_UNIT64;
    // 此时内存块需要具有MALLOC、ACCESS、FREE的完整事件，且在state->events中具有顺序，因此events中倒数第二个事件即为LA.
    if (eventsLen >= MIN_EVENTS_NUM)
    {
        if (state->events[eventsLen - MIN_EVENTS_NUM]->eventType == EventBaseType::ACCESS)
        {
            lastAccessApiId = state->apiId[eventsLen - MIN_EVENTS_NUM];
        }
    }

    // 没有找到LA, 或者LA在当前API中
    if (lastAccessApiId == MAX_UNIT64 || lastAccessApiId == pidState.apiId)
    {
        return;
    }

    // 如果LA与FREE中存在MALLOC的API，则说明为过迟释放
    if (lastAccessApiId < pidState.mallocApiTmpId && pidState.apiId > pidState.mallocApiTmpId)
    {
        if (!state->inefficientType.empty())
        {
            state->inefficientType += ",";
        }
        state->inefficientType += "late_deallocation";
    }
}

void InefficientAnalyzer::TemporaryIdleness(std::shared_ptr<MemoryEvent>& event, MemoryState* state)
{
    // 不同的两个ACCESS API值大于某个阈值，即为临时闲置
    if (state->inefficientType.find("temporary_idleness") != std::string::npos)
    {
        return;
    }
    uint64_t eventLen = state->events.size();
    // 当events不足2个或者倒数第二个事件不为ACCESS时，不用判断，此时最后一个events必定为ACCESS
    if (eventLen < MIN_EVENTS_NUM || state->events[eventLen - MIN_EVENTS_NUM]->eventType != EventBaseType::ACCESS)
    {
        return;
    }
    // 当最后一个API值大于倒数第2个API值 并且 其差值大于阈值时，判为临时闲置
    if (state->apiId[eventLen - LAST_EVENTS_NUM] > state->apiId[eventLen - MIN_EVENTS_NUM] &&
        state->apiId[eventLen - LAST_EVENTS_NUM] - state->apiId[eventLen - MIN_EVENTS_NUM] > THREHOLD)
    {
        if (!state->inefficientType.empty())
        {
            state->inefficientType += ",";
        }
        state->inefficientType += "temporary_idleness";
    }
}

InefficientAnalyzer::~InefficientAnalyzer() { UnSubscribe(); }

const char* InefficientAnalyzer::GetName() const { return "inefficient"; }

void InefficientAnalyzer::Subscribe()
{
    auto func = std::bind(&InefficientAnalyzer::EventHandle, this, std::placeholders::_1, std::placeholders::_2);
    std::vector<EventBaseType> eventTypes{EventBaseType::MALLOC, EventBaseType::ACCESS, EventBaseType::FREE,
                                          EventBaseType::OP_LAUNCH};
    EventDispatcher::GetInstance().Subscribe(SubscriberId::INEFFICIENT_ANALYZER, eventTypes,
                                             EventDispatcher::Priority::High, func, GetName());
}

void InefficientAnalyzer::UnSubscribe() const
{
    EventDispatcher::GetInstance().UnSubscribe(SubscriberId::INEFFICIENT_ANALYZER);
}

}  // namespace MemScope
