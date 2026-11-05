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
#include "mstx_manager.h"

#include <cstring>

#include "aten_manager.h"
#include "bit_field.h"
#include "call_stack.h"
#include "event_report.h"
#include "log.h"
#include "memory_pool_trace/atb_memory_pool_trace.h"
#include "memory_pool_trace/memory_pool_trace_manager.h"
#include "memory_pool_trace/mindspore_memory_pool_trace.h"
#include "memory_pool_trace/pta_caching_pool_trace.h"
#include "memory_pool_trace/pta_workspace_pool_trace.h"
#include "op_handler.h"
#include "record_info.h"
#include "securec.h"

namespace MemScope
{

aclError GetStreamID(aclrtStream stream, int32_t* streamId)
{
    char const* sym = "aclrtStreamGetIdImpl";
    using AclrtGetStreamID = decltype(&GetStreamID);
    static AclrtGetStreamID vallina = nullptr;
    if (vallina == nullptr)
    {
        vallina = VallinaSymbol<ACLImplLibLoader>::Instance().Get<AclrtGetStreamID>(sym);
    }
    if (vallina == nullptr)
    {
        LOG_ERROR("vallina func get FAILED: %s", __func__);
        return ACL_ERROR_RT_FAILURE;
    }
    aclError ret = vallina(stream, streamId);
    return ret;
}

// 组装普通打点信息
void MstxManager::ReportMarkA(const char* msg, aclrtStream stream, MemScopeCommType type)
{
    // 处理sanitizer-op算子上报信息
    if (msg && strncmp(msg, SANITIZER_OP_MSG, strlen(SANITIZER_OP_MSG)) == 0)
    {
        if (!SanitizerOpHandler::IsEnabled())
        {
            return;
        }
        const char* opMsg = msg + strlen(SANITIZER_OP_MSG);
        SanitizerOpHandler::GetInstance().Handle(opMsg, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream)));
        return;
    }

    if (!EventTraceManager::Instance().IsTracingEnabled())
    {
        return;
    }

    if (msg && strncmp(msg, TRITON_MSG, strlen(TRITON_MSG)) == 0)
    {
        EventReport::Instance(type).ReportTritonMessage(msg + strlen(TRITON_MSG));
        return;
    }

    int32_t streamId = -1;
    GetStreamID(stream, &streamId);

    // 处理aten算子上报信息
    if (msg && strncmp(msg, ATEN_MSG, strlen(ATEN_MSG)) == 0)
    {
        const char* atenMsg = msg + strlen(ATEN_MSG);
        AtenManager::GetInstance().ProcessMsg(atenMsg, streamId);
        return;
    }

    // msg来自被插桩应用,可能为空,判空后按空串处理(与ReportDomainCreateA一致)
    std::string markMsg = msg ? std::string(msg) : "";
    if (!EventReport::Instance(type).ReportMark(MarkType::MARK_A, markMsg, streamId, onlyMarkId_))
    {
        LOG_ERROR("Report Mark FAILED");
    }
}

// 组装Range开始打点信息
uint64_t MstxManager::ReportRangeStart(const char* msg, aclrtStream stream)
{
    uint64_t rangeId = GetRangeId();
    // msg来自被插桩应用,可能为空,判空后按空串处理(与ReportDomainCreateA一致)
    std::string markMsg = msg ? std::string(msg) : "";
    int32_t streamId = -1;
    GetStreamID(stream, &streamId);
    if (!EventReport::Instance(MemScopeCommType::SHARED_MEMORY)
             .ReportMark(MarkType::RANGE_START_A, markMsg, streamId, rangeId))
    {
        LOG_ERROR("Report Mark FAILED");
    }
    return rangeId;
}

// 组装Range结束打点信息
void MstxManager::ReportRangeEnd(uint64_t id)
{
    std::string msg = "Range end from id " + std::to_string(id);
    if (!EventReport::Instance(MemScopeCommType::SHARED_MEMORY).ReportMark(MarkType::RANGE_END, msg, -1, id))
    {
        LOG_ERROR("Report Mark FAILED");
    }
}

uint64_t MstxManager::GetRangeId() { return rangeId_++; }
// MSTX针对内存池的分析功能 这里进行代码重构 和上面的打点功能剥离
mstxDomainHandle_t MstxManager::ReportDomainCreateA(char const* domainName)
{
    // 构造一次用于后续多次比较，避免重复构造临时string
    const std::string domain(domainName ? domainName : "");
    // 后续收编所有通过MSTX打点的内存池trace
    if (domain == "atb")
    {
        if (MemoryPoolTraceManager::GetInstance().RegisterMemoryPoolTracer("atb", &ATBMemoryPoolTrace::GetInstance()))
        {
            return MemoryPoolTraceManager::GetInstance().CreateDomain(domainName);
        }
    }
    if (domain == "mindsporeMemPool")
    {
        if (MemoryPoolTraceManager::GetInstance().RegisterMemoryPoolTracer("mindsporeMemPool",
                                                                           &MindsporeMemoryPoolTrace::GetInstance()))
        {
            return MemoryPoolTraceManager::GetInstance().CreateDomain(domainName);
        }
    }
    // PTA会进行多次内存池注册 已经注册过就返回之前注册的domain
    if (domain == "ptaCaching" || domain == "msleaks")
    {
        MemoryPoolTraceManager::GetInstance().RegisterMemoryPoolTracer("ptaCaching",
                                                                       &PTACachingPoolTrace::GetInstance());
        return MemoryPoolTraceManager::GetInstance().CreateDomain(
            "ptaCaching");  // 考虑到PTA的兼容性，目前不论接受到老的还是新的都统一为ptaCaching
    }
    // PTAWorkspace与PTACaching由不同的内存池进行管理
    if (domain == "ptaWorkspace")
    {
        MemoryPoolTraceManager::GetInstance().RegisterMemoryPoolTracer("ptaWorkspace",
                                                                       &PTAWorkspacePoolTrace::GetInstance());
        return MemoryPoolTraceManager::GetInstance().CreateDomain(domainName);
    }
    return nullptr;
}

mstxMemHeapHandle_t MstxManager::ReportHeapRegister(mstxDomainHandle_t domain, mstxMemHeapDesc_t const* desc)
{
    auto tracer = MemoryPoolTraceManager::GetInstance().GetMemoryPoolTracer(domain);
    if (tracer)
    {
        return tracer->Allocate(domain, desc);
    }
    return nullptr;
}

void MstxManager::ReportHeapUnregister(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap)
{
    auto tracer = MemoryPoolTraceManager::GetInstance().GetMemoryPoolTracer(domain);
    if (tracer)
    {
        return tracer->Deallocate(domain, heap);
    }
}

void MstxManager::ReportRegionsRegister(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const* desc)
{
    auto tracer = MemoryPoolTraceManager::GetInstance().GetMemoryPoolTracer(domain);
    if (tracer)
    {
        return tracer->Reallocate(domain, desc);
    }
}

void MstxManager::ReportRegionsUnregister(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const* desc)
{
    auto tracer = MemoryPoolTraceManager::GetInstance().GetMemoryPoolTracer(domain);
    if (tracer)
    {
        return tracer->Release(domain, desc);
    }
}

}  // namespace MemScope
