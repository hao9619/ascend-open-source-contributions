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

#ifndef EVENT_REPORT_H
#define EVENT_REPORT_H

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ascend_hal.h"
#include "config_info.h"
#include "dump.h"
#include "host_mem_hooks/host_mem_hooks.h"
#include "kernel_hooks/acl_hooks.h"
#include "kernel_hooks/kernel_event_trace.h"
#include "kernel_hooks/runtime_hooks.h"
#include "process.h"
#include "record_info.h"
#include "trace_manager/event_trace_manager.h"

namespace MemScope
{

constexpr mode_t REGULAR_MODE_MASK = 0177;
constexpr char TRITON_MSG[] = "memscope-triton:";
constexpr char ATEN_MSG[] = "memscope-aten-";
constexpr char ATEN_BEGIN_MSG[] = "b:";
constexpr char ATEN_END_MSG[] = "e:";
constexpr char ACCESS_MSG[] = "ac:";
constexpr char SANITIZER_OP_MSG[] = "sanitizer-op:";

struct MstxStepInfo
{
    uint64_t currentStepId = 0;
    bool inStepRange = false;  // 不在mstx_range_start和mstx_range_end之间的数据，不采集

    // 暂存用mstx标记Step的RangeId,与用mstx标记host采集的RangeId区分开
    // 可能存在多线程操作，例如线程A mstx_range_start,线程B mstx_range_start,线程A mstx_range_end,线程B mstx_range_end
    // 因此是数组
    std::vector<uint64_t> stepMarkRangeIdList;
};

/*
 * EventReport类主要功能：
 * 1. 将劫持记录的信息传回到工具进程
 */
class EventReport
{
   public:
    static EventReport& Instance(MemScopeCommType type);
    bool ReportHalCreate(uint64_t addr, uint64_t size, const drv_mem_prop& prop, CallStackString&& stack);
    bool ReportHalCreate(uint64_t addr, uint64_t size, const drv_mem_prop& prop);  // shadow mode (no callstack)
    bool ReportHalRelease(uint64_t addr, CallStackString&& stack);
    bool ReportHalMalloc(uint64_t addr, uint64_t size, unsigned long long flag, CallStackString&& stack);
    bool ReportHalMalloc(uint64_t addr, uint64_t size, unsigned long long flag);  // shadow mode (no callstack)
    bool ReportHalFree(uint64_t addr, CallStackString&& stack);
    bool ReportHalFree(uint64_t addr);  // shadow mode (no callstack)
    bool ReportHostRegister(uint64_t addr, uint64_t size, CallStackString&& stack);
    bool ReportHostUnregister(uint64_t addr, CallStackString&& stack);
    bool ReportKernelLaunch(const AclnnKernelMapInfo& kernelLaunchInfo);
    bool ReportKernelExcute(const TaskKey& key, std::string& name, uint64_t time, RecordSubType type);
    bool ReportAclItf(RecordSubType subtype);
    bool ReportTraceStatus(const EventTraceStatus status);
    bool ReportMark(MarkType type, std::string& msg, uint32_t streamId, uint64_t rangeId);
    bool ReportMemPoolRecord(EventSubType type, const MemoryUsage& info, CallStackString&& stack);
    bool ReportAtbOpExecute(const char* name, size_t nameSize, const char* attr, size_t attrSize, RecordSubType type);
    bool ReportAtbKernel(const char* name, size_t nameSize, const char* attr, size_t attrSize, RecordSubType type);
    bool ReportAtbAccessMemory(const char* name, size_t nameSize, const char* attr, size_t attrSize, uint64_t addr,
                               uint64_t size, AccessType type);
    bool ReportAtenLaunch(const std::string& name, bool isStart, std::string&& pystack);
    bool ReportAtenAccess(const std::string& name, const std::string& attr, AccessType type, uint64_t addr,
                          uint64_t size, std::string&& pystack);
    bool ReportAddrInfo(EventSubType type, uint64_t addr,
                        const std::vector<std::pair<OwnerLevel, std::string>>& labels);
    bool ReportPyStepRecord();
    // step控制字回显:进程级step计数只读(与ReportPyStepRecord同源pyStepId_)
    uint64_t GetPyStepId() const;
    bool ReportMemorySnapshot(const MemorySnapshotInfo& memory_info, CallStackString&& stack);
    void ReportMemorySnapshotOnOOM(const CallStackString& stack = CallStackString());
    bool ReportOOMTrigger(const OOMTriggerInfo& info);
    bool ReportOOMMemRecord(const OOMMemRecord& record, EventSubType subType);
    bool ReportCpuTensor(uint64_t addr, uint64_t size, bool isAlloc, std::string&& pyStack);
    // 窗口边界系统事件(HOST_LEAK_STAGE_START/END,挂SYSTEM基础类型):钩子经bind回调report_stage上报,
    // isStart区分开/闭窗,stageId即windowId。窗口生命周期事件是分析器唯一的窗口驱动
    // (无逐分配事件流,分析器在闭窗时经dump_*拉取冻结快照)
    bool ReportHostStage(bool isStart, uint64_t timestamp, uint64_t stageId);
    // host-leak快照统计查询:闭窗聚合完成后经svc表拉取冻结统计(get_stats,有效=窗口
    // 关闭态;开启态为尽力而为值),含全局申请/释放累计、溢出通道/开窗前free统计、
    // 采样率与截断标注(bit0块表满转溢出/bit1栈表满/bit2溢出账本满记账停止);
    // 钩子未装配/bind未就绪返回false
    bool GetHostMemStats(MsmemscopeHostmemStats& stats);
    // 钩子块表存活块全量投影(block_detail数据源):仅窗口关闭态有效(热路径冻结);
    // emit回调由分析器提供(纯C签名,经bind svc表直传),钩子未装配返回false
    bool DumpHostMemLiveBlocks(void (*emit)(void* ctx, uint64_t addr, uint64_t size, uint64_t allocTs,
                                            uint64_t stackId),
                               void* ctx);
    // 闭窗栈统计快照(leak_overview数据源):对每个栈emit一行(per-stack申请/释放/未释放
    // +生命周期三字段maxAllocTsNs/freedLifetimeSumNs/liveAgeSumNs+闭窗符号化文本,
    // stackId=0为未知桶行);仅窗口关闭态有效;钩子未装配返回false
    bool DumpHostMemStackStats(void (*emit)(void* ctx, uint64_t stackId, uint64_t allocCount, uint64_t allocBytes,
                                            uint64_t freedCount, uint64_t freedBytes, uint64_t unfreedCount,
                                            uint64_t unfreedBytes, uint64_t maxBlockSize, uint64_t maxAllocTsNs,
                                            uint64_t freedLifetimeSumNs, uint64_t liveAgeSumNs, const char* frameDesc,
                                            size_t len),
                               void* ctx);
    // 闭窗节拍快照序列(leak_overview置信度数据源):预热线程1s节拍对全局top-K栈
    // (按liveBytes降序,K=256)每拍一行(beat/liveBytes/liveCount,flags bit0=槽被驱逐);
    // 按栈分组、栈内按beat升序;仅窗口关闭态有效;钩子未装配返回false
    bool DumpHostMemUnfreedSeries(void (*emit)(void* ctx, uint64_t stackId, uint32_t beat, uint64_t liveBytes,
                                               uint32_t liveCount, uint32_t flags),
                                  void* ctx);
    // 闭窗大小排布(leak_overview数据源):存活块按大小范围分桶;仅窗口关闭态有效
    bool DumpHostMemSizeDist(void (*emit)(void* ctx, uint64_t rangeLow, uint64_t rangeHigh, uint64_t blockCount,
                                          uint64_t blockBytes),
                             void* ctx);
    // 开窗前free大小排布快照(leak_overview数据源):本窗口free未命中块表与溢出账本
    // (开窗前分配/记账被跳过)的事件按大小归桶;仅窗口关闭态有效;钩子未装配返回false
    bool DumpHostMemPreWindowDist(void (*emit)(void* ctx, uint64_t rangeLow, uint64_t rangeHigh, uint64_t blockCount,
                                               uint64_t blockBytes),
                                  void* ctx);
    // 窗口中间快照(display host_leak summary数据源):窗口开启态调用——
    // 钩子冻结记账门控→持锁聚合(per-stack未释放/大小桶/开窗前free/节拍序列)→
    // 符号化(模块快照路径)→恢复门控;不闭窗不清零,窗口数据零丢失。emit签名与
    // dump_*对应项一致(ctx统一传入),stats见MsmemscopeInterimStats(含frozenSkip
    // 冻结期跳过计数与snapshotDegraded本次快照降级位);关闭态stats清零返回false。
    // 钩子未装配返回false
    bool DumpHostMemInterimSnapshot(void (*emitStack)(void* ctx, uint64_t stackId, uint64_t allocCount,
                                                      uint64_t allocBytes, uint64_t freedCount, uint64_t freedBytes,
                                                      uint64_t unfreedCount, uint64_t unfreedBytes,
                                                      uint64_t maxBlockSize, uint64_t maxAllocTsNs,
                                                      uint64_t freedLifetimeSumNs, uint64_t liveAgeSumNs,
                                                      const char* frameDesc, size_t len),
                                    void (*emitSizeDist)(void* ctx, uint64_t rangeLow, uint64_t rangeHigh,
                                                         uint64_t blockCount, uint64_t blockBytes),
                                    void (*emitPreWindow)(void* ctx, uint64_t rangeLow, uint64_t rangeHigh,
                                                          uint64_t blockCount, uint64_t blockBytes),
                                    void (*emitSeries)(void* ctx, uint64_t stackId, uint32_t beat, uint64_t liveBytes,
                                                       uint32_t liveCount, uint32_t flags),
                                    MsmemscopeInterimStats* stats, void* ctx);
    // 测试缝:UT环境无钩子so(bind不执行,svcHostMem_==nullptr),测试注入假svc表驱动分析器;
    // 仅测试用,注入后窗口开关/快照拉取全走注入表
    void SetHostMemSvcForTest(const MsmemscopeHostmemSvc* svc);
    bool ReportTritonMessage(const char* message);
    void UpdateAnalysisType();

    // 进程退出闭窗入口(钩子so经msmemscope_hostmem_exit_close dlsym调用,契约见
    // event_report.cpp文件头注释):必须由钩子exit拦截器/main-return trampoline在
    // 真exit()之前调用——本so内任何atexit注册都在teardown期执行,均晚于可用时点。
    // 经g_hostMemReportInstance直达指针调用实例,不经Instance()的magic-static guard
    // (退出期其他线程可能仍在guard上等待,重入guard即死锁)
    static void HostMemExitHandler();

   private:
    void Init();
    explicit EventReport(MemScopeCommType type);
    ~EventReport();

    bool IsNeedSkip(int32_t devid);
    void SetStepInfo(MarkType type, std::string msg, uint64_t rangeId);

    // 退出闭窗(msmemscope_hostmem_exit_close调用):窗口仍开时置stop闩锁并关窗,
    // 让STAGE_END走完整路径(丢包差分+O4校准+报告落盘,区别于~HostLeakAnalyzer
    // 析构兜底的退化报告);有界等待钩子排空完成,超时/符号不可得留给析构兜底(降级链)
    void CloseHostMemWindowAtExit();

    // host堆钩子bind注册:dlsym解析钩子so的msmemscope_hostmem_bind,
    // 注册api回调表并保存返回的svc表;钩子so未装配时svcHostMem_为null,host功能整体不激活
    void BindHostMemHook();
    // host-leak窗口重算(交集语义):(analysis含host-leaks) && !(stop闩锁) → svc set_enabled
    void UpdateHostMemWindow();
    // host-leak独占通道短路判定:analysis含host-leaks且collectCpu关闭。成立时host堆钩子
    // 是host地址空间唯一记录者,HOST_*事件直连EventDispatcher派发,跳过hostPtrs_地址簿
    // 与EventRouter阶段一/三(MSM账本对该类事件无消费者,见实现处注释)
    bool HostEventBypassEnabled() const;

    // 查询整卡显存用量（dcmi_get_device_hbm_info，与 npu-smi 同源；内部已有
    // EventReportSuppressor 防递归上报）；查询失败返回 -1（不更新缓存，限频告警一次）。查询结果缓存到
    // MemoryStateManager（槽位与事件 device 同源：HAL 事件 flag 解析/池事件 TransDeviceId
    // 均归一为物理卡号）；HOST（DEVICE_ID_CPU）/无效（GD_INVALID_NUM）devId 不写缓存
    int64_t QueryDeviceUsed(int32_t devId);
    // 查询本进程在该设备上的显存占用（dcmi_get_npu_proc_mem_info 按 (card_id, device_id)
    // 查该卡所有进程列表后按本进程 pid 过滤，与 npu-smi Process memory 同源；内部已有
    // EventReportSuppressor 防递归上报）；失败返回 -1（不更新缓存，限频告警一次），
    // 缓存机制与 QueryDeviceUsed 一致
    int64_t QueryProcessUsed(int32_t devId);

   private:
    std::atomic<uint64_t> recordIndex_;
    std::atomic<uint64_t> kernelLaunchRecordIndex_;
    // python接口标识step和mstx标识step两种方式不允许同时存在
    std::atomic<uint64_t> pyStepId_;

    MstxStepInfo stepInfo_;
    std::mutex mutex_;

    // 已分配的HAL块地址 → 分配时device映射：FREE事件据此过滤未注册地址，
    // 并回填device（分配时语义，与MALLOC事件flag解析同源），保证FREE事件能按key(pid, device, addr)定位
    std::unordered_map<uint64_t, int32_t> halPtrs_;
    // 跨通道存活HOST地址簿(先到先得): addr→登记通道(取值为HOST_PTR_FROM_*常量,
    // 定义于event_report.cpp: 0=HAL锁页内存 1=cpu-tensor采集)。
    // alloc侧已被任一通道占用即拒绝;free侧仅登记通道可归还——防他通道误擦致
    // 占有方后续free被拒(块状态永久悬挂/假泄漏)
    std::unordered_map<uint64_t, uint8_t> hostPtrs_;
    std::atomic<bool> destroyed_{false};

    // host堆钩子svc表(bind返回,进程生命周期内不变;null=钩子未装配)与窗口stop闩锁
    // (stop()置位关窗,start()复位可重开,config变更经UpdateHostMemWindow重算)
    const MsmemscopeHostmemSvc* svcHostMem_ = nullptr;
    std::atomic<bool> hostMemStopLatch_{false};

    // 查询失败限频告警标记（首个失败日志后静默）
    std::atomic<bool> deviceUsedQueryWarned_{false};
    std::atomic<bool> processUsedQueryWarned_{false};
};

class GetDeviceInfo
{
   public:
    static GetDeviceInfo& Instance();
    bool GetDeviceId(int32_t& devId);
    bool TransDeviceId(int32_t& devId);
    // 查询设备 HBM 用量（dcmi_get_device_hbm_info，与 npu-smi 同源，单位 MB 转出）；
    // 成功返回 true 并填写 usedMb/totalMb，失败（未 init/未建表/查询失败）返回 false
    bool GetDeviceHbmInfo(int32_t devId, uint64_t& usedMb, uint64_t& totalMb);
    // 查询本进程在该设备上的显存占用，字节单位；成功返回 true 并填写 usedBytes。
    // 主路径 devdrv 直连（/dev/davinci_manager ioctl → devmm 记账列表 + H2D 设备侧查询，
    // 与 npu-smi 同源硬性一致）；devdrv 不可用（节点/权限/ioctl 失败）时降级 dcmi 接口
    // （dcmi 内核入口的 UDIS 分支可能先命中空记账返回 0，值不可信但查询本身有效）
    bool GetDeviceProcMemInfo(int32_t devId, uint64_t& usedBytes);

   private:
    // devdrv 直连查询实现：open /dev/davinci_manager + GET_DEV_RESOURCE_INFO ioctl
    // （DEVDRV_PROCESS_RESOURCE + DEVDRV_DEV_PROCESS_PID/MEM，常量与结构见 event_report.cpp）
    bool QueryDevDrvProcMemInfo(int32_t devId, uint64_t& usedBytes);
    // dcmi 接口查询实现（原 dcmi_get_npu_proc_mem_info 路径，devdrv 降级兜底）
    bool QueryDcmiProcMemInfo(int32_t devId, uint64_t& usedBytes);
    // dcmi_init 一次性初始化（失败则本进程内永久降级，不再重试）
    bool EnsureDcmiInit();
    // 一次性建表：acl 逻辑号 → (card_id, device_id)（dcmi_get_card_list + num_in_card + logic_id）
    bool BuildDeviceMap();
    // 惰性 env 解析 + 受限模式探测（一次性）：镜像 runtime GetVisibleDevices 的解析语义
    // （不 trim/纯数字 token/单逗号分隔/非法或越界 prefix break/乱序或重复整体拒绝），
    // 并经 aclrtGetDeviceCountImpl 探测 runtime 是否处于可见设备受限模式——
    // 受限才做 user→real 转换（详见 event_report.cpp 实现处注释）
    void EnsureEnvParsed();

    // env 解析延迟到首次使用时（EnsureEnvParsed），构造函数不解析
    GetDeviceInfo() = default;

   private:
    ~GetDeviceInfo() = default;

    GetDeviceInfo(const GetDeviceInfo&) = delete;
    GetDeviceInfo& operator=(const GetDeviceInfo&) = delete;
    GetDeviceInfo(GetDeviceInfo&&) = delete;
    GetDeviceInfo& operator=(GetDeviceInfo&&) = delete;

    bool setVisibleDevice = false;  // env 有效且被接受（解析成功后置位，决定是否走转换）
    std::unordered_map<int32_t, int32_t> visibleDeviceMap;

    std::once_flag envParseFlag_;
    // runtime 是否处于可见设备受限模式：受限时 GetDeviceCount 返回可见设备数而非物理数，
    // 池事件 deviceIndex/GetDeviceId 为 user 域，TransDeviceId 需做 user→real 转换；
    // 非受限（旧 runtime/芯片不支持可见设备特性/env 未设置）时事件 device 已是物理域，直传
    bool runtimeRestricted_ = false;
    // TransDeviceId 越界直传的限频告警标记（首个越界日志后静默）
    std::atomic<bool> transDevIdWarned_{false};

    std::once_flag dcmiInitFlag_;
    bool dcmiReady_ = false;  // dcmi_init 成功后才置位
    std::once_flag dcmiMapInitFlag_;
    bool dcmiMapReady_ = false;  // 设备映射建表成功后才置位
    // acl 逻辑号 → (card_id, device_id)，建表后只读
    std::unordered_map<int32_t, std::pair<int32_t, int32_t>> devIdToDcmiMap_;
};

MemOpSpace GetMemOpSpace(unsigned long long flag);

}  // namespace MemScope
#endif
