<p align="center">
  <img src="assets/cgemm3m-cover.webp" width="100%" alt="两幅复数矩阵格网，经三条实数乘积路径合成为输出矩阵">
</p>

<h1 align="center">为华为昇腾实现复数矩阵乘法</h1>
<p align="center"><strong>aclblasCgemm3m</strong> · 从三乘法公式到 Atlas 950PR 上的 Ascend C 内核</p>
<p align="center">
  <a href="https://gitcode.com/cann/ops-blas/pull/735"><img alt="PR 735 已合入" src="https://img.shields.io/badge/PR_%23735-Merged-7654A8?style=flat-square"></a>
  <img alt="目标硬件 Atlas 950PR" src="https://img.shields.io/badge/Atlas-950PR-117F82?style=flat-square">
  <img alt="工具链 CANN 9.1.0" src="https://img.shields.io/badge/CANN-9.1.0-344B72?style=flat-square">
  <img alt="数据类型 complex64" src="https://img.shields.io/badge/BLAS-complex64-A25031?style=flat-square">
</p>
<p align="center">
  <a href="ops-blas/blas/gemm/arch35/cgemm3m_host.cpp"><b>读算子源码</b></a> ·
  <a href="ops-blas/test/gemm/cgemm3m">看配套测试</a> ·
  <a href="https://gitcode.com/cann/cann-ops-competitions/pull/2506">设计文档</a> ·
  <a href="https://gitcode.com/cann/ops-blas/pull/735">上游合入记录 ↗</a>
</p>

## 项目背景 · 华为昇腾与 CANN

<a href="https://github.com/Ascend"><img src="assets/ascend-org-avatar.png" align="right" width="64" alt="Huawei Ascend 官方 GitHub 组织标识"></a>

昇腾是华为的 AI 计算平台。CANN 是面向昇腾硬件的异构计算软件栈，其中的 **ops-blas** 提供矩阵乘法等基础线性代数运算，应用可以直接调用这些接口。我们参与的是 CANN 社区的一项算子开发任务，成果于 **2026 年 11 月 21 日**合入 `cann/ops-blas`。[CANN 项目介绍](https://www.hiascend.com/cann) · [ops-blas 官方仓库](https://gitcode.com/cann/ops-blas)

## 社区任务 · 让 Atlas 950PR 支持 Cgemm3m

给定两个复数矩阵 A、B，以及原有的输出矩阵 C，调用者希望用一次 BLAS 调用完成 **C ← α·op(A)·op(B) + β·C**。A 和 B 各自可以保持原样、转置或共轭转置。接口需要支持全部九种组合，并正确处理列主序存储与前导维。

名字里的 **3m** 点出了实现约束。普通复数乘法展开后有四项实数乘积，这个算子要用**三次实数矩阵乘法**算出同一个复数矩阵乘积。交付时还要在真机上对照参考库检查精度、验证同输入双跑确定性，并用设备计时与任务给定的 GPU 基线比较性能。[昇腾社区任务](https://www.hiascend.com/activities/task-center/details/b6f3238e73084a0ebba5bbfc1312cdf8?menu=tasks) · [接口声明](ops-blas/include/cann_ops_blas.h)

## 开发思路 · 从三次乘法到设备内核

**先把第四次乘法省下来。** 把操作后的输入写成 A = Ar + iAi、B = Br + iBi。直接展开时，虚部需要分别计算 Ar × Bi 和 Ai × Br。如果先把实部和虚部相加再相乘，就能一次取回这两个交叉项。非融合路径用到的三个实矩阵乘积如下。

| P₀ | P₁ | P₂ |
| :---: | :---: | :---: |
| Ar × Br | Ai × Bi | (Ar + Ai) × (Br + Bi) |

由此得到 **Re(AB) = P₀ − P₁** 和 **Im(AB) = P₂ − P₀ − P₁**，再应用 α 和 β。省下的乘法换来了额外的加减和中间结果，实际速度取决于这些数据在哪里算、搬运多少次。

**再按形状分配工作。** 我们复用已有 Cgemm 的计算内核。在适合融合的形状上，输入按面板送入片上缓冲，三个乘积采用另一组等价分解，在片上合并结果。非融合路径则联合选择输出块大小和 K 方向的 1～4 个分区，以便在输出块较少、点积很长时仍能让多个核心分担计算。[调度实现](ops-blas/blas/gemm/arch35/cgemm_host.cpp) · [两条路径的完整公式](ops-blas/blas/gemm/README.md)

**最后处理 FP32 舍入。** 代数等价的表达式，在长点积中可能得到不同的浮点低位。对于满足条件的 2048～4096 阶连续方阵，我们分段累加，保留部分和，再筛出对舍入敏感的位置，按固定顺序重算。这样可以把额外的计算集中在需要纠正的输出上。[舍入纠正内核](ops-blas/blas/gemm/arch35/cgemm3m_rounding_kernel.cpp)

## 真机验证

以下结果来自 **2026 年 11 月 9 日**同一构建产物的自测。设备为 Atlas 950PR，工具链为 CANN 9.1.0，参考库为 Netlib BLAS/CBLAS 3.10.0。实测版本为 `0f8c125` 加两处测试框架 SDK 兼容补丁，本仓收录的源码对应合入时的 PR 源提交 `a2626f23`。[实测记录与原始附件](https://gitcode.com/cann/ops-blas/pull/735?did=42c2afae0d665e6e0ce90e42d345bb20eaf574a6#tid-199509752)

| 检查项 | 结果 | 判据或观察 |
| :--- | :---: | :--- |
| Cgemm3m 全量精度 | **1208 / 1208** | 对照参考结果，适用用例以相同输入双跑，逐字节比较输出 |
| 原 Cgemm 接口回归 | **51 / 51** | 验证本次改动涉及的共享计算路径 |
| 正式性能 | **200 / 200** | GPU 基线耗时 ÷ NPU 耗时，单项最低 **0.41465**，平均 **0.63542** |
| 内存快速检查 | **11 / 12 组通过** | ED 组有 5 项数值断言失败和 1 条 8 字节 GM 写告警，本轮未定位根因 |

性能用例各预热 5 次，再用 msprof 采集 10 次。一次调用涉及多个 kernel 时，先将这些 kernel 的设备耗时相加，再对 10 个样本取算术平均。任务要求每个用例的倍率至少为 0.4，全部用例的平均倍率至少为 0.5。

## 源码与复现

本仓的 `ops-blas/` 保留了 PR 的全部 26 个变更文件、两个相关的共享内核头文件及上游许可证，目录结构与上游一致。阅读时可以从接口入口跟进调度和内核；实际编译需要完整的 ops-blas 工程和对应的 CANN 开发环境。

| 想看什么 | 从这里开始 |
| :--- | :--- |
| 公共接口如何进入共享计算路径 | [Cgemm3m 入口](ops-blas/blas/gemm/arch35/cgemm3m_host.cpp) → [共享调度](ops-blas/blas/gemm/arch35/cgemm_host.cpp) |
| 输入拆分、乘积计算和结果合并 | [输入预处理](ops-blas/blas/gemm/arch35/cgemm_kernel.cpp) → [乘积计算](ops-blas/blas/gemm/arch35/cgemm_compute_kernel.cpp) → [输出合并](ops-blas/blas/gemm/arch35/cgemm_epilogue_kernel.cpp) |
| 怎样构建、运行用例、比较结果 | [完整固定版本](https://gitcode.com/gzh1630/ops-blas/tree/a2626f23ec202b1f8bdd3d637cae8dbf6b7decb8) · [构建与测试说明](ops-blas/test/gemm/cgemm3m/README.md) · [测试用例](ops-blas/test/gemm/cgemm3m/arch35/cgemm3m_test.csv) |

上游源码和测试适用随附的 CANN Open Software License Agreement 2.0。[上游许可证](ops-blas/LICENSE) · [贡献范围与来源](CONTRIBUTIONS.md)

## 致谢

谢谢把 GitCode 账号 **gzh1630** 借给我提交这次贡献的朋友。很高兴能和你分享代码合入的这一刻。

感谢华为昇腾 CANN 社区的维护者和评审同学。你们追问了两条计算路径各自使用的公式，逐项核对了真机记录，也在 CI 排查中提供了帮助。这些具体的反馈，让今天读到这份代码的人既能看懂算法，也能找到支撑结论的测试证据。
