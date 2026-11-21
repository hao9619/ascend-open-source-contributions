<p align="center">
  <img src="assets/cgemm3m-cover.webp" width="100%" alt="两幅矩阵格网经三条乘积路径，合成为一幅复数矩阵的概念插画">
</p>

<h1 align="center">aclblasCgemm3m</h1>
<p align="center"><strong>在昇腾 Atlas 950PR 上，用三次实数矩阵乘法计算复数矩阵乘积。</strong></p>

<p align="center">
  <a href="https://gitcode.com/cann/ops-blas/pull/735"><img alt="上游 PR 735 已合入" src="https://img.shields.io/badge/PR_%23735-Merged-7654A8?style=flat-square"></a>
  <img alt="目标硬件 Atlas 950PR" src="https://img.shields.io/badge/Atlas-950PR-117F82?style=flat-square">
  <img alt="开发套件 CANN 9.1.0" src="https://img.shields.io/badge/CANN-9.1.0-344B72?style=flat-square">
  <img alt="数据类型 complex64" src="https://img.shields.io/badge/BLAS-complex64-A25031?style=flat-square">
</p>

<p align="center">
  <a href="https://gitcode.com/cann/ops-blas/pull/735"><b>上游 PR ↗</b></a> ·
  <a href="https://gitcode.com/cann/cann-ops-competitions/pull/2506">设计算法与接口 ↗</a> ·
  <a href="https://gitcode.com/gzh1630/ops-blas/blob/a2626f23ec202b1f8bdd3d637cae8dbf6b7decb8/blas/gemm/arch35/cgemm3m_host.cpp">阅读实现 ↗</a>
</p>

2026 年 11 月 21 日，华为昇腾 CANN 社区将这项贡献合入了 `cann/ops-blas`。新增的 BLAS 接口计算 **C ← α·op(A)·op(B) + β·C**，支持列主序 `complex64` 矩阵。A 和 B 均可独立选择原矩阵、转置或共轭转置。

## 为什么三次乘法就够了？

先把操作后的矩阵写成 A = Ar + iAi、B = Br + iBi。按定义展开 AB，实部和虚部各需要两次实矩阵乘法。我们在非融合路径中复用前两个乘积，用一次额外的乘法同时取回两个交叉项。

| 第一次乘法 | 第二次乘法 | 第三次乘法 |
| :---: | :---: | :---: |
| P₀ = Ar × Br | P₁ = Ai × Bi | P₂ = (Ar + Ai) × (Br + Bi) |

此时 **Re(AB) = P₀ − P₁**，**Im(AB) = P₂ − P₀ − P₁**。最后再应用复数标量 α 和 β。少做一次矩阵乘法的代价，是多做输入加法、保存中间结果，并承担加减顺序改变带来的浮点舍入误差。这三点决定了内核的设计。

## 从公式到 Ascend C 内核

- **让中间结果尽量留在片上。** 我们复用已有 Cgemm 的计算组件。对适合融合的形状，按面板准备输入，在片上执行三乘积；融合分解采用 P₀ = (Ar + Ai) × Br、P₁ = Ar × (Bi − Br)、P₂ = Ai × (Br + Bi)，以 P₀ − P₂ 和 P₀ + P₁ 分别得到实部和虚部，减少全局工作区的读写。
- **让矩阵形状决定怎么分工。** 输出能切出足够多的块时，各核心可以并行计算不同输出块。输出块少而 K 很长时，非融合路径会联合选择输出块大小和 K 方向的 1～4 个分区，在核心利用率和部分和的搬运成本之间取舍。
- **单独处理舍入敏感的输出。** 两种分解在实数算术下等价，FP32 的累加顺序却会改变低位。针对特定的大方阵路径，我们分段累加并记录部分和，对需要纠正的位置按固定顺序重算，再用 Netlib 参考结果检查精度，用同输入双跑检查确定性。

## 真机验证

下表来自 2026 年 11 月 9 日的 **Atlas 950PR** 自测。实测源码为 `0f8c125` 加两处测试框架 SDK 兼容补丁，环境为 CANN 9.1.0，参考库为 Netlib BLAS/CBLAS 3.10.0。合入时的 PR 源分支提交为 `a2626f23`。[实测证据附件](https://gitcode.com/cann/ops-blas/pull/735?did=42c2afae0d665e6e0ce90e42d345bb20eaf574a6#tid-199509752)

| 自测项目 | 结果 | 测了什么 |
| :--- | :---: | :--- |
| Cgemm3m 全量测试 | **1208 / 1208** | 检查参考精度，适用用例用相同输入运行两次，逐字节比对输出 |
| 已有 Cgemm 回归 | **51 / 51** | 在同一构建产物上检查共享实现影响到的既有接口 |
| 正式性能用例 | **200 / 200** | GPU 基线耗时 ÷ NPU 耗时，单项最低 **0.41465**，平均 **0.63542** |
| 内存快速检查 | **11 / 12 组通过** | ED 组出现 5 项数值断言失败和一条 8 字节 GM 写告警，本轮未定位根因 |

性能用例各预热 5 次，再用 msprof 记录 10 个有效样本；先对一次调用涉及的各 kernel 耗时求和，再对 10 次取算术平均。任务门槛是单项倍率至少 0.4、全部用例平均倍率至少 0.5。

## 致谢

谢谢把 GitCode 账号 **gzh1630** 借给我提交这次贡献的朋友。很高兴能和你分享代码合入的这一刻。

感谢华为昇腾 CANN 社区的维护者和评审同学。从追问两条路径各自用了哪组三乘积公式，到逐项核对真机日志、帮助排查 CI，你们的反馈让实现更容易读懂，也让后来的开发者能沿着记录复核结果。
