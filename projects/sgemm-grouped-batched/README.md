# Grouped Batched GEMM · A2 / A3

[返回合集](../../README.md) · [源码目录](source/) · [逐文件 SHA256](source-manifest.json)

## 工作内容

实现FP32分组批量矩阵乘法，按形状、批量和归约规模组织执行路径。验证alpha/beta、padding、特殊值、异步生命周期与独立双流行为。

## 阅读代码

- 实现入口：[blas/gemm_grouped_batched/arch22](source/blas/gemm_grouped_batched/arch22)。
- 测试入口：[test/gemm_grouped_batched/arch22](source/test/gemm_grouped_batched/arch22)。
- 构建或使用说明：[test/gemm_grouped_batched/arch22/README.md](source/test/gemm_grouped_batched/arch22/README.md)。

本目录收录实际贡献文件的当前本地快照，保留上游路径、文件头及许可证。修改过的公共文件以完整文件形式保存，其中包含上游原有内容，不表示整文件均由本人原创。

## 验证记录

v41同一份源码在910B2与A3 Ascend910_9382分别完成1973/1973功能、70/70定向精度、独立双流和200/200性能验证。Mean60与Trim50分别保留；910B2不等同于任务指定910B3实测。

以上为历史指定版本的开发侧记录。本轮仅整理源码与说明，没有重新编译或上卡测试，也未将本地快照与所有远端提交做字节等价认证。正式验收、CI和合入状态需以对应版本的官方记录为准。

## 复现方式

这里是贡献源码档案，**不是独立可构建的完整上游工程**。复现需取得完整上游仓库、对应版本的公共组件及CANN等依赖，再按项目构建说明执行。

- 本地快照所基于的Git提交：`617bb065849dc60fbc543c364dcf36f6e7660905`。
- 此提交仅标识本地上游基线，不包含未提交的贡献文件；归档内容以本目录清单逐项SHA256为准。
- 从原始PR复现时，应检出PR对应的固定提交；不要把本地快照无差别覆盖到上游最新分支。
- 本轮未提供一键覆盖或安装脚本，避免把不同版本混合成未经验证的工程。

## 上游与贡献记录

- 上游：https://gitcode.com/cann/ops-blas
- 评审入口：https://gitcode.com/cann/cann-ops-competitions/pull/2067
- 上述评审入口为设计PR，不是代码PR。

本人完成技术设计、实现、验证和后续提交整改；部分任务报名账号不同，具体归属说明见[贡献声明](../../CONTRIBUTIONS.md)。
