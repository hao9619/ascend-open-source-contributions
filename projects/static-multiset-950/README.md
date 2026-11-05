# StaticMultiset · Atlas 950

[返回合集](../../README.md) · [源码目录](source/) · [逐文件 SHA256](source-manifest.json)

## 工作内容

实现支持重复键的静态多重集合，覆盖插入、条件操作、查询、计数、检索和设备侧访问。重点处理容量约束、重复输入、批次边界与资源生命周期。

## 阅读代码

- 实现入口：[include/static_multiset.h](source/include/static_multiset.h)。
- 测试入口：[tests/static_multiset](source/tests/static_multiset)。
- 构建或使用说明：[scripts/build_static_multiset.sh](source/scripts/build_static_multiset.sh)。

本目录收录实际贡献文件的当前本地快照，保留上游路径、文件头及许可证。修改过的公共文件以完整文件形式保存，其中包含上游原有内容，不表示整文件均由本人原创。

## 验证记录

2026-09-20 CI 整改版：15个功能程序、60个注册测试、1,706,450条断言通过；44项性能达到配套门槛。性能沿用原测试的Host wall-clock口径。后续本地修订不自动继承该版全量结论。

以上为历史指定版本的开发侧记录。本轮仅整理源码与说明，没有重新编译或上卡测试，也未将本地快照与所有远端提交做字节等价认证。正式验收、CI和合入状态需以对应版本的官方记录为准。

## 复现方式

这里是贡献源码档案，**不是独立可构建的完整上游工程**。复现需取得完整上游仓库、对应版本的公共组件及CANN等依赖，再按项目构建说明执行。

- 本地快照所基于的Git提交：`42eb075b91f2ef5f71a25f0b6a9144f414390ba6`。
- 此提交仅标识本地上游基线，不包含未提交的贡献文件；归档内容以本目录清单逐项SHA256为准。
- 从原始PR复现时，应检出PR对应的固定提交；不要把本地快照无差别覆盖到上游最新分支。
- 本轮未提供一键覆盖或安装脚本，避免把不同版本混合成未经验证的工程。

## 上游与贡献记录

- 上游：https://gitcode.com/cann/ops-collections
- 评审入口：https://gitcode.com/cann/ops-collections/pull/50

本人完成技术设计、实现、验证和后续提交整改；部分任务报名账号不同，具体归属说明见[贡献声明](../../CONTRIBUTIONS.md)。
