# Wan2.2-I2V 推理优化

在 Ascend A3 单卡上为 Wan2.2-I2V-A14B 接入 TaylorSeer、缓存卸载和异步预取。TaylorSeer 复用历史特征以跳过部分去噪计算。Offload 将缓存迁移至 Host，Prefetch 使用拷贝流提前将后续步骤需要的缓存传回 NPU。针对旧三阶配置出现的画质退化，2026-11-03 修复将默认阶数调整为一阶，并使用间隔 4、预热 3 步和末尾保留 1 步。

## 实现与测试入口

- [推理入口](source/generate.py)、[图生视频管线](source/wan/image2video.py)与 [DiT 模型](source/wan/modules/model.py)
- [TaylorSeer](source/wan/cache/taylorseer.py)与[缓存接口](source/wan/cache/cache_block.py)
- 单卡配置：[NoCache](source/config/14b_single_nocache.yaml)、[TaylorSeer](source/config/14b_single_taylorseer.yaml)、[Offload](source/config/14b_single_taylorseer_offload.yaml)、[Prefetch](source/config/14b_single_taylorseer_prefetch.yaml)
- [TaylorSeer 单元测试](source/tests/test_taylorseer.py)与 [NPU 集成测试](source/tests/test_taylorseer_integration_npu.py)

运行推理和测试需要完整的 `cann-recipes-infer` 工程和对应依赖。文件取自本地 WanTOP 提交，具体版本和逐文件 SHA256 见[源码清单](source-manifest.json)，许可证见[上游许可文件](source/LICENSE.txt)。

## 历史 A3 自验

2026-11-03 使用 BF16、40 步、61 帧，在同卡同图条件下，每档预热 1 次并正式计时 3 次。耗时为同步 Host 侧生成时间，计时区间不含模型加载和视频落盘。

| 模式 | 平均生成耗时 |
| :--- | ---: |
| NoCache | 231.014 秒 |
| TaylorSeer | 106.977 秒 |
| TaylorSeer + Offload | 115.075 秒 |
| TaylorSeer + Offload + Prefetch | 109.375 秒 |

完整组合较 NoCache 耗时下降 52.65%。开发自验还记录了四档共 20 次完整请求，以及一次在途拷贝故障后的 5 次完整恢复请求。以上为该历史版本的开发自验结果，官方验收状态以社区记录为准。

## 社区记录

[代码 PR #1073](https://gitcode.com/cann/cann-recipes-infer/pull/1073) · [设计 PR #2239](https://gitcode.com/cann/cann-ops-competitions/pull/2239)
