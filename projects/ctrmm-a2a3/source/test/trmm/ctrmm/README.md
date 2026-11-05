# CTRMM 自验证入口

CSV 位于 `arch22/ctrmm_test.csv`，覆盖任务提供的 1000 个功能参数和 200 个性能参考参数。
正式 GTest 还包含原地/独立 padding（48 组合）及设备回退布局（24 组合）两个测试。
不运行 TC_PF 参数时，共注册 1002 项；“1002 项”不是内部断言数。

精度 golden 使用私有目录中的 Netlib BLAS，固定最大绝对误差 0.01，
实虚部匹配率各至少 99%，逐元素容差为 2^-13×(1+abs(golden))；非有限值按分类和符号检查。
测试在未使用的三角区域、UNIT 对角填入 NaN，以检验实现没有引用这些元素。
`NonfiniteLayouts` 补充 192 个内部组合，覆盖左右乘、上下三角、N/T/C、单位对角、
A/B 的 Inf/NaN、实数及复数 alpha 和 leading dimension padding。
非有限值参考行为按 Fortran 复数运算核对；不同发行包即使同名 Netlib，也可能采用不同恢复规则。
复现时请记录实际加载的参考库及 SHA256，不通过放宽阈值兼容不同规则。

以下在包含 `ops-blas`、`scripts`、`third_party/netlib` 的交付工程根目录运行：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
cmake -S scripts -B build-ctrmm -DCTRMM_ORDER_CHUNK=2048
cmake --build build-ctrmm -j8
BLAS_DEVICE_ID=0 msprof \
  --application='./build-ctrmm/ctrmm_test --gtest_filter=-*TC_PF*' \
  --output=profile-functional --ai-core=off --task-time=on --runtime-api=on
```

需要 CANN 9.1.0、CMake、C++17、GTest 及 Netlib BLAS；不依赖 PyTorch 或 torch_npu。
私有 Netlib 文件包含在交付包的依赖目录，保留许可证；不要替换系统默认 BLAS。
若仅检出算子仓，使用仓库原生入口，并按交付依赖说明配置 Netlib 后核对链接的实际库。

五项主性能对应 TC_PF_1001 至 TC_PF_1005。整轮构建、冒烟、性能、全量功能可复现为：

```bash
python3 scripts/validate_integration.py fresh_unique_tag
```

tag 必须唯一，原始结果写入项目 `docs/integrated_<tag>`。
设备固定逻辑 0，其他设备使用前需调整驱动脚本的 BLAS_DEVICE_ID 设置。
性能流程预热 10 次、采样 60 次、按完整调用 kernel 总和去掉最大 10 次后平均。
设备首尾跨度沿用同一组 50 个样本；所有风险检查、打包和回退入口均计入。
不含 Host 输入生成、首次分配或 H2D/D2H，不能称为应用端到端时间。

同时检查 `full.json`、GTest 的最终汇总和 `performance.json`。
msprof 的退出码可能为 0，而内部 GTest 仍失败，因此只看进程退出码不足以宣布通过。
任务规定的五项之外，200 个参考性能参数尚需按需要单独测量；不得冒充已完成全量性能自验。
