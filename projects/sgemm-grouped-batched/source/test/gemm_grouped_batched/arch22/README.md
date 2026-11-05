# A2/A3 grouped SGEMM 测试

实现包括Cube批量计算、顺序AIV大规约、alpha/beta合并及特殊数值回退。
当前验证范围和源码版本记录于工程根目录docs/project_status.md，不以旧版结果代替新版全量验证。
原始任务CSV含1000条精度和200条性能用例，保留原始字节；另有965条正态分布扩展用例。

在CANN9.1.0环境从ops-blas根目录构建（soc参数须对应实际设备）：

```bash
bash build.sh --ops=gemm_grouped_batched --soc=ascend910b3
```

参考后端需显式配置REFBLAS_INCLUDE_DIR、REFBLAS_LIB和REFLAPACK_LIB，检查实际加载路径为Netlib。
测试直接接收Host指针数组，其元素为Device矩阵地址，支持列主序N/T与padding。
负向用例直接调用API，覆盖非法枚举、数组空指针及单个矩阵空指针。

```bash
# 原任务精度及专项测试
BLAS_DEVICE_ID=0 ./build/test/gemm_grouped_batched/gemm_grouped_batched_test --gtest_filter='*-*TC_PF*:*TC_NM*'
# 正态分布扩展测试
BLAS_DEVICE_ID=0 ./build/test/gemm_grouped_batched/gemm_grouped_batched_test --gtest_filter='*TC_NM*'
```

精度对有效m×n输出检查rtol=atol=2^-13、匹配率>=0.99；最大绝对误差上限为
max(0.01,32*ULP(golden))，golden取最大绝对误差处，遵循官方混合容差标准。
NaN/Inf需一致，padding与no-op存储逐字节保持。参考后端为Netlib BLAS3.10.0。
正态输入不截断，均值在[-5,5]、标准差在[0.1,2]取样，各用例固定种子。
专项测试共8个测试体，覆盖异步重复调用、Host元数据修改后的捕获语义、非有限系数、
小矩阵65批次尾包和padding、Inf/NaN乘正负零、扫描半块/整块/尾块，以及共享缓存行/重叠输出、相邻分组输入依赖、64矩阵扫描标志复用。
向量与无需合并的小Cube批次上限96，其他Cube/混合扫描上限64；
性能输入TC_PF_1011/1014另做256矩阵golden检查，覆盖向量/小Cube的96批次尾包。
完整功能回归包含1000条原任务精度、965条正态扩展及8项专项，共1973项；
200条性能输入的golden检查与msprof采样需另行执行，不计入这1973项。

修改失败项后，先复测对应失败项，通过后执行相关边界，最终版本再完整回归：

```bash
BLAS_DEVICE_ID=0 ./build/test/gemm_grouped_batched/gemm_grouped_batched_test --gtest_filter='*-*TC_PF*'
```

性能使用独立msprof设备计时，不以GTest运行时间作为设备耗时。
默认预热10、采样60、去掉最大的10次后取50次平均，保留全部原始数据。
任务正式性能要求指定910B3；其他设备结果分别标注。
