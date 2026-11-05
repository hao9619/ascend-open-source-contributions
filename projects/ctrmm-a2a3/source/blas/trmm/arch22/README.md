# aclblasCtrmm

单精度复数三角矩阵乘。所有矩阵采用列主序，输入输出元素类型为 `aclblasComplex`（complex64）。

## 产品支持

| 产品 | 实现支持 | 本次实测 |
|---|---|---|
| Atlas A2 系列，含 Atlas 800I A2 | 支持，arch22 | Ascend910B2 |
| Atlas A3 系列，含 Atlas 800I A3 | 支持，arch22 | Ascend910_9382 |

使用 CANN 9.1.0。任务指定的 910B3 性能结果不能由上述型号推算；具体版本证据见交付自测报告。

## 接口

```cpp
aclblasStatus_t aclblasCtrmm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, aclblasComplex* C, int ldc);
```

| 参数 | 内存位置与语义 |
|---|---|
| handle | Host，已创建的 BLAS 句柄；使用其绑定的 stream |
| side | LEFT：C=alpha×op(A)×B；RIGHT：C=alpha×B×op(A) |
| uplo | UPPER 或 LOWER，仅引用对应三角区域 |
| trans | N、T、C，分别表示原矩阵、转置、共轭转置 |
| diag | UNIT 时对角视为 1 且不读取；NON_UNIT 时读取对角 |
| m、n | Host 非负整数，B/C 的行列数 |
| alpha | Host complex64 标量指针 |
| A、lda | Device 只读三角矩阵；阶数 d=LEFT?m:n，lda≥d |
| B、ldb | Device 只读 m×n 矩阵，ldb≥m |
| C、ldc | Device 输出 m×n 矩阵，ldc≥m；padding 保持原值 |

非空矩阵需有效 alpha/C；非零 alpha 另需有效 A/B。支持 B=C 且 ldb=ldc；
拒绝 C 与 A 重叠以及 B/C 部分重叠。矩阵缓冲区和句柄必须存活至 stream 上计算结束。
调用异步返回，读取 C 前由调用方同步 stream；算子不释放调用方内存，不进行 Host 输出回读。
同一 handle 的工作区随 stream 使用；不保证并发复用同一 handle 的多个独立 stream 安全。

返回值：空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；非法枚举、维度、步长、
必要空指针或不允许的重叠返回 `ACLBLAS_STATUS_INVALID_VALUE`。
工作区不足返回 `ACLBLAS_STATUS_ALLOC_FAILED`，设备资源查询异常返回内部错误。
m=0 或 n=0 在枚举/非负维度检查后直接成功；alpha=0 在非空输出合法时置零，不读取 A/B。

## 实现

有效三角输入解交织后进行三次实数 Cube 矩阵乘，组合为复数输出；小矩阵融合打包与求和。
满足参数条件的极小矩阵直接用单内核计算；16/32/64阶使用紧凑行块、四列复用和实际规约长度。
右侧窄矩阵将转置交给Cube，输出合并与异常回退共用一个内核；默认向量块为2304个元素。
128行且列数不超过128、临时矩阵与输出均连续时，将输出展开为连续区间，按四个复数对齐分配给向量核。
满足左乘分流条件的128阶使用独立编译的紧凑打包与融合输出内核，避免改变其他尺寸的设备实现。
普通打包风险归约仅扫描实际写入的向量范围，风险阈值和异常处理保持不变。
输出 leading dimension 非四倍数时按四列分组，每组起点保持 32 字节对齐并分配给同一向量核，
避免相邻核写入边界冲突，同时支持多核并行输出。
特定长规约分支共享 Cube 前缀，对筛选出的输出尾段做有序修正，计入全部实际执行 kernel。
打包同时检查有效输入范围；较大幅值或非有限值在设备上转入有序补偿计算。
非单位 alpha 和原地计算直接使用严格路径。计算分流只依赖 API 参数及运行时输入，
不读取 golden、测试名称、随机种子或预先记录的坐标。
非有限值乘法保留 Fortran 运算产生的 NaN/Inf，不额外执行 C 复数乘法的 Inf 恢复。
参考库的编译方式会影响特殊值语义，复现时需要记录实际库文件及其构建信息。

数值策略为经过自测的工程实现，不声明对任意 complex64 输入都有数学误差上界证明。
v36在A3的官方1000项精度和同一二进制200项配套性能通过；性能用例附带的固定0.01精度检查
仍有15项不通过，单列记录。定向阶段64阶曾超限，完整批次通过，两份原始结果均保留。
同版A2（910B2）原生工程构建、官方1000项精度、正文五项及完整200项性能均已通过。
910B3指定型号与社区评审尚未完成，不能据此宣称社区全部验收通过。
不同源码版本的已测范围与失败清单见交付报告。
工作区容量与 UB 会在 Host 检查，超出可用资源时返回失败，不进行越界分配。

## 构建与测试

仓库构建入口：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash build.sh --ops=ctrmm --soc=ascend910b3 --run
```

A3 使用 `--soc=ascend910_9382`。该入口的实测状态以交付报告为准。
严格 Netlib 自测使用交付包的独立 CMake 入口，步骤见 `test/trmm/ctrmm/README.md`；
不要将其他 BLAS 实现产生的参考结果混称为 Netlib 基线。
