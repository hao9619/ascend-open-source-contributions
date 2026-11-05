# GemmGroupedBatched算子

## 算子概述

分组批量矩阵乘法（GEMM Grouped Batched），对多个分组内各批次矩阵独立执行 GEMM 运算，是 BLAS Level 3 核心算子之一。每个分组可拥有不同的 (m, n, k, transa, transb, alpha, beta, lda, ldb, ldc) 参数以及 groupSize 个批次。所有矩阵采用列主序（Column-Major）存储。

数学表达式：

```
C[i] = alpha[g] * op(A[i]) * op(B[i]) + beta[g] * C[i]
```

其中 `i` 属于分组 `g`，`op(A)` / `op(B)` 由 transa / transb 决定：`N` 为不转置，`T` 为转置。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgemmGroupedBatched | 单精度浮点分组批量矩阵乘法 |

## 算子执行接口

### aclblasSgemmGroupedBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品（含 Atlas 800I A3）：支持（arch22）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品（含 Atlas 800I A2）：支持（arch22）

v41已在Ascend910B2（A2）与Ascend910_9382（A3）、CANN9.1.0分别完成验证，
两端均通过1973项完整功能回归。验证范围见arch22测试说明及项目根目录docs/。

#### 函数原型

```cpp
aclblasStatus_t aclblasSgemmGroupedBatched(
    aclblasHandle_t handle, int groupCount,
    const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray,
    const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray, const float* const* Aarray, const int* ldaArray,
    const float* const* Barray, const int* ldbArray, const float* betaArray,
    float* const* Carray, const int* ldcArray, const int* groupSizeArray);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| groupCount | 输入 | int | 分组数量，Host 内存 |
| transaArray | 输入 | const aclblasOperation_t* | 每组的 A 矩阵操作类型数组，当前支持 N/T，Host 内存 |
| transbArray | 输入 | const aclblasOperation_t* | 每组的 B 矩阵操作类型数组，当前支持 N/T，Host 内存 |
| mArray | 输入 | const int* | 每组的输出行数数组，Host 内存 |
| nArray | 输入 | const int* | 每组的输出列数数组，Host 内存 |
| kArray | 输入 | const int* | 每组的内积维度数组，Host 内存 |
| alphaArray | 输入 | const float* | 每组的标量乘数数组（float），Host 内存 |
| Aarray | 输入 | const float* const* | A 矩阵指针数组（Host 侧数组，元素为 Device 矩阵地址），长度 = sum(groupSize[g])，Host 内存 |
| ldaArray | 输入 | const int* | 每组的 A 矩阵 leading dimension 数组，Host 内存 |
| Barray | 输入 | const float* const* | B 矩阵指针数组（Host 侧数组，元素为 Device 矩阵地址），长度 = sum(groupSize[g])，Host 内存 |
| ldbArray | 输入 | const int* | 每组的 B 矩阵 leading dimension 数组，Host 内存 |
| betaArray | 输入 | const float* | 每组的标量乘数数组（类型与 alphaArray 一致），Host 内存 |
| Carray | 输入/输出 | float* const* | C 矩阵指针数组（Host 侧数组，元素为 Device 矩阵地址），长度 = sum(groupSize[g])，Host 内存 |
| ldcArray | 输入 | const int* | 每组的 C 矩阵 leading dimension 数组，Host 内存 |
| groupSizeArray | 输入 | const int* | 每组的批次数量数组，Host 内存 |

#### 约束说明

- groupCount >= 0
- m, n, k >= 0（每组）
- groupSize >= 0（每组）
- lda >= max(1, transa=N 时为 m, transa=T 时为 k)
- ldb >= max(1, transb=N 时为 k, transb=T 时为 n)
- ldc >= max(1, m)
- Aarray/Barray/Carray 为 Host 侧指针数组，元素指向 Device 上的矩阵数据；不可传入 Device 侧指针数组
- alpha=0 时跳过矩阵乘法，仅执行 C = beta * C
- k=0 时等效为 C = beta * C
- transa/transb 各支持 N/T，组合为 NN / NT / TN / TT 四种

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

// 初始化 ACL 和 handle
aclInit(nullptr);
aclrtSetDevice(0);
aclblasHandle_t handle = nullptr;
aclblasCreate(&handle);

// 定义 2 个分组
int groupCount = 2;

// 分组 0: 4×4 GEMM, NN, 3 个 batch
// 分组 1: 8×8 GEMM, TN, 2 个 batch
aclblasOperation_t transaArray[] = {ACLBLAS_OP_N, ACLBLAS_OP_T};
aclblasOperation_t transbArray[] = {ACLBLAS_OP_N, ACLBLAS_OP_N};
int mArray[]      = {4, 8};
int nArray[]      = {4, 8};
int kArray[]      = {4, 4};
float alphaArray[] = {1.0f, 1.0f};
float betaArray[]  = {0.0f, 1.0f};
int ldaArray[]     = {4, 4};   // 分组0: trans=N, lda>=m=4; 分组1: trans=T, lda>=k=4
int ldbArray[]     = {4, 4};
int ldcArray[]     = {4, 8};
int groupSizeArray[] = {3, 2};

// 以下数组需要先填入已分配并初始化的 Device 地址再调用接口。
// 完整分配、上传、同步读回流程见 test/gemm_grouped_batched/arch22/gemm_grouped_batched_test.cpp。
// A/B/C 指针数组（共 5 个 batch，Host 侧数组），元素指向 Device 上的列主序矩阵
const float* Aarray[5];
const float* Barray[5];
float*       Carray[5];

aclblasStatus_t ret = aclblasSgemmGroupedBatched(
    handle, groupCount,
    transaArray, transbArray, mArray, nArray, kArray,
    alphaArray, Aarray, ldaArray,
    Barray, ldbArray, betaArray,
    Carray, ldcArray, groupSizeArray);

// 清理
aclblasDestroy(handle);
aclrtResetDevice(0);
aclFinalize();
```


## A2/A3 实现与验证

- 逐组校验全部参数后下发；非法转置在 arch22 返回 `ACLBLAS_STATUS_INVALID_VALUE`，arch35 保留其既有行为。
- Cube 路径关闭 HF32，常规128×256×64、小矩阵独立入口128×128×64基础块；普通批次最多64个矩阵；参数与Device地址按值捕获。
- m/n/k均不超过64且无需alpha/beta合并时，小Cube批次最多96个矩阵，采用独立扫描修复。
- 普通 alpha/beta 通过 handle workspace 的临时输出和AIV合并处理；长K普通系数及瘦矩阵采用有序AIV累加。
- 小矩阵标量回退最多64个矩阵批量下发；NN向量路径最多96个，各矩阵输出缓存行须互不重叠。
- 其中NN、alpha=1、beta=0、0<k<=64、m/n<=16使用UB向量外积，先检测极值并按需标量回退。
- 满足条件的Cube路径融合输入扫描和数值修复；最多8组独立小NN矩阵共享混合核并行执行。
  跨组输出与所有输入/输出须无缓存行重叠；全核同步使用批调度，避免多流等待死锁。
- 检测到 NaN/Inf/极大数时采用数值回退；扫描And使用16位lane计数覆盖完整FP32位模式。
- 乘积保留Inf/NaN与正负零的传播；alpha=0或k=0不读取A/B，beta=0不读取初始C。
- C与A/B不得重叠；同handle跨stream并发共享workspace未在当前实现中提供保证。
- 构建：`bash build.sh --ops=gemm_grouped_batched --soc=ascend910b2`。
  A3使用实际SoC，如`ascend910_9382`。版本与硬件需按实际环境核对。
- 测试目录：`test/gemm_grouped_batched/arch22/`；原任务1200条CSV另配965条正常分布用例。
- v41版本A2/A3分别通过1973项完整功能及独立双流检查；各70项定向精度含16条性能输入golden。
- A2/A3均先完成固定65项，再完成全量200项；v41在两端均达参考门限，原始计时分别复算通过。
- 当前源码A2/A3自测已完成；指定910B3正式验收仍待补，其他设备自测不作为正式验收结论。
