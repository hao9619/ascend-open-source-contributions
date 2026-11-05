# 950PR FP32 LU 与求逆验证

在 ops-solver 根目录执行下列 Linux 命令。测试调用公开 Device API，CPU 仅生成金标。
准备 CANN 9.1.0、950PR、CMake≥3.16、GCC、Python 3.11；确认目标卡空闲并设置项目独占输出目录。

## 构建

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
cmake -S . -B build-lu950 -DSOC_VERSION=ascend950 -DBUILD_TEST=ON '-DTEST_NAMES=sgetrf;sgetri'
cmake --build build-lu950 -j2
python3 -m venv .venv-lu950
.venv-lu950/bin/pip install numpy==2.3.5 scipy==1.16.3
g++ -std=c++17 -O2 -shared -fPIC test/lu_real/reference_bridge.cpp -o build-lu950/libreference_bridge.so
```

SciPy 的 wheel 必须带 scipy.libs/*openblas*.so，导出 scipy_sgetrf_ 和 scipy_sgetri_。
验证器会记录实际加载库路径、版本、符号和 SHA256。参考桥接库不链接入算子。

## 功能与多流

```bash
.venv-lu950/bin/python -I test/lu_real/validate_semantics.py \
  --library build-lu950/libops_solver.so --multi-repeat 100 --output semantics.json
build-lu950/test/sgetrf/sgetrf_subnormal_pivot_test
```

输出文件使用新名称；验证器拒绝覆盖已有 JSONL。
文件型 sgetrf_test/sgetri_test 还需各自 README 中的输入数据；完整任务验证入口可直接生成数据。

## 全部 712 项精度与性能

```bash
.venv-lu950/bin/python -I test/lu_real/run_acceptance.py \
  --library build-lu950/libops_solver.so \
  --reference-bridge build-lu950/libreference_bridge.so \
  --device 0 --output results-lu950
```

| 接口 | 参数用例数 | 检查范围 |
|---|---:|---|
| Sgetrf | 148 | 每个实际 m/n/lda/pivot，精度、info、padding、重复确定性 |
| Sgetri | 200 | 已有 FP32 LU 的求逆、主元保持、精度与确定性 |
| SgetriBatched | 149 | 全部 8711779 个矩阵，输入与主元保持、输出 padding |
| SmatinvBatched | 215 | 全部 194539114 个原始矩阵，每个矩阵检查 info 与精度 |

实际 batch 全量验证，不做采样替代。输出可很大，请预留磁盘和主机内存。
默认每组16用例串行执行、每例保留独立 stream ID，逐次同步。
发现精度或性能失败后保留证据并停止，以便先独立修复。

精度使用 FP32 LAPACK 参考、FP64 残差评估，epsilon=2^-24：
LU ratio≤max(5*ratio_cpu,0.3)，已有 LU 求逆≤max(5*ratio_cpu,0.05)，原始批量求逆≤max(5*ratio_cpu,0.5)。
原始矩阵按索引交替均匀[-5,5]与正态(-0.7,1.2)，对角+=n，种子20261016+索引。
这些主用例不替代额外的无对角增强、奇异、非法参数和布局自构造验证。

入口调用profile_with_summary.py，使用CANN9.1的msprof --type=db保留完整采集数据，
再调用官方msprof.py export summary导出完整CSV，减少大型JSON时间线导出的空间与耗时。
原始采集数据和CSV均保留；所有kernel仍参与计时，不删除样本。

msprof 完整 API 内所有 lu950 kernel 时长求和；预热10次、采样60次。
Mean60 对照任务 GPU avg_ms*1000/0.35；Trim50 去掉最大10次后取均值。
保存完整60次原始数据。Host 时间、搬运、kernel 间隙不计入 kernel sum。
stream ID 用于关联不同用例；不使用 Host 时钟差计算设备耗时。

## 独立复现一个失败项

```bash
.venv-lu950/bin/python -I test/lu_real/run_acceptance.py \
  --library build-lu950/libops_solver.so --kinds sgetrf --indices 9 \
  --device 0 --output result-getrf9
```

summary.json 的每组及每例字段给出精度/性能和原始路径。
退出码：0=选中项全部通过；1=发现失败；3=运行完成但存在缺失GPU基线；其他非零=运行/证据错误。
SmatinvBatched 索引6和20的原始 GPU perf 为空；可测 NPU 时间，不能判为性能通过。
所有这些是开发者自验结果；官方 CI、设计评审和验收状态单独记录。

## 补充数据与中阶批量边界

```bash
.venv-lu950/bin/python -I test/lu_real/validate_supplemental.py \
  --library build-lu950/libops_solver.so \
  --reference-bridge build-lu950/libreference_bridge.so --output supplemental-lu950
.venv-lu950/bin/python -I test/lu_real/validate_medium_boundary.py \
  --library build-lu950/libops_solver.so \
  --reference-bridge build-lu950/libreference_bridge.so --output medium-boundary-lu950
.venv-lu950/bin/python -I test/lu_real/validate_raw_boundary.py \
  --library build-lu950/libops_solver.so \
  --reference-bridge build-lu950/libreference_bridge.so --output raw-boundary-lu950
```

补充入口包含72项不加对角增强（均匀/正态各半）和11项奇异、近奇异、整数及非连续指针检查。
中阶入口包含27项33至128阶的完整4096矩阵批量/padding边界，以及2项非连续指针、
无主元、内层空指针/重叠/非法主元/奇异info混合批量。每轮使用新的输出目录。
这些自构造项不代替主任务全部712项，也不使用Host时间判定性能。

原始小逆边界入口检查402项：3/4/5及10至16阶的180项分组与布局检查，batch在4096、8192、32768各阈值的前一项、
本项和后一项，紧密布局及独立lda/ldc padding各一组；另将真实10阶近奇异失败输入以batch1/33/4097、
两种padding保留为6项回归；10至13阶另加24项batch5376分组上界检查。输入按原始FP32位模式存于cases/regression_raw10_status.json。
17至32阶另加192项batch511/512/513/4095/4096/4097与两种padding的分组边界。
全部3283014个矩阵检查FP32 CPU金标和info；非奇异项检查
FP64残差、info、输入保持、输出padding及两次调用确定性。

已有LU小逆边界另检查17至32阶、batch128/512/4096和两种独立lda/ldc布局，共96项、151552个矩阵。
使用原FP32 CPU LU金标与已有LU求逆残差判据（底限0.05），不以此代替主任务性能检查。

```bash
.venv-lu950/bin/python -I test/lu_real/validate_small_boundary.py \
  --library build-lu950/libops_solver.so \
  --reference-bridge build-lu950/libreference_bridge.so --output small-boundary-lu950
```

## 内存工具补充检查

任务书3.4不设内存量化门槛；以下检查覆盖47个代表路径，记录同步驻留HBM差值，不能作为峰值或GPU内存比。
新增3/4/5阶中小批量检查覆盖320/192/128矩阵分组及尾部不足一组的路径；
另覆盖10阶按列LU的batch1和4097、13阶单轮分组80/96、16阶batch1/4097及padding路径。新增10项17至32阶缓冲复用路径（含22/27阶group40），覆盖紧密和padding布局、单轮分组以及UB接近上界的分组。原28项完整保留。

```bash
mssanitizer --tool=memcheck --leak-check=yes --log-file=memory-lu950-tool.log -- \
  .venv-lu950/bin/python -I test/lu_real/validate_memory.py \
  --library build-lu950/libops_solver.so --output memory-lu950.json
```

`--`须保留，以免工具把Python的`-I`当作自身选项而仅输出帮助。
同时检查应用JSON的50项结果、工具实际检查的kernel数及错误/泄漏日志；不能只凭工具退出0宣称通过。
内存工具运行与性能采样串行，避免工具插桩影响性能数据。
