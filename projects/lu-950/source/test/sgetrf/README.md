# sgetrf 文件型测试（950PR）

本测试使用950PR公开Device API。构建和完整验收入口见[统一测试说明](../lu_real/README.md)。
文件采用Host行主序FLOAT32；`test/utils/real_solver_test.h`将其转换为Device列主序并同步下载。
公开接口本身的输入是Device列主序，不接受Host矩阵。

## 构建

在ops-solver根目录，加载CANN9.1.0环境后执行：

```bash
cmake -S . -B build-lu950 -DSOC_VERSION=ascend950 -DBUILD_TEST=ON '-DTEST_NAMES=sgetrf;sgetri'
cmake --build build-lu950 -j2
```

## 准备数据并运行

先按统一测试说明创建`.venv-lu950`并安装NumPy/SciPy。下列命令在ops-solver根目录生成32阶原始矩阵。
自定义尺寸时须同时修改输入生成和程序参数，输出目录须预先存在。

```bash
.venv-lu950/bin/python - <<'PYDATA'
from pathlib import Path
import numpy as np
p = Path('test/sgetrf/data')
(p / 'input').mkdir(parents=True, exist_ok=True)
(p / 'output').mkdir(exist_ok=True)
a = np.random.default_rng(20261016).uniform(-5, 5, (32, 32)).astype(np.float32)
a += np.eye(32, dtype=np.float32) * np.float32(32)
a.tofile(p / 'input/A_gm.bin')
PYDATA
build-lu950/test/sgetrf/sgetrf_test 0 32 32
```

程序参数为`deviceId m n`，默认`0 32 32`。输出`A_gm.bin`是合并LU的Host行主序矩阵，
`W_gm.bin`是INT32、1-based主元序列，二者位于`test/sgetrf/data/output/`。
另外可运行`build-lu950/test/sgetrf/sgetrf_subnormal_pivot_test`检查次正规主元。

文件型程序退出0只表示该程序执行成功；正式精度结论还需核对输出残差。
完整712项Device接口精度与msprof性能、补充用例和判据见统一测试说明。
