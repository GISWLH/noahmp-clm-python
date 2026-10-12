# GPU 迁移设计与阶段记录

## 当前状态

阶段 A 的状态规约和阶段 B 的 JAX 默认链路已经写入 `src/noahmp_clm/gpu/`。
本机 Windows 的 JAX 0.11.2 是 CPU wheel；JAX API 使用 `jit`、`vmap`、
`lax.scan`、`lax.switch` 和固定次数 `lax.fori_loop`，迁移到 xuanze 的
CUDA-enabled JAX 时不需要改变物理接口。阶段 C 的 48 步官方/普通版 parity、
阶段 D 性能表、CuPy 对照和 CUDA kernel 仍待完成。

## 设计决策

`ColumnState` 不进入 JAX trace。`pack.py` 将四层土壤、三层固定雪槽、标量
状态和 32 个 `Opt*` 配置拆成 `fvec/ivec`；雪的有效层数是 `ivec[32]` 的 mask。
阶段 F 会把这些字段转为 `field[column]` 的 SoA 内存布局，让一个 CUDA warp 读取
相邻柱时访问连续地址。单柱函数没有全局缓存、文件 I/O 或柱间通信。

普通版的割线求解器有“收敛即返回”和失败后的网格回退。JAX 版以固定 25 次
割线循环替代早停，并在每次更新把 Ts 限制在 `[200, 330] K`；MOST 以固定 20
次循环替代相对变化早停。四层土壤热方程和三层雪热方程采用固定维度 Thomas
展开，避免动态 `scan` 与 Python 列表。

方案分发使用 `lax.switch`。目前降水 1--3/5、土壤水胁迫 1--3、地面阻抗 1--4、
表面拖曳 1--2、气孔 1--2、雪反照率 1--2、雪导热 1--5、冻土和径流 1--4
均有对应纯函数；SNICAR、VIC 和新安江选项沿用阶段 1 的占位范围。

## 精度和运行方式

默认验证使用：

```powershell
$env:XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
$env:JAX_NUM_THREADS="1"
$env:JAX_ENABLE_COMPILATION_CACHE="false"
$env:PYTHONPATH="src"
C:\Users\hhuwl\venvs\noahmp-gpu\Scripts\python.exe -m pytest -q tests/test_gpu_jax.py
```

`jax_enable_x64=True` 的配置由测试显式设置。Windows CPU wheel 在同一进程为
不同形状重复编译时曾触发 XLA HLO verifier 的内部中止；批量 JIT 函数因此在
模块级缓存，测试固定形状，并关闭该 wheel 的持久编译缓存。这是运行时限制，
不是物理分支或容差放宽。CUDA-enabled Linux wheel 需要重新做一次 C 阶段 parity
验证，不能把当前 CPU 编译结果当作 GPU 实测。

float32 与 float64 的日累计偏差、CuPy 和手写 CUDA 的偏差将在阶段 C/E/F 实测后
补入本文件；当前没有伪造这些数字。
