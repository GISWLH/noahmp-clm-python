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

## 阶段 3b / H2：Linux GPU 根因与修复

2026-10-12/13 EDT，在 xuanze RTX 4090、JAX/jaxlib 0.11.2 下复测。
原始代码的 8 柱混合 `jit(vmap(scan))` 在 CUDA 上成功编译，Windows 的 HLO
进程中止不复现；但原始图的最大能量残差为 2158.487487 W/m²，因此原实现
不能通过物理 parity。不能用“不复现 HLO”代替数值验收。

定位到并修复的普通版差异：

- 裸地 `dqs` 错把比湿减去水汽压（Pa），暴雨案例使 LE 约为 −1909 W/m²。
  修为 `qsat - qair`，与 `driver.step_column` 相同。
- MOST 20 次和割线 25 次循环以前继续迭代，改变了普通版早停后的数值；
  改为固定上限内冻结收敛状态，并补齐普通版 131 点扫描、40 次二分回退。
- 雪下土壤现在保持普通版零表面热通量；融空层以固定长度 gather 前移，
  雪面/裸地过渡的冠层水和 G 诊断保持普通版语义。
- 新雪加入阈值与普通版保持一致（大于零），零 SWE 不再压实空雪层。
- 层厚常量按输入 dtype 构造；修复 x64 启用时 float32 `lax.switch` 分支的
  float64/float32 类型不一致，ABI 偏移、长度和普通版代码均未改变。

保留原测试，另加真正同时运行 8 柱 × 48 步的批量门，覆盖暴雨、冻结、三层雪、
消融、近萎蔫点干旱和混合选项；断言实际消融发生，比较诊断量、最终土壤状态
和有效雪层。沿用原容差（普通诊断 1e-6，雪/径流段 1e-4），没有放宽。

```bash
source ~/venvs/noahmp-phase3b/bin/activate
cd ~/work/noahmp-phase3b/repo
unset LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
XLA_PYTHON_CLIENT_PREALLOCATE=false NOAHMP_REQUIRE_GPU=1 PYTHONPATH=src \
python -m pytest tests/test_gpu_parity.py tests/test_gpu_jax.py tests/test_gpu_pack.py -x -q -s
```

最终结果：**7 passed**，输出设备 `CudaDevice(id=0)`；原始失败、GPU 编译探针、
库加载诊断和最终通过日志见 `docs/gpu-results/h2-*.log`。

## H3：默认物理 CUDA kernel 编译

2026-10-13 EDT，`bash src/noahmp_clm/gpu/cuda/build.sh` 在 xuanze 以
nvcc 12.4.99、`-O3 -std=c++14 -arch=sm_89` 成功构建共享库（约 1.3 MiB），
编译输出为空、退出码 0、零 warning。可用 `CUDA_ARCH` 覆盖本机算力。

源码保持 1 thread/column、grid-stride、float32 SoA、`__restrict__` ABI。
补入默认 MOST/Ball--Berry/PM 蒸腾与根区限制、地面和截留蒸发、三层雪mask、
默认 BATS、雪隐式导热与融化、Niu--Yang未冻水/相变热容、土壤入渗/排水、
25次割线及回退、固定Thomas4。所有物理函数注明普通版入口。
编译采用标准数学函数；性能优化不能先于 H4 容差验证。

H3 只确认完整默认物理源码可编译；CUDA 数值 parity 由 H4 独立验收。
