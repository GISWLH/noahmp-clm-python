# GPU 迁移设计与阶段记录

## 当前状态

xuanze Linux 的 JAX-CUDA float64 已通过 8 柱 × 48 步普通版 parity，
默认物理 CUDA kernel 已经 nvcc/NVRTC 编译执行。手写 kernel 的严格 float32
parity 仍未通过；GPU 五档性能已实测，NumPy 大规模单核对照正在补测。
这些结果对照本仓库普通版，不代表完整 Noah-MP 官方站点验收。

## 设计决策

`ColumnState` 不进入 JAX trace。`pack.py` 将四层土壤、三层固定雪槽、标量
状态和 32 个 `Opt*` 配置拆成 `fvec/ivec`；雪的有效层数是 `ivec[32]` 的 mask。
CUDA loader 把这些字段转为 `field[column]` 的 SoA 内存布局，让一个 CUDA warp 读取
相邻柱时访问连续地址。单柱函数没有全局缓存、文件 I/O 或柱间通信。

普通版的割线求解器有“收敛即返回”和失败后的网格回退。JAX 版以固定 25 次
割线循环内冻结已收敛的状态，并在每次更新把 Ts 限制在 `[200, 330] K`；MOST
同样在固定 20 次循环中保留普通版早停语义。四层土壤热方程和三层雪热方程采用固定维度 Thomas
展开，避免动态 `scan` 与 Python 列表。

方案分发使用 `lax.switch`。已验收范围为降水 1--3、土壤水胁迫 1--3、地面阻抗 1--4、
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
不同形状重复编译时曾触发 XLA HLO verifier 的内部中止；批量 JIT 函数在
模块级缓存。原始混合图在 Linux CUDA 未复现该中止，随后完成数值修复和 GPU
parity；尚未单独证明 Windows wheel 中止的内部根因，不能据此断言其必为 wheel 缺陷。

一日 float32/float64 累积偏差见下文 H6；手写 CUDA 的严格失败和数值条件分析
见 [GPU_BLOCKERS.md](GPU_BLOCKERS.md)。

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

## H6：Linux GPU 选项矩阵与一日精度实测

2026-10-13 EDT，xuanze 物理 GPU 1（RTX 4090，进程内为 `CudaDevice(id=0)`）
完成 `benchmarks/check_gpu_options.py`，退出码 0。审计显式要求 CUDA 后端，
并检查所有结果确实位于 GPU、浮点 dtype 与请求一致且无非有限值。
八种独立场景覆盖冻结、已有三层雪、两种土壤消融、暴雨、两种土壤干旱和
274.2 K 雨雪转换；每个配置运行 **24 步**，每次只改变一个选项。

全部 32 个 `PhysicsOptions` 字段、103 个配置接受值均已清点：

| 分类 | 字段数 | 取值数 | 本次结论 |
|---|---:|---:|---|
| 已实现物理选择器 | 12 | 35 | JAX float64 与普通版 parity 全部通过 |
| 已声明但尚未实现的物理取值 | 上述 12 字段中的 4 个 | 11 | 普通版均明确抛出 `NotImplementedError`；不执行 GPU，不计为物理通过 |
| ABI 保留、未消费的选择器 | 18 | 55 | parity 通过，且普通版/JAX 的浮点状态和诊断均与默认配置逐元素完全相同；只证明无效应及 ABI 保留 |
| 只有一个合法取值的字段 | 2 | 2 | 默认值 parity 与无效应检查通过；无法提供两个合法值 |

日志的 `92 passed` 是 **35 个物理取值 + 55 个无效应取值 + 2 个固定取值**，
另有 `11 unsupported_expected`，不能把它写成“92 种物理方案通过”。
已实现的 12 个物理字段均实际检查不少于两个值；“每个 Opt 均有至少两种物理
方案”的字面目标无法满足：18 个字段仍是占位配置，`OptDynamicVeg` 仅允许 4，
`OptPedotransfer` 仅允许 1。未实现取值也没有获得实现支持；JAX 的 `lax.switch`
会夹紧分支索引，不能凭它返回结果就声称支持这些值。以下表格以普通版的
实际分发为准，也更正了早期设计段中把降水取值 5 列为可对照支持的表述。

| 字段 | 类别 | 本次 GPU 检查值 | 已声明但未实现的物理值 |
|---|---|---|---|
| `OptDynamicVeg` | 单值固定 | 4 | — |
| `OptRainSnowPartition` | 物理 | 1, 2, 3 | 4, 5 |
| `OptSoilWaterTranspiration` | 物理 | 1, 2, 3 | — |
| `OptGroundResistanceEvap` | 物理 | 1, 2, 3, 4 | — |
| `OptSurfaceDrag` | 物理 | 1, 2 | — |
| `OptStomataResistance` | 物理 | 1, 2 | — |
| `OptSnowAlbedo` | 物理 | 1, 2 | 3 |
| `OptCanopyRadiationTransfer` | ABI 保留/无效应 | 1, 2, 3 | — |
| `OptSnowSoilTempTime` | ABI 保留/无效应 | 1, 2, 3 | — |
| `OptSnowThermConduct` | 物理 | 1, 2, 3, 4, 5 | — |
| `OptSoilTemperatureBottom` | ABI 保留/无效应 | 1, 2 | — |
| `OptSoilSupercoolWater` | 物理 | 1, 2 | — |
| `OptRunoffSurface` | 物理 | 1, 2, 3, 4 | 5, 6, 7, 8 |
| `OptRunoffSubsurface` | 物理 | 1, 2, 3, 4 | 5, 6, 7, 8 |
| `OptSoilPermeabilityFrozen` | 物理 | 1, 2 | — |
| `OptDynVicInfiltration` | ABI 保留/无效应 | 1, 2, 3 | — |
| `OptTileDrainage` | ABI 保留/无效应 | 0, 1, 2 | — |
| `OptIrrigation` | ABI 保留/无效应 | 0, 1, 2, 3 | — |
| `OptIrrigationMethod` | ABI 保留/无效应 | 0, 1, 2, 3 | — |
| `OptCropModel` | ABI 保留/无效应 | 0, 1 | — |
| `OptSoilProperty` | ABI 保留/无效应 | 1, 2, 3, 4 | — |
| `OptPedotransfer` | 单值固定 | 1 | — |
| `OptGlacierTreatment` | ABI 保留/无效应 | 1, 2 | — |
| `OptSnowCompaction` | 物理 | 1, 2 | — |
| `OptSnowCoverGround` | ABI 保留/无效应 | 1, 2 | — |
| `OptWetlandModel` | ABI 保留/无效应 | 0, 1, 2 | — |
| `OptSnicarSnowShape` | ABI 保留/无效应 | 1, 2, 3, 4 | — |
| `OptSnicarRTSolver` | ABI 保留/无效应 | 1, 2 | — |
| `OptSnicarBandNum` | ABI 保留/无效应 | 1, 2 | — |
| `OptSnicarSolarSpec` | ABI 保留/无效应 | 1, 2, 3, 4, 5, 6 | — |
| `OptSnicarSnwOptic` | ABI 保留/无效应 | 1, 2, 3 | — |
| `OptSnicarDustOptic` | ABI 保留/无效应 | 1, 2, 3 | — |

ABI 保留字段的“—”仅表示没有另行分发并抛错的值，不代表对应物理已实现。
径流两字段确实进入普通版方案分发，但柱诊断只输出地表与地下径流之和；本次
parity 不能单独验证未公开的两个分量。矩阵是逐字段抽查，不是全部选项组合的
笛卡尔积；混合配置由 H2 的独立 8 柱 × 48 步门覆盖。

本次使用 H2 原有容差：诊断 `ts` 至 `prcp_mm`、`canw`、`residual` 和土壤/标量
最终状态 `rtol=atol=1e-6`；`runoff_mm/melt_mm/swe` 与有效雪层
`rtol=atol=1e-4`；整数最终状态完全一致，能量残差绝对值小于 0.5 W/m²。
全部比较中最大的 `abs(error)/(atol + rtol*abs(reference))` 为
**6.74660e-6**（小于 1 才通过），没有修改 ABI 或放宽容差。

一日精度另外使用默认选项运行相同八场景 **48 × 1800 s = 86400 s**，两次
模型计算分别为 JAX float32 与 float64；ET 直接累加每步 `et_mm`，感热按
`sum(h*dt)/1e6` 积分为 MJ/m²。两组归约均使用 float64，避免混入额外的低精度
求和误差。以下偏差均为 **float32 − float64**：

| 场景 | ET 偏差 (mm/day) | 感热偏差 (MJ/m²/day) |
|---|---:|---:|
| 冻结砂土 | +5.93480e-11 | +4.39688e-7 |
| 壤土消融 | +1.19739e-6 | −7.08634e-6 |
| 黏土暴雨 | −1.59418e-6 | −3.90757e-6 |
| 砂土干旱 | −1.05071e-6 | −1.51299e-5 |
| 壤土三层雪 | 0 | −3.34893e-6 |
| 黏土消融 | +1.33341e-6 | −8.33898e-6 |
| 砂土雨雪转换 | +1.70575e-6 | −3.47928e-6 |
| 壤土干旱 | +9.64719e-6 | +2.15644e-5 |

结论：这些默认选项的一日场景中，最大 ET 绝对偏差为 **9.64719e-6 mm/day**，
最大相对偏差 **4.51152e-5（0.00451152%）**；最大感热绝对偏差为
**2.15644e-5 MJ/m²/day**，最大相对偏差 **1.03690e-5（0.00103690%）**。
绝对最大值均出现在壤土干旱场景；感热相对最大值出现在三层雪场景。零 ET 的
相对偏差记为 `null`。这支持本组一日场景累积量的 float32 精度较好，不等同于
长期误差保证，也不能代替手写 CUDA 的严格 H4 parity 门。

复现命令如下；审计拒绝覆盖旧输出，重跑时须选择新的日志文件名。

```bash
cd ~/work/noahmp-phase3b/repo
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true JAX_PLATFORMS=cuda CUDA_PATH=/usr/local/cuda-12.4
export PYTHONPATH=src
~/venvs/noahmp-phase3b/bin/python -u benchmarks/check_gpu_options.py \
  --output ../logs/h6-options.jsonl --summary ../logs/h6-options.json \
  > ../logs/h6-options.log 2>&1
```

原始证据：[完整运行日志](gpu-results/h6-options.log)、
[逐项 JSONL](gpu-results/h6-options.jsonl)、[汇总及逐场景误差 JSON](gpu-results/h6-options.json)。
任何实际 parity 失败、无效应检查失败、设备/dtype/有限性错误都会记录并返回非零
退出码；进程级中止会留下未完成日志，不能据此报告完成。

## J2：有效薄雪层的热容量修复

J4 边界准备暴露了 H6 八场景没有覆盖的有效薄雪层：顶部 SWE=1e-6 m、第二层
SWE=0.001 m 时，JAX `_snow_heat` 把顶部实际 3.33e-6 m 厚度抬到 1e-4 m，
给热容量和融雪能量引入额外质量。普通版与 CUDA 使用实际有效层厚度。
现在仅对无效占位层保留厚度占位，层间导热的分母保护不变。

修复前，1 W/m² 顶通量的雪温偏差约 0.065 K；100 W/m² 下融雪为
0.000504863 m，而普通版为 0.000519117 m。修复后对应融雪分别为
0.0005191167664670495 / 0.0005191167664670498 m。
三个 1/10/100 W/m² 回归和原 H2 parity 在实际 CUDA 上 **5 passed**，未改容差。
证据：[修复前](gpu-results/j2-thin-snow-before.log)、
[修复后与 H2](gpu-results/j2-thin-snow-after-h2.log)。这是独立语义修复，
不代表 CUDA32/JAX32 的全部 J2 数值差异已经解决，H4 继续红灯。
