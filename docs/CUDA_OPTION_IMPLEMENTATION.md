# CUDA 选项移植与扩展验收清单

本清单最初审阅基线为 `a3aec95`，日期 2026-10-13；下文新增 J3 的实测记录。
选项移植以 `f29d7ad` 的 J2 默认运算树修复为基础，保持状态 ABI 和所有验收阈值。
这些结果不替代 H4 的 NumPy64 门，不证明年度稳定性或真实大陆域验证通过。

## J3 实现与实测记录（2026-10-13）

CUDA 已透传下表 12 个字段，实现 35 个值的 device 分发。默认专用路径与运行期
选项路径通过模板共享同一套物理函数；没有复制两套物理。仍为 float32、
1 thread/column、SoA、grid-stride、`__restrict__`，保留
`--fmad=false --prec-div=false`，输入缓冲区所有权规则不变。
loader 对 18 个 ABI-only 字段和 2 个 singleton 只接受其默认值；未实现物理值
明确报出字段、值和柱号。它们没有因放宽 loader 而被虚假标记为支持。

测试入口为 `tests/test_gpu_cuda_options.py`。逐字段 35 值分别运行 H6 的 8 情景、
24 步，并比较完整最终浮点/整数状态和全部 12 项逐步诊断；另有 1024 柱、48 步
混排。CUDA 与实际 JAX-CUDA float32 的门保持 `rtol=atol=1e-6`。径流探针直接
调用生产 device 函数，对 27 组土壤/湿度/地形、16 组 surface/subsurface 选项配对、
4 个入流水量，共 1728 个输入，分别检查两个出口与 JAX32、量化输入后的普通版64
对齐和总量守恒。只比较总径流不足以检验这两个选项，此项没有省略。

首轮为 **52 passed / 2 failed**，原始日志保留于
[j3-options-first.log](gpu-results/j3-options-first.log)。失败为 RainSnow=3 和混排，
首先出现在雨雪过渡砂土情景的残差，原有默认 J2 的 1/48 步门均通过。
定位采用未修改的生产 24 步完整 HLO，以及仅追加中间诊断的临时 JAX 副本。
追加诊断前后全部原状态和 12 项诊断逐位不变；CUDA 探针使用实际 prestate 和
生产求出的 Ts。不能用孤立标量子图的常量折叠来代替这个证据。

根因为气孔温度因子的运算树：JAX 执行
`1 - 0.0016 * ((298 - tair) * (298 - tair))`，旧 CUDA 左结合先乘 `0.0016`。
在 tair=274.2 K 时，两者因子分别为 0.09369701147079468 和
0.09369695186614990；Ts、rnet、H、G 相同，但气孔阻抗和蒸腾改变，最终 LE
相差 7.62939453125e-6 W/m²。修复为先平方后乘系数后，该点的 rs、tr、LE、
residual 逐位一致。证据见
[完整生产 HLO 摘录](gpu-results/j3-temperature-hlo-excerpt.txt)、
[修复前分量](gpu-results/j3-transition-before.json) 和
[修复后分量](gpu-results/j3-transition-after.json)。这与 J4 的四项根系归约树问题
是两条独立原因，不属于 float32 对 float64 的误差预算。

修复后，在原 `f29d7ad` JAX 基线上 **54 passed in 91.02 s**，包括默认 J2、
输入 ownership 和上述选项测试，见
[开发阶段完整日志](gpu-results/j3-options-temp-square.log)。nvcc sm_89 编译 exit 0、
无 warning。该轮 raw JAX hash 含 git archive 导出的 CRLF；LF 规范化后与
`f29d7ad` 完全相同，见 [源码身份记录](gpu-results/j3-reference-identity.json)。
它是后续 JAX 显式根系归约修复之前的结果，不能冒充联合版本的最终验收。

最终联合版本 **55 passed in 105.04 s**，包括上述 54 项和真实 J4 输入的
64800/N1 批量形状回归，见 [最终日志](gpu-results/j3-options-final.log) 和
[最终源码哈希](gpu-results/j3-options-final-sourcehash.txt)。最终 nvcc sm_89 编译
exit 0，stdout/stderr 日志为空（无 warning）。默认 CUDA 内核源码已改变，
先前默认内核的性能数字也必须重跑 checksum/逐值门后才能使用；本节不提供新性能数字。
H4 的 NumPy64 近零残差门仍保持红灯，容差提案未应用。35 值逐字段抽查和混排
通过也不能外推为所有选项笛卡尔积或所有薄雪边界均通过。

最终联合源码中的 JAX 根系四项归约来自 `c6bb8c6`。本轮三个生产文件的 raw/LF
SHA256 相同：CUDA `7640b062a329bf290c3719a62bddf97ea2ec523167b387730e758b8bd1b40454`；
loader `c4afdbed8502242c356ea970e5ce74b375bae8dfaefb082249ae1148c6b4d08a`；
JAX `0e2966918699eb39ee520ec2993eee5819df9d49371a1bfc87901b2dc8e69270`。
复现需要 Linux NVIDIA GPU、CuPy/JAX CUDA，使用新日志路径保存每次运行：

```bash
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NOAHMP_REQUIRE_GPU=1 PYTHONPATH=src
bash src/noahmp_clm/gpu/cuda/build.sh
python -m pytest tests/test_gpu_float32_parity.py tests/test_gpu_cuda_inputs.py \
  tests/test_gpu_cuda_options.py tests/test_gpu_batch_shape.py -q --tb=short
```

定点分量与生产 HLO 的独立复现入口为
`python benchmarks/probe_cuda_option_residual.py --output <新路径>.json`。
它同时保存输入/逐步诊断 NPZ 和完整生产 HLO；只有追加诊断前后全部原状态与
原诊断逐位一致时，才接受仪器化分量。实测完整 HLO/NPZ 另保留在 xuanze
`/home/lhwang/work/noahmp-phase3b/logs/j3-transition-before.*` 和
`j3-transition-after.*`，仓库保存提要、哈希和完整可复现脚本。

## 选项范围：12 个物理字段、35 个已实现值

依据 [PhysicsOptions](../src/noahmp_clm/options.py)、
[普通版驱动](../src/noahmp_clm/driver.py)、
[JAX 核心](../src/noahmp_clm/gpu/jax_core.py) 与
[H6 审计清单](../benchmarks/check_gpu_options.py)，35 是下列各字段可移植值数之和，
不是 35 个不同字段，也不是所有 Noah-MP 原始物理方案均已实现。
表中偏移是 `ivec`/SoA 的 **零起始偏移**；现有 70/33/8 ABI 保持不变。

| 整型偏移 | 字段；默认值 | 可移植值及普通版语义 | 普通版入口 → JAX 入口 | CUDA 分发位置及依赖 |
|---:|---|---|---|---|
| 1 | `OptRainSnowPartition=1` | 1 Jordan；2 BATS；3 冰点阈值 | `precip.partition_precipitation` → `_snow_fraction` | 在 `step` 分解降水前调用相态 `switch`；依赖 tair、dt、prcp，更新雪水和截留雨水 |
| 2 | `OptSoilWaterTranspiration=1` | 1 Noah 体积水；2 CLM 基质势；3 SSiB 基质势平方 | `stress.soil_water_stress` → `_soil_stress` | `beta` 同时提供表层值和四层根系加权值；依赖 smc、root_frac、smcwlt/smcref/smcmax/bexp；接入 `stomata` 和裸土蒸发 |
| 3 | `OptGroundResistanceEvap=4` | 1 Sakaguchi–Zeng；2 Sellers；3 adjusted Sellers；4 无雪同 1 | `ground_evap.ground_resistance` → `_ground_resistance` | `ground_resistance` 分发，供裸土 LE 使用；依赖表层含水、固定 dz、土壤参数；当前雪面无升华，4 的雪阻抗 1 s/m 没有整柱调用路径 |
| 4 | `OptSurfaceDrag=1` | 1 MOST；2 bulk/Chen | `surface_drag.exchange_velocity` → `_exchange_velocity` | `exchange` 分发；初始 PM 和每次能量残差均调用；MOST 保留 20 次上限及收敛冻结，bulk 不执行 MOST |
| 5 | `OptStomataResistance=1` | 1 Ball–Berry；2 Jarvis | `canopy.stomatal_resistance` → `_stomatal_resistance` | `stomata` 分发；两者共享水分胁迫、饱和水汽压、温度/光照响应；Ball–Berry 使用仓库经验光合，并非完整 Farquhar |
| 6 | `OptSnowAlbedo` | legacy ABI：1 BATS、2 CLASS；extensive `CLMColumnState`：3 SNICAR 五波段 | legacy `snow.update_snow_albedo` / extensive `canopy_snow_radiation` | 旧标量 ABI 保留1/2；完整五层 CLM 路径使用 `physics.snicar` 及 CUDA/JAX SNICAR kernels，步末标量反照率仍不等同于五波段光学状态 |
| 9 | `OptSnowThermConduct=1` | 1 Stieglitz；2 Anderson；3 常数；4 Verseghy；5 Douville | `snow.snow_thermal_conductivity` → `_snow_conductivity` | `snow_k(density, temperature, option)`；同时用于 `snow_flux` 与 `snow_heat`，不能只改其中之一；4 使用每层温度 |
| 11 | `OptSoilSupercoolWater=1` | 1 Niu–Yang；2 仓库 Koren 三次平均迭代入口 | `frozen_soil.unfrozen_water_content` → `_unfrozen_water` | `liquid` 分发；`capacity` 的 T±0.02 K 两次调用和所有相态刷新均透传选项 |
| 12 | `OptRunoffSurface=3` | 1 TOPMODEL；2 平衡水位 TOPMODEL；3 free drainage；4 BATS | `runoff.route_runoff` → `_route_runoff` / `_runoff_scheme` | 在土壤水更新完成后对合计候选径流分发；依赖更新后的 smc、terrain_index、groundwater_depth、土壤参数 |
| 13 | `OptRunoffSubsurface=3` | 1/2/3/4 同上 | `runoff.route_runoff` → `_route_runoff` | 若与 surface 不同，对第一轮剩余地下径流再分发；若相同只执行一次。最终输出两分量后求和，不可把二次分发施加到原始总量 |
| 14 | `OptSoilPermeabilityFrozen=1` | 1 倍率全为 1；2 `exp(-7*ice/smcmax)` | `frozen_soil.permeability_factor` → `_soil_water` | `soil_water` 的 Campbell 排水前缓存**入渗后、蒸散前**冰量；不能使用热更新后的冰量或每层扣水后重算的冰量 |
| 23 | `OptSnowCompaction=2` | 1 Anderson；2 较慢温度响应占位方案 | `snow.compact_snow` → `_compact_snow` | `snowfall` 后密度更新分发；warm/load 系数分别 1e-7/2.5e-7 和 6e-8/1.5e-7；不改变 SWE |

计数：`3+3+4+2+2+2+5+2+4+4+2+2 = 35`。上表的支持范围由生产 device 分发、
整柱严格对照和径流局部分量探针共同验证；不能仅凭 `_check_options` 接受值来认定。

## 18 个 ABI-only 字段和 2 个单值字段

以下字段被配置类接受，但普通版和 JAX 驱动物理路径不消费其值。ABI round-trip 或
“改值后输出不变”只能证明兼容性，不能证明实现了相应物理。

| 偏移 | ABI-only 字段 | 配置接受值 | 当前实际行为 |
|---:|---|---|---|
| 7 | `OptCanopyRadiationTransfer` | 1,2,3 | 固定 big-leaf 辐射 |
| 8 | `OptSnowSoilTempTime` | 1,2,3 | 固定时间积分 |
| 10 | `OptSoilTemperatureBottom` | 1,2 | 固定土壤底部零热通量；默认值 2 并没有施加深层规定温度 |
| 15 | `OptDynVicInfiltration` | 1,2,3 | Dynamic VIC 未实现 |
| 16 | `OptTileDrainage` | 0,1,2 | 无瓦片排水 |
| 17 | `OptIrrigation` | 0,1,2,3 | 无灌溉 |
| 18 | `OptIrrigationMethod` | 0,1,2,3 | 无灌溉方式分支 |
| 19 | `OptCropModel` | 0,1 | 无作物生长模型 |
| 20 | `OptSoilProperty` | 1,2,3,4 | 使用传入 SoilParams |
| 22 | `OptGlacierTreatment` | 1,2 | 无冰川物理 |
| 24 | `OptSnowCoverGround` | 1,2 | 固定雪阈值和 `min(1,SWE/0.02)` 蒸腾抑制 |
| 25 | `OptWetlandModel` | 0,1,2 | 无湿地模型 |
| 26 | `OptSnicarSnowShape` | 1,2,3,4 | extensive SNICAR 使用五波段公共表的固定形状近似；selector 保留但不改变当前表映射 |
| 27 | `OptSnicarRTSolver` | 1,2 | extensive SNICAR 使用稳定薄层 adding/doubling 求解器；selector 保留 |
| 28 | `OptSnicarBandNum` | 1,2 | extensive SNICAR 固定五波段；selector 保留 |
| 29 | `OptSnicarSolarSpec` | 1,2,3,4,5,6 | extensive SNICAR 直/散射边界显式输入；selector 保留 |
| 30 | `OptSnicarSnwOptic` | 1,2,3 | extensive SNICAR 使用公共冰雪光学表；selector 保留 |
| 31 | `OptSnicarDustOptic` | 1,2,3 | extensive SNICAR 使用四类气候态尘埃及一类 BC；selector 保留 |

另两个 singleton：偏移 0 `OptDynamicVeg=4` 仅使用输入 LAI；偏移 21
`OptPedotransfer=1` 仅使用传入土壤参数。它们无法提供“每字段至少两个取值”的证据。

明确拒绝的物理值应在 host 校验时报告：RainSnow 4/5、SnowAlbedo 3、
RunoffSurface 5–8、RunoffSubsurface 5–8。普通版对这些值抛出
`NotImplementedError`（雪反照率有积雪时触发）；JAX 的 `lax.switch` 截断索引及
雨雪 4/5、反照率 3 的占位分支不是支持依据。CUDA 不应模仿该静默回退。

## 数值和分发验收中的陷阱

1. **runoff 总量会掩盖未分发。** 目前方案只重分配两个出口，不反馈状态；
   `diag[7]` 是两个出口之和。即使 CUDA 完全忽略两个径流选项，现有整柱诊断和
   最终状态也可能全部通过。因此必须用独立测试 kernel 调用同一 device 函数，
   输出 `(surface, subsurface)` 与普通版/JAX 分量比较；这类测试输出不进入状态 ABI。
   用湿土、高/低 terrain_index 和浅/深 groundwater_depth，覆盖 4×4 配对。
2. **某些已实现值数值相同是当前模型语义。** 无雪地面阻抗 1/4 相同；
   supercool 2 从 Niu–Yang 固定点开始再取三次平均，当前实现可能与 1 完全相同。
   不能强求每个选项都改变整柱输出，也不能因为输出相同而跳过源码分发检查。
3. **默认标签不等于物理实现。** 底温默认 2 仍是零通量；snow compaction 2 和
   Koren 2 是本仓库简化/占位关系；Ball–Berry 没有完整 Farquhar。移植目标先是
   本仓库普通版，不能在 CUDA 扩展时顺便换成另一套文献方程。
4. **雪状态时序不能移动。** 新雪重分层后再压实；能量求解用旧反照率；
   融雪后更新反照率。雪面能量忽略升华、LE 诊断为零，雪下土壤热通量为零；
   本步彻底融化时 G 诊断归零。不要借扩展选项改变这些现有语义。
5. **所有阈值和形状均需边界样例。** 新雪分层 SWE=0.05/0.15 m、雪面开关
   SWE=1e-4 m、保留层 1e-8 m、反照率重置新雪 1e-6 m、三种降水相态阈值应
   各取阈值上下可表示数。已有 JAX `_snow_heat` 对有效薄雪厚度有 1e-4 m floor，
   普通版 `apply_snow_heat` 与 CUDA 有效层厚度未采用同一 floor。此项已由 J2
   最小例证实并修复，见 [GPU.md](GPU.md#j2有效薄雪层的热容量修复)；修复后的
   年度短测另暴露普通版也有的薄雪边界通量稳定性问题，见
   [GPU_EXTENDED_VALIDATION.md](GPU_EXTENDED_VALIDATION.md)。不能把 H6 八情景
   通过外推到所有薄雪边界。
6. **FMA/运算树属于数值规范。** J2 确认的相减、乘除、Thomas 和迭代冻结顺序
   应由所有选项共用；不能只给默认分支保留修复。对 opt 混排柱保持 grid-stride、
   `__restrict__` 与 1 thread/column，并检查输入/输出禁止重叠仍生效。

建议实现顺序：先透传 12 个整型选择器；再完成无雪四类分支（相态、stress、
ground resistance、stomata/drag）；再雪 conductivity/albedo/compaction；再 frozen
liquid/permeability；最后加入两个 runoff 分发与分量探针。每一组先通过局部
device 函数对照，再纳入 8 情景×24 步及混排 batch。仅在 J2 已有的同精度
数值准则下验收；残差容差提案保持独立，未获 Wang 确认前不得写入测试门。

最小矩阵采用 H6 的 8 个情景，逐字段取上表 35 个值，其他字段为默认；额外
混排 1024 柱、48 步验证不同字段联动。由于 runoff 与 supercool 的可观察性限制，
清单应同时记录“输出 parity 通过”和“分支执行/局部分量通过”。禁止把全组合
覆盖从单字段抽查结果中推导出来。

## J4 输入审计：仓库没有真实大陆域输入

审阅了 `src/`、`examples/`、`benchmarks/` 及工作区常见气象/地理数据扩展名。
当前仅发现 [io.synthetic_forcing](../src/noahmp_clm/io.py) 的正弦日循环、固定
第 2 天暴雨，以及 `column.default_soil_params` 的 sand/loam/clay 三类表值；
没有随仓库交付的经纬度网格、地表覆盖、土壤栅格、年气象强迫或大陆域初始化文件。
[parity_noahmp.py](../examples/parity_noahmp.py) 的 `--reference` 读取外部日统计
CSV，并不提供真实域强迫或运行 Fortran。

因此可立即执行的是 **64800 柱合成大陆规模集成压力测试**。它可检验分块、
独立格点、设备划分、全年积分和内存管理，不能称为某一真实大陆的科学验证。
真实域验收还需要有来源与时间轴的 8 强迫、初始土壤/雪状态、土壤参数、植被
参数及网格映射；下载/生成这些输入是单独的数据准备工作。

## J4 最小可重复运行与预算

以下是执行预算，均不是已测耗时或通过结果。所有运行记录代码 SHA、源文件
hash、输入 seed/hash、设备、精度、选项、完成步数与 UTC；非有限数或门失败
立即写出首个 `(column, step, field)` 和可重放的输入/状态片段。

| 项目 | 最小规模与输入 | 检查内容 | 单轮预算 |
|---|---|---|---|
| 年度长稳 | 8 柱×365 天×48 步，dt=1800 s；固定 seed 的年/日周期、周期降雨；包括温带、冻融、干旱、持续冷雪和融雪初态 | JAX x64/x32，CUDA 仅在 J2 后纳入；每日有限性、温度和水量边界、雪层一致性、累计 P/ET/Q/储水；首末日及失败窗口存样 | 每 backend 上限 30 分钟；先 7 天 smoke，剩余时间按实际速率预估，但只报告完成的实测天数 |
| 64800 柱域 | 180×360 仅作整数索引网格，24 或 48 步；固定 seed，将土壤/植被/强迫有界扰动与 8 情景分组组合 | 全量有限性、设备 reduce checksum、CPU 可复算子集；单卡整域 vs 分块拼接、单/双卡分片一致性 | 30 分钟；无年度整域输出常驻显存 |
| 双卡扩展 | 相同64800及100000柱输入，一卡全量 vs 两卡各半；每卡独立 process/context | 3 预热+3 测量；同步起跑、两卡全部完成后计 wall；结果按全局柱 ID 拼接并 checksum | 20 分钟；共享强迫和初始 H2D 的计时边界与单卡一致 |
| 年度普通版锚点 | 相同 8 个小规模年度情景，真实 NumPy 执行；若预算不足至少保留一个完整年度及其余情景完整日窗口 | 比较日/年 ET、H积分、最终状态及水量残差；不得以每日 reset 假装连续365天 | 30 分钟；记录已完成范围，未完成不补算为通过 |

年度输入不能直接把当前 `synthetic_forcing(n_days=365)` 当作充分覆盖：它只有
第 2 天一场雨，缺乏年季节变化和持续冻融。可用固定解析年周期叠加当前日周期，
每 7 天固定时段降水、按情景设置 tmean/tamp/季节幅值和起始 SWE；生成器与
seed 一起入库，标签始终为 synthetic。一个连续年度必须承接前块的完整 fvec/
ivec，不能每天重新初始化。推荐按 48 或 336 步分块，daily 统计用 float64 reduce，
保存 checkpoint 和失败片段；逐步数据可在小 8 柱实验保留，在大域只保留抽样。

定义水储量 `W = Σ(smc[k]·dz[k]) + SWE + canw`（m），soil_ice 是 smc 内部分量，
不能再加一次。每日 `ΔW - (P-ET-Q)` 是守恒审计量；雪层小量裁剪可能造成微小损失，
需要记录绝对累计量，不应只用会相消的年净差。H 应累计 `Σ(H·dt)/1e6` MJ/m²，
ET 直接累加诊断 mm；均同时报告绝对差和非零基准的相对差。
长稳检查不会自动修改 H4 的近零残差门，也不会把经验稳定性当成物理模型验证。

64800×48×12 的诊断精确占用：float32 为 149,299,200 B，float64 为
298,598,400 B。整年同域仅诊断就需 54,494,208,000 B / 108,988,416,000 B，
尚未计输入、临时数组和状态，必须分块/归约。该内存估算来自数组形状，
不是测得的 GPU 内存峰值。8 柱全年 float64 诊断为 13,455,360 B，适合完整留存。

双卡效率必须用实测 `T_one / T_two`；格点独立仅意味着无需每步跨卡通信，
不能据此预先宣称 1.9×。设备同时执行前需要记录其他作业的 GPU 占用。
JAX 当前 loop launch 开销明显，64800 规模的两卡可能反而受相同固定开销限制；
该结果也应如实保留。

独立入口 [check_gpu_extended.py](../benchmarks/check_gpu_extended.py) 已提供 annual/domain
模式；下面命令是待调度的复现入口，本文不宣称已经在 GPU 上执行。annual 默认
8 柱、每日 48 步，默认每次运行预算 1800 s；domain 默认 64800 柱、48 步。
输出必须使用新路径。checkpoint 每完成一天原子写入，保存完整状态、累计水量
误差和已完成天数，续跑要求 backend/dtype/seed/源文件 hash 相同，且上一日未失败。
domain 另做抽样普通版/单柱对照，以及全体列 tair+0.125 K 的敏感性检查。

```bash
python benchmarks/check_gpu_extended.py --mode annual --backend jax64 \
  --days 365 --max-seconds 1800 --output ../logs/j4-annual-jax64.jsonl
python benchmarks/check_gpu_extended.py --mode annual --backend jax64 \
  --days 365 --resume ../logs/j4-annual-jax64.checkpoint.npz \
  --output ../logs/j4-annual-jax64-resumed.jsonl
python benchmarks/check_gpu_extended.py --mode domain --backend jax32 \
  --columns 64800 --steps 48 --output ../logs/j4-domain-jax32.jsonl
```

脚本强制实际 CUDA 设备，生产状态 dtype 为指定精度，device 归约使用 float64。
每日输出 ET/H、能量残差、水量误差、状态/诊断 min/max、普通/加权 checksum 和
输入 hash。退出码 0 为本次请求范围完成且已检查门通过，1 为失败，2 为预算到期
但年度未完成。水量漂移仅报告，不新增验收容差；真实大陆域验证仍未获得输入。

Roofline 和 ncu 必须在 checksum 通过后采集：分别报告实际 FP32/FP64/FMA/
特殊函数指令计数、DRAM 字节、局部内存 spill、寄存器、active warps/occupancy
及 kernel 时间。`N×steps/time` 是柱步吞吐，缺少每柱步的实际指令计数时无法
转换为 TFLOPS，也不能仅据此认定 DCE。先确认输出消费和依赖，再判断实际算术
强度与热点；双卡、48 步融合和 occupancy 调整都保留同一 checksum/数值门。
