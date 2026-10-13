# M1：一次增量温度尝试，已回退

**未采纳生产修改。** 2026-10-13 07:30:35 EDT 开始，本次候选在约 7 分钟内
测出原 CUDA/JAX float32 同精度门回归，遂按 Wang 的“一次尝试、出现回归即回退”
指令结束。没有继续调求根阈值、容差或编译器运算树；两小时上限未用尽。

基线提交是 `b954d9022e5f96925f6d41fea833f740ab0c04f3`。生产文件全部恢复到开工时
逐字节 SHA256，`git diff -- src` 为空；其他并行工作的文件未回退。
[开工清单](gpu-results/m1-baseline-identity.json)、
[候选清单](gpu-results/m1-candidate-identity.json)、
[回退核验](gpu-results/m1-restored-identity.json) 保存完整身份。

## 做了什么

候选在 NumPy、JAX 和手写 CUDA 同步实现：

- 土壤、雪层隐式热传导改解 `A ΔT = boundary_flux − L T_old`，最后加回旧温度；
  均温且零边界通量时，温差 RHS 严格为零。
- 割线和原扫描/二分回退以气温为坐标原点，求 `δ = Ts − Tair`，保留原
  200–330 K 区间、25 次割线、40 次二分和 1e-3 W/m² 停止标准。
- MOST 感热、地表感热及导热直接评价温差；Tetens 在摄氏温差上评价；
  长波使用 `(Tair+δ)^4` 的完整四次展开，没有线性化或把 residual 设为零。
- ABI、储存精度和现有测试阈值均未改变。

完整[候选 patch](gpu-results/m1-candidate.patch)只作为未采用的尝试证据。
该改法能保护热传导均温不变量，但仅有这个代数性质不保证有限精度求根轨迹相同。
既有 K3 取证支持土温轨迹经 AHC 截断区放大；K1 的 137 项是先前未采用的
混合精度原型残留，不能把它写成本次生产基线的残差失败总数。

## 候选实际结果与回退原因

在 xuanze 物理 GPU0 / RTX 4090 上运行原测试，不改阈值：

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 CUDA_PATH=/usr/local/cuda-12.4 \
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/lhwang/venvs/noahmp-phase3b/bin/python -m pytest \
  tests/test_gpu_parity.py tests/test_gpu_float32_parity.py tests/test_gpu_cuda.py -q -s
```

结果 **4 passed / 4 failed，60.08 s**。NumPy64/JAX64 H2 通过；原 H4 的
1/48 步仍失败，且此前绿色的同精度 1/48 步两项新增失败。
原 H4 对 NumPy64 的失败元素分别为 **1024 / 12288**（1 步）和
**61659 / 589824**（48 步），阈值仍为 `rtol=1e-4, atol=1e-6`。
各项终态浮点状态均通过同精度门，整数状态全等，不能抵销诊断失败。

| CUDA vs JAX32 原同精度门 | 失败元素 | 最大绝对差 |
|---|---:|---:|
| 1 步 residual | 256 | 3.385543823e-5 W/m² |
| 48 步 rnet | 6 | 1.144409180e-4 W/m² |
| 48 步 H | 697 | 3.547668457e-4 W/m² |
| 48 步 LE | 66 | 6.103515625e-4 W/m² |
| 48 步 G | 3477 | 5.507469177e-4 W/m² |
| 48 步 residual | 10909 | 6.532669067e-4 W/m² |

同精度门仍为 `rtol=1e-6, atol=1e-6`。48 步候选对 NumPy64 的 residual
最大绝对差为 `6.473382280e-4 W/m²`，有改善也不能接受同精度回归。
[完整 stdout 与失败堆栈](gpu-results/m1-candidate-fast-tests.log)、
[所有诊断和70槽状态逐字段表](gpu-results/m1-candidate-sameprecision.json)均保留。

本次证据只确认候选产生了后端运算差异及其累计影响；没有进一步完成独立的
XLA/NVRTC 运算树因果探针，因此不把具体编译器优化写成已证实根因。
候选未跑完整域或全年：新增必过门回归已经触发预定回退条件。

## 回退源码的完整复验

最终使用恢复后的生产源码，独立目录为
`/home/lhwang/work/noahmp-m-close-m1/baseline`；未修改旧运行目录。
候选留存在同级 `candidate` 目录，不作为最终生产结果。

64800 柱 × 48 步的 JAX32/NumPy64 全量复验已完成，CPU 部分耗时
184.72 s，GPU/CPU 物理源 hash 相同。原域门返回 `passed=false`，仍有
**5565 / 64800 = 8.58796%** 柱有效状态超差；64800 柱均有原严格诊断超差，
整数全等且物理边界通过。全部计数、逐诊断最大误差、逐状态最大误差和完整
冰量分解与此前 K2 修后证据完全相同；全部 64800 行柱记录解压后的 SHA256
也一致，见[复现核验](gpu-results/m1-restored-domain-repeatability.json)。
没有把旧输出当作新运行。

失败柱仍为 `transition_sand` 5529 柱与 `cold_snow_loam` 36 柱，最大终态
冰量差 `5.42444015e-4 m³/m³`；5565/5565 个超差冰量元素的温度轨迹项均
大于公式评价与水量项绝对值之和。冻土敏感应用建议用 **float64 / JAX-x64**。
本轮采用 M2 所允许的已量化、全部归因的域限制，不宣称原 K3 域门已变绿。

[全域汇总](gpu-results/m1-restored-domain-summary.json)、
[64800 柱记录](gpu-results/m1-restored-domain-columns.csv.gz)、
[冰量分解](gpu-results/m1-restored-domain-ice-decomposition.json)、
[5565 项逐项分解](gpu-results/m1-restored-domain-ice-rows.jsonl.gz)可独立复核。
全量 NPY 和逐列误差 NPZ 留在上述目录的 `outputs/m1-domain/`，约 629 MiB。

年度复验使用原八柱、每日 48 步、连续 carry，不重置、不调整输入或阈值。
NumPy 检查逐步完整状态；GPU 检查每步全部诊断及每天块末完整状态，
没有宣称 GPU 每一中间步的全层状态都回读过。

| 回退源码后端，365 天 × 8 柱 | 完成状态 | 最大绝对 residual W/m² | 累计水量误差最大绝对值 m |
|---|---|---:|---:|
| NumPy64 | 已完成，通过 | 0.000999923414128 | 9.936184220e-9 |
| JAX32 | 已完成，通过 | 0.007209777832031 | 6.691502617e-5 |
| JAX64 | 已完成，通过 | 0.000999923414071 | 9.936178716e-9 |
| CUDA32 | 已完成，通过 | 0.007209777832031 | 6.691545038e-5 |

全年稳定性不替代跨后端逐值 parity 或整柱焓守恒验证，也不能抵销其他
收尾门的失败。原始日志：[NumPy](gpu-results/m1-restored-numpy-365.jsonl)、
[JAX32](gpu-results/m1-restored-jax32-365.jsonl)、
[JAX64](gpu-results/m1-restored-jax64-365.jsonl)、
[CUDA](gpu-results/m1-restored-cuda-365.jsonl)。

2026-10-13 **08:20:06 EDT / 12:20:06 UTC** 全部完成。每后端都是连续
17520 步，365 个日记录全部通过；共 **1460 份每日状态 dump** 的 SHA256
与日志逐份核验一致，四后端启动及结束时的全部生产源 hash 都与回退基线一致。
最终[核验报告和 checksum](gpu-results/m1-restored-validation.json)、
[各步骤退出码](gpu-results/m1-exits.txt)保留完整复核结果。
GPU0 已释放，没有重启或截短任何年度运行。

独立复现使用新输出路径；先按仓库环境说明设置 JAX/CUDA 环境：

```bash
python benchmarks/forensic_gpu_domain.py gpu --directory outputs/recheck-domain
python benchmarks/forensic_gpu_domain.py cpu --directory outputs/recheck-domain
python benchmarks/forensic_gpu_domain.py summarize --directory outputs/recheck-domain
python benchmarks/forensic_gpu_domain.py decompose --directory outputs/recheck-domain
python benchmarks/check_annual_numpy.py --days 365 --output outputs/recheck-numpy.jsonl
python benchmarks/check_annual_gpu.py --backend jax32 --days 365 --output outputs/recheck-jax32.jsonl
python benchmarks/check_annual_gpu.py --backend jax64 --days 365 --output outputs/recheck-jax64.jsonl
python benchmarks/check_annual_gpu.py --backend cuda --days 365 --output outputs/recheck-cuda.jsonl
```

原严格域 `summarize` 预期返回非零；继续运行 `decompose` 是保留归因证据，
并非忽略失败后将该原门改成通过。
