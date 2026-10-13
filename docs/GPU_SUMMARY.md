# 阶段 3 GPU 迁移总结

K1仍红：569项已全量复现并取证，全部来自无雪暖湿降雨配置；double割线+F原型将较大反例降为0，但原门仍有137项residual及H/G超差，尚未采纳生产或改变容差。

K2/K3仍待最终验收：K2基线薄雪边界bug已源头修复且NumPy365天通过，GPU年度/回归运行中；K3已完成64800柱全量NumPy64取证，存在5565柱有效状态超差，不能由旧15柱样本代替。

真实数据smoke进行中：已找到NAS上CN-Cng FLUXNET2015真实30天连续强迫，正在同源NumPy/GPU与LE/H观测对照；N300000基线未完成且监督器保持原样，所有红灯关闭和全量pytest全绿前禁止合并main。

当前任务为K1–K7；J1–J5已执行，双卡、profiling与计时审计均冻结，不重复。
首先交付[569项聚类](GPU_RESIDUAL_FORENSICS.md)。未经Wang明确确认，不改变
原H4容差或合并条件；2e-3提案已证伪，不再作为待批准方案。

## 已交付

- `docs/GPU_ENV.md`：保留 Windows 历史环境，并记录 xuanze 双 RTX 4090、nvcc 12.4、Linux venv 和 CUDA 库加载修正。
- `docs/STATE_SPEC.md` 与 `gpu/pack.py`：70 槽 float64、33 槽 int32、8 槽强迫 ABI；雪层有效数由整型控制位保存；50 个随机柱 round-trip 逐字段一致。
- `gpu/jax_core.py`：固定数组、固定三层雪、固定 25 次割线和 20 次 MOST；四层 Thomas 手工展开；方案用 `lax.switch`；批量入口采用模块级 `jit(vmap(scan))`。
- `benchmarks/bench_gpu.py` 与 `docs/BENCHMARK.md`：保留旧五档 GPU 原始计时并标为待 checksum 复核；新基准强制消费全部输出、同步归约回读并与 JAX 对照。
- `gpu/cupy_core.py`：CuPy ABI 对照适配器。它明确把普通版参考物理放在 host 上，不能当作 GPU 性能实现。
- `gpu/cuda/column_kernel.cu`、`build.sh`、`loader.py`：1 thread/column、SoA、grid-stride、完整 LE/雪/冻土/水热物理及12字段35值分发；nvcc零warning，同精度门通过，原NumPy64 H4失败证据保留。
- `benchmarks/check_gpu_options.py`：32 字段范围审计、24 步选项抽查、一日 JAX x32/x64 累积精度；占位和未实现选项明确区分。

## 阶段 3b 验收进度

| 阶段 | 状态 |
|---|---|
| 访问门槛 | 通过：2026-10-12 EDT，SSH + GPU + nvcc |
| H1 Linux venv / JAX-CUDA / CuPy | 通过：JAX 0.11.2 CudaDevice、CuPy 14.2.0 实际设备运算 |
| H2 8 柱 × 48 步 float64 parity | 通过：实际 CUDA 输出；相关回归 7 passed，原容差不变 |
| H3 完整默认物理 CUDA kernel | 通过编译：nvcc 12.4.99，sm_89，零 warning；NVRTC 实际运行 |
| H4 N=1024 CUDA parity | 红灯：原NumPy64门仍失败；CUDA/JAX32的1/48步全部诊断及最终状态逐位一致，同精度1e-6门通过，原容差保留 |
| H5 五档48步GPU基准 | 当前共同源码15组完整输出门通过；N100000对单核三后端均达到100×计时目标，CUDA仍标记H4未通过；N300000的CPU三次预热完成、正式测量中 |
| H6 选项矩阵与 x32/x64 精度 | 当前共同源码92 passed、11 unsupported_expected；35物理值/55 ABI-only/2固定值通过，8场景日累计ET/H相对差均<1e-4 |
| H7 全量测试与合并 | 当前冻结版134 passed / 2 failed / 0 skipped；两项均原NumPy64 H4，禁止合并main |

## J1–J5 新证据

- [GPU_NUMERICS.md](GPU_NUMERICS.md)：运算树、常量折叠、FMA/除法 lowering、MOST与归约的因果探针；N=1输入所有权修复前失败、修后通过，nvcc零warning。
- [GPU_TIMING_AUDIT.md](GPU_TIMING_AUDIT.md)：完整状态/每步诊断的device归约、host读回、JAX逐值门和异质全N敏感性；11项GPU验收仪器测试通过，失败不计measured。
- [GPU_TOLERANCE_PROPOSAL.md](GPU_TOLERANCE_PROPOSAL.md)：7488个普通版求根的局部量化探针；候选residual绝对2e-3尚未经用户确认，也尚不能覆盖所有已知场景。
- [GPU_J4_PROFILE.md](GPU_J4_PROFILE.md)：当前CUDA被采首步4347 FLOP/柱步、2.825 TFLOP/s，三次预热及被采运行全部输出逐位对齐JAX32；不能外推48步，另含JAX精度热点证据。
- [GPU_BATCH_SHAPE.md](GPU_BATCH_SHAPE.md)、[GPU_EXTENDED_VALIDATION.md](GPU_EXTENDED_VALIDATION.md)：固定四层根系归约消除真实列的跨shape复现，64800域15样本对N1均通过；NumPy64域门和年度第3天稳定性仍失败。
- [CUDA_OPTION_IMPLEMENTATION.md](CUDA_OPTION_IMPLEMENTATION.md)：CUDA的35值逐字段、1024柱混排和1728组径流双出口探针通过；18个ABI-only字段仅接受默认，未声称全笛卡尔积覆盖。
- [GPU_OCCUPANCY.md](GPU_OCCUPANCY.md)：256/128/64三配置全部通过严格输出门，计时波动较大，生产保留256。
- 当前JAX双卡四组全部通过完整输出门；N100000/64800的x32加速为1.04682×/1.08759×、x64为1.62227×/1.74957×，均未达1.9×，详见[BENCHMARK.md](BENCHMARK.md)。手写CUDA双卡未测。

## 待完成工作

1. 按 [GPU_BLOCKERS.md](GPU_BLOCKERS.md) 修复 float32 数值条件问题，重新通过原严格 H4；不能隐藏 residual 或放宽容差。
2. 补齐大规模 NumPy 单核三次预热/三次计时，更新 [BENCHMARK.md](BENCHMARK.md) 的完整实测比值。
3. H2+H4+H5 全绿且全量 pytest 通过后，才勾选阶段 3 并合并 main。
