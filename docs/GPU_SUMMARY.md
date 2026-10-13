# 阶段 3 GPU 迁移总结

当前状态：H2 GPU float64 parity保持通过；CUDA/JAX32默认1024柱×1/48步逐位一致，J3的12字段35值分发联合55项通过，最新全量pytest为134通过、2项原H4失败。

性能数字：当前共同源码五档三后端15组通过完整输出门；N=100000×48的NumPy单核/JAX x64/JAX x32/CUDA为3616.904161/21.355447/3.807727/0.196976 s，相对单核为169.37×/949.89×/18362.13×；CPU启动源码hash缺失的来源限制见基准文档。

验收缺口：原H4对NumPy64仍红灯，修后有569项residual差值超过候选2e-3且H/LE/G也有超差；N300000的CPU仍运行，年度第3天及64800柱x32独立物理门失败，阶段3未完成且禁止合并main。

最新工作入口为 [GPU_J_STAGE.md](GPU_J_STAGE.md)：继续 J1–J5，并提出有推导的
分档容差方案。未经用户明确确认，不改变原 H4 容差或合并条件。

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
| H5 五档48步GPU基准 | 当前共同源码15组完整输出门通过；N100000对单核三后端均达到100×计时目标，CUDA仍标记H4未通过；N300000的CPU未完成 |
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

## 待完成工作

1. 按 [GPU_BLOCKERS.md](GPU_BLOCKERS.md) 修复 float32 数值条件问题，重新通过原严格 H4；不能隐藏 residual 或放宽容差。
2. 补齐大规模 NumPy 单核三次预热/三次计时，更新 [BENCHMARK.md](BENCHMARK.md) 的完整实测比值。
3. H2+H4+H5 全绿且全量 pytest 通过后，才勾选阶段 3 并合并 main。
