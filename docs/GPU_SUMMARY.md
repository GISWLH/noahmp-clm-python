# 阶段 3 GPU 迁移总结

可用，带[已知限制](KNOWN_LIMITATIONS.md)：M1增量候选因同精度回归已逐字节回退，生产物理保持b954d90，M2合成1024×48门通过，原K1/H4与K3红灯保留（域状态超差5565/64800=8.58796%且全部已归因）。

EHPC串行/4进程/两节点MPI、xuanze核心门与N100000校验计时、HF ZeroGPU JAX1024×48均完成；四站各30天JAX64通过且观测r/bias/RMSE已报告，ERA5因缺下行辐射和总降水记未测，均不宣称物理已验证。

全量pytest为147 passed / 2 failed / 0 skipped（仍为原H4两项）；M4同量化输入canw通过率99.201389%，仍低于原99.5%门，Wang已明确接受该差异并验收v0.2.0收尾版，授权合并main及发布标签，见[验收记录](GPU_ACCEPTANCE.md)。

本轮按Wang的M0–M5收尾，先提交[固定M2准则](GPU_CLOSEOUT_CRITERIA.md)再运行；
原H4不再单独决定合并；M4数值失败保留，发布依据为Wang后续明确授权的验收例外。
Wang明确接受表面能量闭合、
整柱融尽后的222784.419 J/m²热量路由缺口作为公开限制。双卡/profiling保持冻结。

## M阶段收尾交付

- [GPU_DELTA_ATTEMPT](GPU_DELTA_ATTEMPT.md)：一次ΔT实现、新增同精度回归、完整候选patch和逐字节回退；64800域复验与四后端365天全部完成通过，1460份每日dump hash核验一致。
- [GPU_CLOSEOUT_RESULTS](GPU_CLOSEOUT_RESULTS.md)：1024×48逐步全状态/诊断、153项近空canw精确归因及独立复算；各字段≥99.5%、residual中位 .000293732/p99 .00181961 W/m²，表面/水量门通过。
- [REAL_DATA_CHECKS](REAL_DATA_CHECKS.md)：4站完整30天、实际QC及LE/H指标；同输入canw46/5760超差，CN-Cha39/1440；全量归因仍不更改硬门。初表51项含输入量化差异，另保留为端到端结果。
- [ENV_TESTS](ENV_TESTS.md)：EHPC两作业全输出逐位一致；HF实际Blackwell MIG2g.48gb与xuanze完整float32轨迹checksum一致；用户授权删除4个最旧ZeroGPU Space后完成独立演示。
- [BENCHMARK](BENCHMARK.md)：最终物理N100000×48含校验中位JAX64 22.042122s、JAX32 3.227870s、CUDA .203611s；旧五档及N300000 CPU来源限制继续保留。
- [最终pytest日志](gpu-results/m-final-pytest.log)及[源码/环境元数据](gpu-results/m-final-pytest-metadata.json)：GPU强制启用、147通过/2原H4失败，运行前后源码hash一致；[H6](gpu-results/m-final-options.json)为92通过、11预期未实现。
- Windows git归档换行曾使build.sh失败，已添加`.gitattributes`固定shell/PBS为LF；[原失败](gpu-results/m-final-nvcc-crlf.log)和[原参数重编译成功](gpu-results/m-final-nvcc-lf-metadata.json)均保留，仍有1条既有未使用snow_flux警告。

下方为K/J阶段历史交付，旧“全量原H4全绿才合并”等规则已由本轮Wang的M2指令取代；
它们的原始数字和失败证据不删除。

## K阶段交付

- [GPU_RESIDUAL_FORENSICS.md](GPU_RESIDUAL_FORENSICS.md)：569条完整上下文、全部轨迹、诊断×状态聚类和三变体真实GPU对照；独立审计重算全部计数、生成源码hash和成本，结果一致。
- [GPU_ANNUAL_FIX.md](GPU_ANNUAL_FIX.md)：原失败单柱/强迫切片、隐式雪顶边推导、普通版与两GPU实现、四后端全年日志；完整融尽的既有剩余能量未路由，表面残差通过不等于整体热守恒已验证。
- [GPU_DOMAIN_FIX.md](GPU_DOMAIN_FIX.md)：修前/修后全64800柱的NumPy64对照、真实聚类图、温度/水量/冰量分解与最小trace；K2消除2.35 K雪面尾部，但冻土AHC敏感性及厚雪下均温土壤漂移仍待修。
- [FLUXNET_PLAN.md](FLUXNET_PLAN.md)与[smoke脚本](../examples/fluxnet_smoke.py)：真实CSV映射、来源hash、QC筛选、后端parity及观测指标、独立Phase4计划。
- [全量pytest](gpu-results/k-final-pytest.log)：实际GPU强制开启；生产源码与[冻结测试快照](gpu-results/k-final-pytest-metadata.json)一致。旧J阶段性能只属于旧源码，未重做双卡、profiling或JAX计时审计。

## 已交付

- `docs/GPU_ENV.md`：保留 Windows 历史环境，并记录 xuanze 双 RTX 4090、nvcc 12.4、Linux venv 和 CUDA 库加载修正。
- `docs/STATE_SPEC.md` 与 `gpu/pack.py`：70 槽 float64、33 槽 int32、8 槽强迫 ABI；雪层有效数由整型控制位保存；50 个随机柱 round-trip 逐字段一致。
- `gpu/jax_core.py`：固定数组、固定三层雪、固定 25 次割线和 20 次 MOST；四层 Thomas 手工展开；方案用 `lax.switch`；批量入口采用模块级 `jit(vmap(scan))`。
- `benchmarks/bench_gpu.py` 与 `docs/BENCHMARK.md`：保留旧五档 GPU 原始计时并标为待 checksum 复核；新基准强制消费全部输出、同步归约回读并与 JAX 对照。
- `gpu/cupy_core.py`：CuPy ABI 对照适配器。它明确把普通版参考物理放在 host 上，不能当作 GPU 性能实现。
- `gpu/cuda/column_kernel.cu`、`build.sh`、`loader.py`：1 thread/column、SoA、grid-stride、完整 LE/雪/冻土/水热物理及12字段35值分发；nvcc零warning，同精度门通过，原NumPy64 H4失败证据保留。
- `benchmarks/check_gpu_options.py`：32 字段范围审计、24 步选项抽查、一日 JAX x32/x64 累积精度；占位和未实现选项明确区分。

## J阶段冻结时的验收进度（K修后状态见顶部及下方K交付）

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
- [GPU_TOLERANCE_PROPOSAL.md](GPU_TOLERANCE_PROPOSAL.md)：7488个普通版求根的局部量化历史推导；2e-3候选已证伪并撤回，K1未提出新的容差申请。
- [GPU_J4_PROFILE.md](GPU_J4_PROFILE.md)：当前CUDA被采首步4347 FLOP/柱步、2.825 TFLOP/s，三次预热及被采运行全部输出逐位对齐JAX32；不能外推48步，另含JAX精度热点证据。
- [GPU_BATCH_SHAPE.md](GPU_BATCH_SHAPE.md)、[GPU_EXTENDED_VALIDATION.md](GPU_EXTENDED_VALIDATION.md)：固定四层根系归约消除真实列的跨shape复现，64800域15样本对N1均通过；NumPy64域门和年度第3天稳定性仍失败。
- [CUDA_OPTION_IMPLEMENTATION.md](CUDA_OPTION_IMPLEMENTATION.md)：CUDA的35值逐字段、1024柱混排和1728组径流双出口探针通过；18个ABI-only字段仅接受默认，未声称全笛卡尔积覆盖。
- [GPU_OCCUPANCY.md](GPU_OCCUPANCY.md)：256/128/64三配置全部通过严格输出门，计时波动较大，生产保留256。
- 当前JAX双卡四组全部通过完整输出门；N100000/64800的x32加速为1.04682×/1.08759×、x64为1.62227×/1.74957×，均未达1.9×，详见[BENCHMARK.md](BENCHMARK.md)。手写CUDA双卡未测。

## K阶段历史后续清单（本轮处置见顶部）

1. K1原混合割线原型仍不满足完整精度/成本条件；K3需单独修复土热轨迹及相变热容敏感性，不能把它们都归为相变残差或调阈值解决。
2. N300000原监督器已完成并补表，六次checksum一致；保持CPU启动hash缺失的来源限制，不以这份旧物理基线冒充K2新源码性能。
3. 合并门为K1（混合修复通过，或Wang明确确认且实际通过的分档）+K2+K3全绿+全量pytest全绿；当前留在gpu/migration，main不受影响。
4. 完整FLUXNET Phase 4是下一个独立任务；本轮仅真实数据30天smoke与计划，不宣称物理已验证。
