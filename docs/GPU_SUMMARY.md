# 阶段 3 GPU 迁移总结

当前状态：xuanze 访问、H1 环境、H2 GPU float64 parity 和 H3 完整默认 CUDA kernel 编译已通过；H6 已实现物理选项的 35 个取值通过 GPU 抽查。

性能数字：已收回 NumPy 单核 `N=10000 × 48` 中位数 364.04924 s；旧 GPU 计时（包括 CUDA 0.00728780 s）进入 J2 checksum/同步审计，暂不作为最终验收证据。

验收缺口：H4 float32 严格 parity 为红灯；H5 大规模单核基线尚在补测，不能报告 N=1e5 的完整加速比，阶段 3 未完成且禁止合并 main。

最新工作入口为 [GPU_J_STAGE.md](GPU_J_STAGE.md)：继续 J1–J5，并提出有推导的
分档容差方案。未经用户明确确认，不改变原 H4 容差或合并条件。

## 已交付

- `docs/GPU_ENV.md`：保留 Windows 历史环境，并记录 xuanze 双 RTX 4090、nvcc 12.4、Linux venv 和 CUDA 库加载修正。
- `docs/STATE_SPEC.md` 与 `gpu/pack.py`：70 槽 float64、33 槽 int32、8 槽强迫 ABI；雪层有效数由整型控制位保存；50 个随机柱 round-trip 逐字段一致。
- `gpu/jax_core.py`：固定数组、固定三层雪、固定 25 次割线和 20 次 MOST；四层 Thomas 手工展开；方案用 `lax.switch`；批量入口采用模块级 `jit(vmap(scan))`。
- `benchmarks/bench_gpu.py` 与 `docs/BENCHMARK.md`：保留旧五档 GPU 原始计时并标为待 checksum 复核；新基准强制消费全部输出、同步归约回读并与 JAX 对照。
- `gpu/cupy_core.py`：CuPy ABI 对照适配器。它明确把普通版参考物理放在 host 上，不能当作 GPU 性能实现。
- `gpu/cuda/column_kernel.cu`、`build.sh`、`loader.py`：1 thread/column、SoA、grid-stride、默认完整 LE/雪/冻土/水热物理；nvcc 和 NVRTC 已编译执行，严格 parity 失败证据保留。
- `benchmarks/check_gpu_options.py`：32 字段范围审计、24 步选项抽查、一日 JAX x32/x64 累积精度；占位和未实现选项明确区分。

## 阶段 3b 验收进度

| 阶段 | 状态 |
|---|---|
| 访问门槛 | 通过：2026-10-12 EDT，SSH + GPU + nvcc |
| H1 Linux venv / JAX-CUDA / CuPy | 通过：JAX 0.11.2 CudaDevice、CuPy 14.2.0 实际设备运算 |
| H2 8 柱 × 48 步 float64 parity | 通过：实际 CUDA 输出；相关回归 7 passed，原容差不变 |
| H3 完整默认物理 CUDA kernel | 通过编译：nvcc 12.4.99，sm_89，零 warning；NVRTC 实际运行 |
| H4 N=1024 CUDA parity | 红灯：首步 residual 超差，48 步近零 H/G 等也超差；原容差保留 |
| H5 五档 48 步 GPU 基准 | 旧 GPU 五档计时待 checksum 复核；NumPy N=10000 已收回，N=100000/300000 在算；暂不宣称 ≥100× |
| H6 选项矩阵与 x32/x64 精度 | 已实现物理 35 值通过，55 ABI-only 值无效应、2 固定值通过、11 未实现值如实记录 |
| H7 全量测试与合并 | 44 passed, 2 failed（H4 的 1/48 步）；保持 gpu/migration，不合并 main |

## 待完成工作

1. 按 [GPU_BLOCKERS.md](GPU_BLOCKERS.md) 修复 float32 数值条件问题，重新通过原严格 H4；不能隐藏 residual 或放宽容差。
2. 补齐大规模 NumPy 单核三次预热/三次计时，更新 [BENCHMARK.md](BENCHMARK.md) 的完整实测比值。
3. H2+H4+H5 全绿且全量 pytest 通过后，才勾选阶段 3 并合并 main。
