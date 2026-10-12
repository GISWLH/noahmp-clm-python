# 阶段 3 GPU 迁移总结

当前状态：已完成环境侦察、阶段 A 状态规约、阶段 B JAX 核心、阶段 D 基准入口、阶段 E CuPy 正确性锚点和阶段 F CUDA SoA 源码；普通版仍未改动。

性能数字：本机 `N=100 × 48` 实测 numpy 单核 490.08 µs/柱步（2,040 柱步/s），JAX CPU x64 13.63 µs/柱步（73,351 柱步/s）；没有可报告的 GPU kernel 数字。

阻塞项：Windows JAX 0.11.2 CPU wheel 在混合选项/重复形状的 XLA HLO verifier 中会进程级中止；本机没有 nvcc，CuPy 批量对照缺 CUDA headers，因此 parity、CUDA kernel 和 N≥1e5 性能必须在 xuanze/Linux 复测。

## 已交付

- `docs/GPU_ENV.md`：记录 RTX 4070、驱动/CUDA、JAX/CuPy 安装路径及路线选择。
- `docs/STATE_SPEC.md` 与 `gpu/pack.py`：70 槽 float64、33 槽 int32、8 槽强迫 ABI；雪层有效数由整型控制位保存；50 个随机柱 round-trip 逐字段一致。
- `gpu/jax_core.py`：固定数组、固定三层雪、固定 25 次割线和 20 次 MOST；四层 Thomas 手工展开；方案用 `lax.switch`；批量入口采用模块级 `jit(vmap(scan))`。
- `benchmarks/bench_gpu.py` 与 `docs/BENCHMARK.md`：五档规模入口、预热/中位数、实际小规模记录。
- `gpu/cupy_core.py`：CuPy ABI 对照适配器。它明确把普通版参考物理放在 host 上，不能当作 GPU 性能实现。
- `gpu/cuda/column_kernel.cu`、`build.sh`、`loader.py`：1 thread/column、SoA、grid-stride、固定 Thomas/割线；文件头和测试均标记 `[未实测]`。

## 还需要在 xuanze 完成

1. 安装 CUDA-enabled JAX，启用 x64，运行 `tests/test_gpu_parity.py` 的 8 柱 × 48 步混合选项门槛；记录 ET、感热、表层地温的逐日偏差。
2. 用 CUDA Toolkit 编译 kernel，运行 N=1024 的 float32 parity；修正目前 kernel 中为 ABI/求解器验证保留的简化通量项后再测 1e-4 容差。
3. 执行 `benchmarks/bench_gpu.py` 的 100--300000 柱五档，补齐 GPU 吞吐和相对单核加速比。
4. parity 和性能数据真实落盘后，再把阶段 3 总体项在 `ROADMAP.md` 勾选完成。
