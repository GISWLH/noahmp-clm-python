# GPU 性能基准

基准脚本是 `benchmarks/bench_gpu.py`，每个后端先预热 3 次，再对 measured
runs 取中位数；指标为每柱每步微秒和柱·步/秒。计划规模为 100、1,000、10,000、
100,000、300,000 柱，每柱 48 步。

## 当前机器

- CPU：Windows 11 主机（具体型号由脚本运行时的 `platform.processor()` 输出）
- GPU：NVIDIA GeForce RTX 4070，12 GB
- JAX：Windows CPU wheel（没有 CUDA backend）
- CuPy：14.2.0，CUDA runtime 12.09，可见 RTX 4070
- CUDA C 编译器：`nvcc` 不可用

## 实测记录

2026-10-12 在本机以 `N=100`、48 步、预热 3 次、measured 1 次运行的实际结果：

| 后端 | 总耗时 (s) | µs/(柱·步) | 柱·步/s | 状态 |
|---|---:|---:|---:|---|
| numpy 单核 | 2.3524 | 490.08 | 2,040 | measured |
| JAX CPU x64 | 0.0654 | 13.63 | 73,351 | measured |
| CuPy | — | — | — | unavailable：缺少 CUDA headers |

JAX CPU 在这个小规模上约为 numpy 的 36×；这不是 GPU 加速数字。完整五档规模
尚未在本机整夜跑完，不能填入估算值。运行下面命令会把每个后端实际结果打印到
终端，初始化或硬件失败会明确显示 `unavailable`：

```powershell
$env:PYTHONPATH="src"
python benchmarks/bench_gpu.py --sizes 100 1000 10000 100000 300000 --steps 48
```

阶段 D 的加速目标（GPU 相对单核 numpy 在 `N=100000` 达到 100×）需要在
CUDA-enabled JAX 或手写 kernel 可用的 Linux/xuanze 环境复测；当前 Windows
JAX wheel 不提供该数字。
