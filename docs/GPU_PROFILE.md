# xuanze JAX64 单轮性能采样

2026-10-13 00:37–00:39 EDT，在 xuanze 的物理 GPU 1（RTX 4090）完成一次
`N=100000 × 48` 的 JAX float64 Nsight Systems 采样。**实测热点是大量重复的
loop kernel：`loop_or_select_fusion_15` 调用 378240 次，累计 20.484 秒，占
GPU kernel 累计时间的 94.7%。** 源码中的固定迭代及批量条件分支可能导致了
这一执行形态；这是待 HLO/源位置关联确认的推断，不能据此断言 FP64 算力或
显存带宽已经饱和。

本轮只采样和定位，没有进行性能优化；H4 严格 parity 仍为红灯，见
[GPU_BLOCKERS.md](GPU_BLOCKERS.md)。profiling 结果不能替代 parity 或作为
合并 main 的依据。

## 范围与复现

独立脚本 [profile_gpu.py](../benchmarks/profile_gpu.py) 复用
`bench_gpu._gpu_runner("jax64", ...)` 与相同的默认柱、48 步合成强迫，未修改
JAX、CUDA kernel 或基准执行逻辑。JAX 0.11.2、jaxlib 0.11.2、Python 3.12.2；
Nsight Systems 2023.4.4.54。H6 作业退出并释放 GPU 后才启动采样，采样退出后
才将 GPU 交还全量 pytest；未与本任务的其他 GPU 作业并行。

```bash
cd /home/lhwang/work/noahmp-phase3b/repo
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src
/usr/local/cuda-12.4/bin/nsys profile \
  --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=profile_output/jax64-n100000-t48 \
  /home/lhwang/venvs/noahmp-phase3b/bin/python benchmarks/profile_gpu.py \
  --n 100000 --steps 48 --warmups 3 \
  --output profile_output/jax64-n100000-t48.jsonl

/usr/local/cuda-12.4/bin/nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format json --output profile_output/jax64-n100000-t48-summary \
  profile_output/jax64-n100000-t48.nsys-rep
```

输入已放在 GPU；三次同步预热后，`cudaProfilerStart/Stop` 只包围一次同步
调用。JIT 编译、预热、最初的主机到设备输入传输及最后一个柱步的少量结果回读
都在捕获范围外。保留全部 `(100000, 48, 12)` 诊断输出，调用结束执行
`jax.block_until_ready`。这里测的是与 GPU 基准一致的设备驻留调用边界。
复测时使用新的输出名称，避免覆盖证据；脚本会拒绝覆盖现有 JSONL。

原始约 46 MB `.nsys-rep` 和导出的 SQLite、CSV、工具日志保留在远程：
`/home/lhwang/work/noahmp-phase3b/repo/profile_output/`。
大二进制不纳入 Git；紧凑原始统计链接见下文。

## 测量结果

### 耗时及 GPU 热点

原始运行记录：[jax64-profile-run.jsonl](gpu-results/jax64-profile-run.jsonl)。

| 指标 | 实测 |
|---|---:|
| 第 1 次预热，含编译 | 26.439891 s |
| 第 2 / 3 次预热 | 22.379216 / 22.550605 s |
| 单次捕获调用墙钟 | 23.121362 s |
| 捕获调用每柱步 | 4.816951 µs |
| 无 profiler 的 H5 三次中位数 | 22.003309 s |
| 捕获调用相对该中位数增加 | 约 5.08% |
| GPU kernel 种类 / 总执行次数 | 23 / 806114 |
| GPU kernel 累计执行时间 | 21.637389 s |

单轮 profiler 时间不替换 [BENCHMARK.md](BENCHMARK.md) 的无 profiler 基准。
约 5.08% 的差额包含观测开销和运行间变化，不能全部归因于某一种因素。

原始 kernel 统计：[jax64-profile-kernels.json](gpu-results/jax64-profile-kernels.json)。

| Kernel | 次数 | 累计时间 | 单次中位数 | kernel 时间占比 |
|---|---:|---:|---:|---:|
| `loop_or_select_fusion_15` | 378240 | 20.483877 s | 52.001 µs | 94.7% |
| `loop_add_fusion_3` | 399456 | 0.392078 s | 0.992 µs | 1.8% |
| `loop_select_subtract_fusion_1` | 6240 | 0.208489 s | 32.160 µs | 1.0% |
| `loop_select_subtract_fusion` | 6240 | 0.155577 s | 24.832 µs | 0.7% |

### CPU / CUDA 调度

原始 API 统计：[jax64-profile-api.json](gpu-results/jax64-profile-api.json)。

| API | 次数 | 累计 API 持续时间 | 单次中位数 |
|---|---:|---:|---:|
| `cuLaunchKernel` | 806114 | 20.552543 s | 5.652 µs |
| `cuGraphLaunch` | 19249 | 0.586787 s | 46.770 µs |
| `cuMemcpyDtoDAsync_v2` | 13152 | 0.339850 s | 30.615 µs |

这些是主机 API 持续时间，可能包含驱动等待，并与 GPU 执行重叠；不能把它们
加到 GPU kernel 时间上，也不能将 `cuLaunchKernel` 的 20.553 秒直接称为纯
Python/CPU 计算开销。此次禁用了 CPU 栈采样和调度跟踪，未定位 Python 函数热点。

脚本同时记录了进程 `resource.getrusage` 差值（用户 CPU 15.354 秒、系统 CPU
10.239 秒、自愿切换 846 次、非自愿切换 34 次）。该差值包住
`cudaProfilerStart/Stop`，包含 Nsight 停止捕获与导出开销，**不能用于计算目标
调用的 CPU 利用率**；保留原始值以便审查观测边界。

### 内存与数据移动

设备内存数字来自 JAX allocator 的 `memory_stats()`，不是全卡显存占用，也不
包含其他进程。MB/GB 在本表采用十进制。

| 指标 | 实测或精确数组大小 | 重复/临时数据说明 |
|---|---:|---|
| 输入状态和选项 | 69.200 MB | 基准把默认柱复制到 100000 个独立柱 |
| 共享强迫输入 | 3072 B | 48 × 8 个 float64；初始输入传输在捕获范围外 |
| 返回数组逻辑大小 | 530.000 MB | 最终状态、选项及全部柱步诊断 |
| JAX 捕获前 bytes_in_use | 80.313 MB | 预热结果已释放 |
| JAX 捕获后 bytes_in_use | 686.384 MB | 此时仍持有返回数组 |
| JAX peak_bytes_in_use | 1862.761 MB | 整个进程生命周期内峰值，含预热 |
| JAX pool_bytes | 2283.799 MB | allocator 池容量；不等于当前活跃数组大小 |
| Device-to-Device 复制 | 13152 次 / 10.608002 GB | 反复设备内复制，累计 18.767245 ms |
| Host-to-Device / Device-to-Host 复制 | 捕获范围未记录 | 不代表初始输入或结果导出免费 |
| 跨 GPU 通信 | 本工作负载仅使用一个设备 | 未运行 NCCL 或 GPU–GPU 交换 |

设备复制原始统计：[时间](gpu-results/jax64-profile-memory-time.json)、
[字节数](gpu-results/jax64-profile-memory-size.json)。按累计字节数除以累计复制
时间可得约 565 GB/s，但这是这些 D2D 复制的统计比值，不是整个应用达到的
显存带宽。此次未采 SM 利用率、FP64 指令吞吐、占用率或 DRAM 硬件计数器，
不报告这些指标，也不宣称 memory-bound 或 compute-bound 已被证实。

## 解释与后续优先级

**证据：** 时间高度集中于一个 loop kernel，且单次调用产生约 80.6 万次 kernel
启动；显式 D2D 复制的 18.77 ms 远小于 kernel 累计 21.64 s。
因此下一轮应优先定位并减少重复 loop 工作，不能先把 PCIe 传输当作主瓶颈。

**推断：** JAX 源码有 25 次割线、130 段回退扫描、40 次二分及每次通量内部
20 次 MOST 迭代。状态冻结仍以数组选择实现，且批量 `vmap` 与条件分支的组合
可能执行额外分支工作。当前 trace 的 kernel 名称与次数和这一方向一致，但没有
HLO 源位置映射，尚不能断言 `loop_or_select_fusion_15` 就对应其中某一个具体循环。

1. 下一轮先将主热点关联到 HLO/源码，确认收敛冻结及回退扫描是否仍对整个批量
   执行，然后评估保持数值语义的批量早停/条件处理，预期最可能减少总工作量。
2. 若必须保留相同迭代次数，再评估循环融合与调度方式；从 API 累计时间直接
   推算收益不可靠，仍需新的无 profiler 对照基准和原 parity 门。
3. 最后再用硬件计数器判断 FP64 吞吐、占用率或带宽是否限制单个主热点。
   本轮不运行额外 ncu 回放，不在夜间继续调优。

## Instrumentation changelog

本轮只新增独立采样入口和证据文档；生产物理模块与原基准未因 profiling 改动。

| 文件 | 变更 | 内容 | 行位置 |
|---|---|---|---|
| `benchmarks/profile_gpu.py` | 新增 | 复用现有 runner；三次预热、单次 CUDA profiler 区间、同步计时、设备内存和进程统计、少量区间外回读 | 全文件 |
| `docs/GPU_PROFILE.md` | 新增 | 方法、边界、结果、证据与推断、后续方向和本清单 | 全文件 |
| `docs/gpu-results/jax64-profile-run.jsonl` | 新增 | 原始环境、预热、调用时间、内存和结果形状 | 全文件 |
| `docs/gpu-results/jax64-profile-kernels.json` | 新增 | nsys 原始 kernel 汇总 | 全文件 |
| `docs/gpu-results/jax64-profile-api.json` | 新增 | nsys 原始 CUDA API 汇总 | 全文件 |
| `docs/gpu-results/jax64-profile-memory-time.json` | 新增 | nsys 原始设备复制耗时汇总 | 全文件 |
| `docs/gpu-results/jax64-profile-memory-size.json` | 新增 | nsys 原始设备复制字节数汇总 | 全文件 |
| 远程 `profile_output/jax64-n100000-t48*` | 新增，未入 Git | `.nsys-rep`、SQLite、工具日志、CSV/JSON 导出 | 不适用 |

这些独立 instrumentation 文件和远程 trace 可以在完成审查后按需清理；无需撤销
生产代码改动。本次保留全部证据。
