# J4 硬件计数器与 JAX 精度性能差

2026-10-13 xuanze 物理 GPU0，RTX 4090，N=100000、48 步。旧 CUDA 的约7 ms
已经由 Nsight Systems 证明包含48次真实 kernel 执行；本轮进一步取得实际指令、
访存和 occupancy。**证据不支持“660 TFLOPS，因此必然 DCE”的推断。**
下方历史节保留旧kernel诊断；最新共同源码的计数在第一节。所有计数均不替代H4。

## J3共同源码的最新首步计数（06:37 UTC）

冻结`/home/lhwang/work/noahmp-phase3b/j3-profile-final`，CUDA源码`7640b062…`、
JAX源码`0e296691…`；物理GPU0，PCI `0000:5E:00.0`。当前
[profile_cuda_kernel.py](../benchmarks/profile_cuda_kernel.py)先生成同N的JAX32
参考，三次完整48步预热均执行全输出归约/逐值门，捕获后再校验一次；四次全部
通过且各字段差值为零。实际导入包路径限定在冻结src，已加载CUDA文本与磁盘
规范化hash相同，完整源文件首尾hash一致。探针使用同质默认输入；全N异质
预检由另外的五档正式基准和block实验覆盖，不能说本探针也执行了该预检。

ncu仅采捕获调用的**第一个step_columns**，不是48步累计。以下来自同一次
counter报告：

| 指标 | 实测 |
|---|---:|
| FP32 FADD / FMUL / FFMA线程指令 | 95,400,000 / 114,500,000 / 112,400,000 |
| FADD + FMUL + 2×FFMA | 434,700,000 FLOP |
| 每柱步上述浮点操作 | 4,347 FLOP |
| DRAM读 / 写 | 44.702976 / 64.177280 MB |
| 被采kernel时间 | 153.888 µs |
| 算术强度，实际DRAM层级 | 3.992459 FLOP/B |
| 上述浮点操作吞吐 | 2.824782 TFLOP/s |
| DRAM字节 / kernel时间 | 707.529216 GB/s |
| 寄存器 / 寄存器限制的同驻block数 | 96/thread / 2个block每SM |
| 实际occupancy | 28.77% |

FLOP不包含SFU或整数指令，不能把一次首步工作量乘48当作完整轨迹指令数，
也不能把旧版6460 FLOP/柱步移到新版。当前首步的带宽活动显著，寄存器限制
同驻；这仍不足以证明整48步唯一受带宽限制。三档block完整计时已做一轮，
波动不足以支持切换配置，生产保留256，见[配置实验](GPU_OCCUPANCY.md)。

原始[counter CSV](gpu-results/j4-new-cuda-ncu.csv)、
[输出门与执行源码](gpu-results/j4-new-cuda-ncu.json)、
[单位及公式汇总](gpu-results/j4-new-cuda-ncu-summary.json)、
[执行日志](gpu-results/j4-new-cuda-ncu.log)均保留。CSV单位为nsecond/byte/inst，
汇总用`FADD+FMUL+2*FFMA`、`FLOP/(read+write)`及`FLOP/time`直接计算。
完整ncu-rep留在冻结目录profile_output中。

继续使用`--clock-control none --cache-control none`，ncu相应警告保留。仅使用
已配置凭据为这一命令申请sudo，不改驱动策略。采样前两卡利用率均为0%，
先前约5GB的其他作业已结束；两卡仍有常驻显存，不能声称长期独占整机。
[开始](gpu-results/j4-new-cuda-gpu-before.csv)/
[结束](gpu-results/j4-new-cuda-gpu-after.csv)硬件状态保留。CPU基线同时运行。
这次同精度门通过，但原NumPy64 H4仍红；profile仍标记diagnostic_only。

当前共同源码正式基准的N100000含校验时间为x64 21.355447s、x32 3.807727s，
比值 **5.60845×**，并非固定7×。下方JAX kernel热点证据来自所列历史快照，
可支持精度运算成本与大量短kernel调度共同作用的解释，但不能把旧trace时间
当作此次新版本的分项计时，也未证明本次纯由FP64峰值吞吐限制。

## 旧kernel固定样本的实际算术强度

使用旧 CUDA 源码快照
`/home/lhwang/work/noahmp-phase3b/timing-audit-original-20261013T050056Z`，
源码 hash 见 [J2 清单](gpu-results/j2-old-cuda-sourcehash.json)。独立入口
[profile_cuda_kernel.py](../benchmarks/profile_cuda_kernel.py) 先做3次完整48步预热，
随后捕获一次完整调用。ncu 仅选择该调用的**第一个** `step_columns`，回放3次
收集下表的 counters；调用后的完整输出归约检查 finite/形状，但不冒充 JAX parity。

| 同一次 counter 样本 | 实测 |
|---|---:|
| 生效 FP32 FADD 线程指令 | 84,800,000 |
| 生效 FP32 FMUL 线程指令 | 65,800,000 |
| 生效 FP32 FFMA 线程指令 | 247,700,000 |
| `FADD + FMUL + 2×FFMA` | 646,000,000 FLOP |
| 每个柱步的上述浮点操作 | 6,460 FLOP |
| DRAM 读 / 写 | 44.534912 / 43.081728 MB |
| 被采 kernel 时间 | 132.448 µs |
| 算术强度，FLOP / 实际 DRAM 字节 | 7.37303 FLOP/B |
| 上述浮点操作吞吐 | 4.87739 TFLOP/s |
| DRAM 字节 / kernel 时间 | 661.517 GB/s |

FLOP 计数不包含 transcendental 特殊函数、整数运算或访存指令，亦未把一次
特殊函数随意当作固定数量的 FLOP。它是这个真实首步的硬件计数，不能外推
所有48步都有相同的迭代次数或分支成本。毫秒耗时本身无法确定 TFLOP/s。
原始 [counter CSV](gpu-results/j4-original-cuda-flops.csv)、
[归约与源码身份](gpu-results/j4-original-cuda-flops.json)、
[汇总](gpu-results/j4-cuda-roofline-summary.json) 可复算所有比值。

另一次10-pass ncu 采样取得：94 registers/thread、256 threads/block、391 blocks、
128 SM、1.53 waves/SM；理论 occupancy 33.33%、实际27.72%，每 SM 寄存器上限
只允许两个 block 同驻。DRAM throughput 为719.84 GB/s、报告最大带宽利用率
71.49%，L2 hit69.35%，ncu roofline 报告约6%的 FP32 峰值。
这组采样时间148.064 µs，与上表是不同回放，不能把两次数字拼成一条测量。
原始 [详细报告](gpu-results/j4-original-cuda-ncu-details.txt)、
[全部 metrics](gpu-results/j4-original-cuda-ncu-raw.csv)。

两次均 `--clock-control none --cache-control none`，不改服务器全局频率/缓存策略。
因此频率、其他用户负载及缓存重用会影响结果，特别是基于 DRAM 字节的 AI。
按 [NVIDIA roofline 定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline)，
AI 要与所用内存层级对应；这里使用实际 DRAM counters，不能拿逻辑数组大小替换。
目前可判断带宽活动显著、寄存器限制驻留；没有足够证据将整48步唯一归因于带宽。

128 threads/block 是值得在数值门通过后测一次的候选：较小 block 可能减少
寄存器分配造成的驻留浪费。occupancy 增加不保证应用更快，因此本轮不修改
生产 launch 配置，也不把 ncu 的理论建议当作实测加速比。

## 计数器权限和复现

普通账号首次采样返回 `ERR_NVGPUCTRPERM`，驱动 `RmProfilingAdminOnly=1`。
随后使用 remote-ssh 已配置账号凭据，仅以 sudo 运行该固定探针；成功取得
counter。没有修改驱动配置、停止其他 GPU 进程或放开全局性能计数器权限。
工具为 Nsight Compute 2024.1.0。

在上述快照内、相同 CUDA 环境下执行（sudo 凭据不写入命令或仓库）：

```bash
ncu --profile-from-start off --kernel-name regex:step_columns --launch-count 1 \
  --clock-control none --cache-control none \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
  --export profile_output/cuda-flops-new \
  ~/venvs/noahmp-phase3b/bin/python benchmarks/profile_cuda_kernel.py \
  --n 100000 --steps 48 --output ../logs/cuda-flops-new.json
```

完整 `.ncu-rep` 留在快照 `profile_output/`，不入 Git。该硬件诊断即使 finite
通过，也不改变 [GPU_TIMING_AUDIT.md](GPU_TIMING_AUDIT.md) 中严格数值门的红灯。

## JAX x64/x32：重复 MOST 迭代放大了精度成本

新增 [profile_gpu.py](../benchmarks/profile_gpu.py) 的 dtype 参数、优化 HLO 导出
和完整输出回读门，保持原低层 runner。快照
`/home/lhwang/work/noahmp-phase3b/j4-stable-jax-20261013T0513` 使用薄雪修复后的
固定 JAX 源码，两个 dtype 顺序占用同一GPU0，均三次预热后捕获一次。
优化 HLO 中 `fused_or_select.15` 两种精度有**相同操作结构**：5个 divide、
2个 log、1个 power、1个 atan2，及收敛状态选择。这些计算对应 MOST 的
稳定度修正和交换速度迭代；fp32/fp64 的指令实现和成本可能不同。
完整 HLO 留在远程 `profile_output/`；带 hash 的
[热点摘录及直方图](gpu-results/j4-jax-hlo-hotspot.json) 入库。

| 观测 | JAX x32 新采样 | JAX x64 已成功历史采样 |
|---|---:|---:|
| 所有 GPU kernel 次数 | 799,058 | 806,114 |
| 所有 GPU kernel 累计时间 | 1.208286 s | 21.637389 s |
| `loop_or_select_fusion_15` 次数 | 378,240 | 378,240 |
| 该热点累计时间 | 0.757857 s | 20.483877 s |
| 该热点单次中位数 | 2.016 µs | 52.001 µs |
| 该热点占 kernel 累计时间 | 62.7% | 94.7% |

历史 x64 来自 [此前成功的 nsys 采样](GPU_PROFILE.md)，与新采样不是同一次
运行；默认无雪基准未消费本次薄雪修复的分支，HLO 和热点次数也支持结构对照。
新 x64 同步捕获墙钟21.7734 s、完整输出门通过，但 nsys 导入报
`Wrong event order`，重导入仍失败，故**没有**捏造新 x64 kernel 表。
原始 [新 x64 运行](gpu-results/j4-jax64-profile.jsonl)、
[导入失败日志](gpu-results/j4-jax64-reimport.log) 已保留。
新 x32 捕获墙钟4.66214 s，包含 profiler 开销；其无 tracing 的第二/三次预热
分别3.30686/3.41711 s，不应拿单次 profile 替代三次正式测量的中位数。

这轮解释是：固定迭代在批量条件分支中仍形成大量 MOST 重复计算，双精度让
含除法和特殊函数的主热点显著变慢；x32 加快该热点后，近80万次微小 kernel 的
调度成本所占比例上升，所以整调用的精度速度比不会等于一个浮点硬件峰值比。
ncu 确认该设备 FP32/FP64 峰值比为64:1，但未对 JAX 热点收 FP64 饱和度，
因此“单纯 FP64 峰值瓶颈”仍不是已证结论。优先改进方向是减少收敛后无效
循环/未选分支执行，同时维持根与状态冻结语义；不是根据理论峰值推算加速比。

## 仪器化文件

| 文件 | 改动及目的 |
|---|---|
| `benchmarks/profile_gpu.py` | 新增 x32、HLO/cost 导出、完整结果消费；static HLO cost 明确不当作 executed FLOP |
| `benchmarks/profile_cuda_kernel.py` | 独立 bounded CUDA counter 入口，保留完整输出；不改生产 kernel |
| `docs/gpu-results/j4-*profile*`、`j4-jax-hlo-hotspot.json` | 运行和 HLO 摘录；大 HLO/trace 在忽略目录 |
| `docs/gpu-results/j4-original-cuda-*`、`j4-cuda-roofline-summary.json` | 真实 counter、归约、环境与计算比值 |
| 临时 `profile_output/analyze_hlo.py` | 从已导出的 HLO/CSV 提取摘录与操作直方图，未改生产物理 |

本轮没有基于这些性能数据修改 CUDA/JAX 物理、block size 或收敛阈值。
