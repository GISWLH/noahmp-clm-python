# GPU 性能基准

> 2026-10-13：JAX五档x64/x32的全量device归约→host回读→JAX逐值门已全部通过。
> 下方新表取最终`event=result`，旧计时另存历史节。原H4及年度/x32域验收
> 仍有红灯，checksum不能替代独立物理parity；详见[计时审计](GPU_TIMING_AUDIT.md)。

## J1/J2 旧65c5源码的输出校验计时

以下JAX数字属于根系显式归约修复前的`65c5c3bc…`源码。当前`0e296691…`/
CUDA `7640b062…`共同版本正在独立重跑，不能把旧表当成新版本成绩。

物理GPU0，RTX4090，固定薄雪修复后的JAX源码；每档3次完整预热、3次完整
计时取中位数。每次消费全部70个最终浮点状态、33个整数状态和48×12个逐步
诊断：GPU归约后读回host，同时在device上与同N参考逐值比较，rtol=atol=1e-6，
整数全等。计时前另做完整N的异质输入/敏感性检查，所有列都必须响应。
全src和输入生成器hash由独立sidecar及监督器阶段前后复核。

主表时间包括同步计算、全量归约、逐值验证及host读回。H2D初始输入与参考
生成在计时外；另记录同步compute时间，不能把两种边界混用。JAX参考与被测
实现同源，这证明结果消费及复现，独立物理正确性仍由H2/J4判定。

| N，48步 | NumPy单核 s | JAX-CUDA x64 s | JAX-CUDA x32 s | NumPy/x64 | NumPy/x32 |
|---:|---:|---:|---:|---:|---:|
| 100 | 3.764984 | 4.157730 | 3.228039 | 0.906× | 1.166× |
| 1,000 | 37.731105 | 4.066923 | 3.446797 | 9.278× | 10.947× |
| 10,000 | 364.049241 | 4.097570 | 3.331466 | 88.845× | 109.276× |
| 100,000 | 正式测量中 | 21.732564 | 3.482058 | 待基线 | 待基线 |
| 300,000 | 预热中 | 60.734212 | 3.883731 | 待基线 | 待基线 |

N=100000同步compute中位数为x64 **21.547112 s**、x32 **3.295101 s**；
N=300000分别60.166400/3.310363 s。包含完整校验后，N=100000的x64/x32
比值为6.241×。精度成本和大量MOST kernel的证据见[硬件剖析](GPU_J4_PROFILE.md)。
在N=100000的单核三次测量完成之前，仍不能宣布该规模的≥100×目标达成。

小规模NumPy100/1000已用独立并发重复脚本补做，每个样本固定单物理核2/4/6，
六次有限checksum精确相等；这样补足旧标准bench入口没有保存checksum的缺口。
新数据不同于下方历史单次作业的调度条件，不选取旧较快数值混入新表。
大规模N10000/100000/300000仍采用同一完整重复方案，不分柱并行、不外推。
测试期间主机还运行其他规模的CPU基线和GPU1诊断；不是独占空闲整机的成绩，
共享CPU/驱动调度和其他用户负载均可能影响耗时。

原始文件：[NumPy小规模复测](gpu-results/j1-refreshed-numpy-small.jsonl)、
[NumPy10000](gpu-results/h5-numpy-10000.jsonl)、
[JAX64](gpu-results/j2-final-jax64.jsonl)、[JAX32](gpu-results/j2-final-jax32.jsonl)、
[十组结果的独立核对](gpu-results/j2-final-jax-summary.json)、
[全源码清单](gpu-results/j2-final-gated-sourcehash.json)、
[监督阶段记录](gpu-results/j2-final-status.jsonl)。

修后CUDA默认版本`f29d7ad`的五档**全部在异质完整N预检被拒绝**，没有接受任何
计时：N100/1000/10000仅residual超门，最大容差倍数30.5115；N100000/300000
还涉及rnet/H/G，residual最大倍数1646.3164。全部最终状态仍在门内、所有列均
响应扰动，但这不能抵销逐步诊断失败。N1024固定四种气温的逐位一致结果不能
外推到这些输入及shape。原始[CUDA五档](gpu-results/j2-cuda-f29d7ad-final.jsonl)、
[源码前后复核](gpu-results/j2-cuda-f29d7ad-sourcehash-recheck.json)已保存。
独立NumPy64 H4同样仍未通过；J3选项扩展后的kernel须另验证并标识源码。

## xuanze 旧方法历史记录（不作当前GPU验收依据）

双路 Xeon Gold 5218R、40 物理核；物理 GPU 1 为 RTX 4090 24 GB，sm_89。
Python 3.12.2、NumPy 2.5.4、JAX/jaxlib 0.11.2、CuPy 14.2.0，安装细节见
[GPU_ENV.md](GPU_ENV.md)。保留其他用户进程；本任务的 GPU 基准、选项检查和
profiling 依次执行。基准使用默认柱、共享合成强迫，每柱 48 个半小时时间步。

每项先完成 **3 次完整预热，再做 3 次完整测量，取中位数**。JAX 使用
`block_until_ready`，CUDA 同步当前 stream；初次编译落在预热中。初始主机到
设备传输在计时外，返回诊断留在设备上；计时包括批量驱动内部缓冲区分配和
CUDA AoS/SoA 转换。NumPy 每一柱均真实执行普通版，没有按小 N 外推或缓存物理结果。

下表为总耗时（秒）。**手写 CUDA 的严格 H4 parity 失败，其计时状态为
`measured_unvalidated`，不得作为合格物理实现的加速结果。**

| 柱数 N | NumPy 单核 | JAX-CUDA x64 | JAX-CUDA x32 | 手写 CUDA x32，精度未过门 |
|---:|---:|---:|---:|---:|
| 100 | 3.332289 | 4.211391 | 2.947675 | 0.001921855 |
| 1,000 | 32.824343 | 4.316150 | 3.090554 | 0.001940340 |
| 10,000 | 364.049241 | 4.613225 | 3.125188 | 0.001966745 |
| 100,000 | 补测中 | 22.003309 | 3.144129 | 0.007287800 |
| 300,000 | 补测中 | 61.388568 | 3.245655 | 0.020225205 |

| N | JAX x64 µs/(柱·步) | JAX x32 µs/(柱·步) | CUDA µs/(柱·步)，精度未过门 |
|---:|---:|---:|---:|
| 100 | 877.373183 | 614.098976 | 0.400386440 |
| 1,000 | 89.919789 | 64.386537 | 0.040423746 |
| 10,000 | 9.610885 | 6.510807 | 0.004097385 |
| 100,000 | 4.584023 | 0.655027 | 0.001518292 |
| 300,000 | 4.263095 | 0.225393 | 0.001404528 |

NumPy N=100/1000 分别为 694.226944/683.840483 µs/(柱·步)。
J1 收回的 N=10000 三次完整样本为 364.049241 / 362.408549 / 377.019978 s，
中位数 364.049241 s（758.435918 µs/柱步），三个单核 affinity 和六次 checksum
一致，原始记录见 [h5-numpy-10000.jsonl](gpu-results/h5-numpy-10000.jsonl)。
用旧 JAX x64/x32 耗时算出的比值为 78.914×/116.489×，仅是历史表的算术比，
待 checksum 计时复核后才能进入新验收结论。
**N=1e5 的 NumPy 中位数尚未完成，不能宣称 ≥100×，也不以小 N 外推补表。**

原始三次样本与预热值见 [GPU JSONL](gpu-results/h5-gpu.jsonl)、
[NumPy 首轮 JSONL](gpu-results/h5-numpy.jsonl)。首轮 NumPy 在 N=10000
完成首个预热 331.5124 s 后中止，切换为以下完整重复并发方案；该预热不计入中位数。

## 完整 NumPy 单核重复的夜间执行

`benchmarks/bench_numpy_parallel_repeats.py` 让三个独立完整重复并发，各自绑定
一个不同的物理核并限制数值库为单线程；**没有把一个 N 柱样本分散到多个核**。
三次预热全部结束后才开始三次计时。每个样本保留 wall time、process time、
CPU affinity、完整柱步数与 checksum。并发会影响整机频率和共享资源，因此
明确记录该调度条件，不宣称它是完全空闲主机上的孤立单核成绩。

N=100 小验证成功，六个完整重复的 checksum 一致；详细数据见
[并发重复验证](gpu-results/h5-numpy-parallel-smoke.jsonl)。大规模任务使用
CPU 2/4/6（N=10000）、8/10/12（N=100000）、14/16/18（N=300000）；
这些 CPU ID 对应不同物理核。后台监督器保存所有部分结果，最迟在
2026-10-13 06:45 EDT 停止自己启动的进程组，供天亮前收尾。

```bash
cd ~/work/noahmp-phase3b/repo
PYTHONPATH=src ~/venvs/noahmp-phase3b/bin/python benchmarks/run_cpu_batch.py \
  --deadline 2026-10-13T10:45:00+00:00 \
  --job 10000:2,4,6 --job 100000:8,10,12 --job 300000:14,16,18 \
  --output-dir ../logs/h5-overnight
```

输出目录必须不存在；新一轮使用新目录和未来的截止时间。后台状态和接续步骤
见 [XUANZE_OVERNIGHT.md](XUANZE_OVERNIGHT.md)。运行中的任务不是已完成基准。

## GPU 复现命令与一次 profiling

```bash
source ~/venvs/noahmp-phase3b/bin/activate
cd ~/work/noahmp-phase3b/repo
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export CUDA_PATH=/usr/local/cuda-12.4 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python benchmarks/bench_gpu.py --sizes 100 1000 10000 100000 300000 --steps 48 \
  --backends jax64 jax32 cuda --output ../logs/h5-gpu-new.jsonl
```

JAX x64 的 N=1e5 单轮 Nsight Systems 采样显示 806114 次 GPU kernel 调用，
其中一个重复 loop kernel 占 GPU kernel 总时间的 94.7%；固定迭代与批量条件
分支的展开是待 HLO 进一步确认的来源。此次只做一轮定位，没有为追求分数
延长调优；完整统计与插桩变更见 [GPU_PROFILE.md](GPU_PROFILE.md)。

## Windows 阶段 3a 历史数据

2026-10-12 RTX 4070 主机没有 nvcc，JAX 是 CPU wheel。N=100、48 步，
预热 3 次、仅测量 1 次：NumPy 2.3524 s（490.08 µs/柱步），JAX CPU x64
0.0654 s（13.63 µs/柱步），约 36×。这是原阶段 3a 代码的历史 CPU 数字，
不是 GPU 成绩，也不与已修复普通版语义的阶段 3b 直接比较。
