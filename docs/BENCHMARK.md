# GPU 性能基准

> 2026-10-13：当前共同源码的JAX x64/x32和CUDA五档共15组全部通过
> device归约→host回读→JAX逐值门。下表仅取最终`event=result`；原H4及年度/x32域验收
> 仍有红灯，checksum不能替代独立物理parity；详见[计时审计](GPU_TIMING_AUDIT.md)。

## 当前共同源码的15组计时

物理GPU0，RTX4090，CUDA `7640b062…`、loader `c4afdbed…`、JAX `0e296691…`。
71个src/bench/tests文件冻结于`j2-v2-stable-gated-final`，各阶段首尾身份复核一致。
每档完整N异质预检和所有列输入敏感性通过，正式3次完整预热、3次完整测量，
每次完整状态/逐步诊断的点值及device归约门均通过；不以抽样替代完整输出门。

| N，48步 | NumPy单核 s | JAX x64 s | JAX x32 s | CUDA s | NumPy/x64 | NumPy/x32 | NumPy/CUDA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 3.764984 | 4.165433 | 3.402356 | 0.004517 | 0.904× | 1.107× | 833.47× |
| 1,000 | 37.731105 | 4.178183 | 3.607140 | 0.004846 | 9.031× | 10.460× | 7,786.32× |
| 10,000 | 364.049241 | 4.223078 | 3.655238 | 0.018328 | 86.205× | 99.597× | 19,862.76× |
| 100,000 | 3616.904161 | 21.355447 | 3.807727 | 0.196976 | 169.367× | 949.885× | 18,362.13× |
| 300,000 | 正式测量中 | 59.585120 | 4.123694 | 0.612957 | 待基线 | 待基线 | 待基线 |

GPU主表包含同步计算、完整归约/逐值验证和host回读；初始H2D及参考生成在计时外。
NumPy每个样本是真实单物理核上运行全部N×48，独立重复分配到不同物理核同时执行，
不将一个样本的列分给多个核。两种后端的结果消费方式不同：NumPy逐柱消费终态
标量checksum并释放诊断列表，GPU保留并校验全部输出；不把这种差别隐去。
共享主机仍有其他作业，不能将结果视作独占空闲机器的硬件极限。

N=100000 的正式样本为3543.480384、3616.904161、3648.669605 s，
三次预热及三次测量的有限checksum精确相同；每次都执行480万柱步，
CPU process-time/wall均约0.99645。此次RTX4090上三后端均达到相对该普通
Python/NumPy单核基线的100×计时目标；这不等于A100结果或H4物理验收通过。
CPU作业启动时没有记录已加载源码的hash：事后核查普通物理16文件与启动前
提交一致，但计时runner来自当时未提交工作树，不能追认运行时源码锁定。
原始[CPU日志](gpu-results/j1-numpy-100000-final.jsonl)、
[验收复算](gpu-results/j1-numpy-100000-acceptance.json)、
[来源限制](gpu-results/j1-numpy-100000-provenance.json)保留这一证据边界。

07:55 UTC进度：N300000三次完整预热已于07:43 UTC结束，checksum均为
84253926.96750298，每次1440万柱步、单物理核。三次正式测量已提交，但
尚无正式样本完成或最终result；不将预热时间填入主表，也不由它外推中位数。
[进度日志](gpu-results/j1-numpy-300000-warmups.jsonl)及
[监督器快照](gpu-results/j1-numpy-300000-warmups-supervisor.json)仅记录这一阶段。

| N | x64同步计算 s | x32同步计算 s | CUDA同步计算 s |
|---:|---:|---:|---:|
| 100,000 | 21.170121 | 3.620372 | 0.007101946 |
| 300,000 | 59.018088 | 3.550512 | 0.020489365 |

约7ms的CUDA计算区间现在通过完整JAX32输出门，但公开端到端成绩使用0.196976s，
两种边界不能混用。新版被采首步4347 FLOP/柱步、2.825 TFLOP/s仅描述该counter
样本，不能外推所有48步，见[硬件记录](GPU_J4_PROFILE.md)。原NumPy64 H4仍失败，
CUDA计时状态是`timing_validated_h4_unverified`，加速比不表示物理验收通过。
JAX参考与被测实现同源，只证明结果消费/复现，独立物理门仍需另验。

完整原始数据：[x64](gpu-results/j2-v2-stable-jax64.jsonl)、
[x32](gpu-results/j2-v2-stable-jax32.jsonl)、[CUDA](gpu-results/j2-v2-stable-cuda.jsonl)、
[逐项复算汇总](gpu-results/j2-v2-stable-summary.json)、
[源码首尾核查](gpu-results/j2-v2-stable-sourcehash-recheck.json)。旧源表另节保留历史。

## 当前共同源码的双4090扩展

同一冻结源码另开窗口，单卡完整N对双卡各N/2；每组独立三次预热、三次测量，
每个worker输出与完整N参考逐值比较。四组全部通过，72个worker样本（含预热）
浮点状态/逐步诊断逐位一致、整数全等。下面仅在同组内计算加速比，不与上面
不同运行窗口的单卡时间拼接。计时包含协调、输出门和host回读。

| 精度 | N，48步 | 单卡 s | 双卡 s | 单/双卡 |
|---|---:|---:|---:|---:|
| JAX x32 | 100,000 | 4.022509 | 3.842597 | 1.04682× |
| JAX x32 | 64,800 | 3.808625 | 3.501909 | 1.08759× |
| JAX x64 | 100,000 | 21.480919 | 13.241240 | 1.62227× |
| JAX x64 | 64,800 | 12.865216 | 7.353351 | 1.74957× |

**没有一组达到1.9×。** 独立柱允许分片，但不保证短kernel/调度成本随GPU数减半。
K4结论：双卡工作已冻结，不再调优或重复测试。x32单卡工作量只有约3.8秒，
完整计时中的传输、回读和协调开销占据主要比例，按Amdahl定律，增加设备后
这部分成本不能随计算分片减半，因此1.05×/1.09×的近零扩展是预期行为，
不是双卡实现bug；现有完整逐值输出门均通过。这里的3.8秒是完整计时口径，
不冒充单独kernel计时。原1.9×只是任意设定的探索目标，**不作为验收条件**。
手写CUDA双卡：**未测**。此结论不取消K1/K2/K3和全量pytest合并门。
这些是JAX分片测试，不是手写CUDA双卡测试；通过同源参考也不替代64800合成域
的NumPy独立物理门。归约、完整覆盖、三样本中位数及首尾源码核查见
[双卡复算](gpu-results/j4-v2-scaling-summary.json)和[计时审计](GPU_TIMING_AUDIT.md)。

## J1/J2 旧65c5源码的输出校验计时

以下JAX数字属于根系显式归约修复前的`65c5c3bc…`源码，表内CPU进度也保留
当时状态。当前`0e296691…`/CUDA `7640b062…`共同版本的结果见上表。

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
本旧表发布时单核三次测量尚未完成；当前基线及目标判定见上表。

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
**该历史记录时N=1e5的NumPy中位数尚未完成，未据此宣称≥100×或用小N外推。**

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
