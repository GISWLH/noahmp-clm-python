# J2 GPU 计时可信性审计

2026-10-13 EDT：旧 CUDA 的 `N=100000 × 48`、约 0.007 秒计算时间在冻结源码上
可复现；Nsight 确认实际执行了 **48 次 `step_columns`**，合计 **6.528953 ms**。
本次没有发现“只有 kernel launch、没有执行”的证据，也没有确认 DCE。
但完整输出无法通过同输入 JAX float32 的严格 `rtol=atol=1e-6` 检验，因此
**旧 CUDA 数字仍只作诊断，不能作为通过验收的吞吐/加速比**。

## 冻结代码与原始观测

原始快照：`/home/lhwang/work/noahmp-phase3b/timing-audit-original-20261013T050056Z`。
快照复制源代码和原 `bench_gpu.py`，后续 kernel 数值修复不会改变它；运行设备为
xuanze 物理 GPU 0（RTX 4090），环境 JAX 0.11.2、CuPy 14.2.0，float32。
完整 [SHA256 清单](gpu-results/j2-old-cuda-sourcehash.json) 与每次探针输出的
`sourcehash` 保存实际文件身份。原 kernel SHA256 为
`f607888a7d5f764361c764c718fc1687c4f04cb58c003ae14f60f3e8284a71a8`。
[运行后复核](gpu-results/j2-original-source-recheck.json)确认原始 34 个源码文件
全部与快照清单一致。原探针记录启动磁盘哈希；更新后的探针另外比较 loader
实际加载的 CUDA 源码文本哈希与运行结束哈希，后者尚未用于改写本次旧探针证据。

| 观测 | 真实耗时/数量 | 含义 |
|---|---:|---|
| 原脚本，3 预热后 3 次中位数 | 0.007535301 s | 已同步的原 API 调用；尚未消费全输出 |
| Nsight 捕获样本的同步 compute wall | 0.008149788 s | 包含 loader、发射、同步；不是纯 kernel 累计 |
| 同一样本 CUDA event 跨度 | 0.008110752 s | 当前 CuPy stream 上完整调用跨度，与 wall 同量级 |
| 同一样本所有 `step_columns` 累计 | 0.006528953 s / 48 次 | GPU 实际 kernel 执行时间 |
| 同一样本含全量归约与 host 回读 | 0.197959147 s | 验收要求的端到端边界，包含新增观察开销 |
| 三次探针的 compute wall 中位数 | 0.008010671 s | 诊断值，未通过数值验收 |
| 三次探针含归约回读中位数 | 0.198896267 s | 诊断值，未通过数值验收 |
| 有效列数 / 有效列步数 | 100000 / 4800000 | 每次全输出 finite 与有效温度检查通过 |

设备归约覆盖最终 70 个浮点槽、33 个整数槽和每列全部 48×12 个诊断量。
每个分量分别求 signed sum、absolute sum、按行位置加权 sum、min/max 和 finite
count，再回读这些小数组。额外逐元素比较不允许 checksum 抵消掩盖局部错误。
变更 `SWdown = 1.17 * SWdown + 31 W/m²` 后，温度及能量/水分通量的归约明显
变化，排除了恒定结果/无效 forcing 的这类退化情况。

Nsight 中本次归约观察器占主要时长：`cupy_sum_with_dtype` 为 82.07 ms，
`cupy_max` 为 56.56 ms，`cupy_min` 为 29.32 ms。该开销被明确计入新端到端
指标，同时保留计算阶段指标，不能把两种计时边界混用。这里没有由耗时反推
executed-FLOP，也没有据此声称 660 TFLOPS；FLOP/roofline 需要独立计数器采样。

严格门失败定位到诊断量而非非有限输出：最终浮点状态最大容差比约 0.532，
整数状态完全一致；诊断 H 最大容差比 **345.47**，G **111.29**，能量 residual
**1448.55**。此处容差比为 `abs(actual-reference)/(1e-6+1e-6*abs(reference))`，
超过 1 即失败。六次完整输出检查均失败，没有生成 accepted benchmark。
该现象属于数值差异；不能仅因时间短就把它解释成 DCE 或同步错误。

原始证据：[原脚本复现](gpu-results/j2-old-cuda-reproduction.jsonl)、
[全部归约/逐项门及 event-wall 数据](gpu-results/j2-old-cuda-consumed.jsonl)、
[运行日志](gpu-results/j2-old-cuda-consumed.log)、
[Nsight kernel CSV](gpu-results/j2-old-cuda-stats_cuda_gpu_kern_sum.csv)、
[Nsight CUDA API CSV](gpu-results/j2-old-cuda-stats_cuda_api_sum.csv)。
完整 `.nsys-rep` 留在快照的 `profile_output/j2-old-cuda.nsys-rep`。

## 新基准验收边界

`bench_gpu.py` 先在相同输入上计算 JAX 参考，再验证本次完整 N 列分别扰动初始温度和
辐射的敏感性。CUDA 的参考固定为 JAX float32；JAX 后端使用同 dtype 的 JAX
参考，以检查完整执行/稳定消费，不声称它构成独立物理正确性证明。

每次预热与正式测量都必须完成：实际计算及同步 → 全状态/诊断 device reduce →
归约 host 回读 → finite/有效列步计数 → 逐分量归约 `1e-6` 门 → 浮点逐项
`rtol=atol=1e-6` 与整数逐项全等。归约 sum 按该分量元素数归一化后应用相同
容差，signed/absolute/weighted 和 min/max 各自独立验收。失败返回非零，只记录
`rejected` 或明确的诊断时间，不能写 `measured`；源码运行期间发生变化也拒绝。

有效 GPU `seconds` 改为包含设备归约和回读的端到端耗时，另存
`synchronized_compute_wall_seconds`。CUDA 同时记录 `cuda_event_span_seconds`；
JAX 使用其自身同步后的 wall，不把其他 stream 的 CuPy event 冒充 JAX kernel
时间。`--cuda-parity-passed` 不能绕过本门；缺少独立 H4 通过声明时，即使计时门
通过，也仅标为 `timing_validated_h4_unverified`。checksum 门不能代替 H4。

浮点状态及诊断还必须保持所请求的 float32/float64，整数状态必须为 int32；
数值相同但 dtype 错误也会拒绝。源码证据同时区分原始文件 byte SHA256 与
loader 加载文本的 SHA256，后者使用相同的换行归一化，避免 Windows CRLF
产生无意义的差异。非有限失败量以字符串保存于标准 JSON，不输出 JSON 的
非法 `NaN`/`Infinity` 数字。

`_numpy_runner` 和 `_gpu_runner` 的函数源码与修改前完全相同，因此运行中的 CPU
基准与 `profile_gpu.py` 的低层调用接口不受此变更影响。`profile_gpu.py` 仍是
计算阶段 tracing 工具，它的单次 profile 结果不能代替新的完整基准验收。

## 复现与验证

```bash
cd ~/work/noahmp-phase3b/timing-audit-original-20261013T050056Z
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true JAX_PLATFORMS=cuda CUDA_PATH=/usr/local/cuda-12.4 PYTHONPATH=src
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o profile_output/j2-old-cuda \
  ~/venvs/noahmp-phase3b/bin/python -u benchmarks/audit_gpu_timing.py \
  --n 100000 --steps 48 --capture --output ../logs/j2-old-cuda-consumed.jsonl
```

重跑请使用新的输出文件名以保留证据。原始快照保留旧基准；新门在工作仓库运行：

```bash
PYTHONPATH=src NOAHMP_REQUIRE_GPU=1 python -m pytest tests/test_gpu_benchmark_checksum.py -q
PYTHONPATH=src python benchmarks/bench_gpu.py --sizes 100 --steps 48 \
  --backends jax32 cuda --output ../logs/j2-gated-smoke.jsonl
```

CPU 门单测实际结果：**7 passed, 1 skipped**（跳过需要显式 CUDA 环境的测试）。
涵盖全零假输出、相反误差抵消、顺序错乱、任意状态槽 NaN、少算列以及不响应
forcing 的退化输出。GPU 门单测与小规模基准结果另附本节后续实测记录。

GPU 0 上的独立门测试实际 **8 passed**，见 [测试日志](gpu-results/j2-gate-tests.log)。
随后 N=100、48 步的小规模流程中，JAX32 完成三次预热和三次测量且门全通过：
端到端中位数 **3.0072766 s**，同步 compute wall 中位数 **3.0038134 s**；旧 CUDA
被严格门拒绝，组合命令退出码 1，未输出 CUDA `measured`，见
[小规模 JSONL](gpu-results/j2-gated-smoke.jsonl)及[日志](gpu-results/j2-gated-smoke.log)。
该次小规模运行使用当时四列敏感性版本；之后代码审阅补充了完整 N 列敏感性
预检、显式 dtype 约束及三个 CPU dtype 回归案例，CPU 结果变为
**10 passed, 1 skipped**。后续源码哈希与原始小规模证据不同，应按新脚本重新验收。

相同输入的多次 checksum 通过本身不能证明每一轮均重新执行了 kernel；本次
直接计数证据来自 Nsight 捕获的一次 48 步调用。完整 N 敏感性预检扩大了对
恒定/仅小 N 正确结果的检查范围，但不把它表述为对任意缓存实现的形式化证明。

### 最终版本复验：11 项 GPU 门与完整 N 敏感性

2026-10-13 05:33 UTC，在新的冻结快照
`/home/lhwang/work/noahmp-phase3b/j2-final-gated-20261013T052813Z` 上复验；
物理源码来自 `j4-stable-jax-20261013T0513/src`，包含 JAX 薄雪修复。
完整 [源码清单](gpu-results/j2-final-gated-sourcehash.json) 区分该快照与旧探针；
JAX SHA256 为 `65c5c3bc23e2c378014e8e6cc79c584387e6e1f0da8e7a512f7ca2a10277e8a8`。
新增的 dtype、完整 N 预检和标准 JSON 版本已经在 GPU 0 复验。

`NOAHMP_REQUIRE_GPU=1` 实际 **11 passed in 0.94s**，见
[原始测试日志](gpu-results/j2-final-gate-tests.log)。N=100、48 步的 JAX32
完整 N 异质预检确认 **100/100 列响应**，三次预热与三次测量全门通过，
端到端中位数 **2.991220064 s**，同步 compute wall 中位数 **2.987346679 s**。
见 [最终 smoke JSONL](gpu-results/j2-final-gated-smoke.jsonl)。

同一 smoke 中冻结 CUDA 在完整 N 预检被严格门拒绝：状态最大值归约的容差比
约 1.387，诊断 signed-sum 约 21.97，状态和诊断逐元素门也失败；没有输出
CUDA `measured`。该组合 smoke 退出码为 1，是预期 CUDA 验收红灯，
与 11 项门测试全通过分开报告。两个阶段的运行前后源码哈希均未变化。
正式 JAX64/JAX32 五档完整基准另行记录，不能从此 smoke 推算大 N 吞吐。

### 最终五档 JAX 基准

同一冻结快照与 GPU 0 已完成 JAX64/JAX32 的全部五档；每档完整 N 异质敏感性
预检及三次预热、三次测量均通过。下表只取最终 `event=result` 的中位数，
并复核逐次有效列步、逐元素门及中位数重算。端到端包括归约与 host 回读；
compute 是同步 API 调用耗时，不能等同纯 kernel 时长。

| N | JAX64 端到端 s | JAX64 compute s | JAX32 端到端 s | JAX32 compute s |
|---:|---:|---:|---:|---:|
| 100 | 4.157730043 | 4.154170349 | 3.228039250 | 3.224391975 |
| 1000 | 4.066922523 | 4.063364059 | 3.446796618 | 3.442471378 |
| 10000 | 4.097570240 | 4.081260100 | 3.331466101 | 3.314959794 |
| 100000 | 21.732564241 | 21.547112219 | 3.482058212 | 3.295100853 |
| 300000 | 60.734212235 | 60.166399531 | 3.883731045 | 3.310362689 |

原始数据：[JAX64](gpu-results/j2-final-jax64.jsonl)、
[JAX32](gpu-results/j2-final-jax32.jsonl)、[独立汇总](gpu-results/j2-final-jax-summary.json)、
[完整运行日志](gpu-results/j2-final-run.log)、[阶段状态及进程 PID](gpu-results/j2-final-status.jsonl)。
这次 JSONL 内嵌的来源仍是原五文件清单；外层启动器同时记录和逐阶段检查
全部 `src` Python 文件、CUDA 源码以及基准脚本，包括 `column.py`、`pack.py`、
`io.py` 和普通版物理模块。[完整运行后复核](gpu-results/j2-final-sourcehash-recheck.json)
确认全部文件与启动 sidecar 一致。运行中没有缩小 N，也没有出现 OOM。

这些结果验收的是同 N、同输入、同精度的消费与复现；同源 JAX 参考不能替代
独立 NumPy 物理 parity，也不能证明改变 batch shape 后仍保持同样数值。
CPU 加速比需结合已完成的同负载普通版最终基线，不能用小 N 的每柱时间外推。

### f29d7ad CUDA 修复后的五档预检

在单独快照 `/home/lhwang/work/noahmp-phase3b/j2-cuda-f29d7ad-gated-final` 中，
kernel 与 loader 均精确取自提交 `f29d7ad4427c0a88710fd87835b43f9150604ac7`，
其余物理源码保持同一薄雪修复版 JAX。kernel SHA256 为
`3da106c2ffdc0eb42ab95a91227ba1492861d081e5c1552389fd536625356395`；
loader 为 `5a87015be97d1b8d548c10ade35cd33e149debec0456e7983a4bc34ab3ac7f75`。

五档全部在完整 N 异质预检被拒绝，因此没有开始正式三次预热/三次测量，
也没有产生任何可接受 CUDA `seconds`。各档均确认全部 N 列响应扰动：

| N | 最终状态最大容差比 | 超差诊断及最大容差比 |
|---:|---:|---|
| 100 / 1000 / 10000 | 0（逐项完全相同） | residual 30.5115 |
| 100000 / 300000 | 0.106659（通过） | rnet 2.76807、H 63.8939、G 25.8117、residual 1646.316 |

完整命令退出码 1。[原始 JSONL](gpu-results/j2-cuda-f29d7ad-final.jsonl)、
[日志](gpu-results/j2-cuda-f29d7ad-final.log)、[进程/阶段状态](gpu-results/j2-cuda-f29d7ad-final-status.jsonl)
与[全部源码前后复核](gpu-results/j2-cuda-f29d7ad-sourcehash-recheck.json)保留了失败证据，
源码未变化。单一 N=1024 的同构输入测试通过不能外推到这些异质完整规模，
本次也不以 checksum 或局部 parity 替代独立 H4。

### NumPy 计时返回值的独立修复

独立审阅通过故障注入确认原 `_timed` 会丢弃 NumPy runner 返回的 checksum，
从而把 NaN 温度或 ET 产生的错误结果仍标为 `measured`。工作分支已补上
每次预热和测量的有限标量检查，以及全部重复之间的精确一致性检查，并在
每个 sample 的 JSON 保留 checksum。失败非零退出且不生成最终 `measured`。
`_numpy_runner` 函数本身完全不变，GPU 计时路径也未变。

新增 `tests/test_numpy_benchmark_checksum.py` 在全部六个重复位置分别注入
NaN、Inf 或变化 checksum，另从普通物理返回值注入 NaN 温度和 ET 验证 CLI；
与现有 checksum CPU 测试合跑结果为 **31 passed, 1 skipped**。跳过项仅为
本地未显式请求的 GPU 测试。该修复在上述 GPU 快照冻结之后完成，故快照的
旧 NumPy 分支不具备此门；该快照本次只运行 GPU 基准，未产生 NumPy 基线。

## 双卡脚本的验收边界（尚待实测）

`bench_gpu_scaling.py` 比较单卡完整 N 与双卡各 N/2，使用 `check_gpu_extended.py`
的八类异质合成柱和对应逐柱强迫。单独进程在非计时区计算完整 N 的 JAX 参考，
保存 `.npy`；每张卡只上传自己对应的连续分片及参考。参考生成、文件读写和
H2D 均在计时区外。CUDA 的参考固定 JAX float32，H4 状态仍独立标记。

每张物理 GPU 使用一个新建 spawn 进程。单卡和双卡分别完成三次全量预热、
三次测量；各次由包含父进程的 barrier 同步起跑，只有全部分片逐元素通过
`rtol=atol=1e-6`、整数全等、完整有效列步计数、覆盖与源码身份检查后才给出
可接受加速比。缺少两张不同 PCI 身份的 CUDA 卡或任一卡失败则非零退出。

`inclusive_coordinator_seconds` 从父进程进入 barrier 之前计到全部结果回读和
进程间传输完成，包含 barrier 等待及调度；`compute_coordinator_seconds` 使用
相同起点，截止最后一个 worker 的同步计算结束，所以也包含 barrier 等待。
另外记录 barrier 放行之后各 worker 的 compute/inclusive 时长及最大值，CUDA
另有 event 跨度。该多进程 wall 边界不等同于 Nsight 的纯 kernel 累计时间。

下列命令须在两张 GPU 都已释放、工作快照已冻结后运行；目前只完成语法、
`--help` 和分片/失败路径静态审阅，**尚无双卡实测数字**。重复固定输入的缓存
证明边界同上，逐项对照与 checksum 不单独证明每轮重新发射了 kernel。

```bash
env -u LD_LIBRARY_PATH PYTHONPATH=src CUDA_PATH=/usr/local/cuda-12.4 \
  ~/venvs/noahmp-phase3b/bin/python -u benchmarks/bench_gpu_scaling.py \
  --backend jax32 --sizes 100000 64800 --devices 0 1 --steps 48 \
  --output ../logs/j4-scaling-jax32.jsonl \
  --work-dir profile_output/j4-scaling-jax32-reference
```

首次可先用新输出路径执行 `--sizes 100` smoke，再启动完整 N。脚本拒绝已有
输出与 reference 目录，保留原始证据，不会覆盖前次运行。

## 仪器化改动清单

| 文件 | 类型 | 内容 | 入口行 |
|---|---|---|---:|
| `benchmarks/bench_gpu.py` | 修改 | 全输出门、独立列敏感性、源码哈希、双计时边界、失败退出码 | 86 |
| `benchmarks/gpu_checksum.py` | 新增 | GPU 归约、逐项比较、计数和取消抵消的校验 | 16 |
| `benchmarks/audit_gpu_timing.py` | 新增 | 冻结源码的诊断探针及 CUDA profiler 区间 | 20 |
| `benchmarks/bench_gpu_scaling.py` | 新增 | 两张独立 GPU 分片、完整参考逐项门、同步多进程计时 | 28 |
| `tests/test_gpu_benchmark_checksum.py` | 新增 | 有效执行与损坏输出的回归测试 | 26 |
| `tests/test_numpy_benchmark_checksum.py` | 新增 | NumPy 非有限/非一致 checksum 及 CLI 故障注入 | 19 |
| `docs/GPU_TIMING_AUDIT.md` | 新增 | 实测证据、边界及限制 | 1 |

kernel、JAX 物理和 loader 没有由本计时任务修改。
