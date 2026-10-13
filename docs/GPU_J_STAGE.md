# J1–J5 后续验收（2026-10-13）

> K阶段收尾10:35 UTC：原N300000监督器正常完成，三预热+三正式checksum一致，
> 正式中位数10272.657723秒，已收原始日志并补BENCHMARK；不得重启旧监督器。
> J1–J5、双卡、profiling与计时审计保持冻结，不重复。K2四后端全年已通过，
> K1/K3与原H4仍红，最新入口为GPU_SUMMARY.md和K取证文档，不能合并main。
> 以下“只剩N300000运行”等时间段是历史检查点，不再是当前待执行指令。

用户在阶段 3b 后要求继续 J1→J5。工作分支保持 `gpu/migration`，状态 ABI 不变。
这是当前接续入口；旧夜间 CPU 监督器和访问路径见 `XUANZE_OVERNIGHT.md`。

## 最新接续状态（单卡与双卡均已收齐）

- 单卡最终共同源码五档三后端15组全部完成并通过完整输出门，原始前缀
  `docs/gpu-results/j2-v2-stable-`。JAX/kernel/loader身份分别为
  `0e296691…`/`7640b062…`/`c4afdbed…`，后续没有生产源码修改。
- J1 N100000完成：三次中位数3616.904160872 s，六次checksum完全一致，
  每次480万柱步、单物理核；对当前GPU含校验时间的加速比分别为
  x64 169.367×、x32 949.885×、CUDA18362.13×。CPU启动没有hash，不能
  追认源码锁定，详见`j1-numpy-100000-provenance.json`。
- CPU监督器2293032只剩N300000（子PID/PGID2293035）运行，deadline为
  10:45 UTC；不要重启或用100000外推。只接受三预热+三正式后的result。
  07:55 UTC已收回三次完整预热：checksum精确相同，每次1440万柱步；
  正式三样本于07:43:08 UTC提交，尚无正式结果。证据前缀
  `docs/gpu-results/j1-numpy-300000-warmups`；不能当成已完成基准。
- 新版ncu已完成并推送`b17d78c`，被采首步4347 FLOP/柱步、2.825 TFLOP/s，
  全输出门逐位通过。无需再重复profiling或占卡调优。
- 双卡监督器2611548已正常结束，冻结目录`j2-v2-stable-gated-final`源码
  首尾一致，状态`logs/j4-v2-scaling-status.jsonl`。x32 N100000/64800
  完整时间加速1.04682×/1.08759×，x64为1.62227×/1.74957×；四组完整
  逐值门全部通过，均未达1.9×。GPU0/1已释放，无需重复测试或继续调优。
- 后续heartbeat只需收回原CPU N300000的结果/截止状态，更新各结果入口并
  推送。确认剩余物理缺口已完整交代后结束heartbeat；不能据此宣告J1–J5全绿。
- 全量134passed/2原H4failed、H6 92passed/11预期未实现已收证。
  原H4、第三天年度预检、x32合成域NumPy对照均仍红；候选2e-3已知不足，
  不申请通过该阈值来合并，不改普通物理、不合并main。以下时间段为历史检查点。

## 约束和当前状态

- J1：收回完整夜间 NumPy 基线；N=10000/100000已完成，中位数分别
  **364.049240723/3616.904160872 s**；N300000仍在计算，不能重启或外推。
- J2 数值：先分解并修复 CUDA/JAX float32 的实现差异，目标比较
  `rtol=1e-6, atol=1e-6`，保持原 H4 红灯，不能隐藏 residual。
- J2 计时：旧 GPU 表缺少强制 device-reduce→host→JAX checksum 门，
  当前仅保留为历史待复核记录，不作为最终验收的性能证据。新计时必须实际
  消费所有输出，检查有限值、完整工作量和参考校验，失败不能记 measured。
- 同精度差异不自动证明某个特定根因；运算顺序、FMA、数学函数和迭代分支
  都需要可重复对照。0.007 s 本身也不能给出 FLOP/s，须先测执行工作量。
- 相对 NumPy64 的精度地板须推导并用局部数值灵敏度验证；候选分档为状态量
  rtol 1e-4、residual 绝对 2e-3 W/m²。**这只是用户要求形成的提案，未经
  用户明确确认，不得修改测试/验收容差，不得据此合并。**旧 48 步 residual
  最大差已达 2.5568e-3，不能预先声称 2e-3 能覆盖全部场景。
- J3：在 J2 证据明确后实现 12 字段共 35 个已实现选项值的 CUDA 分发；
  不把 18 个 ABI-only 字段当作完整物理。occupancy 调优必须先校验输出；
  48 步融合是可选项。
- J4：roofline/实际算术强度、ncu、双 4090 扩展、365 天长稳、JAX x64/x32
  约 7 倍差的一轮解释、64800 柱集成验收。没有地理输入时只能称合成大陆规模
  压力测试；约 1.9× 双卡扩展是待测目标，不能预设结果。
- J5：完整记录实测及缺口，适当回归、commit+push；H2+H4+H5 全绿且所有
  必要确认到位前不合并 main，不勾选阶段 3 总体完成。

## 并发资源与文件职责

J2 数值子任务占物理 GPU1，修改 `jax_core.py` / `column_kernel.cu`；
计时审计子任务占物理 GPU0，修改 `bench_gpu.py` 和审计脚本/文档，使用独立
源码快照防止并发修复污染结果。现有 CPU 基线继续运行，`_numpy_runner`
保持不变。J3 选项映射和 J4 长稳脚本先做静态准备，正式 GPU 性能采样须排队。

每阶段都记录源码哈希、原始日志和实际设备。新结果有意义时推送分支；单一
诊断方向卡住约 30 分钟要记录并换方向，不用无关压力负载占用整夜。

## 05:40 UTC 接续检查点

- 已推送J2数值修复：`b3c5ee1`；默认48步只剩residual的504个同精度超差，
  不动阈值。N1调用者输入别名bug已通过修前/修后对照修复。
- 已推送全量checksum计时门：`f9fbd02`；最终GPU仪器11项通过。运行快照
  `/home/lhwang/work/noahmp-phase3b/j2-final-gated-20261013T052813Z`，监督PID
  `2407614`，进度 `/home/lhwang/work/noahmp-phase3b/logs/j2-final-20261013T052813Z-status.jsonl`。
  GPU0按JAX64五档→JAX32五档顺序跑；CUDA预检失败不出正式时间。运行快照
  不随工作树修改，结果用记录的source hash解释。不要重复启动。
- J4年度/域脚本及结果已推送`debeff6`，年度和x32域仍红灯。root采样完成，
  GPU0留给正式基准，GPU1仅做有限时长residual诊断；双卡测试待两卡可用后排队。
- roofline与JAX热点见`GPU_J4_PROFILE.md`；ncu已成功收counter，x64新nsys导入
  `Wrong event order`已重试一次仍失败，保留日志，不继续消耗时间重复导入。
- 新发现CPU标准bench入口未检查NaN checksum，正在修仪器；在途夜间CPU脚本
  是独立实现，已检查其finite及六次checksum相等，不受这个入口缺陷影响。

## 后续 heartbeat

06:13 UTC更新：`c6bb8c6`已推送JAX四层根胁迫显式交叉归约。真实col2跨batch
回归从失败变为逐位一致；64800域15个N1样本均通过1e-6门，原NumPy64域门仍红。
最终JAX源码SHA256为`0e2966918699eb39ee520ec2993eee5819df9d49371a1bfc87901b2dc8e69270`。
J3温度胁迫先平方修复在旧JAX基线上54passed；GPU0正在对新JAX联合版本做最终矩阵。
随后冻结源码，GPU0重新跑五档三后端checksum基准；GPU1做有限block配置实验及
完整pytest/H6，最后两卡一起重跑x32/x64分片门。旧x32双卡100000和64800均失败，
没有可接受的大规模双卡加速数；旧源JAX计时不自动代表新归约版本。
CPU监督器2293032仍运行，N100000正在三次正式测量，N300000仍预热；不要重启。

06:15 UTC更新：J3联合55项通过、nvcc零warning，已推送`3c190db`。GPU0新基准
监督PID2510183，冻结目录`/home/lhwang/work/noahmp-phase3b/j2-v2-stable-gated-final`，
进度`logs/j2-v2-stable-status.jsonl`，按JAX64/JAX32/CUDA五档执行。GPU1的三档
block实验已完成且全部通过严格输出门，详见[GPU_OCCUPANCY.md](GPU_OCCUPANCY.md)；
波动较大，保留生产256。GPU1现转全量pytest、H6及剩余H4精度报告，日志前缀
`j5-final-`，其后释放供双卡复测。旧大规模双卡失败不能提前转绿。

06:24 UTC更新：GPU1全量134passed/2原H4failed/0skipped；H6为92passed、11
预期未实现、无数值失败；新日累计精度及H4报告见GPU.md，569项残差差>2e-3未变。
73个源文件首尾hash一致，证据`logs/j5-final-*`已取回。GPU1已释放，另有非本任务
约5GB进程，勿停止。GPU0正式基准完成后先交root一次新版ncu，再跑双卡；
双卡开始/结束需记录GPU硬件负载，不能当作空闲独占整机成绩。

05:48 UTC更新：`f29d7ad`单行修复后，J2默认N1024×1/48全部诊断/状态逐位一致，
H2/薄雪另5 passed。J3代理正式接手kernel/loader的35值分发；root准备有限block
size实验，双卡仍需排队。原H4 NumPy64 residual差最大0.00250709，569项超过
2e-3，且H/LE/G也超原门；候选容差不能直接解决，未申请/获得放宽授权。数值
代理随后限时诊断JAX32不同batch shape的J4差异，不修改生产JAX。

05:59 UTC更新：JAX十组计时全过，已推送`3a30dbd`；小规模NumPy100/1000也
补齐六次checksum。f29 CUDA五档全被异质preflight拒绝，未生成可接受计时。
J3第一版nvcc零warning、52pass/2fail：默认同精度和16组runoff二出口通过，
RainSnow3及混排仍有residual小差；正在验证气孔温度平方的分组。数值代理已
将JAX N64800/N1差异缩至根系beta归约，准备独立tmp固定四项树的验证。
GPU双卡N100 smoke通过（0.996×，小规模无加速）；N100000/64800正式x32
进程PID2467048，日志前缀`/home/lhwang/work/noahmp-phase3b/logs/j4-scaling-jax32-final`。
两卡暂用于该作业，完成后交回短数值验证，再排x64双卡；不要同时占卡profiling。

`xuanze-gpu` 每 30 分钟接续本聊天。先读取本文件和最新日志，避免重复正在
运行的任务；优先收回 CPU 结果和推进当前未完成阶段。只有状态有意义改变、
完成、失败或需要用户确认时通知。全部工作完成或明确记录无法继续的缺口后，
推送最后状态并结束该 heartbeat；不能仅因 NumPy 完成而宣布 J1–J5 全部完成。
