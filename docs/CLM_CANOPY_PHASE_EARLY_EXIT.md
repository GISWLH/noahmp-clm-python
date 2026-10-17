# 冠层相变回溯提前退出（2026-10-17）

本项用于降低多站气候年核验成本，不新增物理参数化，也不改变 v1.0 科学验收。
CPU/长时物理测试在 EHPC PBS 计算节点，设备测试在 xuanze；本机只做编辑、
静态解析、字节核对和归档。NumPy 原参考已经在第一个下降点退出。

## 改动与计时范围

原 JAX/CUDA 回溯选中第一个严格下降点后，仍执行完剩余的 24 档二分试探，
但不会再更改选中值。设备端现也在选中后退出；主 Newton 与失效分支恢复的
两处回溯同步。分数、运算、严格下降规则、湿度恢复、迭代上限及最终方程/
独立收支/1e-6 同精度门保持。没有采用 M1 全量 ΔT，也没有改变 NumPy 物理。

使用 system-profile 工作流，在独立目录运行 CUDA events 和单独的 Nsight
Systems 追踪，未向生产源插入计时代码。xuanze 两张 RTX4090，本项使用 GPU1；
其他自有回归及 GPU0 年度任务同时运行。因此下表仅是这两组冠层核的受控
A/B 结果，不能外推整柱、全年或独占 GPU 性能。

编译及准备排除在计时之外；每组 3 次预热，20 对 A/B，顺序交替，每样本连续
5 次发射，保存全部原样本。正确版本 B 将实际 G614 的扩展输入预先放到设备。
初版 A 每次重新搬运 8 字节扩展，原记录保留，采用 B 报告：

| 工况 | 基线中位 ms | 提前退出中位 ms | 配对耗时比中位 | 基线 p10–p90 ms | 提前退出 p10–p90 ms |
|---|---:|---:|---:|---:|---:|
| 原 23 组回归工况 | 101.481573 | 67.467163 | 1.504265 | 101.456505–109.637565 | 67.451688–75.618264 |
| 实际 G614 固定边界 | 35.399054 | 25.607271 | 1.382287 | 35.389251–36.803480 | 25.593002–26.147496 |

全部 300 字段均逐位相同，运行前后 473 份基线源未变。单独追踪区分两个
导出核的符号；原核合计耗时占追踪的绝大部分，主机等待调用不能解释为独立
CPU 物理瓶颈。[脚本、原样本、输入/输出、CUDA 源及统计](v1-results/short-circuit-profile/manifest.json)
保留原字节。原始 Nsight 二进制会记录进程环境，GitHub 推送保护识别出其中的
凭据后已从公开提交移除，仅保留私有原件和两份追踪的 SHA256；没有重压缩
隐藏或绕过保护。[发布处理记录](v1-results/short-circuit-trace-publication.json)
保留原完整私有归档哈希，37项公开证据及数值结论不变。

Nsight 首次内存统计遇到 SQLite 导出时间戳检查，失败文本保留；随后显式
重新导出得到[补充记录](v1-results/short-circuit-profile-supplement/manifest.json)。
整段追踪（含准备/结果核对）H2D 为 33 次、0.191 MB、37.793 μs；D2H 为 10 次、
0.144 MB、20.256 μs。它不是计时循环内流量或整柱传输基准，无 GPU 间通信。

## 当前源核验

冻结源归档 SHA256：`8cffbefe052fc8eb476b3205ea98bd99bdbf840320662f3a439d96f2381df72b`。
xuanze 定向 **9 passed /0 failed /0 skipped，198.60 秒**；进程含收尾 202.092792 秒。
原 23 组与实际 G614 的生产 CUDA 全部 300 字段与改动前基线逐位一致，
CUDA32/JAX32 最大差 0。实际 G614 独立最大温差 2.33750e-5 K，水账残差
1.16415e-10 kg/m²、热账 0.327593 J/m²；均按原界核对。
[当前源、JUnit、独立残差及逐位对照](v1-results/short-circuit-target/manifest.json)保留。

原 G614 四柱整步使用单独兼容目录，保留原六字段土壤/冻结曲线，只改两个
设备回溯文件。CUDA **112.387780 秒完成**，全部输出及 16 份下一状态数组
与改动前逐位一致。最大独立水/热/C/N 账仍为 1.09174e-5 kg/m² /
2.09280 J/m² /1.92091e-4 gC/m² /5.18175e-5 gN/m²；五种杂质最大
5.41789e-14 kg/m²。JAX 整步 **1246.104414 秒完成**，相对其改动前基线所有
输出及 16 份下一状态数组也逐位一致。两条完整路径的 CUDA32/JAX32 输出最大
差 3.63798e-12、下一状态差 0，原独立守恒及 1e-6 门满足。
[完整两条重放、源差异与原基线比较](v1-results/short-circuit-legacy/manifest.json)
保存。含编译/并发的整步耗时不用于提速结论。

当前默认收集 1,032 项：710 项 CPU 与 EHPC PBS55 原完整记录逐节点一致，
CPU 运行库、测试及基准字节未变，只有两个设备源文件改变；322 项设备测试
共 61 文件已于 **15:28:59 UTC** 在 xuanze GPU1 启动当前源完整重跑，
[清单/来源/启动绑定](v1-results/short-circuit-full-launch/manifest.json)保留。
该覆盖拆分不是单进程全量 pytest 已完成。
新 I 从相同锁定配置/强迫的物理种子另起首日，
564.367274 秒完成四站各 48 步（并发下耗时，不作整柱提速结论）。全部 18 个
轨迹字段、23 个检查点/累计账数组与不可变 H48 逐位相同；对独立 NumPy 的
所有声明门无失败，日 H/LE 最大偏差仍为 0.000700186% /0.000936399%。
独立水/热/C/N 累计账最大仍为 0.000340857 kg/m² /71.555978 J/m² /
0.00392248 gC/m² /0.000646398 gN/m²，逐步逐柱原界保持。
[首日、源码/配置身份及完整逐位核对](v1-results/short-circuit-climate-i-day/manifest.json)
在续年之前冻结，原 17520 步请求和 `running`/48 状态保留，没有复用或改写 H 检查点。
I 于 **15:28:54 UTC** 从自身 48 步检查点续原 17520 步全年，GPU0 PID3746083。
[启动绑定](v1-results/short-circuit-climate-i-annual-launch/manifest.json)同时声明额外
JAX 整步当时仍在计算；它随后完成，如上，并未将启动记录改成已完成。

H 已主动替换：核对自有 PID/命令后 SIGINT，原运行器捕获 BaseException 并
保存 **failed/KeyboardInterrupt，573 个已接受步**，不是数值验收失败。
[原报告/轨迹/状态/累计账及主动停止原因](v1-results/short-circuit-h-prefix-superseded/manifest.json)
原样保留。停止归档器初先假设报告仍为 running，断言失败；随后只读恢复核验
真实 failed 代次并归档，没有改报告或检查点源码身份。旧 C 队列也在 6 文件/
26 项完成后停止，[活动文件与其余未完成项](v1-results/short-circuit-obsolete-suction-c/manifest.json)
保持未完成，以让当前源码验证使用计算资源。
年度、spinup、官方/观测、留出及默认驱动切换仍未验收。
[当前470份代码/测试/基准及3975项公开归档的静态字节核对](v1-results/short-circuit-static-audit-20261017.json)
另行保存，包含I续年对自身48步状态的绑定；没有本机物理运行。
归档同时检查Git清单：通用 `profile_output/` 忽略规则曾漏掉当前27项原始证据
及旧native-carry-b的一份已在原manifest中的脚本。现仅为版本化证据目录添加
例外并补齐，原字节/SHA不变；排除上述两份私有追踪后，3975项公开证据都必须
同时存在于Git清单。私有追踪继续按原始哈希核对，没有冒充已发布。

## 计时插桩变更清单

下表是计时代码；生产文件仅包含正式提前退出逻辑，没有插桩。
生成的源/输入/输出/日志逐文件清单在上面37项公开和2项私有追踪的manifest中。

| 文件（相对 `outputs/v1/`，远端独立目录同名） | 变更类型 | 内容 | 行 |
|---|---|---|---|
| `profile_phase_short_circuit.py` | 新建 | 初版 CUDA A/B、events、独立追踪入口与逐位核对 | 全文件 |
| `run_phase_short_circuit_profile.py` | 新建 | 分开执行非追踪计时、Nsight、统计 | 全文件 |
| `launch_phase_short_circuit_profile.py` | 新建 | 独立 A 目录、GPU1、进程及启动记录 | 全文件 |
| `prepare_profile_b_locally.py` | 新建 | 静态生成 B，仅调整扩展输入的搬运位置 | 全文件 |
| `profile_phase_short_circuit_b.py` | 新建 | 常驻输入的 B 版计时及相同逐位核对 | 全文件 |
| `run_phase_short_circuit_profile_b.py` | 新建 | B 版计时/追踪控制 | 全文件 |
| `launch_phase_short_circuit_profile_b.py` | 新建 | 独立 B 目录、GPU1、启动记录 | 全文件 |
| `archive_short_circuit.py` | 新建 | 原始计时/追踪/元数据及后续验证证据归档 | 全文件 |
| `preserve_short_circuit.py` | 新建 | 本机静态逐字节核对及确定性压缩证据，不运行物理 | 全文件 |
| `audit_short_circuit_bytes.py`、`check_evidence_tracking.py` | 新建 | 静态身份/原始哈希/Git清单核对 | 全文件 |
| `privatize_nsight_evidence.py` | 新建 | 私有追踪哈希核对后移出公开树，不显示凭据或重压缩绕过保护 | 全文件 |
| 仓库根 `.gitignore` | 修改 | 版本化证据目录例外及两份私有原始追踪排除 | 10–15 |
| A/B `profile_output/baseline.cu`、`candidate.cu` | 生成 | 精确编译输入及区分符号，仅候选两个循环条件变化 | 全文件 |
| `short-circuit-profile-metadata.json` | 生成 | 计时后设备/拓扑/工具元数据，未冒充计时期间独占记录 | 全文件 |

计时脚本留在忽略的本机 `outputs/v1/` 及远端独立 profile 目录，可在用完后清理；
提交的证据归档和正在运行的年度源码不属于待清理插桩。
