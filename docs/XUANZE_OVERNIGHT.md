# xuanze 夜间计算接续记录

> 2026-10-13 J1–J5 更新：执行优先级以 [GPU_J_STAGE.md](GPU_J_STAGE.md) 为准。
> 本文保存 CPU 监督任务的位置；CPU 作业完成不代表后续 J 阶段完成，也不自动结束 heartbeat。
> 旧 GPU 计时现在必须通过新增 checksum 门复核，不再直接用于最终验收加速比。

2026-10-13，工作分支 `gpu/migration`。H1/H2/H3 已通过，H4 严格 parity
红灯，H6 已实现范围审计通过；全量 Linux pytest 为 **44 passed, 2 failed**。
H4 根因分析见 [GPU_BLOCKERS.md](GPU_BLOCKERS.md)，不得因性能良好而合并 main。

## 正在执行

- SSH：remote-ssh 技能中的 `xuanze-neiwang`，公钥已验证。
- 远程仓库：`/home/lhwang/work/noahmp-phase3b/repo`。
- 新 venv：`/home/lhwang/venvs/noahmp-phase3b`。
- 监督器 PID：`2293032`；2026-10-13 00:43:49 EDT 启动。
- 状态：`/home/lhwang/work/noahmp-phase3b/logs/h5-overnight/state.json`。
- 完整单核重复：N=10000 / 100000 / 300000，各自 3 次完整预热、3 次完整测量。
- 父进程/进程组 ID：2293033 / 2293034 / 2293035；对应 CPU ID
  2,4,6 / 8,10,12 / 14,16,18，每次重复只用一个不同物理核。
- 日志：同目录 `numpy-10000.jsonl`、`numpy-100000.jsonl`、`numpy-300000.jsonl`；
  各有同名 `.log`。监督器日志为上一层 `h5-supervisor.log`。
- 自动截止：2026-10-13 **06:45 EDT / 10:45 UTC**；监督器只停止自己创建的进程组，
  保留部分日志，禁止操作服务器上其他用户的进程。
- 本聊天 heartbeat：`xuanze-gpu`，每 30 分钟接续检查；完成后删除此 heartbeat。

单个大样本数小时是计算耗时，不是访问阻塞。不重启正在运行的样本，不把未完成
预热、单次测量或小 N 外推写成中位数。并发的是独立完整重复，不是把一个样本
拆为多核；调度条件和共享资源影响必须在最终基准中继续说明。

## 检查与收尾

```powershell
ssh -o BatchMode=yes xuanze-neiwang 'cat ~/work/noahmp-phase3b/logs/h5-overnight/state.json'
scp xuanze-neiwang:work/noahmp-phase3b/logs/h5-overnight/numpy-10000.jsonl docs/gpu-results/h5-numpy-10000.jsonl
scp xuanze-neiwang:work/noahmp-phase3b/logs/h5-overnight/numpy-100000.jsonl docs/gpu-results/h5-numpy-100000.jsonl
scp xuanze-neiwang:work/noahmp-phase3b/logs/h5-overnight/numpy-300000.jsonl docs/gpu-results/h5-numpy-300000.jsonl
scp xuanze-neiwang:work/noahmp-phase3b/logs/h5-overnight/state.json docs/gpu-results/h5-overnight-state.json
```

只有 JSONL 中 `event=result, status=measured` 且三次 measured 样本完整、affinity
单核、checksum 一致时，才能把 `seconds` 填入表。GPU 对照必须取新 checksum 门
通过的 JSONL；旧 `h5-gpu.jsonl` 只作历史记录。比值须使用相同 N 和计时边界。
手写 CUDA 即使 checksum 通过，也须独立通过 H4 才能作为最终性能验收结果。

新增有效结果时更新 `docs/BENCHMARK.md`、`docs/GPU_SUMMARY.md` 顶部三句话和
README GPU 段，保留 Windows 历史数据并注明版本语义差异。每 1–2 小时有新证据
就 commit + push `gpu/migration`，无变化时安静等待。网络 push 失败可重试，不
用 force push。J4 后续 profiling 的范围和预算以最新工作入口为准。

全部样本完成或截止后，写清所有实际结果/缺口、H5 是否有完整基线及 N=1e5
是否达到 100×，push 文档；J1–J5 完成或明确记录无法继续的缺口后才结束 heartbeat。若没有生产代码改动，
不必重复已保存的全量 pytest。**H4 未通过时，ROADMAP 阶段 3 不得勾选完成，
不得 checkout/merge/push main。**

## 源码与日志可追溯性

远程 HTTPS 拉取曾停滞，因此后续源码通过 SCP 同步到该专用工作目录；远程
Git HEAD 仍为 `2461a95` 且工作区有修改。H6 日志的 `git_revision` 只表示这个
基础提交，不能当作被测源码的完整版本。实际文件以
[source-manifest.json](gpu-results/source-manifest.json) 为准：src/tests/benchmarks
共 **47 个 Python/CUDA/shell 文件**经 CRLF→LF 规范化后，本地与远程 SHA256
逐个一致。日志与对应代码一起提交到本地 `gpu/migration`。

远程 `.nsys-rep` 等大文件保存在 `repo/profile_output/`，不纳入 Git；紧凑统计
已保存于 `docs/gpu-results/`。不要在远程工作目录执行破坏性 reset/clean。
