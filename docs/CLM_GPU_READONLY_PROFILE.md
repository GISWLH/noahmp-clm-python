# xuanze实态回归等待核对（2026-10-17）

稳定Qia实态的三个GPU测试已经正常退出：pytest为2289.114秒，控制器记录
2338.375377秒，零失败/跳过、486份冻结源未变。CUDA32/JAX32全字段最大差0。
JAX32对独立NumPy的H/LE绝对差0.000470721/0.005369960 W/m²，GPP/NEE差
2.55666e-8/1.14972e-7 gC/m²/步；JAX64 H/LE差6.03961e-14/1.88294e-13 W/m²。
原收敛门和同精度1e-6门保持。

CUDA32/JAX32独立水/热/C/N残差为2.77193e-5 kg/m²、9.697795 J/m²、
0.000160255 gC/m²、3.98606e-7 gN/m²；原数值界为0.004026193 kg/m²、
2567.553823 J/m²、0.459763704 gC/m²、0.019505741 gN/m²。
[完整原日志/XML及调度记录](v1-results/most-stable-target-prefix-a/manifest.json)已公开。

## 只读快照所能支持的结论

采用隔离安装的py-spy 0.4.1，对本任务所属PID做三次即时栈采样，未采集局部
变量、环境或原生二进制跟踪，未修改运行源码。公开证据为
[原JSON、GPU快照、失败包装器及恢复记录](v1-results/most-readonly-stack-a/manifest.json)。

| 19:07:59 UTC的快照 | 实际栈顶 | 能支持的观察 |
| --- | --- | --- |
| Qia子进程，两次 | JAX `clear_backends` → `clean_up` | 这两个瞬间已在后端清理，进程存活不等于测试体仍运行 |
| J年度，一次 | CUDA冠层相变表面 → MOST调度 | 该瞬间仍在年度物理调用内；随后年度检查点继续推进 |
| 同时GPU利用率 | GPU0 100%，GPU1 0% | 共享设备的即时总利用率，不能归因到单个进程 |

三次采样集中在同一秒，没有统计代表性，不能测出热点占比、显存峰值、
传输带宽或速度收益。38分09秒是pytest记录的整个测试时间，不能据清理栈
把它全归为清理；控制器与pytest的时间差也不能全部归为一个原因。

原包装器完成采样后，元数据哈希函数对`__file__`字符串调用`read_bytes`而
报AttributeError。恢复工具只读取已有文件，没有重复采样；原每次调用耗时
未保存，明确记为不可用。这个诊断错误不计作模型或测试失败。

当前采取的调度措施是避免重复执行：19:24UTC，已核对完整队列包含余下三个
目标文件、源哈希及本任务PID/UID/目录/参数/进程组后，停止独立定向队列。
已完成的两文件六项保留；活动反馈精度文件的中断日志不算通过，也不算
数值失败。SIGINT后仅向仍活跃且身份未变的本任务子进程发送SIGTERM。
完整325项队列和J年度PID身份未变，分别继续使用GPU1与GPU0。余下检查仍须
由完整队列实际完成；没有完整目标、全量、年度或科学验收结论。

后续先收齐这些计算结果。只有取得代表性、可分离的计时证据后才考虑新的
性能修改；此次快照不支持直接修改JAX清理或冠层物理内核。

## 插桩与文件变更清单

生产代码修改为0。以下均为新建的外部诊断文件或隔离工具，行号为“—”；
本机原件位于`outputs/v1/`，远端原件位于`/home/lhwang/work/noahmp-v1/`。
采样、恢复及归档包装器原件已随上面两份证据归档；清单工具原件保留于本机
忽略目录与远端根目录，其结果单独公开。第三方二进制只发布
[逐文件大小与SHA256清单](v1-results/most-readonly-profile-dependency-a.json)。

| 文件/相对位置 | 类型 | 内容 | 行号 |
| --- | --- | --- | --- |
| `probe_most_slow_a.py` | 新建 | 核身份后只读栈/GPU采样，保留原元数据错误 | — |
| `recover_most_slow_a.py` | 新建 | 只读恢复采样元数据 | — |
| `archive_most_stable_profile_a.py` | 新建 | 安全采样与测试前缀归档 | — |
| `inventory_most_profile_dependency_a.py` | 新建 | 只读隔离依赖文件清单 | — |
| `profile_output/most-slow-a/00-target-3981208.json` | 新建 | 首个目标栈 | — |
| `profile_output/most-slow-a/01-annual-3983588.json` | 新建 | 年度栈 | — |
| `profile_output/most-slow-a/02-target-3981208.json` | 新建 | 第二个目标栈 | — |
| 上述三个同名`.stderr.txt`文件 | 新建 | 三个空stderr文件 | — |
| `profile_output/most-slow-a/gpu-snapshot.txt` | 新建 | 两张卡即时利用率/显存 | — |
| `profile_output/most-slow-a/summary.json` | 新建 | 恢复元数据及原错误说明 | — |
| `profile_output/most-slow-a/probe_most_slow_a.py` | 新建 | 原采样包装器副本 | — |
| `profile_output/most-slow-a/recover_most_slow_a.py` | 新建 | 恢复包装器副本 | — |
| `most-readonly-profile-dependency-a.json` | 新建 | 工具文件清单，公开到docs/v1-results | — |
| `profile-tools-pyspy-a/bin/py-spy` | 新建 | 隔离工具可执行文件 | — |
| `profile-tools-pyspy-a/py_spy-0.4.1.dist-info/INSTALLER` | 新建 | pip安装元数据 | — |
| 同目录`METADATA` | 新建 | 包版本元数据 | — |
| 同目录`RECORD` | 新建 | 安装文件记录 | — |
| 同目录`REQUESTED` | 新建 | 安装标记 | — |
| 同目录`WHEEL` | 新建 | wheel元数据 | — |
| 同目录`licenses/LICENSE` | 新建 | 许可证 | — |

证据副本为`docs/v1-results/most-readonly-stack-a/`的12项清单文件，以及同名
远端/本机忽略目录中的`most-readonly-stack-a-evidence.tar.gz`；回归前缀另列
`most-stable-target-prefix-a/`。这些外部文件可独立清理，不需回退物理源码。
