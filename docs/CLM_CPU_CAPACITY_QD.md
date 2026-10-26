# EHPC 年度终态与容量恢复（QD）

## 恢复后首8窗已闭合，两后继Qia失败

[22件实际闭合与失败小件](v1-results/pj-capacity-resume-qd-b-first-closures-qp/manifest.json)
确认1846—1853全部F0/Stageout0、worker0、各480/480步；共新增3840步，Cng
闭合13968/13968/13488/13488，Qia四根7248。每根原raw与新closed的SHA匹配，
原状态接续与源/硬账条件保持，不计全年或收敛。只读时6个后继仍running。

1864/1865两条CN-Qia field-capacity后继实际F1/Stageout0：7248后started28/
completed27，在绝对步7275抛原Newton iteration30、原消息残差13.757259 W/m²。
closed仍7248，失败partial不计closed，不自动重试。两份约282MB原archive仍在
远端，完整路径/SHA与成员清单已封存，本机未下载或数值解码大包。

失败外层receipt的source_unchanged=False原样保留；它只在整窗成功后更新，
不能据此认定改源。原内层run/receipt的前后验证为True，实际控制代码在模型
失败后仍hash全584源；当前36stage与NM archive/manifest亦实际exact。两作业
临时source目录已被PBS清理，未冒称事后重查那两份目录。原单位混合问题见
[保存路径分析](CLM_CANOPY_PHASE_ENERGY.md)，不改写这两份原消息。

## 17:17UTC后：服务恢复与八条原链实际启动

[管理员恢复原件](v1-results/pbs-root-recovery-qr/manifest.json)不含凭证；原日志
与完整资源回收包独立核验，日志快照不含最终PTY退出，另明确其工具回执转录范围。

管理员入口已获授权，原datastore进程保持，没有重建数据库。16:52正常启动尝试
直接报告spool创建ENOSPC；新pbs_server在后台等待，未重复启动。GPFS远端
fileset扩大inode请求实际拒绝退出1，原50176上限不变。随后只清理已完整备份、
逐文件核验的27个pytest缓存，共117文件/81目录；模型源、状态和结果删除0。
[28件备份/guard/回收原件](v1-results/pbs-inode-repair-qm/manifest.json)。df读数
仍为0时不宣称释放了多少inode；实际零字节spool探针创建成功、删除完成，PBS
Active/三节点free，原1813 F0、1831 F1及Stageout0保留。

新独立作业1843 F0、1844原模型F1、1845 F0均实际完成并交付Stageout0，确认
调度和结果取回恢复。1844科学失败未归一为成功。管理员会话已退出，凭证未存储。

qd-b[39件唯一启动原件](v1-results/pj-capacity-resume-qd-b/manifest.json)记录控制器
99446从8个原成功父状态唯一接续；首快照1846—1853实际R。Cng两根
13488→13968、两根13008→13488；Qia四根6768→7248。原36项stage依赖在
两主机逐字节核验，原进度和8父状态不变，容量门仍12GiB/20000inode；真实
F0/Stageout0闭合后才接下480步窗，17520即停，不开cycle2或评分。活动窗不是
已完成年，JN/GPU的7275失败年仍不续算。17:59:44UTC只读checkpoint为Cng
13872/13872/13344/13344、Qia四根7056，均超过原父；当前窗尚未取得新F0回执，
不把部分checkpoint计作闭合窗、全年、收敛或评分。以下停服快照保留为历史。

最新[16:32UTC服务阻塞原件](v1-results/pbs-service-outage-ql/manifest.json)：同一
1831/1813句柄反复拒连超过30分钟，qstat255；pbs_server无进程，现账号sudo
要求密码。qd-b及新CPU诊断均保持未启动，等实际服务恢复再核对全部原父门。
inode0和日志切日只是观察，未证实停服原因；[阻塞与实际绕行](GPU_BLOCKERS.md)。

15:46后新独立恢复预检核对八个最新父状态、原源/控制及容量实际0，原进度不改。
新控制器提交1831—1838后八窗均F1/Stageout0、0物理步：原worker的完整stage
检查仍要求首个48步归档，而它们随旧容器副本回收，报初始归档FileNotFoundError。
恢复预检漏查这项旧依赖，不能计作年度物理失败或成功接续。
[133件依赖修复原件](v1-results/pj-dependency-repair-qd-b/manifest.json)已保留每案
F1/Stageout0、0启动/完成步及全部原目录；8初始归档同时对原stage及35GB备份
直接原bytes核验，已exclusive恢复到两主机，36/36原stage文件全exact，最新
8父状态哈希不变。16个原失败目录逐文件核验后rename到新审计路径，0删除/覆盖。
旧独立review漏查worker全stage依赖的局限亦纠正并保留旧原件。

16:01起manager PBS qstat拒连errno111；SSH和文件读写正常。独立观察有
pbs_comm/pbs_sched而无pbs_server，系统内存约369GiB可用，根盘inode充足，
HOME仍inode0；这些观察不能证明具体PBS退出原因。新qd-b控制器仅准备、未部署
或启动；须服务实际恢复后再核对原8个F0父回执及容量，不能绕过原门或续失败年。

2026-10-26 07:50 UTC 原链只读快照和随后容量恢复的
[33 件原始证据](v1-results/cpu-capacity-qd/manifest.json)保留实际命令、SSH
退出、完整调度记录、原进度及备份与回收逐文件哈希。没有运行数值模型或年度评价。

PJ 原八条 NumPy 基线同年链的控制器已退出：Cng 两根闭合到 13488、另两根
13008；Qia 四根闭合到 6768。最后各窗均实际 F0/Stageout0，随后原
12 GiB/20000 inode 容量断言终止控制器，无新全年。完整状态和原错误保留；
未通过降低门槛、重置到 48 步或重跑旧控制器恢复。

旧 JN 官方配对链：Qia 成功闭合到 7152，1436 作业实际 F1/Stageout0，
保存到 7275 的失败报告为 Newton iteration29、13.758249 W/m²。
Cha、HaM 各连续成功至 17520；旧保存报告仍显示 running，与完成回执并列保留。
LU1536 为 H，尚未开始独立三站全年评价；这两份旧源完整年不能算 NM 通过。

`/tmp` 归档副本使 compute002 空间降到原容量门以下；HOME GPFS 虽有字节
余量但 inode 为零，两个备份建目录尝试均真实失败，未删数据。
随后将 manager 的全部 174 份 closed 容器只读流到 Windows 备份，
34,879,119,360 字节，SHA256
`e45de1899393b010dcc81691cd9eb86323e5e410c9d4a67ec79c4c4fffd03cf4`。
本机以标准库核验整个包及每个成员的哈希、大小和 offset，没有解码模型数组。
大包留在 outputs，不进入 Git；174 个容器的证明文件进入上述证据包。

回收前重新确认旧控制器不存活、进度未变、所有目标容器与已验证备份完全相同，
只删除旧 closed 容器副本；manager 最新 8 个父状态容器保留。两主机实际 SSH0：
manager 空闲 13,026,697,216→43,969,482,752 字节；compute002
11,891,142,656→42,833,989,632 字节。原 actual/raw 结果归档删除 0，输入及
物理状态改变 0。旧缓存另先备份后只回收 4 文件/3 目录，未声称 inode 立即释放。

容量恢复不等于模型继续运行。恢复须重新核实八个父状态、源/控制哈希及实际
容量，只从成功闭合父状态延续原同年，17520 即停，不开 cycle2 或评分；
Qia 已失败的 JN/GPU 年不续算。
