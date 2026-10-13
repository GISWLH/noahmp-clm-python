# M3 跨环境有界检查

核验日期：2026-10-13（America/New_York）。生产源码采用 `git archive b954d90`，
即 M1 一次尝试回退后的最终物理。每份本项目运行 JSON 均保存源码逐文件 SHA256；
`git archive` 也受本机换行配置影响；逐文件保存原始字节hash，并规范化为LF后
与 `b954d90` 对照文本，不根据运行系统猜测换行或源码身份。
本节为能运行与结果一致性检查，单次 smoke 耗时不是正式性能基准。

## EHPC：串行、多进程、两节点 MPI 均通过

独立目录 `/home/zhangyq/work/noahmp-closeout-m3-20261013/repo`，没有修改既有
环境检查目录。系统 Python 3.6.8 不满足项目要求；本次在共享 home 安装独立
`/home/zhangyq/venvs/noahmp-closeout`：Python **3.11.9**、NumPy **2.2.6**。
计算节点是 CentOS 7 / glibc 2.17、双 Xeon Gold 6248R @ 3.00 GHz，48 物理核，
本次 `compute002` 实查内存 **404141895680 bytes（376.4 GiB）**。
PBS 18.1.1、Open MPI 3.0.0。OMP/OpenBLAS/MKL 线程数均为 1。

同一合成输入为 **100 柱 × 48 步，float64，seed=20261013**，含 8 类情景和
柱内异质扰动。串行、4 进程和 MPI 所有输出均有限。两种并行方式对串行的
完整 **7000 个浮点终态、3300 个整数控制值和 57600 个逐步诊断值逐位相同**，
最大绝对差为 0；因此也通过预定 `rtol=1e-6, atol=1e-6` 精度档。

| 方式 | PBS / 实际节点 | 计算时间 s | 含输出和校验的脚本墙钟 s | 结果 |
|---|---|---:|---:|---|
| 串行 | `1.manager` / compute002 | 4.494118 | 4.539174 | 通过 |
| 4 进程分柱 | 同一 `1.manager` / compute002 | 1.238447 | 1.272282 | 对串行逐位相同 |
| MPI rank 0，50 柱 | `2.manager` / compute002 | 2.348859 | 2.370149 | 有限且完整 |
| MPI rank 1，50 柱 | `2.manager` / compute001 | 2.282925 | 2.323572 | 有限且完整 |
| MPI 汇总与串行对照 | `2.manager` | 最慢 rank 2.348859 | 汇总单独 0.040933 | 对串行逐位相同 |

两作业 `Exit_status=0`，PBS 记录整作业墙钟分别 7 秒、3 秒（整数秒粒度）。
MPI 的汇总时间不包含 `mpirun` 和各 rank 计算，不能拿 0.040933 秒冒充整次运行。
每节点一个 Open MPI rank，按全局柱号奇偶分配；柱间没有状态依赖，结果通过
GPFS 独立文件汇总。检查每柱只出现一次、两个不同主机、各 rank 源码一致。
这覆盖跨节点分柱计算，不是 MPI 通信吞吐测量。

三个完整结果的共同 SHA256：
`e6532abb184518dbe85a3818f0eb8f9f1ede6ed8f48b26f00c9e0fdacecd5a83`。
输入 SHA256：`3c41ee78e10fd2f8bf982dd9513a24a7025f0aae7dc739fcb838b5815c6282c4`。
checksum 由完整浮点终态、整数和逐步诊断连同 dtype/shape 组成；另做完整逐值比较，
不是仅凭 checksum 宣称数值通过。

原始证据：[串行](environment-checks/m3-ehpc-b954d90/serial.json)、
[4 进程](environment-checks/m3-ehpc-b954d90/pool.json)、
[MPI 汇总](environment-checks/m3-ehpc-b954d90/mpi.json)、
[单节点作业日志](environment-checks/m3-ehpc-b954d90/pbs-serial-pool.log)、
[MPI 作业日志](environment-checks/m3-ehpc-b954d90/pbs-mpi.log)。
完整 NPZ 保存在上述远端目录 `outputs/ehpc/`，本地副本在忽略目录
`outputs/m3-env/`。运行命令和 PBS 脚本见 [benchmarks/ehpc/README.md](../benchmarks/ehpc/README.md)。

## HF ZeroGPU：历史环境检查与本项目短测

使用本地已有认证，不查找或索要新凭据。以下为**删除前的历史检查**：原有
[WeatherLearn 演示 Space](https://huggingface.co/spaces/LonghaoWang/weatherlearn-fengwu-smoke)
在 07:38:53 EDT 检查时为 RUNNING，源码提交
`ac601ba2f1f4a35e74229fc0e66819128c22fc4c`。本轮调用现有 `/run_cuda_ping` 成功：

| 项目 | 本轮实测 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb |
| 软件 | PyTorch 2.13.0+cu130；本次接口未报告 Python 或 JAX 版本 |
| 运算 | 64 × 64 随机 CUDA 矩阵乘法，同步后全部有限，`STATUS=OK` |
| 请求墙钟 | 2.452212 秒，含网络/调度；不是纯 GPU 计时 |
| 配额消耗 | 1.269892 GPU 秒；调用后 2397.641964 秒，`overquotaUsed=0` |
| 峰值分配 | 0.0313 GiB，不能当作设备总显存 |
| checksum | 现有接口不返回数值结果/checksum，故未测；不补造 |

当时本项目的 **JAX 1024 × 48 parity 尚未测**：已有 Space 只提供 WeatherLearn / PyTorch
接口，没有 JAX/Noah-MP 执行入口；申请独立私有 ZeroGPU Space
`LonghaoWang/noahmp-clm-gpu-smoke` 时 HF 返回 HTTP 400：账号已有 10 个 ZeroGPU
Spaces，达到上限 10。因此未创建本项目 Space、未上传源文件、未更改/停用其他
Space、未开通付费硬件。该次阻塞按 M3 有界环境例外记录未测，不能把 CUDA ping 当作
Noah-MP 或 JAX 的兼容性通过。

[原始 HF 记录](environment-checks/m3-hf-zerogpu-b954d90.json)保存 runtime、两次
配额与原始 SSE 输出。已备好 [Noah-MP Space 模板](../examples/hf_zerogpu/README.md)，
后续在用户清理名额后完成部署，实际检查见下一节。
模板强制 JAX CUDA、单次 GPU 上限 120 秒、输出真实差异和完整 checksum，
不允许 CPU 静默回退。HF 对动态分配的官方兼容说明主要覆盖 PyTorch，
JAX 仍须实际验证：[官方 ZeroGPU 文档](https://huggingface.co/docs/hub/spaces-zerogpu)。

### 用户清理名额后的续测

上述容量阻塞之后，用户明确要求删除创建时间最旧的 4 个 ZeroGPU Spaces，
已删除并核验不存在，含上述 WeatherLearn Space。因此它的链接不再代表当前可用
演示，历史检查记录保留。删除清单与时间见
[删除记录](environment-checks/m3-hf-deleted-oldest-four.json)。
名额释放后已创建独立私有
[Noah-MP Space](https://huggingface.co/spaces/LonghaoWang/noahmp-clm-gpu-smoke)，
采用 `b954d90` 生产源码、预冻结 H4 输入与 M2 口径，硬件仅 ZeroGPU。
已实测 float32 的 **1024 柱 × 48 步完整逐步状态/诊断**：全部有限，整数对
NumPy64一致，逐步回放与单次48步批运行逐位相同。M2 每字段门通过；仅 canw
有153/49152项超差，通过率99.6887207%，其余8字段100%。residual 中位数
0.000293731689453125、p99 0.001819610595703125 W/m²；表面/逐步水量门均通过。
153项与xuanze同一近空冠层聚类，因果分解见
[独立审计](gpu-results/m2-independent-audit.json)，不再次把聚类标签当作因果证据。

实测设备为 **NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb**，
设备报告总显存 **50868518912 bytes**；Python 3.12.12、NumPy 2.5.3、
JAX 0.11.2、PyTorch 2.13.0+cu130。JAX强制CUDA，实际设备 `cuda:0`。
CPU参考先运行156个唯一输入并映射到1024柱；GPU实际运行全部1024柱。

| float32 smoke 项目 | 本轮实测 |
|---|---:|
| NumPy参考 | 5.301340 秒 |
| GPU含冷编译、整批运行、逐步回放、完整读回 | 16.553062 秒 |
| 函数总墙钟 | 24.028216 秒 |
| API请求墙钟（含网络与调度） | 26.648625 秒 |
| ZeroGPU配额消耗 | 18.449742 GPU秒，`overquotaUsed=0` |

完整JAX轨迹SHA256为
`8851a29bcab8c5396652d1ce269d13c012175b13534317aa5d3aaba5a496e139`，
与xuanze RTX4090的M2 JAX32结果完全相同；完整NumPy参考checksum也相同。
该结论只覆盖这份固定输入。[跨环境checksum与源码复核](environment-checks/m3-hf-cross-environment-comparison.json)
记录比较；部署源码与 `b954d90` 文本一致，Windows归档换行转换的字节差已标明。
Space私有页面HTTP200、命名API实际返回报告；测试commit为
`413e5003ef5b14b89bf9117aecf366826cf7c86e`。
[完整float32原始记录](environment-checks/m3-hf-noahmp-float32.json)保留源码hash、
逐字段差异、153行超差、原始SSE、配额。这里的耗时为一次短测，不进入正式性能表。

可选 **JAX64 也完成同一1024×48短测**：完整有效状态（沿用既有雪层/保留槽mask）
和全部诊断对NumPy64均通过 `rtol=1e-6, atol=1e-6`；全部有限、整数相同，
逐步回放与48步整批运行逐位相同，表面/水平衡门也通过。另保留M2纯相对口径下
canw的46/49152个极近零差异（通过率99.9064128%）；其余8字段100%。
46项均为降雨配置耗尽尾部，一端为0、另一端最多1.6940658945086007e-21 m。
这46项不影响上述带既有绝对容差的JAX64严格parity，逐项值仍在原始记录中。
CPU参考5.230822秒，GPU含JIT/整批与逐步回放/全部读回17.183159秒，
函数总墙钟24.424463秒，API请求25.753589秒；配额消耗18.862804 GPU秒，
两次本项目测试共消耗37.312546 GPU秒，调用后剩2360.329418秒、无超额计费。
完整JAX64轨迹SHA256：
`73fec6e7137d507cc6c104727cb2301097b001ff32d85f36f216c8df631e4dd7`。
[float64原始记录](environment-checks/m3-hf-noahmp-float64.json)与
[最终Space状态](environment-checks/m3-hf-noahmp-final-runtime.json)保留证据。
最终演示commit为`4259f47a3468e56e3747b03231852f240ea594e5`，相对float32实测版
仅将可选float64校验改为既有有效层mask，生产源码、输入、x32检查均未改。
本轮没有测试ZeroGPU上的CuPy/手写CUDA，也不宣称整柱焓守恒或真实物理验证。

## xuanze

最终 `b954d90` 生产源在物理 GPU1 / RTX4090 上重测；Python3.12.2、NumPy2.5.4、
JAX0.11.2、CuPy14.2.0、nvcc12.4.99。GPU0同时有年度作业，属于共享机器测量。
实际源码hash前缀为 JAX `6ec9039c`、CUDA `e1f7cf0d`、loader `ee5846de`。
N=100000 × 48步，每后端3次预热、3次正式，**每次完整输出校验门均通过**。

| 后端 | 含校验正式中位数 s |
|---|---:|
| JAX64 | 22.04212154448 |
| JAX32 | 3.227870002389 |
| CUDA32 | 0.2036111056805 |

CUDA仅同步计算的正式中位数为0.007183358073秒，不能与上述含校验时间混用。
原状态 `timing_validated_h4_unverified` 保留；性能重测不把原H4红灯改为通过。
[原始运行记录](gpu-results/m3-xuanze-benchmark.jsonl)保留各次输出校验和计时。
最终全量pytest为 **147 passed / 2 failed，143.13秒**，两项失败仅为原H4的
1/48步 NumPy64 严格门；8×48 parity相关测试通过，没有修改原H4阈值。
[pytest日志](gpu-results/m-final-pytest.log)与
[源码元数据](gpu-results/m-final-pytest-metadata.json)记录实际GPU、退出码和源码未变。
H6选项抽查为 **92 passed / 11 unsupported_expected / 0 failures**，墙钟
176.35秒；[结构化结果](gpu-results/m-final-options.json)和
[逐项记录](gpu-results/m-final-options.jsonl)保留所有配置。

首次直接运行归档中的`build.sh`因CRLF失败，
[原失败日志](gpu-results/m-final-nvcc-crlf.log)保留。随后添加`.gitattributes`
强制`.sh/.pbs`为LF，用原编译旗标重编译，**exit 0**；只有旧`snow_flux`
未使用的1条warning，没有宣称零warning或变更生产物理。
[LF修正后日志](gpu-results/m-final-nvcc-lf.log)与
[编译元数据](gpu-results/m-final-nvcc-lf-metadata.json)提供对应源码和命令。
