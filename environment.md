# 计算与 GPU 测试环境

核验日期：2026-10-13（America/New_York）。适用于独立格点模拟、数值计算、CPU/GPU 后端对照及类似项目。

共享维护源：`~/.agent/commands/environment.md`；本仓库根目录的 `environment.md` 是同步副本。修改公共环境信息时先更新共享源，再同步到项目。SSH 配置和密钥保留在 `~/.ssh`，本文不保存密码、私钥或访问令牌。

## 1. 环境选择

### 2026-10-13 Noah-MP 收尾补充

以下正文保留早间硬件/访问检查快照；当前项目实测以
`noahmp-clm-python/docs/ENV_TESTS.md` 为准。

- EHPC 已建立共享环境 `/home/zhangyq/venvs/noahmp-closeout`，Python3.11.9 /
  NumPy2.2.6。Noah-MP N100×48 的串行、4进程、两节点MPI完整输出逐位一致；
  PBS 作业 `1.manager`、`2.manager` 均退出0。
- Wang 明确要求删除创建日期最旧的4个ZeroGPU Space，已删除
  `weatherlearn-fengwu-smoke`、`flux2-klein4b-studio`、`gemma4-e4b-omni-chat`、
  `gpt-oss-20b-lab`。正文 WeatherLearn ping 是删除前的历史证据，其链接已失效。
- 新建独立私有 `LonghaoWang/noahmp-clm-gpu-smoke`，使用已有ZeroGPU权限，未付费升级。
  已在实际 Blackwell MIG 2g.48gb 上完成本项目 JAX float32/float64 1024×48 smoke，
  float32完整轨迹checksum与xuanze相同，float64严格parity通过；界面和API可用。用现有HF认证访问：
  https://huggingface.co/spaces/LonghaoWang/noahmp-clm-gpu-smoke 。
- 环境可运行不代表全部项目发布门通过；真实站点的canw近零相对误差仍须按项目
  固定验收门记录，不能用跨环境checksum一致替代真实数据精度验收。

| 环境 | 可用资源 | 适合的测试 | 证据状态 |
|---|---|---|---|
| EHPC 超算 | 3 个计算节点，共 144 物理核、约 1.1 TiB 内存；另有管理节点 | CPU 基线、独立任务批处理、多进程、MPI | SSH 已验证；三节点 PBS/MPI 冒烟测试通过 |
| xuanze | 2 × RTX 4090，每卡 24564 MiB；40 物理核 / 80 线程，124 GiB 内存 | JAX/CuPy/CUDA 开发、数值一致性、持续 GPU 基准 | 免密、硬件和现有 Python 包已复核；既有 GPU 执行记录见项目文档 |
| HF PRO / ZeroGPU | 本次实际分配 RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb | 短时云端兼容性检查、交互式演示 | PRO 与配额已核验；PyTorch CUDA 矩阵运算通过；整卡规格仅查官方资料 |

不同机器的 CPU 和 GPU 数量不能直接换算成加速比。以同一代码、输入、精度和计时范围的实测结果比较。

## 2. SSH 入口与免密状态

以下别名已配置在本机 `C:\Users\hhuwl\.ssh\config`，均使用现有 `~/.ssh/id_ed25519_github` 密钥。超算公钥已追加到共享 home 的 `authorized_keys`，原有条目保留。

| SSH 别名 | IP | 用户 | 主机角色 |
|---|---|---|---|
| `ehpc` / `ehpc-manager` | 10.169.135.247 | zhangyq | manager，登录与提交任务 |
| `ehpc-c0` | 10.169.135.248 | zhangyq | compute000 |
| `ehpc-c1` | 10.169.135.249 | zhangyq | compute001 |
| `ehpc-c2` | 10.169.135.250 | zhangyq | compute002 |
| `xuanze-neiwang` | 10.168.38.211 | lhwang | GPU 服务器；系统 hostname 为 localhost.localdomain |

```powershell
ssh ehpc
ssh ehpc-c0
ssh xuanze-neiwang

# 非交互验证：连接失败时直接报错，不退回密码认证
ssh -o BatchMode=yes -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no ehpc hostname

# 在本地项目的父目录上传；项目只需上传一次
scp -r .\noahmp-clm-python ehpc:/home/zhangyq/
```

四台超算的 SSH 别名和原始 IP 均可使用公钥登录；主机密钥检查为 `StrictHostKeyChecking yes`。本次免密范围为本机到上述普通用户账号；未配置 root 免密，也未分发私钥到计算节点。MPI 可使用已有的 PBS TM 集成。

这些地址属于内网；当前电脑已实测可达，其他网络需要具备相应路由或 VPN。换电脑时应使用那台电脑自己的公钥，并重新配置 SSH 别名。

## 3. EHPC 超算

### 硬件与调度

四台机器均为双路 Intel Xeon Gold 6248R @ 3.00 GHz，每台 48 物理核、48 逻辑 CPU，系统可见内存约 376 GiB。PBS 三个计算节点均登记 `ngpus=0`。

| 项目 | 实查结果 |
|---|---|
| 系统 | CentOS Linux 7，内核 3.10.0-1160.15.2.el7.x86_64 |
| PBS | 18.1.1；队列 `workq`；提交命令 `qsub` |
| 当前计算池 | compute000、compute001、compute002；manager 未加入计算池 |
| Open MPI | 3.0.0，`/opt/openmpi/3.0.0`，含 PBS `ras:tm` / `plm:tm` 组件 |
| 系统 Python | Python 2.7.5 / Python 3.6.8 |
| 编译器 | GCC / gfortran 4.8.5 |
| 本地系统盘 | 每台约 99 GiB，节点间独立 |
| 租期 | 用户提供的到期日：2026-11-09 |

当前 Noah-MP 项目的 `pyproject.toml` 要求 Python >=3.10、NumPy >=1.24，系统 Python 3.6.8 不能直接满足要求。本轮已在共享 home 建立 `/home/zhangyq/venvs/noahmp-closeout`（Python3.11.9、NumPy2.2.6），并完成实际串行、多进程和两节点运行；系统 Python 保持不变。

### 共享存储

文件系统类型为 GPFS，`df` 显示约 5.0 TiB 总容量、4.7 TiB 可用（检查时快照，并非已确认的账号配额）。四台机器挂载一致：

| 路径 | 后端 | 使用方式 |
|---|---|---|
| `/zhangyq` | 共享文件系统顶层 | 当前 root:root、0700，普通用户不能直接进入 |
| `/home` | 同一文件系统的 `/zhangyq/home` 子目录 | 独立挂载入口，普通账号可正常使用自己的 home |
| `/home/zhangyq` | 共享用户目录 | 代码、虚拟环境、输入、日志与输出的工作入口 |
| `/opt` | 同一文件系统的 `/zhangyq/opt` 子目录 | 共享 MPI 等软件 |

可按项目放在 `~/work/<project>`，环境放在 `~/venvs/<project>`。这些为建议目录，按实际项目创建。计算输出在到期前同步到本地或 NAS。

### PBS 最小使用方法

在 manager 提交作业，批量计算通过调度器申请资源。资源申请不会自动使串行程序并行；Python 多进程通常限于单节点，跨节点需要 MPI 或独立作业分发。

保存为 `smoke.pbs`（Linux 换行）：

```bash
#!/bin/bash
#PBS -N env-smoke
#PBS -q workq
#PBS -l select=3:ncpus=1:mpiprocs=1:mem=256mb
#PBS -l place=scatter
#PBS -l walltime=00:02:00
#PBS -j oe
set -eu
cd "$PBS_O_WORKDIR"
export PATH=/opt/openmpi/3.0.0/bin:$PATH
export LD_LIBRARY_PATH=/opt/openmpi/3.0.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cat "$PBS_NODEFILE"
mpirun -np 3 /bin/hostname
```

```bash
qsub smoke.pbs
qstat -u zhangyq
pbsnodes -av
qstat -xf <job-id>       # 历史状态与 Exit_status
qdel <job-id>           # 仅在需要取消时使用
```

2026-10-13 实测作业 `0.manager`：三个节点各申请 1 核，`mpirun -np 3 /bin/hostname` 分别返回 compute002、compute001、compute000，最终 `job_state=F`、`Exit_status=0`。原始脚本与输出保存在 `/home/zhangyq/environment-check-20261013-i5f8c4/`；这是调度和跨节点启动测试，不是性能基准。

单节点 48 核任务可申请 `select=1:ncpus=48:mem=300gb`，并让程序实际启动相应数量的工作进程。三个独立案例可以分别提交三个单节点作业。MPI 程序使用多节点时，按进程数设置 `mpiprocs`。NumPy 多进程基线通常将 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 设为 1，避免每个进程再开一组线程。

## 4. xuanze GPU 环境

| 项目 | 2026-10-13 实查结果 |
|---|---|
| 系统 | CentOS Stream 9 |
| CPU | 2 × Xeon Gold 5218R @ 2.10 GHz，40 物理核 / 80 逻辑 CPU |
| 内存 | 124 GiB |
| GPU 0 / GPU 1 | NVIDIA GeForce RTX 4090，各 24564 MiB；sm_89（既有环境记录） |
| NVIDIA 驱动 | 570.153.02；nvidia-smi 显示驱动支持 CUDA 12.8 |
| CUDA Toolkit | `/usr/local/cuda-12.4/bin/nvcc`，12.4.99 |
| Python | 3.12.2 |
| 现有项目环境 | `/home/lhwang/venvs/noahmp-phase3b` |
| 现有项目目录 | `/home/lhwang/work/noahmp-phase3b/repo` |
| 环境内主要包 | NumPy 2.5.4、JAX / jaxlib 0.11.2、cupy-cuda12x 14.2.0 |
| 磁盘 | `/` 约 4.0 TiB；`/home` 约 9.0 TiB，检查时约 3.6 TiB 可用 |

两张 GPU 的显存各自独立。检查时两卡均有其他进程，运行前重新查看占用，选择合适设备。GPU 索引不等于空闲状态。

以下使用本项目已有环境；其他项目应新建独立环境并记录依赖版本：

```bash
ssh xuanze-neiwang
nvidia-smi
source ~/venvs/noahmp-phase3b/bin/activate
cd ~/work/noahmp-phase3b/repo

# 这里以 GPU 0 为例，实际按实时占用选择
export CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.4
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=src
python -c "import jax; print(jax.devices()); assert all(d.platform == 'gpu' for d in jax.devices())"
NOAHMP_REQUIRE_GPU=1 python -m pytest tests/test_gpu_parity.py -q
```

`CUDA_VISIBLE_DEVICES` 过滤后，进程内逻辑设备 0 对应选中的物理 GPU。既有 JAX 环境曾因系统 `LD_LIBRARY_PATH` 抢先加载 CUDA 库而初始化失败；这里仅清除当前 shell 的变量，配合 `JAX_PLATFORMS=cuda` 避免悄悄退回 CPU。驱动支持版本、nvcc 版本和 Python wheel 自带 runtime 是不同信息，应分别记录。

已有执行证据及尚未通过的数值门见项目的 `docs/GPU_ENV.md`、`docs/GPU_BLOCKERS.md`、`docs/BENCHMARK.md`。本次复核没有重跑 GPU 基准。

## 5. Hugging Face PRO / ZeroGPU

以下为 2026-10-13 查询的官方规格；本次实测覆盖默认的 48 GB 实例：

| 选项 | 官方硬件 / 显存 | 配额消耗 |
|---|---|---|
| `large`（默认） | 半张 RTX PRO 6000 Blackwell / 48 GB | 1× |
| `xlarge` | 整张 RTX PRO 6000 Blackwell / 96 GB | 2× |

PRO 当前含每天 40 分钟 ZeroGPU 配额、最高队列优先级，可托管最多 10 个 ZeroGPU Spaces。GPU 在 `@spaces.GPU` 函数执行时动态分配；当前支持 Gradio，默认单次时长 60 秒，可声明其他时长。官方兼容说明以 PyTorch 为主，JAX / CuPy / 自定义 CUDA 路线须单独验证。来源：[ZeroGPU 官方文档](https://huggingface.co/docs/hub/spaces-zerogpu)。

测试流程：创建 Gradio Space → 选择 ZeroGPU → 用 `@spaces.GPU` 包装短测试 → 在函数内采集 GPU 型号、显存、框架版本及结果 → 导出日志。排队和容器启动时间应与 GPU 计算时间分别统计。

这里可作为第三种硬件的兼容性验证目标。常驻付费 GPU Spaces 属于单独的硬件升级，见 [GPU Spaces 官方说明](https://huggingface.co/docs/hub/spaces-gpus)。

### 本账号实测结果

使用本地环境变量 `HF_TOKEN` 完成认证，账号 `LonghaoWang` 的 `isPro=true`。复用了已有的 [WeatherLearn FengWu 测试 Space](https://huggingface.co/spaces/LonghaoWang/weatherlearn-fengwu-smoke)，未修改或新建 Space。

2026-10-13 11:14:50 UTC（07:14:50 EDT）调用 `/run_cuda_ping`，执行 64 × 64 CUDA 随机矩阵乘法，显式同步并检查输出全部有限：

```text
torch=2.13.0+cu130
cuda_available=True
gpu=NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb
matmul_ok=True
peak_cuda_mem_gb=0.0313
STATUS=OK
```

| 实测项 | 结果 |
|---|---|
| Space 代码提交 | `ac601ba2f1f4a35e74229fc0e66819128c22fc4c` |
| HF runtime hardware 标识 | `zero-a10g`；这是服务标识，实际 GPU 型号以上述 CUDA 查询为准 |
| 调用墙钟时间 | 2.27684 秒，含远程调用开销，不是纯 GPU 性能基准 |
| 调用前 / 后可用配额 | 2400 / 2398.911856 GPU 秒 |
| 本次配额减少 | 1.088144 GPU 秒；`overquotaUsed=0` |
| 显存解释 | CUDA 名称报告 `MIG 2g.48gb`；未另行读取设备总显存字节数，`0.0313` 是 PyTorch 峰值分配 GiB |
| 测试范围 | CUDA 可用性、张量矩阵运算和有限性；未验证本项目的 JAX、CuPy、自定义 CUDA 或数值 parity |

原始结果在本项目 `docs/environment-checks/hf-zerogpu-2026-10-13.json`。不要将这次分配解释为长期独占卡，也不要将整卡 96 GB 或 H200/A100 写成已经实测的设备。

复测时使用安装了 `huggingface_hub`、`gradio_client` 的本地环境；`get_token()` 会读取环境变量或 HF 本地凭据，不要把 token 写进代码：

```python
from huggingface_hub import get_token
from gradio_client import Client

client = Client(
    "LonghaoWang/weatherlearn-fengwu-smoke",
    token=get_token(), verbose=False, analytics_enabled=False,
)
print(client.predict(api_name="/run_cuda_ping"))
```

每次复测前可通过 HF 官方配额 API `GET https://huggingface.co/api/spaces/zero-gpu/quota` 查询剩余额度，认证放在请求头。该接口的本次响应字段为 `base`、`current`、`resetsAt`、`overquotaUsed`；细粒度 token 需要相应的用量读取权限。实际配额随账号使用变化。

## 6. 真实测试数据位置

### ERA5：xuanze 的既有全球网格数据

用户此前在“排查 GPU 三项失败并验证真实数据”聊天中明确提供：

```text
SSH: xuanze-neiwang (lhwang@10.168.38.211)
/data/lhwang/ERA5_raw/ERA5_1940_2023/
├── surface/
└── upper_air/
```

2026-10-13 已只读确认目录及两个子目录存在、可列出。`1940_2023` 是现有目录名，本次未核验全时段完整性、变量清单、分辨率或缺测，不能据此宣称完整覆盖。该数据尚未接入本项目的真实网格测试；不能把合成 64800 柱测试写成 ERA5 验证。

### 通量：NAS WLH 的 FLUXNET 数据库

本地数据目录 / 门户代码仓库为 `F:\Data_center`，真实数据主体在 NAS WLH（10.168.18.205）：

```text
/volume2/Data/Global/FLUXNET-Data-Shop/
├── raw/                     # 原始数据归档
└── processed/site_csv/      # 已处理的站点 CSV
```

门户：[FLUXNET 数据入口](http://10.168.18.205:8013/fluxnet-shop/)。路径依据为 `F:\Data_center\docs\data-mapping.md` 和本项目 `docs/FLUXNET_PLAN.md`；本次 NAS SSH 复核未成功，因此目录描述按既有记录保留。

已经用于本项目 30 天真实数据 smoke 的长岭草地站 CN-Cng 原始文件：

```text
/volume2/Data/Global/FLUXNET-Data-Shop/raw/added_2026-06/ameriflux_downloads/FLX_CN-Cng_FLUXNET2015_FULLSET_2007-2010_1-4.zip
```

对应 [ZIP 下载入口](http://10.168.18.205:8013/fluxnet-shop/files/FLX_CN-Cng_FLUXNET2015_FULLSET_2007-2010_1-4.zip)。2026-10-13 HTTP HEAD 返回 200，Content-Length 为 45,384,649 字节，与已有来源记录一致；本次未重复下载和计算哈希。来源与 ZIP、完整 CSV、子集 SHA256 记录在本项目 `docs/gpu-results/k6-fluxnet-source.json`。

已提取的测试子集为 2007-07-01 至 2007-07-30，共 1440 个半小时记录：

| 位置 | 文件 |
|---|---|
| 当前 Windows 项目 | `D:\OneDrive\GitHub\noahmp-clm-python\outputs\k6\FLX_CN-Cng_FLUXNET2015_FULLSET_HH_20070701-20070730.csv` |
| xuanze 测试目录 | `/home/lhwang/work/noahmp-phase3b/k6-fluxnet-20261013/data/FLX_CN-Cng_FLUXNET2015_FULLSET_HH_20070701-20070730.csv` |

2026-10-13 已确认上述两个子集文件存在；本地还保留原 ZIP 和完整 HH CSV。原始数据在忽略目录或远端数据目录，不随 Git 提交。读取与 QC 方法见 `examples/fluxnet_smoke.py` 及 `docs/FLUXNET_PLAN.md`。该 30 天测试未率定，不能替代物理验证。

## 7. 类似项目的统一测试约定

1. 固定代码提交与未提交修改、输入哈希、随机种子、问题规模、时间步数和数值精度。
2. 先做小规模正确性测试，再增加核数、节点数或 GPU 批量；计算误差、物理守恒与运行速度分别验收。
3. CPU 记录进程数、线程数与绑核方式；GPU 记录物理设备、驱动、工具链和框架版本。共享 GPU 记录同机负载。
4. 分开记录编译 / JIT、预热、主计算、主机与设备传输、结果回读；GPU 计时前后执行框架同步。相同计时范围才能比较。
5. 每次保存原始日志、退出码、PBS 作业号或 HF Space 链接，以及环境清单。标明“实测”“官方规格”“未验证”，不以推算代替结果。

建议分工：EHPC 跑 CPU 对照、参数扫描和独立格点批处理；xuanze 跑持续的 GPU 数值与性能测试；ZeroGPU 先做可独立复现的短测试，再决定是否适配更完整的工作负载。
