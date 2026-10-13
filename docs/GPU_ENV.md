# GPU 环境侦察

记录时间：2026-10-12（Windows 11，PowerShell）

## 原始命令与结果

| 命令 | 结果 |
|---|---|
| `nvidia-smi` | 发现 1 张 NVIDIA GeForce RTX 4070，显存 12,282 MiB（约 12 GB）；驱动/KMD 610.88，CUDA UMD 13.3，WDDM；侦察时显存占用约 1,592 MiB。 |
| `nvcc --version` | 不可用：PowerShell 找不到 `nvcc`。 |
| `python --version` | Python 3.13.12（`E:\miniconda3\python.exe`）。 |
| `pip list`（基础环境） | `numpy 2.4.4`、`torch 2.6.0+cu124`、`torchaudio 2.6.0+cu124`、`torchvision 0.21.0+cu124`；未安装 JAX/CuPy。 |

## 安装方法

没有修改基础环境，也没有使用 `--break-system-packages`。创建了独立虚拟环境：

```powershell
python -m venv C:\Users\hhuwl\venvs\noahmp-gpu
C:\Users\hhuwl\venvs\noahmp-gpu\Scripts\python.exe -m pip install --upgrade pip
C:\Users\hhuwl\venvs\noahmp-gpu\Scripts\python.exe -m pip install "jax[cpu]" cupy-cuda12x
```

安装结果：JAX 0.11.2、jaxlib 0.11.2、CuPy 14.2.0、NumPy 2.5.4。JAX Windows wheel 只发现
`CpuDevice(id=0)`；CuPy 能枚举 RTX 4070（compute capability 8.9，CUDA runtime 12.09）。
CuPy 给出 CUDA_PATH 未设置的 warning；进一步的 device arithmetic 和 RawKernel 编译均因
缺少 CUDA Toolkit headers 失败，因此本机没有可报告的 GPU kernel smoke test。

作为独立的驱动/硬件检查，基础环境中的 `torch 2.6.0+cu124` 报告
`cuda_available=True`，并在 RTX 4070 上完成了 `float32` 张量运算（`arange * 2 + 1`）。
该结果只证明 CUDA 驱动和硬件可用，不代表 Noah-MP CUDA kernel 已执行。

阶段 D 期间尝试了 `pip install "cupy-cuda12x[ctk]"` 以补 CUDA headers；该 extra
会下载约 553 MB 的 NVIDIA toolkit wheels，当前网络下载未完成，虚拟环境仍保持
原先的 CuPy 14.2.0 安装。因此批量 CuPy benchmark 的 headers 缺失仍记录为
`unavailable`，没有把它写成 GPU 实测结果。

## 路线选择

本机有 NVIDIA GPU，因此按 A→H 推进：

- A：状态扁平化与 round-trip 测试。
- B：JAX 全链路；当前 Windows wheel 走 JAX-CPU，保留同一套 `jit/vmap/scan` 代码。
- C：JAX x64/x32 与 NumPy 基准 parity。
- D：NumPy、JAX-CPU 和可用的 CuPy/GPU 基准。
- E：CuPy 对照版，使用已验证的 RTX 4070 runtime。
- F：CUDA kernel 源码和 loader 照写；由于本机没有 `nvcc`，编译/运行标记为 **[未实测]**，不伪造结果。
- G/H：文档、全量测试和推送收尾。

以上为阶段 3a 的历史路线；阶段 3b 已在 Linux/xuanze 新建环境并实际执行，
下面记录对应的 `jax[cuda12]` 安装和验证，GPU 状态接口保持不变。

## xuanze 侦察

记录时间：2026-10-12 23:26 EDT（服务器显示 2026-10-13 11:26 CST）。
通过 `ssh -o BatchMode=yes -o ConnectTimeout=15 xuanze-neiwang` 公钥连接成功；
用户 `lhwang`，主机报告 `localhost.localdomain`，系统 CentOS Stream 9。

| 命令 | 实际结果 |
|---|---|
| `nvidia-smi` | 两张 NVIDIA GeForce RTX 4090，各 24564 MiB；Driver 570.153.02，驱动支持 CUDA 12.8；侦察时各有约 959 MiB 已被其他进程使用，利用率 0%。 |
| `nvcc --version` | CUDA compilation tools release 12.4，V12.4.99，build cuda_12.4.r12.4/compiler.33961263_0。 |
| `command -v nvcc` | `/usr/local/cuda-12.4/bin/nvcc`。 |
| `python3 --version` | Python 3.12.2。 |
| `nvidia-smi --query-gpu=index,name,compute_cap,memory.free,utilization.gpu --format=csv` | GPU 0/1 均为 compute capability 8.9，剩余 23133 MiB，利用率 0%。 |

访问、GPU、nvcc 门槛通过。Linux 将新建独立 venv；保留已有 GPU 进程。
四份必读文档已读完；补充参考 `docs/AGENT_PROMPT_GPU.md` 在本分支中不存在。

### H1 Linux 安装命令

仓库路径 `/home/lhwang/work/noahmp-phase3b/repo`；新建虚拟环境
`/home/lhwang/venvs/noahmp-phase3b`（Python 3.12.2），没有复用 Windows venv。
系统已提供 Toolkit 12.4，无需安装系统包。采用
[JAX 官方的 CUDA 12 pip 安装路径](https://docs.jax.dev/en/latest/installation.html)：

```bash
git clone --branch gpu/migration --single-branch \
  https://github.com/GISWLH/noahmp-clm-python.git ~/work/noahmp-phase3b/repo
python3 -m venv ~/venvs/noahmp-phase3b
source ~/venvs/noahmp-phase3b/bin/activate
python -m pip install --upgrade pip
python -m pip install 'jax[cuda12]' cupy-cuda12x
cd ~/work/noahmp-phase3b/repo
python -m pip install -e '.[dev]'
export CUDA_VISIBLE_DEVICES=1
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true
export CUDA_PATH=/usr/local/cuda-12.4
export PYTHONPATH=src
```

服务器 CPU 为双路 Intel Xeon Gold 5218R @ 2.10 GHz（40 物理核、80 逻辑 CPU）；
GPU 实测固定用物理设备 1（RTX 4090，sm_89）。现有任务仍保留。

### H1 验证输出

以下为 xuanze 上新 venv 的实际输出；`CUDA_VISIBLE_DEVICES=1` 后的逻辑设备 0
对应物理 GPU 1。安装和验证成功，退出码均为 0。

```text
Python: 3.12.2
jax 0.11.2
jaxlib 0.11.2
jax-cuda12-plugin 0.11.2
jax-cuda12-pjrt 0.11.2
cupy-cuda12x 14.2.0
numpy 2.5.4
pytest 9.1.1
noahmp-clm 0.1.0
jax.devices(): [CudaDevice(id=0)]
JAX x64 GPU arithmetic: [1. 3. 5. 7. 9.] float64
CuPy device count: 1
CuPy runtime: 12090
CuPy arithmetic: [1. 3. 5. 7. 9.]
```

JAX CUDA 12 extra 自带的 runtime 为 CUDA 12.9；系统 nvcc 保持 12.4.99。
验证脚本显式执行 `jax.config.update('jax_enable_x64', True)` 并断言 GPU backend。
原始环境日志保存在服务器 `~/work/noahmp-phase3b/logs/h1-env.log`。

### CUDA 库搜索路径修正

服务器登录环境预设 `LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64/:`。
不清除时，JAX 单独初始化 CUDA 会因 cuSPARSE 无法加载而退回 CPU；`ldd`
显示 pip cuSPARSE 12.5.10.65 错误解析到系统 12.4 的 `libnvJitLink.so.12`。
原 H1 探针先导入 CuPy，恰好掩盖了这个加载顺序问题。

后续所有 JAX GPU 验证均 `unset LD_LIBRARY_PATH`，设 `JAX_PLATFORMS=cuda`
禁止 CPU 回退，并在 parity 中设 `NOAHMP_REQUIRE_GPU=1` 断言输出设备。
这仅改变当前 shell 环境，不修改服务器全局配置。最终测试实际输出
`parity devices: {CudaDevice(id=0)}`。
