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

若后续切换到带 CUDA-enabled JAX 的 Linux/xuanze 服务器，只需重新安装对应
`jax[cuda12]` wheel，GPU 代码接口不变。
