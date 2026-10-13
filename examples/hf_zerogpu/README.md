---
title: Noah-MP short GPU smoke
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.25.2
python_version: 3.12
app_file: app.py
pinned: false
license: mit
---

# Noah-MP ZeroGPU compatibility smoke

短时交互演示：1024 个合成柱 × 48 步、JAX GPU 对 NumPy64。
输入和判据沿用预冻结 M2 的 H4 1024 柱集。检查逐步完整状态和 H/LE/G/ET，
并验证逐步回放与单次48步批运行逐位相同，输出版本、源码 hash、输入/输出
checksum、有限性、误差与包含编译/读回的耗时。默认 float32，另可运行 float64。
不是正式性能基准、真实观测验证或校准。

部署时将本目录的三个文件复制到独立 Gradio Space 根目录，再复制仓库
`src/` 和 `benchmarks/` 中的 `check_closeout.py`、`closeout_criteria.json`、
`probe_gpu_numerics.py`，使用已有 ZeroGPU 权限。
不要启用付费硬件，不要上传凭据。依赖和环境设置见 requirements.txt/app.py。
JAX 在 `@spaces.GPU` 内初始化并强制 CUDA；若该平台的动态设备机制不兼容，
报告会保留真实失败，不会退回 CPU 冒充 GPU 通过。

```python
from huggingface_hub import get_token
from gradio_client import Client
client = Client("LonghaoWang/noahmp-clm-gpu-smoke", token=get_token())
print(client.predict("float32", api_name="/run_smoke"))
```

CPU参考对156个唯一输入列运行并映射到1024柱，GPU实际运行全部1024柱。
CPU参考不占 GPU 分配时段。GPU 函数单次最多申请 120 秒（含冷编译与输出回读）；
使用已有免费/PRO 配额，调用前确认剩余配额充足。实际部署、权限与运行结论
见本仓库 `docs/ENV_TESTS.md`。完整收尾门以冻结的 M2 报告为准。
