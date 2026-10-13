# 给接手本仓库的 AI Agent 的工作指令（阶段 3b：xuanze/Linux GPU 实测）

> 这是一段可直接粘贴给另一个 AI agent 的 prompt。那个 agent 持有 Wang 的 GitHub 凭证。
> 背景：阶段 3a 已在 Windows + RTX 4070 机器上完成（分支 `gpu/migration`），
> 但那台机器没有 nvcc、JAX 只有 CPU 版——parity 门、CUDA kernel 实测、
> N≥1e5 基准三件硬事必须搬到有完整 CUDA 工具链的 Linux 机器上做。
> Wang 有一台名为 xuanze 的 Linux 服务器（应有 NVIDIA GPU）。
> 本指令按整夜无人值守设计。

## 0. 开工前先读（30 分钟，不许跳过）

在 `gpu/migration` 分支上读完这四个文档再动手：
- `docs/GPU_SUMMARY.md`——阶段 3a 的交付清单与三句话状态
- `docs/GPU_ENV.md`——Windows 机器的环境与路线选择
- `docs/BENCHMARK.md`——当前唯一的实测数字（N=100：numpy 490µs/柱步，
  JAX-CPU 13.63µs/柱步，36×，**不是 GPU 数字**）
- `docs/STATE_SPEC.md`——状态向量 ABI（70 float64 + 33 int32 + 8 强迫），
  后面所有工作都基于它，**不许改 ABI**（改了则 pack/jax/kernel 全崩）

## 1. 第一步：确认 xuanze 访问（硬门槛）

1. 用你已知的方式连接 xuanze Linux 服务器（SSH 等）。连上后先跑
   `nvidia-smi`、`nvcc --version`、`python3 --version`，把结果追加到
   `docs/GPU_ENV.md` 新的一节 "xuanze 侦察"。
2. **如果 30 分钟内连不上**：写 `docs/XUANZE_ACCESS.md`（试过的方法、
   报错原文、你还需要 Wang 提供什么信息），push 到 `gpu/migration`，
   然后**停下来**，不要编造任何结果。Wang 看到文档后会给你开门。
3. 连上且有 GPU + nvcc：继续。下面阶段编号为 H1→H7。

工作分支：继续用 `gpu/migration`（`git clone https://github.com/GISWLH/noahmp-clm-python`，
`git checkout gpu/migration`，`git pull`）。Linux 上重建虚拟环境，
不要复用 Windows 的 venv。

## 2. H1 — Linux 环境搭建（约 30 分钟）

- 安装 CUDA Toolkit（若没有 `nvcc`；需要 sudo 时记下命令写进文档，
  不要反复试错——先 `which nvcc || ls /usr/local/cuda*/bin/nvcc`）。
- venv + `pip install "jax[cuda12]"`（与本机 CUDA 大版本对应，
  `nvidia-smi` 的 CUDA Version 说了算），`jax_enable_x64=True` 验证：
  ```python
  import jax; jax.config.update("jax_enable_x64", True)
  print(jax.devices())  # 必须看到 cuda 而不是只有 cpu
  ```
- `pip install cupy-cuda12x`（与 CUDA 大版本对应），验证
  `cupy.cuda.runtime.getDeviceCount() > 0`。
- `pip install -e ".[dev]"` 装本仓库。
- 把"装了什么、版本号、验证输出"记进 `docs/GPU_ENV.md` → commit + push。

## 3. H2 — JAX parity 门复测（约 1–2 小时，硬门槛）

Windows 上 `tests/test_gpu_parity.py` 在混合选项下触发 XLA HLO verifier
进程级中止——先在 Linux 上复现：

1. `PYTHONPATH=src python -m pytest tests/test_gpu_parity.py -x -q`。
2. **如果同样中止**：这是 JAX 代码的真 bug（大概率是某处形状依赖了
   Python 期非常量，或 `lax.switch` 分支返回了不一致的形状/dtype）。
   用二分法定位：先单柱默认选项跑通，再逐个加选项/加柱数，
   用 `jax.debug.print` 或分段 `jit` 缩小范围。修好之前不许动 H3。
   修好后在 `docs/GPU.md` 记一笔根因。
3. **如果直接通过**：在 `docs/GPU_SUMMARY.md` 记录"Windows HLO 中止为
   Windows wheel 缺陷，Linux 不复现"，继续。
4. 门槛：8 柱 × 48 步混合选项（暴雨/积雪/消融/干旱全覆盖），
   JAX float64 vs numpy 普通版，diag 相对误差 < 1e-6
   （雪/冻土量放宽 1e-4，理由已在测试注释里）。
   全绿 → commit + push。

## 4. H3 — CUDA kernel 补全（约 2–3 小时，本阶段主体）

现状：`src/noahmp_clm/gpu/cuda/column_kernel.cu` 的 `energy_residual`
只有 `rnet - h - g` 三项——**LE 全家都没搬进去**（土壤蒸发、冠层截留蒸发、
Penman-Monteith 蒸腾、雪分支、冻土），只是一个能编译的骨架。
你的工作是把它补成与 `jax_core.py` 默认选项组合**物理等价**的完整 kernel：

1. 按 `docs/STATE_SPEC.md` 的 SoA 布局，把缺的物理逐项搬成 `__device__` 函数，
   每个函数头注释写 `对应普通版: <模块>.<函数名>`：
   - 气孔阻抗（默认 `OptStomataResistance=1` Ball-Berry；Jarvis 保留接口，
     先实现默认项，`lax.switch` 对应为 `switch(opt)` 或函数指针表）
   - Penman-Monteith 蒸腾 + 根系水量限制
   - 地面蒸发（默认 `OptGroundResistanceEvap=4` 对应的无雪分支）
   - 雪分支：3 层 mask、CLASS/BATS 反照率（默认 BATS）、融化
   - 冻土未冻水反推（默认 Niu & Yang 2006）
   - 割线迭代固定 25 次、Thomas 4x4 已有（保留）
2. 约束：**1 thread = 1 column** 不变；grid-stride loop 不变；
   全部单精度 `float`；`__restrict__` 指针；shared memory 不用
   （每柱状态小，register 够）。
3. `gpu/cuda/build.sh` 跑通 `nvcc -O3 -arch=sm_XX`（按本机算力填，
   `nvidia-smi --query-gpu=compute_cap` 查），**零 warning**编译
   （`-Wall`，warning 当 error 修）。
4. 交付：编译通过的完整 kernel → commit + push。

## 5. H4 — CUDA parity（约 1 小时，硬门槈）

- `loader.py` 跑起来：N=1024 柱，float32，与 numpy 普通版对比，
  容差 **1e-4 相对误差**（单精度）；同时与 JAX float32 版对比，
  两者偏差应 < 1e-5（同为单精度实现，差太多说明 kernel 有 bug）。
- 把 `[未实测]` 标记从已验证的文件/测试中去掉（没验证的不许去）。
- 全绿 → commit + push。

## 6. H5 — 完整 GPU 基准（约 1 小时）

- 跑 `benchmarks/bench_gpu.py --sizes 100 1000 10000 100000 300000 --steps 48`，
  后端：numpy 单核、JAX-GPU（x64 与 x32 各一版）、手写 CUDA kernel。
- 每个配置预热 3 次取中位数；记录 CPU/GPU 型号。
- 真实数字填进 `docs/BENCHMARK.md`（替换"尚未实测"的声明）。
  **目标**：N=1e5 时 GPU（JAX 或 kernel 取优者）相对单核 numpy ≥100×。
  达不到：`ncu`（若有）或 JAX profiling 做一轮，把瓶颈一句话写进文档，
  不许调优熬整夜。
- 交付：BENCHMARK.md 真实表格 → commit + push。

## 7. H6 — 选项矩阵与精度（约 1 小时）

- 在 GPU（JAX-CUDA）上重跑阶段 3a 的选项抽查
  （每个 `Opt*` ≥2 个取值 × 24 步），确认 Windows 上被 HLO 中止掩盖的
  选项组合在 Linux 全绿；新增 `tests/test_gpu_options_linux.py`
  （若 Windows 版测试已覆盖则合并，不重复）。
- float32 vs float64 日累积 ET/感热偏差：写进 `docs/GPU.md` 精度一节，
  给出一句话结论（"单精度日 ET 偏差 < X%，可/不可接受"）。

## 8. H7 — 收尾与合并（约 30 分钟）

1. `docs/GPU_SUMMARY.md`：顶部三句话更新为**真实**状态
   （parity 结果、加速比数字、还剩什么）；删掉"必须在 xuanze 复测"的
   旧表述，换成实测结论。
2. `ROADMAP.md`：阶段 3 总体项勾选完成（仅当 H2+H4+H5 全绿）。
3. `README.md`：GPU 章节把"计划中"换成实测数字 + 运行命令。
4. 全量 `pytest`（普通版 + GPU 全部测试）全绿。
5. 合并：`git checkout main && git merge gpu/migration`（fast-forward 或
   `--no-ff` 均可，写清楚 merge message），`git push origin main`。
   **只有 H2+H4+H5 全绿才合**；任一红灯就留在分支，SUMMARY 写清缺口。

## 9. 整夜纪律（与 3a 相同，继续执行）

1. H1→H7 顺序推进，每阶段交付+push。
2. 每 1–2 小时 push 一次（`gpu/migration` 分支）。
3. 单点阻塞 30 分钟：记 `docs/GPU_BLOCKERS.md`（现象、试过的方法），绕行。
4. 不许删测试、不许放宽容差、不许伪造数字；`main` 在合并前不受影响。
5. 天亮前：push 所有进度，`docs/GPU_SUMMARY.md` 顶部三句话写当前真实状态。

## 10. 参考

- 本仓库 `docs/AGENT_PROMPT.md`（铁律）、`docs/AGENT_PROMPT_GPU.md`
  （阶段 3a 全文，含 JAX 移植细节约束）
- `docs/STATE_SPEC.md`（ABI，不许改）
- JAX 文档（`jax.devices` 后端确认）、CUDA C++ Programming Guide

现在开始：第 1 步，确认 xuanze 访问并记录到 `docs/GPU_ENV.md`。
