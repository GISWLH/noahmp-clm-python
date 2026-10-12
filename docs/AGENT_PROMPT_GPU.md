# 给接手本仓库的 AI Agent 的工作指令（阶段 3：GPU 迁移，整夜运行版）

> 这是一段可直接粘贴给另一个 AI agent 的 prompt。那个 agent 持有 Wang 的 GitHub 凭证。
> 本指令按**整夜无人值守**设计：顺序执行 A→H，每阶段有明确交付物；
> 阻塞时记录并绕行，不许卡死；每 1–2 小时 push 一次。

你是 Wang 的 AI 工程师。你的任务：把 **GISWLH/noahmp-clm-python**
的单柱物理迁移到 GPU。普通版本（`src/noahmp_clm/`，numpy）是正确性基准，
**只读不改**；GPU 代码全部放在新子包 `src/noahmp_clm/gpu/`。

## 0. 环境侦察（开工第一件事，约 15 分钟）

运行并记录结果到 `docs/GPU_ENV.md`：
- `nvidia-smi`（有无 GPU、型号、显存、驱动/CUDA 版本）、`nvcc --version`
- `python3 --version`，`pip list 2>/dev/null | grep -i -E "jax|cupy|torch|numpy"`
- 能否 `pip install`（上次 pip 受 PEP 668 限制，必要时用 `--break-system-packages`
  或 venv；记录你用的方法）

按侦察结果定路线（写进 GPU_ENV.md）：
- **有 NVIDIA GPU**：全路线 A→H（JAX + CuPy 对照 + 手写 CUDA kernel）
- **无 GPU**：做 A→D（JAX-CPU 完整路线）+ E 跳过 + F（kernel 写完但标记 `[未实测]`）
  + G、H
- **JAX 装不上**：回退路线——先做 A，然后用 numpy 做 SoA 向量化版
  （`gpu/numpy_soa.py`，同样 vmap 语义），D 的基准改为 numpy 循环 vs SoA；
  F 的 kernel 照写。把回退原因写进 GPU_ENV.md。

## 1. 铁律

1. **不动普通版本**：`src/noahmp_clm/` 下除 `pyproject.toml`（加 gpu extras）
   和文档外，一律只读。GPU 版是普通版的**逐函数对应移植**，不是重写物理；
   每个 GPU 函数 docstring 首行标注 `对应普通版: <模块>.<函数名>`。
2. **格点独立**：GPU 版同样零柱间通信；`vmap` 的映射轴就是柱轴。
3. 现有全部测试保持全绿；新增 `tests/test_gpu_*.py`，GPU 目录的测试独立命名。
4. Clean-room 约束继续有效。
5. 中文 docstring + 注释；关键设计决策写进 `docs/GPU.md`（为什么这样做，
   不是只贴代码）。
6. **不许伪造结果**：没跑过的测试不许写"通过"；没测的 kernel 标记 `[未实测]`。

## 2. 阶段 A — 状态扁平化规约（约 1 小时）

JAX/CUDA 要求定长、无 Python 对象的状态。设计状态向量并写成文档
`docs/STATE_SPEC.md`（表格：字段 | 偏移 | 长度 | 说明）：

- 浮点段（约 70）：`tsoil[4]`、`smc[4]`、`soil_ice[4]`、`root_frac[4]`、
  `snow_swe_layers[3]`、`snow_t_layers[3]`、`snow_density_layers[3]`、
  标量 `ts, swe, snow_t, snow_albedo, snow_age_days, canw, lai, fveg, z0m,
  albedo, emiss, rsmin, terrain_index, groundwater_depth`（共 14 个），
  `soil` 7 参数（smcmax, smcwlt, smcref, satdk, bexp, quartz + 1 保留位）。
  `dz` 固定 `[0.1, 0.3, 0.6, 1.0]` 为编译期常量，**不进向量**。
- 整型段（约 20）：`PhysicsOptions` 的全部 `Opt*` 逐字段映射为 int
  （`OptDynamicVeg` 等已关闭的也保留占位，保证与上游字段一一对应）。
- 强迫向量（8）：`dt, tair, qair, wind, ps, swdown, lwdown, prcp`。
- 实现 `gpu/pack.py`：`column_state_to_vec(state) -> (fvec, ivec)`、
  `vec_to_column_state(fvec, ivec, dz_const) -> ColumnState`、
  `forcing_to_vec(f)`；写 `tests/test_gpu_pack.py` 做 round-trip 测试
  （随机 50 个柱，打包再解包，逐字段 bit 一致）。
- 交付：`docs/STATE_SPEC.md` + `pack.py` + 测试全绿 → commit + push。

## 3. 阶段 B — JAX 移植（整夜主体，约 3–4 小时）

新建 `src/noahmp_clm/gpu/jax_core.py`（太大可拆 `jax_physics/` 子目录，
与普通版 `physics/` 模块一一对应）。逐条是硬要求：

- **B1**：`jnp` 替换 `np`；Python 内置 `min/max` → `jnp.minimum/maximum`；
  `np.clip` → `jnp.clip`；不许留任何 Python 标量分支依赖追踪期非常量。
- **B2 选项分发**：options 是 int 向量，用 `jax.lax.switch(opt_idx, branches, ...)`
  分发；每个方案写成独立函数。**先实现默认选项组合**（`options.py` 的默认值）
  的全链路跑通，再扩展其余选项。
- **B3 割线迭代**：普通版 `secant_solve` 是变长迭代 → 改为**固定 25 次**
  `jax.lax.fori_loop`（数学等价，收敛性由阶段 C 的 parity 测试验证）。
  Ts 钳制 `[200, 330]` 用 `jnp.clip`。
- **B4 MOST 迭代**（`surface_drag`）：同样改为固定次数 `fori_loop`
  （普通版几次收敛就取几次，parity 验证）。
- **B5 Thomas 求解**：n=4 固定，**手工展开** 4x4 追赶法（避免 scan 开销，
  每行写清楚对应普通版 `_thomas` 的哪一步）。
- **B6 多层雪**：固定 3 层 + active mask；分层/合并逻辑用 `jnp.where(mask)`
  实现，不许用 Python `if len>...`。
- **B7 冻土**（Niu & Yang 2006 反推）：逐层标量运算，直接写向量表达式。
- **B8 顶层签名**：
  `step_column_jax(fvec, ivec, forcing_vec) -> (fvec_new, ivec_new, diag_vec)`，
  `diag_vec` 定长（ts, rnet, h, le, g, et_mm, prcp_mm, runoff_mm, melt_mm,
  swe, canw, residual 共 12 个）。
- **B9**：时间循环用 `jax.lax.scan`，柱批量用 `jax.vmap`，整体 `jax.jit`。
  对外提供 `run_columns_jax(fvecs, ivecs, forcings)`.
- **精度**：先 `jax.config.update("jax_enable_x64", True)` 跑通 parity；
  再做 float32 版（同一套代码，jit 时切换 dtype），记录两者偏差。
- `pyproject.toml` 加 `[project.optional-dependencies] gpu = ["jax[cpu]"]`
  （有 CUDA 的机器 agent 自行换 `jax[cuda12]`），**不许破坏基础安装**。
- 交付：`jax_core.py` + 默认选项全链路可运行 → commit + push。

## 4. 阶段 C — 一致性验证（硬门槛，约 1–2 小时）

- `tests/test_gpu_parity.py`：构造 N=8 个**混合配置**的柱
  （sand/loam/clay × 高/低 LAI × 默认选项与 2–3 套非默认选项组合，
  强迫覆盖：暴雨、冷期积雪、暖期消融、干旱），跑 48 步：
  JAX float64 vs numpy 普通版，逐时间步对比 `diag_vec` 与终态向量，
  容差：相对误差 < 1e-6（雪/冻土相关量放宽到 1e-4，理由写注释）。
- 能量/水平衡：在 JAX 版上重算 residual（用 `diag_vec` 的 rnet/h/le/g），
  同样 < 0.5 W/m²；水平衡用向量版重算 < 1e-9 m。
- 选项矩阵抽查：阶段 1 已实现的每个 `Opt*` 至少 2 个取值跑通 24 步
  （全组合太多，抽样即可，但抽样方案写进测试 docstring）。
- float32 vs float64：记录 ET/感热的日累积偏差量级，写进 `docs/GPU.md`。
- **全部通过才算阶段 C 完成**；通不过就修 B，不许调松容差
  （容差变更必须在 commit message 说明理由）。

## 5. 阶段 D — 性能基准（约 1 小时）

- 写 `benchmarks/bench_gpu.py`：N = 1e2, 1e3, 1e4, 1e5, 3e5 柱 × 48 步，
  测：numpy 单核（普通版 `run_columns`）、JAX-CPU、JAX-GPU（若有），
  指标：每柱每步耗时（µs）、总吞吐（柱·步/秒）。
- 每个配置预热 3 次取中位数；记录机器信息（CPU 型号、GPU 型号）。
- 输出 `docs/BENCHMARK.md` 表格 + 加速比。**目标**：GPU 上相对单核 numpy
  ≥100×（N=1e5 量级）。达不到就做一轮 profiling，把瓶颈一句话写进文档
  （kernel launch 开销 / H2D 搬运 / 某模块），**不许**为调优熬掉整夜。
- 交付：bench 脚本 + BENCHMARK.md → commit + push。

## 6. 阶段 E — CuPy 对照版（仅有 CUDA 且 D 已完成，约 1 小时）

- `src/noahmp_clm/gpu/cupy_core.py`：numpy 代码的机械替换（`cp` 换 `np`），
  保持与普通版相同的循环结构（不追求性能）。
- `tests/test_gpu_cupy.py`：与 numpy 普通版 parity（容差 1e-6）。
- 目的：给手写 kernel 一个"中间正确性锚点"。时间不够可跳过，
  跳过时在 GPU.md 写一句话说明。

## 7. 阶段 F — 手写 CUDA kernel（仅有 CUDA 且 nvcc 可用，约 2 小时）

- **设计**：**1 thread = 1 column**（格点完全独立的最自然映射），
  grid-stride loop 处理 N 柱；block 大小 256。
- **内存**：SoA 布局——状态向量按字段连续存放（`field[c]` 而非 `column[f]`），
  保证合并访问；`docs/STATE_SPEC.md` 追加 GPU 内存布局一节。
- **kernel 内**：Thomas 4x4 手工展开；割线迭代固定 25 次；雪层 mask；
  全部单精度；`__restrict__` 修饰指针。
- 交付物：`gpu/cuda/column_kernel.cu` + `gpu/cuda/build.sh`（nvcc 编译命令）
  + `gpu/cuda/loader.py`（cupy RawKernel 或 ctypes 加载 + 启动配置）。
- `tests/test_gpu_cuda.py`：N=1024 柱，与 JAX float32 / numpy 对比，
  容差 1e-4 相对。
- **若无 GPU/nvcc**：kernel 照写完，保证语法完整，文件头标 `[未实测]`，
  测试文件只留加载器 import 测试；**不许伪造测试结果**。

## 8. 阶段 G — 文档与收尾（约 30 分钟）

- `docs/GPU.md`：架构（为什么 1 thread = 1 column）、状态向量规约、
  精度说明（float64/float32/CUDA 单精度三档偏差）、性能表、未完成项与原因。
- 更新 `README.md`（加 GPU 章节：安装、快速开始、性能数字）、
  `ROADMAP.md`（勾选阶段 3 完成项）。
- `docs/GPU_SUMMARY.md`：中文总结——做了什么、性能数字、哪些没做/为什么、
  下一步建议（多 GPU、CLM 柱子集的 GPU 版等）。
- 最终：全量 `pytest`（含普通版全部测试）全绿 → push。

## 9. 整夜工作纪律

1. 按 A→H 顺序推进，一阶段交付+push 后再进下一阶段。
2. **每 1–2 小时 push 一次**（即使阶段未完）：
   `git commit -am "gpu: <进度>" && git push origin gpu/migration`，
   让 Wang 早上能看到进度曲线。
3. 单点阻塞超过 30 分钟搞不定：记入 `docs/GPU_BLOCKERS.md`（现象、试过的方法），
   绕过去继续，**不许整夜卡死**。
4. 不许为赶进度删测试、放宽容差；`main` 分支永远可用——GPU 代码只在
   `gpu/` 子包和 `gpu/migration` 分支，不影响普通版。
5. 天亮前无论推进到哪：push 所有进度，在 `docs/GPU_SUMMARY.md` 顶部用三句话写
   当前状态（做到哪一步、性能数字、阻塞项）。

## 10. 参考资料

- 本仓库 `docs/AGENT_PROMPT.md`（铁律与阶段 1 上下文）
- JAX 文档：`jit` / `vmap` / `lax.switch` / `lax.scan` / `fori_loop`
- `github.com/ayalahlou/clm-ml-jax`（JAX 冠层物理，风格参考，只读）
- CUDA C++ Programming Guide（内存合并访问一节）

现在开始：第 0 步环境侦察，记录到 `docs/GPU_ENV.md`，然后进阶段 A。
