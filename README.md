# noahmp-clm-python

Noah-MP / CLM **单柱物理**的纯 Python 基线实现（普通版本）。

核心设计只有一条：**格点完全独立**。每一个 1D 陆面柱（土壤—积雪—冠层）的状态全部封装在
`ColumnState` 里，任意两个柱之间没有任何水平耦合。批量驱动只是把 N 个独立的柱放在一起跑，
这个"柱"维度就是将来 `jax.vmap` / CUDA kernel 要向量化的轴——现在用普通 Python 循环，
以后整段替换，不需要改物理代码。

## 这是"普通版本"，不是翻译版本

- 本仓库是**原创的 Python 重实现**（按公开发表的方程与算法流程重写），不是把
  Noah-MP / CTSM 的 Fortran 源码逐行翻译过来的，仓库里不含上游源码。
- 方程与算法流程参考：
  - Noah-MP v5.0（NCAR/noahmp，现代 Fortran 重构版；He et al., 2023, GMD）
  - CLM5 / CTSM（ESCOMP/CTSM，BSD-3-Clause；Lawrence et al., 2019）
- 上游许可归上游，本仓库新增代码采用 MIT 许可。

## 阶段 1 实现了什么

单个时间步（默认 30 分钟）内按 Noah-MP 风格的顺序求解：

1. 降水按气温分配为雨/雪（Jordan 1991 简化）；冠层截留（容量 ∝ LAI）
2. 积雪：最多 3 层的 bulk 雪盖，BATS/CLASS 反照率，五种雪导热率，压实、融化
3. 冠层：big-leaf，Jarvis 或 Ball–Berry 气孔阻抗（经验光合曲线），Penman–Monteith 蒸腾
4. 地表能量平衡：对地表温度 `Ts` 做割线迭代，使 `Rnet − H − LE − G = 0`
5. 土壤水热：4 层（0.1 / 0.3 / 0.6 / 1.0 m，与 Noah-MP 默认一致）
   - 热传导：Crank–Nicolson 隐式求解（Thomas 算法），步长无条件稳定
   - 水分：入渗（fill-and-spill）+ Campbell/Clapp-Hornberger 型排水 + 根系吸水
6. 批量驱动 `run_columns`：N 个柱各自独立推进，可直接对照单柱结果
7. `PhysicsOptions` 对齐 Noah-MP v5.0 `ConfigVarType.F90` 的 `Opt*` 配置；阶段 1
   已实现降水相态 1–3、MOST/Chen97、Ball–Berry/Jarvis、Noah/CLM/SSiB
   胁迫、四种地面阻抗、BATS/CLASS 雪反照率、冻土相变、TOPMODEL/BATS/自由排水径流。

阶段 1 仍不包含 SNICAR、湿球降水相态（OptRainSnowPartition=5）、完整 Farquhar
光合、作物/灌溉/湿地/冰川和横向地下水；这些范围在 ROADMAP.md 中列为后续工作。

## 快速开始

```bash
pip install -e ".[dev]"
python examples/run_single_column.py     # 5 天合成强迫，打印逐日 ET/径流/地温
pytest                                   # 能量闭合、水平衡、批量独立性测试
```

## 阶段 3：GPU 迁移

GPU 代码全部位于 `src/noahmp_clm/gpu/`，普通 numpy 版仍是只读正确性基准。
状态先由 `gpu.pack` 编码为固定长度 `fvec/ivec`，JAX 后端提供
`step_column_jax` 和 `run_columns_jax`（时间 `lax.scan`、柱轴 `vmap`）；
`gpu/cuda/` 保存 SoA CUDA kernel 源码和延迟 loader。

```bash
python3 -m venv ~/venvs/noahmp-phase3b
source ~/venvs/noahmp-phase3b/bin/activate
pip install 'jax[cuda12]' cupy-cuda12x
pip install -e '.[dev]'
unset LD_LIBRARY_PATH  # xuanze 登录环境的系统 CUDA 库会干扰 pip CUDA 库
export CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src
python -c "import jax; print(jax.devices())"
NOAHMP_REQUIRE_GPU=1 python -m pytest tests/test_gpu_parity.py -q
python benchmarks/bench_gpu.py --sizes 100 1000 10000 100000 300000 --steps 48
```

2026-10-13 xuanze RTX 4090 实测，`N=100000 × 48`（三次预热、三次测量取中位数）：
JAX x64 **22.0033 s**，JAX x32 **3.14413 s**。五档 GPU 计时已完成，大规模
NumPy 单核对照正在补测，尚不报告完整加速倍数。`.[gpu]` 本身仍只安装 CPU JAX，
Linux CUDA 安装与库路径说明见 [GPU_ENV.md](docs/GPU_ENV.md)。

JAX float64 混合选项门通过；手写 CUDA 虽已编译执行，但严格 float32 parity
仍失败，其 **0.00728780 s** 计时标为“实测但精度未过门”。不能据此宣布阶段 3
完成。范围、原始计时和失败证据见 [GPU_SUMMARY.md](docs/GPU_SUMMARY.md)、
[BENCHMARK.md](docs/BENCHMARK.md)、[GPU_BLOCKERS.md](docs/GPU_BLOCKERS.md)。

最小调用示例：

```python
import numpy as np
from noahmp_clm import ColumnState, default_soil_params
from noahmp_clm.driver import run_columns
from noahmp_clm.io import synthetic_forcing

state = ColumnState(soil=default_soil_params("loam"))
forcings = synthetic_forcing(n_days=3, dt=1800.0)
diags = run_columns([state], [forcings])[0]
print("3 天总 ET (mm):", sum(d["et_mm"] for d in diags))
```

## 目录结构

```
src/noahmp_clm/
    column.py        # ColumnState（单柱全部状态）与 Forcing（单步强迫）
    driver.py        # step_column / run_column / run_columns（批量轴）
    io.py            # 合成强迫、CSV 输出
    physics/
        canopy.py    # 气孔阻抗、Penman-Monteith
        energy.py    # 地表能量平衡迭代、湍流通量
        soil.py      # 土壤水热求解器
        snow.py      # 多层积雪、反照率、压实与导热
    options.py      # Noah-MP v5.0 Opt* 配置值对象
    gpu/
        pack.py      # 定长状态 ABI
        jax_core.py  # JAX jit/vmap/scan 物理链路
        cupy_core.py # CuPy 正确性锚点
        cuda/        # 默认物理 SoA CUDA kernel；已实测，严格 parity 未通过
examples/run_single_column.py
tests/test_column.py
ROADMAP.md           # 分阶段移植计划：Noah-MP 对标 → CLM 柱子集 → JAX/CUDA
```

## 验证状态

`pytest` 覆盖四条硬约束（每次提交都应通过），并覆盖阶段 1 各物理选项：

- **能量闭合**：每步 `|Rnet − H − LE − G| < 0.5 W/m²`
- **水平衡**：`降水 = ET + 径流 + Δ蓄水`，误差 `< 1e-9 m`
- **格点独立**：批量跑 N 个柱的结果 ≡ 逐个单柱跑的结果（bit 级一致）
- **雪累积/消融**：冷期 SWE 增长，暖期融化减少

当前 Linux GPU 验收结果：

- 全量 `python -m pytest -q -s`：**44 passed, 2 failed**；两项失败分别为 CUDA N=1024 的 1 步与 48 步严格 parity，保留原容差，未隐藏失败。
- JAX-CUDA 的 H2 混合门和相关回归 **7 passed**；H6 已实现物理 35 个选项取值通过，18 个占位字段的 55 个值仅验证无效应，另有 2 个单值字段及 11 个未实现值明确记录。
- 一日 JAX x32/x64 最大累计偏差：ET **9.64719e-6 mm**、感热 **2.15644e-5 MJ/m²**。详细范围见 [GPU.md](docs/GPU.md)。

阶段 3a 的 Windows 历史验收：

- `python -m pytest -q`：**36 passed, 4 skipped**（基础环境没有 JAX/CuPy，Windows CUDA kernel 标记未实测）；`python -m compileall -q src examples tests benchmarks` 通过。
- 35 个已实现选项组合（降水 3、地表拖曳 2、气孔 2、胁迫 3、地面阻抗 4、雪反照率 2、雪导热 5、压实 2、冻土渗透率 2、径流表层/地下各 4）均通过有限值、入口不变性和能量诊断检查；雪方案冷暖转换的 13 个组合水量误差均小于 `5e-16 m`，最大能量残差小于 `1e-3 W/m²`。
- `python examples/run_single_column.py` 和 `python examples/parity_noahmp.py --days 5` 均可运行；默认 5 日示例最大能量残差为 `9.538e-4 W/m²`。

### NCAR/noahmp v5.0 对照

工作区用 NCAR/noahmp v5.0 `src/` 的 **135 个 Fortran 模块全部编译通过**（Windows
conda-forge Flang 19.1.7；SNICAR 输入模块使用 netCDF Fortran module），并编译运行了
`AtmosForcingMod` 单柱烟囱测试。以与官方相同的 IEEE single-precision 温度输入，Jordan
(1991)、BATS 和冰点阈值在 `Tair−273.15 = −1, 0, 1, 2, 3 K` 的 15 个输出上与官方
`FrozenPrecipFrac`/`RainfallRefHeight` 一致，最大冻结比例差为 `1.22e-6`。恰好冰点的
双精度/单精度舍入差会改变阈值分支，但不是方程差异。这项测试覆盖 `AtmosForcingMod.F90`
的阶段 1 降水分配入口；其余日尺度 ET/感热/地温需要 HRLDAS 的站点强迫和参数表。

`examples/parity_noahmp.py` 接受官方单柱 CSV（列：`et_mm,h_wm2,tsoil1_k`）并报告日偏差：

```bash
python examples/parity_noahmp.py --reference official_daily.csv
```

没有官方 CSV 时脚本仍生成 Python 日诊断，但不会伪造官方偏差数值。
