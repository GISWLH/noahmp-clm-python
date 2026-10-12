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

当前本地验收结果：

- `python -m pytest -q`：**34 passed**；`python -m pytest -q --doctest-modules src tests`：**34 passed**；`python -m compileall -q src examples tests` 通过。
- 35 个已实现选项组合（降水 3、地表拖曳 2、气孔 2、胁迫 3、地面阻抗 4、雪反照率 2、雪导热 5、压实 2、冻土渗透率 2、径流表层/地下各 4）均通过有限值、入口不变性和能量诊断检查；雪方案冷暖转换的 13 个组合水量误差均小于 `5e-16 m`，最大能量残差小于 `1e-3 W/m²`。
- `python examples/run_single_column.py` 和 `python examples/parity_noahmp.py --days 5` 均可运行；默认 5 日示例最大能量残差为 `9.538e-4 W/m²`。

### NCAR/noahmp v5.0 对照

工作区用 NCAR/noahmp v5.0 `src/` 的 **135 个 Fortran 模块全部编译通过**（Windows
conda-forge Flang 19.1.7；SNICAR 输入模块使用 netCDF Fortran module），并编译运行了
`AtmosForcingMod` 单柱烟囱测试。Jordan (1991)、BATS 和冰点阈值在
`Tair−273.15 = −1, 0, 1, 2, 3 K` 上与官方 `FrozenPrecipFrac`/`RainfallRefHeight`
逐点一致（官方单精度输出的最大舍入差小于 `3e-7`）。这项测试覆盖 `AtmosForcingMod.F90`
的阶段 1 降水分配入口；其余日尺度 ET/感热/地温需要 HRLDAS 的站点强迫和参数表。

`examples/parity_noahmp.py` 接受官方单柱 CSV（列：`et_mm,h_wm2,tsoil1_k`）并报告日偏差：

```bash
python examples/parity_noahmp.py --reference official_daily.csv
```

没有官方 CSV 时脚本仍生成 Python 日诊断，但不会伪造官方偏差数值。
