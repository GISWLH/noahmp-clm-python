# 给接手本仓库的 AI Agent 的工作指令（阶段 1：Noah-MP v5.0 对标）

> 这是一段可直接粘贴给另一个 AI agent 的 prompt。那个 agent 持有 Wang 的 GitHub 凭证。

---

你是 Wang 的 AI 工程师。你的任务：在 GitHub 仓库 **GISWLH/noahmp-clm-python**
上实现「阶段 1：Noah-MP v5.0 对标」。你持有 Wang 的 GitHub 访问凭证，
可直接 clone / push 该仓库，不要索要新的凭证。

## 1. 仓库现状（v0.1，已验证通过）

纯 Python + numpy 的 Noah-MP/CLM 单柱物理基线。`src/noahmp_clm/` 结构：

- `column.py` — `ColumnState`（单柱全部状态）、`Forcing`（单步强迫）、`SoilParams`
- `driver.py` — `step_column(state, forcing)` 纯函数单步推进；
  `run_columns` 批量循环（N 个柱各自独立）
- `physics/canopy.py` — 简化 Jarvis 气孔阻抗 + Penman-Monteith 蒸腾
- `physics/energy.py` — 湍流通量、饱和水汽压、对 Ts 的割线迭代求解器
- `physics/soil.py` — 土壤热传导（Crank–Nicolson + Thomas）、土壤水
  （fill-and-spill 入渗 + Campbell 排水 + 根系吸水）
- `physics/snow.py` — 单层 bulk 积雪、反照率衰减、融化
- `io.py` — 合成强迫、CSV 输出；`tests/test_column.py` — 4 个硬约束测试

开工前先通读 `README.md`、`ROADMAP.md` 和 `driver.py`。

## 2. 铁律（违反任何一条即返工）

1. **格点完全独立**：`ColumnState` 之外不允许任何跨柱共享状态；不许引入柱间通信。
2. `step_column` 保持纯函数语义（状态进、状态出），数组在入口复制；
   将来这段要整体换成 `jax.vmap` / CUDA kernel，不要写 vmap 不友好的代码
   （无原地修改、无 Python 端分支依赖全局量、无 I/O）。
3. `tests/test_column.py` 的 4 个测试必须永远全绿：
   能量闭合残差 < 0.5 W/m²、水平衡误差 < 1e-9 m、批量与单柱 bit 级一致、雪累积/消融。
4. **Clean-room**：可以阅读 NCAR/noahmp 的 Fortran 源码来理解方程，
   但不许逐行翻译、不许复制代码段；变量名与结构自己组织，docstring 注明方程出处。
5. 中文 docstring + 英文术语；每个物理方案注明论文/技术文档出处。

## 3. GitHub 工作流

- `git clone https://github.com/GISWLH/noahmp-clm-python`，main 分支
- 每个模块一个分支 `phase1/<模块名>`（如 `phase1/snow-albedo`），小步提交后 push 分支
- 合并到 main 前本地跑全量测试；commit message 用英文简短句式，
  如 `phase1: add CLASS snow albedo (OptSnowAlbedo=2)`

## 4. 阶段 1 实现清单（按顺序一次做一个模块）

### 4.0 选项框架（先做）

新建 `src/noahmp_clm/options.py`，`PhysicsOptions` dataclass，字段与
Noah-MP v5.0 `ConfigVarType.F90` 的 `Opt*` 一一对应（选项含义以该文件注释为准，
我已核实，抄录如下）：

- `OptRainSnowPartition`：1=Jordan(1991)，2=BATS（Tair<冰点+2.2 全雪），
  3=冰点阈值，5=湿球温度（Wang et al. 2019）
- `OptSoilWaterTranspiration`：1=Noah（土壤水），2=CLM（基质势），3=SSiB（基质势）
- `OptGroundResistanceEvap`：1=Sakaguchi & Zeng 2009，2=Sellers 1992，
  3=调低湿土阻抗的 Sellers，4=无雪用 1、有雪用表值
- `OptSurfaceDrag`：1=Monin-Obukhov 相似理论，2=原 Noah（Chen et al. 1997）
- `OptStomataResistance`：1=Ball-Berry，2=Jarvis
- `OptSnowAlbedo`：1=BATS，2=CLASS，3=SNICAR（阶段 1 不做 3）
- `OptSnowSoilTempTime`：1=半隐式通量边界，2=全隐式温度边界，3=同 1 但雪盖参与表温
- `OptSnowThermConduct`：1=Stieglitz(Yen 1965)，2=Anderson 1976，3=常数，
  4=Verseghy 1991，5=Douville(Yen 1981)
- `OptSoilTemperatureBottom`：1=底部零通量，2=8m 深部温度边界
- `OptSoilSupercoolWater`：1=Niu & Yang 2006（无迭代），2=Koren 1999（迭代）
- `OptRunoffSurface` / `OptRunoffSubsurface`：1=TOPMODEL+地下水，2=TOPMODEL+平衡水位，
  3=Noah 原始自由排水，4=BATS，5=MMF 地下水，6=VIC，7=新安江，8=Dynamic VIC
- `OptSoilPermeabilityFrozen`：1=线性（更透水），2=非线性（更不透水）
- `OptSnowCompaction`：1=Anderson 1976，2=Abolafia-Rosenzweig 2024
- `OptSnowCoverGround`：1=Niu & Yang 2007，2=AR 2025
- `OptDynamicVeg` 固定为 4（关闭动态植被，用输入 LAI）；灌溉/作物/冰川/湿地/
  瓦片排水全部关闭，不在阶段 1 范围。

`ColumnState` 增加 `options: PhysicsOptions` 字段；`step_column(state, forcing)`
签名不变，从 `state.options` 读取选项做分发。默认选项值以 NCAR/noahmp 配套的
HRLDAS 示例 namelist 为准（去上游仓库找），在 `options.py` 顶部注释写明出处。

### 4.1 逐模块实现（按此顺序，依赖由弱到强）

每个选项实现为纯函数，放在 `physics/` 下对应模块，分发字典如
`SNOW_ALBEDO = {1: albedo_bats, 2: albedo_class}`。

1. **降水分配** `physics/precip.py`：把 driver 内联的 Jordan91 抽出为函数，
   补 Opt=2（BATS）、Opt=3（冰点阈值）。参考上游 `SnowfallBelowCanopyMod.F90`
   （冠层截雪本阶段跳过）。
2. **地表拖曳** `physics/surface_drag.py`：v0.1 是中性 bulk。实现 Opt=1 的
   MOST 稳定度迭代（参考 `ResistanceAboveCanopyMostMod.F90` /
   `PsychrometricVariableMod.F90`：迭代 Obukhov 长度更新 CH），Opt=2 保留现有
   bulk。CH 进入 H/LE/G 全部分量，收敛判据写清楚。
3. **气孔阻抗** `physics/canopy.py` 扩展：已有简化 Jarvis（=Opt 2）。
   补 Opt=1 Ball-Berry（参考 `ResistanceCanopyStomataBallBerryMod.F90`），
   需要光合速率 A——本阶段用经验光合-光响应曲线代替完整 Farquhar，
   在 docstring 明确标注简化点（完整 Farquhar 是阶段 2 的工作）。
4. **土壤水胁迫** `physics/stress.py`：实现 Opt=1/2/3 三种 beta
   （Noah 分段线性 / CLM 基质势 / SSiB 基质势），同时作用于气孔阻抗与土壤蒸发。
   参考 `SoilWaterTranspirationMod.F90`。
5. **地面蒸发阻抗** `physics/ground_evap.py`：Opt=1 Sakaguchi & Zeng 2009
   干表层阻抗，替换 v0.1 的 beta 法。参考 `ResistanceGroundEvaporationMod.F90`。
6. **雪反照率** `physics/snow.py` 扩展：现有指数衰减≈BATS 简化（=Opt 1）。
   补 Opt=2 CLASS（参考 `SnowAlbedoClassMod.F90`：按雪龄/温度的分段衰减+新雪重置）。
7. **多层雪** `physics/snow.py`：v0.1 单层 bulk。实现最多 3 层——
   `SnowLayerDivideMod`（按 SWE 阈值分层）、`SnowLayerCombineMod`（合并）、
   每层独立雪温与热传导（参考 `SnowThermalPropertyMod.F90` +
   `SoilSnowThermalDiffusionMod.F90`）；层间能量用隐式耦合，稳定性自己用测试验证。
8. **雪压实** Opt=1 Anderson 1976（参考 `SnowpackCompactionMod.F90`：
   破坏性沉降 + 超载压实，随温度）。
9. **冻土** `physics/frozen_soil.py`：Opt=1 Niu & Yang 2006
   （参考 `SoilWaterSupercoolNiu06Mod.F90`：由温度反推未冻水含量）；
   相变潜热用 apparent heat capacity 法进入土壤热传导
   （参考 `SoilSnowWaterPhaseChangeMod.F90`）；
   `OptSoilPermeabilityFrozen=2` 非线性（导水率随冰含量指数衰减）。
10. **径流** `physics/runoff.py`：v0.1 自由排水≈Opt 3。实现 Opt=1/2 TOPMODEL
    （参考 `RunoffSurfaceTopModelEquiMod.F90`：地形指数→饱和缺水→产流；
    需给 `ColumnState` 新增字段：地形指数、地下水位深）与 Opt=4 BATS。
    VIC / 新安江列为可选。
11. **雪热传导选项**：`OptSnowThermConduct=1` Stieglitz(Yen 1965) vs 现有常数，
    小改动，直接做进雪模块。

### 4.2 验证要求

- 4 个硬约束测试永远全绿；新选项不得破坏能量/水平衡（冻土相变、融雪水
  同样计入平衡检查）。
- 每个新选项至少 1 个单元测试，测物理合理性而非数值巧合，例如：
  冻土 −5°C 时未冻水 < 总水量；雪温到 0°C 后能量转为融化而非升温；
  Ball-Berry 与 Jarvis 在湿润 conditions 下量级一致。
- parity 测试：用 NCAR/noahmp 官方单柱驱动在相同强迫下跑参考，
  对比日 ET / 感热 / 表层地温，把对比脚本放 `examples/parity_noahmp.py`，
  偏差目标：与 Noah-MP 自身参数不确定性同量级；在 README 记录偏差结果。
- 每个选项完成后跑全量测试，不许跳过。

### 4.3 节奏

按 4.1 顺序一次只做一个模块：实现 → 测试 → commit → push。
不要并行开多个模块，不要一次提交整个阶段。

## 5. 代码规范

- numpy 为主；不引入 Fortran/C 扩展，不调用上游二进制。
- 函数纯函数化；`ColumnState.copy()` 已提供深复制。
- 类型注解 + 中文 docstring；每个方程注明出处（论文或技术文档章节号）。
- 不复制上游源码：变量命名与代码结构自己组织。

## 6. 验收标准（全部满足才算阶段 1 完成）

- [ ] `options.py` 完整，默认值有出处注释
- [ ] 4.1 中第 1–9 项全部实现且有测试；第 10 项至少完成 TOPMODEL-equilibrium，
      第 11 项完成
- [ ] 4 个硬约束测试全绿；`examples/parity_noahmp.py` 可运行且偏差已记录
- [ ] README 更新阶段 1 状态；ROADMAP 勾选完成项
- [ ] 所有代码已 push 到 `GISWLH/noahmp-clm-python` 的 main 分支

## 7. 参考资料（只读、理解方程用，不复制代码）

- `NCAR/noahmp` 仓库 `src/*.F90`（v5.0 模块化源码；上面点名的 Mod 文件即入口）
- He et al., 2023, GMD-16-5131（v5.0 重构说明）
- Noah-MP v5.0 Technical Note, doi:10.5065/ew8g-yr95（方程手册，按章节查）
- Niu et al., 2011（Noah-MP 原始论文）；Niu & Yang, 2006, JHM（冻土）

现在开始：先读仓库 README/ROADMAP/driver.py，然后做 4.0 选项框架。
