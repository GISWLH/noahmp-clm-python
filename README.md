# noahmp-clm-python

Noah-MP / CLM **单柱物理**的纯 Python 基线实现（普通版本）。

**下一版本：v1.0.0 单柱物理开发中，尚未发布。** 当前工作覆盖生物地球物理、
碳氮循环和主要扩展，进度见 [v1.0 执行记录](docs/V1_PROGRESS.md)。
Wang 已确认新的[科学验收口径](docs/V1_ACCEPTANCE.md)：关键变量偏差 ≤10%、
瞬时通量 r >0.85、近零绝对容差、收支守恒独立硬要求。旧 H4/K3/M4 门已退役，
下述 v0.2.0 数字是保留的历史记录，不是本阶段合并条件。

开发分支已完成[P1物理修复](docs/V1_P1_RESULTS.md)，并建立三后端
[单叶光合](docs/CLM_PHOTOSYNTHESIS.md)和[多层辐射](docs/CLM_CANOPY_RADIATION.md)
组件及[三后端冠层能量联立](docs/CLM_CANOPY_ENERGY.md)。新的无雪柱入口已接入
NumPy和[原生JAX/CUDA无雪整柱步](docs/CLM_GPU_COLUMN.md)，并继续扩展到下述
完整五层雪/冠层焓柱。稳定度连续回归与观测评价待完成，默认驱动仍是v0.2路径。
单叶实测仍有未达标曲线。
新增[NumPy五层雪组件](docs/CLM_SNOW_DESIGN.md)：质量/焓、压实、粒径老化、
五波段SNICAR近似与气候态沉降映射。新增[雪—土壤联合焓及冬季冠层边界](docs/CLM_COLUMN_ENTHALPY.md)，
联合热/质量转移365天累计焓残差最大1.878e-6 J/m²；
已建立[NumPy冠层—雪—土壤质量/焓时间步](docs/CLM_EXTENSIVE_COLUMN.md)，
连接水汽、融水和排水携热。新增[联合热量原生JAX/CUDA](docs/CLM_GPU_ENTHALPY.md)：
252个热工况同精度全字段最大差0，7天连续热柱float32最大温差0.0001205 K。
新增[原生JAX/CUDA雪辐射及冠层短波耦合](docs/CLM_GPU_SNOW_RADIATION.md)：
368个雪况最大逐层光谱吸收偏差0.6048%，349个耦合工况最大雪吸收偏差0.1853%；
该检查点Linux全量460 passed / 0 skipped。新增[雪质量/焓/杂质原生输送](docs/CLM_GPU_SNOW_MASS.md)：
2015个单步工况和4032次连续柱操作同精度最大差0，并修复全融化残余杂质丢失；
该检查点Linux全量466 passed / 0 skipped。
[雪表层水汽/霜露、压实、老化和沉降](docs/CLM_GPU_SNOW_PROCESSES.md)现已同步原生JAX/CUDA；
全表/随机工况粒径最大偏差6.616%，12096次连续组件操作同精度最大差0。
该检查点Linux全量475 passed / 0 skipped。
新增[冬季固定表温冠层与储水携热原生边界](docs/CLM_GPU_CANOPY_BOUNDARY.md)：
425工况同精度全字段最大差0，H/LE最大偏差0.1039%/0.2307%。
该检查点Linux全量481 passed / 0 skipped。
新增[土壤入渗/取水/排水质量与携热原生输送](docs/CLM_GPU_SOIL_TRANSPORT.md)：
1062同输入工况和12096次连续柱操作同精度最大差0，日底部出流偏差最大0.00733%；
Linux全量487 passed / 0 skipped，float32累计舍入残差详见报告。
新增[原生冠层—雪—土壤非线性表面步](docs/CLM_GPU_CANOPY_SURFACE.md)：
50工况同精度全字段最大差0，H/LE最大绝对差0.01409/0.01139 W/m²，
完整质量/焓收支独立核算，最终Linux全量492 passed / 0 skipped；七天4032柱步完成，同精度全轨迹最大差0。
新增[降水至排水的完整质量/焓柱原生入口](docs/CLM_GPU_EXTENSIVE_COLUMN.md)：
16类工况目标4 passed / 0 skipped，同精度全字段最大差4.76837e-7，
最大土温差0.0000403881 K、LE差0.0183929 W/m²；最终Linux全量496 passed /
0 skipped；七天4032柱步完成，同精度全轨迹最大差0，日H最大偏差0.00845985%。
新增[分层冠层截留/冰水相态与水汽携焓组件](docs/CLM_CANOPY_INTERCEPTION.md)：
1728个独立方程工况、365天NumPy守恒及428个设备工况、四后端七天过程链完成，
CUDA32/JAX32全字段最大差0，日冰出流偏差最大0.0000271%；最终全量510 passed / 0 skipped。
随后分层冠层冰水已接入原生完整柱；MOST反馈及外部科学验收仍待完成。
正在开发[NumPy分层冠层焓—叶温—空气湿度联立](docs/CLM_CANOPY_PHASE_ENERGY.md)：
75个独立方程工况及三类冠层七天1008步完成；该NumPy检查点全量528 passed / 0 skipped，累计焓残差最大2.519 J/m²。
新增[原生冠层相变联立](docs/CLM_GPU_CANOPY_PHASE_ENERGY.md)：22项目标测试、75个设备对照工况及七天4032次联立求解完成，同精度全字段最大差0，LE最大偏差0.00542%；该检查点全量532 passed / 0 skipped。
[分层冠层完整柱](docs/CLM_LAYERED_CANOPY_COLUMN.md)NumPy 44项目标通过，随后冻结b全量554 passed / 0 skipped；新增原生JAX/CUDA完整柱4项目标通过，16工况同精度全输出最大差4.76837e-7，最大土温差1.91494e-5 K。最终冻结全量563 passed / 0 skipped，四后端七天4032柱步已完成，日H最大偏差0.00876828%，累计float32水/焓残差0.00166426 kg/m² / 834.633502 J/m²，数值界见报告。
新增[MOST实际通量反馈](docs/CLM_CANOPY_MOST.md)：60个独立积分对照最大偏差0.534%，100个系数工况CUDA32/JAX32全输出最大差0；完整设备耦合5项目标通过，同精度最大差2.98023e-8。连续检查发现稳定反馈收敛慢，改用有界求根后真实失败旧态10次收敛、11项NumPy回归通过；修后三柱一天144步完成，累计焓残差最大0.000237588 J/m²；修后设备6项目标通过，独立float32水/焓残差最大1.67303e-5 kg/m² / 26.69451 J/m²。
新增[Wang湿球相态](docs/CLM_WETBULB.md)：334个独立方程工况、1000个原生三后端工况完成，最大float32温差2.67421e-5 K；选项5已接入NumPy及设备完整柱，整柱及选项矩阵69项目标通过，CUDA32/JAX32完整柱最大差0。论文方程与官方迭代的差异已明确记录，官方科学对标仍待完成。

新增[VIC可变入渗与ARNO基流](docs/CLM_VIC.md)：756个独立入渗对照最大差5.68434e-14 kg/m²；NumPy完整柱及原生转移32项目标通过，组件同精度全部输出最大差0；8个原生完整柱工况同精度最大差4.76837e-7。后续13项回归12项通过、旧湿球合同测试1项失败，修正合同后已在新目标复验；官方站点验收待完成。
新增[GDD季节落叶物候](docs/CLM_PHENOLOGY.md)：两年解析LAI最大差8.88178e-16，512个设备组件工况同精度差0；选项5已接入三后端实际LAI/截留/水热，28项目标通过，8类柱连续两步同精度全输出差0。长期站点及植物C/N连接待做。
新增[CENTURY风格凋落物与三库土壤C/N分解](docs/CLM_DECOMPOSITION.md)：NumPy独立日步碳库/呼吸偏差0.22448%/0.52504%；原生组件13项目标通过，512工况及四后端七天同精度差0。植物分配已完成下述组件；整柱NEE已接通，真实观测验证待做。该分解检查点全量662 passed / 0 skipped，不包含后续植物/整柱代码。
新增[植物C/N分配、呼吸与氮限制](docs/CLM_PLANT_CN.md)：14项目标通过，512状态及四后端七天植物—土壤联算同精度差0；float32累计C/N残差0.07559 gC/m² / 0.001246 gN/m²，独立数值界见报告。[NumPy整柱BGC连接](docs/CLM_BGC_COLUMN.md)及相关36项回归已完成：叶碳决定实际LAI，叶N反馈真实冠层GPP，整柱C/N残差最大2.34379e-12 gC/m² / 8.84393e-14 gN/m²；原生整柱8类柱连续两步同精度全输出差0，float32独立水/热/C/N最大残差3.13232e-5 kg/m² / 5.86774 J/m² / 0.000178888 gC/m² / 4.70704e-5 gN/m²；数值界和失败记录见报告。BGC冻结E全量700 passed / 0 skipped（4010.29秒，不含后续作物/灌溉）；七天4032柱步已完成、全轨迹同精度差0，日GPP/NEE后端偏差0.000303%/0.000908%，观测验收待做。

新增[简化作物](docs/CLM_CROP.md)：积温/出苗/收获与种子、产品、凋落C/N账已接通三后端整柱；8类柱两步全输出同精度差0。冠层湿度边界修复后目标33 passed；90天NumPy及七天四后端轨迹运行中，产量/站点验收待做。
新增[土壤水触发灌溉](docs/CLM_IRRIGATION.md)：三后端组件及NumPy整柱25项相关回归完成，输入水量/焓独立路由；原生四种整柱ABI已接通，设备完整目标复核中。
新增[简化湿地甲烷三后端组件](docs/CLM_WETLAND.md)：10项目标通过，512工况同精度最大差1.05345e-38；两条设备路径8柱×365日累计碳残差最大1.68447e-6 gC/m²。NumPy整柱及相关32项回归通过；原生整柱24项通过，四种组合两步全部输出同精度差0；真实反馈连续轨迹和观测验收待完成。
新增[简化瓦片排水组件](docs/CLM_TILE_DRAINAGE.md)：NumPy 5项通过、108组独立高精度水量对照最大差2.78e-17 kg/m²；真实土壤扣水携热已实现，原生组件10项通过、4柱×480步全部输出同精度差0；NumPy完整柱及相关回归27项、原生完整柱14项通过，完整状态连续两步同精度差0。
新增[有限冰川柱](docs/CLM_GLACIER.md)：5层雪与4层纯冰共享界面，NumPy独立方程最大温差2.84e-14 K；NumPy气象驱动整柱26项目标与1460步年度守恒完成，累计焓残差3.40e-6 J/m²。原生完整气象时间步F目标25项通过，含中性/MOST、耗尽及真实旧态求根回归。四后端365天1460步完成，同精度全轨迹差0；日ET/H/LE最大差0.0000503 mm / 0.001435 / 0.001649 W/m²，float32最大表温差0.01240 K。官方/观测验收未完成。

**状态：可用，带[已知限制](docs/KNOWN_LIMITATIONS.md)。** NumPy、JAX x64/x32 和
CUDA 后端均可运行；`v0.2.0` 为 GPU 移植收尾版，已获 Wang 验收。
四站真实数据同量化输入 canw 通过率99.2014%，仍未达到原99.5%门；
Wang明确接受这项已量化差异，见[验收记录](docs/GPU_ACCEPTANCE.md)。
原H4/K3红灯完整保留，见[收尾数字](docs/GPU_CLOSEOUT_RESULTS.md)和
[真实数据结果](docs/REAL_DATA_CHECKS.md)。冻土敏感应用建议用 float64 / JAX-x64。

计算资源、SSH 入口及 CPU/GPU 测试方法见 [environment.md](environment.md)，包含 EHPC、xuanze 和 Hugging Face ZeroGPU 的核验记录。

核心设计只有一条：**格点完全独立**。每一个 1D 陆面柱（土壤—积雪—冠层）的状态全部封装在
`ColumnState` 里，任意两个柱之间没有任何水平耦合。批量驱动只是把 N 个独立的柱放在一起跑，
这个“柱”维度分别由普通 Python 循环、`jax.vmap` 和 CUDA kernel 并行执行。

## 这是"普通版本"，不是翻译版本

- 本仓库是**原创的 Python 重实现**（按公开发表的方程与算法流程重写），不是把
  Noah-MP / CTSM 的 Fortran 源码逐行翻译过来的，仓库里不含上游源码。
- 方程与算法流程参考：
  - Noah-MP v5.0（NCAR/noahmp，现代 Fortran 重构版；He et al., 2023, GMD）
  - CLM5 / CTSM（ESCOMP/CTSM，BSD-3-Clause；Lawrence et al., 2019）
- 上游许可归上游，本仓库新增代码采用 MIT 许可；Leafweb实验数据采用其
  单独的CC-BY-4.0许可和[来源说明](docs/CLM_PHOTOSYNTHESIS.md)。

## 阶段 1 实现了什么

单个时间步（默认 30 分钟）内按 Noah-MP 风格的顺序求解：

1. 降水按气温分配为雨/雪（Jordan 1991 简化）；冠层截留（容量 ∝ LAI）
2. 积雪：最多 3 层的 bulk 雪盖，BATS/CLASS 反照率，五种雪导热率，压实、融化
3. 冠层：big-leaf，Jarvis 或 Ball–Berry 气孔阻抗（经验光合曲线），Penman–Monteith 蒸腾
4. 地表能量平衡：对地表温度 `Ts` 做割线迭代，使 `Rnet − H − LE − G = 0`
5. 土壤水热：4 层（0.1 / 0.3 / 0.6 / 1.0 m，与 Noah-MP 默认一致）
   - 热传导：冻结系数的后向 Euler 隐式求解（Thomas 算法）；整柱边界耦合不保证任意步长下稳定
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

GPU 代码全部位于 `src/noahmp_clm/gpu/`，普通 NumPy64 版是独立数值对照基准。
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
export CUDA_PATH=/usr/local/cuda-12.4
export PATH="$CUDA_PATH/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src
python -c "import jax; print(jax.devices())"
NOAHMP_REQUIRE_GPU=1 python -m pytest tests/test_gpu_parity.py -q
python benchmarks/bench_gpu.py --sizes 100 1000 10000 100000 300000 --steps 48
```

历史性能表属于 **K2修复前** 的冻结源（JAX `0e296691…`、CUDA `7640b062…`），
不能标为当前物理源码性能。`N=100000 × 48` 三预热/三正式的中位秒数如下：

| NumPy单核 | JAX x64 | JAX x32 | CUDA x32 |
|---:|---:|---:|---:|
| 3616.904161 | 21.355447 | 3.807727 | 0.196976 |

GPU计时包含同步计算、完整输出归约、逐值校验和host回读；CPU仅逐柱消费标量
checksum。N300000单核基线已收齐，中位10272.657723秒；CPU启动时未记录加载源码hash，
来源限制保留。本轮最终 `b954d90` 物理源码的N100000复测单列于
[BENCHMARK.md](docs/BENCHMARK.md)，不拿旧CPU基线拼出新源码加速比。
`.[gpu]` 只安装CPU JAX，Linux CUDA安装见[GPU_ENV.md](docs/GPU_ENV.md)。

本轮 `b954d90` 最终物理源码、同样三预热/三正式、同样完整输出校验口径：
N100000×48的中位数为 **JAX x64 22.042122 s / JAX x32 3.227870 s /
CUDA 0.203611 s**。所有计时样本输出门通过；原H4与M4失败仍单独报告。

JAX64混合门、CUDA/JAX32同精度门通过，原NumPy64 H4仍失败；M2合成门各字段
通过率≥99.5%，residual中位 .000293732、p99 .00181961 W/m²。K2四后端365天
可靠性已通过；K3原域门仍有5565/64800柱状态超差（8.58796%），全部归因于
已记录的冻土/厚雪下温度轨迹误差。融尽后仍有222784.419 J/m²热量路由缺口，
表面闭合不等于整柱焓守恒。范围、原始计时和失败证据见
[GPU_SUMMARY.md](docs/GPU_SUMMARY.md)、
[BENCHMARK.md](docs/BENCHMARK.md)、[GPU_BLOCKERS.md](docs/GPU_BLOCKERS.md)。

各环境运行方式：

```bash
# EHPC：共享home中，先串行/多进程，再跨两节点MPI
qsub benchmarks/ehpc/serial_pool.pbs
qsub benchmarks/ehpc/mpi_two_nodes.pbs

# xuanze：使用上述CUDA环境；输出目录必须是新的
python benchmarks/check_closeout.py --directory outputs/m2-recheck
python benchmarks/bench_gpu.py --sizes 100000 --steps 48 --backends jax64 jax32 cuda
```

EHPC脚本与环境变量说明见[EHPC README](benchmarks/ehpc/README.md)；
HF ZeroGPU演示和客户端调用见[Space说明](examples/hf_zerogpu/README.md)。
各环境的实测/未测边界、硬件、软件、耗时与checksum见[ENV_TESTS](docs/ENV_TESTS.md)。

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
        cuda/        # 35值分发 SoA CUDA kernel；同精度门通过，原NumPy64 H4仍失败
examples/run_single_column.py
tests/test_column.py
ROADMAP.md           # 分阶段移植计划：Noah-MP 对标 → CLM 柱子集 → JAX/CUDA
```

## 验证状态

`pytest` 覆盖四条硬约束（每次提交都应通过），并覆盖阶段 1 各物理选项：

- **能量闭合**：每步 `|Rnet − H − LE − G| < 0.5 W/m²`
- **水平衡**：`降水 = ET + 径流 + Δ蓄水`，误差 `< 1e-9 m`
- **格点独立**：普通版测试范围内，批量跑 N 个柱与逐个单柱跑的结果逐位一致；GPU各shape另设数值回归
- **雪累积/消融**：冷期 SWE 增长，暖期融化减少

当前 Linux GPU 验收结果：

- K2生产源全量 `python -m pytest -q -s`：**147 passed, 2 failed, 0 skipped**；两项失败分别为 CUDA N=1024 的 1 步与 48 步原NumPy64 H4，保留原容差，未隐藏失败。本轮最终复验日志见[GPU_SUMMARY](docs/GPU_SUMMARY.md)。
- JAX-CUDA 的 H2 混合门和相关回归 **7 passed**；H6 已实现物理 35 个选项取值通过，18 个占位字段的 55 个值仅验证无效应，另有 2 个单值字段及 11 个未实现值明确记录。
- 当前共同源码8场景一日 JAX x32/x64 最大累计偏差：ET **9.64719e-6 mm**、感热 **2.15644e-5 MJ/m²**；相对差均小于1e-4，不能外推年度或原H4。详细范围见 [GPU.md](docs/GPU.md)。

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
