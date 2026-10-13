# FLUXNET：M4 有界检验与 Phase 4 计划

本轮 M4 使用 NAS `processed/site_csv/` 的 CN-Cng、CN-HaM、CN-Qia、CN-Cha
四站各 30 天，不校准、不调参。完整输入来源、QC、逐步轨迹、JAX64 parity、
M2 分档和 LE/H 观测指标集中在 [REAL_DATA_CHECKS.md](REAL_DATA_CHECKS.md)，
可复现入口为 [examples/fluxnet_multisite.py](../examples/fluxnet_multisite.py)。
CN-HaM 的 LE 无 QC=0 样本，明确报告 n=0；其余三站可计算 LE/H 指标。
ERA5 路径可达但缺下行辐射/总降水，且实际为日均文件，模型对照记“未测”。

Wang 已预先定义 [M2 收尾验收标准](GPU_CLOSEOUT_CRITERIA.md)，它是本轮合并判断的
口径；以下 K6 数据及原严格门结论保留为历史，不能把旧源码的单站结果冒称为 M4 复测。
M4 实际结果为：JAX64 parity 与同精度门通过，但 x32/CUDA canw 合格率
99.201389% 低于 99.5%（CN-Cha 单站 97.291667%），所以 **M4 数值门仍未通过**。
Wang已明确接受这项差异并验收本次收尾，授权合并/发布，见[验收记录](GPU_ACCEPTANCE.md)。
正式同量化输入参考下全部 46 项已归因，仍不改变硬门或删除站点。原始 double 输入参考的 51 项端到端差异也原样保留；观测指标继续使用原始 double 基线。
完整物理验证仍属于下方 Phase 4 独立工作。

## K6 历史单站记录

本任务只建立真实强迫读取、30 天连续运行、后端 parity 和观测指标报告。
**未率定、未完成物理验证；不替代 K1–K3，不改变原 H4 容差或合并条件。**
普通版新增 loader 放在 [examples/fluxnet_smoke.py](../examples/fluxnet_smoke.py)，
没有为 K6 修改 `src/noahmp_clm/` 或 STATE_SPEC。

## 真实输入及可追溯性

选择中国长岭草地站 CN-Cng（44.5934°N, 123.5092°E；IGBP GRA），
固定窗口为 **2007-07-01 00:00 至 2007-07-31 00:00（右端不含）**，
1440 个连续半小时。窗口在运行模型前选定，目的为暖季读取/运行 smoke；
没有根据模拟指标挑选日期，不能由它推断冬季或所有站点表现。
站点来源：[FLUXNET CN-Cng](https://fluxnet.org/sites/siteinfo/CN-Cng)，
数据引用：[Dong, FLUXNET2015 CN-Cng Changling](https://doi.org/10.18140/FLX/1440209)。

`F:\Data_center` 是用户的数据目录/门户仓库，真实 ZIP 位于 NAS WLH：
`/volume2/Data/Global/FLUXNET-Data-Shop/raw/added_2026-06/ameriflux_downloads/FLX_CN-Cng_FLUXNET2015_FULLSET_2007-2010_1-4.zip`。
本次从该 NAS 既有 8013 门户只读取得 ZIP（45,384,649 bytes），解出
FULLSET_HH CSV；从该文件按时间范围原字节选取 header + 1440 行，未修改数据值。
三份文件的完整 SHA256、原始路径、来源 URL 见
[k6-fluxnet-source.json](gpu-results/k6-fluxnet-source.json)。
原始观测与模拟逐步 CSV 留在忽略目录 `outputs/k6/` 及远端私有运行目录，不提交 git。

## 映射、缺失值与 QC

读取列名而非固定列号。根据 `TIMESTAMP_START/END` 求 `dt`，仅接受连续且
等间隔的 1800/3600 秒 HH/HR 数据；缺行、重复、缺 forcing、非法湿度或负降水
均报错，不自动填补。`-9999` 作为缺失值；时间保持当地标准时间，不引入夏令时。
时间/缺失定义见 [FLUXNET2015 dataset](https://fluxnet.org/data/fluxnet2015-dataset/)。

| CSV 列 | Forcing | 转换 |
|---|---|---|
| TA_F，°C | tair，K | +273.15 |
| VPD_F，hPa；或显式 `--humidity rh` 的 RH，% | qair，kg/kg | `es=611.2 exp(17.67 Tc/(Tc+243.5))`；VPD 路径 `e=es−100 VPD`，RH 路径 `e=es RH/100`；`q=0.622 e/(p−0.378 e)` |
| WS_F，m/s | wind，m/s | 原值；本 smoke 未做塔高至模型参考高度校正 |
| PA_F，kPa | ps，Pa | ×1000 |
| SW_IN_F / LW_IN_F，W/m² | swdown / lwdown | 原值 |
| P_F，每条记录的 mm | prcp，m/s | `/1000/dt`，不是 mm/s 或日降水 |

单位与 consolidated `_F` 定义来自
[官方 FULLSET 变量表](https://fluxnet.org/data/fluxnet2015-dataset/fullset-data-product/)。
VPD 和 RH 为显式替代路线，默认 VPD，缺值时不静默切换。
湿度计算使用普通版相同的水面 Tetens 约定；不把不同饱和水汽压公式造成的不一致
隐藏为裁剪。真实月份所有必需 forcing 有值：TA/VPD/WS/LW/P 的 QC 均为 0；
PA 有 1 条 QC=2（ERA），SW 有 1 条 QC=1（良好填补），其余均 QC=0。
这些已有产品 gapfill 作为 forcing 使用并完整计数，不称全观测强迫。

LE/H 使用未做能量闭合校正的 `LE_F_MDS/H_F_MDS`，仅有限且各自 **QC=0** 的
记录进入观测指标。LE 为 1360/1440、H 为 1414/1440；不要求二者同时有效，
不把 gapfill 当作实测。bias 定义为模型减观测；相关系数无方差时记 null。
QC 与通量产品含义见 [Variables quick start](https://fluxnet.org/data/fluxnet2015-dataset/variables-quick-start-guide/)。
档案文件版本为 `1-4`，该月份 LE/H QC 无缺失；历史 QC 缺失问题见
[Known issues](https://fluxnet.org/data/fluxnet2015-dataset/known-issues/)，不可把别的月份缺 QC 自动当 0。

## 运行范围与判定

所有后端使用默认 `ColumnState`：loam、LAI=3、fveg=0.8、z0m=0.05 m，
初始土温/地表温度 285 K；未 spinup，未读取本站土壤/植被参数，未校准。
这是一份数据和代码管线 smoke，其观测误差包含参数、初值、塔高、尺度和结构误差。
观测 LE/H 不输入模型，不参与日期、参数或后端选择。

NumPy64 正常读取 double 强迫；JAX64 使用相同输入。JAX32/CUDA32 parity 的
NumPy64 参考从**同一 float32 量化后的状态与强迫**解包，避免把输入舍入混入运算
parity；观测指标还保留原始 NumPy64 运行。所有 12 个逐步诊断均报告最大绝对差
与超差数，包括 residual。x32 保持原 `rtol=1e-4, atol=1e-6`；x64 此 smoke
使用 `rtol=1e-6, atol=1e-6`。最终有效状态按 H2 的有效雪层 mask 比较，整数
完全一致；未用雪槽填充差异另列。GPU 必须断言实际设备，不能 CPU fallback。

## 实测结果：输入与执行就绪，x32 parity 仍红

2026-10-13 在 xuanze 物理 GPU1（RTX 4090）完成同一个 30 天连续轨迹，
NumPy64、JAX64、JAX32、CUDA32 输出均有限。独立源码快照为
`/home/lhwang/work/noahmp-phase3b/k6-fluxnet-20261013`，未重置日间状态。
首次全流程运行在约 8 分钟内结束；此处没有进行性能基准，不能从它给出加速比。
完整输入 QC、所有逐字段差异、设备、Python/NumPy 版本及源码 SHA256 保存在
[k6-fluxnet-smoke.json](gpu-results/k6-fluxnet-smoke.json)。

| GPU 后端 | 对相同输入的 NumPy64 parity | 最终有效状态/整数 | 逐步诊断说明 |
|---|---|---|---|
| JAX64 | 通过 `1e-6 + 1e-6` | 通过/完全相同 | 最大诊断绝对差 1.6115e-11（LE）；所有 12 字段通过 |
| JAX32 | **失败**，原 x32 门不变 | 通过/完全相同 | 见下表 |
| CUDA32 | **失败**，原 x32 门不变 | 通过/完全相同 | 见下表 |

| x32 对 NumPy64（JAX32/CUDA32 各自结果） | 超差步数 / 1440 | 最大绝对差 |
|---|---:|---:|
| Rnet，W/m² | 52 | 0.00506167415 |
| H，W/m² | 402 | 0.00789728585 |
| LE，W/m² | 36 | 0.01322944523 |
| G，W/m² | 218 | 0.00766327876 |
| ET，mm/步 | 1 | 0.00001139703 |
| residual，W/m² | 1433 | 0.00278519689 |

JAX32 和 CUDA32 的该表计数/最大值相同，不代表逐值完全相同。其余诊断通过本次
原门；有效最终状态通过不能覆盖已失败的逐步通量。这再次说明“能运行真实数据”
和“GPU parity 通过”是独立结论，**K6 没有使 K1–K3 任何红灯转绿**。

仅对 QC=0 的观测计算如下，bias/RMSE 单位 W/m²，bias=模型−观测：

| 后端 | 观测 | n | r | bias | RMSE |
|---|---|---:|---:|---:|---:|
| NumPy64 | LE | 1360 | 0.86934154 | 0.452131 | 47.681827 |
| JAX64 | LE | 1360 | 0.86934154 | 0.452131 | 47.681827 |
| JAX32 | LE | 1360 | 0.86934206 | 0.452154 | 47.681732 |
| CUDA32 | LE | 1360 | 0.86934206 | 0.452154 | 47.681732 |
| NumPy64 | H | 1414 | 0.89737293 | 6.490629 | 35.385035 |
| JAX64 | H | 1414 | 0.89737293 | 6.490629 | 35.385035 |
| JAX32 | H | 1414 | 0.89737190 | 6.491630 | 35.385671 |
| CUDA32 | H | 1414 | 0.89737190 | 6.491630 | 35.385671 |

**这些数值是未校准、未 spinup 的单站短期描述，不是物理验证通过。**
来源版本、初态、默认参数、QC 样本数和未做塔高校正限制必须随表保留。

源码身份边界：实测采用 K2 第一版稳定的雪热耦合快照，报告记录源码首尾完全一致。
随后 K2 修改 `physics/snow.py`、`gpu/jax_core.py` 和 `gpu/cuda/column_kernel.cu`
的雪响应运算树；本次暖季窗口气温 13.6–38.07°C、初始无雪，但不能因此追认后续
源码已经复测。报告保留原 hash，**不是最终整合源码的重测证据**。
`examples/fluxnet_smoke.py` 的日志增强也在该次实测后加入：每完成一个 backend
立即 flush，并原子保存 `run_completed=false` 的部分报告，最终才标 true；
此增强已用同一真实月份 NumPy 路径重跑并通过 loader 测试，没有重写原 GPU 结果。

复现命令（在独立快照目录，物理 GPU 编号按排队分配）：

```bash
env -u LD_LIBRARY_PATH PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 \
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  CUDA_PATH=/usr/local/cuda-12.4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 /home/lhwang/venvs/noahmp-phase3b/bin/python \
  examples/fluxnet_smoke.py \
  --csv data/FLX_CN-Cng_FLUXNET2015_FULLSET_HH_20070701-20070730.csv \
  --site CN-Cng --start 2007-07-01 --days 30 \
  --output k6-smoke.json --series data/k6-series.csv
```

局部验证：`PYTHONPATH=src python -m pytest tests/test_fluxnet_loader.py -q`：7 passed。
这些合成单元测试只检查单位、降水守恒、RH/VPD 路径、QC 筛选、缺失与时间断档，
不充当真实站点结果。

## Phase 4：下一个独立任务

1. **固定站点集与留出集。** 覆盖草地/农田/森林、干湿气候、季节冻土和积雪，
   至少多年连续数据，按站点和年份划分校准/验证，保留独立站点。先审计下载版本、
   许可、时区、测量高度、土壤层深、植被和土壤参数、管理事件及数据缺测。
2. **预处理与初值。** 为每站建立明确的测量高度至模型参考高度映射；统一雨雪、
   辐射和湿度单位，保留 forcing QC/ERA 来源。规定 spinup 的周期、终止标准、
   水/热初始条件及缺测处理，不能从评价期观测反推最有利初态。
3. **评价设计。** 预先锁定 LE/H 的 r、bias、RMSE、MAE、日/季节/年汇总及样本数；
   辅以 ET、土温/土水、SWE、径流和水/能量守恒。按昼夜、干旱、雨后、冻融分组，
   报告不确定性。未闭合通量与闭合校正产品分别评价，避免混为同一观测。
4. **参数校准方案。** 先用独立元数据定土壤/植被参数与合理范围，敏感性分析后只
   校准可辨识子集；多目标、有限预算、固定随机种子。校准期与保留评价期隔离，
   记录结构误差，不能通过参数拟合掩盖 K1–K3 数值错误。本轮没有执行校准。
5. **GPU 角色。** 用通过 parity/可靠性验收的 GPU 加速多站、多年和参数集合，
   每次重要图/精度变更保留 NumPy64 子集对照及源码/输入 hash。GPU 加速不是观测
   验证证据。本轮收尾按 Wang 预注册的 M2 门评估；原 H4/K1/K3 红灯及已知限制
   保留，未来 Phase 4 不用调参掩盖后端数值差异。
