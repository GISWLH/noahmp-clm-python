# noahmp-clm-python

Noah-MP / CLM 单柱陆面物理的 Python 重实现，包含 NumPy 参考和原生 JAX、CUDA
后端。各陆面柱独立，没有水平耦合。代码依据公开方程与算法流程编写，未复制
Noah-MP / CTSM 的 Fortran 实现。

**已发布版本为 v0.2.0；v1.0.0 正在开发，尚未完成科学验收或发布。**
v0.2.0 的 GPU 移植已获 Wang [验收](docs/GPU_ACCEPTANCE.md)。当前阶段目标是
完整单柱物理，进度见 [ROADMAP](ROADMAP.md) 和 [执行记录](docs/V1_PROGRESS.md)。
此前 README 的全部运行快照、性能数字与失败记录已移到
[历史记录](docs/README_DEVELOPMENT_HISTORY.md)，并保留原提交和文件哈希。

## 当前验证状态

截至 2026-10-25，开发分支已接入下表中的物理过程。表面求根与新雪密度修订
完成 31 项相关 GPU 专项后已集成，代码与远端已测试候选一致；SOC 评价器按
严格 **低于 5%/百年** 判断，不把等于 5% 记作通过。

| 已取得的证据 | 实际结果与范围 |
|---|---|
| 最终候选 NM 全量 CPU | EHPC 作业 1585 实际退出 0；1168 项通过，835.74 秒，零失败、错误或跳过；[源与完整节点](docs/v1-results/climatology-final-cpu-nr/manifest.json) |
| 表面求根永久 GPU 回归 | 旧源重现原失败；候选 CUDA32/JAX32 的 345 字段最大绝对差 0；对独立 NumPy 的 H/LE 最大差 0.002164094/0.000090556 W/m²；[原始结果](docs/CLM_SURFACE_ROOT.md) |
| 修复相关 GPU 专项 | CUDA1 原 8 个完整模块、31 项均实际退出 0；控制器 7206.43 秒，零失败、错误或跳过，源不变；[完整日志与源](docs/v1-results/surface-root-regression-mx/manifest.json) |
| 最终候选冷启动 | 12 配置各 48 步；EHPC1586 独立核对 576 对柱步，H/LE 最大绝对差 0.005750100/0.003746741 W/m²；原水热碳氮与杂质条件保持；[完整范围](docs/CLM_MULTISITE_SPINUP.md) |
| 最终完整 GPU 回归 | CUDA1 于 UTC 07:29 实际启动；07:38 快照 73/408＝41 项本轮实际通过＋32 项有来源和依赖核验的专项；尚无完整通过结果；[队列与估时](docs/GPU_QUEUE_REPLACEMENT.md) |

这些后端比较不能替代观测验收。当前完整年度、气候及 SOC spinup、多站多年、
留出集和校准仍未完成；已有观测差距保留，例如当前源 Qia 一月 H 偏差
−23.08%～−38.55%、LE 偏差 −31.51%～−11.42%。详见
[一月诊断](docs/CLM_JANUARY_DIAGNOSTIC.md)、[FLUXNET 计划](docs/FLUXNET_PLAN.md)
和 [已知限制](docs/KNOWN_LIMITATIONS.md)。

## 单柱功能

下表列出开发路径中的实现；组件或短窗口验证完成不表示整套物理已通过站点验收。
各报告保留 NumPy 参考、设备验证的实际数字和对应源码版本。

| 过程 | 当前实现与说明 |
|---|---|
| 多层冠层与光合 | 默认 10 层，直散射双流辐射，Farquhar 的 Rubisco/RuBP/TPU 限制，Ball–Berry 气孔联立，叶温和冠层空气温湿度；[光合](docs/CLM_PHOTOSYNTHESIS.md)、[辐射](docs/CLM_CANOPY_RADIATION.md)、[完整冠层柱](docs/CLM_LAYERED_CANOPY_COLUMN.md) |
| 冠层截留与能量 | 分层冰/液水截留、凝结/蒸发、相变与水汽携焓，MOST 实际通量反馈；[截留](docs/CLM_CANOPY_INTERCEPTION.md)、[相变能量](docs/CLM_CANOPY_PHASE_ENERGY.md)、[MOST](docs/CLM_CANOPY_MOST.md) |
| 积雪 | 最多 5 层雪，分层/合并、压实、粒径老化、蓄泄、融化与升华；五波段 SNICAR 简化反照率和黑碳/尘埃气候态沉降，不做气溶胶传输；[雪设计](docs/CLM_SNOW_DESIGN.md)、[设备雪过程](docs/CLM_GPU_SNOW_PROCESSES.md) |
| 土壤水热与冻土 | 雪—土壤联合焓求解、冻融、入渗与根系取水、排水携热、Richards 水分输送；[焓柱](docs/CLM_COLUMN_ENTHALPY.md)、[冻土](docs/CLM_FREEZING_REFERENCE.md)、[Richards](docs/CLM_RICHARDS.md) |
| 降水与径流 | 既有降水相态和径流选项，新增湿球相态选项 5、VIC 可变入渗和 ARNO 基流选项 6；[湿球](docs/CLM_WETBULB.md)、[VIC](docs/CLM_VIC.md) |
| 动态植被 | GDD 生长季与落叶物候，实际 LAI 连接截留、辐射、水热和植物碳库；[物候](docs/CLM_PHENOLOGY.md) |
| 碳氮循环 | 凋落物及快/慢/被动土壤碳库，温湿调节，植物叶/茎/根与储备分配、周转、矿化/固持、氮限制光合；输出自养/异养呼吸、GPP、NPP、NEE；[分解](docs/CLM_DECOMPOSITION.md)、[植物 C/N](docs/CLM_PLANT_CN.md)、[整柱连接](docs/CLM_BGC_COLUMN.md) |
| 作物与灌溉 | 简化积温、出苗/收获和种子/产品 C/N 账，土壤水触发灌溉并计入水量/焓；[作物](docs/CLM_CROP.md)、[灌溉](docs/CLM_IRRIGATION.md) |
| 湿地与甲烷 | 简化厌氧分数、甲烷生成及输送，与整柱水热和碳账连接；[湿地](docs/CLM_WETLAND.md) |
| 冰川与瓦片排水 | 有限冰川柱的雪—冰水热过程；简化瓦片排水实际扣水并携热；[冰川](docs/CLM_GLACIER.md)、[瓦片排水](docs/CLM_TILE_DRAINAGE.md) |
| 多后端与状态 | NumPy、JAX x64/x32、CUDA32；公开 `step_column/run_column/run_columns` 接收完整 `CLMColumnState` 或 `GlacierColumnState`；[公开驱动](docs/CLM_PUBLIC_DRIVER.md)、[原生完整柱](docs/CLM_GPU_EXTENSIVE_COLUMN.md) |

旧 `ColumnState` 入口保留 v0.2.0 的 big-leaf、最多 3 层雪和 Noah-MP 风格物理。
完整 CLM 应显式使用 `CLMColumnState`，见下方示例；旧入口不自动等同于完整 CLM。
独立 CuPy 物理直译没有实现，原 CuPy host-reference 只作 CPU 对照锚点。

新安江径流 7、Dynamic VIC 8 和可选火过程仍未实现。本阶段不包含整机
CESM/WRF 耦合、河道演算、侧向地下水、土地利用变化或城市子系统。

## 验收口径

- 日/月 ET、H、LE、GPP、NEE、土壤温湿、径流和 SWE 与官方参考或观测比较，
  非近零量相对偏差不超过 10%；瞬时通量要求 r > 0.85。GPP/NEE 按任务中
  指定的偏差或相关条件报告。近零量使用明确的绝对容差。
- 水、热、碳、氮及杂质守恒单独核算，不用 10% 精度口径豁免收支问题。
- 新模块先 NumPy 独立参考与单元验证，再同步 JAX/CUDA；同精度门保持 1e-6，
  整数与 dtype 条件保留。旧 H4/K3/M4 的 1e-4 红灯是已接受的历史限制。
- 完整科学验收仍须真实多站多年、测高敏感性、spinup、留出集和校准。
  ERA5 已检查的文件缺少 ssrd/strd/tp，区域一年验证目前记为未测。

## 运行与计算环境

本项目的 CPU 模型、参考、pytest 和数值分析在 **EHPC PBS 计算节点**运行，
GPU 在 **xuanze 物理 CUDA1**。Windows 工作区用于编辑、Git 和静态文件核验。
环境与提交方式见 [environment.md](environment.md)、
[EHPC README](benchmarks/ehpc/README.md)、[GPU 环境](docs/GPU_ENV.md)。

```bash
# EHPC 计算节点：安装后运行显式 CLM 示例
pip install -e '.[dev]'
python examples/run_clm_column.py
```

[CLM 示例](examples/run_clm_column.py)包含显式 C/N 初库、湿球/VIC/MOST 和
公开驱动调用。真实站点输入、测高假设与验证步骤见
[FLUXNET 计划](docs/FLUXNET_PLAN.md)；官方参考进度见
[HRLDAS 对标](docs/HRLDAS_CURRENT_SOURCE.md)和 [CTSM 参考](docs/CTSM_REFERENCE_PREPARATION.md)。

```bash
# xuanze：使用已配置的 CUDA Python 环境
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=GPU-cf7322ad-e2e0-ddd3-76d7-a07e2dd73452
export CUDA_PATH=/usr/local/cuda-12.4
export PATH="$CUDA_PATH/bin:$PATH"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src
NOAHMP_REQUIRE_GPU=1 python -m pytest tests/test_gpu_parity.py -q
```

该命令只运行指定测试文件，不代表全量验收。`.[gpu]` 只安装 CPU JAX；CUDA
依赖按环境文档安装。既有全量任务使用冻结源码、独立缓存和设备完成屏障，
结果连同实际父进程退出码归档。历史速度测试见 [BENCHMARK](docs/BENCHMARK.md)，
不能将旧物理版本的速度标为当前 CLM 性能。

## 代码与许可

- `src/noahmp_clm/physics/`：NumPy 物理参考；`gpu/` 与 `gpu/cuda/`：原生设备实现。
- `driver.py` 和 `clm_enthalpy_driver.py`：公开入口及完整 CLM 状态推进。
- `benchmarks/`：参考比较、站点评价和计算环境入口；`tests/`：组件、整柱和设备回归。
- `docs/`：方程、限制、验收口径和原始结果；历史失败保留原源码身份。

方程参考 Noah-MP v5.0 与 CLM5/CTSM。本仓库新增代码采用 MIT 许可，上游许可
归上游；Leafweb 数据采用其单独的 CC-BY-4.0 许可，见
[来源和使用边界](docs/CLM_PHOTOSYNTHESIS.md)。
