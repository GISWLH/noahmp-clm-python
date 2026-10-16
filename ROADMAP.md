# ROADMAP — CLM 单柱物理

## 当前阶段：v1.0.0 完整单柱物理

按 Wang 2026-10-13 指定的 P0–P6 顺序推进，见
[执行清单](docs/V1_PROGRESS.md)与[验收规范](docs/V1_ACCEPTANCE.md)。
旧 H4/K3/M4 数值门已退役为已接受限制，历史复验结论保留在下方。
本阶段排除城市模型、土地利用变化、整机耦合、河道和侧向地下水。

- [x] P1指定的融尽余热路由、AHC平滑和厚雪均温漂移修复及365天复验。
- [x] P2单叶Farquhar/Ball–Berry和多层短/长波独立组件，NumPy/JAX/CUDA同步。
- [x] P2 NumPy叶温/空气温湿/共享供水联立与无雪开发柱入口，110方程工况、1008柱步守恒。
- [x] P2冠层联立组件JAX/CUDA：302工况收敛，同精度全字段最大差0。
- [x] P2无雪新柱步原生JAX/CUDA：88柱＋三类柱7天，同精度全状态/柱诊断最大差0。
- [x] P2 NumPy五层雪组件：质量/焓、蓄泄及阻水径流、压实/粒径、SNICAR与沉降映射；独立方程及雪组件365天守恒。
- [x] P2 NumPy雪—土壤共享传热及融水焓入渗，联合365天守恒；固定雪面冠层/空气与直散射边界的独立方程核验。
- [x] P2 NumPy冠层—五层雪—土壤质量/焓状态和连续时间步；水汽、根水/冠层储水显热及排水携焓，独立方程核对。
- [x] P2土壤焓反演及雪—土壤共享传热原生JAX/CUDA；252工况和四后端7天固定质量热柱验证，详见[热组件报告](docs/CLM_GPU_ENTHALPY.md)。
- [x] P2五波段SNICAR及冠层—雪短波原生JAX/CUDA；368个雪况、349个耦合工况，最大光谱吸收偏差0.6048%，详见[雪辐射报告](docs/CLM_GPU_SNOW_RADIATION.md)。
- [x] P2降雪重分层、雨混合、雪蓄泄/阻水径流和冰升华原生JAX/CUDA；2015个单步工况、4032次连续操作同精度最大差0，详见[输送报告](docs/CLM_GPU_SNOW_MASS.md)。
- [x] P2雪表层混合相水汽/霜露、压实/粒径/沉降原生JAX/CUDA；2035/4739工况（含全部老化表节点），粒径最大偏差6.616%，详见[过程报告](docs/CLM_GPU_SNOW_PROCESSES.md)。
- [x] P2分层冠层截留/冰水相态/水汽携焓组件NumPy/JAX/CUDA：1728独立方程工况、365天NumPy守恒、428设备工况与四后端七天过程链；同精度最大差0，见[冠层截留报告](docs/CLM_CANOPY_INTERCEPTION.md)。
- [ ] P2冠层冰雪与叶温/整柱能量连接、完整稳定度连接、默认驱动切换和叶片/冠层实测达标。
  NumPy焓—叶温—空气湿度联立开发中，75个独立方程工况、三类冠层七天1008步完成；18项目标测试通过，累计焓残差最大2.519 J/m²，
  该NumPy检查点全量528 passed / 0 skipped，见[当前结果与失败记录](docs/CLM_CANOPY_PHASE_ENERGY.md)。
  [原生相变联立](docs/CLM_GPU_CANOPY_PHASE_ENERGY.md)22项目标测试、75个设备工况完成，同精度最大差0；原生七天4032次联立求解完成，同精度最大差0，最终全量532 passed / 0 skipped。
  [分层完整柱](docs/CLM_LAYERED_CANOPY_COLUMN.md)NumPy 44项目标及三柱七天1008步完成；原生整柱4项目标通过，同精度最大差4.76837e-7，最终冻结全量563 passed；四后端七天4032柱步已完成，日H最大偏差0.00876828%。
  [MOST通量反馈](docs/CLM_CANOPY_MOST.md)原生完整耦合5项目标通过，同精度最大差2.98023e-8；连续稳定工况收敛慢修复后11项NumPy回归通过，修后三柱一天144步完成；修后设备6项目标通过。
- [ ] P3湿球选项5：334个独立根及1000个原生组件工况完成；完整驱动及设备选项矩阵69项目标通过，同精度差0；连续验证和官方差异核对待完成，见[湿球记录](docs/CLM_WETBULB.md)。[VIC](docs/CLM_VIC.md)原生完整柱8工况同精度最大差4.76837e-7；旧湿球合同失败修正后已复验。
- [x] P3 [GDD物候组件与整柱LAI](docs/CLM_PHENOLOGY.md)：选项5接入实际截留/辐射/水热，28项目标通过、8类柱两步同精度差0；长期站点与C/N联动仍待验收。
- [x] P4 [凋落物及三库土壤C/N分解独立组件](docs/CLM_DECOMPOSITION.md)：日步对连续ODE最大碳库/呼吸偏差0.22448%/0.52504%；三后端13项目标和四后端七天完成，同精度差0。
- [x] P4植物C/N、储备、组织分配/凋落、维持/生长呼吸与叶N反馈组件：三后端14项目标通过，七天植物—土壤联算同精度差0；[完整数字及简化](docs/CLM_PLANT_CN.md)。实际整柱GPP/NEE与长期反馈验证尚未完成。
- [x] P4 [NumPy整柱C/N连接](docs/CLM_BGC_COLUMN.md)：实际叶碳LAI、叶N容量、凋落、土壤N竞争、RA/RH/NEE，36项新目标及相关回归。
- [x] P4整柱C/N原生JAX/CUDA连接：8类柱连续两步全部状态/诊断同精度差0，独立四预算通过。
- [ ] P4完整反馈连续验证、真实强迫spinup及GPP/NEE观测验收；新全量及七天审计运行中。
- [ ] P5作物：热量物候/收获及C/N转移NumPy组件8项通过，整柱/GPU和作物观测待做；[简化与结果](docs/CLM_CROP.md)。
- [x] P2固定雪/土表温冠层、冰面交换和液水储热/根水携热原生JAX/CUDA；425工况同精度全字段最大差0，详见[边界报告](docs/CLM_GPU_CANOPY_BOUNDARY.md)。
- [x] P2土壤入渗/取水/重力排水及携热原生JAX/CUDA；1062工况、三柱7天过程链，日底部出流偏差最大0.00733%，详见[土壤输送报告](docs/CLM_GPU_SOIL_TRANSPORT.md)。
- [x] P2冬季冠层—雪—土壤表面求根、实际水汽/根水携焓、联合传热及雪融水入渗接成原生JAX/CUDA时间步；50工况同精度最大差0，最终全量492 passed / 0 skipped，详见[表面步报告](docs/CLM_GPU_CANOPY_SURFACE.md)。七天4032柱步完成，同精度全轨迹最大差0。
- [x] P2降水、液水截留、雪/土壤入渗、重力排水及沉降接入完整原生焓柱；16类工况目标4 passed / 0 skipped，同精度最大差4.76837e-7，最终全量496 passed / 0 skipped，详见[完整柱报告](docs/CLM_GPU_EXTENSIVE_COLUMN.md)。连续七天4032柱步完成，同精度全轨迹最大差0，日H最大偏差0.00845985%。
- [ ] P2完整焓柱连续核验与官方/观测雪验证；P3选项；P4碳氮；P5扩展；P6完整验证与发布。

截至2026-10-13，**v0.2.0 的阶段3/M0–M5 GPU移植收尾已完成并验收**，
已合并main并发布标签；这不表示下面全部物理开发和科学验证已完成。
NumPy64、JAX64、JAX32、CUDA32在已测范围内可用、总体一致，原H4/K3及M4
近零冠层水差异仍保留，发布例外见[验收记录](docs/GPU_ACCEPTANCE.md)，
使用边界见[已知限制](docs/KNOWN_LIMITATIONS.md)。

## 设计不变量

贯穿所有阶段：**单柱状态全封装、柱间零耦合**。任何阶段都不允许出现"柱 A 读柱 B 状态"
的代码。批量轴（格点 × 时间步）永远是独立的，这是 CUDA/JAX 向量化的前提。

## 阶段 0 — v0.1 初始基线（已完成的历史阶段）✅

纯 Python + numpy，可读优先：

- 4 层土壤水热（Noah-MP 默认分层）、单层 bulk 积雪、big-leaf 冠层
- 表面能量平衡割线迭代；土壤热传导采用冻结系数的后向 Euler；土壤水 fill-and-spill + Campbell 排水
- 合成强迫示例 + 三条硬约束测试（能量闭合 / 水平衡 / 批量独立性）

## 阶段 1 — Noah-MP v5.0 对标

目标：与 NCAR/noahmp 离线单柱运行在 FLUXNET 强迫下定量可比。

- [x] 多物理选项开关：precipitation（Jordan/BATS/freezing threshold）、surface drag
  （MOST/Chen97）、runoff（TOPMODEL/BATS/free drainage）、snow albedo（BATS/CLASS）、
  stomatal（Ball-Berry/Jarvis）、soil moisture stress（Noah/CLM/SSiB）和四种 ground resistance
- [x] 冻土过程（soil ice + apparent heat capacity）、多层积雪（最多 3 层）
- [x] MOST surface drag、Sakaguchi--Zeng ground resistance、Anderson snow compaction
- [x] Noah-MP `ConfigVarType.F90` 的 `Opt*` 配置框架与 HRLDAS 默认值；动态植被固定为 4，
  作物/灌溉/湿地/冰川/瓦片排水关闭
- 冠层分层辐射（双流近似简化版）与植被能量平衡迭代
- [x] 官方源码验收：NCAR/noahmp v5.0 `src/` 的 135 个 Fortran 模块可编译，
  `AtmosForcingMod` 的 Jordan/BATS/冰点阈值与 Python 实现逐点一致
- [ ] 对标实验：选 3–5 个 FLUXNET 站点，对比日 ET / 感热 / 土壤温湿度的偏差与相关系数；
  `examples/parity_noahmp.py` 已提供 CSV 对比入口，待提供带站点参数和强迫的 HRLDAS 官方输出。

## 阶段 2 — CLM 柱子集

以下为尚未完成的物理扩展，不属于v0.2.0的GPU移植验收范围。

- CLM5 生物地球物理核心的柱版本：多层冠层（~10 层）光合—气孔耦合（Farquhar + Ball-Berry）
- 保留"柱"抽象：CLM 的 gridcell → landunit → column → PFT 层级在本仓库中
  坍缩为"一个 column = 一个独立 PFT 柱"，批量轴天然对应 PFT × 格点
- 与阶段 1 共用 driver：`run_columns` 签名不变，只换 `step_column` 里的物理包

## 阶段 3 — 批量加速（JAX / CUDA，v0.2.0已验收）

- [x] 环境侦察、固定状态/强迫 ABI（`docs/STATE_SPEC.md`）和 50 柱 round-trip
- [x] JAX 定长单柱核心：`jit`、`vmap`、`lax.scan`、`lax.switch`、固定迭代 Thomas/割线
- [x] CuPy host-reference 正确性锚点（CUDA headers 缺失时明确 unavailable）
- [x] 性能基准脚本和本机 N=100 实测记录
- [x] 已实现默认物理的 SoA CUDA kernel、256-thread grid-stride loader；xuanze nvcc编译并实际执行，发布前编译成功且保留1条既有未使用变量warning
- [x] JAX `vmap(scan)` 批量入口及 Linux GPU float64 8 柱 × 48 步混合门
- [x] GPU 已实现选项 35 个取值抽查和一日 float32/float64 累积精度实测
- [ ] CUDA32/NumPy64 N=1024 原H4 parity（同精度CUDA/JAX32已通过，NumPy64的residual、近零H/G等仍超差）
- [x] 最终物理源码N100000三预热/三正式含校验计时，四后端365天及64800柱域检查完成
- [x] EHPC串行/多进程/两节点MPI、xuanze及HF ZeroGPU兼容性检查
- [x] M0–M5证据归档、Wang明确验收、main合并与v0.2.0标签
- [ ] 独立CuPy物理直译实现（可选后续）；现有CuPy host-reference为CPU对照锚点，
  不能称为另一个GPU物理实现，手写CUDA kernel已经完成
- [ ] A100硬件性能检查（未测、非本次收尾门）

原H4两项仍失败；Wang以M2总体一致性口径及随后明确的M4验收例外完成收尾，
不把旧门改绿。全量GPU测试147通过/2原H4失败/0跳过。N300000 CPU基线已收齐；
旧五档性能来源与最终物理N100000复测分开报告，不把旧CPU基线拼成新版本加速比。
最终数字和来源边界见[GPU_SUMMARY](docs/GPU_SUMMARY.md)及[BENCHMARK](docs/BENCHMARK.md)。

## 阶段 4 — 完整真实数据验证（后续）

- [x] 四个FLUXNET站点各30天有界smoke、后端对照和LE/H的r/bias/RMSE；未调参
- [ ] 站点元数据及测高映射、spinup、多站多年评价、独立留出集和校准
- [ ] 完整官方HRLDAS站点输出对标（与阶段1未完成项关联）
- [ ] ERA5格点驱动对照：本轮可达数据缺下行辐射和总降水，保持未测

具体站点、指标和校准设计见[FLUXNET_PLAN](docs/FLUXNET_PLAN.md)。四站短期
后端数值吻合不等于观测物理验证；完整CLM核心、11个明确未实现物理选项等
开发范围也不能因GPU移植验收而标为完成。

## 不做的事

- 不移植整机 CESM/WRF，只做单柱物理
- 不做河道演算、地下水侧向流（那是另一个并行模式）
- 新增物理先建立NumPy参考与可核验用例，再同步对应GPU实现和精度检查
