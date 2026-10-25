# 柱内垂向水文边界复核（OC，2026-10-25）

[完整作业终态、日/月/年数值、原源码与配置索引](v1-results/ctsm-vertical-hydrology-oc/manifest.json)。

当前已实现柱内垂向土壤水分输送，含显式零底通量 Richards 及携焓版本；**尚无独立含水层库存、预报地下水位及土柱—含水层双向交换**。该限制不能与本项目明确排除的柱间侧向地下水耦合混为一谈。原官方 R 本身也没有启用 aquifer 边界，不能以 WA 字段存在或冷态变化作为“缺少含水层造成偏差”的证据。

本次只读复核 12 个当前源文件，它们与冻结 NM584 逐字节相同；另核对原 KG3 和 NP12 批次实际引用的 15 份配置。索引为 `clm-vertical-groundwater-oc-index.json`，生成器仅用标准库读取文本、JSON 和哈希。没有更改源码、配置、观测、QC、科学门或当前年度轨迹，也未在 Windows 计算物理数组。

## 当前两条路径与真实库存

| 路径 | 实际做法及范围 |
|---|---|
| 未设置 `soil_hydrology` | `run_fluxnet_column.hydrology_configuration` 返回 None；柱调用 `drain_soil`，逐层向下搬运液水及其焓，受剩余孔隙和液水约束。底层出流实际离开土柱。 |
| 当前 KG3／NP12 的 VIC6 | 全部实际引用配置均未设置 `soil_hydrology`，`OptRunoffSurface=OptRunoffSubsurface=6`、`OptTileDrainage=0`。因此这些冻结年度用 VIC 入渗与 ARNO 底层出流，并未采用显式 Richards。ARNO 的液水“库”就是最底土层，非独立含水层；扣水与携热出口记入原总账。 |
| 显式 `richards_zero_flux` | NumPy、JAX、CUDA 实际调用路径已接通。土层内部液水可沿压力／重力梯度重新分配并携焓；水文底边界无液水及其平流焓通量，底出流置零。地表接纳溢流、雪径流与可显式启用的瓦片排水另记真实出口。要求地下径流选项3；不能在请求 ARNO 时静默去掉出口。这里不把水文边界等同于整个热传导模块的热底界。 |
| `groundwater_depth` 与 TOPMODEL | 深度是静态站点参数；简化 TOPMODEL 用它重新分配既有候选径流，不推进含水层水量或水位。`CLMColumnState.water_kgm2` 和原 native 水热总账只有土壤、雪与冠层库存。16 个基础状态及23个当前冷检查点字段中没有另一个含水层库存。 |

代码位置：`clm_enthalpy_driver.py:50,239,264`；`physics/soil_drainage.py:12`；`physics/vic.py:54`；`physics/richards.py:15,277`；`gpu/clm_richards_column.py:9,25`；`clm_state.py:44,276`；`benchmarks/run_fluxnet_native.py:85`。`richards.py` 开头“not yet wired”是过时说明，与实际调用和后续文档不符；本次未改冻结源码。`docs/KNOWN_LIMITATIONS.md:259,267` 已明确保留完整 CTSM 含水层／饱和带出流限制；其中其他更早的“未接柱”等文字不能当作当前实现状态。

## 原 R 的零底通量不等于没有地下径流

固定 CTSM commit `fdfc03a0abec838ec0b66663a6a2f5e619cd2611` 的源语义及 R/S 实际 `lnd_in` 一致：`lower_boundary_condition=2`、`soilwater_movement_method=1`、`baseflow_scalar=.001`。`SoilWaterMovementMod.F90:50–55,196–211,1354–1375` 将2定义为零底通量，`use_aquifer_layer()` 仅对3／4返回真。因此 R 实际关闭独立 aquifer，QCHARGE 不能从字段名称猜测。

无 aquifer 分支仍调用 `PerchedWaterTable`／`ThetaBasedWaterTable`，从土壤湿度诊断水位；随后调用 `PerchedLateralFlow`／`LateralFlowPowerLaw`。后者按局部地形、冰阻和水位计算饱和区出流，从水位以下至基岩底之间的**实际土壤液水层**扣除，受可移出水量限制；前者在具备条件时从冻层以上的滞水层扣水。它们是本地土柱参数化出口，没有在此审阅中要求实现相邻柱之间的地下水交换。源位置：`HydrologyNoDrainageMod.F90:211–229`、`HydrologyDrainageMod.F90:129–147`、`SoilHydrologyMod.F90:1747–1812,1865–1915,2074–2128,2219–2226`。

原 OA 读回的 WA 从冷文件4000 mm变为首 restart 5000 mm，不能解释为1000 mm降水或补给。`BalanceCheckMod.F90:143–153` 在相应水位条件下将 WA 设为基线；`clm_varcon.F90:128` 的基线是5000 mm。`TotalWaterAndHeatMod.F90:254–266,525–541` 用 **WA−5000** 计入相关总量，并说明此默认 CLM5 水库为不参与实际交换的虚拟基线。此次没有重建首次写入的逐调用轨迹，也没有把字段跳变加入水量收支。

## 原全年实际贡献

EHPC `1592.manager` 已实际 **F/Exit0**，PBS墙钟17秒、reader **3.3851491529494524秒**，模型运行次数0。读回采用原 R 封存件与固定 portable 环境，原17521存储行只排除启动 `[-1800,0]`，保留全部17520个正半小时区间；时间界与原气象输入逐区间绑定。365日、12月按原试验的2003本地年（UTC+8）分组，所有7个字段原掩码／非有限数均0，没有填补。完整归档 SHA 见 `ctsm-vertical-hydrology-oc-package.json`；数值来自 EHPC 读回 JSON。

| 实际陆面字段 | 全年 mm | 非零半小时数 |
|---|---:|---:|
| QDRAI | 125.39910458945087 | 1457 |
| QOVER | 122.19870647717673 | 3177 |
| QRUNOFF | 247.59781107155328 | 4495 |
| QCHARGE | 0 | 0 |
| QDRAI_PERCH／QH2OSFC／QRGWL | 各0 | 各0 |

QDRAI 占原全年 QRUNOFF 的 **50.64628966094203%**。它只在5月（64.70769935331191 mm，占当月径流68.1214039229464%）和6月（60.69140523613896 mm，占86.80952969631122%）非零，其余10个月为0；原313首48步为0的结论仍成立，但不能推广为全年。完整逐日／逐月累计和非零步数均在原结果 JSON。比例只是诊断贡献，不是模型相对误差；分母绝对累计≤0.05 mm时报告 null 及绝对量，不新增验收门。

## 结论与下一最小动作

已证实的区别是：R 有来自土壤饱和区的季节性参数化出口；当前 VIC6 从最底土层按 ARNO 退出；显式封闭 Richards 没有该饱和区出口。这项原参考年贡献不小，值得在已有全年结果完成后优先比较，**尚未证明它造成当前模型任何通量偏差，也未证明加入含水层会改善结果**。

下一步仅用冻结年度已有输出，在原时间窗对照5–6月及其余月份的总径流、
土层水与 ET；先列明土质、土层／根区、气象和 SP/CN 差异，再判断库存响应。
当前原生年度的固定 TRACE 仅保存总 `runoff_mm`，没有逐步地表／地下拆分；
续算额外保存的是 BGC 子集。模型的全年分项因此仍未测，不能从总量或末库存
反演后冒充实际出流。原冷48步和首次续步中的分项不代表5–6月。
若未来需要分项诊断，应另行声明并验证实际捕获，不改变活动年度控制。
不得把同名 `runoff_deep` 自动当成相同机制。无需重跑 R/S48、换当前年配置
或扩展柱间侧向地下水。若未来确需独立垂向含水层，应另立物理库存、初态与
水热硬守恒合同，独立验证后开新轨迹；不能把新库存接入当前16状态／23字段冻结链。

本次没有追逐已满足 P0 的 OB 沉降末位差，没有新增科学通过信用。
