# GDD 季节落叶物候（组件及三后端整柱接入）

独立依据 [CLM5 技术说明 2.20.3](https://escomp.github.io/CTSM/release-clm5.0/tech_note/Vegetation_Phenology_Turnover/CLM50_Tech_Note_Vegetation_Phenology_Turnover.html)
实现温带季节落叶控制器。第三土层温度高于0°C的部分乘 dt/86400 累积为GDD；
可用年均气温的公开阈值关系生成站点参数，也可显式输入阈值。
冬至重启累积，春季超过阈值开始展叶；夏至前未触发则等待下一季。
日长缩短到39300秒后开始落叶，默认展叶30天、落叶15天。

简化部分：叶面积比例在展叶/落叶期间线性变化，尚未由植物碳转移库控制。
此组件输出LAI及叶面积增减比例，不产生或消耗虚构的植物碳；P4需将其接到
植物分配/凋落物的实际C/N转移。适用季节落叶类型，不能作为热带、常绿或
水分胁迫落叶型的通用算法。日长、季节趋势和冬至跨越由显式日历/测站输入，
没有默认北半球日期。完整分层焓柱已启用 `OptDynamicVeg=5`；植被覆盖比例
保持输入的最大覆盖比例，LAI由叶面积比例状态计算。

`PhenologyState` 保存GDD、叶面积比例、阶段计时器、阶段与本季可触发标志。
计时器使用秒，常用半小时步长可准确表示30/15天；避免每日分数反复相减
累积的触发时刻漂移。阶段在步边界切换，触发的当前步计入新阶段。
只允许每个冬至周期一次展叶。实际年长/南北半球/极地日历映射及长期整柱
运行仍需验证，不以组件结果替代站点LAI或GPP验收。

先完成NumPy及独立解析对照：404个温度/步长组合与60位Decimal的积温对照，
最大差 **2.22044605e-16 K·day**；两年逐日分段解析LAI最大差
**8.88178420e-16 m²/m²**，展叶日11/376、落叶日300/665。
错过春季、休眠再启动、半小时阶段计时与非法状态均已覆盖。

冻结 `p3-phenology-b` 的原生JAX32/JAX64、CUDA32及NumPy目标：
**9 passed / 0 skipped，24.34秒**。512个量化单步状态覆盖全部阶段与开关，
float32对NumPy最大GDD差 **2.13623047e-5 K·day**、叶面积比例差
**5.50187664e-8**、剩余时间差 **0秒**、LAI差 **4.43352594e-7 m²/m²**。
float64全部输出差0；CUDA32/JAX32全部浮点输出差0，整数阶段/事件逐项相等，
仍保留1e-6同精度标准。此时完整柱LAI尚未由本组件驱动。

首轮A的设备测试在NumPy断言格式化数组绝对容差时抛出TypeError，
5个NumPy测试完成但整个目标失败；B将三种状态量分别按各自单位检查，
不改变原定容差和物理。失败及成功日志均保留：
[A失败](v1-results/p3-phenology-a-target.log.gz)、
[B完成](v1-results/p3-phenology-b-target.log.gz)。

2026-10-14冻结 `p3-phenology-column-a` 已连接第三土层旧态温度、GDD、
LAI、降水截留、辐射、气孔、叶温及土壤水热。NumPy独立于设备实现；
设备路径先原生反演第三层焓，再原生更新物候，最后进入原生完整分层柱。
冠层叶面积减少不会丢弃截留水或冰：超出新容量的物质及其焓由现有出口
路由到雪/土壤。零LAI的旧冠层存量全部转移，GPP归零。

入口为 `CLMColumnState.with_layered_canopy().with_phenology(parameters, initial)`，
NumPy步需显式 `phenology_environment=PhenologyEnvironment(daylength_seconds,
increasing_daylength,winter_solstice,latitude_degrees)`。该简化方案限定纬度绝对值
大于19.5°的季节落叶类型。设备使用 `pack_clm_phenology_columns` 和
`step_clm_phenology_columns_jax/cuda`；`next_phenology_arrays`保留下一步所需
的LAI、GDD、阶段/计时器以及全部物理存量。旧70浮点ABI明确拒绝选项5。

整柱及相关组件目标 **28 passed / 0 skipped，520.12秒**。8类暖/冷雪工况、
四个物候阶段连续两步，包含湿球选项5和VIC选项6。CUDA32/JAX32全部输出
最大差 **0**；float32对NumPy最大LAI差 **2.38418580e-7 m²/m²**、土温差
**1.30884390e-5 K**、H/LE差 **0.00140934 / 0.00215707 W/m²**、
GPP差 **5.71925638e-5 µmol/m²/s**。float64最大土温差1.13687e-13 K。

独立float32水/焓残差最大 **3.08838207e-5 kg/m² / 10.1744385 J/m²**，
逐工况数值界最大 **0.00466156 kg/m² / 2213.78478 J/m²**；float64残差
**1.30812028e-13 kg/m² / 0.000142913439 J/m²**，最大热界0.000158387325 J/m²。
守恒检查未使用10%豁免。证据：[整柱摘要](v1-results/p3-phenology-column-a-summary.json)、
[完整日志](v1-results/p3-phenology-column-a-target.log.gz)。
真实站点LAI、多季整柱、植物C/N分配/凋落物连接尚未完成。
