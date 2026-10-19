# ROADMAP — CLM 单柱物理

## 当前阶段：v1.0.0 完整单柱物理

按 Wang 2026-10-13 指定的 P0–P6 顺序推进，见
[执行清单](docs/V1_PROGRESS.md)与[验收规范](docs/V1_ACCEPTANCE.md)。
旧 H4/K3/M4 数值门已退役为已接受限制，历史复验结论保留在下方。
本阶段排除城市模型、土地利用变化、整机耦合、河道和侧向地下水。

19:02UTC当前检查点：

- [x] [日长容量／十日温度记忆／垂直叶N组件及NumPy完整柱](docs/CLM_CANOPY_CAPACITY.md)：先NumPy41／49／53及32次连续DP原硬审计，再设备；532F完整981CPU、组件／逐层／原相变20实际0，连续容量积分最大偏差0.04271%、新逐层300原始字段差0。
- [ ] 原生20/23数组整柱三项在GPU1运行，须完成两精度、结构／MOST开关、两步全部返回字段及携带数组1e-6；当前实际收集1344=981+363，其余340GPU尚待安排。旧完整任务／519H等待器继续。
- [ ] 将新容量接站点JSON／20/23数组连续检查点，完成新选项气候年／spinup／多站多年／留出／测高敏感性及科学验收；非球形PFT角度、冠层冰雪组织光学和K78等原缺口保留。
- [ ] 旧519H原冷年NumPy576／CUDA1200快照，后者跨过原1110时间；均非新容量年度或完整年，未续失败轨迹，未发布v1.0.0。

17:57UTC当前检查点：

- [x] [1110精确拐点局部Jacobian](docs/CLM_QIA1110_PHASE.md)：NumPy先行29、最终H完整943CPU及xuanze相变14实际0；原整步两端实际0，全部380字段最大差3.64e-12，原单步／累计MHCN与杂质界满足。
- [x] [PFT最终D整柱BGC3](docs/v1-results/pft-optics-full-gpu3-d/manifest.json)实际0、全部原始同精度字段差0；[早期506完整1241](docs/v1-results/phase-final-check-full-a/manifest.json)实际0，保持各自源码身份。
- [ ] H当前规范1297余340GPU排队；D完整1291／旧511完整1254仍实际运行，不停止实际worker。新原冷NumPy295快照144/17520，新CUDA原冷年已启动，旧两端failed1110原样保留。
- [ ] 日长容量／十日温度记忆／垂直叶N、K78原1e-6、完整气候年／spinup／多站多年／留出／测高敏感性／科学验收和发布。CPU只在EHPC PBS计算节点，GPU只在xuanze。

下方带时间的运行状态为当时快照，不代表后续终态。

16:58UTC最终检查点：

- [x] [PFT光学最终D](docs/CLM_CANOPY_OPTICS.md)完整940CPU、设备目标20实际0，47例及完全散射全部624字段差2.00e-36；A整柱BGC3另有实际0，所有B/C失败保留。
- [ ] [最终D整柱3及完整规范1291](docs/v1-results/pft-optics-full-deferred-d/manifest.json)：整柱GPU0运行，完整GPU在其实际退出后用GPU0执行；旧511实际GPU1任务未停止，只以UID／无子进程守卫替换自己的闲置D等待器。
- [ ] [原冷NumPy年912快照](docs/v1-results/pft-optics-ehpc-checkpoint-d/manifest.json)，原CUDA1110失败／隔离候选未生产验收，K78及完整年/spinup/多站多年/测高敏感性/科学验收/发布仍待。

16:49UTC当前补充：

- [x] [显式PFT叶／枝光学接入](docs/CLM_CANOPY_OPTICS.md)：独立47例叶PAR最大相对差0.04847%，EHPC937及940CPU实际0；NumPy/JAX/CUDA、相变／MOST／BGC／站点输入与叶片PAR分配连接。
- [ ] B/C完全散射边界设备失败保留，D统一CUDA与JAX更新顺序后受控624字段差2.00e-36；D完整940CPU／目标20、规范1291及整柱BGC三项仍待。
- [ ] [原CUDA年F失败1110](docs/CLM_QIA1110_PHASE.md)，NumPy参考同样失败；隔离单侧Jacobian候选22次收敛，未生产／设备验收。NumPy年864快照、K78、完整年/spinup/多站多年/留出及发布仍待。
- [ ] 日长容量、十日温度记忆、垂直叶N正式连接；非球形PFT叶倾角和冠层积雪组织光学未实现。

下方为历史快照。

15:25：新增[完整冠层输入及首日诊断结果](docs/CLM_CTSM_CANOPY_INPUTS.md)已归档。
NOAA近似日长容量上界后H−14.2811%、GPP均值+59.7611%/r0.990215；不选择默认值。
正式PFT光学/日长/温度记忆/垂直叶N连接仍待。NumPy年384快照、CUDA自身有效
48经审计后同源续跑96快照；完整年/科学与发布仍待。下方为此前检查点。

14:58UTC当前状态（下方为历史快照）：

- [x] [已有湿层相变端点](docs/CLM_QIA735_PHASE.md)：先NumPy26/独立方程，再三后端；最终EHPC262全CPU911、xuanze相变11实际0，原735整步热残差3.84025e-8 J/m²、300原始同精度字段差0。A版测试调用失败与B版仅动态参数绑定的修正均保留。
- [x] [3798独立双端整步](docs/CLM_QIA3798_PHASE.md)：JAX实际0，EHPC259重新审计两端全部380字段差0、原MHCN/杂质硬界满足。
- [ ] 最终511源实际收集1254=911CPU+343GPU，余332已排队；新原冷NumPy年144步快照、CUDA年GPU0启动，未计全年通过。K78原始同精度仍差1.52588e-5 J/m²。
- [ ] 完整气候年、spinup、多站多年/留出、测高敏感性与官方同条件科学验收；合并main和v1.0.0发布。

15:18补充：P2还须正式连接[已定位的冠层输入缺口](docs/CLM_CTSM_CANOPY_INPUTS.md)：
PFT叶/枝光学、日长容量因子、十日温度记忆与垂直叶N。原组件通过不抵充这些
物理连接。首日入射散射比例两端均约0.860234425；NOAA近似因子0.5716只用于
预先声明的EHPC267/268诊断，未替换默认值。新CUDA E物理48后的父后处理1已
审计，F同源有效续跑96快照；原错误/未测/完整年待办均保留。

13:32UTC：

- [x] 当前510 NumPy原冷首周336及同源有效前缀续跑审计；EHPC PBS245原7天/四土层[后端工程对照](docs/CLM_QIA3798_PHASE.md)全部偏差满足既定门，全周H+0.000580%、GPP+0.000140%。
- [x] [官方S48生理诊断](docs/CLM_CTSM_PHYSIOLOGY_INPUTS.md)：模型/PBS0、27原字段精确R重放、48原始重启；缺测量明确保留，预先声明PFT参数首日对照。
- [ ] 原509C CUDA全年在3798失败，隔离数值候选尚未验收；当前K78原始1e-6仍失败，差1.52588e-5 J/m²。当前1250余GPU、Python完整年、spinup、多站多年/留出和发布仍待。旧快照中的全年running不代表其后终态。

12:14UTC：[原506F严格首48](docs/CLM_QIA_FIRST_DAY.md)实际worker/controller0，
原逐步CUDA32/JAX32最大差3.63798e-12，不抵充当前510规范1250。当前510
NumPy已running/192/17520，509C CUDA年12:09为1680/17520、同精度null，
分别跨过旧182/1519；当前余332GPU、完整年/spinup/观测/发布仍待。
以下为此前检查点，完整证据见[执行记录](docs/V1_PROGRESS.md)。

12:07UTC当前记录：

- [x] P6 [官方Q/R水热及冠层初态实际对照](docs/CLM_CTSM_COLD_INITIALIZATION.md)：各17520正区间、模型/PBS0，R水热/冠层读回误差0，全年从首区间起LAI3/SAI0.549363315；原O/Q全部指标由相同NumPy前48步精确复现，原错误/两份R原始NetCDF保留。
- [ ] 官方同条件科学验收：R首日H仍−33.1266%、LE绝对差5.16639W/m²、GPP均值+192.851%/r0.967753；土质、深度、根系/胁迫/地下水/SP-CN等仍不同，完整Python年、spinup、多站多年/留出和发布仍待。

下方为此前工程检查点，不抵充这些科学待办。

11:29UTC当前记录（下方均为历史快照）：

- [x] [Qia182新凝结水端点数值修订](docs/CLM_QIA182_PHASE_EDGE.md)：EHPC全CPU910/相变25、原163/182整步与xuanze相变8完成；300原始CUDA32/JAX32字段差0，对NumPy叶温差0.000448 K、H/LE差0.0000396/0.000791 W/m²，原守恒及tol/80次不变。
- [ ] 当前510源规范1250：同源910CPU/8GPU完成，其余332已排队等旧506实际退出；旧509C仅等待器已核实后替换，原模型/严格首日/全年未停。新EHPC PBS225从原冷种子试算，旧failed182不续跑。
- [ ] [C3作物全部文件独立光响应](docs/CLM_LEAFWEB_CROP.md)：63文件全登记，14文件/162点可评价，3达标、11未达标、49未测，原草地失败保留。不是P2单叶整体通过。
- [ ] Python完整气候年、spinup、多站多年/留出与同条件官方科学验收、合并main和v1.0.0发布。

10:30UTC补充：[新509C CUDA自身48及原后处理失败](docs/v1-results/qia-physical-cuda-day-c/manifest.json)
已归档，物理模型running/48，父进程1只因严格影子/主状态字段集断言；只读
核验26主字段及此前CUDA48的22轨迹字段完全相同、实际0。模型源码/输入未变，
[D续自身有效前缀](docs/v1-results/qia-physical-cuda-year-launch-d/manifest.json)实际96/17520。
不从原failed163/1519或变更源码的失败物理年续跑；完整GPU340、年/spinup/科学验收仍待。

10:21UTC终态补充（下方10:05及更早段落保留为历史）：

- [x] 数值修订工程验证：[最终509C全CPU909/MOST14/相变24](docs/v1-results/most-warm-cpu-c/manifest.json)在EHPC PBS209实际0，完整CPU470.937863秒；[xuanze原1519整步](docs/v1-results/most-physical-gpu-c/manifest.json)最终CUDA32/JAX32实际0，361原始字段最大差0，原MHCN界不变。原206/207/208失败全部保留。
- [ ] [当前规范1249完整回归](docs/v1-results/most-full-deferred-c/manifest.json)：909CPU同源已完成，全部340GPU已排队等前一1241实际退出，不抵充旧源/1519探针。前一1241为1064覆盖。
- [ ] 新原冷初态气候试算：[EHPC PBS210 NumPy](docs/v1-results/qia-physical-numpy-launch-c/manifest.json)先到336、原17520请求保留；[xuanze CUDA-only](docs/v1-results/qia-physical-cuda-launch-c/manifest.json)先验自身48再续年。归档均为running/0快照，非完成周/年/科学验收；原failed163/1519和严格配对首日未改。

10:05UTC更新（下方为此前快照）：

- [x] P6 [官方O全年直接强迫核对](docs/CLM_CTSM_FORCING_AUDIT.md)：PBS200模型/PBS0、17520正区间，气象含比湿及四短波全部符合历史精度；原N的77项湿度差异保留，冷初态/土质/SP-CN和科学验收仍待。
- [ ] P2/P6 [NumPy163因果初值候选](docs/CLM_QIA163_PHASE_FAILURE.md)：正式代码已重放原整步，水/热/C/N原界满足，独立方程温差8.75389e-12K；相变24目标通过。905全CPU有旧近零LE快照门1失败，实际差1.88425e-8W/m²；按既定绝对门修订，当前909另验。
- [ ] P6 [MOST物理一致性候选](docs/CLM_QIA1519_MOST_PHYSICS.md)：原CUDA年failed/1519保留；只在原50次物理稳定后另验完整反馈目标，硬守恒和原始1e-6不变。CPU14/相变24目标通过，当前完整CPU909及设备重放待终态。
- [ ] P6 前一终止检查源完整1241：09:05后已实际启动GPU1，10:05覆盖1061/1241；未计当前509源候选，原严格配对首日保持独立运行。

- [x] P6 [Qia CUDA-only首48与NumPy工程对照](docs/CLM_QIA_FIRST_DAY.md)：实际299.065秒，H/LE日偏差0.000101%/0.000783%，原MHCN界满足；26冷状态/账目数组与严格A相同，原A继续且门不变。CUDA全年384/17520仍运行，不是科学验收。
- [x] P6 [官方CTSM首日实际执行/输出读取](docs/CLM_QIA_FIRST_DAY.md)：EHPC PBS178模型/PBS0、49样本含起始区间；PBS179/180提取原48区间。原I/J失败保留，上游2930源字节不变。
- [x] P6 [官方SP全年执行/边界诊断](docs/CLM_CTSM_FORCING_AUDIT.md)：实际17520区间、模型0，原后处理使父PBS1；原输出重读0，全部错误保留，未改上游源。四短波分量符合历史精度，散射比例最大差2.47267e-8。
- [ ] P6 官方同条件比较：首日时序/四短波分量已对齐；全年比湿77项仍不同，最大6.91925e-5kg/kg。土质/冷初态/SP-CN未匹配，冷首日H偏差−12.4116%、LE近零绝对差10.7473W/m²，NEE未测，不以执行0充当10%验收。
- [ ] P2/P6 [Qia163相变](docs/CLM_QIA163_PHASE_FAILURE.md)：原NumPy周试算failed/163；末次更新误报已补最终检查，两项回归/完整CPU901/GPU8通过，同精度最大差0。整柱后续真未收敛仍待修复，30分钟后记录绕行；生产容差/80次预算不变，不将原失败年续跑或计完成周。
- [ ] P6 [当前1241完整回归](docs/v1-results/phase-full-deferred-a/manifest.json)：同源CPU901/GPU8已完成，余332排队；旧1239仅等待器已替换，原旧493/年度/严格首日进程保留，旧结果不抵充当前。
- [ ] P6 前一506B规范1239回归（历史）：CPU899与GPU22原证据保留；仅等待器已由上条当前1241队列替换，未执行的318不计通过，旧493结果不抵充当前。

- [x] P6 [19数组spinup/HRLDAS评价入口](docs/CLM_SPINUP_ASSESSMENT.md)：EHPC A版全CPU899/B版目标69实际通过，真实设备四步检查点读核与短年拒绝完成；汇总仅表示水热/SAI端点，不是实际spinup或碳氮平衡。
- [ ] P2/P6 [Qia SAI站点新冷启动](docs/CTSM_PFT_GEOMETRY_INPUTS.md)：显式气候态SAI0.549363、站点冠高11m/形态下沿假设5.5m，全部原水热/C/N/LAI/测高保留；EHPC PBS163首48步完成，累计热残差最大0.000498821J/m²。xuanze CUDA32／JAX32首日配对运行中；完整年、spinup及观测对照未完成。

- [x] P2 SAI站点入口和检查点保留：[实际四步记录](docs/CLM_STEM_STATION.md)，NumPy全CPU879项，原生主/影子19数组、连续/分段48数组逐位相同，同精度最大差7.28e-12；合成验证不抵充真实站点/全年。
- [x] P2 [SAI规范22项设备矩阵](docs/CLM_STEM_STATION.md)：506E版其余19项与仅修正float64测试输入的F版3项覆盖完整规范22项，实际工作进程均退出0；E原输入失败保留，不称单次E22通过。同精度最大差7.28e-12，不代表全部GPU340项或科学验收。
- [ ] P2 SAI真实站点几何采用及完整年：Qia显式输入首日NumPy完成，GPU配对运行中，全部站点的完整年、spinup及观测仍待完成。

- [ ] P2 [完整柱茎面积SAI接入](docs/CLM_STEM_AREA_GAP.md)：参考最大偏差1.1300%；后续EHPC全CPU864项完成，xuanze原生组件4项/800矩阵最大偏差0.0001404%、同精度差0。CUDA完整柱28柱步H/LE绝对差0.005369/0.002882W/m²；原生规范22项目标随后完成，但真实站点全年/观测仍未完成，不能勾选整体科学验收。
- [ ] P6 [K78同精度热量差异](docs/CLM_K78_PRECISION.md)：原年度failed/78；原态两端独立守恒通过，但329字段中首雪层残差差1.52588e-5 J/m²未过1e-6，已保留并按30分钟规则绕行。
- [ ] P6 [L6198 MOST系数未收敛](docs/CLM_L6198_MOST_FAILURE.md)：原独立CUDA年度failed/6198；有界50次原态重放复现，Qia末轮系数残差0.000244273338，原上限/守恒/收敛门未放宽。全年、spinup和观测未完成。

最新493源仍在验证：[雪消失事件与冠层差分修订](docs/CLM_NUMPY_MATERIAL_EVENTS.md)
已保留初版492源CPU808中1失败及设备集成失败；EHPC PBS142四个原失败步和旧
相变用例完成后，PBS144完整CPU808与原GPU64参考矩阵25例均完成；xuanze设备
12项目标已完成。PBS145另从原冷初态完成四站48步独立对照，日H/LE最大偏差
0.000700181%/0.000936390%；13:02UTC完整回归967/1135节点，未恢复失败年。
下方489源1129项是上一检查点的完整回归，不代表493源或科学验收已完成。

上一检查点完整1129清单使用[独立工作进程](docs/PYTEST_WORKER_LIFECYCLE.md)验证489源：
EHPC PBS126实际CPU804项与xuanze GPU325项全过，零失败/错误/跳过，规范节点
并集恰为1129；旧486源结果没有抵充新源。完整年和科学验收独立待办。

- [ ] P6 [官方CTSM单柱参考](docs/CTSM_REFERENCE_PREPARATION.md)：固定2930原源及111包环境，后续EHPC PBS178首日实际模型/PBS0，49样本及原48正区间已读回；所有中间失败保留。强迫时序/辐射、初态/土壤/SP-CN及同条件科学偏差待完成，此处不开发整机耦合或河道。
- [x] P6 [J5498雪—土壤停滞修订及完整回归](docs/CLM_J_PHASE_FAILURE.md)：局部割线重估同步JAX/CUDA；原输入12轮、独立热残差−1.06854J/m²在原界，对NumPy温差7.75942e-6K、同精度全字段差0，整步重放及新489源1129节点完整回归完成。K首日独立NumPy/影子验证完成后从自身48步续全年；全年和科学验收仍未完成。
- [ ] P6 [独立NumPy完整年四个失败](docs/CLM_NUMPY_ANNUAL_FAILURES.md)：原PBS121 failed/2657、479、4037、6456保持原样；原态全复现，PBS142候选四步与旧相变用例完成，新493源完整CPU/设备目标12及冷首日已完成，完整设备回归仍进行；不修改失败记录或放宽守恒，完整NumPy年仍未完成。

2026-10-17连续验证修订：[薄雪分支括区](docs/CLM_SNOW_BRANCH.md)和
[联合热量局部舍入](docs/CLM_HEAT_COORDINATE.md)已进入三后端开发路径，当前
另已修订[精确相变端点前进](docs/CLM_HEAT_PHASE_EVENT.md)，EHPC当前完整CPU部分
688项通过；xuanze当前热量10项通过，完整GPU回归及四站气候年仍未完成。
这些修订不勾选观测、气候spinup或发布待办。

- [x] P6 [气候循环收敛核对入口](docs/CLM_SPINUP_ASSESSMENT.md)：EHPC79项目标完成，平/闰年、原身份/冷种子/16数组和累计账目衔接、两次水热端点、真实/折算百年及七库总量/SOC三库区分覆盖；原J48拒绝。当前CPU804逐节点分布式覆盖、设备325原节点/代码一致；真实spinup及科学验收未完成。

- [ ] P6 [慢碳库初始化研究](docs/CLM_SPINUP_ACCELERATION.md)：EHPC两组合成季节条件对连续独立参考最大偏差0.130848%/0.136814%；缺氮负对照不适用。只完成条件方法核对，真实植物/氮反馈、站点采用及正常模式spinup待完成。

当前[稳定Qia三项及只读等待核对](docs/CLM_GPU_READONLY_PROFILE.md)完成，同精度
差0；定向两文件六项保留，余下三文件由原完整GPU325队列执行，GPU0年度/
GPU1回归继续。下方“其余定向仍运行”为此前并行调度，未将中断文件算通过。

- [ ] P6 [I4016 MOST振荡](docs/CLM_MOST_OSCILLATION.md)：原年度失败已保留，NumPy目标21项/完整CPU780项及实态设备核心3项完成，CUDA32/JAX32全字段最大差7.27596e-12。J首日对独立NumPy日H/LE最大偏差0.000700186%/0.000936399%，所有轨迹/状态与原I48逐字节一致，从自身48步续全年。其余定向及当前完整325项设备回归仍运行，原50次与守恒/同精度门不变。
- [x] P4 [固定日呼吸分量诊断](docs/CLM_RESPIRATION_DIAGNOSTIC.md)：xuanze原48步、全部16轨迹字段/23状态账数组逐字节一致；Qia叶/茎/根维持呼吸0.894/2.057/0.869 gC/m²/day，未据此拟合或宣称NEE通过。

- [ ] P6发现的[土质基质势冻土缺口](docs/CLM_FREEZING_REFERENCE.md)：新NumPy8080点最大偏差0.953774%；开发焓/输送参数链已接入，EHPC完整CPU部分708项及设备核心26项完成，同精度最大差0、联合热量最大温差0.000588K；完整柱GPU回归和连续科学核验待完成。
- [ ] P6 [G614冠层分支导数恢复](docs/CLM_CANOPY_PHASE_JACOBIAN.md)：CPU当前710项、设备9项及原四柱CUDA/JAX整步完成，实际300字段差0、整步输出差3.63798e-12/下一状态差0，独立热账最大2.09280J/m²；当前I从已验证首日续全年，年度/观测未验收，不勾选发布。
- [ ] P6 [冠层回溯提前退出](docs/CLM_CANOPY_PHASE_EARLY_EXIT.md)：当前设备9项与原四柱CUDA/JAX整步完成，全部输出/下一状态相对改动前逐位一致；独立核配对耗时比1.50/1.38。当前收集1032项，CPU710原EHPC记录字节/节点一致，设备322项已完整启动；I首日18轨迹字段/23检查点数组与H48逐位一致后，从自身48步续全年，未作为科学验收。
- [x] P6 [完整年/月数值评价入口](docs/HRLDAS_NATIVE_YEAR.md)：EHPC53项目标完成，明确日历、积分单位、原来源、原守恒界及最终检查点存量核对；实际I336年度请求拒绝。此项只完成评价工具，完整年、spinup及科学验收仍未完成。
- [ ] P6 [首个完整1月科学诊断](docs/CLM_FIRST_MONTH_RESULTS.md)：EHPC94项目标完成，三站官方均未整体达标，Qia H/LE月偏差−62.3938%/+82.9974%；FLUXNET有相关、均值及QC缺口。原I年度随后failed/4016、spinup和多站多年/留出均未完成。
- [x] P2/P5 [公开驱动接入完整CLM/冰川状态](docs/CLM_PUBLIC_DRIVER.md)：逐步物候/沉降/管理输入、全部库状态及批量独立性接通，EHPC17项目标与实际C/N示例完成；完整当前CPU759项、330.03秒全过。172份其余运行库与I一致，观测/年度及发布条件未勾选。

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
- [ ] P2冠层冰雪与叶温/整柱能量连接、完整稳定度连续科学核验和叶片/冠层实测达标；公开驱动完整状态连接见上方已完成项。
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
- [ ] P4完整反馈连续验证、真实强迫spinup及GPP/NEE观测验收；BGC冻结E全量700 passed / 0 skipped；七天4032柱步已完成、全轨迹同精度差0，日GPP/NEE后端偏差0.000303%/0.000908%，后续作物/灌溉不在该冻结源内。
- [ ] P5作物：三后端组件和整柱连接完成，8类柱两步同精度差0；七天四后端4032柱步已完成，90天NumPy12960柱步已完成，作物观测未验收；[简化与结果](docs/CLM_CROP.md)。
- [ ] P5灌溉：三后端组件和完整柱已接通，设备B目标27项与联合管理七天4032柱步完成；管理观测待验收；[数字及限制](docs/CLM_IRRIGATION.md)。
- [x] P5湿地甲烷三后端独立组件：512工况、8柱×365日两条设备路径完成，全部字段同精度门通过；[碳账与简化](docs/CLM_WETLAND.md)。
- [ ] P5湿地整柱水热/BGC连接及CH₄观测验收：NumPy及相关32项、原生整柱24项完成；联合七天4032柱步完成、同精度差0；最终全量和观测待验收。
- [ ] P5 [简化瓦片排水](docs/CLM_TILE_DRAINAGE.md)：NumPy组件5项完成，真实土壤扣水携热已实现；原生组件10项及4柱×480步两条设备路径完成；NumPy完整柱及相关27项、原生完整柱14项完成，同精度差0；长期完整柱及观测待验收。
- [ ] P5 [有限冰川柱](docs/CLM_GLACIER.md)：NumPy完整柱26项目标及1460步年度守恒完成；原生完整气象时间步F目标25项完成；实际第355步求根停滞已修复，四后端365天1460步及月积分完成，同精度全轨迹差0；官方/观测验收待做。
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
- [ ] 完整官方HRLDAS站点输出对标（与阶段1未完成项关联）；[三站各17,520步官方输出](docs/HRLDAS_CLIMATE_PROTOCOL.md)已在EHPC完成；CLM首周科学诊断三站均未整体达标，原CLM全年/最终配对验收待完成。
- [ ] ERA5格点驱动对照：本轮可达数据缺下行辐射和总降水，保持未测

具体站点、指标和校准设计见[FLUXNET_PLAN](docs/FLUXNET_PLAN.md)。四站短期
后端数值吻合不等于观测物理验证；完整CLM核心、11个明确未实现物理选项等
开发范围也不能因GPU移植验收而标为完成。

## 不做的事

- 不移植整机 CESM/WRF，只做单柱物理
- 不做河道演算、地下水侧向流（那是另一个并行模式）
- 新增物理先建立NumPy参考与可核验用例，再同步对应GPU实现和精度检查
