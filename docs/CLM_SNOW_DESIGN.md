# 五层雪开发前核查

本文保留五层雪的初始设计与分阶段证据。当前已接入
[NumPy冠层—雪—土壤质量/焓时间步](CLM_EXTENSIVE_COLUMN.md)，
[联合热量](CLM_GPU_ENTHALPY.md)及[雪辐射/冠层短波](CLM_GPU_SNOW_RADIATION.md)
已同步原生JAX/CUDA；[质量/焓/杂质输送](CLM_GPU_SNOW_MASS.md)也已同步。
[雪表层水汽、霜露和微结构演化](CLM_GPU_SNOW_PROCESSES.md)现也已同步。
完整冬季柱GPU、官方/观测验收仍待完成。
下文“尚未连接/同步”描述对应历史检查点。范围按 Wang 的五层上限执行。

CTSM 官方雪水文技术说明同时保存液水、冰、厚度和温度，分层/并层需保存质量
与焓。当前在线 `release-clm5.0` 页面描述的可选上限已是十二层，因此这里的
五层是本项目已确认的范围，不把网页的其他上限扩进本阶段。
来源：[Snow Hydrology §2.8.3、2.8.7](https://escomp.github.io/CTSM/release-clm5.0/tech_note/Snow_Hydrology/CLM50_Tech_Note_Snow_Hydrology.html)。

现有 `physics/snow.py` 按 `C_SNOW = 300 × 2100` 乘实际厚度计算储热；
在密度随压实变化时，这不等于雪冰质量乘比热。新雪热模块需使用实际冰/液水
质量，保持重分层前后的总焓；雪下导热也必须与土壤共用同一个接口通量。
依据：[Soil and Snow Temperatures，式2.6.90](https://escomp.github.io/CTSM/release-clm5.0/tech_note/Soil_Snow_Temperatures/CLM50_Tech_Note_Soil_Snow_Temperatures.html)。

SNICAR 简化版需要粒径、杂质浓度、下垫面反照率和直/散射条件驱动的分层
双流吸收，不可只把现有雪龄经验曲线重新命名。按任务要求采用黑碳/尘埃
气候态输入，不展开气溶胶传输。依据：
[Surface Albedos §2.3.2](https://escomp.github.io/CTSM/release-clm5.0/tech_note/Surface_Albedos/CLM50_Tech_Note_Surface_Albedos.html)。

先做NumPy质量/焓状态与独立导热、融化、再冻结、流出验证，再接入冠层地面
边界与JAX/CUDA。P1已修复的融尽余热必须继续路由到土壤。表面收支闭合不能
替代这一核算，也不能因为后端吻合就宣称冬季物理验收完成。

## 已取得的官方参考数据

来源为 [UCAR 输入数据目录](https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/lnd/clm2/snicardata/)，
2026-10-13下载到本地 `outputs/v1/snow-reference/` 和xuanze
`/home/lhwang/work/noahmp-v1/snow-reference/`，未作为物理代码复制或修改。

| 文件 | 核查内容 | SHA256 |
|---|---|---|
| `snicar_optics_5bnd_c090915.nc` | 5波段×1471粒径；冰的直/散射光学量及BC/OC/四类尘埃 | `a17f1b42616ce0b5c0534ccc0f92c33f0020db574bcc2888d326c7da6b69c305` |
| `snicar_drdt_bst_fit_60_c070416.nc` | 温度11×梯度31×雪密度8；tau/kappa/drdsdt0老化表 | `75aa2dbb5e770cca38915ed2ed4f9c31e076857fd56c701ced2fef6bdb78e5e6` |
| `aerosoldep_monthly_2000_mean_1.9x2.5_c090421.nc` | 96纬×144经；BC、OC及四粒级尘埃干/湿沉降 | `2e920f2558685c790e904185bd3a9150435f31e3dca2d2ebb1aa1f6b60ae61ff` |

NetCDF头与变量范围已实际读取，`metadata.json`保存在同目录并归档为
`v1-results/p2-snow-reference-metadata.json.gz`。当前只完成数据准备，
在下面的NumPy组件检查之前，这一步仅是数据准备。

## NumPy组件开发检查点（2026-10-14）

新增`snowpack.py`保存最多五层质量、焓、厚度、粒径与BC/四类尘埃质量；
重分层采用体积重叠积分。粒径按质量加权的逆半径混合以保持光学比表面积
代理量，区别于技术说明中的算术平均。雪厚不足1 cm仍保留显式储热层。

`snow_heat.py`使用实际质量热容和共享界面通量，采用后向欧拉焓求解；
不是CTSM温度先行的Crank–Nicolson算法。底边界返回一个需要土壤实际接收
的通量。目前只检查雪组件，不能把这个接口误写成已经完成雪土联立。

`snicar.py`读取校验固定身份的UCAR光学表，使用五波段、delta缩放、可见光
Eddington及近红外半球平均闭合，通过原创建倍算法求各层通量。直/散射
分别查表；输入为可见/近红外直射与散射通量，以便后续连接冠层。
压缩查表文件SHA256：`43aaf77afdfec87990c64250418d13db89ebfc66c8e21e5a75023aaa70ced357`；
`benchmarks/prepare_snow_tables.py`可从上述原件重建，数据归属见包内NOTICE。

本地组件测试：雪状态与热29项、辐射8项，共37项。参考包含单层解析焓、
独立SciPy多层能量求解、SciPy矩阵指数辐射边值解和无限厚雪解析极限；
不是实际CTSM输出或雪站观测。具体诊断例：初始2 kg/m²冰、268.15 K，
输入1,800,000 J/m²后融尽，流出水携带1,779,000 J/m²，初始焓−21,000 J/m²，
未路由能量为0 J/m²。此为跨越全融化区间的强迫压力测试，不是气候工况。
100 kg/m²、54.5 µm洁净雪，800 W/m²入射时反照率0.8623489594，
反射689.8791676 W/m²、土壤吸收0.1466788 W/m²，收支残差1.237621e-13 W/m²。

随后补充的老化、压实、沉降映射及阻水路由见下方。BC目前合并为亲水
类别，液水用周围冰光学性质近似，不能模拟积水光学；这些均不是已经
通过CLM完整雪水文验收的声明。

## NumPy雪组件独立核验与年度收支

最终组件冻结快照：`/home/lhwang/work/noahmp-v1/p2-snow-numpy-b`；
源包SHA256 `245f8a809b7144c234203262f0054b2cb0e245dcfa15a2c7d4332e265a94c460`。
组件49项测试全部通过，3.69秒；数值核验23.93秒。
同一b快照的Linux全量测试 **317 passed / 0 skipped**，222.83秒，
[完整日志](v1-results/p2-snow-full-pytest.log.gz)及
[执行源码清单](v1-results/p2-snow-full-source-sha256.json.gz)已归档。
171个打包文件中170个与本地逐项一致；唯一变化为随后构建wheel重新生成的
`src/noahmp_clm.egg-info/SOURCES.txt`文件清单，所有物理、测试和基准源码均一致。
全量测试中的GPU检查覆盖既有模块，不能当作新雪模块GPU已实现的证据。
[原始数值报告](v1-results/p2-snow-numpy-audit.json)包含10个源码/数据SHA256，
已逐一核对本地源码。`benchmarks/check_snow_numpy.py`可重复运行。

| 核查 | 实际差异/残差 |
|---|---:|
| 200个含相变雪传热工况，对独立SciPy求解 | 最大焓差3.725290e-9 J/m²；温差5.684342e-14 K；液水差1.421085e-14 kg/m² |
| 同组热量预算，最多6次迭代 | 最大残差2.910383e-9 J/m² |
| 240个双流工况，对独立SciPy矩阵指数边值解 | 最大通量差8.606094e-11 W/m²；大于1 W/m²的量最大相对差2.172151e-10% |
| 同组辐射预算 | 最大残差4.405365e-13 W/m² |
| 四组干/湿老化，对独立自适应ODE积分 | 最大粒径偏差0.432998%，湿雪54.5 µm经半小时，参考70.947242 µm，本实现71.254442 µm |
| 雪组件365天、17520步 | 最大累计水差2.592060e-11 kg/m²；焓差2.235174e-7 J/m²；杂质差7.155734e-18 kg/m² |

年度案例达到五层，最大SWE 0.145976 m，雪温256.276452–273.15 K；
输入278.18 kg/m²，流出及升华191.546474 kg/m²，其中底排181.120474 kg/m²，
本年度没有触发阻水径流。阻水及接收层冻结膨胀分别有专门单测覆盖，不能
用年度径流为零代替这两项。收支包括降水、流出水和移除冰的平流焓；
上边界为指定净热量加隐式空气换热，底热量导向记录中的外部储库，
不是闭合的冠层—雪—土壤气候模拟。365份日诊断在同一JSON中。

首轮`p2-snow-numpy-a`采用不反馈雪温的指定热通量，能量闭合但温度边界
不宜作为季节物理证据，保留其[原始报告](v1-results/p2-snow-earlier-fixed-flux-audit.json.gz)。修正为隐式温差换热
后从头重跑为上面的b；没有把a的收支结果当作最终温度验证。

`snow_evolution.py`实现破坏性变质、上方载荷和融化塌缩压实，厚度不小于
冰/液水实际体积；冻结速率用指数积分。粒径老化用三线性查表及干/湿
增长分裂积分，重新冻结的冰按1000 µm混合。尚无漂移雪再分布。
老化表`tau`属性为hours、`drdsdt0`属性为m²/kg/hour，但其描述及
[官方SnowSnicarMod实际用法](https://raw.githubusercontent.com/ESCOMP/CTSM/master/src/biogeophys/SnowSnicarMod.F90)
将其用于微米长度尺度和微米/小时增长。这里只读核对单位，未复制Fortran；
保留原表及原属性记录，使用官方方程中的微米/小时解释。

液水向下转移同时满足接收层孔隙及再冻结体积限制；阻水时在上层保水，
超过孔隙容量部分连同焓与杂质作为局部径流输出。调用方必须使用
`return_ledger=True`分别消费底排和径流。完全融尽的层不保留虚构雪骨架。

`snow_forcing.SnowDepositionClimatology`读取官方月沉降文件，纬度/经度
双线性与月中点周期插值，处理跨经度零点和闰年；湿/干沉降汇总为BC和
四粒级尘埃。文件身份、实际读取的12×96×144×5数据及四个抽样坐标见
[沉降映射报告](v1-results/p2-snow-deposition-audit.json)。例如45°N、125°E，
一月中点BC通量1.765592e-12 kg/m²/s。该点只是映射检查，未称作站点观测。

包已构建并确认包含光学/老化表及NOTICE；wheel仍为0.2.0开发包，
489299字节，SHA256 `8b3b9a98ce326e7ffb6ec19887cecd6bea65a0ec219c1a24f3d8fb1d1c1169a2`。
后续已完成NumPy雪/土壤共享通量、实际平流焓路由及固定雪面冠层边界，
见[联合焓开发核验](CLM_COLUMN_ENTHALPY.md)。完整冬季时间步、三后端状态
扩展、默认驱动切换与真实外部输出验证仍待完成。
