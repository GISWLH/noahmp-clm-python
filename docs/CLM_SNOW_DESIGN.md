# 五层雪开发前核查

状态：NumPy五层质量/焓状态、传热及五波段辐射组件初版；尚未接入整柱，
尚未同步JAX/CUDA，未完成冬季/官方输出验收。范围按 Wang 的五层上限执行。

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

尚待完成：粒径老化、压实、沉降时间/空间映射、冬季表面交换/雪土联立、
三后端状态扩展与默认驱动切换、真实外部输出验证。当前液水处理为串联
蓄泄桶，密实冰孔隙溢水继续向下传递，尚未实现接收层阻水导致的积水与
径流分流。BC目前合并为亲水类别，液水用周围冰光学性质近似，不能模拟
积水光学；这些均不是已经通过CLM完整雪水文验收的声明。
