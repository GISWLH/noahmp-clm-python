# 五层雪开发前核查

状态：设计依据与现有缺口核查，尚未实现。范围按 Wang 的五层上限执行。

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
没有运行SNICAR对照，也没有声明雪反照率误差通过。
