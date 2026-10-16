# 土壤/凋落物碳氮分解（原生组件完成，整柱BGC待连接）

按[CLM5分解说明](https://escomp.github.io/CTSM/release-clm5.0/tech_note/Decomposition/CLM50_Tech_Note_Decomposition.html)
的CENTURY框架独立实现：三种凋落物及粗木库进入快、慢、被动三种SOM库。
表2.21.3的周转时间为0.066、0.25、0.25、4.1、0.17、6.1、270年；
SOM接收物质的C:N为8、11、11。温度因子Q10=1.5、参考25°C，水势限制
在−2.5至−0.002 MPa之间按对数变化。路径及砂土依赖的快库呼吸比例参考
[CTSM release-clm5.0结构](https://raw.githubusercontent.com/ESCOMP/CTSM/release-clm5.0/src/soilbiogeochem/SoilBiogeochemDecompCascadeBGCMod.F90)，
只读理解，没有复制Fortran实现。粗木纤维素比例0.76为本简化配置，可显式修改。

本实现为柱尺度聚合，气候条件作为显式输入，未做垂直C/N混合、完整FUN、
同位素或CLM原版Century温度归一化。表中时间按本模型25°C参考直接使用；
不能因此声称与CTSM站点输出已经吻合。氧因子由后续湿地/甲烷模块提供，
关闭时为1。冻结时用Clapeyron吸力限制水势。

`DecompositionState`携带7个有机C库、7个有机N库和矿质N库。植物凋落、
沉降、矿质N输入及植物吸收需求均为显式步长内质量，单位gC/gN每平方米。
供体采用指数存活率，接收的有机物留到下一步分解；矿化产生的N也在下一步
进入竞争。当前矿质N按植物需求和微生物固持需求的比例分配，缺N的分解
路径同步削减C/N转移；净矿化路径照常进行。每条路径都返回供/受C/N台账。
呼吸是实际损失的C，植物吸收是显式N出口，没有凭空补齐目标C:N。

NumPy先完成8项独立检查，再实现原生JAX/CUDA。48组环境因子与60位Decimal
对照最大绝对差2.22044605e-15。108组碳库对独立连续时间矩阵指数解：

| 步长 | 最大碳库偏差 | 最大呼吸偏差 |
|---|---:|---:|
| 1800秒 | 0.000133949% | 0.0109925% |
| 3600秒 | 0.000531530% | 0.0219827% |
| 86400秒 | 0.224481% | 0.525040% |

氮受限解析例中共同满足比例0.133333333333，C/N残差
6.08888e-15 gC/m² / −5.18249e-17 gN/m²。固定气候且氮充足时，由实际日步
构造线性转移算子并用矩阵幂推进5000年；后续百年最大相对漂移
3.60385e-9，距离散平衡1.17622e-8，离散/连续平衡最大差2.06119%。
这是分解库的理想化验证，**不是**完整气候/植被/C-N反馈spinup验收。

冻结 `p4-decomposition-b`：**13 passed / 0 skipped，49.16秒**。
512个量化状态原生对照、8柱四后端七天10752次推进；CUDA32/JAX32全部
输出和轨迹最大差0，保留1e-6同精度门。每步独立float32 C/N残差最大
0.00137821542 gC/m² / 8.01116767e-5 gN/m²，逐工况数值界最大
0.449285230 gC/m² / 0.0427265794 gN/m²。float64残差为
7.31050e-12 gC/m² / 2.53785e-13 gN/m²。
七天float32累计C/N残差0.189312057 gC/m² / 0.00437101118 gN/m²，
累计数值界97.0291324 gC/m² / 2.45401293 gN/m²；最终最大状态差
0.122074989 g/m²。float64累计C/N残差3.27170e-11 / 3.41061e-13。
这些是明确报告的有限精度残差，没有用科学10%门替代守恒审核。

A首次设备目标误用了2e-5相对科学门，在RH=0.539394 gC/m²处因实际偏差
1.49813e-5 gC/m²（0.00277743%）中止，8项NumPy已通过。按照Wang已确认
的P0口径修正为10%及分单位近零绝对界；未改物理、独立守恒界和同精度门。
A失败日志保留，不为该微小差异继续优化。

float32相对NumPy的最大状态绝对差0.000993339 g/m²，异养呼吸差
1.49608e-5 gC/m²/步（3.98710e-5 µmol/m²/s）；float64最大状态差
4.54747e-13 g/m²。原始证据：[NumPy](v1-results/p4-decomposition-numpy-a.log.gz)、
[A失败](v1-results/p4-decomposition-a-target.log.gz)、
[B完整目标](v1-results/p4-decomposition-b-target.log.gz)、
[结构化结果](v1-results/p4-decomposition-b-summary.json)。

分解与植物分配、真实凋落物、叶N容量及整柱NEE已在后续
[NumPy整柱连接](CLM_BGC_COLUMN.md)中接通；FLUXNET验收待完成。
冻结B全量现已结束：**662 passed / 0 skipped，2 warnings，4866.91秒**。
两个警告来自既有高密度低温雪黏度指数上溢；测试未失败。这是分解检查点的
全量结论，不包含后续植物C/N及整柱BGC代码。
[全量日志](v1-results/p4-decomposition-b-full.log.gz)和
[源码清单](v1-results/p4-decomposition-b-full-summary.json)已归档。
