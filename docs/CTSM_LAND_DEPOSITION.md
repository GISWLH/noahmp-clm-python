# CTSM R 实际陆面沉降读回（OB）

[原作业终态、完整数组与控制证据](v1-results/ctsm-land-deposition-ob/manifest.json)。

EHPC `1591.manager` 实际 `F / Exit_status=0`，worker 墙钟17秒，固定 portable
环境 SHA `6c535c3391287f26e49ec1089bc71c255651ab0de97983c1dc5c2b6a1e978a20`。
读回耗时3.519869528秒，模型前向执行0次。全部新文件在 manager/compute002
各自 `/tmp` 与 PBS `TMPDIR`；生产、原输入/QC、活动CPU/GPU年度均未修改。

已读既有 AERO 输入审计与 OA1589，未重复原气象/driver值验收。这里直接从原R
私有归档 `14891786b9c492d4f34f3077476181333f6e0ccb6c3209663ad0a5194f77b1b0`
中的两份 `.clm2.h0` 文件读取陆面 `BCDEP/DSTDEP/OCDEP`。原文件分别17520/1
存储行；沿原CF边界只排启动 `[-1800,0]`，其余17520个正半小时时段逐项等于
原固定当地2003年的区间。没有重新标注、移位、筛选较好窗口或用driver量替代land。

三字段单位均为 `kg/m^2/s`、`time: mean`、float32、维度 `(time,lndgrid=1)`。
原填充值/缺测属性完整保留；全部正区间 masked和nonfinite数量均0。结果NPZ
保留全部17521行、原掩码/非有限掩码、来源文件索引、原/整数时间边界、正区间
选择器和原14个气溶胶月中心锚点、14分量输入及5物种插值。没有补零缺测值。

## 实际偏差

以下数值直接摘自EHPC结果JSON，本机未重算积分、均值、相关或物理数组。
对照量是原R输入 `CN-Qia.aero.2002-2004.nc` 的原14月中心线性插值，在相同
原区间中点求值；旧输入协议已独立核对过它与原Python五物种沉降方案的关系。

| 陆面量 | 最大绝对速率差 kg/m²/s | 年陆面量 kg/m² | 年原输入量 kg/m² | 年差 kg/m² | 年偏差 % | 瞬时r |
|---|---:|---:|---:|---:|---:|---:|
| BCDEP | 1.6041647607426671e-15 | 0.00014581592008493065 | 0.00014581592008920894 | -4.278303345794809e-15 | -2.9340440626629655e-9 | 0.9999999526936973 |
| DSTDEP | 4.652293203471002e-14 | 0.003921686247393419 | 0.003921686246917253 | 4.761655307845938e-13 | 1.2141856864732525e-8 | 0.9999999619006285 |
| OCDEP | 0 | 0 | 0 | 0 | null | null |

**BCDEP与DSTDEP各17520项都不等于原中点输入的float32表示**，不能称逐步强迫
完全相等，也不能把全部差异归为写出量化。年度积分接近和r很高并不识别其
计算时刻；本作业没有进行移位拟合。BC最大差在原正区间索引3057（从0开始），
陆面/输入分别7.345471453312769e-12、7.347075618073511e-12 kg/m²/s；
DST最大差在14903，分别1.6144596770573116e-10、1.6149249063776587e-10。
实际时段插值/接收时刻的逐值一致未证明。年偏差和瞬时相关已远在 Wang 的
10%／r>0.85 口径内；此微小差异予以接受，不增加逐元素门或发布阻塞。

OC所有17520项均严格0，最大绝差0、历史精度不同项0；零参考不算相对偏差或r。
这是原共同五物种试验显式OC=0的读回，不能推广为真实大气OC沉降已验证。

## 映射与未测项

- BCDEP对应原 `BCDEPWET + BCPHODRY + BCPHIDRY` 总黑碳；未将OC合并进BC。
- DSTDEP对应原四个尘档各自湿/干共8分量的总和。两份原history只有总尘字段，
  **各尘档实际陆面接收仍未测**；不能用输入四档或雪内库存冒充实际陆面通量。
- 原五物种输入年积分依序BC、dust1–4为
  `[0.00014581592008920834, 0.00022755789982760585, 0.0006750293855386729,
  0.0009163064650068633, 0.00210279249654411]` kg/m²。这只是输入协议的数值，
  不给四档陆面接收新增通过信用；BC求和次序的末位差保留原JSON。
- 此次没有观测比较、额外10%科学门、参数校准、模型重跑、全年Python验收、
  SP/CN等价或新的rootfr数值。

## 根字段为何不能用参数替代

固定官方CTSM提交仍为 `fdfc03a0abec838ec0b66663a6a2f5e619cd2611`。只读源码语义，
未复制Fortran实现或计算任何根分布；每份源码SHA及具体行段见结果JSON的
`root_semantic_index`。

1. [SoilStateType.F90](https://github.com/ESCOMP/CTSM/blob/fdfc03a0abec838ec0b66663a6a2f5e619cd2611/src/biogeophys/SoilStateType.F90#L234)
   234–259行将ROOTFR history注册限制在 `use_dynroot`，ROOTR限制在 `use_cn`
   且默认inactive；391–412行的rootfr restart读取同样有dynroot条件，缺失时调用
   分开的water/carbon默认初始化。原R实际 `lnd_in` 31/33/38行分别是
   `use_cn/use_dynroot/use_hydrstress=.false.`，123–124行water/carbon方法均1。
   这与当前原件未输出实际根分数一致，不等于柱内没有根过程。
2. [RootBiophysMod.F90](https://github.com/ESCOMP/CTSM/blob/fdfc03a0abec838ec0b66663a6a2f5e619cd2611/src/biogeophys/RootBiophysMod.F90#L110)
   110–172行区分water/carbon方法/变体及床岩限制；175–219、231–272行分别涉及
   roota/rootb与rootprof_beta参数的不同可选分布。实际分层还依赖几何、PFT和
   方法选择；单独的参数表值不能视作已读回ROOTFR。
3. [SoilMoistStressMod.F90](https://github.com/ESCOMP/CTSM/blob/fdfc03a0abec838ec0b66663a6a2f5e619cd2611/src/biogeophys/SoilMoistStressMod.F90#L223)
   223–310行ROOTFR_UNF是未冻根归一化内部数组；363–438行ROOTR进一步依赖
   实际水、温和阻力并归一化。因此静态根参数不能替代实际吸水权重。

不为上述微小差异继续定位时序或重跑模型。四档陆面通量、
ROOTFR/ROOTR/ROOTFR_UNF没有原输出支撑时
继续记未测；不能从参数或驱动反演后宣称实际读回。

## 原证据

`ctsm-land-deposition-ob-evidence.tar.gz`：5,383,430字节、25项，SHA
`84a703d27ef8fa782b4357d48ef134ea916b67b3167108c4fc4945f5e6f38637`。
包含真实F/Exit0、原PBS出口、控制/reader/wrapper字节、stdout/stderr、元数据和
完整结果数组；下载后25项SHA与外包SHA逐项复核。数组SHA
`8aef15a572bea7e7d17cb299962bde4dc1bb58d11a0dd913524cb636d91a96c5`。
原OA的1587/1588导入失败与1589成功仍在原包中，本包不改写或省略它们的历史。
