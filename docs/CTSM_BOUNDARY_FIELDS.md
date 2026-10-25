# CTSM R/S 原边界字段只读回溯（OA）

[完整三次作业、原读回数据与控制证据](v1-results/ctsm-boundary-fields-oa/manifest.json)。

1587、1588 均真实 F/Exit1，原因是 `netCDF4` 导入失败，发生在读取 NetCDF
前；错误日志原样保留。1589 使用固定 portable 环境后真实 F/Exit0，实际读回
3.552007 秒，核对 R/S 共 12 个原文件，前向模型运行次数为 0。原 reader SHA
`0fb6bc718615fe0882754af2b3e4e9097013f893abbb8bb9784aae9c310a7cfb`
未改；环境 SHA 为
`6c535c3391287f26e49ec1089bc71c255651ab0de97983c1dc5c2b6a1e978a20`。

manager 与 compute002 三次 `/tmp` stage 全部原文件、dispatch、qsub 记录、
PBS 脚本/出口、result 包和其中日志/结果均保留。配置的 `pbs-output.log` 在两端
均未出现，qstat 的 Stageout_status 为 0；不补造空文件。实际 stdout/stderr 已
由原 PBS 控制重定向到 result 包。1589 首次 E/Exit0 快照也保留，但仅后续 F
快照用于闭合。

以下数字直接摘自 EHPC 读回 JSON，没有在 Windows 重算物理、求均值、积分或
改变时间索引。主结果 SHA 为
`a4ebe215ebf1ea1b224be0b6357709df6ec1444b64be452f11d1444901a3542f`。
全文件变量目录、原单位、维度、掩码、已读最值与逐层数值见该原 JSON。

| 字段/对象 | 实际读回与边界 |
|---|---|
| WATSAT | R 首个 history 的 15 个 `levgrnd`，`mm3/mm3`；原值范围 0.4234800041–0.6111107469。与当前 nominal loam 的常数孔隙率不能视为同条件。 |
| SUCSAT | 同 15 层，原单位 `mm`，正值 90.99365234–186.2010040；保留原符号与单位，不把它直接当成 Python 单位为 m 的有符号势。 |
| BSW | 同 15 层，`unitless`，4.958487034–8.343965530。 |
| HKSAT | 同 15 层，原单位 `mm s-1`，0.004498150200–0.01049637143；不把它与 m/s 的输入直接作数值比较。 |
| 深度 | ZSOI、DZSOI 都实际存在，单位 m，维度 `(levgrnd=15,lndgrid=1)`。ZSOI 中心首/末值为 0.007100635208/35.177619934；DZSOI 首/末值为 0.017512818798/13.851152420。完整 15 层原值在 JSON，不能用末层中心冒充总深度。既有 WP 记录的 15 热层/10 水层与 Python 4 层仍不等价。 |
| 实际根分数 | 所选 12 个原文件的完整变量目录中均无 ROOTFR、ROOTR、ROOTFR_UNF；写“这些原件中缺字段”。参数表有 `roota_par`、`rootb_par`（1/m）和 `rootprof_beta`（unitless，C/N 垂直离散参数）；它们不是实际吸水根分数或 beta 权重。未重新计算根分布。 |
| 冷输入地下水 | R/S `CN-Qia.official-interpolated-cold.nc` 的 WA=4000 mm，ZWT=8.80188191232272 m，ZWT_PERCH=3.8018819123227208 m。 |
| 推进后的地下水 | S 第一个实际 restart `2002-12-31-59400` 已为 WA=5000 mm，ZWT=3.8018819123227208 m，ZWT_PERCH=2.8646071131796917 m；S 末 restart 和 R 年末亦有这三个字段。不能把冷文件的指定值等同于运行后的内部状态。 |
| 初始土壤水热 | 冷文件 T_SOISNO/H2OSOI_LIQ/H2OSOI_ICE 维度 `(column=1,levtot=20)`，单位分别 K、kg/m2、kg/m2；前 5 雪槽温度为 0，后 15 土槽温度 278.722 K。液水前 5 雪槽和末 5 深土槽为 0，10 水层库存原值保留；全部冰槽为 0。这不意味着 20 个同等活跃水层，也不证明与四层柱总库存相同。 |
| 出口 | history 有 QOVER、QDRAI、QRGWL、QRUNOFF，均 mm/s、time: mean；QRGWL 的原说明包含冰川/湿地/湖泊及 QSNWCPICE 融冰，QRUNOFF 是不含土地利用变化修正的液态总径流。这里只记录定义与可用性，不重新归并收支。 |
| RAIN/SNOW | history 实际存在，mm/s，time: mean；原 long_name 明确是按温度重分相后的大气雨/雪。R 主 history 17520 存储行，RAIN/SNOW 原最大值分别 0.009611111134/0.000504666648 mm/s；R 末 history 另 1 行，S history 49 行。不能把存储行数直接称为对应正积分区间数；未做新逐步相态配对或换算累计量。 |
| 沉降通量 | R/S history 的 BCDEP、DSTDEP、OCDEP **实际存在**，均 `kg/m^2/s`、time: mean，long_name 为大气干+湿总沉降。原 reader 未将这三个名字纳入选值集合，因此本包只有这些字段的元数据，数值读回/时间对齐仍未测，不能写成字段不存在。 |
| 雪杂质库存 | 冷输入与所选 restart 有 mss_bcpho/mss_bcphi、mss_ocpho/mss_ocphi、mss_dst1–4，单位 kg m-2、五个 levsno 槽，已读原值；冷输入均 0。库存不是沉降速率，也不能替代逐步 land 接收通量核对。 |

R 只选了唯一年末 restart；S 的 48 个 restart 中仅选首、末两个。这里不是完整
时间审计，不把 S 首/末相同推广为全部中间状态相同。ROOTFR 的缺失限定于这
12 个实际读取文件，不外推为所有 CTSM 文件都没有该字段。

后续[垂向水文复核](CLM_VERTICAL_HYDROLOGY.md)确认 R 实际关闭 aquifer，
全年 QCHARGE 为0。无 aquifer 路径的 WA=5000 mm 是基线，不能把上表冷文件
至 restart 的变化直接解释为1000 mm物理补给。R 全年仍有125.3991 mm饱和区
排水，不能从其首日排水为0推断全年封闭。

这些结果补充[参考边界台账](CTSM_REFERENCE_BOUNDARIES.md)：土质、深度、
地下水初值/运行状态已有直接字段证据；实际根分数仍未测。后续 EHPC1591
已完成[总 BC／尘／OC 陆面通量数值读回](CTSM_LAND_DEPOSITION.md)，年度
BC／尘偏差约 −2.93e-9%／+1.21e-8%，瞬时 r 均超过 .99999995；四尘档
实际接收仍未测。本表沉降行描述的是 OA 原 reader 范围，不能覆盖后续 OB。
SP 对 CN、SAI/PFT/容量与水热底界差异依然保留，不更改生产参数、观测 QC、
原参考、科学门，也不新增同条件或科学通过信用。
