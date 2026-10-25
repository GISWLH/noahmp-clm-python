# CN-Cha 植被与冬季冠层：一手证据核查

本次只查原始站点文献、官方元数据及已冻结输入，没有运行新数值、修改模型或站点配置。
**证据足以确定长白山塔周围存在常绿红松与落叶阔叶混交，以及非零枝干结构；尚不足以唯一确定模型所需的 PFT 面积权重、冬季活叶 LAI、逐月 LAI/SAI 或分 PFT 的 C/N 库。**
现有 Cha 整个聚合叶库按落叶物候休眠的配置未显式表示常绿组分；这是待检验的配置简化，不等于已经证明冬季 H/LE 偏差的根因。

## 站点与原始量的对应

| 一手来源及精确位置 | 原文中的量与方法 | 对配置的约束与限制 |
|---|---|---|
| [FLUXNET CN-Cha 官方站点页](https://fluxnet.org/sites/siteinfo/CN-Cha)，Overview / Publication(s) | 坐标 42.4025°N、128.0958°E；IGBP=MF；FLUXNET2015 为 2003–2005；指定 Guan et al. (2006) 为站点描述论文。 | MF 是分类；其中通用的“各林型不超过 60%”定义不是测得的该塔 PFT 权重。 |
| [Guan et al. (2006), AFM 137:138–149](https://www.chinaflux.org/manage/eWebEditor/uploadfile/2006828163357793.pdf)，§2.1，印刷 pp.139–140（PDF pp.2–3） | 塔坐标 42°24′09″N、128°05′45″E，与 CN-Cha 一致。红松、紫椴、色木槭、水曲柳、蒙古栎等混交；最大树高 23–28 m，报告最大面积指数 5.8。 | 确认混交组成，不提供活叶面积或塔足迹的 PFT 覆盖率。Fig.1 是树干数量/胸径分布，不能换成叶面积比例。 |
| 同一 [Guan 论文](https://www.chinaflux.org/manage/eWebEditor/uploadfile/2006828163357793.pdf)，§2.2 p.141；Eq.(16) 后 p.143（PDF pp.4、6） | LAI-2000 沿塔西南 200 m 样线，观测时段为 4 月下旬至 10 月下旬。p.143 明写 L₀=1.5 是 “the estimation of L of stems”。 | 1.5 是该论文采用的枝干面积估计；不是已证实的冬季活叶 LAI，也不是冬季逐日直接观测。 |
| [Zhang et al. (2006), AFM 137:176–187](https://www.chinaflux.org/manage/eWebEditor/UploadFile/2006828122332756.pdf)，§2.3.4 p.179；Fig.3 p.180；§4.1 p.184（PDF pp.4、5、9） | 2003 年上下冠层 PAR、消光系数 k=0.55 反演冠层面积，Fig.3 与 LAI-2000 点比较；p.184 采用非生长季背景 LAIᵦ=1.5。 | 文中称它为 “background LAI”，没有在这一定义处给出常绿针叶与枝干的独立拆分。不能将此 1.5 与 Guan 的 1.5 分别作为活叶和枝干相加。 |
| [Shi et al. (2010), JGR 115, DOI 10.1029/2009JD012832](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009JD012832)，§3.1 ¶32；§3.4 ¶39 | 同一精确塔坐标。LAI-2000 在 1 m 高、塔西南 200 m 的 10 点测 PAI；报告 10–4 月枝干面积约 0.8，并以 LAI=PAI−0.8 得到叶面积。 | 0.8 是更明确区分 PAI/LAI 的站点枝干文献值，可作为独立诊断情景。方法没有证明本项目每个日期、整个足迹都恰好为 0.8。该研究的模型评价集中在 2003–2007 年 5–9 月。 |
| 同一 [Shi 论文](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009JD012832)，§2.3 ¶¶27–28、Table 1 | 红松叶生物量 3.13 t/ha，占五个优势树种叶生物量的 55.30%；该生物量调查转引 Li et al. (1981)。文中另有不同树种冠层上下沿调查。 | **55.30% 是叶生物量比例，不是 LAI、冠盖或 PFT 地表面积比例。** 本次未获取 Li (1981) 原调查全文，不能将转引升级为已核验的 2003 年塔足迹调查；也未取得匹配的分树种 SLA/面积换算约定。 |

上述 0.8 与 1.5 来自不同方法或资料时段，**不构成置信区间**。可预先声明为两个结构诊断情景，并保留原 SAI=0 对照；不能根据 H/LE 得分选一个作为“正确观测值”。冬季保持常绿针叶与冬季仍有正净光合也不是同一个命题，不能用冬季碳通量反求活叶比例。

## 已锁定输入实际来自哪里

- 本地 `docs/v1-results/fluxnet-2015-bif-three-sites.json.gz` 中 CN-Cha 的 `GRP_HEIGHTC`，group_id 40000651–40000653，分别给 2003、2004、2005 年 HEIGHTC=32 m；`GRP_IGBP`=MF。其原元数据包 SHA256 为 `c1b4304272c77ffeb9f6b1bdb706eba47886a3b15a68d036f28b3e088b5faa7f`。这份提取没有分 PFT 叶面积/比例，也缺 TA_F、VPD_F、WS_F 合并字段测高。
- `docs/v1-results/fluxnet-height-sensitivity-protocol.json` 已锁定 32 m 冠高和 61.8 m 的 T/q/风基线；测高属于 Wang 授权的文献假设，粗糙度 d/h=2/3、z0m/h=0.1 也明确是假设。[原测高引用论文](https://journals.ametsoc.org/view/journals/apme/51/6/jamc-d-11-0243.1.xml) §2a 描述平均冠高约 26 m、LAI 冬约 2/夏约 6，并转引既有文献。该描述不是新冬季 LAI 实测，且 26 m 与 BIF 32 m 的定义/时期差异未解决；本次不替换已锁定冠高或测高。
- 冻结 MH 包 `outputs/v1/climatology-failed-year-mh-export.tar.gz` 的 `inputs/configs/CN-Cha-2003-wilting-hydraulic-clim.json`：初始名义 LAI=3、最大 LAI=5、SLA=0.02、叶 C=150 gC/m²；物候 `lai_minimum=0`、初始 `leaf_fraction=0`；没有 `canopy_structure`。其 assumptions 明说一个聚合 C3 PFT、名义叶/CN 参数、未分混交比例。这些值不因与某篇论文数量接近就成为站点实测。其余已确认的 SAI=0/冬叶=0 事实沿用 MW，不重复计算。
- `docs/CTSM_PFT_GEOMETRY_INPUTS.md` 及对应固定版本官方输入只提供候选 PFT1/PFT7 的粗网格月值与参数来源。它们可作为显式气候态假设，不能提供站点常绿/落叶混合权重；也不能以粗网格最大 PFT、树干数量、胸高断面积或已有通量评分替代权重。

**目前能独立固定的是“混交类别、非零枝干结构的待检验情景”，不能固定唯一完整站点配置。** 最小后续工作是原状态下的结构隔离诊断：只在明确的枝干面积约定下比较 SAI=0/0.8/1.5，保留活叶及全部库存、强迫和 MOST 几何，先检查守恒与辐射/地面交换响应。真正修订常绿叶与落叶叶的 C/N 物候，仍需分组面积定义、SLA/叶碳和独立季节依据，不能把叶面积凭空加进原状态。

核验记录：Guan p.143 和 Zhang p.184 已逐页查看原 PDF 图像。下载于 2026-10-25 的原 PDF 分别为 657080 B、SHA256 `ef3af6367d175915a13461ab84455827f61f000385d4870765e0917ea5ac2994`；557293 B、SHA256 `8b74dc80f6cf7cf12d27bf5b7a893d32b1bb8ea7b345946c36b04991c6d33567`。Shi 使用出版方全文的段落和表号定位；不依赖 ResearchGate、综述或搜索摘要来确定 0.8/55.30 的含义。未获取分 PFT 冬季活叶面积原记录；本备忘不作物理验收结论。
