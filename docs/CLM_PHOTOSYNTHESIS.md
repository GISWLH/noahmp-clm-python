# P2 光合参考模块（开发中）

已实现 NumPy、JAX64/JAX32、CUDA32 单叶组件，尚未接入柱驱动；不能
把它计为已完成多层冠层或已通过 FLUXNET GPP 验证。

方程来源为 [CLM5 技术手册 2.9 节](https://escomp.github.io/CTSM/release-clm5.0/tech_note/Photosynthesis/CLM50_Tech_Note_Photosynthesis.html)，
参考 [PhotosynthesisMod](https://github.com/ESCOMP/CTSM/blob/release-clm5.0/src/biogeophys/PhotosynthesisMod.F90)
和 [CanopyFluxesMod](https://github.com/ESCOMP/CTSM/blob/release-clm5.0/src/biogeophys/CanopyFluxesMod.F90)
的责任划分，仅阅读结构，不复制实现。官方 CLM5 默认气孔模型为 Medlyn；本项目
按 Wang 指定使用 Ball–Berry，并按指定取 Rubisco/RuBP/TPU 限制的最小值。

单叶输入显式区分吸收 PAR（µmol photon/m² leaf/s）、CO₂（ppm）、气孔/边界层
导度（mol H₂O/m² leaf/s）和面积基准。电子输送率使用二次方程的小根；温度响应
使用 Arrhenius 和高温抑制。氮限制只下调光合容量，呼吸参数独立。
气孔导度与叶面湿度联立，叶内 CO₂ 用扩散供给与生化需求的括区求根闭合；
夜间允许呼吸导致叶内 CO₂ 高于环境，不把负净光合截为零。

已用独立 Decimal 60位运算建立21个固定叶内CO₂/光照参考点（不导入生产实现）。
NumPy 最大绝对差 **3.552714e-15 µmol CO₂/m² leaf/s**，最大非零相对差
**2.219913e-16**；12项测试同时覆盖夜间向外扩散、叶面湿度耦合、TPU、缺氮和
干旱胁迫。原始表为 `tests/fixtures/leaf_clm5_equations.json`，生成入口为
`benchmarks/reference_leaf_table.py`。这是一份公开方程核对，**不是叶片观测数据
或官方CTSM输出对照**；不能据此宣称文献实测光响应和冠层GPP已经验收。

## GPU 数值证据

6,075组笛卡尔积输入覆盖光照0–2000、268–320 K、280–800 ppm、湿度、
边界层导度、缺氮和水分胁迫；另用固定种子711生成2,048组随机输入和第二套
参数。CUDA32/JAX32在全部12个输出上最大差为**0**，扩散残差也参与1e-6门。
JAX32对同量化输入NumPy64的最大净光合差为**2.237390e-5 µmol/m²/s**，
叶内CO₂最大差0.001099683 ppm；JAX64最大净光合差2.220446e-16 µmol/m²/s。
源码、设备及完整逐字段数字见 `v1-results/p2-component-audit.json`。

首轮5个扩散残差项相差最多5.722046e-6，原因是CUDA默认舍入平方根与XLA的
近似平方根不同。CUDA改为同一运算后消失，没有删除残差、抬高精度或放宽门。

## Leafweb 实测光响应：探索性跨曲线评价

参考 Han、Zhang、Sun、Telesphore、Grodzinski、Yin、Porcar-Castell、Berry、Gu
(2022)，[Leafweb December 2022 Release](https://doi.org/10.25581/ornlsfa.027/1887896)。
本次使用2025-09-17发布对象中的完整 `C3_grass.zip`，物种为
*Dichanthelium clandestinum*，Cornell Botanic Garden，2020-07-15。
原始压缩包以 **CC-BY-4.0** 许可保存在 `tests/fixtures/leafweb_c3_grass.zip`；
下载URL与SHA256记录在 `benchmarks/evaluate_leafweb.py` 和结果JSON中。
原始文件未改写，不用模型生成值替代观测。

每文件独立使用暗测定的呼吸和最大PSII效率，用 A–Ci 曲线拟合 Vcmax/Jmax/TPU，
**不使用光响应行拟合**。用实测叶温和叶内CO₂预测其独立光响应行；因此这是
生化组件评价，不验证Ball–Berry气孔。光吸收率0.84来自CLM5草地叶片光学表，
是参数假设，不是本站实测；曲率0.7保持固定。该协议经过本数据探索，不能称
盲测。部分参数的Jacobian秩不足3，尤其TPU未受充分约束，结果中明确报告。

| 文件编号 | 留出光响应点数 | 平均相对偏差 | RMSE/观测均值 | r |
|---|---:|---:|---:|---:|
| 1 | 13 | −4.7093% | 20.5571% | 0.956252 |
| 2 | 13 | +5.0394% | 13.7503% | 0.981445 |
| 3 | 13 | −11.3617% | 34.4809% | 0.827356 |
| 4 | 13 | +3.2112% | 7.5523% | 0.995340 |
| 5 | 0 | 未测 | 未测 | 未测 |

以整条光响应曲线归一化RMSE≤10%评估，只有第4条达到该指标；第1/2条虽平均
偏差<10%，曲线形状仍未通过，第3条平均偏差和r也未通过。**不能宣称全部单叶
观测已验收**。近零净光合用0.5 µmol/m²/s绝对容差，逐点失败数和每一对预测/
观测都保存在 `v1-results/p2-leafweb.json`，不以相对误差卡夜间接近零的值。
当前Farquhar组件在低于CO₂补偿点时将生化碳收益截为零，尚不能完整描述
低CO₂实验中的光呼吸碳损失；后续应与容量参数可辨识性一起审查，不盲调留出曲线。

复算：`python benchmarks/evaluate_leafweb.py tests/fixtures/leafweb_c3_grass.zip
--output outputs/leafweb.json`。

下一步：将已验证的[辐射组件](CLM_CANOPY_RADIATION.md)与逐层叶温、冠层空气
温湿和水量限制联立，再把GPP/ET接入柱驱动。数学方程核对、后端核对、叶片
观测和FLUXNET观测验证分别报告，不互相替代。
