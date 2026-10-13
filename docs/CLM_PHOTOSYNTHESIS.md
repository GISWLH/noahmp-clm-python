# P2 光合参考模块（开发中）

当前只实现 NumPy 单叶组件 `physics/photosynthesis.py`，尚未接入柱驱动；不能
把它计为已完成多层冠层或已通过 FLUXNET GPP 验证。JAX/CUDA 同步也尚未完成。

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

下一步：补充文献/观测光响应对照与JAX/CUDA单叶同步；完成逐层双流
辐射、叶温能量平衡、冠层空气温湿和水量限制，再将冠层 GPP/ET 接入驱动。
数学方程核对、后端核对和 FLUXNET 观测验证分别报告，不互相替代。
