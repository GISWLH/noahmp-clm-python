# P2 光合参考模块（开发中）

当前NumPy、JAX64/JAX32、CUDA32单叶组件已接入分层冠层和完整BGC柱；
实际FLUXNET GPP与下列未达标单叶曲线仍须独立评价。早期组件检查按日期保留。

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
当前Farquhar组件在低于CO₂补偿点时将生化碳收益截为零，这也是CLM5说明
2.9.3/2.9.4明确给出的约束，不能仅凭本数据拟合误差就判定为编码缺陷并移除。
它对低CO₂实验的适用性仍需与容量参数可辨识性一起审查，不盲调留出曲线。

复算：`python benchmarks/evaluate_leafweb.py tests/fixtures/leafweb_c3_grass.zip
--output outputs/leafweb.json`。

已将温度系数移到48次ci迭代之外；4096个相同随机输入的12项输出与重构前
逐位相同。新增[NumPy叶温/温湿/水量联立与无雪柱连接](CLM_CANOPY_ENERGY.md)，
该连接尚待JAX/CUDA和冬季同步。数学方程核对、后端核对、叶片观测和
FLUXNET观测验证分别报告，不互相替代。


## 零光照解析扩散与真实输入耗时（2026-10-16）

Bondville固定LAI/MOST首日执行暴露出多层求解嵌套的CPU开销。对保持运行的
Linux原实现做15秒、20 Hz的py-spy采样，共299个有效样本、采样错误0；
单叶气体交换占240/299（80.27%）含子调用样本，冠层相态求解占295/299。
这是执行热点，不能仅凭采样推断物理不收敛。GDB首次采样因缺少Python调试
辅助命令未取得Python栈，随后用独立目录的py-spy 0.4.1取得栈与speedscope文件。

NumPy现在仅在整批PAR精确为0时使用扩散方程的闭式解：零电子输送使GPP=0，
净同化为−Rd，Ball–Berry返回最小导度；以原浮点表达式得到的gs计算
`ci = ca + Rd × (1.4/gb + 1.6/gs)`。保留所有12项输出及实际扩散残差，
混合明暗/非零光照继续原48次括区迭代。没有改白天方程、参数、守恒或收敛门；
JAX/CUDA原生迭代保留，作为独立数值对照。

对修改前提交25e8b52的不可变源码比较12,288个随机黑暗工况：GPP、净同化、
gs、叶面RH逐位一致；ci最大差5.00222e-12 ppm，Rubisco限制差5.68434e-14，
扩散残差差2.50483e-13 µmol/m²/s。10层黑暗叶100次调用、重复5轮，本机
中位时间0.120065→0.0161771秒（7.42倍）；这是叶组件计时，不冒充完整柱加速比。

NumPy新增解析/混合批次独立检查在本机15项通过。冻结B（407源文件）Linux
单叶、实际JAX/CUDA及冠层相态目标 **39 passed / 0 skipped，141.57秒**。
1215纯黑暗输入的CUDA32/JAX32最大差0；JAX32对NumPy解析解最大净同化差
4.55919e-7 µmol/m²/s、ci差0.000395068 ppm、gs差9.49949e-11 mol/m²/s；
JAX64 ci差2.27374e-12 ppm、净同化差0。原6075格点及2048随机输入也仍满足门。
完整柱修改前F与修改后G均保留隔离快照运行，尚未据部分轨迹宣布全天性能或
官方偏差验收。[采样、逐字段数字、原生日志及源码身份](v1-results/dark-leaf-b/manifest.json)。

本次性能核查的文件改动：

| 文件 | 类型 | 用途 |
|---|---|---|
| `outputs/v1/profile_output/compare_dark_leaf.py` | 新建独立测量脚本 | 加载旧提交比较12项输出并重复计时 |
| `outputs/v1/profile_output/bondville-15s.json`、`bondville-python-stack.txt` | 采样输出 | 对原Linux进程只读采样；没有在生产循环插入计时 |
| `src/noahmp_clm/physics/photosynthesis.py` | 数值等价优化 | 仅零光照解析扩散分支，不是临时插桩 |
| `tests/test_photosynthesis.py`、`tests/test_gpu_photosynthesis.py` | 新增回归 | 解析关系、原迭代对照及原生精度门 |

测量脚本和原始采样存放在忽略的输出目录，压缩证据单独归档；生产代码没有
待清理的计时或调试插桩。
