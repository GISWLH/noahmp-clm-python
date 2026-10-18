# 完整柱的SAI接口缺口（2026-10-18）

## 当前进度：NumPy完整柱与原生独立组件完成，原生完整柱待接入

NumPy参考组件在EHPC PBS149完成33项，对独立连续月衰减/叶落源解析解的
最大偏差1.13000364%（20天、日步长、generic crop）；半小时非作物无叶落
偏差0.00578821%。[原始证据及固定495源](v1-results/stem-reference-cpu-a-complete/manifest.json)
包含全部12组实际数字。没有把SAI诊断当作新增碳库或用通量评分选面积。

完整NumPy柱496源随后在EHPC PBS150完成40项目标及全部848项CPU回归，
零失败/错误/跳过，完整回归556.558283秒。[实际记录](v1-results/stem-column-numpy-a-complete/manifest.json)
覆盖无叶有茎的雨截留/光学/能量、显式MOST、雪埋后旧冰的携焓路由、
prescribed SAI、C/N实际周转/生长、收获0.25→休耕0.25→再种植0的状态携带。
水热/C/N仍按原独立硬界检查，未改变原70状态ABI。

新状态显式携带总`sai`及`CanopyStructureParameters`；用
`with_vegetation_structure`启用。prescribed模式必须给实际初始SAI，动态模式
使用固定参考规则：非作物月损失0.5/下限0.5，generic crop月损失1/下限0.05，
本项目C3年作物SAI=0.2LAI、收获残茬0.25。总面积与雪埋后的暴露面积独立；
几何在本步降水前以旧雪深诊断，用于截留、辐射、能量、地面交换及MOST。
C/N柱以同一原SAI/LAI分别计算物理阶段所用几何和最终已获资金的叶面积几何，
没有二次衰减。真实站点的PFT/几何/初态采用仍需显式配置，尚未改原科学年度。

原生独立组件已同步JAX32/64与CUDA32，在xuanze完成4项、每矩阵800工况。
[最终500源B证据](v1-results/stem-native-component-b/manifest.json)：float32对NumPy
最大绝对差9.77479902e-7 m²/m²、非近零最大相对偏差0.000140357234%，
float64最大绝对差8.88178420e-16；CUDA32/JAX32全14字段最大差0。
[首版A失败](v1-results/stem-native-component-a/manifest.json)保留：4项中2失败，
分别为完全雪埋的倒数除法舍入使暴露面积非精确0，及NVRTC找不到`math.h`。
修订明确雪埋几何端点并去掉不必要头文件；没有放宽10%/1e-6或删零端点断言。

**此项仍未全部完成：** 原生完整柱的辐射/能量/MOST/截留以及物候/BGC/作物/
灌溉/湿地已形成502源候选，通过独立三数组扩展保留SAI/高度/方案，不改变
旧70状态ABI。14:10:48UTC在xuanze启动19项目标（含旧入口回归），尚待终态，
[原启动快照](v1-results/stem-native-column-launch-a/manifest.json)只确认当前组件4项。
旧native packer仍明确拒绝丢弃结构，候选使用`pack_clm_structure_columns`显式
保留；只有验证后才能采用。另EHPC PBS151已完成16项实际SAI湿地/灌溉/
作物水热/C/N/CH4目标（31.291253秒），503源仅新增该测试文件，其余502字节
完全相同；[16项实际记录](v1-results/stem-extensions-targets-a/manifest.json)已归档，
随后[全CPU864项](v1-results/stem-extensions-full-a/manifest.json)在594.577818秒实际完成，
零失败/错误/跳过，PBS151实际退出0、503源哈希不变。

另在空闲GPU0完成同一502源的[CUDA32独立整柱28柱步](v1-results/stem-native-cuda-prefix-a/manifest.json)：
分层、物候、BGC、作物、湿地、灌溉各2柱×2步，另2柱×2步显式MOST。
全部通过原NumPy10%及独立守恒审计；H/LE最大绝对差0.00536936/0.00288245 W/m²，
SAI差1.77986e-7 m²/m²。独立水/热/C/N最大残差2.54996e-5 kg/m²、
27.9184113 J/m²、5.62374e-4 gC/m²、3.85394e-5 gN/m²，原数值界保留。
记录包含设备同步，但detach入口没有独立进程退出码，不计正式19项worker终态；
JAX32/64、同精度及旧入口回归仍待完成。上述不能抵充尚未完成的原生完整柱、全年
或观测验收。下面493源诊断作为历史保留；原K后来failed/78，L也已failed/6198。

当前493源的独立冠层组件接受`stem_area`，将叶、茎共同用于光学/热交换/
截留；`coverage`的干叶面积比例排除了茎，避免茎发生气孔蒸腾。但完整
`CLMColumnState`及`step_clm_enthalpy_column`没有SAI状态/输入，整柱的分层
面积、截留容量、地面交换衰减及MOST植物面积只由LAI构造。对应的原生
完整柱也没有茎面积输入。这是组件未接入完整柱的缺口，属于P2完成范围。

EHPC PBS147已实际读取[官方全球地表数据及四站邻域原值](v1-results/ctsm-reference-input-next-b/manifest.json)。
其`MONTHLY_SAI`为time×lsmpft变量；四个邻居、全部PFT原值中，Cng/Qia/Cha/HaM
最大SAI分别2.59220747、1.21784350、3.56976241、2.19315722 m²/m²。这些是
粗网格各PFT的范围，不是站点实测、拟合参数或已选定的站点SAI。
数据证明有真实参考字段可用，尚未证明它是此前H/土温等偏差的原因。

已只读核对固定上游
[`CNVegStructUpdateMod`](https://github.com/ESCOMP/CTSM/blob/fdfc03a0abec838ec0b66663a6a2f5e619cd2611/src/biogeochem/CNVegStructUpdateMod.F90)：
CN模式的总SAI具有前步记忆、叶面积减少项及PFT下限/衰减，另区分雪埋后的
暴露SAI；作物还有独立几何/收获残茬处理。因此不能任意用一个未验证的
“茎碳×系数”替代全部情形，也不能只把读取SAI字段当作完成物理接入。
这里只理解接口和物理定义，不复制上游实现。

后续实现需明确规定SAI与动态C/N结构的关系，守恒状态及检查点携带；把实际
叶/茎几何一致接入截留、辐射、能量、地面交换和MOST，并区分总量/雪埋暴露量。
先完成NumPy独立参考，覆盖无叶有茎、落叶/收获、雨雪截留和水热/C/N，再同步
JAX/CUDA，在原10%数值、同精度1e−6及独立守恒门内核对。不得按现有通量评分
选择SAI或改原年度/初态；当前K/L和493源完整回归继续保持原身份，不能作为
该尚未实现扩展的验证。
