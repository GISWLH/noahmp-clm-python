# CLM饱和基质势与冻土平衡液水（2026-10-17）

**新曲线已接入开发柱的参数、焓正逆变换、传热和输送，核心设备26项已完成；
完整冠层/BGC回归仍在运行。** 已运行的旧气候年使用原近似，不能将下面结果
算作全年或观测验收完成。

核对冷启动冰水后发现，当前`soil_enthalpy`/历史`frozen_soil`用统一的`5e-5`
吸力尺度，未传入土质相关的饱和基质势；温度比例也使用融点近似。
[CLM5技术说明Eq2.6.52](https://escomp.github.io/CTSM/release-clm5.0/tech_note/Soil_Snow_Temperatures/CLM50_Tech_Note_Soil_Snow_Temperatures.html#equation-6-52)
要求饱和基质势和当前土温。以正吸力头`psi_sat_m`（m）表示，平衡液水上限为
`theta_sat * [Lf*(Tf-T)/(g*T*psi_sat_m)]**(-1/B)`，再受总水量限制。
CLM文献中的mm吸力头已在本式转换为m；参数不是水汽压，也不是为站点评分调参。

使用已冻结[官方土壤表](https://github.com/NCAR/noahmp/blob/17751dcd7a2442a3c57138f530f523daf08dc1c5/parameters/NoahmpTable.TBL)
的STAS1/6/9/12：吸力头分别0.069/0.355/0.263/0.468m。海北STAS9同初温、
同总水量的初始液水，当前CLM为0.1781898、官方0.1588106m³/m³，相差约12.2%；
这不是旧1e-4精度门的问题。长白山还存在名义土壤与STAS6参数差别，不能把它的
全部冰水差异只归于一个常数。

## 先NumPy、后设备的实际验证

NumPy新函数位于[clm_freezing.py](../src/noahmp_clm/physics/clm_freezing.py)，
显式接收总水量、温度、孔隙度、B和正吸力头，返回液水及温度/总水量解析导数。
保留P1的局部4%平滑上限，其相对直接min的最大改变量约1%；没有调整P1验收或
重新尝试M1全量ΔT。当前库使用Tf=273.15K、Lf=334000J/kg、g=9.80616m/s²，
不宣称与CTSM/Noah常数逐位相同。

EHPC PBS41在计算节点完成**10 passed /0 skipped，0.37秒**，包括三份已锁定
官方初始液水点、温度/含水量导数及质量边界。独立压力头公式另覆盖每土质
404温度×5含水量，共8080点，全部保留、无拟合：

| 土质 | 旧曲线对独立方程最大相对差 | 新曲线最大相对差 |
|---|---:|---:|
| STAS1砂土 | 148.8295% | 0.675902% |
| STAS6壤土 | 18.8209% | 0.953774% |
| STAS9黏壤土 | 15.8967% | 4.44e-14% |
| STAS12黏土 | 5.59686% | 0.773481% |

零液水不计算相对差，绝对差也保存在原记录。
[NumPy原始数字、测试与464文件未变证明](v1-results/soil-freezing-numpy-a/manifest.json)
固定先行参考结果。

随后xuanze独立组件目标完成**3 passed /0 skipped，2.55秒**：JAX32/JAX64各
8000个量化工况；CUDA32/JAX32液水及两个导数的最大差均0，原rtol/atol=1e-6
同精度门保持。JAX32相对同量化输入NumPy64的液水最大绝对差0.000591740m³/m³，
温度/总水量导数最大差0.795765K⁻¹/0.0119660；均满足预先10%或近零绝对门。
JAX64对应最大差5.55112e-17m³/m³、5.32907e-14K⁻¹、2.77556e-15。
这些是组件量，不是日土温误差或整柱热账。

首版CUDA只因NVRTC找不到不需要的宿主`math.h`在编译前失败，JAX两项通过；
修订仅移除该头文件，工况、方程及门不变。
[原失败](v1-results/soil-freezing-gpu-a/manifest.json)与
[修后完整设备证据及468文件未变证明](v1-results/soil-freezing-gpu-b/manifest.json)
分别保存，没有把编译失败算作CUDA已测。

## 耦合接入及当前验证

`SoilParams.saturated_suction_m`明确使用正吸力头m；名义砂/壤/黏土默认
0.069/0.355/0.468m。其余名义孔隙度/B仍是原近似，不能称整套参数等同STAS。
历史v0.2 AHC/Thomas及70浮点打包路径保持原物理实现，不使用这个新字段。
开发焓路径的Option1采用上述CLM方程；Option2保留历史简化三次迭代，**尚非
独立验收的完整Koren1999实现**。

NumPy焓及解析热容已接入；JAX/CUDA焓正逆变换、雪土联合热量、携热土水输送、
VIC、完整柱准备/收尾、分层冠层、物候/BGC、灌溉和瓦片排水均传递该字段。
设备内部仍使用原温度偏移坐标，无M1全量ΔT改写。新开发数组宽度为：热力学6、
联合热量土性4、紧凑土水参数5、完整土性7；旧宽度明确报错，不静默补默认值。
历史报告缺少吸力头时，观测土壤算子和官方冠层诊断也拒绝按新曲线解码，须用
其原冻结源码。原气候检查点不能改写身份后接着运行。

先行EHPC PBS43的NumPy焓/传热/完整柱目标**44 passed，9.16秒**。PBS44
完整CPU部分707项中701通过、6失败：五份独立表面参考仍调用旧冻结曲线，另一份
Cng测试使用旧曲线金值。前五份已改成独立压力头方程；Cng原金值保留为历史，
新曲线检查守恒及独立液水方程，设备另与当前NumPy比较。此处不声称新物理
必须复现旧物理的通量。[原失败及冻结源码](v1-results/suction-numpy-a/manifest.json)
全部保存。

随后PBS45为**703 passed /4 failed，284.22秒**，四份历史夹具仍是六字段，
触发新七字段契约。派生输入明确追加0.355m并保持原质量/焓，故解码温度可改变；
这不是原旧曲线的逐位重放。原夹具不覆盖、不删除，质量/焓和原数值门保持。
[第二轮失败证据](v1-results/suction-numpy-b/manifest.json)保留。
PBS46较早快照为704 passed /4 failed（274.14秒），同四项旧宽度问题，
[该失败也完整保存](v1-results/suction-numpy-c/manifest.json)。
修订后PBS47在compute002完成**708 passed /0 failed /0 skipped，289.40秒**，
469份冻结文件未变，[完整CPU选择清单、JUnit及源码](v1-results/suction-numpy-d/manifest.json)
保存。GPU文件另在xuanze执行，不能把708称作全GPU/最终全量验收。

xuanze核心耦合验证**26 passed /0 skipped，69.12秒**，469份冻结文件未变：

| 对比 | 最大实际差 |
|---|---:|
| CUDA32/JAX32，各已比较字段 | 0（rtol/atol=1e-6） |
| 焓反演JAX32/NumPy64，温度 | 3.28470e-5 K |
| 焓反演JAX32/NumPy64，液水质量 | 3.78430e-4 kg/m² |
| 土水输送JAX32/NumPy64，温度 | 2.41676e-5 K |
| 雪土传热JAX32/NumPy64，温度 | 5.88236e-4 K |
| 雪土传热JAX32/NumPy64，焓 | 19.7615 J/m² |
| 雪土传热JAX64/NumPy64，温度 | 4.77485e-12 K |

独立双精度累计的单步水/热残差，土水输送最大3.43323e-5kg/m² /50J/m²，
联合传热最大30.0197J/m²；逐层/逐工况均满足原浮点储量运算界，并非用10%
放行守恒。[完整日志、JUnit、原生源码及未变证明](v1-results/suction-gpu-core-b/manifest.json)
可追溯。热容在接近冻结时很大，日志中的J/m²/K绝对差不是土温偏差；仍按原
组件10%/近零绝对门检查，未宣称逐元素严格float64相同。

更完整的冠层/BGC/管理模块逐文件测试已在xuanze启动，第一份冠层表面文件
5 passed /0 skipped，422.88秒，其余尚未收齐。
[首份已完成文件的证据](v1-results/suction-surface-prefix-c/manifest.json)已独立保存。
当前174份运行库文件与核心B、
整柱C和CPU最终D逐字节一致；C到D只改两份夹具适配测试，
[完整静态身份及归档核对](v1-results/suction-coupling-static-audit-20261017.json)单列。
接入后的
连续气候柱、官方/观测重新评价仍待完成，没有据此勾选完整年度或v1.0发布。
