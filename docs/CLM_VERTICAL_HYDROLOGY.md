# 2026-10-26 排水适配最新证据

[PC-d实际小probe85件](v1-results/drainage-default-geometry-pc-d/manifest.json)：
唯一worker/parent0、controller完成无timeout、3实际PID退出，原源未应用patch。
float32现有PTX div.full及原HLO常量/分组的局部候选，在GPU内逐位重现4组×9
default-JAX几何输出；6个明确primitive合法边界均finite/domain有效，两倒数
常量RN guard失败0。没有新JAX或完整物理积分。原4组的严格/own RN差仍为
12/8、6/3、9/5、6/4；RN只是独立描述，未作为新的同精度/科学门。

[EHPC1867 Fraction独核](v1-results/drainage-default-geometry-pc-d-fraction/manifest.json)
实际F0/Stageout0/run1，4原输入/context、10域、两常量及全部eligible硬RN bits/
flags与精确有理数RNE吻合；原半途/次正规/溢出/有符号零自检真实assert通过。
0模型/GPU/replay。该试验编译ftz=false，不能替代原CuPy wrapper的完整模块
验证。实际共享CUDA1，17.82秒仅交付耗时，不作性能结论；其他训练未改动。
596源的完整候选仅准备，595其他文件不变，仍须原8案/参考/1e−6/整数/P0/硬账。

[PC-e完整设备候选](v1-results/clm-drainage-device-pc-e/manifest.json)实际CUDA/JAX
各8案均返回0，原wrapper `-ftz=true` 的真实加载/cubin/cache证明通过；EHPC1873
F1/Stageout0。P0、行为和原64位硬守恒16/16满足，但75字段的同精度/整数仅5/8。
perched_stress出流差0.0114059448 kg/m²、焓差4151.25 J/m²且接受步数差1；
option2出流差0.0154600143 kg/m²、焓差5625.75 J/m²；two_bands两能量残差
差0.015625 J/m²。原生产wrapper和门不改，候选不推广。

[独立Fraction核验](v1-results/drainage-geometry-pb-v-fraction/manifest.json)1845实际
F0/Stageout0：纯有理数和IEEE binary32最近偶数舍入，含半途/次正规/有符号零/
溢出自检，8个上下文合法；分类与PB-z硬件参考一致，0新物理或GPU调用。
例如frozen saturation原JAX为3f71afd6，精确RN为3f71afd5。该诊断不抵充完整积分。

[固定版本编译器源码](v1-results/jax-division-source-pc-a/manifest.json)及
[实际隔离进程试验](v1-results/jax-division-probe-pc-b/manifest.json)进一步定位：
本机安装JAX/XLA的NVPTX默认单精度除法模式1使用近似div.full，模式2使用IEEE
精确除法。单个JIT的公开compiler_options不支持该嵌套选项映射，见
[精确接口核查](v1-results/jax-jit-option-source-pc-c/manifest.json)；未发明局部API。

两独立CUDA1进程总15.689秒、均实际0，x64均false，HLO无f64。baseline无
XLA_FLAGS，原PB-v全部payload逐字节重现；precision2只设置已核实的进程级
`--xla_backend_extra_options=-nvptx-prec-divf32=2`。四案实际操作数RN差从
8/3/5/4降至0/1/3/1，变量除法差全部消失，余差均为/1000或/917的常量折叠。
没有改生产flag、源、物理式或门限，没有完整新积分/同精度通过信用。下一步
只评估局部算术一致性候选；新模块仍须原NumPy参考、1e−6、整数和硬账验证。

[PB-v 54件设备证据](v1-results/drainage-geometry-pb-v/manifest.json)：CUDA/JAX
各4次原诊断及4次扩展诊断，parent均0，总38.47秒；8个后端/案例的原29字段、
扩展29字段及对应geometry原位均逐字节重现，来源/输入/控制不变。
原及扩展HLO都将常量除法折叠为float32乘法，编译文本本身不证明数值因果。

[PB-z 42件独立硬件RN诊断](v1-results/drainage-rn-primitives-pb-z/manifest.json)：
CUDA1单kernel处理32层行/9步，worker7.002秒、parent9.04秒，实际0；0模型/
相反演/CPU参考调用。GPU内用显式最近偶数舍入add/mul/div分别比较严格链及
每步实际操作数，输出原/参考bits、相等标志及分组计数。

| 原数据来源 | 严格链不相同次数 | 实际操作数不符合RN次数 | 非法输入行 |
|---|---:|---:|---:|
| CUDA，四案例各自 | 0 | 0 | 0 |
| JAX frozen | 12 | 8 | 0 |
| JAX perched_stress | 6 | 3 | 0 |
| JAX option2 | 9 | 5 | 0 |
| JAX two_bands | 6 | 4 | 0 |

frozen每层的saturation/ice_quotient变量除法都出现实际操作数RN差；其他JAX案
还出现/1000或/917差，故只阻止常量fold不足。add、dz*porosity和917*dz等
实际操作未出现该差。RN含义按[NVIDIA CUDA12.4原语说明](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html)。
此独立参考直用NVRTC显式ftz=false，保存实际加载CUBIN及flags；原CuPy缓存
helper会追加ftz=true，生产路径原样保留。设备结果定位了原语差值，后续Fraction
参考见上；修订后的完整积分仍待，不增加原同精度/P0/年度通过信用。

[PB-w失败原件](v1-results/drainage-loaded-cuda-pb-w/manifest.json)及
[PB-x ELF结构](v1-results/drainage-elf-tail-pb-x/manifest.json)确认原缓存所有
代码/section完整，仅最后program-header表少1字节。工具4次255/2次1均保留。
[PB-y独立诊断副本](v1-results/drainage-elf-zero-copy-pb-y/manifest.json)只在
额外副本补1个尾零，原前缀及代码段不变，六次工具实际0、可读SASS；副本从未
加载/编译。没有将工具失败归因版本，也不把副本冒称原磁盘字节。

[PB-t 85件实际设备与比较证据](v1-results/clm-drainage-device-pb-t/manifest.json)：
唯一新CUDA/JAX parent均0，各8/8案返回、无超时，worker25.1718/127.7275秒，
parent40.4281/150.6389秒；只改头部成功解决NVRTC编译失败。EHPC1828真实
F1/Stageout0，全部16后端案的独立P0/行为/原M/H硬账满足，但75字段原1e−6及
整数精确配对仅4/8通过，不能推广或计完整设备同步。

失败frozen能量残差差1.14440918e−5 J/m²、error_ratio差1.16634354e−4；
two_bands残差差.015625 J/m²。perched_stress/option2还出现真实积分分叉，
总出口质量差.0145597458/.00302410126 kg/m²、焓差5299/1100.5 J/m²，
accepted_substeps差1。后两案初始phase及参数字节相同而首个diagnosis已经
不同；不将其直接归因最终求和或阈值。继续定位同输入原语，不放宽原门。

[PB-u离线43件证据](v1-results/drainage-rounding-pb-u/manifest.json)：1839真实
F0/Stageout0，零新物理/GPU调用。29字段按实际IEEE位核对，初始ice fraction
差1 ULP、saturation差1–2 ULP，frozen冰阻因子16 ULP、main rate10 ULP；
stress/option2初始滞水面差1.7881393433e−7 m/6 ULP。这些差发生在adaptive
分支前，但没有内部每步操作数，尚不能确定除法、编译舍入或具体首个算子。

[PB-r 149件原始包](v1-results/clm-drainage-warm-phase-pb-r/manifest.json)：
EHPC1704实际F0/Stageout0，24项全过，17参考记录、20金样647字段完全一致。
唯一NumPy64暖态修正直接使用原总水量，避免非均匀层厚体积回乘产生−1 ULP冰量；
冷态、焓、原非负guard和64eps硬界保持。私有596源未并入NM冻结年。

[PB-s 73件设备/比较原件](v1-results/clm-drainage-device-pb-s/manifest.json)：
CUDA/JAX实际parent分别1/0、无超时；CUDA只完成一次相反演，首个排水NVRTC编译
因math.h不可用失败，0个完整案。JAX8案/600数组返回，worker133.883秒、
parent142.328秒；EHPC1820真实F1/Stageout0，明确保留CUDA缺项及配对未完成。
JAX8案原独立64位物理/行为/三组M/H硬界均满足；stress案最大分层质量/焓/温差
.03275756 kg/m²、11878.90394 J/m²、.000143906 K，滞水总出流差−.03177222 kg/m²。
其combined水/热残差−2.98023224e−8 kg/m²/−3.0986328125 J/m²，原界
.0108337402 kg/m²/3261.5234375 J/m²。额外NumPy32同精度报告仍有4案差异及
accepted_substeps±1，未删整数条件；主要CUDA/JAX同精度尚未得到结果。

PB-t仅去掉两个host C头文件，定义精确2^-23/2^-52 epsilon；595文件及kernel
计算body不变，重新CUDA1→JAX→EHPC比较。此项不抵充整柱、post-Richards32、
JAX64、年度或科学验收。以下记录按历史保留。

# 柱内垂向水文边界复核（OC，2026-10-25）

2026-10-25 PB-n包化一致性：EHPC1673真实F0／Stageout0，在新593文件候选中
直接使用1665冻结输入32，完成20次原语、17次完整积分和5768次实际相反演。
全部1106数值返回字段的dtype／shape／C-order bytes与原NumPy32金样完全相同；
数值子进程44.649228秒，worker45.265075秒。最大水／热残差仍为
9.771436453e−6 kg/m²／5.375 J/m²，对应原界.0108337402344／6119.685546875，
最大界占比.00101008／.00106114，未修改原64eps账。
974个独立文件前后不变：候选源及清单594、准备及清单24、执行控制11、金样345。
其中593候选源包含全部586父源；两个CUDA文本
只作包资源核对，未运行。此轮仅包化／import路径／资源一致性，不计64位包化、
JAX／CUDA、整柱、native适配或生产集成。
[270件原始证据](v1-results/clm-drainage-package-pb-n/manifest.json)、
[原件核验及独立复审](v1-results/clm-drainage-package-pb-n-review/manifest.json)。

PB-o随后在EHPC1678真实F0／Stageout0完成12项CPU输入前置检查测试，
零失败／错误／跳过；这些检查证明非法dtype／shape／phase控制在后端导入、
转换和相计算前被拒绝，保留调用方显式pack32。再用原1673执行器和1665输入32
重复20原语／17完整积分，5768次相反演、1106字段仍全部dtype／shape／bytes相同；
原水／热最大残差与界不变。pytest0.30秒、测试父进程.714678秒、数值43.779411秒，
worker45.132005秒；975源／控制／参考／金样文件前后不变，594候选内包含原586。
[274件原件](v1-results/clm-drainage-guard-pb-o/manifest.json)及
[控制／实际结果独审](v1-results/clm-drainage-guard-pb-o-review/manifest.json)仅支持新排水组件的
CPU入口合同和NumPy32一致性；实际JAX／CUDA、64位完整积分、native整柱guard／
适配和生产接入仍未完成，不能把sentinel负测试计为设备计算。

PB-p-b隔离适配随后在EHPC1695真实F1／Stageout0：24项中23通过、1失败，
零错误／跳过；数值父进程17.434406秒、worker18.476799秒。新入口接收实际
水／焓库存，先相反演及纯诊断，再依滞水／主排水顺序推进；尚未接入生产柱。
唯一失败为nonuniform_dz-float64，第二个主排水半步被原输入域validate拒绝，
尚未取得该案最终预算或物理比较；同案float32通过，后续单案捕获见下。
不将域异常称为10%物理偏差失败，也不跳过或修改原guard。

16份已完成独立参考记录及原20组金样保留；纯诊断／单步647返回字段的
dtype／shape／bytes与金样一致，不等同于前轮17完整积分重跑。暖态64主出口
.5462453722 kg/m²；32位滞水压力案为8.26828289、独立参考8.28549276 kg/m²，
最大逐层出流／携焓绝对差.01774577 kg/m²／6434.90394 J/m²，温差8.28708e−5K。
该案原水／热残差3.17022e−6 kg/m²／2.001953125 J/m²，小于原界
.0108337402／3261.523193；完成记录均保持各自原硬界。1033源／控制／金样文件
前后不变。[139件原结果](v1-results/clm-drainage-adapter-pb-p-b/manifest.json)
及[独立实际复核](v1-results/closed-diagnostics-pw-pbp-review/manifest.json)
仅支持上述CPU组件范围，未运行JAX／CUDA、native整柱或年度。

[PB-q单案捕获](v1-results/clm-drainage-nonuniform-pb-q/manifest.json)在EHPC1702
真实F1／Stageout0复现原ValueError。唯一拒绝谓词为第二层ice_negative：
T=290K、dz=.375m，W=164.92674916302653、liquid=164.92674916302656 kg/m²，
ice=−2.84217094304e−14 kg/m²，恰为−1 ULP。液水与原体积到质量的回乘逐值相同，
其余全部原谓词false，重建bad与原bad一致。原参考1次返回、适配1次失败，
630原文件及控制不变；诊断完成不表示该测试通过。下一候选只令NumPy64暖态
直接使用总水质量、冰量0；原冷态、输入检查及水热守恒门保持，修订验证仍待。

2026-10-25后续开发：补齐诊断土柱水位、饱和带及冻层上方滞水带排水的NumPy
参考组件，再同步JAX／CUDA及整柱连接。[CLM5公开水文方程§2.7.3—2.7.5](https://escomp.github.io/CTSM/release-clm5.0/tech_note/Hydrology/CLM50_Tech_Note_Hydrology.html)
的这条路径使用零底通量土水求解，随后从实际土层移出径流，不以独立WA水库
为前提。新组件应保留水位与分层取水的物理关系，输出逐层水量及携焓账；
水位、分层库存和排水量都需独立参考验证，不能只核总量守恒。在跑冻结年
继续用于原基线评价；新组件尚未接入生产。

后续PB-c已在EHPC1614真实F0／Stageout0完成22项测试、子进程55.630566秒；
原18个测试函数AST不变，直接复用1610七份参考原件，没有重跑官方三例。
修订完整层主冰权重、滞水相邻层阻力和下一层供体后，无冰总出流偏差仍为
−0.195814%；冰层案例的出流绝对差为1.566564e−10 kg/m²；滞水出流绝对差为
8.410906e−15 kg/m²，官方与组件都从第三层扣水，四层最终总水差均0。
两个无量纲冰阻系数分别为0.0010167384647167886、0.010111280751435874，
与原调用器实际值相同。三例水残差最大9.951462e−15 kg/m²、热残差最大
3.929017e−9 J/m²，均逐例小于各自原硬界；原解析及ODE结果保留。
[PB-c完整结果](v1-results/clm-saturated-drainage-pb-c/manifest.json)、
[原件及未改测试核验](v1-results/clm-saturated-drainage-pb-c-root-review.json.gz)。
跨层、供体耗尽、官方最小液水／过饱和修正激活、整柱和JAX／CUDA仍待验证；
这三个成功剖面不表示整个水文缺口已闭合。下文PB及1610保留修订前的真实差异。

EHPC1615的[原官方边界调用](v1-results/clm-drainage-official-pb-d-reference/manifest.json)
已真实F0／Stageout0，7个新输入进程、25个官方排水步及1个PB-c无冻积分。
同172800秒主排水总量在官方1／4／16步细化时分别为52.5424412／43.8752752／
42.8116941 kg/m²，固定PB-c为42.5034476，偏差−19.1064%／−3.1267%／−0.7200%。
16步第三层刚进入供水，不能只用总量和末库存宣布分层出口已收敛，继续细化。
极端滞水耗尽仅先取得官方真值，尚未作同热相边界的完整配对。

三个修正激活案例分清渠道：干柱增加0.04 kg/m²由同量负q_drain补偿，rsub_sat
已包含其中，WA／QCHARGE未供水；液超孔隙的200 kg/m²转到地表水库存，排水0；
冰超量35.01 kg/m²进入独立固态出口。后两项是超出NM合法孔隙域的官方压力输入，
不是组件合法状态或普通土壤验收。逐步独立合计25份完整有符号水账，最大残差
9.325873e−14 kg/m²，均在原界内；[根核对](v1-results/clm-drainage-official-pb-d-root-review.json.gz)。
不把修正后的层净变化都称为排水，也不把负排水补偿复制为未记账的水热源。

EHPC1616后续[时间细化及滞水对照](v1-results/clm-drainage-official-pb-e-reference/manifest.json)
已真实F0／Stageout0，6个新官方进程、596步，复用原PB-d主积分，没有重跑原三小例。
主排水第三层出口在64／256官方子步时为1.06984968／0.90666559 kg/m²，PB为
0.87488772；对应差−18.2233%／−3.5049%。256步第二层差+0.08085%、总量差
+0.004352%，迁移前沿绝对差0.000328300m，各层均满足原10%或近零绝对门。
64步的第三层失败保留；没有为匹配粗步改参数。

极端滞水压力案例仍未闭合：PB第二／三层出流8.24984843／13.38571719 kg/m²，
官方192步为9.54227807／11.79258539，差−13.5442%／+13.5096%；总量仅差+1.4094%
不能替代逐层结果。PB实际携焓并重新相平衡，第三层温度270→267.755207K、
冰137.219456→129.770703 kg/m²；原官方水文调用的温冰保持输入值。该对照混有
热相边界差异，不能单独据此改供体规则。PB实际105接受／4拒绝子步中几何cap
及缺液限制均未激活；单个冻结率试步的cap不是实际自适应积分的限制量。
PB水／热残差8.881784e−15 kg/m²／4.656613e−9 J/m²，分别小于原界
2.017941e−11／6.075061e−6；[逐原stdout及596份水账核验](v1-results/clm-drainage-official-pb-e-root-review.json.gz)。
主排水与诊断的隔离后端同步继续，滞水演进需同热相边界证据；整柱尚未接入。

后续[PB-f隔离后端原语](v1-results/clm-drainage-backends-pb-f/manifest.json)在EHPC1629
实际F0／Stageout0，22项测试、10份NumPy诊断，子进程1.815942秒。此前1628的
20通过／2失败原件保留；仅修零厚度预期键和浅基岩用例初态两处测试错误，三个
物理文件和阈值逐字不变。冻结率主试步最大水残差1.021405e−14 kg/m²／原界
2.557954e−11，热残差6.228220e−9 J/m²／原界1.139880e−5。
[独立核验](v1-results/clm-drainage-backends-pb-f-independent-review-wo.json.gz)确认
此轮仅执行NumPy，原语自比差0不能计作JAX／CUDA同精度证据。

[PB-g完整主排水自适应流程](v1-results/clm-drainage-adaptive-pb-g/manifest.json)在EHPC1632
实际F0／Stageout0，8项测试／5份诊断，子进程7.372660秒。实际逐试步相反演、
整步／两半步拒绝、Kahan库存及硬水热账均执行；暖／冻短积分对PB-c库存／携热
最大差0。两日跨层第三层对原官方256步偏差−3.504916%，对独立连续ODE为
+2.606307%；非零层均在10%内，零量用.001 kg/m²绝对门。该例水／热残差为
9.992007e−16 kg/m²／−1.169974e−8 J/m²，原界2.557954e−11／1.139880e−5。
只完成NumPy64主带，JAX64／JAX32／CUDA32尚未执行，滞水设备积分及整柱未完成；
明确拒绝将64位输入静默转为32位相反演。
[原stdout与完整范围核验](v1-results/clm-drainage-adaptive-pb-g-root-review.json.gz)。

[PB-h更细官方参考及冰阻诊断](v1-results/clm-drainage-hydraulic-pb-h/manifest.json)在EHPC1631
实际F0／Stageout0：只新增384／960两条原官方剖面，共1344真时钟步，以及两次
固定初态冰阻系数的NumPy诊断。旧192步及动态PB-c复用原1616，不重算或重计。
新960步官方第二／三层出口8.55054468／11.93394038 kg/m²；原动态PB-c对照为
−3.516691%／+12.165109%，仍未满足第三层10%要求，总量+5.619280%不抵充。
只固定率系数取样初态时，两层偏差降为−3.516691%／+0.411631%；更细控制为
−4.041743%／+0.431023%。实际温冰仍变化、Sy诊断始终2m、缺液和几何cap均0，
说明冰阻反馈是该压力案例的重要贡献，但不能将诊断替代实际模块或称同热边界。
官方384→960总量不单调，不据此宣称渐近阶。全部1344份原有符号水账独立复核，
最大残差1.365574e−14 kg/m²；两诊断最大水／热残差7.105427e−15／4.656613e−9，
均在原界内，没有外加热或改供体。[原渠道与边界核验](v1-results/clm-drainage-hydraulic-pb-h-root-review.json.gz)。

[PB-i显式NumPy32主流程](v1-results/clm-drainage-adaptive-pb-i/manifest.json)在EHPC1641
实际F0／Stageout0，18项＝原G8项＋新增32位10项，16份诊断，子进程39.560044秒。
原G模块／测试及5份64位诊断均精确保留。原NM三个相函数仅作NumPy导入、float32
常量与有界循环适配；状态、相反演和出口保持32位，Python时间步决策及独立64位
参考／账目重求明确区分，没有64位相态结果转存32位冒充原生链。
默认跨层第三层出流.874603689 kg/m²，对原官方256步差−3.536244%，对连续ODE
+2.572996%；更细控制为.858045220 kg/m²，对应−5.362548%／+0.631029%。各非零层
均在10%内，近零层用.001 kg/m²；更细32对默认64的差异包含不同时间控制，不能
称纯dtype偏差。两冻结选项的相反演温差最大1.318359e−5 K；完整自适应仅测选项1，
选项2完整积分仍待。浅基岩用例只验证下层受保护，前沿未实际触底。
32位最大水残差8.046627e−6 kg/m²、热残差绝对值5.375 J/m²，分别小于原界
.0137329102／6119.68555；原8epsilon相重构与64epsilon硬账因子未改。原准备文字
预判225 kg/m²量化后变224.9999847有误，实际32运算仍为225、水量化差0，原件
保留并由实际诊断更正，没有剪裁库存。JAX／CUDA、滞水设备、整柱及年度未测。
[独立原日志／dtype范围核验](v1-results/clm-drainage-adaptive-pb-i-independent-review-wo.json.gz)。

后续[PM补齐选项2完整暖／冻积分](v1-results/clm-drainage-option2-pm/manifest.json)：
EHPC1642实际F0／Stageout0，仅2项新增测试、4次完整组件积分，2.316605秒，
没有重跑原18项。19次实际32位相反演均保持float32／option2并收敛，原I物理
文件不变。暖／冻出口质量最大差2.39070e−8／5.29034e−10 kg/m²，携热差
.0172478／.000207897 J/m²，温差0／1.57879e−5 K，前沿差
−3.62588e−8／+3.95411e−8 m。32与64各自原硬水热界及零时钟残差均满足，
原名义初值／时间控制相同，显式初始焓量化约12 J/m²另列，不称纯dtype误差。
[原日志、同控制和硬账复核](v1-results/clm-drainage-option2-pm-root-review.json.gz)。

[PB-j匹配共享热闭包的官方水文](v1-results/clm-drainage-common-thermal-pb-j/manifest.json)在
EHPC1647真实F0／Stageout0：仅原耗尽剖面960×90秒，原公开水文例程与库保持，
961个17位有效数字IPC帧逐库存／温冰精确回传，960次共享NM64相反演；迁移前沿
持久、不逐步重置，main Sy诊断始终2m，全部WA／qcharge／地表／冰雪和附加
渠道保留且未激活非纯出口。原1616动态结果复用，不重跑三小例或H固定冰试验。

新参考第二／三层出口8.550544684／13.334225300 kg/m²，动态PB-c偏差
−3.516691%／+0.386163%，均在原P0的10%内；此前H的+12.165109%混有不同
热闭包，原结果保留，不能继续把它当同热边界的算法误差。前沿PB .6026666733、
新参考.6045689741 m，绝对差−.0019023008 m、相对−.314654%。原额外1mm诊断
flag为false且保留；该前沿非近零量，按Wang已确认的10%物理口径已满足，不为
额外毫米级诊断追精度或另改参数／原始flag。

共享累计水残差最大1.421085e−14 kg/m²／原界2.017941e−11，热7.450581e−9 J/m²／
原界6.075061e−6；单步最大水／热绝对残差2.842171e−14／1.123408e−8，原界
保持。独立性限于官方水文的供体、速率与前沿；液水携焓和相反演为明确共享的
已验证NM闭包，不称官方热真值。匹配热闭包对照支持继续后端同步，尚不构成
JAX／CUDA、整柱、年度或完整模块验收。
[独立原stdout、19220个IPC标量和有符号账核验](v1-results/clm-drainage-common-thermal-pb-j-independent-review-wo.json.gz)
确认645份原库／模块文件未改，961帧的W/H/T/液冰往返精确，原时钟0→960。

[PB-k滞水完整NumPy64／32链](v1-results/clm-drainage-perched-pb-k/manifest.json)在
EHPC1649真实F0／Stageout0：18项测试（两精度各9项）、20条诊断，29.748889秒。
每个试步后实际相反演，整步／两半步拒绝、逐层供体切换、缺液／几何cap分账、
Kahan库存及原硬水热界均执行；原主带6个依赖和5份参考原件逐字不变。
对J共享热闭包官方水文，64位第二／三层出流偏差−3.516691%／+0.386163%，
前沿−0.314654%；32位对应−3.787384%／+0.418871%、前沿−0.338374%，均在P0
10%内。原J和新1mm诊断仍false，保留原值，不增加毫米级合并条件。

冻结选项2也完成两精度各1800秒完整滞水积分，对PB-c参考的温差最大
0.000127706 K；这是对既有NumPy参考的验证，不称独立官方选项2认证。
32位最大水／热残差绝对值9.771436e−6 kg/m²／3.4609375 J/m²，对应原界
0.01083374／3261.5239；64位最大约1.24e−14／7.450581e−9，均在各自原界内。
此轮没有执行JAX／CUDA、整柱或年度，不能将已准备设备源码计作同步通过。
[独立原件、XML、20条诊断及范围复核](v1-results/clm-drainage-perched-pb-k-independent-review-wo.json.gz)。

[PB-l真实NumPy土柱候选](v1-results/clm-drainage-column-pb-l/manifest.json)在EHPC1654
实际F0／Stageout0：12项、13条诊断，7.022044秒。候选由NM584派生为586源文件，
583份原件精确保持，只改NumPy柱驱动并新增两个模块，完整差异与旧driver原字节
保留。它有独立显式配置，按原零底通量Richards→滞水→重新诊断主带→瓦片排水
顺序更新真实库存及携焓，不调用legacy重力排水／ARNO，不增加独立含水层库存。
两真实土壤出口和瓦片排水归深径流，原底边界通量仍0；下一步表温、根区环境
和BGC读取最终实际相态。床岩要求等于柱底；NumPy实际3／6层均完成，未将整柱
悄悄限制为设备当前4层接口，新配置在未同步native入口明确拒绝。

暖柱主出流0.546245372 kg/m²；暴雨地表160.000000、主出流0.546245372；
瓦片案例另出0.208333333 kg/m²；冻柱滞水仅1.922025e−7 kg/m²，是近零出口。
融尽雪／五种杂质账、旧None／Richards全返回数组逐字节／标量数值一致及零坡
一致性都实测。
实际完成15个耦合柱步＝13候选＋2原NM，其中10个新CLM物理步、2个BGC步；
额外10次Richards／10次PB-c／1次tile重算用于核查接线，分别统计，不重复计柱步。
最大水残差绝对值1.212364e−13 kg/m²／该例原界1.223555e−11，最大热残差
1.209846e−7 J/m²／该例原界1.638991e−5，各自原C/N／杂质界也保持。

这两步BGC氮需求与吸收实际均0，只证明实际根区环境、叶N容量映射和原C/N账
贯通，不证明活跃氮竞争或非零吸收反馈；后续PB-m补案单列如下。PB-l是隔离新源候选，
不是生产接入或官方整柱认证，不改变当前NM冻结年度，也没有JAX／CUDA或年度
验收信用。
[静态源码独审](v1-results/clm-drainage-column-pb-l-static-independent-review-wo.json.gz)与
[实际闭合独审](v1-results/clm-drainage-column-pb-l-actual-independent-review-wo.json.gz)
核对了586／583份源码、9＋3份控制、实际F0／Stageout0、12项XML与13条原诊断，
没有将重放次数计作新增柱步或把零需求案例计作活跃N反馈。

[PB-m贫氮真实柱补案](v1-results/clm-drainage-active-n-pb-m/manifest.json)在EHPC1658
实际F0／Stageout0，只运行新增1项／两个1800秒柱步，3.567872秒；没有重跑原12项，
PB-l586源和生理参数不变。初始叶氮因子0.5、矿质N .01 gN/m²，每步原定沉降
1e−5 gN/m²。两步GPP为0.344987349／0.389217152 gC/m²，潜在N需求
2.404907766／2.395433719 gN/m²，真实吸收0.009997444545／9.986776596e−6；
微生物固持为1.255545537e−5／1.322340403e−8 gN/m²。两步新矿化均0，第二步
起始矿质库0、供应仅来自固定沉降，不能据此声称新矿化反馈已测。

叶氮因子真实推进0.5→0.5020829198→0.5020850006，第二步使用实际周转后C/N。
1028次原光合生化函数调用的实际Vcmax与记录的N／温度／水分因子乘积最大差
1.421085e−14，逐次在原代数界内；不把跨步Vcmax差归为纯N效应。真实post-drain
根环境、原allocation／decomposition／plant重算一致。原M/H/C/N硬账均满足；
第二步热残差1.897387556e−4 J/m²／原界2.099679343e−4（90.37%），没有放宽界。
[独审](v1-results/clm-drainage-active-n-pb-m-independent-review-wo.json.gz)和
[根标量复核](v1-results/clm-drainage-active-n-pb-m-root-review.json.gz)分别重求氮池及
供氮账，两步总N残差各−2.524757e−14 gN/m²／约3.85547e−11原界。此补案支持
局部真实机制和接线，不构成官方／年度／设备或一般BGC科学验收。

[PO-b设备验证参考输入](v1-results/drainage-gold-po-b/manifest.json)在EHPC1665实际
F0／Stageout0／父0：9项序列化与比较合同检查、20个案例，实测40次原语、34次
完整NumPy积分及5768次成功32位相反演，88.146098秒。80个NPZ／2372个数组的
dtype、shape和原payload SHA均核验；输入、输出、源、控制及金样封条均固定。
NumPy32／64是相同名义配方、分别构造并明确记录输入量化，不称相同原始输入；
完整积分最大出口相对偏差为水2.596848%、热2.596600%，最大温差0.000842511 K，
前沿最大绝对差0.000227440m，原10%／近零绝对门和各自原硬账均满足。
32位最大完整积分水残差9.771436e−6 kg/m²／原界.0108337402、热−5.375 J/m²／
原界6119.68555；64位最大水／热绝对残差1.465494e−14／1.169974e−8，分别在原界
2.557954e−11／1.139880e−5内。

[首次1659失败](v1-results/drainage-gold-po-failed/manifest.json)原件保留：8项检查完成，
首例实测1原语／1完整积分／13次相反演后，JSON元数据中的float32标量不可序列化，
实际F1／Stageout0。PO-b仅转换元数据标量为Python float、增加一项序列化回归，
使用独立目录重新执行；物理、输入、硬界不变，首例66个重复数组payload精确一致。
[完整差异、原失败、字节核验及设备交接](v1-results/drainage-gold-po-supplement/manifest.json)
保持身份。这些是NumPy金样，JAX32／CUDA32尚未运行；未来两设备必须读取同一份
固定float32输入，全部数值返回按原1e−6、离散返回精确，不能用这批CPU结果抵充。
[独立原件及范围核验](v1-results/drainage-gold-po-b-independent-review-wo.json.gz)
确认5768为实际成功NumPy32逆算返回；64位路径未插桩，原counter为0不表示未逆算。

[PB原型实际结果](v1-results/clm-saturated-drainage-pb/manifest.json)已在EHPC
1606真实F0／Stageout0完成18项测试，pytest 56.76秒。它包含实际局部物理积分，
没有整柱、站点年或GPU运行。独立DOP853＋quad无冻土参考的迁移前沿差为
0.000277190m（0.022667%）、总出流偏差0.046005%、最大非零单层出流偏差
0.130059%（绝对差0.029632717kg/m²）；恒定给水度解析案例最大单层出流偏差
0.885763%。零供体层单独用绝对差检查。步长10000／2500／625秒时，独立
前沿误差依次为5.564010e−4／1.385353e−4／3.459872e−5m。

恒定给水度案例的水残差7.105427e−15kg/m²、热残差−1.862645e−9J/m²，分别
小于原界2.557954e−11／1.156825e−5；没有采用百分比守恒容差。首层滞水位
插值已与主水位首层特例分开，真实结果0.6956521739130435m、独立几何参考
0.6956521739130436m。[根代理原件复核及量化](v1-results/clm-saturated-drainage-pb-root-review.json.gz)。

**这些结果尚不构成CLM排水模块验收。** EHPC1610随后直接链接原固定CTSM
已编译模块，调用原水位与排水公开例程，真实F0／Stageout0。三个相同初态、
1800秒剖面分别执行官方及PB原型，共六次组件运行，没有整柱、年度或GPU运行。
原官方源和对象未改；调用程序只提供原输入、时钟及诊断，不重写官方物理。

| 实际剖面 | 官方出流 kg/m² | PB原型出流 kg/m² | PB−官方 |
|---|---:|---:|---:|
| 无冰饱和 | 0.5473170958734158 | 0.5462453722302109 | −0.0010717236432049582（−0.195814%） |
| 冰层阻力 | 0.0005564783437715881 | 0.00005595660489618981 | −0.0005005217388753983 kg/m² |
| 冻层上方滞水 | 0.00000020044345051805153 | 0.000010884537020951375 | +0.000010684093570433324 kg/m² |

后两项为微小出流，报告绝对差，不以相对误差卡近零量。实际滞水案例的官方
取水来自第三层，原型来自第二层；主排水的完整层冰权重及滞水相邻层导水阻力
也需修订。该结构差异不能用总量守恒或无冻土ODE通过抵消。官方完整水量残差
依次为3.408385e−14、−3.875231e−14、−1.540556e−15 kg/m²；原WA、QCHARGE、
地表水及其他出口均为0，三例的最小液水修正和超孔隙重分配均未激活。官方例程
不返回热出口；归档携焓量由实际逐层移出水量独立计算，不称为官方热输出。

[官方实际调用及两次失败原件](v1-results/clm-drainage-official-pb-reference/manifest.json)、
[逐项原输出核验](v1-results/clm-drainage-official-pb-root-review.json.gz)。1608因调用器
导入名错误、1609因缺少原ESMF时钟初始化失败，均真实F1／Stageout0；1610只修
调用环境后成功。下一步在新组件版本修正上述权重和供体语义，复用原三例真值，
补齐跨层及耗尽案例，再同步JAX／CUDA和整柱。诊断theta水位、局部Sy迁移前沿、
排水后重诊断水位保持分别输出；在跑冻结年度不更换物理或初态。

原1605因遗漏既有pytest依赖路径，在收集前失败（真实F1／Stageout0）；1606
只修执行环境，原组件和18测试字节不变。两次原日志均保留。原worker的诊断
计数0是行首解析漏掉pytest点号后的输出，归档直接从原stdout提取11条真实
记录，未重算数值或改写原receipt；原`model_runs=0`字段仅表示未跑整柱。

[完整作业终态、日/月/年数值、原源码与配置索引](v1-results/ctsm-vertical-hydrology-oc/manifest.json)。

当前已实现柱内垂向土壤水分输送，含显式零底通量 Richards 及携焓版本；**尚无独立含水层库存、预报地下水位及土柱—含水层双向交换**。该限制不能与本项目明确排除的柱间侧向地下水耦合混为一谈。原官方 R 本身也没有启用 aquifer 边界，不能以 WA 字段存在或冷态变化作为“缺少含水层造成偏差”的证据。

本次只读复核 12 个当前源文件，它们与冻结 NM584 逐字节相同；另核对原 KG3 和 NP12 批次实际引用的 15 份配置。索引为 `clm-vertical-groundwater-oc-index.json`，生成器仅用标准库读取文本、JSON 和哈希。没有更改源码、配置、观测、QC、科学门或当前年度轨迹，也未在 Windows 计算物理数组。

## 当前两条路径与真实库存

| 路径 | 实际做法及范围 |
|---|---|
| 未设置 `soil_hydrology` | `run_fluxnet_column.hydrology_configuration` 返回 None；柱调用 `drain_soil`，逐层向下搬运液水及其焓，受剩余孔隙和液水约束。底层出流实际离开土柱。 |
| 当前 KG3／NP12 的 VIC6 | 全部实际引用配置均未设置 `soil_hydrology`，`OptRunoffSurface=OptRunoffSubsurface=6`、`OptTileDrainage=0`。因此这些冻结年度用 VIC 入渗与 ARNO 底层出流，并未采用显式 Richards。ARNO 的液水“库”就是最底土层，非独立含水层；扣水与携热出口记入原总账。 |
| 显式 `richards_zero_flux` | NumPy、JAX、CUDA 实际调用路径已接通。土层内部液水可沿压力／重力梯度重新分配并携焓；水文底边界无液水及其平流焓通量，底出流置零。地表接纳溢流、雪径流与可显式启用的瓦片排水另记真实出口。要求地下径流选项3；不能在请求 ARNO 时静默去掉出口。这里不把水文边界等同于整个热传导模块的热底界。 |
| `groundwater_depth` 与 TOPMODEL | 深度是静态站点参数；简化 TOPMODEL 用它重新分配既有候选径流，不推进含水层水量或水位。`CLMColumnState.water_kgm2` 和原 native 水热总账只有土壤、雪与冠层库存。16 个基础状态及23个当前冷检查点字段中没有另一个含水层库存。 |

代码位置：`clm_enthalpy_driver.py:50,239,264`；`physics/soil_drainage.py:12`；`physics/vic.py:54`；`physics/richards.py:15,277`；`gpu/clm_richards_column.py:9,25`；`clm_state.py:44,276`；`benchmarks/run_fluxnet_native.py:85`。`richards.py` 开头“not yet wired”是过时说明，与实际调用和后续文档不符；本次未改冻结源码。`docs/KNOWN_LIMITATIONS.md:259,267` 已明确保留完整 CTSM 含水层／饱和带出流限制；其中其他更早的“未接柱”等文字不能当作当前实现状态。

## 原 R 的零底通量不等于没有地下径流

固定 CTSM commit `fdfc03a0abec838ec0b66663a6a2f5e619cd2611` 的源语义及 R/S 实际 `lnd_in` 一致：`lower_boundary_condition=2`、`soilwater_movement_method=1`、`baseflow_scalar=.001`。`SoilWaterMovementMod.F90:50–55,196–211,1354–1375` 将2定义为零底通量，`use_aquifer_layer()` 仅对3／4返回真。因此 R 实际关闭独立 aquifer，QCHARGE 不能从字段名称猜测。

无 aquifer 分支仍调用 `PerchedWaterTable`／`ThetaBasedWaterTable`，从土壤湿度诊断水位；随后调用 `PerchedLateralFlow`／`LateralFlowPowerLaw`。后者按局部地形、冰阻和水位计算饱和区出流，从水位以下至基岩底之间的**实际土壤液水层**扣除，受可移出水量限制；前者在具备条件时从冻层以上的滞水层扣水。它们是本地土柱参数化出口，没有在此审阅中要求实现相邻柱之间的地下水交换。源位置：`HydrologyNoDrainageMod.F90:211–229`、`HydrologyDrainageMod.F90:129–147`、`SoilHydrologyMod.F90:1747–1812,1865–1915,2074–2128,2219–2226`。

原 OA 读回的 WA 从冷文件4000 mm变为首 restart 5000 mm，不能解释为1000 mm降水或补给。`BalanceCheckMod.F90:143–153` 在相应水位条件下将 WA 设为基线；`clm_varcon.F90:128` 的基线是5000 mm。`TotalWaterAndHeatMod.F90:254–266,525–541` 用 **WA−5000** 计入相关总量，并说明此默认 CLM5 水库为不参与实际交换的虚拟基线。此次没有重建首次写入的逐调用轨迹，也没有把字段跳变加入水量收支。

## 原全年实际贡献

EHPC `1592.manager` 已实际 **F/Exit0**，PBS墙钟17秒、reader **3.3851491529494524秒**，模型运行次数0。读回采用原 R 封存件与固定 portable 环境，原17521存储行只排除启动 `[-1800,0]`，保留全部17520个正半小时区间；时间界与原气象输入逐区间绑定。365日、12月按原试验的2003本地年（UTC+8）分组，所有7个字段原掩码／非有限数均0，没有填补。完整归档 SHA 见 `ctsm-vertical-hydrology-oc-package.json`；数值来自 EHPC 读回 JSON。

| 实际陆面字段 | 全年 mm | 非零半小时数 |
|---|---:|---:|
| QDRAI | 125.39910458945087 | 1457 |
| QOVER | 122.19870647717673 | 3177 |
| QRUNOFF | 247.59781107155328 | 4495 |
| QCHARGE | 0 | 0 |
| QDRAI_PERCH／QH2OSFC／QRGWL | 各0 | 各0 |

QDRAI 占原全年 QRUNOFF 的 **50.64628966094203%**。它只在5月（64.70769935331191 mm，占当月径流68.1214039229464%）和6月（60.69140523613896 mm，占86.80952969631122%）非零，其余10个月为0；原313首48步为0的结论仍成立，但不能推广为全年。完整逐日／逐月累计和非零步数均在原结果 JSON。比例只是诊断贡献，不是模型相对误差；分母绝对累计≤0.05 mm时报告 null 及绝对量，不新增验收门。

## 结论与下一最小动作

已证实的区别是：R 有来自土壤饱和区的季节性参数化出口；当前 VIC6 从最底土层按 ARNO 退出；显式封闭 Richards 没有该饱和区出口。这项原参考年贡献不小，值得在已有全年结果完成后优先比较，**尚未证明它造成当前模型任何通量偏差，也未证明加入含水层会改善结果**。

对当前偏差的下一步诊断使用冻结年度已有输出，在原时间窗对照5–6月及其余月份的总径流、
土层水与 ET；先列明土质、土层／根区、气象和 SP/CN 差异，再判断库存响应。
当前原生年度的固定 TRACE 仅保存总 `runoff_mm`，没有逐步地表／地下拆分；
续算额外保存的是 BGC 子集。模型的全年分项因此仍未测，不能从总量或末库存
反演后冒充实际出流。原冷48步和首次续步中的分项不代表5–6月。
若未来需要分项诊断，应另行声明并验证实际捕获，不改变活动年度控制。
不得把同名 `runoff_deep` 自动当成相同机制。无需重跑 R/S48、换当前年配置
或扩展柱间侧向地下水。若未来确需独立垂向含水层，应另立物理库存、初态与
水热硬守恒合同，独立验证后开新轨迹；不能把新库存接入当前16状态／23字段冻结链。

本次没有追逐已满足 P0 的 OB 沉降末位差，没有新增科学通过信用。

1878既有数据首差分析实际Exit0（qstat状态E且无Stageout字段，不能写Stageout0），0模型/forward/GPU/JAX。stress、option2、two_bands的CUDA首差均在diagnosis_before：输入和phase_cells_before完全相同；stress/option2各5个初诊断字段先差，随后phase差最大1176/4648 J等效并放大到诊断8.84533e-5/3.46780e-4，接受步差1；two_bands初诊断5字段差、post-perched仅4项差、final phase exact。JAX三案与PB-t逐数组bit-exact。已有数据没有pow、heads或逐步adaptive轨迹，不能归因具体算子。证据见 v1-results/clm-drainage-pce-earliest-analysis/manifest.json。
