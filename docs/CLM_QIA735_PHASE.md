# Qia735：已有冠层湿层的相变端点

15:18后续：新CUDA E已物理完成48步；[原父退出1及正确只读审计](v1-results/qia-all-phase-cuda-day-e/manifest.json)
确认父脚本误把无影子26字段与含影子的48字段集作整体相等断言。26个原冷
主状态/索引/账目字段逐值相同，额外22字段全部为参考JAX影子，原守恒和模型
状态running/48未失败。正确审计实际0，原E错误未修改。
[F同源续跑](v1-results/qia-all-phase-cuda-year-launch-f/manifest.json)只复制这段
自身有效前缀，所有输入/状态/轨迹/账目逐字节一致；15:14快照running/96，
worker3414721/controller3414720仍在。无JAX影子、同精度null，不计完整年或
当前K78/完整回归通过。F继承的scope字符串中“error/audit1”有误，归档已明确
更正：E父进程1，正确审计0，没有另一个失败的E数组审计。

2026-10-19 14:58UTC。CPU计算与独立分析在EHPC PBS计算节点，GPU在xuanze；
本机只编辑、检查语法/哈希、整理归档。完整气候年、spinup和科学验收仍未完成。

## 原失败与数值修订

[原510 NumPy全年终态](v1-results/qia-fresh-phase-numpy-terminal-b/manifest.json)
为failed/735/17520，最大相变残差0.00010874183 W/m²，超过原1e-7/80次。
原轨迹、因果暖初猜、旧水热储量及失败记录保留，没有续写失败状态。

原方案只允许旧质量为零的新凝结水试算精确相变端点；已有微量冰水的潜热平台
也会使二分回溯接近端点但进展缓慢。修订沿原Newton方向试算遇到的第一个精确
相变端点，推广到全部相变活跃层，只有完整耦合残差严格下降才接纳。
原物理方程、旧储量、数值残差、迭代上限和守恒界均不变。NumPy先完成参考与
测试后才同步JAX/CUDA；没有采用M1全量土壤ΔT方案。

[EHPC254/255/256原态与参考](v1-results/qia735-numpy-probes-a/manifest.json)
保留原整步失败和F修订成功。修订整步水/热/C/N残差分别为−6.77548e-14 kg/m²、
+3.84025043e-8 J/m²、+7.11292e-13 gC/m²、−4.65157e-16 gN/m²；原界分别为
8.04841e-12、1.45253112e-5、8.57550e-10、3.63348e-11，全部满足。
独立SciPy原方程参考的最大温差5.68434e-14 K，H/LE绝对差1.25588e-12/
3.49942e-13 W/m²；修订23次收敛。原255基础GPU打包器拒绝MOST扩展的失败
也保留；256明确分开打包基础字段与原MOST扩展，没有改动输入值。

[EHPC257 NumPy先行检查](v1-results/all-phase-numpy-target-a/manifest.json)
确认原实现的新735回归预期失败，新实现26项相变测试实际退出0。
新夹具`qia_735_phase_oldwet.json`的SHA256为
`93bf18fcbfba472509ee03a32681a5746c57190e4decac75697f0af7d521ffc6`，
量化32/64位的NumPy参考也均由EHPC计算。

## 新增设备测试的原失败与修正

[A版设备测试](v1-results/all-phase-gpu-target-a/manifest.json)实际10通过/1失败，
不是完整通过：新735测试原始300字段有57项超过rtol=atol1e-6，最大差
0.182373046875 J/m²。两项32/64位独立参考比较已过，不能抵充该同精度失败。

[原捕获及全部调用方式对照](v1-results/qia735-device-staging-abc/manifest.json)
与[EHPC261独立分析](v1-results/qia735-staging-analysis-c/manifest.json)表明，
新增测试把叶参数与MOST扩展捕获为JAX编译常量；实际生产入口按动态设备数组
传入。保持九组原输入、参数数值、物理、精度和所有容差不变，按生产方式传参，
300原始字段与CUDA逐值相同。显式使用同一辐射仍复现常量路径57项失败；
动态参数路径在内部/显式辐射两种情况下均为0，584个原追踪辐射字段也逐值相同。
这证明本次调用方式差异；没有据此臆测某条底层机器指令或分别归因某个参数。
捕获A额外辐射展示的形状错误和B另算的未追踪辐射均标明，不混作原追踪路径。

B仅修改新增测试的参数绑定，全部生产文件、CPU测试及夹具与A逐字节一致。
[B版11项设备目标](v1-results/all-phase-gpu-target-b/manifest.json)在xuanze实际
worker/controller退出0，0失败/跳过，pytest 167.288秒；setup/call/teardown、
设备同步、精确节点和源码未变均有回执。735全部300原始字段最大差0，整数精确。
JAX32相对EHPC量化参考最大叶温差2.73907e-5 K、H/LE差4.10080e-5/
0.000307560 W/m²，水残差6.98492e-10 kg/m²、水相焓账2.86102e-6 J/m²。
这些测试沿用已披露的解释器atexit绕过，只在全部测试阶段/同步/回执后退出，
不代表验证了JAX解释器关闭行为。

## 最终源码与正在执行的验证

[A全CPU911](v1-results/all-phase-cpu-final-a/manifest.json)和
[最终B全CPU911](v1-results/all-phase-cpu-final-b/manifest.json)分别由EHPC258/262
实际完成，均0失败/跳过；B用时460.724秒，未插桩735整步也满足原硬界。
最终B共511文件，清单SHA256为
`37187bea427930cdf1772c5db1d07b679c7dda15d8033e85e5a013a06089517f`；
三后端生产函数/完整CUDA词元与独立3798候选F一致，详见[3798配对](CLM_QIA3798_PHASE.md)。
[源码/控制程序及节点核对](v1-results/all-phase-source-control-ab/manifest.json)另保存
A/B完整清单和构建过程：当前498份运行文件与B冻结包逐字节相同，全部1254收集
节点中的911 CPU与11相变GPU逐项匹配XML，不计旧源码结果。

[实际完整收集](v1-results/all-phase-full-deferred-b/manifest.json)为1254项：
911 CPU、343 GPU；同源911与相变11已完成，其余332排队等待旧506工作进程
实际退出。仅核实后替换了尚无子进程的旧510等待器，原模型工作进程未终止。
收集和排队不算完整执行。

30分钟单点阻塞已[记录并绕行](GPU_BLOCKERS.md)。
[新NumPy年](v1-results/qia-all-phase-numpy-year-launch-c/manifest.json)在EHPC260
从原冷种子启动；14:52不可变快照running/144/17520，累计热残差
0.000915885961 J/m²。它使用A生产源码，与最终B生产完全相同，差别只有GPU测试。
[新CUDA年](v1-results/qia-all-phase-cuda-year-launch-e/manifest.json)14:55在xuanze
GPU0启动，最终B源码、原冷种子；没有JAX影子，同精度明确为null，不能抵充
完整逐步同精度验收。原735/3798失败年都未续写。

当前K78原始字段同精度仍有1.52588e-5 J/m²差异，未豁免。完整1254、气候年、
spinup、多站多年/留出与官方同条件科学验收、合并main及v1.0.0发布仍待。
