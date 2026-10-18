# 官方CTSM单柱参考准备（2026-10-17）

目前EHPC上的官方完整构建和动态库检查已完成，尚无单柱输出或与本
模型的偏差数字。这条参考链补充已有HRLDAS与FLUXNET，不替代二者或将环境
自检算成科学验收。实际CPU构建/模型运行安排在EHPC PBS计算节点。

## 实际Qia气象序列准备（2026-10-18）

EHPC PBS127 compute002已将锁定Qia17520时段序列化为官方DATM字段并重新打开，
总13.462687秒；气温、比湿、风、气压、下行短/长波六字段差0，降水由m/s
转mm/s后还原最大差8.47033e-22m/s。保持原强迫/配置哈希与39.6m温/湿/风
高度；当地2003年00:00起点按固定UTC+8转为2002-12-31 16:00UTC，原半小时
边界与中点完整保留，没有编造边界外天气或沿用Brazil的null气象流。
[执行源码、原字段/单位/时间/哈希和实际回读数字](v1-results/ctsm-qia-meteorology-a/manifest.json)
保留，实际NetCDF留在服务器。原CO₂ 372.93–376.54ppm作为明确的辅助字段保留，
尚未把它冒充已经接通的官方CO₂流。匹配域/表面/PFT/土壤/初态、气溶胶及CO₂
映射、实际DATM时间对齐检查和官方模型运行仍待完成，不能作为官方输出验收。

## 预检与绕行结果

EHPC PBS106 compute002实际预检：系统GCC/GFortran4.8.5可用，但Git、CMake、
NetCDF-C/Fortran配置工具和MPI包装器缺失；计算节点对GitHub、raw及NCAR输入
数据域名均DNS失败。home仅余283 inode，节点/tmp有约75 GB及619万空闲inode。
因此使用节点临时盘解包、构建和运行，只将报告/必要结果压缩存回home。
原失败及[EHPC预检](v1-results/leaf-pressure-ctsm-preflight-a/manifest.json)完整保留。

在联网机器仅准备依赖，显式设置目标glibc2.17，锁定87个conda-forge包，下载
共288121207字节；安装后逐包SHA256核对，打包为427299473字节。没有改公共
Anaconda、系统编译器或原物理运行环境。
[精确包计划](v1-results/ctsm-reference-dependency-plan-a/manifest.json)与
[实际安装/打包及来源记录](v1-results/ctsm-sources-portable-a/manifest.json)分开保存。
工具链压缩包SHA256为`8f1d5ba12ed58a68e18293c182991d84cacbccd53f7dd8c86e06f687ea719769`，
二进制包留在服务器，不加入本仓库。

EHPC PBS108 compute002已实际完成：节点临时盘安全解包、重定位，GCC/GFortran、
Git、CMake、nc-config/nf-config返回0；用新编译器编译C程序写NetCDF，再由
Fortran读入并写另一份文件，Python交叉读取1.25/2.5及277.5/3.0完全一致。
18个命令阶段均退出0，包含解包的总耗时12.333732秒。
[源测试程序、全部日志/动态库与节点结果](v1-results/ctsm-portable-smoke-a/manifest.json)保留。
这解除该依赖组合在EHPC运行的障碍，不表示CTSM源已成功编译。

## 固定上游来源

| 来源 | 锁定提交 | 用途 |
| --- | --- | --- |
| [CTSM release-clm5.0](https://github.com/ESCOMP/CTSM/tree/fdfc03a0abec838ec0b66663a6a2f5e619cd2611) | `fdfc03a0abec838ec0b66663a6a2f5e619cd2611` | CLM5官方参考源码 |
| [CIME cime5.6.47](https://github.com/ESMCI/cime/tree/d8e9b894d7eb96fe96f8f8a6943b421147f94606) | `d8e9b894d7eb96fe96f8f8a6943b421147f94606` | 上游指定构建/离线驱动框架 |
| [FATES指定版本](https://github.com/NGEET/fates/tree/3f6749d471e8766a212dd0a2b0c4c95aacd97f65) | `3f6749d471e8766a212dd0a2b0c4c95aacd97f65` | 上游要求的源码依赖，不新增Python物理功能 |

原Git fetch实际连接超时并终止，退出记录/错误保存。改用相同提交的官方
GitHub源码归档：CTSM/CIME/FATES三包SHA256分别为
`75f386ad3141760b0efe1e05603f2e6e37d36ae92f1ee74dd63f066aaa0f6372`、
`342f0bec6870d00a3e9b03e226cdf99d799c5ce4e7164899cf7f9eebf7121af0`、
`eb9d606155df6d28b94d873cbfef520541bca342887745f88eb2cf546f3541a6`。
归档检出不含.git历史，后续不得把本地合成提交冒充原上游提交。

离线布局包保留2930份上游普通文件的原字节/哈希和22个构建链接，放入指定
CTSM/CIME/FATES路径，没有更改Fortran物理。计划用stub runoff/glacier做
单柱参考；河道、整机耦合或FATES开发不进入本项目范围。官方说明支持
[单点、单处理器配置](https://escomp.github.io/CTSM/release-clm5.0/users_guide/running-single-points/running-single-point-configurations.html)，
当前仍待机器构建配置、输入数据与强迫映射、实际编译和连续运行。

EHPC PBS110随后完成该布局2930份原文件哈希及CIME入口核对，14.202345秒，
源码未变。PBS109两条CLI本已正常返回，但诊断脚本用了新版CIME目录路径，
收集元数据时FileNotFoundError；修订B仅更正路径，原失败保留。
[两轮节点/日志与完整上游哈希锁](v1-results/ctsm-source-preflight-ab/manifest.json)另列。
单柱SP、stub runoff/glacier case已能创建，机器环境适配及输入预览仍在核查；
没有把case创建当作模型构建成功或站点科学结果。

后续PBS111/112分别暴露模板modulecmd不存在、请求无限栈超出PBS硬限。
仅改新生成case的模块设置/私有NetCDF路径及资源请求上限，不改上游源码或
系统限额。PBS113的case.setup实际返回0，2930份原文件仍相同；但
preview_namelists因缺少Perl XML::LibXML返回1，该轮总体退出仍为1。
[三轮配置、原错误与适配记录](v1-results/ctsm-case-config-abc/manifest.json)保留。
PBS114另证实系统Perl5.16.3也缺LibXML/Reader，实际返回码0/2/2；
[原可用性记录](v1-results/ctsm-system-perl-a/manifest.json)没有把缺失模块算通过。
随后用另一份工具链补齐该依赖，原87包压缩环境不覆盖。

## XML依赖与实际构建

CF单渠道的首次补包计划因缺少perl-xml-libxml退出1，原失败保留；增加Bioconda
后成功解出111包，保留显式URL/MD5及每个下载包SHA256。仅Perl从5.44改为
5.32.1并增加24个Perl依赖，原其余86包相同；原环境/压缩包仍保留。
新环境包432276606字节，SHA256为
`6c535c3391287f26e49ec1089bc71c255651ab0de97983c1dc5c2b6a1e978a20`。
[两次求解与实际安装/打包](v1-results/ctsm-portable-b-a/manifest.json)保留，
没有将已缓存包遗漏于111包精确锁。

EHPC PBS115 compute002对该包重做安全解包/重定位、Perl LibXML/Reader与实际
XML属性解析、C/Fortran编译及NetCDF双向精确读取，19阶段均退出0，12.581721秒。
[程序、原输出与节点记录](v1-results/ctsm-portable-smoke-b/manifest.json)保留。
此项只验证构建依赖，不是官方模型或科学通过。

PBS116在同一节点首次完成case.setup及preview_namelists（均0），开始真实
CTSM/CIME构建；mpi-serial已编译，GPTL在perf_utils.F90外部MPI_BCAST的
INTEGER/LOGICAL调用处报类型不匹配，case.build退出1（7.016649秒）。全部
2930份上游文件仍相同；[实际构建失败与私有原工作区哈希](v1-results/ctsm-case-build-d/manifest.json)
保留，不能称为完整构建成功。

PBS117只在新生成case的Macros.make/Macros.cmake追加
`-fallow-argument-mismatch`后重新构建，未修改上游物理或精度/优化选项。
此选项的作用是把外部过程参数类型不一致的编译错误降为警告，适用范围及
限制见[GNU Fortran官方说明](https://gcc.gnu.org/onlinedocs/gfortran/Fortran-Dialect-Options.html)。
PBS117的共享库构建完成后，在FireEmisFactorsMod.F90:230的整型参数
`z'000053db'`报BOZ常量语法错误，case.build退出1、109.164467秒，原件见
[该轮原日志/编译适配与私有工作区哈希](v1-results/ctsm-case-build-e/manifest.json)。
PBS118再仅追加`-fallow-invalid-boz`（同一GNU官方说明列明的旧语法兼容开关）
重试，仍未改源、精度或优化设置。当前仍待终态，不用已完成共享库或CLM构建
脚本的局部打印替代最终构建退出码和可执行文件。

PBS118实际159.742834秒编译到最终可执行文件链接，未解析的dgtsv/dgesv/dgbsv
来自LAPACK，整体仍退出1，源未变；[该轮原错误](v1-results/ctsm-case-build-f/manifest.json)
保留。PBS120在新case中追加私有现有LAPACK/BLAS链接，后续仅实际构建0、唯一
cesm.exe哈希及动态库全部可解析时才记构建完成；仍不等同运行/科学通过。

PBS120的显式SLIBS同时抑制了旧CIME的默认NetCDF回退，最终仍退出1，
[原错误/生成case记录](v1-results/ctsm-case-build-g/manifest.json)保留。
PBS122改为显式使用已验证`nf-config --flibs`的完整NetCDF链接项再加LAPACK/
BLAS，9阶段均返回0；case.build160.828343秒，总计178.391814秒。
唯一cesm.exe为25779696字节，SHA256为
`12911b249bb40ed9259d16dbeea3ff627808ecd6f9b7ab3345307d13c735ddd6`，
ldd返回0且没有缺失共享库，2930份上游源仍原字节。
[完整实际构建、适配、链接输出及私有工作区绑定](v1-results/ctsm-case-build-h/manifest.json)
保留；可执行文件留服务器，没有将它加入Python模型或公共仓库。
构建完成不勾选模型运行、有效强迫、spinup或配对科学验收。

1x1_brazil/SP配置目前用于构建链检查；DATM1PT强迫尚有null文件项，不能作为
可运行气候驱动。后续需要有效单点气象/输入数据，并按目标站点建立共同
边界后才配对；此配置不冒充四站匹配参考，也未产生官方物理输出。

EHPC PBS127后续已用原CN-Qia 2003年17520个半小时间隔序列化单点气象，
[实际节点、变量/单位映射及逐值回读](v1-results/ctsm-qia-meteorology-a/manifest.json)
保留。六字段回读最大差0；降水m/s到mm/s乘1000，往返差8.47e−22。
气温/比湿/风速均39.6m，未按评分挑测高；原UTC+8完整年对应UTC
2002-12-31T16至2003-12-31T16，CF时间为区间中点并保留边界，不补虚构小时。
文件SHA256 `34ed3f0128820575e78a5faa2ed8373912d9bbd6ceae23dd555db8b7df9a8f74`。
CO2原372.93–376.54ppm只作辅助变量保留，尚未声称DATM已实际读取。

[固定上游DATM合同及生成case核对](v1-results/ctsm-forcing-contract-a/manifest.json)
确认温度/比湿/风/气压/短长波/降水/高度分别使用TBOT/QBOT/WIND/PSRF/
FSDS/FLDS/PRECTmms/ZBOT对应的tbot/shum/wind/pbot/swdn/lwdn/precn/z。
这只是只读配置证据；匹配的domain/地表PFT/土壤/初态、气溶胶/CO2实际映射及
DATM日界对齐仍须准备并真正运行，不能把Brazil静态输入当作Qia匹配参考。

上游指南的Brazil示例使用I2000Clm50SpGs；CLM1PT的自带强迫示例另用于特定
单点，并要求按实际强迫时期设置起止日期。后续参考将显式准备目标站点强迫
流和共同输入边界，不把当前null项扩展成常值天气。依据见
[CLM5单点运行说明](https://escomp.github.io/CTSM/release-clm5.0/users_guide/running-single-points/running-single-point-configurations.html)。

对PBS116真实输入清单逐一做HEAD，10份官方文件均200；按每文件不超过10MiB
先暂存其中7份、2738092字节，保留原ETag/长度/NetCDF签名/SHA256，原数据
仅在服务器。其余气溶胶/初态/urban默认文件为6205649532/693632420/343955156
字节，尚未暂存；urban只是上游默认文件项，本项目没有新增城市物理。
[网络/下载原记录](v1-results/ctsm-input-staging-a/manifest.json)完整保留。
EHPC PBS119 compute002在节点临时盘按原哈希打开全部7份NetCDF、读取维度/
变量/单点土壤和PFT等元数据，13.085429秒，未改原数据。
[实际节点与元数据](v1-results/ctsm-input-inspection-a/manifest.json)保留。
这不是完整输入集；两条null气象项仍在，尚无可运行的官方气候参考。

数据/初态、PFT/土壤层、测高、光学/气孔约定及spinup需在配对之前明列；
每日/月ET/H/LE/GPP/NEE/土壤温湿/径流/SWE及近零门仍按原验收规范，原水热
C/N独立账不能由环境自检豁免。只有真正输出后才能报告模型−官方偏差。
