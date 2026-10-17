# 官方CTSM单柱参考准备（2026-10-17）

目前完成的是离线源码和可移植构建依赖，尚无官方CTSM编译/单柱输出或与本
模型的偏差数字。这条参考链补充已有HRLDAS与FLUXNET，不替代二者或将环境
自检算成科学验收。实际CPU构建/模型运行安排在EHPC PBS计算节点。

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
当前正在为独立工具链补齐该依赖，原87包压缩环境不覆盖。

1x1_brazil/SP配置目前用于构建链检查；DATM1PT强迫尚有null文件项，不能作为
可运行气候驱动。后续需要有效单点气象/输入数据，并按目标站点建立共同
边界后才配对；此配置不冒充四站匹配参考，也未产生官方物理输出。

数据/初态、PFT/土壤层、测高、光学/气孔约定及spinup需在配对之前明列；
每日/月ET/H/LE/GPP/NEE/土壤温湿/径流/SWE及近零门仍按原验收规范，原水热
C/N独立账不能由环境自检豁免。只有真正输出后才能报告模型−官方偏差。
