# NM 完整 GPU 回归终态（QB）

2026-10-26 07:50:11 UTC 实际只读核验：CUDA1 原完整回归已经结束，
72 个新模块共 376 项：367 通过、9 失败、零 error/skip；另 32 项原专项的
来源及依赖已核验，共 **399/408**。69 个新模块全部通过，3 个模块失败。
原控制器记录耗时 65290.594 秒（18.136 小时），原 worker/parent 回执与
预期节点、XML 均保留；detached controller 的操作系统退出码没有捕获，记 null。
584 份源码执行前后完全一致，没有全量通过或科学验收信用。
[448 件原始记录](v1-results/nq-terminal-qb/manifest.json)。

| 失败模块 | 实际结果 |
|---|---|
| MOST stagnation | 3 项中 2 失败：要求 bisection_restarts=1，实际 0；float64 reference 项通过。float32 的 H/LE/GPP/NEE 物理断言在计数断言前已经通过；同精度项在计数断言处停止，后续同精度和硬账尚未执行，不能推断通过。 |
| CLM Richards column | 5 项中 4 失败；GPP 为 .0084934/.00849366，对历史参考 .00587198；CUDA32/JAX32 的 phase reconstruction residual 为 1 J/0 J，同精度失败保留。 |
| FLUXNET Richards runner | 3 项均失败；原四点 H/LE 最大相对差分别 18.089462%/12.552835%（另后端 12.553396%）。 |

历史 Richards NPZ 实际来自 EHPC383/BF，早于 NM 的土壤水分胁迫及
Ball–Berry 修订。这证明参考来源发生过物理版本变化，不能据此断言 GPU 正确，
也不能直接刷新金样。原 pytest-460 三后端 whole4/split2+2 的 78 份输入、
配置、状态和轨迹另已完整取回：[86 件保真包](v1-results/nq-station-retained-qb/manifest.json)。
当前 NumPy 对原保存输入的 44 次独立重算已在 EHPC1827 真实F0/Stageout0，
44/44返回、113秒，584源和原输入不变，原硬账执行。
[408件参考原件](v1-results/nq-current-reference-qc/manifest.json)的408为证据件数，
不改变原GPU399/408终态。三份原站点GPU对当前NumPy全部满足原物理门：
CUDA32最大H/LE绝对差 .0006629078/.0014292167 W/m²，JAX64为
5.80434e−8/2.14829e−8 W/m²，按实际dtype重建的原初态逐字节相同。
这支持旧参考版本差的诊断，不能代替观测验证或先行刷新fixture。
所有旧失败保留，未改变模型、fixture、1e-6、整数或原守恒门。

现有 legacy 两包各 102 字段 CUDA/JAX 逐字节相同；aggregate 包 126 字段中
117 相同、9 不同，包含土壤状态、输送和相变残差；EHPC1827数值分析确认
其中8差在原1e−6门内，唯一越门仍是phase reconstruction residual的1 J。
MOST 原计数语义和相变交叉重放分别独立调查，不把两个原因混为一谈。

[QF 188件实际交叉证据](v1-results/nq-phase-cross-qf/manifest.json)：CUDA1/JAX
各两次原inverse全返回、parent均0；EHPC1842实际F0/Stageout0。两套原末态分别
给两个算子，全部9字段/计数逐字节相同，且原对角线结果完全重现：CUDA原末态
在两算子均得[1,0,0,0] J，JAX原末态均得[0,0,0,0] J。这证明本案原差来自
更早输送产生的不同输入，原1J列同精度失败仍未修，不删除诊断或改变门。

[QG 101件原值证据](v1-results/nq-reference-values-qg/manifest.json)：1841实际
F0/Stageout0、0新物理；五层GPP32第一步BF .00587197917→当前NumPy
.00849365748，64为.00587198031→.00849365909。404个expected来自当前
NumPy，250个原输入及站点12个input/config/meta数组均逐字节保留；拟fixture
仅在私有结果，checkout未改。新QH私有测试语义候选尚未执行或推广，不能提前
把原9失败改计为通过。CPU原MOST5节点依赖与NR1585已核验同源，完整GPU原
3节点仍须真正设备执行及原门独立比较。

2026-10-26 15:34 UTC，实际 nvidia-smi 显示物理 CUDA1
`GPU-cf7322ad-e2e0-ddd3-76d7-a07e2dd73452` 利用率 0%，空闲 22210 MiB。
完整回归旧剩余估时已退役；完整年仍受 7275 步冠层异常阻塞，没有可靠剩余时间。
短排水 PB-t 使用独立路径和缓存，不能作为全量回归或年度完成信用。
