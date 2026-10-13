# K2：薄雪顶边耦合修复与连续年度检查

M阶段补充：生产物理保持本K2修复；本轮回退源全年度复验见
[GPU_DELTA_ATTEMPT](GPU_DELTA_ATTEMPT.md)。原H4不再独立决定合并，按
[预先固定M2](GPU_CLOSEOUT_CRITERIA.md)与真实数据门判断；整柱融尽热量缺口
经Wang明确接受为公开限制，不以表面闭合冒称整柱焓守恒。

普通版 NumPy 的第 3 天失败已经重新复现，证实是基线数值耦合缺陷，因而本次按
K0/K2 授权在普通版源头修复，并同步 JAX 与手写 CUDA。没有修改 STATE_SPEC、
求根阈值、0.5 W/m² 能量门或 parity 容差。本项可靠性通过也不能替代 K1/H4、
K3 或全量 pytest 合并门，更不代表真实物理已验证。

## 原始失败与最小复现

原始冻结快照 `/home/lhwang/work/noahmp-phase3b/k2-baseline` 来自开工 HEAD
`f716548`；没有修改旧 CPU N300000 监督器或其工作目录。
[本轮原始 NumPy 重放](gpu-results/k2-original-replay.json) 保存源码 SHA256、
失败日前 11 步的完整状态、强迫、能量分量及冻结 F(Ts) 的扫描。

| 零基 day2、column5 | 原始 NumPy 实测 |
|---|---:|
| step9 前 SWE | 0.0012875198030814809 m |
| step9 顶雪温 | 273.15 K |
| step9 G | −133.9752146124391 W/m² |
| step9 更新后雪温 | 182.0636305781052 K |
| step10 返回 Ts | 200 K |
| step10 残差 | −245.45365667557718 W/m² |
| step10 扫描 [200,330] 的变号区间数 | 0 |

土壤水、冰、SWE、雪层数和地下水位在该切片没有先行异常。根因是用旧雪温求得
的 G 在整个 1800 秒保持固定；单层雪热量更新退化为
`Tnew = Told + dt*G/(C_SNOW*depth)`。小热容量雪层被过冷约 91 K，下一步的根
便跑出原 200–330 K 保护区间。层间 Thomas 隐式并没有使这个顶边也隐式。

[最小单柱 fixture](../tests/fixtures/annual_thin_snow.json) 保留原 step9 的完整
70/33 槽状态及 step9–10 的两条强迫；对应回归直接运行普通版源头。
这不是仅 GPU 才会出现的 float32 问题，也不是首次降雪分层数组越界。

## 源头修复

原雪热系统可以写成 `A*Tnew = C*Told/dt + e0*G`。从底层开始消元，得到顶层
`Ttop_new = a + G/B`，其中 `a` 是零顶边通量时的顶层温度，`B>0` 为等效导热
响应系数。与表面导热 `G = K*(Ts-Ttop_new)` 联立，得到

`G = K/(1+K/B) * (Ts-a)`。

驱动在每步压实后计算一次 `(a, K/(1+K/B), K)`，F(Ts) 迭代和随后的热更新
使用同一个 G。对单层非融化情形，这使新雪温成为旧雪温与 Ts 的凸组合，消除
旧边界强迫造成的薄雪过冲。多层版本使用与原 `_implicit_heat_step` 相同的热容、
厚度、层间导热系数和底部零通量，不添加虚假厚度或热容量。

若未受融点约束的顶层温度会超过 273.15 K，则采用 `G = K*(Ts-273.15)`，并由
原热量/潜热更新处理相同 G 的融雪能量。原有多层相变投影和整层融空处理的简化
仍然存在；本次是数值稳定性修复，并非完整雪焓物理验证。

具体已知限制：`test_melted_out_step_keeps_its_actual_heat_flux` 的薄雪融尽切片
中，`G*dt = 259524.4193350902 J/m²`，全部雪的融化潜热只有
`36740 J/m²`，剩余 `222784.4193350902 J/m²` 仍未路由到土壤/融水热量。
这是原全融空处理已有的限制，当前没有在 K2 中重设计它。因此本文“能量门”
严格指表面 F 的 0.5 W/m² 闭合门，不能写成整柱焓能量已经守恒。

同时修正三后端融尽当步的 G 诊断：原逻辑用更新后的 SWE 再判断雪分支，可能把
本步实际进入雪热的 G 写成 0。现在按本步已执行的能量分支报告 G，因此诊断
`rnet-H-LE-G` 与已求得的 residual 一致。

### 新增回归的取证与修正

第一版直接消元绝对 Kelvin 温度，造成 `OptSurfaceDrag=2` 三层雪 step3 的
零通量参考温度在 CUDA/JAX 之间相差 1 ULP。相同输入下分别为
268.0032043457031 与 268.003173828125 K；Ts 相同，G/residual 差
2.6628375e-5 W/m²，原同精度门失败。该失败已保存，未放宽门限。

最终版本改为消元 `T-Told_top` 的温差 RHS，最后再加回旧顶层温度。均温雪的
RHS 严格为零，从而零通量参考温度严格等于原温度。这是改善消去误差的代数
等价写法；普通版、JAX、CUDA 都采用同一表达。最终雪边界与选项回归
**57 passed、0 failed、0 skipped**，涵盖 35 值、混排和双径流出口门。

## 最终冻结源码与验收范围

最终快照：`/home/lhwang/work/noahmp-phase3b/k2-final`。
原始、第一候选、最终快照的完整源码身份和取证文件校验见
[k2-source-identities.json](gpu-results/k2-source-identities.json)。

| 文件 | SHA256 |
|---|---|
| `driver.py` | `4d5008604fe12a513b6ab2510a0985ce2c4ef36ceafb9f3af165688872eebd09` |
| `physics/snow.py` | `e56b13921e5f138f216e8ead446e8ae66a2df4bc26d1c692c6e74dd31a7a49b2` |
| `gpu/jax_core.py` | `af3f9dd5c84970b58b771687fbe85b82ede196af2f1b0791bb08902669de20f0` |
| `gpu/cuda/column_kernel.cu` | `b514c3b883bbb5c1e2ce25e95b48fcb556bd1fbdc3091919a303164a098f459f` |

NumPy 使用 [check_annual_numpy.py](../benchmarks/check_annual_numpy.py)，逐步
检查所有完整状态与诊断，保存每个日块的 49 个状态截面、48 个强迫/诊断。
GPU 使用 [check_annual_gpu.py](../benchmarks/check_annual_gpu.py)，调用原生产
积分器，每日读取全部状态及全部半小时诊断，在主机应用原 extended gate。
GPU 每日块内部的全层状态没有全部读回，不能宣称逐中间步的全层物理边界都已检查。

两类脚本沿用原八柱、seed20261013、每日48步和连续 carry，不重置、不缩步、
不丢弃失败日；检查 NaN、土壤水上下界、冰量界、非负雪水/冠层水、有效雪层数、
0.5 W/m² 能量闭合，另记录地下水位/水量余额。水量余额是实测报告，没有新增
接受阈值。GPU 日回读方式只用于正确性审计，不计入正式加速比或替换旧基准。

最终 NumPy 已完成 365 天：逐步最大 residual 为 0.0009999234141275792 W/m²，
最低有效雪温 253.0738680631838 K，地下水位深度最大 2 m，累计净水量误差绝对
最大 9.936184220423036e-9 m。最终 CUDA 已完成 365 天：最大 residual 为
0.00720977783203125 W/m²，每日块末有效雪温最低 255.3595428466797 K，累计
净水量误差绝对最大 6.691545038487422e-5 m。最终JAX64和JAX32也已分别于
08:47/08:48 UTC完成365天；**K2规定的年度可靠性门通过**。第一版曾完成
NumPy/CUDA 全年，但随后发现上述同精度回归，故最终源码独立重跑四后端；旧
candidate 的两项 JAX 年度被明确终止，不能作为全年结果。

| 最终后端，8柱连续17520步 | 完成天数 | 最大\|residual\| W/m² | 累计净水量误差最大绝对值 m | 原年度门 |
|---|---:|---:|---:|---|
| NumPy64 | 365 | .000999923414128 | 9.93618422e-9 | 通过 |
| JAX64 | 365 | .000999923414071 | 9.93617872e-9 | 通过 |
| JAX32 | 365 | .007209777832031 | 6.69150262e-5 | 通过 |
| CUDA32 | 365 | .007209777832031 | 6.69154504e-5 | 通过 |

最终逐日原始日志：[NumPy](gpu-results/k2-final-numpy-365.jsonl)、
[JAX64](gpu-results/k2-final-jax64-365.jsonl)、[JAX32](gpu-results/k2-final-jax32-365.jsonl)、
[CUDA](gpu-results/k2-final-cuda-365.jsonl)。这些结果是稳定性检查，**不是年度
跨后端逐值parity或整体焓守恒通过**；检查范围及既有融尽能量缺口如上。

同一生产源码全量pytest为 **147 passed / 2 failed / 0 skipped**，两项失败仍是
原NumPy64 H4的1/48步，不能因K2通过而忽略。H2与CUDA/JAX32同精度回归通过，
修后默认48步residual的CUDA/JAX32最大差为2.98e-8，不再声称修后全部逐位一致。
[完整pytest日志](gpu-results/k-final-pytest.log)及[快照核验](gpu-results/k-final-pytest-metadata.json)
记录91个归档文件与测试后文件一致、实际CUDA强制标志开启及非零退出码。
最终kernel另用nvcc、sm_89及生产编译选项编译成功；
[日志](gpu-results/k-final-nvcc.log)有1条旧`snow_flux`辅助函数未使用警告，没有
抑制警告或称零warning。Windows快照的build.sh为CRLF，直接bash启动失败后
使用该脚本中相同的显式nvcc命令完成编译，未改生产源码。
全量快照之后仅FLUXNET runner日志/QC显示与域取证图表仪器有改动；最终loader
已另测[7 passed](gpu-results/k-final-fluxnet-loader.log)，生产物理哈希仍与全量快照一致。

[K3 全域取证](GPU_DOMAIN_FIX.md) 的独立复验表明，本修复把全域最大 Ts
后端差值从 2.34994 K 降至 0.0747771 K，原 seasonal_loam 雪面大偏差尾部消失。
但全部 5565 列有效状态超差及其冰量误差仍存在，主要涉及无雪砂土 AHC 灵敏度；
因此 K2 不是全部 K3 误差的共同根因。

复现命令（新输出文件名；用实际物理卡号）：

```bash
cd /home/lhwang/work/noahmp-phase3b/k2-final
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/lhwang/venvs/noahmp-phase3b/bin/python benchmarks/check_annual_numpy.py \
  --days 365 --output outputs/recheck-numpy365.jsonl
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
  JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
  /home/lhwang/venvs/noahmp-phase3b/bin/python benchmarks/check_annual_gpu.py \
  --backend jax64 --days 365 --output outputs/recheck-jax64-365.jsonl
```

GPU 后端参数另取 `jax32` 或 `cuda`。每份日志开头记录完整源码 hash，日记录包含
强迫 hash、每日 dump hash 与检查结果。完整 dump 留在上述冻结目录的
`outputs/k2-final-*.states/`，原始失败 JSON 和正式年度日志同步到仓库结果目录。
