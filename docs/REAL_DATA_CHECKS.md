# M4 真实数据收尾检查

**M4 数值门仍未通过：四站 canw 的同量化输入 M2 合格率为 99.201389%，低于预注册的 99.5%；
CN-Cha 单站为 97.291667%。已经全部归因也不将此硬门改成通过。**
JAX64 parity、CUDA/JAX32 同精度、其余八类状态/通量、residual 和表面/水平衡通过；
ERA5 因必需 forcing 不完整为“未测”。Wang收到完整结果后明确接受上述差异并验收，
授权继续合并/发布；[验收记录](GPU_ACCEPTANCE.md)与自动数值判定分开保存。

本轮只检验真实数据读取、连续运行和后端数值一致性；不调参、不校准、不宣称物理验证。
验收口径固定为 [M2 预注册标准](GPU_CLOSEOUT_CRITERIA.md)，原 K6/H4 的严格红灯保留。

## FLUXNET：四站各 30 天

直接从 NAS WLH 的 `/volume2/Data/Global/FLUXNET-Data-Shop/processed/site_csv/`
只读取得处理后 CSV，未回退到 raw。全文件 SHA256、大小和抽取片段 SHA256 见
[来源清单](gpu-results/m4-fluxnet-sources.json)。抽取仅保留原 header 与原行字节，
没有改值、插值或补行；原观测和完整轨迹保留在忽略目录 `outputs/m4/` 及远端私有目录。

| 站点 | 固定窗口（当地标准时间，右端不含） | 步数 | LE QC=0 样本 | H QC=0 样本 |
|---|---|---:|---:|---:|
| CN-Cng | 2007-07-01 至 2007-07-31 | 1440 | 1360 | 1414 |
| CN-HaM | 2007-07-01 至 2007-07-31 | 1440 | 0 | 971 |
| CN-Qia | 2003-07-01 至 2003-07-31 | 1440 | 1393 | 1406 |
| CN-Cha | 2003-07-01 至 2003-07-31 | 1440 | 982 | 1099 |

窗口在模型运行前固定。最初列入的 CN-Du2 在处理后文件中没有 2007 年 7 月行，
所以没有运行它；改选目录可达的 CN-Qia，按目录首年取 2003 年 7 月。
CN-HaM 的 LE 没有 QC=0 实测，保留该缺测结论，并增加同样取目录首年 7 月的
CN-Cha，使三站具有可算的 LE/H 观测指标。没有根据模型拟合优劣挑选日期或站点。

读取沿用 [fluxnet_smoke.py](../examples/fluxnet_smoke.py) 的列名、单位、时间和 QC
规则，全部四站通过 1440 个连续半小时 forcing 检查。未做额外 gapfill；产品原有
gapfill/ERA forcing 的 QC 全部计数。例如 CN-HaM 的 PA 和 P 全部为 QC=2（ERA），
不能称为全观测强迫。LE/H 只使用各自有限且 QC=0 的未闭合校正通量。

所有站点统一使用未改动的 `ColumnState` 默认参数（loam、LAI=3、fveg=0.8、
z0m=0.05 m、初始 285 K），没有 spinup、站点率定或塔高修正。
观测不进入模拟参数。bias=模型−观测；无有效观测时 n=0、其余指标为 null。

NumPy64 与观测的实际描述指标如下，bias/RMSE 单位均为 W/m²。
这份短期、默认参数、无 spinup 的表不能宣称物理验证通过。

| 站点 | 通量 | n | r | bias | RMSE |
|---|---|---:|---:|---:|---:|
| CN-Cng | LE | 1360 | 0.869342 | 0.452131 | 47.681827 |
| CN-Cng | H | 1414 | 0.897373 | 6.490629 | 35.385035 |
| CN-HaM | LE | 0 | — | — | — |
| CN-HaM | H | 971 | 0.238231 | −22.542932 | 56.526686 |
| CN-Qia | LE | 1393 | 0.838350 | −40.194766 | 79.236683 |
| CN-Qia | H | 1406 | 0.890822 | 10.476968 | 46.739421 |
| CN-Cha | LE | 982 | 0.858376 | 8.408103 | 63.318907 |
| CN-Cha | H | 1099 | 0.703453 | −6.177463 | 42.226466 |

全部四后端的完整 r/bias/RMSE 保存在 [运行报告](gpu-results/m4-fluxnet.json)。
JAX64 相对上表最大变化仅 r=1.05e-15、bias=2.05e-13 W/m²；JAX32/CUDA
最大变化分别为 r=4.37e-6、bias=0.001001 W/m²、RMSE=0.000636 W/m²。
这些小幅后端差异不等于观测误差小，也不取消 canw 的未通过结论。

## 正式 M2：同量化输入复算

只读终审发现，原运行保留的 NumPy64 使用原始 double 强迫，包含额外输入量化差，
与预注册“相同量化输入”门不是同一口径。原 51 项及其所有文件保留为端到端检查，
没有删项或改标准。本次使用原有 GPU 全轨迹，只将初态和 forcing 转为与 GPU
完全相同的 float32 值，再用未改动的 NumPy64 物理重跑参考；无需重复 GPU。
原始 double 的 NumPy/JAX64 和观测指标全部保留，不被新参考替换。

[同输入副报告](gpu-results/m4-fluxnet-matched-input.json)确认 GPU 运行与复算生产
源码相同（允许 LF/CRLF 字节差），保留原报告/input/output/source hash；
[复算脚本](../benchmarks/audit_fluxnet_matched_inputs.py)按同一冻结 criteria 判门。
canw 仍未通过：**46/5760 超差，99.201389% 合格**；其余八个状态/通量字段
全部 100% 合格。CUDA/JAX32 的 canw 状态逐位一致，二者失败坐标相同。

| 站点 canw | 同输入超差 / 1440 | 合格率 | 同输入 M2 |
|---|---:|---:|---|
| CN-Cng | 2 | 99.861111% | 通过 |
| CN-HaM | 1 | 99.930556% | 通过 |
| CN-Qia | 4 | 99.722222% | 通过 |
| CN-Cha | 39 | **97.291667%** | **未通过** |

同输入全部 canw 元素最大绝对差 **9.406406491e-8 m**，其中 46 个超差项的
最大绝对差为 **2.650765660e-8 m**，超差两端最大存水 2.346753306e-5 m。
不能把全体最大误差与失败项最大误差混用。各字段完整最大误差和 signed mean
保存在副报告，土温最大差 0.007717330 K、H/LE/G 最大差分别
0.023749172/0.043341347/0.020918205 W/m²，ET 等价功率最大差 0.043296971 W/m²。

全部 46 项已重新精确分解：**42 项干燥相对敏感性 + 4 项耗尽舍入**，
见[归因汇总](gpu-results/m4-canopy-matched-attribution.json)和
[46 行完整证据](gpu-results/m4-canopy-matched-attribution.jsonl.gz)。
本步评价、温度路径、前态路径的最大绝对贡献分别为
1.615889012e-10、2.089922761e-8、5.887764859e-9 m；同输入强迫量化项为 0，
参考公式重构误差和总差重组误差均为 0。归因没有使 99.5% 硬门改绿。

GPU 输出未改变，故 residual 自身中位数/p99、表面闭合、水量闭合和同精度结果
与原报告相同，仍全部通过。副报告的 `automated_checks_passed=false`、
`acceptance_status=failed_numeric_gate`；脚本应退出 2。
新 runner 会同时保存原始 double 基线与 `numpy64_matched32`，前者用于观测及
JAX64 parity，后者用于 x32/CUDA 正式 M2；原始输入差仍单列端到端结果。

同输入完整 NPZ 保存在 `outputs/m4/m4-fluxnet-matched-input.npz`，SHA256 为
`8de6a8489077fcbfeb5f6546fd8c77ac7fb9f84f21519ac7b48bdbeb772f90e2`。
复现 CPU 副报告：

```bash
PYTHONPATH=src python benchmarks/audit_fluxnet_matched_inputs.py \
  --report docs/gpu-results/m4-fluxnet.json \
  --trajectories outputs/m4/m4-fluxnet.npz \
  --output docs/gpu-results/m4-fluxnet-matched-input.json \
  --arrays outputs/m4/m4-fluxnet-matched-input.npz
```

## 原始 double 输入的端到端对照（保留）

以下 51 项统计含输入量化影响，作为原始端到端结果保留；正式同输入 M2 为上节 46 项。

最终生产源码 `b954d90`，xuanze 物理 GPU1（RTX 4090），四站 × 1440 步连续运行，
无每日重置。NumPy64、JAX64、JAX32、CUDA32 全部有限；源码首尾 hash 完全一致。
NumPy64 使用原始 double 输入，x32/CUDA 使用其 float32 量化输入，量化差也保留在
本次跨精度对照中。JAX/CUDA trace 与各自生产批量入口的完整结果逐位相同。

- JAX64：224796 个有效物理状态和 69120 个诊断在 `rtol=1e-6, atol=1e-6`
  全通过；最大状态差 9.66e-13、最大诊断差 1.61e-11，整数逐步一致。
- CUDA/JAX32：全部 403480 个状态逐位一致；69120 个诊断最大差
  1.52587890625e-5，全部通过同精度 `rtol=1e-6, atol=1e-6`；整数一致。
- x32/CUDA 对 NumPy64 的表中计数相同。每字段单独计算，不跨字段稀释失败。

| 端到端字段（同样阈值） | 元素数 | 超差数 | 合格率 | 最大绝对差 |
|---|---:|---:|---:|---:|
| tsoil | 23040 | 0 | 100% | 0.007717405 K |
| smc | 23040 | 0 | 100% | 5.659788e-6 m³/m³ |
| soil_ice | 23040 | 0 | 100% | 0 m³/m³ |
| swe | 5760 | 0 | 100% | 1.246514e-9 m |
| **canw** | 5760 | **51** | **99.114583%** | 9.424830e-8 m |
| H | 5760 | 0 | 100% | 0.023724 W/m² |
| LE | 5760 | 0 | 100% | 0.043194 W/m² |
| G | 5760 | 0 | 100% | 0.020847 W/m² |
| ET（等价潜热功率） | 5760 | 0 | 100% | 0.043158 W/m² |

| 站点 canw | 超差数 / 1440 | 合格率 | 全时段最大绝对差 m | 端到端结果 |
|---|---:|---:|---:|---|
| CN-Cng | 3 | 99.791667% | 1.671037e-8 | 通过 |
| CN-HaM | 2 | 99.861111% | 4.994498e-8 | 通过 |
| CN-Qia | 4 | 99.722222% | 3.510322e-8 | 通过 |
| CN-Cha | 42 | **97.083333%** | 9.424830e-8 | **未通过** |

residual 自身绝对值的四站中位数为 **0.0003242493 W/m²**，p99 为
**0.0020554352 W/m²**，最大值为 0.0106811523 W/m²；预注册门限制中位数/p99，
没有把最大值当作 p99。四站分别计算的 residual 门也全通过。
表面代数闭合最大误差 8.773804e-5 W/m²，为相应 float32 界的 0.02309 倍；
逐步水闭合最大误差 5.097315e-8 m，为相应界的 0.01040 倍。各站 30 天水残差
累积绝对最大 1.221141e-6 m。整柱融尽热量缺口沿用 Wang 已接受的已知限制，
不称为整柱焓守恒验证。

## 原端到端 51 项 canw 超差的完整归因

[归因仪器](../benchmarks/attribute_fluxnet_canopy.py) 对每条失败重新评价普通版
double 冠层截留/蒸发公式，分别置换前态、地表温度与 float32 量化 forcing。
[51 行证据](gpu-results/m4-canopy-attribution.jsonl.gz) 和
[汇总](gpu-results/m4-canopy-attribution.json) 保留全部加减项：

`GPU−NumPy = 本步评价舍入 + 温度路径 + forcing量化 + 前态路径 + 参考重构误差`。

所有失败均落在冠层水接近耗尽的阶段：**42 项为干燥过程的相对误差敏感性，
9 项为耗尽截断附近的舍入**。全部无雪，其中 1 步有降雨，其余无雨。
失败项本身的最大存水为 2.346756e-5 m，失败项最大绝对误差为
**2.653101e-8 m（0.000026531 mm）**。注意这与上表“全部 canw 元素”的最大
绝对误差不同：绝对误差较大的含水较多状态可能仍满足相对门。

各分解项的最大绝对值分别为 1.615889e-10、2.102587e-8、1.965929e-10、
5.903261e-9 m；参考重构误差最大 1.69e-21 m，总差重组误差为 0。
CN-Cha 有一段 step1373–1410 的连续干燥敏感轨迹，不把连续超差拆成无关噪声。
51/51 条都有可复核归因，**但门仍未通过**，没有删除 CN-Cha、改阈值或再调参。

## 可复现运行

生产源码从 `b954d90` 归档到独立目录
`/home/lhwang/work/noahmp-m-close-m4/`，仪器为
[fluxnet_multisite.py](../examples/fluxnet_multisite.py) 与统一的
[check_closeout.py](../benchmarks/check_closeout.py)。每一步的全部状态、整数槽、
诊断及输入保存在 NPZ；JAX/CUDA trace 另外与生产完整 1440 步输出逐位核验。
JSON 记录每个源码 SHA256、输入 SHA256、设备、各后端耗时和完整输出 checksum。
这是正确性检查，包含编译和 trace 仪器，不用其耗时宣称 GPU 加速比。

```bash
cd /home/lhwang/work/noahmp-m-close-m4
env -u LD_LIBRARY_PATH PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 \
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  CUDA_PATH=/usr/local/cuda-12.4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 /home/lhwang/venvs/noahmp-phase3b/bin/python \
  examples/fluxnet_multisite.py --manifest data/manifest.json \
  --output results/m4-fluxnet.json
```

manifest 是数组，每项包含 `site`、相对 manifest 的 `csv`、`start`、`days`，以及
可选的源文件路径和 hash。使用 `--backends` 空列表可仅生成 NumPy64 参考。
M2 对每个字段分别计数，ET 先乘 `2.5e6/dt` 从 mm/步换为等价 W/m²。
residual 判门使用它自身的绝对值，而非只看后端 residual 差。
全部超差坐标保存在 JSONL.GZ，少量超差也必须解释；数值门和观测 r/bias/RMSE 分开报告。

实际 NumPy64/JAX64/JAX32/CUDA 耗时为 4.779975/318.791307/185.256730/
2.777078 秒（含各自编译、保存轨迹和生产入口复核）。完整 NPZ SHA256 为
`223c0d0123c38b74bdb2c91265646e028fb1a09fca28eae3c7b8b49ac13317bd`。
运行日志见 [m4-fluxnet.log](gpu-results/m4-fluxnet.log)。

仪器边界：首次 runner 的退出码只表达“计算完成”，原 JSON 已明确
`numeric_thresholds_passed=false`。随后只修正 CLI 退出状态：现在统一检查数值门、
逐站门及守恒，失败退出 2。使用保存的实际报告复核该函数返回 false，证据为
[验收副报告](gpu-results/m4-fluxnet-acceptance.json)。没有重写原运行报告、
源码 hash 或任何数值；生产源码和阈值没有因此改变。

## ERA5：未测，必需 forcing 不完整

xuanze 的 `/data/lhwang/ERA5_raw/ERA5_1940_2023/surface/` 可达，
用 [只读库存脚本](../examples/era5_point_inventory.py) 核验 2007 年 7 月，
请求 CN-Cng 附近点 44.5934°N、123.5092°E，实际最近格点为 44.5°N、123.5°E。
11 份实际 NetCDF 的文件 hash、变量单位、聚合 history 和 30 个时间戳见
[库存证据](gpu-results/m4-era5-inventory.json)。文件不是逐小时序列：每月只有
31 个日均时间片，history 明确写出 `daily_reduce(... how=mean)`。

可用字段含 2m 气温、2m 露点、10m U/V、地表气压、对流/大尺度雨率，以及
**净**短波/长波。目录没有向下短波、向下长波或总降水字段。
净辐射不能代替模型需要的下行辐射；雨率也不能未经核验替代含雪的总降水。
因此没有构造虚假 ERA5 强迫，没有运行 ERA5 GPU/CPU parity，此项明确为“未测”。
`upper_air/` 不是缺失地表下行辐射的替代来源；`F:\Data_center` 仅作备用门户说明。

完整输入可用后需要的映射为：

| 模型 forcing | ERA5 输入和转换 |
|---|---|
| tair | 2m temperature，K，原值 |
| qair | 露点 `Td_C=Td_K−273.15`，`e=611.2 exp(17.67 Td_C/(Td_C+243.5))`，`q=0.622 e/(ps−0.378 e)` |
| wind | `sqrt(u10²+v10²)`，m/s；不假装已做高度订正 |
| ps | surface pressure，Pa；不用海平面气压替代 |
| swdown/lwdown | 仅向下辐射；若为 J/m² 累积量，按实际产品累积时间转 W/m² |
| prcp | 总降水 m/实际累积秒数，或经过单位/雪分量核验的完整总降水率 |
| dt | 由真实时间/累积定义确定；日均数据不能恢复小时日变化 |

```bash
/home/lhwang/anaconda3/bin/python examples/era5_point_inventory.py \
  --root /data/lhwang/ERA5_raw/ERA5_1940_2023/surface \
  --month 200707 --latitude 44.5934 --longitude 123.5092 \
  --output results/m4-era5-inventory.json
```

完整物理验证仍是独立 Phase 4，站点集、评价指标和校准方案见
[FLUXNET_PLAN.md](FLUXNET_PLAN.md)。
