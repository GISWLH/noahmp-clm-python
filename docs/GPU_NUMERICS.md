# J2：CUDA/JAX float32 数值对齐记录

当前 **同精度 48 步门及原 H4 仍为红灯，禁止合并 main**。容差未改：新增同精度门为 `rtol=atol=1e-6`；原 H4 保留 NumPy64 `rtol=1e-4, atol=1e-6` 和 JAX32 `rtol=1e-5, atol=1e-6`。没有删除、skip/xfail 失败用例，也没有剔除残差。

## 冻结候选的实测结果

2026-10-13，xuanze GPU 1，CUDA `sm_89`，JAX GPU/x64 配置开启，输入及手写 kernel 为 float32。`tests/test_gpu_float32_parity.py` 独立报告全部诊断和最终状态，避免被 NumPy 断言或首个分量挡住。

| 检查 | 结果 |
|---|---|
| N=1024 × 1 步，同精度 | 通过；12 诊断与 70 浮点状态逐位一致 |
| N=1024 × 48 步，同精度 | 失败；仅 residual 超门，504 元素，最大绝对差 `3.0517578125e-5` |
| 48 步其他诊断 | ts/rnet/H/G/prcp/runoff/melt/SWE/canw 逐位一致；LE 最大差 `1.52587890625e-5`、ET 最大差 `7.450580596923828e-9`，均在 1e-6 门内 |
| 48 步最终状态 | 全部浮点和整数状态逐位一致 |
| GPU H2 + 薄雪回归 | 通过 |
| 原 H4，1/48 步 | 仍失败，未改容差 |
| 组合 pytest | `3 failed, 8 passed in 45.52s` |
| nvcc | `-O3 --fmad=false --prec-div=false -std=c++14 -arch=sm_89` 成功，编译输出为空（零 warning） |

最坏同精度残差在零起始索引 `[column=5, step=17]`：CUDA `0.0001373291015625`，JAX32 `0.0001068115234375`。这不是 1e-6 门通过；不能用其他分量通过代替它。证据：[正式组合日志](gpu-results/j2-candidate-strict.log)、[编译原始输出](gpu-results/j2-candidate-nvcc.log)。

## 已定位并保留的修复

1. **薄雪语义错误（JAX）**：活动雪层厚度被下限 `1e-4 m` 人工放大，改变热容量及融雪能量；普通版只保护层间导热分母。取消活动层厚度 floor，保留非活动层保护。三组薄雪回归与 H2 共 5 项 GPU 测试通过。[修前](gpu-results/j2-thin-snow-before.log)、[修后/H2](gpu-results/j2-thin-snow-after-h2.log)。该项是物理语义修复。
2. **乘法、常量折叠与残差树（CUDA）**：黑体辐射先算 `T^4` 再乘 SIGMA；求根残差按 JAX 顺序逐项减去 eg/ec/tr；冻土系数、相变差商、土壤导热率及水汽/PM 常量按实际优化 HLO 的 float32 树计算。相变差商原本会把 1 ULP 未冻水差放大为约 `124.5 J m^-3 K^-1` 容量差。修后当前 156 样本的容量、导热率、qsat、PM、实际土壤 Thomas 解及冰量逐位一致。[初始](gpu-results/j2-components-initial.json)、[容量常量](gpu-results/j2-components-constants.json)、[水汽/PM](gpu-results/j2-components-vapor.json)、[通量 HLO](gpu-results/j2-flux-before.hlo.txt)。这些是运算契约对齐，不笼统称为缺失物理。
3. **编译行为（CUDA）**：FMA 探针中的 JAX 输出匹配分开舍入的乘加，CUDA 默认 FMA 不匹配；`--fmad=false` 修正此项。`optimization_barrier` 在该最小探针的优化 HLO 中被移除，不能据此宣称它对整个生产图有效。[FMA 原始结果](gpu-results/j2-barrier.json)。仅禁 FMA 的完整 1/48 步测试仍失败：[默认 FMA](gpu-results/j2-order1-same-precision.log)、[禁 FMA](gpu-results/j2-order1-no-fma-same-precision.log)。
4. **GPU 除法 lowering（CUDA）**：相同雪温分子 `58.42400360107422`、分母 `0.21799999475479126`，RN 除法和 RN 倒数乘法得到 `268.0000305175781`，JAX 和 `__fdividef` 为 `268.0`。真实土壤矩阵也定位到除法。加入单独 `--prec-div=false`，与 nvcc/NVRTC 两条编译路径保持一致；没有打开 `--use_fast_math`。[分解](gpu-results/j2-stages-snow-div.json)、[完整对照](gpu-results/j2-fastdiv-same-precision.log)。
5. **MOST（CUDA）**：逐次追踪第一次分歧出现在第 0 次 Obukhov 更新；之前 ce/H 完全一致。JAX 的 Python 常量 `KARMAN*GRAVITY` 先计算后转 f32，结果为 `3.924f`；CUDA `.4f*9.81f` 额外舍入。连同 `ustar**3` 的分组对齐后，156 输入 × 20 次 × 9 个中间量全部逐位一致，并使 N1024 首步完整门通过。[修前迭代](gpu-results/j2-most-trace.json)、[修后迭代](gpu-results/j2-most-trace-constant.json)。
6. **降雪及根系归约（CUDA）**：旧雪层先归约再加入新雪；修复第 2 步雪密度的 1 ULP 偏移。四元素根系胁迫使用实测 XLA 的 `(0+2)+(1+3)` 归约树；顺序归约和相邻成对归约分别有 19/16 个第二步样本不同。气孔 RH 的 `(a/b)/c` 在 HLO 中化为 `a/(b*c)`，按其树对齐。[降雪修前](gpu-results/j2-stages-step2.json)、[修后](gpu-results/j2-stages-step2-sum.json)、[归约探针](gpu-results/j2-scalar-step2.json)、[RH HLO](gpu-results/j2-canopy-step2.hlo.txt)。

以上测试范围限定于记录的输入、batch shape、软件版本及 GPU；XLA 可随 shape/版本改变 lowering，不能从 N156 或单步逐位一致外推所有规模。归约与除法方向分别保存原始证据后切换到下一个子问题，没有通过调容差或隐藏分量结束定位。

## 独立的 CUDA 输入所有权修复

N=1 时 AoS 转置可仍连续，`ascontiguousarray` 因此可能返回调用者设备内存视图。ping-pong 第 2 步写回该视图，改变传入状态。入口现对 state/options 显式 `copy(order="C")`，强迫只读。

因果对照在独立 Python 进程中使用**修后物理 + 仅恢复旧 loader 两行**，同一回归失败；真实修后 loader 回归 `1 passed in 0.78s`。没有回滚工作树。[旧 loader 因果对照](gpu-results/j2-input-ownership-before.log)、[修后](gpu-results/j2-input-ownership-after.log)。

## 仍需解释的残差

当前剩余同精度失败集中于潜热的少量舍入及残差逐项相减。近零残差对这种差异敏感是数值条件分析，**不是它可以自动获得更宽容差的理由**，也未证明所有剩余差异都是精度地板。

普通版最终裸地诊断采用 `rn-h-(eg+ec+tr)-g`；当前 JAX 最终诊断与求根都采用 `rn-h-eg-ec-tr-g`。这项分组差异属于与 NumPy64 对照时的独立误差来源，不应混入 CUDA/JAX 同精度门。下一轮应固定最坏柱的前一步状态和 Ts，直接分解 eg/ec/tr 与最终残差树；原 H4 及 main 合并门继续保持红灯。

## 重放

在 Linux venv，设置 `CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src`，并移除旧 `LD_LIBRARY_PATH`：

```bash
python -m pytest tests/test_gpu_float32_parity.py tests/test_gpu_parity.py tests/test_gpu_numerical_components.py tests/test_gpu_cuda.py -q -s
python -m pytest tests/test_gpu_cuda_inputs.py -q -s
CUDA_ARCH=sm_89 bash src/noahmp_clm/gpu/cuda/build.sh
python benchmarks/probe_gpu_numerics.py --output components.json
python benchmarks/probe_gpu_trajectory.py --output trajectory
```

`probe_gpu_numerics.py` 比较默认 FMA、禁 FMA、禁 FMA+近似除法三组；`probe_gpu_stages.py` 分解真实热方程矩阵；`probe_gpu_most.py` 记录 20 次 MOST 迭代；`probe_gpu_flux_math.py` 与 `probe_gpu_scalar_math.py` 是保留原始表达式的诊断对照，部分字段故意不调用修后函数，不能把其中旧表达式的偏差误认为当前生产回归。
