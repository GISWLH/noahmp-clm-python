# J4 float32 batch shape 差异定位

2026-10-13，保持生产代码、输入和 `rtol=atol=1e-6` 不变，复现
`GPU_EXTENDED_VALIDATION.md` 中 global column 2、zero-based step 17 的失败。
使用冻结目录 `/home/lhwang/work/noahmp-phase3b/j4-stable-jax-20261013T0513`，
JAX 源 SHA256 `65c5c3bc23e2c378014e8e6cc79c584387e6e1f0da8e7a512f7ca2a10277e8a8`。
实际 JAX 0.11.2、CUDA GPU1、float32，输入仍为 N=64800、seed=20261013、
day0、48 步，N=1 直接切片该全域输入的 `[2:3]`。

## 已实测的定位链

1. 原始完整 48 步图重现 H/G/residual 三项超差，残差最大绝对差
   `0.001068115234375 W/m²`。Ts 另有一个 float32 间距的差异，但未超过
   Ts 相对误差门；最终完整浮点状态逐位相同。这不消除逐步诊断失败。
2. 两种 shape 分别运行前 17 步，其诊断与对应完整图的前缀逐位一致，返回的
   **完整 pre-step float/int 状态相同**。该状态下再次运行第17步，重现原差异。
   因此本次差异没有来自此前状态积累的证据。
3. 临时内存探针将该步的两个求根调用都替换为输入中的同一个 Ts，保持其余
   全步物理计算。H、G、rnet 变为逐位相同，但 LE 仍差
   `7.62939453125e-6 W/m²`，残差仍差 `1.52587890625e-5 W/m²`。
4. 在相同 fixed-Ts 图上追加潜热分量和中间量输出，两个 shape 的前12项
   与上一步探针各自逐位一致。结果将差异缩小到根区水分胁迫、气孔阻抗及蒸腾：

| 中间量 | N=64800 | N=1 | 大批量减单柱 |
|---|---:|---:|---:|
| beta_root | 0.5157148241996765 | 0.5157147645950317 | 5.960464477539063e-8 |
| stomatal resistance | 300.7878723144531 | 300.7879333496094 | −6.103515625e-5 |
| transpiration LE W/m² | 42.97646713256836 | 42.97645950317383 | 7.62939453125e-6 |
| ground evaporation LE W/m² | 1.2072699069976807 | 1.2072699069976807 | 0 |
| canopy evaporation LE W/m² | 0 | 0 | 0 |

available_root、rn_canopy、ce_initial、rho 同样逐位相同。原始求根返回 Ts 分别为
`290.9654846191406 K` 与 `290.96551513671875 K`，相差一个 float32 间距。
两个返回值的残差绝对值均小于原求根停止标准 `1e-3 W/m²`；尚未输出完整
迭代轨迹，不能声称已经证明两图在哪一次迭代首次改变停止分支。

## HLO 证据、推断及候选

完整 48 步、17 步前缀、同状态单步、fixed-Ts 单步均保存了优化后 HLO。
fixed-Ts HLO 的根胁迫归约在 N=1 中是 `f32[4] -> f32[]`，在 N=64800
中是 `f32[64800,4] -> f32[64800]`，都保留 `reduce(add)`。它们没有在这份
HLO 文本中明确给出 GPU 机器指令的四项相加树；不能仅凭 shape 不同就断言 FMA
或具体机器归约顺序。

CPU float32 对**相同 pre-step 状态**生成四个 weighted-stress 叶值后，顺序
`((w0+w1)+w2)+w3` 得到 `0.5157148241996765`，相邻配对和交叉配对均得到
`0.5157147645950317`，分别精确命中两个 GPU beta_root。这个控制实验支持
归约树假设；GPU 四个叶值及机器级归约顺序尚未直接导出，所以目前属于强候选，
不是已经证明每个节点的因果结论。

先在独立临时源码目录准备并验证唯一变更：把 `_soil_stress` 的四项 `jnp.sum`
改为显式 `(w0+w2)+(w1+w3)`。三项候选验证通过后，约2026-10-13 06:09 UTC
该唯一运算改动已应用到生产 JAX，随后只增加一行说明固定顺序原因的注释。

临时目录 `/home/lhwang/work/noahmp-phase3b/j4-cross-candidate` 由 f29d7ad
重建；CUDA/loader 保持该提交版本，不包含并行 J3 候选。新的 JAX SHA256 为
`d49b4dd62e0b3b503325e4f0bdb67d9c990f02ce57b95c5645cbc08c25bc061c`。
包含说明注释的最终生产源 raw/LF SHA256 均为
`0e2966918699eb39ee520ec2993eee5819df9d49371a1bfc87901b2dc8e69270`。
这两个版本的运算完全相同，下面实验日志保留实际执行时的 d49 源身份，不把
后来增加注释的 hash 回填为当时测量值。

| 候选验证 | 实测结果 |
|---|---|
| 原 column2 的 N64800/N1 完整48步诊断、最终完整float状态 | 逐位相同 |
| 17步前缀对完整图、自身shape比较、同prestate单步、fixed-Ts单步 | 逐位相同 |
| J2 CUDA/JAX32 N1024×1、×48，全部12诊断和70浮点状态 | 逐位相同；整数状态相等 |
| H2 混合选项和3个有效薄雪组件回归 | 通过 |
| 合并测试调用 | 7 passed in 35.28s |

这次仅替换根胁迫四项归约的干预消除了原跨shape复现，并保住已有J2/H2门，
为本例归约实现导致的数值差异提供因果证据。尚未直接导出旧GPU机器归约树，
因此仍不把旧N64800归约明确命名为某一种四项顺序。

这轮未验证所有64800列对N1，也未验证完整双卡分片或全部五档异构输入。
该局部候选成功不能把那些旧失败自动改为通过；采用新JAX后，必须按新源
身份重跑这些完整门，旧基准仍属于旧源身份。

原始记录：[候选完整重放](gpu-results/j4-shape-cross-probe.json)、
[7项测试日志](gpu-results/j4-shape-cross-tests.log)、
[候选源身份](gpu-results/j4-shape-cross-metadata.json)、
[唯一候选补丁](gpu-results/j4-shape-cross-candidate.patch)。

生产回归 [test_gpu_batch_shape.py](../tests/test_gpu_batch_shape.py) 使用上述
实际 pre-step 完整 f/i/u 的[小型 fixture](../tests/fixtures/gpu_batch_shape_column2.json)，
记录原始日志、JAX源和数组hash。将这一个真实列复制为N64800以隔离shape因素，
与N1运行同一个实际step，比较全部诊断、状态和整数状态；没有重新拼造物理状态，
也不把这组复制列当作异构全域输入。只运行一个时间步，GPU实测约13秒。

同一新回归在旧冻结JAX上[失败](gpu-results/j4-shape-regression-before.log)：
194400个诊断元素超差（每复制列3项），max_abs仍为`0.001068115234375`。
在新生产JAX上[通过](gpu-results/j4-shape-regression-after.log)：全部float诊断/
state逐位相同，`1 passed in 12.89s`。GPU要求模式下CPU回退直接失败，不会
把CPU执行标成CUDA通过；无GPU的一般开发环境可以明确报告能力缺失跳过。

## 新树的64800列完整域审计

随后使用最新版有效雪层mask审计脚本、相同异构输入，在独立冻结目录运行
JAX32完整64800列×48步。[原始日志](gpu-results/j4-domain-jax32-64800-fixed-tree.jsonl)
记录实际执行的 d49 源身份，未受并行J3源码编辑影响。15个样本对N1的完整状态
和逐步诊断均通过原 `1e-6/1e-6` 门（原来14/15），样本诊断最大绝对差为
`9.5367431640625e-7`。这15列没有要求逐位相等；不能将其表述为全域64800列
全部逐值对照通过。

原NumPy64门保持不变：诊断仍为0/15通过，有效物理状态13/15通过；整个审计
**仍失败、exit1**。全域finite/物理水量边界、0.5 W/m²能量门及全域输入扰动
检查通过，最大绝对残差仍为0.014354705810546875 W/m²。完整审计用时
61.43547秒，包含编译及参考比较，不能作为GPU性能基准。年度第3天的物理
稳定性问题未改，也未重复长跑。

## 原始证据和 instrumentation changelog

- [重放报告](gpu-results/j4-shape-probe.json) 含环境、输入身份、各 shape 的
  HLO hash、完整逐步诊断与 pre-step 比较。
- [潜热组件](gpu-results/j4-shape-components.json)、
  [完整单柱 pre-step 输入](gpu-results/j4-shape-prestep-column2.json)、
  [CPU 相加顺序控制](gpu-results/j4-shape-root-arithmetic.json)。
- [元数据](gpu-results/j4-shape-metadata.json) 和两个实际执行的临时脚本
  [重放](gpu-results/j4-shape-probe.py)、[组件](gpu-results/j4-shape-components.py)。
  这些为实验记录，脚本内明确保留了远程输出路径。
- 大型 pre-step NPZ 与 HLO 只留在远程
  `/home/lhwang/work/noahmp-phase3b/profile_output/j4-shape/`，不提交大二进制。

第一轮 instrumentation 只在内存复制的 step 函数里固定 Ts，第二轮只追加
组件输出，不改生产文件、不改实际输入、不改容差。组件版确认前12项输出没有
因增加观测点而改变；这项检查不能证明编译器所有内部节点都保持原样。

生产改动仅通过上述已列范围；旧版完整域和双卡失败记录保留，需分别复验。
原 H4 NumPy64 门未改变，任何阈值均未放宽。
