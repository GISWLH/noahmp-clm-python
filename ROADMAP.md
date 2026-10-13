# ROADMAP — 从普通版本到 CUDA

## 设计不变量

贯穿所有阶段：**单柱状态全封装、柱间零耦合**。任何阶段都不允许出现"柱 A 读柱 B 状态"
的代码。批量轴（格点 × 时间步）永远是独立的，这是 CUDA/JAX 向量化的前提。

## 阶段 0 — v0.1 基线（本仓库当前状态）✅

纯 Python + numpy，可读优先：

- 4 层土壤水热（Noah-MP 默认分层）、单层 bulk 积雪、big-leaf 冠层
- 表面能量平衡割线迭代；土壤热传导采用冻结系数的后向 Euler；土壤水 fill-and-spill + Campbell 排水
- 合成强迫示例 + 三条硬约束测试（能量闭合 / 水平衡 / 批量独立性）

## 阶段 1 — Noah-MP v5.0 对标

目标：与 NCAR/noahmp 离线单柱运行在 FLUXNET 强迫下定量可比。

- [x] 多物理选项开关：precipitation（Jordan/BATS/freezing threshold）、surface drag
  （MOST/Chen97）、runoff（TOPMODEL/BATS/free drainage）、snow albedo（BATS/CLASS）、
  stomatal（Ball-Berry/Jarvis）、soil moisture stress（Noah/CLM/SSiB）和四种 ground resistance
- [x] 冻土过程（soil ice + apparent heat capacity）、多层积雪（最多 3 层）
- [x] MOST surface drag、Sakaguchi--Zeng ground resistance、Anderson snow compaction
- [x] Noah-MP `ConfigVarType.F90` 的 `Opt*` 配置框架与 HRLDAS 默认值；动态植被固定为 4，
  作物/灌溉/湿地/冰川/瓦片排水关闭
- 冠层分层辐射（双流近似简化版）与植被能量平衡迭代
- [x] 官方源码验收：NCAR/noahmp v5.0 `src/` 的 135 个 Fortran 模块可编译，
  `AtmosForcingMod` 的 Jordan/BATS/冰点阈值与 Python 实现逐点一致
- [ ] 对标实验：选 3–5 个 FLUXNET 站点，对比日 ET / 感热 / 土壤温湿度的偏差与相关系数；
  `examples/parity_noahmp.py` 已提供 CSV 对比入口，待提供带站点参数和强迫的 HRLDAS 官方输出。

## 阶段 2 — CLM 柱子集

- CLM5 生物地球物理核心的柱版本：多层冠层（~10 层）光合—气孔耦合（Farquhar + Ball-Berry）
- 保留"柱"抽象：CLM 的 gridcell → landunit → column → PFT 层级在本仓库中
  坍缩为"一个 column = 一个独立 PFT 柱"，批量轴天然对应 PFT × 格点
- 与阶段 1 共用 driver：`run_columns` 签名不变，只换 `step_column` 里的物理包

## 阶段 3 — 批量加速（JAX / CUDA）

- [x] 环境侦察、固定状态/强迫 ABI（`docs/STATE_SPEC.md`）和 50 柱 round-trip
- [x] JAX 定长单柱核心：`jit`、`vmap`、`lax.scan`、`lax.switch`、固定迭代 Thomas/割线
- [x] CuPy host-reference 正确性锚点（CUDA headers 缺失时明确 unavailable）
- [x] 性能基准脚本和本机 N=100 实测记录
- [x] 完整默认物理 SoA CUDA kernel、256-thread grid-stride loader；xuanze nvcc 零 warning 编译并实际执行
- [x] JAX `vmap(scan)` 批量入口及 Linux GPU float64 8 柱 × 48 步混合门
- [x] GPU 已实现选项 35 个取值抽查和一日 float32/float64 累积精度实测
- [ ] CUDA32/NumPy64 N=1024 原H4 parity（同精度CUDA/JAX32已通过，NumPy64的residual、近零H/G等仍超差）
- [ ] CuPy 直译版作为对照；之后手写 CUDA kernel：一个 thread block 处理一批柱，
  土壤三对角求解用 batch Thomas
- [ ] 性能目标：在 A100 上相对单核 Python 基线 ≥100×（N=10⁵ 柱量级）

阶段 3 总体未完成：H2 和 H3 已通过，J3的35值CUDA分发联合55项通过，H4仍红；
新共同源码五档GPU与H5大规模单核基线正在补测。只有H2+H4+H5全绿才可合并main；
当前状态与阻塞原因见 `docs/GPU_SUMMARY.md`。

## 不做的事

- 不移植整机 CESM/WRF，只做单柱物理
- 不做河道演算、地下水侧向流（那是另一个并行模式）
- 阶段 1 完成前不碰 GPU：先保证物理是对的
