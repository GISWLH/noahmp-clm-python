# ROADMAP — 从普通版本到 CUDA

## 设计不变量

贯穿所有阶段：**单柱状态全封装、柱间零耦合**。任何阶段都不允许出现"柱 A 读柱 B 状态"
的代码。批量轴（格点 × 时间步）永远是独立的，这是 CUDA/JAX 向量化的前提。

## 阶段 0 — v0.1 基线（本仓库当前状态）✅

纯 Python + numpy，可读优先：

- 4 层土壤水热（Noah-MP 默认分层）、单层 bulk 积雪、big-leaf 冠层
- 表面能量平衡割线迭代；土壤热传导 Crank–Nicolson；土壤水 fill-and-spill + Campbell 排水
- 合成强迫示例 + 三条硬约束测试（能量闭合 / 水平衡 / 批量独立性）

## 阶段 1 — Noah-MP v5.0 对标

目标：与 NCAR/noahmp 离线单柱运行在 FLUXNET 强迫下定量可比。

- 多物理选项开关（Noah-MP 标志性设计）： runoff（TOPMODEL/Schaake）、
  snow albedo（BATS/CLASS）、stomatal（Ball-Berry/Jarvis）、
  soil moisture stress（Noah/SMC）
- 冻土过程（soil ice + 相变潜热）、多层积雪（最多 3 层，Noah-MP 风格）
- 冠层分层辐射（双流近似简化版）与植被能量平衡迭代
- 对标实验：选 3–5 个 FLUXNET 站点，对比日 ET / 感热 / 土壤温湿度的偏差与相关系数；
  目标：偏差在 Noah-MP 参数不确定性范围内

## 阶段 2 — CLM 柱子集

- CLM5 生物地球物理核心的柱版本：多层冠层（~10 层）光合—气孔耦合（Farquhar + Ball-Berry）
- 保留"柱"抽象：CLM 的 gridcell → landunit → column → PFT 层级在本仓库中
  坍缩为"一个 column = 一个独立 PFT 柱"，批量轴天然对应 PFT × 格点
- 与阶段 1 共用 driver：`run_columns` 签名不变，只换 `step_column` 里的物理包

## 阶段 3 — 批量加速（JAX / CUDA）

- 把 `run_columns` 的 Python 循环换成 `jax.vmap(step_column)`：要求物理函数
  纯函数化（状态进、状态出，无原地修改）——阶段 0 已按此约束编写
- CuPy 直译版作为对照；之后手写 CUDA kernel：一个 thread block 处理一批柱，
  土壤三对角求解用 batch Thomas
- 性能目标：在 A100 上相对单核 Python 基线 ≥100×（N=10⁵ 柱量级）

## 不做的事

- 不移植整机 CESM/WRF，只做单柱物理
- 不做河道演算、地下水侧向流（那是另一个并行模式）
- 阶段 1 完成前不碰 GPU：先保证物理是对的
