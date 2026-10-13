# 给接手本仓库的 AI Agent 的工作指令（阶段 4：综合收尾验收）

> 这是一段可直接粘贴给另一个 AI agent 的 prompt。那个 agent 持有 Wang 的 GitHub 凭证。
> 背景：阶段 3/3b/3c/3d 已执行完毕，分支 `gpu/migration`，最新提交 b954d90，
> 未合并 main。全量 147 passed / 2 failed（原 H4 两项）。K1（137 项残留）、
> K3（5565 柱冰量超差）、H4 仍红；K2（365 天）已绿；N=300k 基线收齐；
> CN-Cng 30 天 smoke 已完成。开工先读 `docs/GPU_SUMMARY.md`、
> `docs/GPU_RESIDUAL_FORENSICS.md`、`docs/GPU_DOMAIN_FIX.md`、
> `docs/GPU_ANNUAL_FIX.md`、`docs/BENCHMARK.md`。

## M0. 收尾原则（先读，这是 Wang 的决定）

1. **目标是系统收尾，不是继续攻关。** Wang 已确认：**数值总体对的上即验收**。
   本 prompt 的 M2 是该指令的具体化——预先定义、可量化、跑完对照、
   **不许事后改标准**。不要再把原 H4 的逐元素严格门当作合并条件，
   但必须报告实际数字与已知限制。
2. 已知红灯不删除、不隐藏、不放宽到"通过"：收尾 = "修得掉就修，
   修不掉就量化并写进已知限制，然后验收"。
3. 合并 `gpu/migration` → `main` 是收尾动作，仅当 M1–M4 的门按 M2 通过。
4. 每阶段 push 到 `gpu/migration`；单点阻塞 30 分钟记
   `docs/GPU_BLOCKERS.md` 并绕行；不伪造数字；天亮前 push 所有进度。

## M1. 增量式 ΔT 修复（限时 2 小时，一次尝试）

K1 的 137 项与 K3 的 5565 柱已定位到**同一根因**：float32 对绝对温度
运算的相消（割线求解的残差 + Thomas 求解的土壤温漂）。做一次
限时尝试：

1. Thomas 求解改解 **ΔT（增量式）**，能量平衡同步改增量形式；
   numpy + JAX + CUDA 三处同步改，保持三方同精度对齐。
2. 重跑：H4 1/48 步、K3 域门、365 天年度。
3. 成功（H4 全绿或仅剩已解释的聚集项）：采纳，更新文档。
4. **2 小时搞不定或出现回归**：回退到修复前源码，记录尝试与原因，
   继续 M2，**不许整夜卡死**。回退本身不是失败——收尾接受已知限制。

## M2. 收尾验收门（预先定义，跑完对照，不许事后改）

- **同精度门（必须全绿）**：CUDA vs JAX-x32（float32），1024 柱 × 48 步，
  全部诊断 `rtol=1e-6`——已通过，保持。
- **跨精度门（总体对的上）**：CUDA/JAX-x32 vs NumPy64（float64）：
  - 状态量（tsoil/smc/soil_ice/swe/canw）与通量（H/LE/G/ET）：
    **≥99.5%** 元素满足 `rtol=1e-3`（近零通量用 `atol=1e-2 W/m²`）；
    超差项必须**全部聚类归因**，不许有未解释的均匀漂移。
  - residual 诊断：中位数 < 1e-3 W/m²，p99 < 1e-2 W/m²
    （float32 地板量级，推导见 `docs/GPU.md` 精度一节）。
  - 守恒：能量/水平衡闭合在 float32 地板内。
  - 长跑：365 天四后端无 NaN、无发散（K2 已绿，保持）。
  - 域门：64800 柱——若 M1 修复则应通过；若 M1 未修复，
    超差柱比例与量级必须量化并写入已知限制（见 M5），
    且全部归因于已记录的冰量漂移。
- 若 M1 未修复 K3，接受条件 = 超差全部归因 + 文档明确标注
  "冻土敏感应用建议用 float64 / JAX-x64"。

## M3. 跨环境简单测试（EHPC / xuanze / HF ZeroGPU）

目标是"每个环境都能跑、结果对得上"，不是深度基准。每个环境记
硬件/软件/耗时/checksum 到 `docs/ENV_TESTS.md`。

- **EHPC 超算**（SSH 已验证；3 计算节点，共 144 物理核、约 1.1 TiB 内存，
  另有管理节点；PBS/MPI 冒烟已通过）：
  1. 1 节点串行 smoke：N=100 × 48 步，CPU 基线；
  2. 多进程分柱：同一作业内把柱分到多进程，结果与串行在精度档内一致；
  3. MPI 跨节点：2 个节点各起 rank，分柱并行，结果与串行在精度档内一致。
  PBS 脚本存入 `benchmarks/ehpc/`（`*.pbs` + README 说明提交命令）。
  SSH 30 分钟连不上：记"未测"并继续其它环境，不阻塞收尾。
- **xuanze**（2 × RTX 4090，每卡 24564 MiB；40 物理核 / 80 线程，
  124 GiB 内存；免密已验证）：最终代码重跑核心 GPU 门——
  parity 8×48、选项抽查、1 档基准 N=1e5（含校验计时）。
- **HF PRO / ZeroGPU**（本次实际分配 RTX PRO 6000 Blackwell Server
  Edition MIG 2g.48gb；短时云端兼容性检查、交互式演示）：
  JAX smoke N=1024 × 48 parity + 演示 Space 可用性检查，短时完成。
  **仅用已有访问权限；无权限则记"未测"，不许索要或寻找凭证。**

## M4. 真实数据检验（有界，不调参，不宣称验证）

- **FLUXNET 处理后数据**（NAS WLH：
  `/volume2/Data/Global/FLUXNET-Data-Shop/processed/site_csv/`；
  原始数据在 `.../raw/`，处理后不可达时才用原始）：
  选 **3–5 个**站点 × 30 天，LE/H 对比观测，只报告 r / bias / RMSE，
  **不调参、不校准**；JAX64 parity 通过为门，x32/CUDA 在 M2 精度档内。
  脚本：`examples/fluxnet_multisite.py`。
- **ERA5**（xuanze：`/data/lhwang/ERA5_raw/ERA5_1940_2023/`，
  含 `surface/`、`upper_air/`）：取 **1 个格点** × 30 天，
  用 ERA5 地表强迫（2m 气温、露点→比湿、风速、气压、短波/长波、降水）
  驱动模型，GPU vs CPU 对比。
- **数据门户**（本地 `F:\Data_center`）：仅作备用说明，不要求远程访问。
- 任一路径不可达：记"未测"并用可达的继续，**不许编造**。
- 交付：`docs/FLUXNET_PLAN.md` 更新（完整验证仍为 Phase 4 后续，
  含站点集、评价指标、校准方案）。

## M5. 收尾交付

1. `docs/KNOWN_LIMITATIONS.md`（新增，合并后随 README 可见）：全部量化写清——
   K1 残留项（数量/聚集/量级）、K3 冰量漂移（比例/量级/建议用 float64 的场景）、
   融雪后热量路由缺口、CPU 基线 provenance（启动无源码 hash）、
   双卡效率（x32 传输主导，属预期）、性能表属 K2 修前源码（修复后需重测
   或已重测则更新）。
2. `README.md`：状态改为"可用，带已知限制"；性能表标注源码版本与校验方式；
   各环境运行命令（EHPC PBS / xuanze GPU / HF ZeroGPU）。
3. 合并 `gpu/migration` → `main`（仅当 M1–M4 按 M2 通过）；打 tag
   `v0.2.0`（GPU 移植收尾版）。
4. `docs/GPU_SUMMARY.md`：顶部三句话更新为最终状态。
5. 全量 `pytest` → push。

现在开始：M1，增量式 ΔT 修复（计时 2 小时）。
