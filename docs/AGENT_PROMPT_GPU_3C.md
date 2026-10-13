# 给接手本仓库的 AI Agent 的工作指令（阶段 3c：H4 根因修复 + CUDA/JAX 深化 + 深度加速测试）

> 这是一段可直接粘贴给另一个 AI agent 的 prompt。那个 agent 持有 Wang 的 GitHub 凭证。
> 背景：阶段 3b（`docs/AGENT_PROMPT_GPU_XUANZE.md`）在 xuanze（双路 Xeon Gold 5218R，
> 双 RTX 4090，CUDA 12.4，sm_89）上完成了 H1/H2/H3/H6，分支 `gpu/migration`。
> 当前红灯：H4（CUDA float32 严格 parity，2 个测试失败）、H5 的 NumPy 大规模基线
> 仍在后台补测。44 passed、2 failed，容差未放宽，未合并 main。
> 本指令先读 `docs/GPU_SUMMARY.md`、`docs/GPU_BLOCKERS.md`、`docs/BENCHMARK.md`
> 再动手；按 J1→J5 顺序整夜推进。

## J0. 铁律（延续）

1. `src/noahmp_clm/` 普通版只读；`docs/STATE_SPEC.md` 的 ABI（70 float64 + 33 int32
   + 8 强迫）不许改。
2. 不删测试、不放宽容差、不伪造数字；`main` 在全部绿灯前不受影响。
3. 每 1–2 小时 push 到 `gpu/migration`；单点阻塞 30 分钟记
   `docs/GPU_BLOCKERS.md` 并绕行。
4. 天亮前 push 所有进度，`docs/GPU_SUMMARY.md` 顶部三句话写真实状态。

## J1. 收尾 H5：补齐基线（约 30 分钟，先做）

后台的 NumPy 单核大规模任务（`benchmarks/run_cpu_batch.py`，deadline
2026-10-13 06:45 EDT）应该已结束：
1. 收集 `../logs/h5-overnight` 的三次重复中位数，填进 `docs/BENCHMARK.md`
   的 NumPy 列（N=1e4/1e5/3e5）。
2. 算出真实加速比：JAX x64 / x32 / CUDA（若过门）相对单核。
   已知 JAX 在 N=1e5×48：x64=22.003s（4.58µs/柱步），x32=3.144s（0.655µs/柱步）。
3. **不许用小 N 外推**；哪个格子没数就写"未测"，不许编。
4. 若加速比 ≥100×（x64 已接近），如实记录；若某档没达标，写清原因。
   → commit + push。

## J2. H4 根因修复（约 2–3 小时，本阶段关键路径）

`docs/GPU_BLOCKERS.md` 的 H4 节已有定位证据，先读透。按以下三步走，
**顺序不能乱**：

### J2a. 真 bug 猎杀：CUDA vs JAX-x32 对齐到 1e-6（必须先做）

两者同为 float32，任何 >1e-6 的差异都是真 bug，不是精度地板：
1. 48 步探针显示 CUDA–JAX32 的 G 有 26769 个超差元素——从 G 入手。
2. 统一三处：① 残差/通量的**运算结合顺序**（CUDA 是 `rn - h - (eg+ec+transp) - g`，
   普通版/JAX 是逐项相减——三方统一为同一顺序）；② 割线/MOST 的**迭代次数**
  （CUDA 必须与 JAX 的固定 25/20 次逐次对应，检查早停逻辑是否一致）；
   ③ FMA 行为（nvcc `--fmad=false` 与 XLA 融合的差异——先测 `fmad` 开关的影响，
   记录结论）。
3. 写 `tests/test_gpu_cuda_jax32.py`：N=1024 单步，CUDA vs JAX-x32，
   全部 12 个诊断 `rtol=1e-6, atol=1e-9`。**全绿才算 J2a 完成**。
   这一步修的是真 bug，不涉及容差哲学。

### J2b. benchmark 的 checksum 门（与 J2a 并行可做，约 30 分钟）

当前手写 CUDA 报 N=1e5×48 只用 0.007s（=1.5ns/柱步 ≈ 660 TFLOPS，
物理上不可能）——极可能是 kernel 被编译器 DCE，或只测了 launch 开销：
1. 给 `benchmarks/bench_gpu.py` 的 CUDA 后端加** checksum 门**：
   device 端 reduce（`ts` 与 `le` 的和）拷回 host，与 JAX-x32 的对应值
   对比（容差 1e-3 相对）；checksum 不过则该后端记 `unvalidated`，
   **不许记录计时**。
2. 确认计时包含 `cudaDeviceSynchronize`（或 CUDA event 计时）。
   修正后重跑五档，把 `docs/BENCHMARK.md` 里 CUDA 列的旧数字替换为
   checksum 过门后的真实数字（旧数字保留一行标注"方法论修正前"）。
3. 若修正后 CUDA 仍比 JAX-x32 快 >50×，先怀疑方法论，再怀疑人生——
   把怀疑过程写进文档。

### J2c. float32 残差地板的科学处理（J2a 之后，约 1 小时）

J2a 修完真 bug 后，CUDA vs numpy（float64）的残差差仍会停在 ~1e-3 量级——
这是 float32 的固有地板，不是 bug：
1. 把推导写进 `docs/GPU.md` 精度一节：float32 在 285K 处 ulp=3.05e-5 K；
   Ts 求解差 1 ulp，经 ~10–30 W/(m²·K) 的 conductance 放大，
   通量差 ~3e-4–9e-4 W/m²；残差是 O(1e2) 量级通量的相消，
   地板约 1e-3 W/m²。引用 blocker 文档里的实测数（CUDA 残差 2.68e-4
   vs numpy 2.96e-7）作为佐证。
2. 基于推导提出**分档门**（不是放宽，是按推导设定）：
   - 状态量与通量（Ts、H、LE、G、ET、终态 tsoil/smc）：保持
     `rtol=1e-4`（J2a 已证明可达）；
   - `residual` 诊断（求解器闭合度）：改用**绝对容差 2e-3 W/m²**
    （= 2× 推导地板），并在测试注释里写清推导出处。
3. **分档变更必须单独列出、附推导，经 Wang 确认后才可合入**；
   在此之前 H4 保持红灯、不合并 main。把待确认事项写进
   `docs/GPU_SUMMARY.md` 顶部，措辞让 Wang 一眼能做决定。

## J3. CUDA kernel 深化（约 2–3 小时）

H3 的 kernel 只有**默认选项**。深化：

1. **选项分发**：给 kernel 加 `switch(opt)` 分支表，覆盖 H6 已验证的
   35 个选项值（至少：`OptStomataResistance` 1/2、`OptSoilWaterTranspiration`
   1/2/3、`OptSnowAlbedo` 1/2、`OptGroundResistanceEvap` 1/4、
   `OptSurfaceDrag` 1/2；35 个全覆盖则更好）。每个分支头注释
   `对应普通版: <模块>.<函数名>`。
2. **occupancy 调优**：block 大小扫 128/256/512（`build.sh` 参数化），
   `nvcc --ptxas-options=-v` 记录寄存器/共享内存占用，选最优并写进
   `docs/GPU.md`（一句话：为什么这个 block 最好）。
3. **时间循环融合（可选，测了再说）**：试一个把 48 步 fuse 进 kernel
   的版本（persistent kernel，grid-stride 外层再套时间循环），对比
   per-step launch 的版本；只保留实测更快的，另一个删掉并记录结论。
4. 约束不变：1 thread = 1 column，SoA，单精度，零 warning 编译。
5. 新增 `tests/test_gpu_cuda_options.py`：kernel 的选项分发 vs JAX-x32
   同选项，N=256，`rtol=1e-6`（同为 float32，对齐 J2a 标准）。
   → commit + push。

## J4. 深度加速测试（约 2 小时）

这是"支持 GPU 深度加速测试"的主体——建方法、跑数据、写结论：

1. **Roofline**：算一柱一步的算术强度（FLOP 数/访存字节数——按
   STATE_SPEC 的 ~444B 读 + ~280B 写估算，分母实测为准）；
   用简单 stream 式 microbenchmark 测本机实际 DRAM 带宽（RTX 4090
   理论 ~1008 GB/s）；判断 kernel 是 memory-bound 还是 compute-bound，
   一句话写进 `docs/GPU.md`，指导 J3 的优化方向。
2. **ncu profiling**（若有 Nsight Compute）：对 hot kernel 跑一轮，
   记录 achieved occupancy、合并访问效率、主要 stall 原因；
   没有 ncu 就用 `nvprof`/`nsys` 替代，都没有就写"未测"。
3. **双卡扩展**：xuanze 有**双 RTX 4090**——把 N 柱手动对半分到两卡
   （`CUDA_VISIBLE_DEVICES` + 双进程，无需通信，格点独立天然支持），
   测 N=1e5/3e5 的加速比，期望 ~1.9×+；记录实测数。
4. **长稳测试**：N=1e4 × 365 天 × 48 步（约 1.75 亿柱步），
   检查 NaN/inf、能量漂移、水平衡闭合；这是 kernel 第一次长跑，
   任何异常都记 blocker。
5. **x64/x32 七倍之谜**：JAX x64（22.0s）vs x32（3.14s）差 7×，
   远超 2× 字节比——做一轮 profiling（XLA HLO 或 ncu），
   在 `docs/GPU.md` 用一段话解释原因（很可能是 x64 codegen 问题）。
6. **真实域集成测试**：构造一个"大陆域"合成测试——如 360×180 经纬网格
   （64800 柱），土壤/植被/LAI 按纬度带状分布 + 随机扰动，跑 30 天，
   验证：无 NaN、ET 纬度分布合理（热带高、极地低）、与单卡 JAX 结果一致
   （抽查 1000 柱，rtol=1e-4）。这是 GPU 版第一次"像真模型一样"跑。
   → 每项 commit + push。

## J5. 收尾（约 30 分钟）

1. `docs/GPU.md`：补精度推导（J2c）、roofline 结论、双卡数、x64/x32 解释。
2. `docs/BENCHMARK.md`：最终真实表格（含 NumPy 基线、双卡）。
3. `docs/GPU_SUMMARY.md`：顶部三句话更新；H4 若按 J2c 分档通过且经 Wang
   确认，标绿；否则保持红灯并写清待确认项。
4. 全量 `pytest` 全绿（除待 Wang 确认的容差项外）。
5. 合并 `main`：**仅当 H2 + H4（按确认后的分档）+ H5 全绿**；
   否则留在 `gpu/migration`，SUMMARY 写清缺口。

现在开始：J1，收集夜间 NumPy 基线，补全 BENCHMARK.md。
