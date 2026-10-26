# P2–P4 当前实现与验收审计（2026-10-26）

这份审计区分“代码模块存在”和“Wang 的外部科学验收已通过”。后者仍须真实
FLUXNET/HRLDAS/CTSM、多年 spinup 和留出集数字，组件 parity 不能替代它们。

| 范围 | 当前状态 | 已有证据/缺口 |
|---|---|---|
| P2.5 十层冠层、Farquhar、Ball–Berry、叶温 | 代码与 NumPy/JAX/CUDA 组件已实现 | 单叶/组件测试存在；FLUXNET 多站瞬时通量仍未达到完整验收闭环 |
| P2.6 双流辐射与植被能量迭代 | 已实现 | 方程、守恒和设备 parity 测试存在；缺 CTSM/Noah-MP 官方同条件比较 |
| P2.7 冠层空气温湿诊断 | 已实现（简化诊断） | 进入叶—气交换；无独立 prognostic 空气质量状态，需参考验证 |
| P2.8 五层雪/SNICAR | **extensive 路径部分实现，入口不一致** | `snowpack.py`、`snicar.py`、JAX/CUDA 光学内核和测试存在；legacy 三层 ABI 仍拒绝冬季，JAX/CUDA legacy option 3 会退回/拒绝，缺安全 sidecar ABI（半径、BC、四尘类）；不能宣称所有入口完成 |
| P3.9 湿球相态 | 已实现 | option 5 及三后端组件测试；外部 Wang/HRLDAS 多站验证仍待 |
| P3.10 径流 | VIC(6) 已实现 | 新安江(7)、Dynamic VIC(8) 明确未实现；至少 VIC 满足本阶段最低要求 |
| P3.11 动态植被 | 已实现（GDD 物候/LAI） | 三后端组件测试；季节/多年观测验证仍待 |
| P4.12–P4.15 C/N | 模块、BGC 接线和三后端组件已实现 | 仍缺长期准平衡（漂移 <5%/百年）及 FLUXNET GPP/NEE 多站留出数字 |

P1 源码审计另见 [P1_SOURCE_AUDIT.md](P1_SOURCE_AUDIT.md)：融雪余热路由、AHC
局部平滑和厚雪均温不变式已有实际证据；AHC 全域 5618/5529 差异和非均温砂土
float32 漂移仍是已知限制，未重复 M1 全量 ΔT 方案。

## 当前阻塞

为运行 extensive 五层雪/SNICAR NumPy 验收，1882 因 compute002 看不到 manager
`/tmp` stage 在 setup 阶段 Exit94；1883 改用共享 `/home` 后因 fileset ENOSPC
在解包阶段 Exit94。两次均为 0 pytest/0 模型执行，不能写成科学失败或通过。下一
次绕行应直接使用 compute002 已存在的源目录、把结果写 `$TMPDIR`，不再向满
fileset 解包 stage。
