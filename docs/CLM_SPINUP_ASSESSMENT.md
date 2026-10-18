# 完整气候循环的收敛核对入口（2026-10-17）

`benchmarks/assess_clm_spinup.py`用于读取真正完成的原生完整年及其循环衔接，
补齐[四站气候周期协议](FLUXNET_FIRST_CLIMATE_CYCLE.md)的端点核对。
当前尚无实际收敛的完整气候spinup结果；下列测试完成的是评价入口。

2026-10-18后续：评价器现按显式配置读取16或19数组，完整保留SAI/几何/方案，
检查主/影子冷种子及跨年全部数组/账目的原字节。年度HRLDAS检查也校验19数组
及最终SAI与轨迹/固定配置的一致性，SAI不作为额外碳库计入守恒存量。
[新版69项目标](v1-results/stem-assessment-targets-b/manifest.json)已在EHPC PBS164
compute002完成，9.676266秒，0失败/错误/跳过、PBS退出0。
另对xuanze实际两柱四步检查点核对，两柱独立土温解码差5.026704e-6/
9.116343e-6 K，SAI为0.8000000119/0.7988895774；原守恒界通过。入口将该
短轨迹明确拒绝为“必须恰好一个完整原强迫日历年”，未生成实际spinup通过。
早一输出命名版本的[同一实际检查点核对](v1-results/stem-assessment-actual-a/manifest.json)
另存；[A版完整CPU899](v1-results/stem-assessment-full-a/manifest.json)随后实际全部
通过，813.372725秒、零失败/错误/跳过、PBS160退出0。B只将水热/SAI汇总键
及其两条测试断言改名，已独立重跑69项；不将A899描述为单次B899执行。

每段必须保留原17520/17568步（或逐小时等价）的完整平年/闰年强迫、原请求、
`complete`状态、源和输入哈希、不可变UUID报告/轨迹与检查点。全局完成/身份
检查先于跟随列路径；列顺序、固定地方标准时、配置、全部气象区间逐一核对。
首次零步的全部16/19数组须与原配置/首步环境的同精度冷种子逐字节一致；后续
只接受`cycle`链，原强迫不变，全部状态及累计账目除段内index外须同前段末态
逐字节一致。不能省略此前循环、用普通连续年冒充重复气候，或重置数值预算。

水热门沿用预先锁定值：各土层温差最大≤0.1 K、各层总土水差最大≤0.001 m³/m³、
SWE差≤0.5 mm，至少两轮且连续两次端点检查满足。首轮将完整年末与已认证的
冷种子同季节相位比较，后续与前一完整年末比较。温度对原焓存量使用同一NumPy
解码器；另报其与原生末态温度的差，差异超过0.1 K收敛尺度时拒绝评价。
总土水/SWE的原生末态别名与检查点存量须精确对应。原逐柱逐步/累计水热C/N
数值界和最终广延存量核对独立执行，不能用上述收敛阈值豁免守恒。

配置SAI时另检验同季节端点：非近零面积相对变化≤10%；原面积≤1e-4 m²/m²
使用固定绝对变化≤1e-6 m²/m²，不对零面积做相对除法。同样至少两轮、连续
两次满足，分别报告 `canopy_structure_converged` 和水热结论；组合键为
`all_columns_water_heat_sai_converged`，只表示所列水热/SAI端点，不能解释为
植物C/N、物候或全部状态已经达到平衡。旧水热/SOC/百年变化及数值守恒界不变。

碳部分按名称保留全部七库的绝对变化和有符号速率，并分别汇总七库总量与
快/慢/被动三库SOC。凋落物及粗木不混成SOC，凋落物增加不能抵消土壤碳减少。
完整一气候年的各组总C变化折算为`10000 × ΔC/C_before`，单位为%/100重复气候年；
该速率与≤5%条件分别报告，明确是年度速率折算。总C或单库≤1 gC/m²时不做
相对除法，保留绝对变化及null，未新增近零碳的通过门。累计获得至少100个
真实完整循环后，另列最近100循环的七库总量、SOC三库合计及各单库实际变化百分比。短期折算不变成真实
百年观测，工具不会自动宣称长期土壤碳平衡、总N稳态或观测/校准/留出验收。
正氮沉降、简化N损失边界及其他[已知限制](KNOWN_LIMITATIONS.md)仍保留。

在EHPC PBS计算节点使用；输入文件可按原basename复制到`--input-root`，原
batch字节与每份强迫/元数据/配置哈希保持不变。示例中的循环目录应替换为
将来实际完成的目录，当前J未完成年度不能直接用于下列成功路径：

```bash
python benchmarks/assess_clm_spinup.py \
  --batch ORIGINAL_BATCH.json --input-root COPIED_ORIGINAL_INPUTS \
  --model-source-manifest ORIGINAL_MODEL_SOURCE_SHA256.json \
  --segments COMPLETE_COLD_CYCLE COMPLETE_NEXT_CYCLE \
  --output NEW_ASSESSMENT.json
```

EHPC compute002的PBS99完成79项目标，6.14秒、零失败/跳过，488源前后哈希不变。
包括24项新增端点/平闰年/身份/断链/账目/近零/伪百年及凋落物/SOC抵消检查，以及55项既有原生
运行器和年度评价回归。原J48的原请求/配置/强迫身份不改，实际调用被明确
拒绝为`Unchanged complete original year required`，未生成spinup成功报告。

PBS96首轮77项中1项因改列身份后先读取不存在路径而抛FileNotFoundError，
没有接受坏输入。修订将全局完成/身份/不可变描述符检查前移，新增缺失描述符
拒绝案例，PBS97的78项完成；随后C另列三库SOC及一项抵消检查，形成当前79项。
[原A失败/B目标、真实J48拒绝和输入元数据](v1-results/spinup-assessment-b/manifest.json)与
[当前C目标和覆盖记录](v1-results/spinup-assessment-c/manifest.json)均保留。

PBS100逐节点核对当前804个CPU测试：原PBS95的780项加PBS99新增24项，55项
交叠只计一次；原483份代码/测试/基准文件及174份运行库字节一致。此处报告
分布式覆盖，保留每份原日志与范围，本次单次目标pytest输出为79项。
[xuanze完整预收集](v1-results/spinup-current-collection-c/manifest.json)为1129项，
325个设备节点与继续运行的MOST冻结队列完全相同。没有改动该队列或J年度的
物理源码，也未把收集节点当作设备执行通过；完整设备、年度和实际spinup仍待完成。
