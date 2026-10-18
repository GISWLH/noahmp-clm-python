# 完整柱的SAI接口缺口（2026-10-18）

当前493源的独立冠层组件接受`stem_area`，将叶、茎共同用于光学/热交换/
截留；`coverage`的干叶面积比例排除了茎，避免茎发生气孔蒸腾。但完整
`CLMColumnState`及`step_clm_enthalpy_column`没有SAI状态/输入，整柱的分层
面积、截留容量、地面交换衰减及MOST植物面积只由LAI构造。对应的原生
完整柱也没有茎面积输入。这是组件未接入完整柱的缺口，属于P2完成范围。

EHPC PBS147已实际读取[官方全球地表数据及四站邻域原值](v1-results/ctsm-reference-input-next-b/manifest.json)。
其`MONTHLY_SAI`为time×lsmpft变量；四个邻居、全部PFT原值中，Cng/Qia/Cha/HaM
最大SAI分别2.59220747、1.21784350、3.56976241、2.19315722 m²/m²。这些是
粗网格各PFT的范围，不是站点实测、拟合参数或已选定的站点SAI。
数据证明有真实参考字段可用，尚未证明它是此前H/土温等偏差的原因。

已只读核对固定上游
[`CNVegStructUpdateMod`](https://github.com/ESCOMP/CTSM/blob/fdfc03a0abec838ec0b66663a6a2f5e619cd2611/src/biogeochem/CNVegStructUpdateMod.F90)：
CN模式的总SAI具有前步记忆、叶面积减少项及PFT下限/衰减，另区分雪埋后的
暴露SAI；作物还有独立几何/收获残茬处理。因此不能任意用一个未验证的
“茎碳×系数”替代全部情形，也不能只把读取SAI字段当作完成物理接入。
这里只理解接口和物理定义，不复制上游实现。

后续实现需明确规定SAI与动态C/N结构的关系，守恒状态及检查点携带；把实际
叶/茎几何一致接入截留、辐射、能量、地面交换和MOST，并区分总量/雪埋暴露量。
先完成NumPy独立参考，覆盖无叶有茎、落叶/收获、雨雪截留和水热/C/N，再同步
JAX/CUDA，在原10%数值、同精度1e−6及独立守恒门内核对。不得按现有通量评分
选择SAI或改原年度/初态；当前K/L和493源完整回归继续保持原身份，不能作为
该尚未实现扩展的验证。
