# GPU 状态向量规约

阶段 A 将一个 `ColumnState` 编码为两个定长数组。这样 JAX 的 `jit/vmap/scan`
和后续 CUDA kernel 不需要处理 dataclass、可变长度数组或 Python 字符串。`dz`
是固定的 `[0.1, 0.3, 0.6, 1.0] m`（也可以由调用者传入同样长度的编译期常量），
不进入向量；`SoilParams.name` 只是诊断标签，不进入 ABI，解包时恢复为
`"packed"`。

## 浮点段 `fvec`（float64，长度 70）

明确的当前物理字段占用前 45 个槽位，槽位 45 是一个保留位。槽位 46--69
为零填充扩展区，使浮点段与阶段 3 约 70 槽位的 ABI 预算一致；后续增加状态时
可以使用这些槽位而不改变已有偏移。

| 字段 | 偏移 | 长度 | 说明 |
|---|---:|---:|---|
| `tsoil` | 0 | 4 | 四层土壤温度 (K) |
| `smc` | 4 | 4 | 四层体积含水量 |
| `soil_ice` | 8 | 4 | 四层冰含量 |
| `root_frac` | 12 | 4 | 四层根系权重 |
| `snow_swe_layers` | 16 | 3 | 三层雪水当量，短雪包用 0 填充 |
| `snow_t_layers` | 19 | 3 | 三层雪温，短雪包用 `snow_t` 填充 |
| `snow_density_layers` | 22 | 3 | 三层雪密度，短雪包用 300 kg/m³ 填充 |
| `ts` | 25 | 1 | 地表温度 |
| `swe` | 26 | 1 | 总雪水当量 |
| `snow_t` | 27 | 1 | bulk 雪温 |
| `snow_albedo` | 28 | 1 | 雪反照率 |
| `snow_age_days` | 29 | 1 | 雪龄 |
| `canw` | 30 | 1 | 冠层截留水 |
| `lai` | 31 | 1 | 叶面积指数 |
| `fveg` | 32 | 1 | 植被覆盖度 |
| `z0m` | 33 | 1 | 动量粗糙度 |
| `albedo` | 34 | 1 | 无雪地表反照率 |
| `emiss` | 35 | 1 | 发射率 |
| `rsmin` | 36 | 1 | 最小气孔阻抗 |
| `terrain_index` | 37 | 1 | TOPMODEL 地形指数 |
| `groundwater_depth` | 38 | 1 | 地下水位深度 |
| `soil.smcmax` | 39 | 1 | 孔隙度 |
| `soil.smcwlt` | 40 | 1 | 萎蔫点 |
| `soil.smcref` | 41 | 1 | 田间持水量 |
| `soil.satdk` | 42 | 1 | 饱和导水率 |
| `soil.bexp` | 43 | 1 | Clapp--Hornberger 指数 |
| `soil.quartz` | 44 | 1 | 石英含量 |
| `reserved0` | 45 | 1 | 阶段 3 ABI 保留位，当前为 0 |
| `reserved[1:24]` | 46 | 24 | 阶段 3 扩展保留区，当前为 0 |

## 整型段 `ivec`（int32，长度 33）

偏移 0--31 按 `PhysicsOptions` dataclass 声明顺序保存全部 Noah-MP
`Opt*` 字段，包括阶段 1 关闭但必须保持一一对应的选项。偏移 32 是
`snow_layer_count`（1--3），用于在固定三层数组中恢复原始有效层数。

| 偏移 | 字段 |
|---:|---|
| 0--31 | `PhysicsOptions` 的 `fields(PhysicsOptions)` 顺序 |
| 32 | `snow_layer_count` |

## 强迫向量 `forcing_vec`（float64，长度 8）

| 偏移 | 字段 | 单位 |
|---:|---|---|
| 0 | `dt` | s |
| 1 | `tair` | K |
| 2 | `qair` | kg/kg |
| 3 | `wind` | m/s |
| 4 | `ps` | Pa |
| 5 | `swdown` | W/m² |
| 6 | `lwdown` | W/m² |
| 7 | `prcp` | m/s 水当量 |

## GPU 内存布局约定

阶段 A 的 `column_state_to_vec` 是便于调试和 round-trip 的单柱 AoS 形式。
阶段 F 的 CUDA kernel 会把同一字段转为 SoA：`field[column]`，例如
`tsoil0[c]`、`smc0[c]`，使一个 warp 读取相邻柱时访问连续地址。柱之间没有
任何共享状态或通信；`snow_layer_count` 只屏蔽本柱的无效雪层。
