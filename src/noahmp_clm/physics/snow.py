"""积雪 bulk 状态、反照率与融化。

v0.1 简化：雪盖存在时地表即雪面，能量平衡按雪反照率求解 Ts，
雪内热传导加热雪体，雪温到 0°C 后的盈余能量转为融化。
反照率方案参考 Noah-MP ``SnowAlbedoBatsMod.F90`` 与
``SnowAlbedoClassMod.F90``；SNICAR (option 3) 留给后续阶段。
"""
from __future__ import annotations

import numpy as np

LF = 3.34e5    # 融化潜热 J/kg
RHO_W = 1000.0
RHO_SNOW = 300.0
C_SNOW = RHO_SNOW * 2100.0  # 雪体积热容量 J/(m3 K)
K_SNOW = 0.30               # 雪导热系数 W/(m K)
MAX_SNOW_LAYERS = 3
SNOW_LAYER_THRESHOLDS = (0.05, 0.15)  # m SWE, SnowLayerDivideMod 风格阈值


def snow_depth(swe: float) -> float:
    return swe * RHO_W / RHO_SNOW


def snow_thermal_conductivity(
    density: float, temperature: float = 273.15, option: int = 3
) -> float:
    """按 ``OptSnowThermConduct`` 返回雪导热系数 (W m-1 K-1)。

    option 1 为 Stieglitz/Yen (1965)，2 为 Anderson (1976)，3 为常数，
    4 为 Verseghy (1991)，5 为 Douville/Yen (1981)。温度参数为接口一致性
    保留，Verseghy 分支用它反映暖湿雪的轻微增益。
    """
    rho = float(np.clip(density, 50.0, 917.0))
    ratio = rho / 1000.0
    if option == 1:
        value = 2.2 * ratio**1.88
    elif option == 2:
        value = 2.9 * ratio**2.0
    elif option == 3:
        value = K_SNOW
    elif option == 4:
        value = 0.023 + 0.234 * ratio**2.0
        value *= 1.0 + 0.05 * np.clip(float(temperature) - 268.15, 0.0, 5.0) / 5.0
    elif option == 5:
        value = 2.2 * ratio**1.88 * (1.0 + 0.1 * ratio)
    else:
        raise ValueError(f"OptSnowThermConduct={option} 无效；允许值为 1、2、3、4、5")
    return float(np.clip(value, 0.05, 2.5))


def divide_snow_layers(swe: float, *, max_layers: int = MAX_SNOW_LAYERS) -> np.ndarray:
    """按 SnowLayerDivideMod 的 SWE 阈值返回各层 SWE（top -> bottom）。"""
    total = max(0.0, float(swe))
    n = 1 if total <= SNOW_LAYER_THRESHOLDS[0] else 2 if total <= SNOW_LAYER_THRESHOLDS[1] else max_layers
    n = int(np.clip(n, 1, MAX_SNOW_LAYERS))
    if total <= 0.0:
        return np.zeros(1, dtype=float)
    if n == 1:
        return np.array([total], dtype=float)
    weights = np.ones(n, dtype=float) / n
    return total * weights


def combine_snow_layers(
    swe_layers: np.ndarray, temperatures: np.ndarray, densities: np.ndarray,
    *, max_layers: int = MAX_SNOW_LAYERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """合并小层并限制至最多 3 层（SnowLayerCombineMod 风格）。"""
    swe = np.asarray(swe_layers, dtype=float)
    temp = np.asarray(temperatures, dtype=float)
    dens = np.asarray(densities, dtype=float)
    keep = swe > 1.0e-8
    if not np.any(keep):
        return np.zeros(1), np.array([273.15]), np.array([RHO_SNOW])
    swe, temp, dens = swe[keep], temp[keep], dens[keep]
    while len(swe) > max_layers:
        i = len(swe) - 2
        mass = swe[i] + swe[i + 1]
        temp[i] = (swe[i] * temp[i] + swe[i + 1] * temp[i + 1]) / mass
        dens[i] = mass / (swe[i] / dens[i] + swe[i + 1] / dens[i + 1])
        swe[i] = mass
        swe = np.delete(swe, i + 1)
        temp = np.delete(temp, i + 1)
        dens = np.delete(dens, i + 1)
    return swe.copy(), temp.copy(), dens.copy()


def add_snowfall(state, snowfall_m: float, snow_temperature: float = 273.15) -> None:
    """把新雪加入顶层并按阈值重分层；状态对象已由驱动复制。"""
    amount = max(0.0, float(snowfall_m))
    if amount <= 0.0:
        return
    old_swe = np.asarray(state.snow_swe_layers, dtype=float)
    old_t = np.asarray(state.snow_t_layers, dtype=float)
    old_d = np.asarray(state.snow_density_layers, dtype=float)
    if np.sum(old_swe) <= 1.0e-8:
        new_swe = np.array([amount])
        new_t = np.array([snow_temperature])
        new_d = np.array([100.0])
    else:
        new_swe = np.concatenate(([amount], old_swe))
        new_t = np.concatenate(([snow_temperature], old_t))
        new_d = np.concatenate(([100.0], old_d))
    target = divide_snow_layers(float(np.sum(new_swe)))
    mean_t = float(np.average(new_t, weights=new_swe))
    mean_d = float(np.average(new_d, weights=new_swe))
    state.snow_swe_layers = target
    state.snow_t_layers = np.full(len(target), mean_t)
    state.snow_density_layers = np.full(len(target), mean_d)
    state.swe = float(np.sum(target))
    state.snow_t = float(state.snow_t_layers[0])


def surface_snow_flux(state, surface_temperature: float) -> float:
    """按顶层厚度计算雪面向下导热通量 (W/m2)。"""
    top_swe = max(float(state.snow_swe_layers[0]), 1.0e-8)
    top_depth = top_swe * RHO_W / max(float(state.snow_density_layers[0]), 50.0)
    conductivity = snow_thermal_conductivity(
        state.snow_density_layers[0], state.snow_t_layers[0],
        state.options.OptSnowThermConduct
    )
    return conductivity * (float(surface_temperature) - float(state.snow_t_layers[0])) / max(
        0.5 * top_depth, 0.01
    )


def compact_snow(state, dt: float, option: int = 1) -> None:
    """更新雪层密度（Anderson, 1976；状态已在驱动入口复制）。

    option 1 使用温度驱动的破坏性沉降和上覆载荷压实；option 2 采用
    Abolafia--Rosenzweig et al. (2024) 的更慢温度响应占位实现。两者都只
    改变密度/厚度，不改变 SWE。
    """
    if option not in (1, 2):
        raise ValueError(f"OptSnowCompaction={option} 无效；允许值为 1 或 2")
    swe, temp, density = combine_snow_layers(
        state.snow_swe_layers, state.snow_t_layers, state.snow_density_layers
    )
    if np.sum(swe) <= 1.0e-8:
        return
    step = max(0.0, float(dt))
    # Anderson (1976) 的温度依赖沉降：暖雪更快接近 550 kg/m3。
    warm_rate = (1.0e-7 if option == 1 else 6.0e-8) * np.exp(
        np.clip(0.08 * (temp - 258.15), -4.0, 4.0)
    )
    overburden = np.cumsum(swe[::-1])[::-1] - swe
    load_rate = (2.5e-7 if option == 1 else 1.5e-7) * overburden
    density = np.minimum(550.0, density + step * (warm_rate * (550.0 - density) + load_rate * 1.0e4))
    state.snow_swe_layers = swe
    state.snow_t_layers = temp
    state.snow_density_layers = density
    state.swe = float(np.sum(swe))
    state.snow_t = float(temp[0])


def _implicit_heat_step(
    temperatures: np.ndarray, thickness: np.ndarray, dt: float,
    top_flux: float, conductivity: np.ndarray,
) -> np.ndarray:
    """三对角隐式层间导热，底边界零通量。"""
    n = len(temperatures)
    cap = C_SNOW * thickness / max(dt, 1.0e-6)
    conductance = np.zeros(max(0, n - 1))
    for i in range(n - 1):
        k_if = 2.0 / (1.0 / conductivity[i] + 1.0 / conductivity[i + 1])
        conductance[i] = k_if / max(0.5 * (thickness[i] + thickness[i + 1]), 1.0e-4)
    lower = np.zeros(n)
    diag = cap.copy()
    upper = np.zeros(n)
    rhs = cap * temperatures
    rhs[0] += top_flux
    for i, g in enumerate(conductance):
        diag[i] += g
        diag[i + 1] += g
        upper[i] = -g
        lower[i + 1] = -g
    # Thomas algorithm, written for the fixed small layer count.
    for i in range(1, n):
        factor = lower[i] / max(diag[i - 1], 1.0e-12)
        diag[i] -= factor * upper[i - 1]
        rhs[i] -= factor * rhs[i - 1]
    result = np.empty(n, dtype=float)
    result[-1] = rhs[-1] / max(diag[-1], 1.0e-12)
    for i in range(n - 2, -1, -1):
        result[i] = (rhs[i] - upper[i] * result[i + 1]) / max(diag[i], 1.0e-12)
    return result


def albedo_bats(age_days: float, snow_temperature: float, new_snow: bool = False) -> float:
    """BATS (Dickinson et al., 1993) 指数雪龄衰减，返回雪反照率。"""
    del snow_temperature  # BATS v0.1 方案只按雪龄衰减
    age = 0.0 if new_snow else max(0.0, float(age_days))
    return float(np.clip(0.55 + 0.30 * np.exp(-age / 3.0), 0.50, 0.90))


def albedo_class(age_days: float, snow_temperature: float, new_snow: bool = False) -> float:
    """CLASS (Verseghy, 1991) 的雪龄/温度分段反照率。

    冷雪采用较慢的 grain-growth 衰减，接近融点的湿雪采用较快衰减；
    降雪会将雪龄重置为零。分段系数是单柱基线的 CLASS 简化，保持与
    Noah-MP ``SnowAlbedoClassMod.F90`` 相同的物理方向。
    """
    age = 0.0 if new_snow else max(0.0, float(age_days))
    fresh = 0.84 if float(snow_temperature) < 271.15 else 0.70
    rate = 0.02 if float(snow_temperature) < 268.15 else 0.06
    floor = 0.65 if float(snow_temperature) < 271.15 else 0.50
    return float(np.clip(fresh - rate * np.sqrt(age), floor, fresh))


SNOW_ALBEDO = {1: albedo_bats, 2: albedo_class}


def update_snow_albedo(
    swe: float,
    albedo: float,
    age_days: float,
    snow_temperature: float,
    dt: float,
    snowfall_m: float,
    option: int,
) -> tuple[float, float]:
    """纯函数形式更新 ``(albedo, age_days)``，供状态驱动调用。"""
    if swe <= 1.0e-6:
        return 0.8, 0.0
    if option not in SNOW_ALBEDO:
        raise NotImplementedError(
            f"OptSnowAlbedo={option} 尚未实现；阶段 1 支持 1 (BATS)、2 (CLASS)"
        )
    is_new = snowfall_m > 1.0e-6
    age = 0.0 if is_new else max(0.0, float(age_days)) + max(0.0, float(dt)) / 86400.0
    value = SNOW_ALBEDO[option](age, snow_temperature, new_snow=is_new)
    return float(value), float(age)


def age_albedo(state, dt: float, snowfall_m: float, option: int = 1) -> None:
    """更新状态中的反照率（状态已在 ``step_column`` 入口复制）。"""
    state.snow_albedo, state.snow_age_days = update_snow_albedo(
        state.swe, state.snow_albedo, state.snow_age_days, state.snow_t,
        dt, snowfall_m, option
    )


def apply_snow_heat(state, dt: float, g_snow: float) -> float:
    """用隐式层间导热加热雪体；雪温超过 0°C 的能量转为融水 (m)。"""
    if state.swe <= 1e-6:
        return 0.0
    swe, temps, density = combine_snow_layers(
        state.snow_swe_layers, state.snow_t_layers, state.snow_density_layers
    )
    thickness = swe * RHO_W / np.maximum(density, 50.0)
    conductivity = np.array([
        snow_thermal_conductivity(d, t, state.options.OptSnowThermConduct)
        for d, t in zip(density, temps)
    ])
    updated = _implicit_heat_step(temps, thickness, dt, g_snow, conductivity)
    melt = 0.0
    for i in range(len(updated)):
        if updated[i] > 273.15:
            excess = (updated[i] - 273.15) * C_SNOW * thickness[i]
            layer_melt = min(swe[i], excess / (LF * RHO_W))
            swe[i] -= layer_melt
            melt += layer_melt
            updated[i] = 273.15
    swe, updated, density = combine_snow_layers(swe, updated, density)
    state.snow_swe_layers = swe
    state.snow_t_layers = updated
    state.snow_density_layers = density
    state.swe = float(np.sum(swe))
    state.snow_t = float(updated[0])
    return float(melt)
