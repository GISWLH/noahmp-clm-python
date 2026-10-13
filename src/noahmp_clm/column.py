"""单柱状态与强迫。

一个 ColumnState 就是一个完全独立的 1D 陆面柱（土壤—积雪—冠层）的全部记忆。
两个 ColumnState 之间不存在任何引用与共享，这是"格点完全独立"的代码级保证。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .options import PhysicsOptions


@dataclass
class SoilParams:
    """土壤水力/热力参数（一组典型值，非逐站率定）。

    典型值量级参考 Cosby et al. (1984) 与 Noah LSM 参数表风格；
    v0.1 只求量级正确，精确对标是阶段 1 的工作。
    """
    name: str = "loam"
    smcmax: float = 0.45   # 孔隙度 (m3/m3)
    smcwlt: float = 0.12   # 萎蔫点 (m3/m3)
    smcref: float = 0.28   # 田间持水量 (m3/m3)
    satdk: float = 1.0e-5  # 饱和导水率 (m/s)
    bexp: float = 5.5      # Clapp-Hornberger 指数 b
    quartz: float = 0.5    # 石英含量（热传导用）


def default_soil_params(name: str = "loam") -> SoilParams:
    table = {
        "sand": dict(smcmax=0.40, smcwlt=0.06, smcref=0.18,
                     satdk=5.0e-5, bexp=4.0, quartz=0.9),
        "loam": dict(smcmax=0.45, smcwlt=0.12, smcref=0.28,
                     satdk=1.0e-5, bexp=5.5, quartz=0.5),
        "clay": dict(smcmax=0.50, smcwlt=0.25, smcref=0.40,
                     satdk=1.0e-6, bexp=9.0, quartz=0.2),
    }
    if name not in table:
        raise ValueError(f"unknown soil texture '{name}', choose from {list(table)}")
    return SoilParams(name=name, **table[name])


@dataclass
class ColumnState:
    """单柱全部状态。numpy 数组在 step_column 入口处被复制，保证纯函数语义。"""
    dz: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.3, 0.6, 1.0]))
    tsoil: np.ndarray | None = None   # 各层土壤温度 (K)
    smc: np.ndarray | None = None     # 各层体积含水量 (m3/m3)
    ts: float = 285.0                 # 地表（雪面）温度 (K)
    swe: float = 0.0                  # 雪水当量 (m)
    snow_t: float = 273.15            # 雪温 (K)
    snow_albedo: float = 0.8          # 雪反照率
    snow_age_days: float = 0.0        # 雪龄 (天)，控制反照率衰减
    canw: float = 0.0                 # 冠层截留水 (m)
    lai: float = 3.0                  # 叶面积指数
    fveg: float = 0.8                 # 植被覆盖度
    z0m: float = 0.05                 # 动量粗糙度 (m)
    albedo: float = 0.2               # 无雪地表反照率
    emiss: float = 0.97               # 地表发射率
    rsmin: float = 120.0              # 最小气孔阻抗 (s/m)
    soil: SoilParams = field(default_factory=SoilParams)
    root_frac: np.ndarray | None = None  # 各层根系权重（归一）
    options: PhysicsOptions = field(default_factory=PhysicsOptions)
    terrain_index: float = 0.0          # TOPMODEL 地形指数
    groundwater_depth: float = 2.0      # 地下水位深度 (m)
    soil_ice: np.ndarray | None = None  # 各层冰含量 (m3/m3)
    snow_swe_layers: np.ndarray | None = None  # 各层雪水当量 (m), 最多 3 层
    snow_t_layers: np.ndarray | None = None    # 各层雪温 (K)
    snow_density_layers: np.ndarray | None = None  # 各层密度 (kg/m3)

    def __post_init__(self):
        n = len(self.dz)
        if self.tsoil is None:
            self.tsoil = np.full(n, 285.0)
        if self.smc is None:
            self.smc = np.full(n, 0.5 * (self.soil.smcwlt + self.soil.smcref))
        if self.soil_ice is None:
            self.soil_ice = np.zeros(n, dtype=float)
        else:
            self.soil_ice = np.asarray(self.soil_ice, dtype=float).copy()
            if len(self.soil_ice) != n:
                raise ValueError("soil_ice 与 soil 层数必须一致")
        if self.snow_swe_layers is None:
            self.snow_swe_layers = np.array([max(0.0, self.swe)], dtype=float)
        else:
            self.snow_swe_layers = np.asarray(self.snow_swe_layers, dtype=float)
            if self.snow_swe_layers.ndim != 1 or len(self.snow_swe_layers) == 0:
                raise ValueError("snow_swe_layers 必须是一维且至少含一层")
            self.snow_swe_layers = self.snow_swe_layers[-3:].copy()
            if self.swe == 0.0:
                self.swe = float(np.sum(self.snow_swe_layers))
        n_snow = len(self.snow_swe_layers)
        if self.snow_t_layers is None:
            self.snow_t_layers = np.full(n_snow, self.snow_t, dtype=float)
        else:
            self.snow_t_layers = np.asarray(self.snow_t_layers, dtype=float).copy()
            if len(self.snow_t_layers) != n_snow:
                raise ValueError("snow_t_layers 与 snow_swe_layers 长度必须一致")
        if self.snow_density_layers is None:
            self.snow_density_layers = np.full(n_snow, 300.0, dtype=float)
        else:
            self.snow_density_layers = np.asarray(self.snow_density_layers, dtype=float).copy()
            if len(self.snow_density_layers) != n_snow:
                raise ValueError("snow_density_layers 与 snow_swe_layers 长度必须一致")
        if self.root_frac is None:
            zc = np.cumsum(self.dz) - 0.5 * self.dz
            rf = np.exp(-3.0 * zc)
            self.root_frac = rf / rf.sum()

    def copy(self) -> "ColumnState":
        import dataclasses
        return dataclasses.replace(
            self,
            dz=self.dz.copy(), tsoil=self.tsoil.copy(), smc=self.smc.copy(),
            soil_ice=self.soil_ice.copy(),
            root_frac=self.root_frac.copy(),
            snow_swe_layers=self.snow_swe_layers.copy(),
            snow_t_layers=self.snow_t_layers.copy(),
            snow_density_layers=self.snow_density_layers.copy(),
        )


@dataclass
class Forcing:
    """单步大气强迫（陆面模型标准强迫变量）。"""
    dt: float       # 步长 (s)
    tair: float     # 2m 气温 (K)
    qair: float     # 2m 比湿 (kg/kg)
    wind: float     # 10m 风速 (m/s)
    ps: float       # 气压 (Pa)
    swdown: float   # 入射短波 (W/m2)
    lwdown: float   # 入射长波 (W/m2)
    prcp: float     # 降水率 (m/s，水当量)
