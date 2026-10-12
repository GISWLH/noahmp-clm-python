"""单柱状态与强迫。

一个 ColumnState 就是一个完全独立的 1D 陆面柱（土壤—积雪—冠层）的全部记忆。
两个 ColumnState 之间不存在任何引用与共享，这是"格点完全独立"的代码级保证。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


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

    def __post_init__(self):
        n = len(self.dz)
        if self.tsoil is None:
            self.tsoil = np.full(n, 285.0)
        if self.smc is None:
            self.smc = np.full(n, 0.5 * (self.soil.smcwlt + self.soil.smcref))
        if self.root_frac is None:
            zc = np.cumsum(self.dz) - 0.5 * self.dz
            rf = np.exp(-3.0 * zc)
            self.root_frac = rf / rf.sum()

    def copy(self) -> "ColumnState":
        import dataclasses
        return dataclasses.replace(
            self,
            dz=self.dz.copy(), tsoil=self.tsoil.copy(), smc=self.smc.copy(),
            root_frac=self.root_frac.copy(),
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
