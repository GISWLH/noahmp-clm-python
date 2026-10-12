"""示例：单个柱 5 天合成强迫运行，打印逐日汇总并输出 CSV。"""
import os
import numpy as np

from noahmp_clm import ColumnState, default_soil_params, synthetic_forcing
from noahmp_clm.driver import run_columns
from noahmp_clm.io import write_csv

os.makedirs("outputs", exist_ok=True)

state = ColumnState(soil=default_soil_params("loam"), lai=3.0)
forcings = synthetic_forcing(n_days=5, dt=1800.0)
(diags, final), = run_columns([state], [forcings])

steps_per_day = int(86400 / 1800)
print(f"{'day':>3} {'ET(mm)':>8} {'PRCP(mm)':>9} {'RUNOFF(mm)':>10} {'Tsoil1(C)':>9} {'SWE(mm)':>8}")
for d in range(5):
    chunk = diags[d * steps_per_day:(d + 1) * steps_per_day]
    et = sum(x["et_mm"] for x in chunk)
    pr = sum(x["prcp_mm"] for x in chunk)
    ro = sum(x["runoff_mm"] for x in chunk)
    t1 = np.mean([x["tsoil"][0] for x in chunk]) - 273.15
    swe = chunk[-1]["swe"] * 1000.0
    print(f"{d+1:>3} {et:>8.2f} {pr:>9.2f} {ro:>10.2f} {t1:>9.2f} {swe:>8.2f}")

worst = max(abs(x["residual"]) for x in diags)
print(f"\n最大能量闭合残差: {worst:.3e} W/m2")
print(f"终态表层土壤含水量: {final.smc[0]:.3f} m3/m3")

write_csv(diags, "outputs/single_column_5d.csv")
print("CSV 已写入 outputs/single_column_5d.csv")
