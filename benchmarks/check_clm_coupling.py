"""Snow-free NumPy column conservation test, NOT a site/observational score."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from noahmp_clm.column import ColumnState, Forcing
from noahmp_clm.clm_driver import step_clm_column
from noahmp_clm.physics.energy import qsat


def audit(days=7):
    dt = 1800.
    states = {
        "vegetated":ColumnState(ts=294.,tsoil=np.full(4,292.),smc=np.full(4,.28)),
        "drying":ColumnState(ts=294.,tsoil=np.full(4,292.),smc=np.array([.121,.125,.14,.14])),
        "bare":ColumnState(ts=294.,tsoil=np.full(4,292.),smc=np.full(4,.28),fveg=0.),
    }
    summaries,series = {},{}
    fields = ("h","le","g","rnet","gpp_umol_m2s","et_mm","runoff_mm",
              "prcp_mm","water_residual_m","latent_water_residual_wm2","residual",
              "canw","ts","canopy_temperature","canopy_humidity","canopy_iterations")
    start = time.perf_counter()
    for name,initial in states.items():
        s = initial
        rows = []
        for step in range(days*48):
            hour = step*.5 % 24.
            sun = max(0.,np.sin(np.pi*(hour-6.)/12.))
            tair = 294.+5.*np.sin(2.*np.pi*(hour-9.)/24.)
            # Identical forcing across columns; a two-hour rain pulse on day 2.
            rain = 2e-6 if 48+16 <= step < 48+20 else 0.
            f = Forcing(dt,tair,.75*qsat(tair,101325.),3.,101325.,800.*sun,340.,rain)
            s,d = step_clm_column(s,f,cosine_zenith=max(.1,sun),diffuse_fraction=.35)
            rows.append([float(d[k]) for k in fields])
        values = np.asarray(rows)
        series[name] = values
        arrays = dict(zip(fields,values.T))
        storage0 = np.dot(initial.smc,initial.dz)+initial.canw
        storage1 = np.dot(s.smc,s.dz)+s.canw
        water = storage1-storage0+(arrays["et_mm"]+arrays["runoff_mm"]-arrays["prcp_mm"]).sum()/1000.
        summaries[name] = {
            "steps":len(rows),"days":days,"final_water_residual_m":float(water),
            "maximum_step_water_residual_m":float(np.max(np.abs(arrays["water_residual_m"]))),
            "maximum_latent_water_residual_wm2":float(np.max(np.abs(arrays["latent_water_residual_wm2"]))),
            "maximum_surface_energy_residual_wm2":float(np.max(np.abs(arrays["residual"]))),
            "cumulative_surface_energy_residual_jm2":float(arrays["residual"].sum()*dt),
            "total_et_mm":float(arrays["et_mm"].sum()),"total_runoff_mm":float(arrays["runoff_mm"].sum()),
            "total_precipitation_mm":float(arrays["prcp_mm"].sum()),
            "gpp_gcm2":float(arrays["gpp_umol_m2s"].sum()*12.0107e-6*dt),
            "max_iterations":int(arrays["canopy_iterations"].max()),
            "final_soil_temperature_k":s.tsoil.tolist(),"final_soil_moisture":s.smc.tolist(),
        }
        print(name,json.dumps(summaries[name]),flush=True)
    root = Path(__file__).resolve().parents[1]
    paths = ["src/noahmp_clm/clm_driver.py","src/noahmp_clm/physics/canopy_energy.py",
        "src/noahmp_clm/physics/soil_uptake.py","src/noahmp_clm/physics/soil.py",
        "src/noahmp_clm/physics/photosynthesis.py","src/noahmp_clm/physics/canopy_radiation.py",
        "benchmarks/check_clm_coupling.py"]
    report = {"scope":"synthetic snow-free coupling/conservation only; not CTSM/HRLDAS/FLUXNET validation",
        "seconds":time.perf_counter()-start,"cases":summaries,"fields":fields,
        "source_sha256":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}}
    return report,series


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--days",type=int,default=7)
    args = parser.parse_args()
    report,series = audit(args.days)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"),**series)
    report["series_sha256"] = hashlib.sha256(args.output.with_suffix(".npz").read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    assert all(abs(c["final_water_residual_m"])<1e-10 and
               c["maximum_latent_water_residual_wm2"]<1e-8 and
               c["maximum_surface_energy_residual_wm2"]<6e-4 for c in report["cases"].values())
