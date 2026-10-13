"""Reproducible NumPy canopy energy audit; references are synthetic equations.

Run from repository root: python benchmarks/check_canopy_energy.py --output ...
Uses the independent nonlinear solver in the tests (SciPy required).
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

from noahmp_clm.physics.canopy_energy import CanopyExchange, solve_canopy_energy
from noahmp_clm.physics.soil_uptake import SoilWaterSupply

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("energy_reference",ROOT/"tests/test_canopy_energy.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def random_cases(count=96):
    rng = np.random.default_rng(20261013)
    for i in range(count):
        tair = rng.uniform(274.,307.)
        es = 611.2*np.exp(17.67*(tair-273.15)/(tair-29.65))
        pressure = rng.uniform(70000.,102000.)
        vapor = rng.uniform(.25,.95)*es
        humidity = .622*vapor/(pressure-.378*vapor)
        e = CanopyExchange(tair,humidity,pressure,rng.uniform(.008,.05),
            rng.uniform(.001,.008),rng.uniform(.015,.06),tair+rng.uniform(-5.,2.),
            rng.uniform(5.,35.),ground_resistance=rng.uniform(0.,500.))
        area = rng.uniform(.01,1.,10)*rng.uniform(.1,1.5)
        kwargs = {"vegetated_fraction":rng.uniform(.05,1.),"wet_fraction":rng.uniform(0.,.9),
            "water_stress":rng.uniform(.1,1.),"nitrogen_factor":rng.uniform(.4,1.),
            "cosine_zenith":rng.uniform(.1,1.),"diffuse_fraction":rng.uniform(.1,1.)}
        if i % 3 == 0:
            kwargs.update(canopy_evaporation_limit=rng.uniform(0.,2e-5),
                transpiration_limit=rng.uniform(0.,5e-5),ground_evaporation_limit=rng.uniform(0.,2e-5))
        if i % 2 == 0:
            roots = rng.uniform(0.,3e-5,4)
            kwargs["soil_supply"] = SoilWaterSupply(roots,rng.uniform(0.,1.,4),roots[0]+rng.uniform(0.,1e-5))
        yield f"random_{i:03d}",area,rng.uniform(0.,1000.),rng.uniform(.7,1.)*5.670374419e-8*tair**4,e,kwargs


def audit(count=96):
    cases = list(reference.CASES)+list(random_cases(count))
    rows = []
    for name,area,sw,lw,e,kw in cases:
        start = time.perf_counter()
        try:
            actual = solve_canopy_energy(area,sw,lw,e,**kw)
            seconds = time.perf_counter()-start
            expected = reference.independent_energy_reference(area,sw,lw,e,**kw)
        except (RuntimeError,AssertionError,ValueError) as error:
            rows.append({"name":name,"error":str(error)})
            continue
        row = {"name":name,"iterations":actual["iterations"],"numpy_seconds":seconds,
            "temperature_max_abs_k":float(np.max(np.abs(np.r_[actual["leaf_temperature"],actual["ground_temperature"]]-expected["temperature"]))),
            "humidity_abs_kgkg":abs(float(actual["canopy_humidity"]-expected["canopy_humidity"])),
            "layer_residual_max_wm2":float(np.max(np.abs(actual["layer_residual"]))),
            "column_residual_wm2":float(actual["residual"]),
            "air_heat_residual_wm2":float(actual["air_heat_residual"]),
            "air_water_residual_kgm2s":float(actual["air_water_residual"])}
        for key in ("h","le","ground_heat","gpp"):
            row[key] = {"value":float(actual[key]),"reference":float(expected[key]),
                "absolute_error":abs(float(actual[key]-expected[key]))}
        rows.append(row)
    good = [r for r in rows if "error" not in r]
    summary = {"cases":len(rows),"failed_cases":len(rows)-len(good),
               "reference":"independent SciPy simultaneous nonlinear equation solve; not CTSM/FLUXNET observations"}
    for key in ("temperature_max_abs_k","humidity_abs_kgkg","layer_residual_max_wm2",
                "column_residual_wm2","air_heat_residual_wm2","air_water_residual_kgm2s"):
        summary[key] = max(abs(r[key]) for r in good)
    summary["maximum_iterations"] = max(r["iterations"] for r in good)
    for key,zero in (("h",1.),("le",1.),("ground_heat",1.),("gpp",.1)):
        records = [r[key] for r in good]
        nonzero = [r for r in records if abs(r["reference"])>zero]
        nearzero = [r for r in records if abs(r["reference"])<=zero]
        summary[key] = {"max_absolute_error":max(r["absolute_error"] for r in records),
            "max_relative_error_nonzero":max((r["absolute_error"]/abs(r["reference"]) for r in nonzero),default=None),
            "nearzero_count":len(nearzero),"nearzero_max_absolute_error":max((r["absolute_error"] for r in nearzero),default=None)}
    paths = ["src/noahmp_clm/physics/canopy_energy.py","src/noahmp_clm/physics/soil_uptake.py","src/noahmp_clm/physics/photosynthesis.py",
             "src/noahmp_clm/physics/canopy_radiation.py","tests/test_canopy_energy.py","benchmarks/check_canopy_energy.py"]
    return {"summary":summary,"source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},"cases":rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--random-cases",type=int,default=96)
    args = parser.parse_args()
    result = audit(args.random_cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result["summary"],indent=2))
    if result["summary"]["failed_cases"]:
        raise SystemExit(1)
