"""Exploratory cross-curve C3 evaluation; never fits the light-response rows.

Requires the unmodified CC-BY-4.0 C3_grass.zip from Leafweb (DOI below),
NumPy and SciPy. Uses measured ci, so this tests biochemistry, not stomata.
Protocol: dark respiration and maximum PSII yield from the separate dark
record; Vcmax/Jmax/TPU fit only to the measured A-ci rows. Incident PAR is
converted with grass absorptance .84 (CLM5 table 2.3.1); theta=.7. All five
files are retained, including the file with no light-response observations.
The protocol was explored with these data: this is not a blind validation.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.optimize import least_squares
from noahmp_clm.physics.photosynthesis import LeafParameters, farquhar_rates

DOI = "https://doi.org/10.25581/ornlsfa.027/1887896"
SHA256 = "936cff54c647e84bb4fc59e380c6d234888f8e47722635b00d24e55068c83cd5"
DATA_URL = "https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-44175486264e001-20250917T161101540122"


def evaluate(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Leafweb archive differs from the recorded source version")
    result = {"doi": DOI, "data_url": DATA_URL, "sha256": SHA256,
        "license": "CC-BY-4.0", "incident_par_absorptance_assumption": .84,
        "scope": "Exploratory biochemical cross-curve evaluation at measured ci; not a stomatal test or blind holdout",
        "near_zero_definition": "abs(observed net assimilation) <= 1 micromol/m2/s",
        "near_zero_absolute_tolerance": .5,
        "light_curve_acceptance": "NRMSE (RMSE/mean observed net rate) <= 10%; report bias and r separately",
        "files": []}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in sorted(archive.namelist()):
            # The second row is units, not a missing observation.
            rows = list(csv.DictReader(io.StringIO(archive.read(name).decode("cp1252"))))[1:]
            keys = ("CO2i","PARi","Tleaf","AnetCO2","DataType","FmorFm.","FoorFs")
            values = np.array([[float(row[k]) for k in keys] for row in rows])
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Nonfinite measurement in {name}")
            dark = values[values[:,4] == 0]
            calibration = values[(values[:,4]>=11)&(values[:,4]<=25)]
            validation = values[(values[:,4]>=31)&(values[:,4]<=45)]
            if len(dark) != 1 or len(calibration) < 4:
                raise ValueError(f"Missing calibration/dark records in {name}")
            rd_factor = float(farquhar_rates(420.,0.,dark[0,2]+273.15,
                LeafParameters(respiration25=1.))["respiration"])
            rd = -dark[0,3]/rd_factor
            quantum = 1.-dark[0,6]/dark[0,5]

            def parameters(x):
                return LeafParameters(vcmax25=x[0],jmax25=x[1],tpu25=x[2],
                    respiration25=rd,quantum_yield_psii=quantum)

            def predict(points,x):
                return farquhar_rates(points[:,0],.84*points[:,1],points[:,2]+273.15,
                    parameters(x))["net"]

            fit = least_squares(lambda x:predict(calibration,x)-calibration[:,3],
                [60.,100.,10.],bounds=([1.,1.,.1],[300.,600.,50.]),max_nfev=2000)
            if not fit.success:
                raise RuntimeError(f"Calibration failed for {name}: {fit.message}")
            entry = {"file":name,"calibration_n":len(calibration),"validation_n":len(validation),
                "parameters":asdict(parameters(fit.x)),
                "calibration_rmse":float(np.sqrt(np.mean(fit.fun**2))),
                "calibration_jacobian_rank":int(np.linalg.matrix_rank(fit.jac)),
                "identifiability_note":"A rank below 3 means some fitted capacities are not independently constrained",
                "status":"unmeasured"}
            if len(validation):
                prediction = predict(validation,fit.x)
                observed = validation[:,3]
                errors = prediction-observed
                near = abs(observed)<=1.
                rmse = float(np.sqrt(np.mean(errors**2)))
                nrmse = rmse/float(observed.mean())
                r = float(np.corrcoef(prediction,observed)[0,1])
                entry.update({"status":"pass" if nrmse<=.1 else "fail",
                    "mean_bias":float(errors.mean()),"relative_mean_bias":float(errors.mean()/observed.mean()),
                    "rmse":rmse,"normalized_rmse":nrmse,"r":r,
                    "near_zero_max_abs":float(np.max(abs(errors[near]))) if np.any(near) else None,
                    "pointwise_relative_failures":int(np.sum(abs(errors[~near])>.1*abs(observed[~near]))),
                    "near_zero_failures":int(np.sum(abs(errors[near])>.5)),
                    "light_rows":[{"ci_ppm":float(v[0]),"incident_par":float(v[1]),
                        "leaf_c":float(v[2]),"observed_net":float(v[3]),"predicted_net":float(a)}
                        for v,a in zip(validation,prediction)]})
            result["files"].append(entry)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=evaluate(args.archive)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps([{k:r.get(k) for k in ("file","status","relative_mean_bias","normalized_rmse","r")}
                      for r in result["files"]]))
