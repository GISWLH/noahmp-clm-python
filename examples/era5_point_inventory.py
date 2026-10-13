"""Read-only bounded ERA5 forcing inventory; never substitute net radiation.

This reports one point and up to 30 daily/hourly records from each available
surface field. It does not claim a model run if downward SW/LW are missing.
Requires netCDF4 and numpy; the inventory records the actual units/aggregation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    from netCDF4 import Dataset, num2date
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="ERA5 surface directory")
    parser.add_argument("--month", default="200707")
    parser.add_argument("--latitude", type=float, default=44.5934)
    parser.add_argument("--longitude", type=float, default=123.5092)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = {
        "tair": "2m_temperature", "dewpoint": "2m_dewpoint_temperature",
        "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind",
        "ps": "surface_pressure", "swdown": "surface_solar_radiation_downwards",
        "lwdown": "surface_thermal_radiation_downwards", "precipitation": "total_precipitation"}
    candidate_extra = ["convective_rain_rate", "large_scale_rain_rate",
                       "surface_net_solar_radiation", "surface_net_thermal_radiation",
                       "mean_surface_net_short_wave_radiation_flux",
                       "mean_surface_net_long_wave_radiation_flux"]
    report = {"utc": datetime.now(timezone.utc).isoformat(), "root": str(args.root),
              "month": args.month, "requested_point": [args.latitude, args.longitude],
              "surface_directories": sorted(p.name for p in args.root.iterdir() if p.is_dir()),
              "required_fields": required, "available": {}, "missing": [],
              "model_run": False, "status": "not_tested",
              "mapping": {
                  "tair": "2m temperature, K, unchanged",
                  "qair": "Td_C=Td_K-273.15; e=611.2*exp(17.67*Td_C/(Td_C+243.5)); q=0.622*e/(ps-0.378*e)",
                  "wind": "sqrt(u10**2+v10**2), m/s; no tower-height adjustment",
                  "ps": "surface pressure, Pa; sea-level pressure is not a substitute",
                  "swdown_lwdown": "downward radiation only; accumulated J/m2 requires the documented accumulation interval before conversion to W/m2; net radiation cannot replace it",
                  "precipitation": "total precipitation in m divided by its accumulation interval, or complete rain+snow rates with verified units; rain-only cannot represent snow",
                  "time": "use actual timestamps and aggregation metadata; daily means cannot recover an hourly diurnal cycle"}}
    for field, directory in required.items():
        if not (args.root / directory / ("ERA5_" + args.month + ".nc")).exists():
            report["missing"].append(field)
    for directory in list(required.values()) + candidate_extra:
        path = args.root / directory / ("ERA5_" + args.month + ".nc")
        if not path.exists():
            continue
        with Dataset(path) as dataset:
            lat, lon = dataset.variables["latitude"], dataset.variables["longitude"]
            ilat = int(np.argmin(abs(lat[:] - args.latitude)))
            ilon = int(np.argmin(abs(lon[:] - (args.longitude % 360))))
            clock = dataset.variables["valid_time"]
            dates = num2date(clock[:], clock.units, calendar=getattr(clock, "calendar", "standard"))
            variables = {}
            for name, value in dataset.variables.items():
                if not {"valid_time", "latitude", "longitude"}.issubset(value.dimensions):
                    continue
                selectors = tuple(slice(0, 30) if dim == "valid_time" else ilat if dim == "latitude"
                                  else ilon if dim == "longitude" else 0 for dim in value.dimensions)
                series = np.asarray(np.ma.filled(value[selectors], np.nan), dtype=float)
                variables[name] = {"units": getattr(value, "units", None),
                                   "long_name": getattr(value, "long_name", None),
                                   "standard_name": getattr(value, "standard_name", None),
                                   "step_type": getattr(value, "GRIB_stepType", None),
                                   "values": [float(v) if np.isfinite(v) else None for v in series],
                                   "finite": bool(np.isfinite(series).all())}
            report["available"][directory] = {
                "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
                "gridpoint": [float(lat[ilat]), float(lon[ilon])],
                "time_count": len(dates), "selected_time_count": min(30, len(dates)),
                "times": [str(t) for t in dates[:30]],
                "history": getattr(dataset, "history", None),
                "time_shift": getattr(clock, "time_shift", None), "variables": variables}
    report["reason"] = ("Required forcing fields missing: " + ", ".join(report["missing"]) +
                        "; available net radiation and daily aggregation are retained as metadata, not silently substituted")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "model_run": False,
                      "available": len(report["available"]), "missing": report["missing"]}))


if __name__ == "__main__":
    main()
