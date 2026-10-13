"""Read actual NetCDF headers in every surface directory, without field reads.

By default inspect first/last and 200707 files in each directory. --all inspects
every file. Report the scope explicitly; never infer downward from net radiation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    from netCDF4 import Dataset, num2date
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    report = {"utc": datetime.now(timezone.utc).isoformat(), "root": str(args.root),
              "scope": "all_headers" if args.all else "first_last_200707_per_directory",
              "directories": {}, "variables": [], "errors": [], "model_run": False}
    variables = set()
    for directory in sorted(p for p in args.root.iterdir() if p.is_dir()):
        paths = sorted(directory.rglob("*.nc"))
        chosen = paths if args.all else sorted(set(paths[:1] + paths[-1:] +
                    [p for p in paths if p.name == "ERA5_200707.nc"]))
        records = []
        for path in chosen:
            try:
                with Dataset(path) as data:
                    fields = {name: {"units": getattr(v, "units", None),
                        "long_name": getattr(v, "long_name", None),
                        "standard_name": getattr(v, "standard_name", None),
                        "dimensions": list(v.dimensions), "shape": list(v.shape)}
                        for name, v in data.variables.items() if len(v.dimensions) >= 2}
                    variables.update(fields)
                    clock = data.variables.get("valid_time", data.variables.get("time"))
                    times = [] if clock is None else [str(num2date(clock[k], clock.units,
                        calendar=getattr(clock, "calendar", "standard")))
                        for k in sorted(set([0, min(1, len(clock)-1), len(clock)-1]))]
                    records.append({"path": str(path), "bytes": path.stat().st_size,
                        "variables": fields, "times": times,
                        "time_shift": getattr(clock, "time_shift", None),
                        "history": getattr(data, "history", None)})
            except Exception as exc:
                report["errors"].append({"path": str(path), "error": str(exc)})
        report["directories"][directory.name] = {"file_count": len(paths), "headers": records}
        print(json.dumps({"directory": directory.name, "files": len(paths), "headers": len(records)}), flush=True)
    report["variables"] = sorted(variables)
    report["required_missing_in_inspected_files"] = sorted({"ssrd", "strd", "tp"} - variables)
    report["status"] = "not_tested" if report["required_missing_in_inspected_files"] or report["errors"] else "forcing_headers_available"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
