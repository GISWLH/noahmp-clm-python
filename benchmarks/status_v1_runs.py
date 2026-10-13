"""Compact, read-only status for P1 yearly logs (no repeated state dumps)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for name in ("numpy365", "jax32-365", "jax64-365", "cuda-365"):
        path = args.directory / "outputs" / (name + ".jsonl")
        lines = path.read_text().splitlines() if path.exists() else []
        try:
            row = json.loads(lines[-1]) if lines else {}
        except json.JSONDecodeError:
            row = json.loads(lines[-2])
        print(json.dumps({"run": name, **{key: row[key] for key in
            ("event", "status", "completed_days", "day_index_zero_based", "continuous_steps_completed",
             "energy_abs_residual_max_W_m2", "water_cumulative_abs_max_m", "wall_seconds") if key in row}}))


if __name__ == "__main__":
    main()
