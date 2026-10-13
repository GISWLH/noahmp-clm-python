"""Verify all four continuous P1 runs and their 1460 saved daily artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    result = {"backends": {}, "verified_daily_dumps": 0, "all_complete": True}
    physics_hashes = None
    for name in ("numpy365", "jax32-365", "jax64-365", "cuda-365"):
        path = root / "outputs" / (name + ".jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        begin, final = rows[0], rows[-1]
        days = [r for r in rows if r["event"] == "day"]
        assert final["event"] == "summary" and final["status"] == "passed"
        assert final["continuous_steps_completed"] == 17520 and len(days) == 365
        assert [r["day_index_zero_based"] for r in days] == list(range(365))
        hashes = {k: v for k,v in begin["source_hashes"].items() if k.startswith("src/")}
        if physics_hashes is None:
            physics_hashes = hashes
        assert hashes == physics_hashes
        for file, expected in hashes.items():
            assert hashlib.sha256((root / file).read_bytes()).hexdigest() == expected
        previous = None
        for row in days:
            dump = root / row["state_dump"]
            assert hashlib.sha256(dump.read_bytes()).hexdigest() == row["dump_sha256"]
            with np.load(dump) as data:
                if name == "numpy365":
                    first, last = data["f"][:,0], data["f"][:,-1]
                else:
                    first, last = data["initial_f"], data["f"]
                if previous is not None:
                    np.testing.assert_array_equal(first, previous)
                previous = last.copy()
            result["verified_daily_dumps"] += 1
        result["backends"][name] = {**final,
            "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result["source_hashes"] = physics_hashes
    result["scope"] = "continuous synthetic reliability and artifact integrity; not full-column enthalpy or observational validation"
    (root / "outputs/p1-annual-verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
