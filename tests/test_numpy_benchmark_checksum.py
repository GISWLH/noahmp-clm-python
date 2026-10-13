"""A failed ordinary-model checksum cannot become accepted benchmark timing."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_gpu.py"
_spec = importlib.util.spec_from_file_location("bench_gpu_checksum_test", _path)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_every_numpy_repeat_checksum_is_consumed_and_recorded():
    progress = []
    _, samples = bench._timed(lambda: 285.125, 3,
                              lambda *row: progress.append(row))
    assert len(samples) == 3
    assert [(row[0], row[1]) for row in progress] == [
        (phase, index) for phase in ("warmup", "measured_run") for index in range(3)
    ]
    assert [row[3] for row in progress] == [285.125] * 6


@pytest.mark.parametrize("bad_index", range(6))
@pytest.mark.parametrize("bad_value", (np.nan, np.inf, 286.0))
def test_bad_checksum_in_any_warmup_or_measurement_rejects_run(bad_index, bad_value):
    values = [285.0] * 6
    values[bad_index] = bad_value
    iterator = iter(values)
    with pytest.raises(ValueError, match="checksum"):
        bench._timed(lambda: next(iterator), 3, lambda *_: None)


@pytest.mark.parametrize("fault", ("surface_temperature", "evapotranspiration"))
def test_cli_rejects_nan_physics_without_measured_result(fault, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(_path.parent))
    monkeypatch.setattr("sys.argv", ["bench_gpu.py", "--sizes", "2", "--steps", "1",
                                    "--backends", "numpy"])
    monkeypatch.setattr(bench, "run_column", lambda *_: (
        [{"et_mm": np.nan if fault == "evapotranspiration" else 0.0}],
        SimpleNamespace(ts=np.nan if fault == "surface_temperature" else 285.0),
    ))
    assert bench.main() == 1
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[-1]["event"] == "result"
    assert rows[-1]["status"] == "rejected"
    assert "nonfinite checksum" in rows[-1]["error"]
    assert "seconds" not in rows[-1]
    assert not any(row.get("status") == "measured" for row in rows)
