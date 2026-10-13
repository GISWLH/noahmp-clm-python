"""The benchmark must reject skipped, corrupted and cancellation-hidden output."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


_path = Path(__file__).resolve().parents[1] / "benchmarks" / "gpu_checksum.py"
_spec = importlib.util.spec_from_file_location("gpu_checksum", _path)
checksum = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checksum)


def _result():
    floats = np.ones((3, 70), dtype=np.float32)
    integers = np.ones((3, 33), dtype=np.int32)
    diagnostics = np.arange(3 * 4 * 12, dtype=np.float32).reshape(3, 4, 12) / 100.0
    diagnostics[:, :, 0] = 285.0
    return floats, integers, diagnostics


def test_full_output_passes_and_counts_every_column_step():
    digest = checksum.numpy_digest_for_test(_result())
    assert digest["counts"] == {"finite_column_steps": 12, "effective_column_steps": 12, "valid_columns": 3}
    assert checksum.validate_digest(digest, digest, 3, 4)["passed"]


def test_finite_zeros_are_not_valid_model_execution():
    reference = checksum.numpy_digest_for_test(_result())
    zero = tuple(np.zeros_like(array) for array in _result())
    digest = checksum.numpy_digest_for_test(zero)
    gate = checksum.validate_digest(digest, reference, 3, 4)
    assert not gate["passed"]
    assert digest["counts"]["effective_column_steps"] == 0


def test_equal_opposite_errors_cannot_cancel_in_signed_sum():
    arrays = _result()
    reference = checksum.numpy_digest_for_test(arrays)
    arrays[2][0, 0, 3] += 10
    arrays[2][0, 1, 3] -= 10
    digest = checksum.numpy_digest_for_test(arrays)
    # A single sum would miss this large error to float32 rounding accuracy.
    np.testing.assert_allclose(digest["groups"]["diagnostics"]["signed_sum"][3],
                               reference["groups"]["diagnostics"]["signed_sum"][3], atol=1e-6)
    assert not checksum.validate_digest(digest, reference, 3, 4)["passed"]


def test_reordering_is_visible_even_when_signed_absolute_and_range_agree():
    arrays = _result()
    reference = checksum.numpy_digest_for_test(arrays)
    arrays[2][:, :, 3] = arrays[2][::-1, ::-1, 3].copy()
    digest = checksum.numpy_digest_for_test(arrays)
    for metric in ("signed_sum", "absolute_sum", "minimum", "maximum"):
        assert digest["groups"]["diagnostics"][metric][3] == reference["groups"]["diagnostics"][metric][3]
    assert not checksum.validate_digest(digest, reference, 3, 4)["passed"]


def test_nan_in_any_state_slot_is_rejected():
    arrays = _result()
    reference = checksum.numpy_digest_for_test(arrays)
    arrays[0][-1, -1] = np.nan
    digest = checksum.numpy_digest_for_test(arrays)
    gate = checksum.validate_digest(digest, reference, 3, 4)
    assert not gate["passed"]
    assert any("nonfinite" in item for item in gate["failures"])


def test_partial_output_shape_is_rejected():
    arrays = _result()
    with pytest.raises(ValueError, match="output shapes"):
        checksum._reduce_arrays((arrays[0][:2], arrays[1], arrays[2]), 3, 4, np)


def test_unchanged_output_fails_forcing_sensitivity():
    arrays = _result()
    original = checksum.numpy_digest_for_test(arrays)
    assert not checksum.nondegenerate_change(original, original)["passed"]
    arrays[2][:, :, 2] += 1.0
    changed = checksum.numpy_digest_for_test(arrays)
    assert checksum.nondegenerate_change(original, changed)["passed"]


@pytest.mark.parametrize("mode", ("float64_in_float32_run", "mixed_float_dtypes", "floating_integer_state"))
def test_dtype_contract_cannot_pass_by_matching_numerical_values(mode):
    f, i, d = _result()
    if mode == "float64_in_float32_run":
        f, d = f.astype(np.float64), d.astype(np.float64)
    elif mode == "mixed_float_dtypes":
        d = d.astype(np.float64)
    else:
        i = i.astype(np.float32)
    with pytest.raises(ValueError):
        checksum._validate_output_dtypes((f, i, d), np.float32)


@pytest.mark.skipif(os.environ.get("NOAHMP_REQUIRE_GPU") != "1", reason="Explicit CUDA environment required")
def test_gpu_gate_checks_every_item_and_exact_integer_state():
    # An explicitly requested GPU gate must fail if CUDA/CuPy is unavailable.
    import cupy as cp
    assert not cp.cuda.runtime.is_hip and cp.cuda.runtime.getDeviceCount() > 0
    reference = tuple(cp.asarray(array) for array in _result())
    digest, gate = checksum.consume_device_result(reference, 3, 4)
    assert gate["passed"]
    changed = tuple(array.copy() for array in reference)
    changed[0][-1, -1] += cp.float32(1e-3)
    changed[1][1, 2] += 1
    _, gate = checksum.consume_device_result(changed, 3, 4, reference, digest)
    assert not gate["passed"]
    assert gate["pointwise_tolerance_ratios"]["integer_state"] == 1
    assert gate["pointwise_tolerance_ratios"]["float_state"][-1] > 1
