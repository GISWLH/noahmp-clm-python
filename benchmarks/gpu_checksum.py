"""Full-output device reductions and the strict benchmark-consumption gate.

This guard proves that the timed result was consumed and agrees with the chosen
JAX reference. It is deliberately separate from, and cannot replace, H4 physics
parity. CuPy performs GPU reductions for both CuPy and JAX (DLPack) outputs.
"""
from __future__ import annotations

import numpy as np

RTOL = 1.0e-6
ATOL = 1.0e-6
GROUPS = ("float_state", "integer_state", "diagnostics")


def json_safe(value):
    """Preserve failure evidence in strict JSON (JSON has no NaN/Infinity)."""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def _validate_output_dtypes(result, expected_dtype=None):
    floating, integer, diagnostics = (np.dtype(array.dtype) for array in result)
    if floating not in (np.dtype("float32"), np.dtype("float64")) or diagnostics != floating:
        raise ValueError(f"Unexpected floating output dtypes: {floating}, {diagnostics}")
    if integer != np.dtype("int32"):
        raise ValueError(f"Integer ABI output must be int32, got {integer}")
    if expected_dtype is not None and floating != np.dtype(expected_dtype):
        raise ValueError(f"Requested {np.dtype(expected_dtype)}, got {floating}")


def _reduce_arrays(result, n, steps, xp):
    """Reduce every element; xp is explicitly NumPy in unit tests, CuPy in runs."""
    expected = ((n, 70), (n, 33), (n, steps, 12))
    if tuple(array.shape for array in result) != expected:
        raise ValueError(f"Unexpected output shapes: {[a.shape for a in result]}; expected {expected}")
    _validate_output_dtypes(result)
    reduced = {}
    for name, array in zip(GROUPS, result):
        flat = array.reshape((-1, array.shape[-1])).astype(xp.float64, copy=False)
        finite = xp.isfinite(flat)
        safe = xp.where(finite, flat, 0.0)
        weights = (xp.arange(flat.shape[0], dtype=xp.float64) + 1.0) / flat.shape[0]
        reduced[name] = {
            "rows": int(flat.shape[0]), "fields": int(flat.shape[1]),
            "finite_count": xp.sum(finite, axis=0, dtype=xp.int64),
            "signed_sum": xp.sum(safe, axis=0, dtype=xp.float64),
            "absolute_sum": xp.sum(xp.abs(safe), axis=0, dtype=xp.float64),
            "weighted_sum": xp.sum(safe * weights[:, None], axis=0, dtype=xp.float64),
            "minimum": xp.min(safe, axis=0), "maximum": xp.max(safe, axis=0),
        }
    f, i, d = result
    finite_steps = xp.all(xp.isfinite(d), axis=2)
    # Finite zeros or an uninitialized/empty result must not count as effective
    # column steps. The benchmark's default-column surface temperature is Kelvin.
    effective_steps = finite_steps & (d[:, :, 0] > 150.0) & (d[:, :, 0] < 400.0)
    valid_columns = xp.all(xp.isfinite(f), axis=1) & xp.all(xp.isfinite(i), axis=1)
    valid_columns &= xp.all(effective_steps, axis=1)
    counts = {
        "finite_column_steps": xp.sum(finite_steps, dtype=xp.int64),
        "effective_column_steps": xp.sum(effective_steps, dtype=xp.int64),
        "valid_columns": xp.sum(valid_columns, dtype=xp.int64),
    }
    return reduced, counts


def _host_reductions(reduced, counts, to_host):
    digest = {}
    for name, group in reduced.items():
        digest[name] = {
            key: value if isinstance(value, int) else np.asarray(to_host(value)).tolist()
            for key, value in group.items()
        }
    return {"groups": digest, "counts": {key: int(to_host(value)) for key, value in counts.items()}}


def validate_digest(digest, reference, n, steps):
    """Strict per-field reduction comparison, normalized to per-element units."""
    failures = []
    counts = digest["counts"]
    for name, target in (("finite_column_steps", n * steps), ("effective_column_steps", n * steps),
                         ("valid_columns", n)):
        if counts.get(name) != target:
            failures.append(f"{name}={counts.get(name)}, expected {target}")
    max_ratio = 0.0
    for name in GROUPS:
        group, expected = digest["groups"][name], reference["groups"][name]
        rows = n * steps if name == "diagnostics" else n
        fields = {"float_state": 70, "integer_state": 33, "diagnostics": 12}[name]
        if group["rows"] != rows or group["fields"] != fields:
            failures.append(f"{name}: wrong reduction dimensions")
            continue
        if np.asarray(group["finite_count"]).shape != (fields,) or any(value != rows for value in group["finite_count"]):
            failures.append(f"{name}: output contains nonfinite values")
        for metric in ("signed_sum", "absolute_sum", "weighted_sum", "minimum", "maximum"):
            actual = np.asarray(group[metric], dtype=np.float64)
            target = np.asarray(expected[metric], dtype=np.float64)
            if actual.shape != (fields,) or target.shape != (fields,) or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(target)):
                failures.append(f"{name}.{metric}: wrong shape/nonfinite checksum")
                continue
            # Normalize sums by the exact number of elements of each component.
            # The gate remains atol=rtol=1e-6 in original per-component units.
            normalizer = rows if metric.endswith("sum") else 1
            ratio = np.abs(actual / normalizer - target / normalizer) / (
                ATOL + RTOL * np.abs(target / normalizer)
            )
            maximum = float(np.max(ratio, initial=0.0))
            max_ratio = max(max_ratio, maximum)
            if maximum > 1.0:
                failures.append(f"{name}.{metric}: tolerance ratio {maximum:.9g} > 1")
    return {"passed": not failures, "rtol": RTOL, "atol": ATOL,
            "max_reduction_tolerance_ratio": max_ratio, "failures": failures}


def numpy_digest_for_test(result):
    """CPU-only reference implementation for corruption/regression unit tests."""
    n, steps = result[2].shape[:2]
    groups, counts = _reduce_arrays(result, n, steps, np)
    return _host_reductions(groups, counts, np.asarray)


def as_cupy_result(result):
    """No bulk host copy: borrow GPU arrays, using DLPack for JAX outputs."""
    import cupy as cp
    arrays = []
    for array in result:
        if isinstance(array, cp.ndarray):
            value = array
        else:
            devices = array.devices()
            if not devices or any(device.platform != "gpu" for device in devices):
                raise RuntimeError(f"Checksum requires device output, got {devices}")
            value = cp.from_dlpack(array)
        if value.device.id != cp.cuda.Device().id:
            raise RuntimeError("Checksum output is on a different CUDA device")
        arrays.append(value)
    return tuple(arrays)


def consume_device_result(result, n, steps, reference_arrays=None, reference_digest=None, expected_dtype=None):
    """Consume every state/diagnostic element on device; copy only reductions."""
    import cupy as cp
    arrays = as_cupy_result(result)
    requested = expected_dtype if expected_dtype is not None else (
        reference_arrays[0].dtype if reference_arrays is not None else None
    )
    _validate_output_dtypes(arrays, requested)
    groups, counts = _reduce_arrays(arrays, n, steps, cp)
    # Full pointwise comparisons make cancellation-resistant fingerprints an
    # additional signal, never a substitute for checking individual elements.
    itemwise = {}
    if reference_arrays is not None:
        for name, actual, expected in zip(GROUPS, arrays, reference_arrays):
            if actual.shape != expected.shape:
                raise ValueError(f"{name}: reference/output shapes differ")
            if name == "integer_state":
                itemwise[name] = cp.sum(actual != expected, dtype=cp.int64)
            else:
                a, b = actual.astype(cp.float64, copy=False), expected.astype(cp.float64, copy=False)
                ratio = cp.abs(a - b) / (ATOL + RTOL * cp.abs(b))
                ratio = cp.where(cp.isfinite(ratio), ratio, cp.inf)
                itemwise[name] = cp.max(ratio, axis=tuple(range(actual.ndim - 1)))
    digest = _host_reductions(groups, counts, cp.asnumpy)
    gate = validate_digest(digest, reference_digest or digest, n, steps)
    gate["actual_dtypes"] = [str(array.dtype) for array in arrays]
    itemwise_host = {name: cp.asnumpy(value).tolist() for name, value in itemwise.items()}
    gate["pointwise_tolerance_ratios"] = itemwise_host
    for name, value in itemwise_host.items():
        if name == "integer_state":
            if value != 0:
                gate["failures"].append(f"integer_state: {value} exact mismatches")
        elif not np.all(np.isfinite(value)) or np.max(value, initial=0.0) > 1.0:
            gate["failures"].append(f"{name}: pointwise tolerance gate failed")
    gate["passed"] = not gate["failures"]
    cp.cuda.get_current_stream().synchronize()
    return digest, gate


def assert_gate(gate):
    if not gate["passed"]:
        raise AssertionError("GPU benchmark checksum gate failed: " + "; ".join(gate["failures"]))


def nondegenerate_change(first_digest, changed_digest):
    """Forcing perturbation must change the calculated physics, not just metadata."""
    left = first_digest["groups"]["diagnostics"]
    right = changed_digest["groups"]["diagnostics"]
    a = np.asarray(left["signed_sum"], dtype=np.float64) / left["rows"]
    b = np.asarray(right["signed_sum"], dtype=np.float64) / right["rows"]
    ratios = np.abs(a - b) / (ATOL + RTOL * np.abs(a))
    # Surface temperature and turbulent/energy fluxes, excluding precipitation.
    sensitive = ratios[[0, 1, 2, 3, 4, 5]]
    return {"passed": bool(np.all(np.isfinite(sensitive)) and np.max(sensitive) > 1.0),
            "sensitive_component_tolerance_ratios": sensitive.tolist()}
