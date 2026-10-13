"""Predeclared v1 scientific bias metrics; independent of backend roundoff gates.

Rates must be in the units below. Supply interval lengths to obtain weighted
daily/monthly means (equivalent to comparing integrated amounts). Missing
samples require an explicit mask; no gaps or nonfinite values are hidden.
"""
from __future__ import annotations

import numpy as np

ABSOLUTE_TOLERANCES = {
    "et_mm_day": 0.05, "runoff_mm_day": 0.05,
    "h_wm2": 1.0, "le_wm2": 1.0, "g_wm2": 1.0,
    "gpp_gc_day": 0.1, "nee_gc_day": 0.1,
    "tsoil_c": 0.5, "smc": 0.002, "swe_m": 1e-4, "canw_m": 1e-7,
}


def compare_series(model, reference, variable: str, *, weights=None, mask=None) -> dict:
    """Return signed bias and instantaneous correlation with explicit validity.

    Near-zero comparisons use an absolute tolerance exclusively. ``bias_pass``
    and ``correlation_pass`` are separate because observational flux validation
    needs both, whereas backend numerical and BGC criteria differ.
    """
    atol = ABSOLUTE_TOLERANCES[variable]
    a, b = np.asarray(model, float), np.asarray(reference, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("model/reference must be matching one-dimensional series")
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    keep = np.ones(a.shape, bool) if mask is None else np.asarray(mask, bool)
    if w.shape != a.shape or keep.shape != a.shape:
        raise ValueError("weights/mask must match the series")
    a, b, w = a[keep], b[keep], w[keep]
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))
            and np.all(np.isfinite(w)) and np.all(w > 0)):
        raise ValueError("selected observations and weights must be finite; weights >0")
    if not a.size:
        return {"n": 0, "status": "unmeasured", "bias_pass": None, "correlation_pass": None}
    ma, mb = float(np.average(a, weights=w)), float(np.average(b, weights=w))
    bias = ma - mb
    near_zero = abs(mb) <= atol / 0.1
    r = None
    if a.size >= 3 and np.ptp(a) > 0 and np.ptp(b) > 0:
        r = float(np.corrcoef(a, b)[0, 1])
    return {
        "n": int(a.size), "status": "measured", "model_mean": ma, "reference_mean": mb,
        "bias": bias, "absolute_bias": abs(bias), "near_zero": near_zero,
        "relative_bias": None if near_zero else bias / abs(mb),
        "absolute_tolerance": atol, "relative_tolerance": 0.1,
        "bias_pass": abs(bias) <= (atol if near_zero else 0.1 * abs(mb)),
        "r": r, "correlation_pass": None if r is None else r > 0.85,
        "rmse": float(np.sqrt(np.average((a - b)**2, weights=w))),
        "max_absolute_difference": float(np.max(np.abs(a - b))),
    }
