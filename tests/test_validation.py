import pytest
from noahmp_clm.validation import compare_series


def test_near_zero_signed_flux_uses_absolute_units():
    result = compare_series([-0.4, 0.6], [-0.5, 0.5], "h_wm2")
    assert result["near_zero"] and result["relative_bias"] is None
    assert result["bias_pass"] and result["correlation_pass"] is None


def test_integrated_bias_uses_elapsed_time_and_reports_correlation():
    result = compare_series([11., 33., 22.], [10., 30., 20.], "le_wm2", weights=[1., 2., 1.])
    assert result["reference_mean"] == 22.5
    assert result["relative_bias"] == pytest.approx(0.1)
    assert result["bias_pass"] and result["correlation_pass"]


def test_missing_is_not_a_pass_or_silent_nan_filter():
    assert compare_series([], [], "swe_m")["bias_pass"] is None
    with pytest.raises(ValueError, match="finite"):
        compare_series([1., float("nan")], [1., 2.], "smc")
    assert compare_series([1., float("nan")], [1., 2.], "smc", mask=[True, False])["n"] == 1
