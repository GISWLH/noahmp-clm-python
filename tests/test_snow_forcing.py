import numpy as np
import pytest
from noahmp_clm.snow_forcing import SnowDepositionClimatology


def table():
    lat=np.array([-60.,0.,60.]);lon=np.array([10.,100.,190.,280.])
    rates=np.zeros((12,3,4,5))
    rates[:]=np.arange(1,6)*1e-12
    rates*=np.arange(1,13)[:,None,None,None]
    rates*=np.array([1.,2.,3.])[None,:,None,None]
    rates*=np.array([1.,2.,3.,4.])[None,None,:,None]
    return SnowDepositionClimatology(lat,lon,rates)


def test_exact_month_center_spatial_average_and_longitude_wrap():
    data=table()
    np.testing.assert_allclose(data.at_site(0.,100.,15.5,2026),np.arange(1,6)*4e-12,rtol=1e-15)
    np.testing.assert_allclose(data.at_site(30.,145.,15.5,2026),np.arange(1,6)*6.25e-12,rtol=1e-15)
    np.testing.assert_allclose(data.at_site(0.,325.,15.5,2026),np.arange(1,6)*5e-12,rtol=1e-15)
    np.testing.assert_array_equal(data.at_site(0.,-35.,15.5,2026),data.at_site(0.,325.,15.5,2026))


def test_year_wrap_and_leap_calendar():
    data=table()
    np.testing.assert_array_equal(data.at_site(0.,100.,0.,2026),data.at_site(0.,100.,365.,2026))
    # February center is day 45 in nonleap and 45.5 in leap years.
    np.testing.assert_allclose(data.at_site(0.,100.,45.,2026),np.arange(1,6)*8e-12,rtol=1e-15)
    np.testing.assert_allclose(data.at_site(0.,100.,45.5,2028),np.arange(1,6)*8e-12,rtol=1e-15)
    assert np.all(data.at_site(90.,0.,1.,2026)>=0.)


def test_invalid_deposition_table_and_site():
    data=table()
    with pytest.raises(ValueError): data.at_site(91.,0.,1.,2026)
    with pytest.raises(ValueError): SnowDepositionClimatology([0.,-1.],[0.,180.],np.zeros((12,2,2,5)))
