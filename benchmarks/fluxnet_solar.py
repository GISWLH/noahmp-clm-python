"""Explicit solar inputs for station evaluation, independent of model backends.

NOAA's published harmonic solar equations supply geometric zenith, without
refraction. Erbs (1982) supplies an estimated diffuse fraction, never labelled
as a tower measurement. Inputs are local STANDARD times with a fixed UTC offset.
Sources and approximations are recorded in docs/FLUXNET_PLAN.md.
"""
from calendar import isleap
from datetime import datetime, timedelta
import math


def solar_geometry(local_time: datetime, latitude: float, longitude: float, utc_offset_hours: float):
    """Position at an explicit instant; caller chooses interval midpoint.

    Fractional year is evaluated in UTC so equivalent local-time representations
    give the same position. Daylength uses the geometric horizon, not refracted
    sunrise/sunset. No implicit UTC+8 or daylight-saving correction is applied.
    """
    if local_time.tzinfo is not None:
        raise ValueError('Use a naive local-standard timestamp and explicit fixed UTC offset')
    if (not all(math.isfinite(x) for x in (latitude, longitude, utc_offset_hours))
            or not -90 <= latitude <= 90 or not -180 <= longitude <= 180
            or not -12 <= utc_offset_hours <= 14):
        raise ValueError('Invalid station coordinates or fixed UTC offset')
    utc = local_time - timedelta(hours=utc_offset_hours)
    hour = utc.hour + utc.minute / 60 + (utc.second + utc.microsecond / 1e6) / 3600
    day = utc.timetuple().tm_yday
    gamma = 2 * math.pi / (366 if isleap(utc.year) else 365) * (day - 1 + (hour - 12) / 24)
    equation = 229.18 * (.000075 + .001868 * math.cos(gamma) - .032077 * math.sin(gamma)
                         - .014615 * math.cos(2 * gamma) - .040849 * math.sin(2 * gamma))
    declination = (.006918 - .399912 * math.cos(gamma) + .070257 * math.sin(gamma)
                   - .006758 * math.cos(2 * gamma) + .000907 * math.sin(2 * gamma)
                   - .002697 * math.cos(3 * gamma) + .001480 * math.sin(3 * gamma))
    local_minutes = local_time.hour * 60 + local_time.minute + (local_time.second + local_time.microsecond / 1e6) / 60
    angle = math.radians((local_minutes + equation + 4 * longitude - 60 * utc_offset_hours) / 4 - 180)
    lat = math.radians(latitude)
    centre = math.sin(lat) * math.sin(declination)
    amplitude = math.cos(lat) * math.cos(declination)
    mu = max(-1., min(1., centre + amplitude * math.cos(angle)))
    if amplitude < 1e-14:
        daylength = 86400. if centre > 0 else 0.
    else:
        sunset_cos = -centre / amplitude
        daylength = 86400. * math.acos(max(-1., min(1., sunset_cos))) / math.pi
    # Spencer orbital-distance factor, 365-day Fourier convention, at UTC day.
    orbit = 2 * math.pi * (day - 1) / 365.
    dni_extra = 1366.1 * (1.00011 + .034221 * math.cos(orbit) + .00128 * math.sin(orbit)
                         + .000719 * math.cos(2 * orbit) + .000077 * math.sin(2 * orbit))
    return {'cosine_zenith': mu, 'zenith_degrees': math.degrees(math.acos(mu)),
            'daylength_seconds': daylength, 'declination_radians': declination,
            'equation_of_time_minutes': equation, 'extraterrestrial_normal_wm2': dni_extra}


def diffuse_fraction_erbs(shortwave_wm2: float, cosine_zenith: float, extraterrestrial_normal_wm2: float):
    """Erbs hourly relation used for each station interval (an approximation).

    The low-sun guard follows the published pvlib default (87 degrees). Retain
    all observed shortwave energy as diffuse there, including positive nighttime
    sensor offsets; report the condition instead of changing total radiation.
    """
    if (not all(math.isfinite(v) for v in (shortwave_wm2, cosine_zenith, extraterrestrial_normal_wm2))
            or shortwave_wm2 < 0 or not -1 <= cosine_zenith <= 1 or extraterrestrial_normal_wm2 <= 0):
        raise ValueError('Invalid shortwave geometry/input')
    clearness_raw = shortwave_wm2 / (extraterrestrial_normal_wm2 * max(.065, cosine_zenith))
    clearness = min(1., clearness_raw)
    if shortwave_wm2 == 0 or cosine_zenith < math.cos(math.radians(87)):
        fraction = 1.
    elif clearness <= .22:
        fraction = 1 - .09 * clearness
    elif clearness <= .8:
        k = clearness
        fraction = .9511 - .1604 * k + 4.388 * k**2 - 16.638 * k**3 + 12.336 * k**4
    else:
        fraction = .165
    return {'diffuse_fraction': fraction, 'clearness_index_raw': clearness_raw,
            'clearness_index_for_erbs': clearness,
            'low_sun_all_diffuse': cosine_zenith < math.cos(math.radians(87)),
            'diffuse_horizontal_wm2': shortwave_wm2 * fraction,
            'direct_horizontal_wm2': shortwave_wm2 * (1 - fraction)}
