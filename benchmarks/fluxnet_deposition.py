"""Explicit constant or monthly deposition for station validation runners.

Monthly values are rates at calendar-month centers. Interpolate cyclically at
each forcing interval midpoint, with the actual Gregorian month lengths.
Nitrogen is an explicit constant rate; no trend or loss process is invented.
"""
import calendar
from datetime import datetime, timedelta
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=32)
def month_centers(year):
    lengths = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)])
    ends = lengths.cumsum(); centers = ends - lengths/2
    return np.r_[centers[-1]-ends[-1], centers, centers[0]+ends[-1]]


class DepositionSchedule:
    def __init__(self, config):
        keys = ('aerosol_deposition_kg_m2_s', 'aerosol_deposition_monthly_kg_m2_s')
        if sum(k in config for k in keys) != 1:
            raise ValueError('Specify exactly one constant or monthly aerosol deposition')
        self.monthly = keys[1] in config
        rates = np.array(config[keys[self.monthly]], dtype=float, copy=True)
        if (rates.shape != ((12, 5) if self.monthly else (5,))
                or not np.isfinite(rates).all() or np.any(rates < 0)):
            raise ValueError('Explicit finite nonnegative five-species aerosol deposition required')
        self.rates = np.concatenate((rates[-1:], rates, rates[:1])) if self.monthly else rates
        self.rates.setflags(write=False)
        self.nitrogen_rate = float(config['nitrogen_deposition_gn_m2_s'])
        if not np.isfinite(self.nitrogen_rate) or self.nitrogen_rate < 0:
            raise ValueError('Explicit finite nonnegative nitrogen deposition required')

    def aerosol(self, start, dt_seconds):
        if not self.monthly:
            return self.rates
        midpoint = start + timedelta(seconds=float(dt_seconds)/2)
        day = (midpoint-datetime(midpoint.year, 1, 1)).total_seconds()/86400
        times = month_centers(midpoint.year)
        return np.array([np.interp(day, times, self.rates[:, k]) for k in range(5)])
