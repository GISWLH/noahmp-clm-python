"""Physical consistency of the last MOST trial and its feedback target.

This check cannot replace either inner solve's conservation requirements.
It only judges whether an unresolved auxiliary transfer coefficient affects
the physical solution. All arrays have a leading column dimension.
"""


def surface_consistent(reference,candidate,time_step,xp):
    """Per-column fixed physical bounds, evaluated on the chosen backend.

    Fluxes use the predeclared v1 near-zero/10% bounds in daily-equivalent
    units. Temperatures use a stricter .05 K absolute bound, soil water
    .002 m3/m3, SWE .1 mm, and canopy humidity .05 g/kg. No rounding.
    """
    def within(key,absolute,*,scale=1.,near=None):
        a=xp.asarray(reference[key])*scale;b=xp.asarray(candidate[key])*scale
        bound=absolute if near is None else xp.where(xp.abs(a)<=near,absolute,.1*xp.abs(a))
        good=xp.isfinite(a)&xp.isfinite(b)&(xp.abs(a-b)<=bound)
        return xp.all(good,axis=tuple(range(1,good.ndim)))
    checks=[within(k,1.,near=10.) for k in ('h','le')]
    checks.append(within('gpp',.1,scale=12.0107e-6*86400.,near=1.))
    checks.extend(within(k,.05,scale=86400./time_step,near=.5) for k in ('et_kgm2','runoff_kgm2'))
    checks.extend(within(k,.05) for k in ('leaf_temperature','ground_temperature','canopy_temperature','soil_temperature'))
    checks.extend((within('canopy_humidity',5e-5),within('soil_water_content',.002),within('swe',.0001)))
    return xp.all(xp.stack(checks,axis=1),axis=1)
