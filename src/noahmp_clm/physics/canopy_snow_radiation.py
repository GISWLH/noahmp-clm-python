"""Conservative two-band canopy / five-band SNICAR boundary coupling.

Snow has distinct direct and diffuse effective albedos. Upward snow flux
re-enters the canopy and its reflected diffuse flux returns to the snow;
the adding solver includes every reflection. NIR sub-band weights retain
the SNICAR climatological direct/diffuse assumptions after canopy scattering.
The pack covers both vegetated and bare fractions of this single column.
"""
import numpy as np
from .canopy_radiation import canopy_shortwave
from .snicar import snow_shortwave


def canopy_snow_shortwave(leaf_area,pack,swdown,*,vegetated_fraction=1.,
                          diffuse_fraction=.5,cosine_zenith=.5,ground_albedo=(.15,.3)):
    """Return canopy, snow and soil absorptions plus atmospheric reflection.

    Fluxes are W/m2 total ground area. ``canopy`` preserves leaf-area based
    PAR per vegetated leaf area for biochemical exchange. Snow/soil absorption
    belongs in the thermal cells, not a massless canopy-ground skin balance.
    """
    fv=float(vegetated_fraction)
    if not 0<=fv<=1: raise ValueError('Invalid vegetation fraction')
    options=dict(ground_albedo=ground_albedo,cosine_zenith=cosine_zenith)
    direct=snow_shortwave(pack,[1.,1.],[0.,0.],**options)
    diffuse=snow_shortwave(pack,[0.,0.],[1.,1.],**options)
    def two_band(values): return np.array([values[0],np.sum(values[1:])])
    ad=two_band(direct['spectral_reflected']);af=two_band(diffuse['spectral_reflected'])
    canopy=canopy_shortwave(leaf_area,swdown,diffuse_fraction=diffuse_fraction,
        cosine_zenith=cosine_zenith,ground_albedo=af,ground_albedo_direct=ad)
    top_direct=np.full(2,.5*swdown*(1.-diffuse_fraction))
    top_diffuse=np.full(2,.5*swdown*diffuse_fraction)
    bottom_direct=np.array([b['direct'][-1] for b in canopy['bands']])
    bottom_diffuse=np.array([b['diffuse_down'][-1] for b in canopy['bands']])
    snow=snow_shortwave(pack,fv*bottom_direct+(1.-fv)*top_direct,
        fv*bottom_diffuse+(1.-fv)*top_diffuse,**options)
    reflected=fv*canopy['reflected']+(1.-fv)*np.sum(ad*top_direct+af*top_diffuse)
    leaf=fv*canopy['absorbed']
    return dict(canopy=canopy,canopy_absorbed=leaf,snow_absorbed=snow['absorbed'],
        soil_absorbed=snow['ground_absorbed'],reflected=float(reflected),
        residual=float(swdown-reflected-leaf.sum()-snow['absorbed'].sum()-snow['ground_absorbed']),
        snow_direct_albedo=ad,snow_diffuse_albedo=af,snow=snow)
