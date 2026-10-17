"""Independent pressure-head form for phase/energy reference solves.

No production inverse, analytic derivative or potential helper is called.
The existing compact cap is specified piecewise; option2 is the retained
simplified controller, not a claim of an independent full Koren reference.
"""
import numpy as np

def fixture_suction(parameters,saturated_suction_m):
    """Explicitly derive new-law challenge inputs from an immutable old fixture.

    Extensive H is held fixed; decoded temperature therefore changes. These
    inputs are not an exact replay of the old constitutive model. The caller
    supplies the suction assumption, rather than inferring it from old scores.
    """
    p=np.asarray(parameters)
    return np.concatenate((p,np.full((*p.shape[:-1],1),saturated_suction_m,dtype=p.dtype)),axis=-1)

def reference_liquid(theta,temperature,params,option=1):
    theta,t=np.broadcast_arrays(np.asarray(theta,float),np.asarray(temperature,float))
    pressure_head=334000.*np.maximum(273.15-t,1e-12)/(9.80616*np.minimum(t,273.15))
    upper=params.smcmax*np.power(params.saturated_suction_m/pressure_head,1./params.bexp)
    def cap(water):
        width=np.maximum(.04*water,1e-12);gap=np.abs(water-upper)
        correction=np.where(gap<width,(width-gap)**2/(4*width),0.)
        return np.maximum(np.minimum(water,upper)-correction,0.)
    value=cap(theta)
    if option==2:
        for _ in range(3):value=.5*(value+np.minimum(theta,cap(value)))
    return np.where(t>=273.15,theta,value)
