"""CUDA Richards water/heat adapter; all physical arrays stay on device."""
from ..richards_enthalpy import _run


def run_richards_enthalpy_cuda(c,p,options,dt,**controls):
    import cupy as cp
    from .enthalpy import run_soil_thermodynamics_cuda
    from .richards import run_richards_cuda
    c,p,dt=[cp.asarray(v,cp.float32) for v in (c,p,dt)]
    return _run(c,p,cp.asarray(options),dt,cp,run_soil_thermodynamics_cuda,run_richards_cuda,**controls)
