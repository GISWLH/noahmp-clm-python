"""Exact Qia cold-cycle cells where the coupled float32 direction stalled."""
from pathlib import Path
import hashlib,json
import numpy as np
from test_gpu_column_heat import numpy_reference
from clm_soil_oracle import fixture_suction

def saved(dtype=np.float32):
    path=Path(__file__).with_name('fixtures')/'qia_heat_coordinate_inputs.npz'
    reference=json.loads(path.with_name('qia_heat_coordinate_reference.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['input_sha256']
    with np.load(path,allow_pickle=False) as z:
        a=tuple(z[k].astype(np.int32 if k=='options' else dtype) for k in ['cells','props','options','boundary'])
        original=z['original']
    a=(a[0],fixture_suction(a[1],.355),*a[2:])
    assert not original[0,75] and np.max(abs(original[0,45:54])/original[0,62:71])>1.
    return a,reference

def test_saved_qia_heat_numpy_reference_and_conservation():
    (c,p,o,b),expected=saved();h,t,d=numpy_reference(c[0],p[0],o[0],b[0])
    np.testing.assert_allclose(h,expected['enthalpy_jm2'],rtol=.1,atol=.1)
    assert np.max(abs(t-expected['temperature_k']))<=.5
    n=int(o[0,0])+4
    incoming=float(b[0,0])*(d['top_flux_wm2']+float(c[0,:n,4].astype(float).sum()))
    residual=float((h-c[0,:n,1].astype(float)).sum()-incoming)
    bound=n*d['convergence_tolerance_jm2']+16*np.finfo(float).eps*max(abs(h).sum()+abs(c[0,:n,1]).sum()+abs(incoming),1.)
    assert abs(residual)<=bound
    print('Qia heat independent NumPy',residual,bound,t.tolist(),flush=True)
