"""Independent reference at a represented snow phase-event barrier."""
from pathlib import Path
import hashlib,json
import numpy as np
import pytest
from test_gpu_column_heat import numpy_reference
from clm_soil_oracle import fixture_suction

def saved(dtype=np.float32):
    path=Path(__file__).with_name('fixtures')/'qia_heat_phase_event.npz'
    metadata=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata['fixture_sha256']
    with np.load(path,allow_pickle=False) as z:x=dict(z)
    assert not np.any(x['original'][:,75])
    assert np.all(np.max(abs(x['original'][:,45:54])/x['original'][:,62:71],axis=1)>1.)
    inputs=tuple(x[k].astype(np.int32 if k=='options' else dtype) for k in ['cells','props','options','boundary'])
    inputs=(inputs[0],fixture_suction(inputs[1],.355),*inputs[2:])
    return inputs,x

@pytest.mark.parametrize('case',range(3))
def test_captured_phase_event_numpy_reference_and_independent_energy(case):
    inputs,x=saved();c,p,o,b=(a[case] for a in inputs)
    h,t,d=numpy_reference(c,p,o,b)
    np.testing.assert_allclose(h,x['reference_enthalpy'][case],rtol=.1,atol=.1)
    assert np.max(abs(t-x['reference_temperature'][case]))<=.5
    n=int(o[0])+4
    incoming=float(b[0])*(d['top_flux_wm2']+float(c[:n,4].astype(float).sum()))
    residual=float((h-c[:n,1].astype(float)).sum()-incoming)
    bound=n*d['convergence_tolerance_jm2']+16*np.finfo(float).eps*max(abs(h).sum()+abs(c[:n,1]).sum()+abs(incoming),1.)
    assert abs(residual)<=bound
    print('Qia phase event NumPy',case,residual,bound,flush=True)
