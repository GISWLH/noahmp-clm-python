"""Capture failed broad-audit input and compare its independent CPU solution."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
import jax.numpy as j
from noahmp_clm.gpu.column_heat import _compiled,unpack_joint_heat,run_joint_heat_jax

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
audit=load('audit_module','benchmarks/check_enthalpy_gpu.py');reference=load('reference_module','tests/test_gpu_column_heat.py')
def probe(*inputs):
    raw=unpack_joint_heat(_compiled()(*(j.asarray(v) for v in inputs)))
    result={k:np.asarray(v) for k,v in raw.items()};failed=np.flatnonzero(~result['converged'])
    print('dtype',inputs[0].dtype,'failed',failed.tolist(),flush=True)
    if len(failed):
        np.savez('failed-heat-inputs.npz',cells=inputs[0],props=inputs[1],options=inputs[2],boundary=inputs[3])
        for row in failed:
            detail=dict(row=int(row),cells=inputs[0][row].tolist(),props=inputs[1][row].tolist(),options=inputs[2][row].tolist(),boundary=inputs[3][row].tolist(),
                result={k:v[row].tolist() for k,v in result.items()})
            try:
                h,t,d=reference.numpy_reference(*(v[row] for v in inputs));detail['numpy']=dict(h=h.tolist(),t=t.tolist(),iterations=d['iterations'],residual=d['residual_jm2'])
            except Exception as exc: detail['numpy_error']=str(exc)
            print(json.dumps(detail),flush=True)
        raise RuntimeError('Probe preserved failed heat inputs')
    return raw
audit.run_joint_heat_jax=probe
audit.main()
