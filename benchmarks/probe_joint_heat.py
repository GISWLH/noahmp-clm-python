"""Inspect raw nonconverged device columns without suppressing public errors."""
import importlib.util,json
from pathlib import Path
import numpy as np
import jax.numpy as j
import cupy as cp
from noahmp_clm.gpu.column_heat import _compiled,unpack_joint_heat
from noahmp_clm.gpu.cuda.column_heat import _kernel

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('heat_cases',root/'tests/test_gpu_column_heat.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
inputs=module.cases(np.float32);n=len(inputs[0])
jax=unpack_joint_heat(_compiled()(*(j.asarray(v) for v in inputs)))
c,p,o,b=[cp.ascontiguousarray(cp.asarray(v)) for v in inputs];raw=cp.empty((n,77),cp.float32)
_kernel()(((n+63)//64,),(64,),(c,p,o,b,raw,np.int32(n)))
cuda=unpack_joint_heat(tuple(raw[:,i:i+9] for i in range(0,54,9))+(raw[:,54:62],raw[:,62:71],raw[:,71:]))
for backend,result in [('jax',jax),('cuda',cuda)]:
    r={k:np.asarray(v) if backend=='jax' else cp.asnumpy(v) for k,v in result.items()}
    failed=np.flatnonzero(~r['converged'])
    print(backend,'failed',failed.tolist(),flush=True)
    for row in failed:
        h,t,d=module.numpy_reference(*(v[row] for v in inputs))
        print(json.dumps(dict(row=int(row),ns=int(inputs[2][row,0]),iterations=float(r['iterations'][row]),
            residual=r['layer_residual_jm2'][row].tolist(),tolerance=r['layer_tolerance_jm2'][row].tolist(),
            enthalpy=r['enthalpy_jm2'][row].tolist(),reference_h=h.tolist(),
            temperature=r['temperature_k'][row].tolist(),reference_t=t.tolist())),flush=True)
for k in jax:
    a,b=np.asarray(jax[k]),cp.asnumpy(cuda[k])
    if a.dtype==bool: print(k,'differ',int(np.count_nonzero(a!=b)))
    else: print(k,'difference',float(np.max(np.abs(a-b))), 'failed',int(np.count_nonzero(np.abs(a-b)>1e-6+1e-6*np.abs(a))))
