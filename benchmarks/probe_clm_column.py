"""Locate device-stage differences before interpreting whole-column errors."""
import json
import sys
from pathlib import Path
import numpy as np
import cupy as cp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tests"))
from test_gpu_clm_column import column_cases
from noahmp_clm.gpu.clm_column import _compiled_stages,step_clm_columns_jax,CLM_DIAGNOSTICS
from noahmp_clm.gpu.cuda.clm_column import _kernels,step_clm_columns_cuda


def differences(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    with np.errstate(invalid="ignore"): delta=np.abs(a-b)
    delta=np.where(a==b,0.,delta)
    idx=np.unravel_index(np.argmax(delta),delta.shape)
    bad=~np.isclose(a,b,rtol=1e-6,atol=1e-6)
    return {"maximum_abs":float(delta[idx]),"failures":int(bad.sum()),
        "maximum_index":list(map(int,idx)),"jax":float(a[idx]),"cuda":float(b[idx])}


s,i,f,v=column_cases(random_count=32)
ja=_compiled_stages()[0](s,i,f,v)
ca=[cp.empty(x.shape,cp.float32) for x in ja]
_kernels()[0]((1,),(64,),(*map(cp.asarray,(s,i,f,v)),*ca,np.int32(len(s))))
result={"preparation":{str(k):differences(a,cp.asnumpy(b)) for k,(a,b) in enumerate(zip(ja,ca))}}
result["preparation_element_differences"]={str(k):np.argwhere(np.asarray(a)!=cp.asnumpy(b)).tolist()
    for k,(a,b) in enumerate(zip(ja,ca))}
from noahmp_clm.gpu.canopy_energy import CANOPY_INPUTS
result["prepared_boundaries"]={name:differences(ja[1][:,k],cp.asnumpy(ca[1][:,k])) for k,name in enumerate(CANOPY_INPUTS)}
np.savez_compressed("prepare-probe.npz",**{f"jax_{k}":np.asarray(x) for k,x in enumerate(ja)},
    **{f"cuda_{k}":cp.asnumpy(x) for k,x in enumerate(ca)},s=s,i=i,f=f,v=v)
Path("prepare-hlo.txt").write_text(_compiled_stages()[0].lower(s,i,f,v).compile().as_text())
try:
    j=step_clm_columns_jax(s,i,f,v,return_canopy=True)
    c=step_clm_columns_cuda(s,i,f,v,return_canopy=True)
    result["state"]=differences(j[0],cp.asnumpy(c[0]))
    result["diagnostics"]={name:differences(j[2][:,k],cp.asnumpy(c[2][:,k])) for k,name in enumerate(CLM_DIAGNOSTICS)}
    result["canopy"]={name:differences(a,cp.asnumpy(c[3][name])) for name,a in j[3].items() if name!="gas_exchange"}
    result["leaf_gas"]={name:differences(a,cp.asnumpy(c[3]["gas_exchange"][name])) for name,a in j[3]["gas_exchange"].items()}
except Exception as e:
    result["error"]=str(e)
print(json.dumps(result,indent=2))
