"""Locate grain-growth discrepancies on all source-table nodes."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
import test_gpu_snow_processes as ref
from noahmp_clm.gpu.snow_evolution import run_snow_evolution_jax
from noahmp_clm.physics.snowpack import Snowpack

c,o,u=ref.evolution_cases(np.float64,0,table_grid=True)
result={k:np.asarray(v) for k,v in run_snow_evolution_jax(c,o,u).items()}
fail=[]
for row in range(len(c)):
    q=ref.evolution_reference(c[row],o[row],u[row]);error=float(np.max(np.abs(result['cells'][row,:,3]-q['cells'][:,3])))
    if error>1e-5:
        n=int(o[row,0]);p=Snowpack(c[row,:n,0],c[row,:n,1],c[row,:n,2],c[row,:n,3])
        fail.append(dict(row=row,error_um=error,temperature=p.temperature.tolist(),density=p.density.tolist(),gradient=u[row,1:1+n].tolist(),
            actual=result['cells'][row,:n,3].tolist(),reference=q['cells'][:n,3].tolist(),
            gpu_parameters=result['aging_parameters'][row,:n].tolist(),reference_parameters=q['aging_parameters'][:n].tolist()))
print(json.dumps(dict(failed=len(fail),cases=sorted(fail,key=lambda v:-v['error_um'])[:30]),indent=2),flush=True)
