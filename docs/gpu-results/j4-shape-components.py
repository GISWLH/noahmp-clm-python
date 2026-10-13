import hashlib,inspect,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from noahmp_clm.gpu import jax_core as jc
out=Path('/home/lhwang/work/noahmp-phase3b/profile_output/j4-shape')
p=np.load(out/'prestep.npz');f=p['f'].copy();i=p['i'];u=p['u'];f[:,25]=p['ts_large']
# Other columns' pinned temperature is irrelevant to column 2's independent physics.
source=inspect.getsource(jc.step_column_jax).replace('_secant_solve(residual, tair, tair + 1.0)','fvec[25]')
source=source.replace('jnp.asarray(0.0, dtype=fvec.dtype), rnet - h - g_snow,','jnp.asarray(0.0, dtype=fvec.dtype), rnet - h - g_snow,\n            jnp.asarray(0.0, dtype=fvec.dtype), jnp.asarray(0.0, dtype=fvec.dtype),')
source=source.replace('rnet - h - le_g - le_c - tr_le - g_soil,','rnet - h - le_g - le_c - tr_le - g_soil, le_g, le_c,')
source=source.replace('melt, canw_after_energy, residual,','melt, canw_after_energy, residual, le_g_dbg, le_c_dbg,')
source=source.replace('    rs = _stomatal_resistance(', '    beta_root_dbg = _soil_stress(smc, root_frac, smcmax, smcwlt, smcref, bexp, _option(ivec, \"OptSoilWaterTranspiration\"))[0]\n    rs = _stomatal_resistance(')
source=source.replace('swe, canw, residual,','swe, canw, residual, le_g_dbg, le_c_dbg, tr_le, rs, avail_root, rn_canopy, ce_initial, rho,\n        beta_root_dbg,')
(out/'fixed-components.py').write_text(source)
ns=jc.__dict__.copy();exec(source,ns);fn=jax.jit(jax.vmap(ns['step_column_jax']))
r={'created_utc':datetime.now(timezone.utc).isoformat(),'fields':['ts','rnet','h','le','g','et_mm','prcp_mm','runoff_mm','melt_mm','swe','canw','residual','ground_evap','canopy_evap','transpiration','stomatal_resistance','available_root','rn_canopy','ce_initial','rho','beta_root_before_water'],'hlo':{}}
results=[]
for label,sl in [('n64800',slice(None)),('n1',slice(2,3))]:
 args=tuple(jnp.asarray(x[sl]) for x in (f,i,u)); exe=fn.lower(*args).compile();hlo=exe.as_text();(out/(label+'-components.hlo.txt')).write_text(hlo);r['hlo'][label]=hashlib.sha256(hlo.encode()).hexdigest();d=np.asarray(jax.block_until_ready(exe(*args))[2]);results.append(d[2] if label=='n64800' else d[0])
r['large']=results[0].tolist();r['single']=results[1].tolist();r['difference']=(results[0].astype(np.float64)-results[1]).tolist()
(out/'components.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)


