"""Diagnostic-only replay of J4 column 2 across batch shapes."""
import hashlib, importlib.util, inspect, json, os, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from noahmp_clm.gpu import jax_core as jc
root=Path.cwd()
out=Path('/home/lhwang/work/noahmp-phase3b/profile_output/j4-shape')
out.mkdir(exist_ok=True,parents=True)
spec=importlib.util.spec_from_file_location('extended',root/'benchmarks/check_gpu_extended.py')
ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
f,i=ext.initial_columns(64800,np.float32)
u=ext.daily_forcing(np.arange(64800),0,np.float32,20261013,48)
report={'started_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'jax':jax.__version__,'devices':str(jax.devices()),'env':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','JAX_PLATFORMS','JAX_ENABLE_X64','XLA_FLAGS']},'jax_source_sha256':hashlib.sha256(Path(jc.__file__).read_bytes()).hexdigest(),'input_sha256':ext.digest_arrays(f,i),'stages':{},'hlo':{}}
def save():
 (out/'report.json').write_text(json.dumps(report,indent=2))
def run(label,fn,*args):
 print('RUN',label,flush=True)
 compiled=fn.lower(*args).compile()
 hlo=compiled.as_text();(out/(label+'.hlo.txt')).write_text(hlo)
 report['hlo'][label]={'sha256':hashlib.sha256(hlo.encode()).hexdigest(),'bytes':len(hlo)}
 r=jax.block_until_ready(compiled(*args));save();return tuple(np.asarray(a) for a in r)
def compare(a,b):
 delta=np.abs(a.astype(np.float64)-b)
 return {'max_abs':float(delta.max(initial=0)),'different':int(np.count_nonzero(delta)),'over_1e6':int(np.count_nonzero(delta>1e-6+1e-6*np.abs(b))),'different_indices':np.argwhere(delta>0).tolist(),'large':a.tolist(),'single':b.tolist()}
a=run('n64800-t48',jc._BATCH_JIT,jnp.asarray(f),jnp.asarray(i),jnp.asarray(u))
b=run('n1-t48',jc._BATCH_JIT,jnp.asarray(f[2:3]),jnp.asarray(i[2:3]),jnp.asarray(u[2:3]))
report['stages']['original_step17_diag']=compare(a[2][2,17],b[2][0,17])
report['stages']['original_all_diag']=compare(a[2][2],b[2][0])
report['stages']['original_final_state']=compare(a[0][2],b[0][0]);save()
ap=run('n64800-t17',jc._BATCH_JIT,jnp.asarray(f),jnp.asarray(i),jnp.asarray(u[:,:17]))
bp=run('n1-t17',jc._BATCH_JIT,jnp.asarray(f[2:3]),jnp.asarray(i[2:3]),jnp.asarray(u[2:3,:17]))
report['stages']['prefix_diag_vs_original_large']=compare(ap[2][2],a[2][2,:17])
report['stages']['prefix_diag_vs_original_single']=compare(bp[2][0],b[2][0,:17])
report['stages']['prestep_state']=compare(ap[0][2],bp[0][0]);save()
np.savez(out/'prestep.npz',f=ap[0],i=ap[1],u=u[:,17],f_single=bp[0],i_single=bp[1],ts_large=a[2][2,17,0],ts_single=b[2][0,17,0],original_large=a[2][2],original_single=b[2][0])
step=jax.jit(jax.vmap(jc.step_column_jax))
as1=run('n64800-step',step,jnp.asarray(ap[0]),jnp.asarray(ap[1]),jnp.asarray(u[:,17]))
bs1=run('n1-step',step,jnp.asarray(ap[0][2:3]),jnp.asarray(ap[1][2:3]),jnp.asarray(u[2:3,17]))
report['stages']['same_input_step_diag']=compare(as1[2][2],bs1[2][0]);save()
source=inspect.getsource(jc.step_column_jax).replace('_secant_solve(residual, tair, tair + 1.0)', 'fvec[25]')
if source==inspect.getsource(jc.step_column_jax):raise RuntimeError('fixedTs injection failed')
(out/'fixed-step.py').write_text(source)
ns=jc.__dict__.copy();exec(source,ns)
fixed=jax.jit(jax.vmap(ns['step_column_jax']))
ff=ap[0].copy();ff[:,25]=a[2][:,17,0]
ax=run('n64800-fixedts',fixed,jnp.asarray(ff),jnp.asarray(ap[1]),jnp.asarray(u[:,17]))
bx=run('n1-fixedts',fixed,jnp.asarray(ff[2:3]),jnp.asarray(ap[1][2:3]),jnp.asarray(u[2:3,17]))
report['stages']['same_input_fixedTs_diag']=compare(ax[2][2],bx[2][0]);save()
report['finished_utc']=datetime.now(timezone.utc).isoformat();save();print('DONE',flush=True)
