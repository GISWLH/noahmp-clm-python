"""Print a saved failing column's device Newton direction and merit values."""
import inspect
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp
from noahmp_clm.gpu import column_heat as heat

source=inspect.getsource(heat.joint_heat_kernel_jax)
source=source.replace('        def search(s):',
    "        jax.debug.print('ITER {c} H {h} RES {r} NORM {n} D {d} DELTA {v} SCALE {s}',c=count,h=h,r=res,n=norm,d=deriv,v=delta,s=scale)\n        def search(s):")
source=source.replace('            return tries+1,scale*.5,j.where(take,trial,candidate),take',
    "            jax.debug.print('SEARCH {i} SCALE {s} MAX {m} L1 {l} GOOD {g} TAKE {a}',i=tries,s=scale,m=j.max(j.abs(r)/weights),l=merit_of(r,weights),g=good,a=take)\n            return tries+1,scale*.5,j.where(take,trial,candidate),take")
scope=dict(vars(heat),jax=jax)
exec(compile(source,str(Path(__file__)),'exec'),scope)
inputs=np.load(sys.argv[1]);row=int(sys.argv[2])
result=jax.jit(scope['joint_heat_kernel_jax'])(*[jnp.asarray(inputs[k][row]) for k in ('cells','props','options','boundary')])
jax.block_until_ready(result)
print(np.asarray(result[-1]))
