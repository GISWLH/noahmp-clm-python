"""Development probe: preserve optimized HLO and discrepant leaf rows."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
import numpy as np
import jax
import cupy as cp
from test_gpu_photosynthesis import leaf_cases
from noahmp_clm.physics.photosynthesis import LeafParameters
from noahmp_clm.gpu.photosynthesis import leaf_kernel_jax, pack_leaf_parameters
from noahmp_clm.gpu.cuda.photosynthesis import run_leaves_cuda

u=leaf_cases(np.float32)
p=LeafParameters(tpu25=8.)
compiled=jax.jit(jax.vmap(leaf_kernel_jax, in_axes=(0,None))).lower(
    u,pack_leaf_parameters(p,np.float32)).compile()
Path("leaf-hlo.txt").write_text(compiled.as_text())
a=np.asarray(compiled(u,pack_leaf_parameters(p,np.float32)))
b=cp.asnumpy(run_leaves_cuda(u,p))
bad=np.flatnonzero(np.any(abs(a-b)>1e-6+1e-6*abs(a),axis=1))
for i in bad:
    print(i, "u",u[i].tolist(),"jax",a[i].tolist(),"cuda",b[i].tolist())
