"""Original warm leafy-MOST SAI0.8 split/rounding regression, all380 paths."""
from pathlib import Path
import json
import numpy as np
import pytest
from test_gpu_clm_layered_column import host,science
from test_gpu_clm_bgc_column import reference
from test_gpu_snicar import assert_science_close
from benchmarks import run_fluxnet_native as native


def packet(dtype):
    with np.load(Path(__file__).with_name('fixtures')/'stem_station_second_step.npz',allow_pickle=False) as z:d=dict(z)
    arrays=[d['a'+str(k)].copy() if d['a'+str(k)].dtype.kind in 'iu' else d['a'+str(k)].astype(dtype) for k in range(19)]
    if dtype==np.float64:
        # Float32-normalized parameter vectors need normalization at the
        # promoted precision, just as root weights do on NumPy reconstruction.
        # The original float32 regression packet and all conserved stores stay
        # unchanged; this branch is a separate float64 reference case.
        for section in (slice(0,3),slice(9,12)):
            arrays[12][:,section]/=arrays[12][:,section].sum(1,keepdims=True)
        arrays[6]/=arrays[6].sum(1,keepdims=True)
        arrays[5][:,:2]=arrays[7].sum(1)
        arrays[4][:,0]=arrays[11][:,0]*arrays[12][:,19]/arrays[4][:,1]
        arrays[4][:,11]=np.where(arrays[11][:,0]>0.,np.minimum(1.,arrays[11][:,3]*arrays[12][:,3]/np.maximum(arrays[11][:,0],np.finfo(float).tiny)),0.)
    arrays[4][:,7:10]=d['environment'].astype(dtype)
    kwargs=dict(environment=d['calendar_values'].astype(dtype),calendar=d['calendar_flags'],
        nitrogen_deposition_gn_m2=d['nitrogen'].astype(dtype),deposition=d['deposition'].astype(dtype),
        most_geometry=d['geometry'].astype(dtype),return_surface=True)
    return tuple(arrays),d['forcing'].astype(dtype),kwargs


def audit(arrays,f,kwargs,out):
    following=tuple(host({'x':v})['x'] for v in native.next_station_arrays(arrays,out))
    _,residual,bound,_=native.audit_step(arrays,following,f,kwargs['deposition'],kwargs['nitrogen_deposition_gn_m2'],out)
    return dict(maximum_absolute=np.max(abs(residual),axis=0).tolist(),maximum_original_bound=bound.max(0).tolist())


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_original_stem_split_station_numpy_reference_and_independent_budgets(dtype):
    jax=pytest.importorskip('jax')
    if dtype==np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.clm_structure_column import step_clm_structure_columns_jax
    from noahmp_clm.physics.canopy_most import MOSTGeometry
    arrays,f,kw=packet(dtype);out=host(step_clm_structure_columns_jax(arrays,f,kind='bgc',**kw));maximum={}
    for k,row in enumerate(zip(*arrays)):
        ref=reference(row[:16],f[k],kw['deposition'][k],kw['environment'][k],kw['calendar'][k],kw['nitrogen_deposition_gn_m2'][k],1.,
            most_geometry=MOSTGeometry(*map(float,kw['most_geometry'][k])),canopy_structure=row[-3:])
        one={key:({a:b[k] for a,b in value.items()} if key=='diagnostics' else value[k]) for key,value in out.items()
            if key in ('snow_cells','soil_cells','options','storage','diagnostics','impurity_output_kgm2','condensate')}
        errors=science(one,ref,dtype)
        for key in ('h','le'):maximum[key]=max(maximum.get(key,0.),errors[key])
        for key in ('gpp_gc_m2','nee_gc_m2'):assert_science_close(out['diagnostics'][key][k],ref['diagnostics'][key],relative=.1,zero_floor=.001,absolute=.0001)
        assert_science_close(out['structure_state'][k],ref['structure_state'],relative=.1,zero_floor=.0001,absolute=1e-6)
    print('Original stem split station reference',str(dtype),json.dumps(dict(maximum_absolute=maximum,budgets=audit(arrays,f,kw,out))),flush=True)


def test_original_stem_split_station_cuda_jax_all380_paths():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.clm_structure_column import step_clm_structure_columns_cuda,step_clm_structure_columns_jax
    arrays,f,kw=packet(np.float32);a=host(step_clm_structure_columns_cuda(arrays,f,kind='bgc',**kw),cp.asnumpy)
    b=host(step_clm_structure_columns_jax(arrays,f,kind='bgc',**kw));pa=native.next_station_arrays(arrays,a);pb=native.next_station_arrays(arrays,b)
    maximum=native.compare_same_precision(dict(output=a,next_state={str(k):v for k,v in enumerate(pa)}),dict(output=b,next_state={str(k):v for k,v in enumerate(pb)}))
    print('Original stem split all380 same precision',json.dumps(dict(maximum_absolute=maximum,cuda_budgets=audit(arrays,f,kw,a),jax_budgets=audit(arrays,f,kw,b))),flush=True)
