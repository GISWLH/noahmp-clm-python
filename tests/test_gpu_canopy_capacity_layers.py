"""Per-column and per-layer coefficients through the actual phase kernel."""
from pathlib import Path
import os
import numpy as np
import pytest
from test_gpu_clm_layered_column import host
from test_gpu_snicar import assert_science_close


def saved():
    with np.load(Path(__file__).parent/'fixtures/clm_canopy_capacity_reference.npz',allow_pickle=False) as archive:
        return dict(archive)


def capture(name,**values):
    if os.environ.get('NOAHMP_CAPACITY_COLUMN_DIAGNOSTICS'):
        folder=Path(os.environ['NOAHMP_CAPACITY_COLUMN_DIAGNOSTICS']);folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/(name+'.npz'),**values)


@pytest.mark.parametrize('precision',['32','64'])
def test_layer_coefficients_phase_matches_ehpc_reference(precision):
    jax=pytest.importorskip('jax')
    if precision=='64' and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.canopy_phase_energy import run_canopy_phase_jax
    data=saved();prefix='phase'+precision+'/'
    arrays=tuple(data[prefix+'a'+str(k)] for k in range(9));parameters=data[prefix+'parameters']
    got=host(run_canopy_phase_jax(*arrays,parameters=parameters,tolerance=.005 if precision=='32' else 1e-7,return_raw=True))
    maximum={}
    for name in ['h','le','gpp']:
        expected=data[prefix+'ref/'+name];maximum[name]=float(abs(got[name]-expected).max())
        assert_science_close(got[name],expected,relative=.1,zero_floor=.1 if name=='gpp' else 10.,absolute=.01 if name=='gpp' else 1.)
    np.testing.assert_allclose(got['leaf_temperature'],data[prefix+'ref/leaf_temperature'],rtol=0.,atol=.05)
    assert np.all(got['converged'])
    capture('phase-reference'+precision,raw=got['raw'])
    print('layer coefficient phase reference',precision,maximum,flush=True)


def test_layer_coefficients_cuda_jax_all300_raw_values():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.canopy_phase_energy import run_canopy_phase_jax
    from noahmp_clm.gpu.cuda.canopy_phase_energy import run_canopy_phase_cuda
    data=saved();arrays=tuple(data['phase32/a'+str(k)] for k in range(9));parameters=data['phase32/parameters']
    a=host(run_canopy_phase_cuda(*arrays,parameters=parameters,return_raw=True),cp.asnumpy)
    b=host(run_canopy_phase_jax(*arrays,parameters=parameters,return_raw=True))
    capture('phase-allraw32',cuda=a['raw'],jax=b['raw'],parameters=parameters)
    np.testing.assert_allclose(a['raw'],b['raw'],rtol=1e-6,atol=1e-6)
    for name in ['iterations','converged','restarts']:np.testing.assert_array_equal(a[name],b[name])
    print('layer coefficient all300raw maximum',float(abs(a['raw'].astype(float)-b['raw'].astype(float)).max()),flush=True)
