"""Actual N=27 height trial: shared feedback arithmetic, original hard gates."""
from pathlib import Path
import numpy as np
import pytest
from test_gpu_clm_layered_column import host,compare_same_precision,budgets
from test_gpu_clm_bgc_column import cn_budgets
from clm_soil_oracle import fixture_suction


def test_actual_height_feedback_same_precision_and_four_budgets():
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.cuda.clm_bgc_column import step_clm_bgc_columns_cuda
    from noahmp_clm.gpu.clm_bgc_column import step_clm_bgc_columns_jax
    with np.load(Path(__file__).parent/'fixtures/height_most_feedback_rounding.npz',allow_pickle=False) as z:v=dict(z)
    a=tuple(v[f'a{k}'].copy() for k in range(16));a[4][:,7:10]=v['forcing_environment']
    a=(*a[:2],fixture_suction(a[2],.355),*a[3:])
    kwargs=dict(environment=v['environment'],calendar=v['calendar'],most_geometry=v['geometry'],return_surface=True)
    cu=host(step_clm_bgc_columns_cuda(*a,v['forcing'],**kwargs),cp.asnumpy)
    ja=host(step_clm_bgc_columns_jax(*a,v['forcing'],**kwargs))
    maximum=compare_same_precision(cu,ja)
    print('height feedback N=27 same precision',maximum,flush=True)
    print('height feedback iterations',cu['surface']['most']['iterations'].tolist(),flush=True)
    print('height feedback M/H',budgets(a[:8],v['forcing'],np.zeros((27,5),np.float32),cu),flush=True)
    print('height feedback C/N',cn_budgets(a,cu,np.zeros(27,np.float32)),flush=True)
