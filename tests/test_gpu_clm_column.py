from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.column import ColumnState,Forcing,default_soil_params
from noahmp_clm.clm_driver import step_clm_column
from noahmp_clm.gpu.clm_column import pack_clm_columns,site_array,validate_clm_batch,CLM_DIAGNOSTICS
from noahmp_clm.gpu.pack import forcing_to_vec,vec_to_column_state,forcing_vec_to_forcing
from noahmp_clm.physics.photosynthesis import LeafParameters
from noahmp_clm.gpu.photosynthesis import pack_leaf_parameters


def column_cases(dtype=np.float32,random_count=0):
    states=[];forcings=[]
    for case in range(24):
        s=ColumnState(ts=298.,tsoil=np.full(4,295.))
        sw,qa,rain,lw=800.,.01,0.,350.
        if case%8 in (1,3): sw,lw=0.,280.
        if case%8==2: rain=1e-6
        if case%8==3: qa=.019
        if case%8==4: s.smc[:]=s.soil.smcwlt
        if case%8==5: s.fveg=0.
        if case%8==6: s.lai=0.
        if case%8==7:
            s.canw=.004;s.smc[:]=[.15,.25,.35,.42];s.root_frac[:]=[0.,.3,.2,.5]
        s.options=replace(s.options,OptSurfaceDrag=1+case//16,
            OptGroundResistanceEvap=1+case%4,OptSoilWaterTranspiration=1+case%3,
            OptRunoffSurface=1+case%4,OptRunoffSubsurface=1+(case+1)%4,
            OptSoilSupercoolWater=1+case%2,OptSoilPermeabilityFrozen=1+case%2)
        states.append(s);forcings.append(forcing_to_vec(Forcing(1800.,298.,qa,3.,101325.,sw,lw,rain)))
    sites=list(site_array(len(states),dtype))
    rng=np.random.default_rng(6729)
    from noahmp_clm.physics.energy import qsat
    for case in range(random_count):
        p=default_soil_params(("sand","loam","clay")[case%3])
        tair=rng.uniform(277.,306.)
        s=ColumnState(ts=tair+rng.uniform(-4.,4.),tsoil=rng.uniform(271.,300.,4),
            smc=rng.uniform(p.smcwlt*.8,p.smcmax,4),soil=p,lai=rng.uniform(.05,6.),
            fveg=rng.uniform(.05,1.),canw=rng.uniform(0.,.001),z0m=rng.uniform(.01,.2))
        s.options=replace(s.options,OptSurfaceDrag=1+case%2,OptGroundResistanceEvap=1+case%4,
            OptSoilWaterTranspiration=1+case%3,OptSoilSupercoolWater=1+case%2,
            OptSoilPermeabilityFrozen=1+case%2)
        states.append(s)
        forcings.append(forcing_to_vec(Forcing(rng.uniform(300.,2400.),tair,rng.uniform(.4,.9)*qsat(tair,100000.),
            rng.uniform(.5,6.),100000.,rng.uniform(0.,900.),rng.uniform(270.,400.),1e-6 if case%3==0 else 0.)))
        sites.append([rng.uniform(370.,480.),rng.uniform(.1,1.),rng.uniform(.1,.9),rng.uniform(.02,.06),rng.uniform(.2,1.)])
    sf,si=pack_clm_columns(states,dtype)
    return sf,si,np.asarray(forcings,dtype),np.asarray(sites,dtype)


def reference(s,i,f,v):
    p=LeafParameters(*pack_leaf_parameters(dtype=s.dtype).astype(float))
    out=[];diagnostics=[]
    for state,opts,forcing,site in zip(s,i,f,v):
        new,d=step_clm_column(vec_to_column_state(state,opts,[.1,.3,.6,1.]),forcing_vec_to_forcing(forcing),
            co2_ppm=float(site[0]),cosine_zenith=float(site[1]),diffuse_fraction=float(site[2]),
            leaf_dimension=float(site[3]),nitrogen_factor=float(site[4]),leaf_parameters=p)
        out.append(pack_clm_columns([new])[0][0])
        diagnostics.append([d.get(k,1.) for k in CLM_DIAGNOSTICS])
    return np.asarray(out),np.asarray(diagnostics)


def test_clm_batch_rejects_unsupported_winter_options_and_geometry():
    inputs=column_cases()
    validate_clm_batch(*inputs,np)
    snow=inputs[0].copy();snow[0,26]=.01
    with pytest.raises(NotImplementedError,match="five-layer"):
        validate_clm_batch(snow,*inputs[1:],np)
    bad=inputs[2].copy();bad[0,0]=0.
    with pytest.raises(ValueError): validate_clm_batch(*inputs[:2],bad,inputs[3],np)
    state=ColumnState();state.dz[0]=.2
    with pytest.raises(ValueError,match="thicknesses"): pack_clm_columns([state])


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
def test_jax_clm_column_against_numpy_and_water_budget(dtype):
    jax=pytest.importorskip("jax")
    if dtype is np.float64 and not jax.config.x64_enabled: pytest.skip("enable JAX x64")
    from noahmp_clm.gpu.clm_column import step_clm_columns_jax
    inputs=column_cases(dtype,64)
    expected,diag=reference(*inputs)
    new,opts,d=map(np.asarray,step_clm_columns_jax(*inputs))
    assert np.all(np.isfinite(new))
    np.testing.assert_array_equal(opts,inputs[1])
    np.testing.assert_allclose(new[:,:4],expected[:,:4],rtol=0,atol=.01)
    np.testing.assert_allclose(new[:,4:12],expected[:,4:12],rtol=.01,atol=1e-5)
    for k in (2,3,4,12): np.testing.assert_allclose(d[:,k],diag[:,k],rtol=.1,atol=.1)
    assert np.max(np.abs(d[:,17])) < (2e-7 if dtype is np.float32 else 1e-12)
    assert np.max(np.abs(d[:,18])) < (.001 if dtype is np.float32 else 1e-9)
    assert np.all(d[:,20:]==1)


def test_cuda_clm_column_same_precision_and_multiple_steps():
    pytest.importorskip("jax")
    cp=pytest.importorskip("cupy")
    from noahmp_clm.gpu.clm_column import step_clm_columns_jax
    from noahmp_clm.gpu.cuda.clm_column import step_clm_columns_cuda
    s,i,f,v=column_cases(random_count=64)
    js,cs=s,s
    for step in range(8):
        jf=step_clm_columns_jax(js,i,f,v,return_canopy=True)
        cf=step_clm_columns_cuda(cs,i,f,v,return_canopy=True)
        js,cs=jf[0],cf[0]
        for a,b in zip(jf[:3],cf[:3]): np.testing.assert_allclose(cp.asnumpy(b),np.asarray(a),rtol=1e-6,atol=1e-6)
        for name,a in jf[3].items():
            if isinstance(a,dict):
                for key,value in a.items(): np.testing.assert_allclose(cp.asnumpy(cf[3][name][key]),np.asarray(value),rtol=1e-6,atol=1e-6)
            else: np.testing.assert_allclose(cp.asnumpy(cf[3][name]),np.asarray(a),rtol=1e-6,atol=1e-6)
        assert np.max(np.abs(np.asarray(jf[2])[:,17])) < 2e-7
