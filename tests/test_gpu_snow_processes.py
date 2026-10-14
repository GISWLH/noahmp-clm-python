"""Snow vapor/microstructure devices against NumPy and independent equations."""
import numpy as np
import pytest
from noahmp_clm.physics.snowpack import Snowpack,TF,CI,CW,LF,regrid
from noahmp_clm.physics.snow_vapor import exchange_vapor
from noahmp_clm.physics.snow_evolution import compact,grain_growth,deposit_impurities,aging_parameters,fresh_snow_properties
from noahmp_clm.gpu.snow_mass import pack_snow_transfer
from noahmp_clm.gpu.snow_vapor import pack_snow_vapor,validate_snow_vapor
from noahmp_clm.gpu.snow_evolution import pack_snow_evolution,validate_snow_evolution
from test_gpu_snow_mass import budgets


def stack(rows):return tuple(np.stack([r[k] for r in rows]) for k in range(3))


def vapor_cases(dtype=np.float64,random_count=200):
    rows=[]
    def add(p,amount,**kwargs):rows.append(pack_snow_vapor(p,amount,dtype=dtype,**kwargs))
    for t,f in ((260.,0.),(TF,0.),(TF,.3),(TF,.9),(278.,1.)):
        p=Snowpack([3.],[3*((CI*(1-f)+CW*f)*(t-TF)+LF*f)],[.02],impurities=np.full((1,5),1e-6))
        add(p,.6);add(p,3.);add(p,0.)
        for dep in (260.,TF,280.):add(p,-.4,temperature=dep)
    p=Snowpack.from_ice([2.,5.],260.);p=Snowpack(p.mass,p.enthalpy,p.thickness,impurities=np.full((2,5),1e-6))
    add(p,2.);add(p,2.,remap=False);add(Snowpack.empty(),-.3,temperature=260.);add(Snowpack.empty(),0.)
    add(Snowpack.from_ice([.05],TF),np.nextafter(dtype(.05),dtype(0)))
    rng=np.random.default_rng(912220)
    for case in range(random_count):
        n=1+case%5;m=10.**rng.uniform(-3.,2.,n);h=m*rng.uniform(-CI*40.,1.1*LF,n)
        p=Snowpack(m,h,m/rng.uniform(50,700,n),rng.uniform(54.5,1500,n),m[:,None]*10.**rng.uniform(-12,-5,(n,5)))
        amount=rng.uniform(.01,.98)*m[0] if case%2 else -10.**rng.uniform(-5.,1.)
        add(p,amount,temperature=rng.uniform(250.,290.),remap=case%3!=0)
    return stack(rows)


def vapor_reference(c,n,u,dtype):
    c=np.asarray(c,float);n=int(n);u=np.asarray(u,float)
    p=Snowpack(c[:n,0],c[:n,1],c[:n,2],c[:n,3],c[:n,4:]);amount=u[0]
    # The documented exhaustion guard scales with storage precision; compare
    # the physically executed amount, including that explicitly bounded snap.
    if amount>0 and abs(amount-c[0,0])<=64*np.finfo(dtype).eps*max(amount,c[0,0]):amount=c[0,0]
    q,d=exchange_vapor(p,amount,deposition_temperature=TF+u[1],remap=bool(u[2]))
    cells,nn,_,_=pack_snow_transfer(q,0)
    return dict(cells=cells,counts=nn,outward=np.r_[d['mass_kgm2'],d['advected_enthalpy_jm2'],d['residue_kgm2']],
        ice_kgm2=d['ice_kgm2'],liquid_kgm2=d['liquid_kgm2'],latent_energy_jm2=d['latent_energy_jm2'])


def evolution_cases(dtype=np.float64,random_count=200,table_grid=False):
    rows=[]
    def add(p,mode,dt,**kwargs):rows.append(pack_snow_evolution(p,mode,dt,dtype=dtype,**kwargs))
    for mode in range(4):add(Snowpack.empty(),mode,1800.,deposition_rates=np.arange(1,6)*1e-12)
    for temp,grad,rho,radius,f in ((TF-15,0.,300.,54.5,0.),(TF-3,100.,100.,100.,0.),(TF,50.,200.,54.5,.1),(TF,150.,350.,500.,.033)):
        p=Snowpack([10.],[10*LF*f if f else 10*CI*(temp-TF)],[10/rho],[radius])
        add(p,1,1800.,gradient=grad)
    p=Snowpack([10.,50.],[2*LF,20*LF],[.03,.08]);add(p,0,1e8,ice_before=[10.,50.],remap=False)
    add(p,3,1800.,ice_before=[10.,50.],refrozen_mass=[1.,2.],gradient=[10.,20.])
    p=Snowpack.from_ice([10.,20.],TF,300.,100.);add(p,1,0.,refrozen_mass=[5.,0.])
    rng=np.random.default_rng(330161)
    for case in range(random_count):
        n=1+case%5;mode=case%4;m=10.**rng.uniform(-2.,2.,n);h=m*rng.uniform(-CI*55.,LF*1.05,n)
        p=Snowpack(m,h,m/rng.uniform(40,850,n),rng.uniform(30,1500,n),m[:,None]*10.**rng.uniform(-12,-5,(n,5)))
        add(p,mode,rng.uniform(0.,86400.),gradient=rng.uniform(0.,500.,n),ice_before=p.ice*rng.uniform(.5,2.,n),
            refrozen_mass=p.ice*rng.uniform(0.,.8,n),deposition_rates=10.**rng.uniform(-14.,-9.,5),
            air_temperature=rng.uniform(200.,280.),wind10=rng.uniform(0.,25.),remap=case%3!=0)
    if table_grid:
        for temp in np.linspace(TF-50,TF,11):
            for grad in np.linspace(0.,300.,31):
                for rho in np.linspace(50.,400.,8):
                    add(Snowpack.from_ice([10.],temp,rho,54.5),1,86400.,gradient=grad)
    return stack(rows)


def evolution_reference(c,o,u):
    c,u=np.asarray(c,float),np.asarray(u,float);n,mode,pre,ref,remap=map(int,o)
    p=Snowpack(c[:n,0],c[:n,1],c[:n,2],c[:n,3],c[:n,4:]);q=p;params=np.zeros((5,3));bare=np.zeros(5)
    if mode in (0,3):q=compact(q,u[0],ice_before=u[6:6+n] if pre else None,remap=False)
    if mode in (1,3):
        if n:params[:n]=np.array(aging_parameters(q.temperature,u[1:1+n],q.density)).T
        q=grain_growth(q,u[0],temperature_gradient=u[1:1+n],refrozen_mass=u[11:11+n] if ref else None)
    if mode in (0,3) and remap:q=regrid(q)
    if mode==2:q,bare=deposit_impurities(q,u[16:21],u[0])
    cells,nn,_,_=pack_snow_transfer(q,0);density,radius=fresh_snow_properties(TF+u[21],u[22])
    return dict(cells=cells,counts=nn,bare_deposition=bare,aging_parameters=params,fresh_density=density,fresh_radius=radius)


def scientific_close(a,b,floor,absolute,dtype):
    a,b=np.asarray(a),np.asarray(b);error=np.abs(a-b);normal=np.abs(b)>floor
    if dtype==np.float32:assert np.all(error<=np.where(normal,.1*np.abs(b),absolute)),(a,b,error)
    else:np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-7)


def cell_science(a,b,dtype):
    scientific_close(a,b,np.array([.001,1.,.0001,1.,*([1e-12]*5)]),
        np.array([.0001,.1,.00001,.1,*([1e-13]*5)]),dtype)


def process_budgets(c,result,addition=None):
    zero=np.zeros((len(c),7));r=dict(cells=result['cells'],incoming=zero.copy(),drainage=zero,runoff=zero,sublimation=zero)
    if 'outward' in result:r['sublimation']=result['outward']
    else:
        r['incoming'][:,2:]=addition;r['sublimation']=np.column_stack((np.zeros((len(c),2)),result['bare_deposition']))
    return budgets(c,r)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('kind',['vapor','evolution'])
def test_snow_process_reference_and_budget(dtype,kind):
    jax=pytest.importorskip('jax')
    if dtype is np.float64 and not jax.config.x64_enabled:pytest.skip('JAX x64 required')
    from noahmp_clm.gpu.snow_vapor import run_snow_vapor_jax
    from noahmp_clm.gpu.snow_evolution import run_snow_evolution_jax
    inputs=vapor_cases(dtype) if kind=='vapor' else evolution_cases(dtype)
    run=run_snow_vapor_jax if kind=='vapor' else run_snow_evolution_jax
    result={k:np.asarray(v) for k,v in run(*inputs).items()};maximum=np.zeros(9)
    for row,values in enumerate(zip(*inputs)):
        ref=vapor_reference(*values,dtype) if kind=='vapor' else evolution_reference(*values)
        assert result['counts'][row]==ref['counts'],(row,kind)
        cell_science(result['cells'][row],ref['cells'],dtype)
        maximum=np.maximum(maximum,np.max(np.abs(result['cells'][row]-ref['cells']),axis=0))
        if kind=='vapor':
            scientific_close(result['outward'][row],ref['outward'],np.array([.001,1.,*([1e-12]*5)]),np.array([.0001,.1,*([1e-13]*5)]),dtype)
            for key in ('ice_kgm2','liquid_kgm2','latent_energy_jm2'):scientific_close(result[key][row],ref[key],.001 if 'kgm2' in key else 1.,.0001 if 'kgm2' in key else .1,dtype)
        else:
            for key in ('fresh_density','fresh_radius'):scientific_close(result[key][row],ref[key],1.,.1,dtype)
            np.testing.assert_allclose(result['aging_parameters'][row],ref['aging_parameters'],rtol=1e-3 if dtype is np.float32 else 1e-8,atol=1e-5)
    c,o,u=inputs;addition=None if kind=='vapor' else np.where((o[:,1]==2)[:,None],u[:,16:21]*u[:,:1],0.)
    print(kind,dtype,'cell max absolute',maximum,'budget',process_budgets(c,result,addition),flush=True)


@pytest.mark.parametrize('kind',['vapor','evolution'])
def test_snow_process_cuda_same_precision(kind):
    cp=pytest.importorskip('cupy');pytest.importorskip('jax')
    from noahmp_clm.gpu.snow_vapor import run_snow_vapor_jax
    from noahmp_clm.gpu.cuda.snow_vapor import run_snow_vapor_cuda
    from noahmp_clm.gpu.snow_evolution import run_snow_evolution_jax
    from noahmp_clm.gpu.cuda.snow_evolution import run_snow_evolution_cuda
    inputs=vapor_cases(np.float32) if kind=='vapor' else evolution_cases(np.float32)
    jaxrun,cudarun=(run_snow_vapor_jax,run_snow_vapor_cuda) if kind=='vapor' else (run_snow_evolution_jax,run_snow_evolution_cuda)
    a,b=jaxrun(*inputs),cudarun(*inputs)
    for key in a:
        x,y=np.asarray(a[key]),cp.asnumpy(b[key]);print(kind,'CUDA/JAX32',key,np.max(x!=y) if x.dtype==bool else np.max(np.abs(x-y)),flush=True)
        np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-6)


def test_snow_process_contracts():
    c,n,u=vapor_cases(np.float64,0);validate_snow_vapor(c,n,u,np)
    bad=u.copy();bad[0,0]=4.
    with pytest.raises(ValueError,match='supply'):validate_snow_vapor(c,n,bad,np)
    c,o,u=evolution_cases(np.float64,0);validate_snow_evolution(c,o,u,np)
    bad=u.copy();bad[0,0]=-1.
    with pytest.raises(ValueError,match='step'):validate_snow_evolution(c,o,bad,np)


def test_gpu_grain_growth_independent_ode():
    pytest.importorskip('jax');solve_ivp=pytest.importorskip('scipy.integrate').solve_ivp
    from noahmp_clm.gpu.snow_evolution import run_snow_evolution_jax
    c,o,u=evolution_cases(np.float32,0);got=np.asarray(run_snow_evolution_jax(c,o,u)['cells'])
    errors=[]
    for row in range(4,8):
        p=Snowpack(c[row,:1,0],c[row,:1,1],c[row,:1,2],c[row,:1,3]);grad=float(u[row,1])
        tau,kappa,rate=[float(x[0]) for x in aging_parameters(p.temperature,grad,p.density)]
        fraction=min(.1,float(p.liquid[0]/p.mass[0]));wet=422000*fraction**3/(4*np.pi)
        def rhs(_,radius):return rate/3600*(tau/(radius-54.5+tau))**(1/kappa)+wet/radius**2
        ref=solve_ivp(rhs,(0,1800),p.radius_um,rtol=1e-11,atol=1e-11).y[0,-1]
        error=abs(float(got[row,0,3])-ref)/ref*100;errors.append(error);assert error<1.
    print('Independent grain ODE relative percent',errors,flush=True)
