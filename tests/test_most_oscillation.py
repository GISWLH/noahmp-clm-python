"""Growing alternating MOST maps need lower gain, not weaker acceptance."""
from pathlib import Path
import math,sys
import numpy as np
import pytest
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_most import MOSTGeometry,solve_surface_most

NAMES=('friction_velocity','heat_velocity','water_velocity','leaf_velocity','ground_velocity')


def recorded_case(dtype=np.float32):
    with np.load(Path(__file__).parent/'fixtures/i4016_most_oscillation.npz',allow_pickle=False) as z:
        v={k:(x.astype(dtype) if x.dtype.kind=='f' else x.copy()) for k,x in z.items()}
    v['a4'][:,7:10]=v['forcing_environment']
    if dtype is np.float64:
        v['a6']/=v['a6'].sum(1,keepdims=True)
        p=v['a12'];p[:,:3]/=p[:,:3].sum(1,keepdims=True);p[:,9:12]/=p[:,9:12].sum(1,keepdims=True)
        plant,site=v['a11'],v['a4'];site[:,0]=plant[:,0]*p[:,19]/site[:,1]
        site[:,11]=np.divide(plant[:,3]*p[:,3],plant[:,0],out=np.zeros(len(plant)),where=plant[:,0]>0.).clip(0.,1.)
        v['a5'][:,:2]=v['a7'].sum(1)
    return tuple(v[f'a{k}'] for k in range(16)),v


@pytest.mark.parametrize('has_root',[True,False])
def test_numpy_analytic_alternation_or_no_root(monkeypatch,has_root):
    import noahmp_clm.physics.canopy_most as module
    def velocities(zeta,*args,**kwargs):return {key:math.exp(zeta) for key in NAMES}
    def feedback(*args):
        zeta=math.log(args[6]);target=-.25-3.*(zeta+.25) if has_root else zeta-.2
        return dict(zeta=target,raw_zeta=target,effective_wind=1.,buoyancy=0.,convective_wind=0.)
    monkeypatch.setattr(module,'canopy_exchange',velocities);monkeypatch.setattr(module,'flux_stability',feedback)
    def surface(exchange):return None,dict(h=0.,et_kgm2=0.,leaf_temperature=np.zeros(10))
    def run():return solve_surface_most(surface,CanopyExchange(280.,.005,100000.,1.,1.,1.,280.,1.),1.,MOSTGeometry(),time_step=1800.)
    if not has_root:
        with pytest.raises(RuntimeError,match='50 iterations'):run()
    else:
        result=run()[1]['most'];assert result['iterations']<50 and abs(result['used_zeta']+.25)<1e-12
        assert result['coefficient_residual']<=1e-4


@pytest.mark.parametrize('has_root',[True,False])
def test_native_schedule_analytic_map_without_gpu_claim(monkeypatch,has_root):
    monkeypatch.setitem(sys.modules,'cupy',np)
    import noahmp_clm.gpu.cuda.canopy_most as coefficients_module
    import noahmp_clm.gpu.cuda.canopy_phase_surface as surface_module
    from noahmp_clm.gpu.canopy_most_surface import run_most_surface
    def coefficients(geometry,atmosphere,state):
        zeta=state[:,0];target=np.float32(-.25)-np.float32(3.)*(zeta+np.float32(.25)) if has_root else zeta-np.float32(.2)
        return dict(**{key:np.exp(zeta) for key in NAMES},zeta=target,raw_zeta=target,
            effective_wind=np.ones(len(zeta),np.float32),potential_temperature=atmosphere[:,0],
            buoyancy=np.zeros(len(zeta),np.float32),convective_wind=np.zeros(len(zeta),np.float32))
    def surface(*args,**kwargs):return dict(h=np.zeros(1,np.float32),et_kgm2=np.zeros(1,np.float32),leaf_temperature=np.zeros((1,10),np.float32))
    monkeypatch.setattr(coefficients_module,'run_canopy_most_cuda',coefficients);monkeypatch.setattr(surface_module,'run_canopy_phase_surface_cuda',surface)
    arrays=[np.zeros((1,1),np.float32) for _ in range(9)]
    arrays[5]=np.array([[280.,.005,100000.]+[1.]*20],np.float32);arrays[7][:]=1800.
    def run():return run_most_surface(arrays,np.zeros((1,9),np.float32),np.ones(1,np.float32),.04,3.,backend='cuda')
    if not has_root:
        with pytest.raises(RuntimeError,match='50 iterations'):run()
    else:
        result=run()['most'];assert result['iterations'][0]<50 and abs(result['used_zeta'][0]+.25)<1e-6
        assert result['coefficient_residual'][0]<=1e-4


def test_recorded_i4016_numpy_whole_column_convergence_and_budgets():
    from test_gpu_clm_bgc_column import reference
    a,v=recorded_case();maximum=np.zeros(4)
    for k in range(4):
        row=[x[k] for x in a];out=reference(row,v['forcing'][k],v['deposition'][k],v['environment'][k],v['calendar'][k],v['nitrogen'][k],1.,most_geometry=MOSTGeometry(*map(float,v['geometry'][k])))
        d=out['diagnostics'];m=d['most'];b=d['biogeochemistry']
        assert m['iterations']<=50 and m['coefficient_residual']<=1e-4 and m['heat_change_wm2']<=.05 and m['vapor_change_kgm2s']<=2e-8 and m['leaf_temperature_change_k']<=.01
        assert abs(d['water_residual_kgm2'])<1e-8 and abs(d['energy_residual_jm2'])<.02
        assert abs(b['carbon_residual_gc_m2'])<=b['carbon_bound_gc_m2'] and abs(b['nitrogen_residual_gn_m2'])<=b['nitrogen_bound_gn_m2']
        maximum=np.maximum(maximum,abs(np.array([d['water_residual_kgm2'],d['energy_residual_jm2'],b['carbon_residual_gc_m2'],b['nitrogen_residual_gn_m2']])))
        print('I4016 NumPy',k,'H/LE',d['h'],d['le'],'iterations',m['iterations'],'coefficient residual',m['coefficient_residual'],flush=True)
    print('I4016 NumPy independent M/H/C/N maxima',maximum.tolist(),flush=True)
