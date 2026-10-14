"""Independent simultaneous equations for the layered canopy phase solve."""
from dataclasses import replace
import numpy as np
import pytest
from noahmp_clm.physics.canopy_energy import CanopyExchange
from noahmp_clm.physics.canopy_phase_energy import solve_canopy_phase_energy
from noahmp_clm.physics.canopy_interception import CanopyCondensate
from noahmp_clm.physics.canopy_radiation import canopy_shortwave,canopy_longwave
from noahmp_clm.physics.photosynthesis import leaf_gas_exchange
from noahmp_clm.physics.soil_uptake import SoilWaterSupply


def cases():
    area=np.full(10,.3);supply=SoilWaterSupply(np.full(4,1e-5),np.full(4,.25),1.5e-5)
    rows=[]
    for name,mass,old_t,fraction,air,humidity,sw,lw in (
            ('liquid',.1,285.,1.,290.,.008,500.,350.),
            ('ice',.2,260.,0.,263.,.001,50.,210.),
            ('melt',.1,270.,0.,280.,.004,300.,300.),
            ('freeze',.02,275.,1.,265.,.001,0.,220.),
            ('dry',0.,295.,1.,295.,.01,600.,350.),
            ('mixture',.2,273.15,.5,273.15,.0035,80.,310.),
            ('thin',1e-5,280.,1.,295.,.003,600.,320.),
            ('dew',0.,280.,1.,280.,.009,0.,260.),
            ('frost',0.,265.,0.,265.,.004,0.,200.)):
        c=CanopyCondensate.from_temperature(np.full(10,mass),old_t,liquid_fraction=fraction)
        e=CanopyExchange(air,humidity,101325.,.02,.003,.03,air-1.,15.)
        kw=dict(soil_supply=supply,root_enthalpy=np.full(4,334000.+4180.*(air-273.15)))
        rows.append((name,area.copy(),sw,lw,e,c,kw))
    name,a,sw,lw,e,c,kw=rows[0]
    rows.append(('stems',a*.7,sw,lw,e,c,{**kw,'stem_area':a*.3,'vegetated_fraction':.65}))
    rows.append(('bare',a,sw,lw,e,CanopyCondensate.empty(),{**kw,'vegetated_fraction':0.}))
    a=np.array([.3,0.,.3,0.,.3,0.,.3,0.,.3,0.]);c=CanopyCondensate.from_temperature(np.where(a>0.,.1,0.),285.)
    rows.append(('inactive_layers',a,sw,lw,e,c,kw))
    from noahmp_clm.physics.snowpack import Snowpack
    for tg in (250.,265.,273.15):
        a=np.full(10,.3);c=CanopyCondensate.from_temperature(np.full(10,.2),min(tg+1.,273.15))
        e=CanopyExchange(tg+2.,.001,101325.,.02,.003,.03,274.,15.)
        kw=dict(soil_supply=supply,root_enthalpy=np.full(4,334000.),fixed_ground_temperature=tg,
            ground_is_ice=True,ground_shares_soil=False,ground_evaporation_limit=1e-6,
            snowpack=Snowpack.from_ice([2.,4.,8.,16.,32.],tg),vegetated_fraction=.7)
        rows.append((f'snow_{tg}',a,350.,240.,e,c,kw))
    return rows


def independent_reference(area,swdown,lwdown,e,old,dt,**kw):
    """Independently assemble mass, phase, air and heat equations for SciPy.

    Reuse only tested radiation and biochemistry. In particular do not call
    production phase/energy/water/air helpers or use production iterates.
    """
    least_squares=pytest.importorskip('scipy.optimize').least_squares
    leaf=np.asarray(area);area=leaf+np.broadcast_to(kw.get('stem_area',0.),leaf.shape);n=len(area)
    fv=kw.get('vegetated_fraction',1.) if area.sum() else 0.;active=fv*area>0.
    sw=canopy_shortwave(area,swdown);mass=old.mass;old_h=old.enthalpy;coupled=None
    if kw.get('snowpack') is not None:
        from noahmp_clm.physics.canopy_snow_radiation import canopy_snow_shortwave
        coupled=canopy_snow_shortwave(area,kw['snowpack'],swdown,vegetated_fraction=fv);sw=coupled['canopy']
    supply=kw['soil_supply'];root_h=kw['root_enthalpy'];root_limit=supply.roots[supply.weights>0.].sum()
    rho=e.pressure/(287.*e.air_temperature*(1.+.61*e.specific_humidity));mol=e.pressure/(8.314462618*e.air_temperature)
    capacity=.1*fv*area;fw=np.minimum(1.,np.divide(mass,capacity,out=np.zeros(n),where=capacity>0.)**(2./3.))
    fd=(1.-fw)*np.divide(leaf,area,out=np.zeros(n),where=area>0.)
    def evaluate(x,details=False):
        heat=mass*4180.*x[:n]
        liq=np.minimum(mass,np.maximum(0.,heat/334000.));ice=mass-liq
        leaf_t=np.full(n,273.15)
        below=(heat<0.)&(mass>0.);above=(heat>334000.*mass)&(mass>0.)
        leaf_t[below]+=heat[below]/(2100.*mass[below]);leaf_t[above]+=(heat[above]-334000.*mass[above])/(4180.*mass[above])
        leaf_t[mass==0.]=273.15+x[:n][mass==0.]
        t=np.r_[leaf_t,x[n]];q=x[-1]*.001
        es=611.2*np.exp(17.67*(t-273.15)/(t-29.65));ql=.622*es/(e.pressure-.378*es)
        ei=np.exp(9.550426-5723.265/t+3.53068*np.log(t)-.00728332*t);qi=.622*ei/(e.pressure-.378*ei)
        fraction=np.divide(liq,mass,out=np.asarray(leaf_t>273.15,dtype=float),where=mass>0.)
        qs=qi[:-1]+fraction*(ql[:-1]-qi[:-1])
        gas=leaf_gas_exchange(sw['absorbed_par'],leaf_t,co2_ppm=e.co2_ppm,relative_humidity=q*e.pressure/(.622+.378*q)/es[:-1],
            boundary_conductance=e.leaf_velocity*mol,nitrogen_factor=1. if root_limit>0. else 0.)
        wanted=rho*fv*area*e.leaf_velocity*np.where(qs>q,fw,1.)*(qs-q)*dt
        loss=np.minimum(np.maximum(wanted,0.),mass);deposit=np.maximum(-wanted,0.)
        wet=(loss-deposit)/dt
        # Independent phase-specific parcel H and final extensive quantities.
        removal=np.divide(loss,mass,out=np.zeros(n),where=mass>0.)*heat
        deposit_h=deposit*((1.-fraction)*2100.*(leaf_t-273.15)+fraction*(334000.+4180.*(leaf_t-273.15)))
        final_m=mass-loss+deposit;final_h=heat-removal+deposit_h;adv=(removal-deposit_h)/dt
        tr=rho*fv*area*e.leaf_velocity*fd*gas['gs_mol']/(e.leaf_velocity*mol+gas['gs_mol'])*np.maximum(ql[:-1]-q,0.)
        if tr.sum()>root_limit:tr*=root_limit/tr.sum()
        rooted=supply.weights>0.;lo=0.;hi=np.max(supply.roots[rooted]/supply.weights[rooted],initial=0.)
        for _ in range(50):
            mid=(lo+hi)/2.;uptake=np.minimum(supply.roots,mid*supply.weights)
            if uptake.sum()<tr.sum():lo=mid
            else:hi=mid
        uptake=np.minimum(supply.roots,(lo+hi)/2.*supply.weights)
        qground=qi[-1] if kw.get('ground_is_ice',False) else ql[-1]
        ge=rho*np.array([fv*(qground-q)/(1./e.ground_velocity+e.ground_resistance),
            (1.-fv)*(qground-e.specific_humidity)/(1./e.atmosphere_velocity+e.ground_resistance)])
        pos=np.maximum(ge,0.);neg=np.minimum(ge,0.);limit=kw.get('ground_evaporation_limit',np.inf)
        if kw.get('ground_shares_soil',True):limit=min(limit,max(0.,supply.surface-uptake[0]))
        if pos.sum()>0.:pos*=min(1.,(limit-neg.sum())/pos.sum())
        ge=pos+neg
        weights=np.r_[area*e.leaf_velocity,e.ground_velocity]
        tc=e.air_temperature+weights@(t-e.air_temperature)/(weights.sum()+e.atmosphere_velocity) if fv else e.air_temperature
        sensible=rho*1004.*fv*weights*(t-tc);sensible[-1]+=rho*1004.*(1.-fv)*e.atmosphere_velocity*(t[-1]-e.air_temperature)
        longwave=canopy_longwave(area,leaf_t,lwdown,t[-1],ground_emissivity=.97)
        rn=np.r_[fv*(sw['absorbed']+longwave['net_absorbed']),fv*(sw['ground_absorbed']+longwave['ground_net'])+
            (1.-fv)*(swdown*.775+.97*(lwdown-5.670374419e-8*t[-1]**4))]
        if coupled is not None:rn[-1]=coupled['snow_absorbed'].sum()+coupled['soil_absorbed']+fv*longwave['ground_net']+(1.-fv)*.97*(lwdown-5.670374419e-8*t[-1]**4)
        latent=np.r_[(2834000.-334000.*fraction)*wet+2500000.*tr,(2834000. if kw.get('ground_is_ice',False) else 2500000.)*ge.sum()]
        root_mean=float(uptake@root_h/uptake.sum()) if uptake.sum()>0. else 0.
        extra=tr*(334000.+4180.*(leaf_t-273.15)-root_mean)
        g=e.soil_conductance*(t[-1]-e.soil_temperature)+e.soil_flux_offset
        residual=rn-sensible-latent;residual[:-1]-=(final_h-old_h)/dt+adv+extra
        if kw.get('fixed_ground_temperature') is not None:g=residual[-1]
        residual[-1]-=g
        if kw.get('fixed_ground_temperature') is not None:residual[-1]=kw['fixed_ground_temperature']-t[-1]
        residual[:-1]=np.where(active,residual[:-1],e.air_temperature-leaf_t)
        qres=(wet.sum()+tr.sum()+ge[0]-rho*fv*e.atmosphere_velocity*(q-e.specific_humidity))*2500000. if fv else (q-e.specific_humidity)*1e5
        result=np.r_[residual,qres]
        if details:return dict(residual=result,temperature=t,h=sensible.sum(),le=latent.sum(),g=g,gpp=fv*leaf@gas['gross'],mass=final_m,enthalpy=final_h,humidity=q)
        return result
    initial=np.r_[np.divide(old_h,mass*4180.,out=np.full(n,e.air_temperature-273.15),where=mass>0.),kw.get('fixed_ground_temperature',e.air_temperature),e.specific_humidity*1000.]
    lower=np.r_[np.where(mass>0.,(200.-273.15)*2100./4180.,200.-273.15),200.,0.]
    upper=np.r_[np.where(mass>0.,334000./4180.+350.-273.15,350.-273.15),350.,190.]
    ambient=initial.copy()
    ambient[:n]=np.where(mass>0.,2100./4180.*(e.air_temperature-273.15) if e.air_temperature<273.15
        else 334000./4180.+e.air_temperature-273.15,e.air_temperature-273.15)
    candidates=[]
    # A reflective trust region can stop on a phase boundary even with large
    # residual. Try the independent rectangular trust region, never seed with
    # production iterates and never accept the optimizer's status by itself.
    for method in ('trf','dogbox'):
        for guess in (initial,ambient):
            fitted=least_squares(evaluate,guess,method=method,bounds=(lower,upper),x_scale='jac',max_nfev=400,xtol=1e-12,ftol=1e-12,gtol=1e-10)
            candidates.append(evaluate(fitted.x,True))
            if np.max(abs(candidates[-1]['residual']))<1e-6:return candidates[-1]
    result=min(candidates,key=lambda r:np.max(abs(r['residual'])))
    assert np.max(abs(result['residual']))<1e-6,(fitted.message,result['residual'])
    return result


@pytest.mark.parametrize('name,area,sw,lw,e,c,kw',cases(),ids=[row[0] for row in cases()])
def test_phase_energy_independent_network(name,area,sw,lw,e,c,kw):
    got=solve_canopy_phase_energy(area,sw,lw,e,c,1800.,**kw)
    ref=independent_reference(area,sw,lw,e,c,1800.,**kw)
    np.testing.assert_allclose(np.r_[got['leaf_temperature'],got['ground_temperature']],ref['temperature'],rtol=0.,atol=1e-4)
    for key in ('h','le','gpp'):np.testing.assert_allclose(got[key],ref[key],rtol=.1,atol=.001)
    np.testing.assert_allclose(got['condensate'].mass,ref['mass'],rtol=.1,atol=1e-7)
    np.testing.assert_allclose(got['condensate'].enthalpy,ref['enthalpy'],rtol=.1,atol=.1)
    np.testing.assert_allclose(got['condensate'].mass+1800.*got['wet_evaporation'],c.mass,rtol=0.,atol=1e-14)
    assert np.all(got['root_uptake']<=kw['soil_supply'].roots+1e-15)
    if name=='melt':assert np.all(got['condensate'].ice==0.)
    if name=='freeze':assert np.all(got['condensate'].liquid==0.)
    if name=='mixture':assert np.any((got['condensate'].ice>0.)&(got['condensate'].liquid>0.))
    assert np.max(abs(got['layer_residual']))<5e-5
    assert abs(got['air_water_residual'])<2.1e-11
    assert abs(got['air_heat_residual'])<1e-9
    total=sw+lw-got['outgoing_shortwave']-got['outgoing_longwave']-got['h']-got['le']-got['ground_heat']
    total-=got['canopy_storage_change_wm2']+got['wet_advected_heat_wm2']+got['transpiration_advected_heat_wm2']-got['root_advected_heat_wm2']
    assert got['transpiration_advected_heat_wm2']==pytest.approx(float(got['transpiration']@(334000.+4180.*(got['leaf_temperature']-273.15))),abs=1e-12)
    assert got['root_advected_heat_wm2']==pytest.approx(float(got['root_uptake']@kw['root_enthalpy']),abs=1e-12)
    assert got['transpiration_net_heat_wm2']==pytest.approx(got['transpiration_advected_heat_wm2']-got['root_advected_heat_wm2'],abs=1e-12)
    assert abs(total)<6e-4
    print(name,dict(iterations=got['iterations'],temperature_error=float(np.max(abs(np.r_[got['leaf_temperature'],got['ground_temperature']]-ref['temperature']))),energy_residual=float(total)))


def test_phase_energy_rejects_unowned_root_heat_and_inactive_water():
    _,a,sw,lw,e,c,kw=cases()[0]
    with pytest.raises(ValueError):solve_canopy_phase_energy(a,sw,lw,e,c,1800.)
    with pytest.raises(ValueError):solve_canopy_phase_energy(a,sw,lw,e,c,1800.,vegetated_fraction=0.,**kw)


def test_plateau_vapor_transfers_both_phases_with_exact_total_heat():
    from noahmp_clm.physics.canopy_phase_energy import _equilibrium_vapor
    old=CanopyCondensate.from_temperature([2.],273.15,liquid_fraction=.5)
    final,p=_equilibrium_vapor(old,np.array([.4]),np.array([273.15]))
    np.testing.assert_allclose([p['ice_kgm2'][0],p['liquid_kgm2'][0]],[.2,.2],rtol=0.,atol=1e-15)
    assert p['latent_energy_jm2'][0]==pytest.approx(1066800.,rel=1e-14)
    assert p['advected_enthalpy_jm2'][0]==pytest.approx(66800.,rel=1e-14)
    assert final.enthalpy[0]==pytest.approx(267200.,rel=1e-14)
    assert final.temperature[0]==273.15


def test_complete_evaporation_releases_the_redundant_plateau_coordinate():
    # Saved continuous cold-canopy step 14: the fourth layer loses all water
    # while its trial heat coordinate is on the melting plateau.
    mass=[0.,0.,0.,.00647506555267528,.01710930969501606,.027346464531692684,.03612239454314759,.04392351269441901,.05119005342416928,.05834543445506625]
    heat=[0.,0.,0.,-5.391749907703349,-34.08079016085131,-70.8682358081651,-112.36370366109665,-157.03485902221186,-205.2610373001825,-259.00956965442833]
    c=CanopyCondensate(mass,heat);e=CanopyExchange(266.7877205179804,.001486775896997012,101325.,.02,.003,.03,263.,15.)
    s=SoilWaterSupply(np.full(4,1e-5),np.full(4,.25),1.5e-5)
    r=solve_canopy_phase_energy(np.full(10,.3),473.46506474755284,258.53395590685693,e,c,1800.,
        vegetated_fraction=.8,soil_supply=s,root_enthalpy=np.full(4,334000.+4180.*(263.-273.15)),
        fixed_ground_temperature=263.,ground_is_ice=True,ground_shares_soil=False,ground_evaporation_limit=1e-5)
    assert np.max(abs(r['layer_residual']))<5e-5
    assert r['condensate'].mass[3]==0. and r['condensate'].enthalpy[3]==0.
    assert r['leaf_temperature'][3]<273.15
