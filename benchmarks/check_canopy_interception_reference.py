"""CLM analytic interception reference and prescribed-heat annual budget.

Decimal evaluates the published bulk relations independently of the NumPy
implementation. The annual experiment checks component conservation; it is
not an observed canopy-flux or complete land-column validation.
"""
import argparse,hashlib,itertools,json,math,time
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
from noahmp_clm.physics.canopy_interception import CanopyCondensate,intercept_precipitation,exchange_vapor
from noahmp_clm.physics.snowpack import TF

ROOT=Path(__file__).resolve().parents[1]


def decimal_reference(kind,area,fv,amount,wind,dt):
    with localcontext() as context:
        context.prec=50
        a,v,m,u,t=map(lambda x:Decimal(str(x)),(area,fv,amount,wind,dt))
        tf=Decimal('273.15');temperature={'rain':Decimal('283.15'),'snow':Decimal('260.'),'unload':tf}[kind]
        if kind=='rain':
            ex=(2*a).exp();fraction=v*(ex-1)/(ex+1)
            old=Decimal('.05')*v*a;retained=min(Decimal('.1')*v*a,old+fraction*m)
            unloading=Decimal(0);specific=Decimal(334000)+Decimal(4180)*(temperature-tf)
        else:
            fraction=v*(1-(-a/2).exp());old=2*v*a
            retained=min(6*v*a,old+fraction*m)
            rate=u/Decimal('156000')+max(Decimal(0),temperature-Decimal('270.15'))/Decimal('187000')
            unloading=retained*min(Decimal(1),t*rate);retained-=unloading
            specific=Decimal(2100)*(temperature-tf)
        ground=m+old-retained
        return {key:float(value) for key,value in dict(mass=retained,enthalpy=retained*specific,
            ground_mass=ground,ground_enthalpy=ground*specific,intercepted=m*fraction,unloaded=unloading).items()}


def analytic():
    maximum={};cases=0;budget=np.zeros(2)
    for kind,area,fv,amount,wind,dt in itertools.product(('rain','snow','unload'),(.1,.5,3.,6.),(0.,.2,.8,1.),(.001,.1,3.,30.),(0.,3.,15.),(60.,1800.,3600.)):
        temperature={'rain':283.15,'snow':260.,'unload':TF}[kind]
        old_mass=(.05 if kind=='rain' else 2.)*fv*area
        state=CanopyCondensate.from_temperature(np.full(10,old_mass/10.),temperature)
        final,d=intercept_precipitation(state,amount if kind=='rain' else 0.,0. if kind=='rain' else amount,
            temperature,dt,np.full(10,area/10.),vegetated_fraction=fv,wind=wind)
        expected=decimal_reference(kind,area,fv,amount,wind,dt)
        actual=dict(mass=final.mass.sum(),enthalpy=final.enthalpy.sum(),
            ground_mass=d['ground_liquid_kgm2']+d['ground_ice_kgm2'],
            ground_enthalpy=d['ground_liquid_enthalpy_jm2']+d['ground_ice_enthalpy_jm2'],
            intercepted=d['intercepted_rain_kgm2']+d['intercepted_snow_kgm2'],unloaded=d['canopy_unloading_kgm2'].sum())
        for key,value in actual.items():
            error=abs(float(value)-expected[key]);floor=1. if 'enthalpy' in key else 1e-8;absolute=1e-5 if 'enthalpy' in key else 1e-10
            assert error<=(.1*abs(expected[key]) if abs(expected[key])>floor else absolute),(kind,key,value,expected[key])
            metrics=maximum.setdefault(key,dict(max_absolute=0.,max_relative_percent=0.))
            metrics['max_absolute']=max(metrics['max_absolute'],error)
            if abs(expected[key])>floor:metrics['max_relative_percent']=max(metrics['max_relative_percent'],100.*error/abs(expected[key]))
        budget=np.maximum(budget,[abs(d['water_residual_kgm2']),abs(d['energy_residual_jm2'])]);cases+=1
    return dict(cases=cases,decimal_precision=50,max_errors=maximum,max_budget=budget.tolist(),
        snow_unit_example=decimal_reference('snow',3.,.8,2.,3.,1800.))


def annual(steps):
    state=CanopyCondensate.empty();area=np.full(10,.3);fv=.8;records=[];max_step=np.zeros(2);max_cumulative=np.zeros(2)
    max_ice=max_liquid=0.
    for step in range(steps):
        old=state;day=step//48;hour=(step%48+.5)/48.;season=2.*np.pi*(day-90.)/365.
        air=TF+10.*np.sin(season)+4.*np.sin(2.*np.pi*hour)
        precipitation=.12 if day%5==0 and step%48<8 else 0.
        rain=precipitation if air>TF else 0.;snow=precipitation-rain
        state,d=intercept_precipitation(state,rain,snow,air,1800.,area,vegetated_fraction=fv,wind=3.)
        # Prescribed heating exercises freezing/melting without claiming an
        # actual leaf radiative-energy solution. Its entire Q is in the ledger.
        target=CanopyCondensate.from_temperature(state.mass,air+np.linspace(-2.,2.,10))
        heating=.05*(target.enthalpy-state.enthalpy);state=state.with_enthalpy(state.enthalpy+heating)
        request=np.full(10,.003*max(0.,np.sin(2.*np.pi*hour))-.001*max(0.,-np.sin(2.*np.pi*hour)))
        state,vapor=exchange_vapor(state,request,leaf_temperature=air+np.linspace(-2.,2.,10))
        ground_m=d['ground_liquid_kgm2']+d['ground_ice_kgm2'];ground_h=d['ground_liquid_enthalpy_jm2']+d['ground_ice_enthalpy_jm2']
        outgoing_m=float(vapor['mass_kgm2'].sum());outgoing_h=float(vapor['advected_enthalpy_jm2'].sum())
        row=[precipitation,d['precipitation_enthalpy_jm2'],ground_m,ground_h,float(heating.sum()),outgoing_m,outgoing_h]
        records.append(row)
        error=np.array([state.mass.sum()-old.mass.sum()-row[0]+row[2]+row[5],
            state.enthalpy.sum()-old.enthalpy.sum()-row[1]+row[3]-row[4]+row[6]])
        scale=np.array([abs(state.mass.sum())+abs(old.mass.sum())+row[0]+row[2]+abs(row[5]),
            abs(state.enthalpy).sum()+abs(old.enthalpy).sum()+sum(abs(row[k]) for k in (1,3,4,6))])
        assert np.all(abs(error)<=64.*np.finfo(float).eps*np.maximum(scale,1.)+np.array([1e-13,1e-7]))
        max_step=np.maximum(max_step,abs(error));max_ice=max(max_ice,float(state.ice.sum()));max_liquid=max(max_liquid,float(state.liquid.sum()))
        if step%48==47 or step==steps-1:
            sums=np.array([math.fsum(row[k] for row in records) for k in range(7)])
            residual=np.array([state.mass.sum()-sums[0]+sums[2]+sums[5],
                state.enthalpy.sum()-sums[1]+sums[3]-sums[4]+sums[6]])
            max_cumulative=np.maximum(max_cumulative,abs(residual))
    assert max_cumulative[0]<1e-9 and max_cumulative[1]<1e-5
    return dict(steps=steps,days=steps/48.,max_step_budget=max_step.tolist(),max_cumulative_budget=max_cumulative.tolist(),
        max_ice_kgm2=max_ice,max_liquid_kgm2=max_liquid,final_mass_kgm2=float(state.mass.sum()),final_enthalpy_jm2=float(state.enthalpy.sum()),
        totals=dict(zip(('precipitation_mass','precipitation_H','ground_mass','ground_H','supplied_heat','vapor_mass','vapor_advected_H'),sums.tolist())))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--steps',type=int,default=17520);args=parser.parse_args()
    start=time.monotonic();result=dict(analytic=analytic())
    if args.steps:result['annual']=annual(args.steps)
    result['sources']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
        'src/noahmp_clm/physics/canopy_interception.py','tests/test_canopy_interception.py','benchmarks/check_canopy_interception_reference.py')}
    result['references']=['https://escomp.github.io/CTSM/release-clm5.0/tech_note/Hydrology/CLM50_Tech_Note_Hydrology.html',
        'https://raw.githubusercontent.com/ESCOMP/CTSM/master/src/biogeophys/CanopyHydrologyMod.F90']
    result['seconds']=time.monotonic()-start;args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
