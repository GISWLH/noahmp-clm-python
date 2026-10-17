"""Independent analytic CLM5 Eq2.6.52 comparison; no fitting or trajectory claim."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from noahmp_clm.column import SoilParams
from noahmp_clm.physics.frozen_soil import unfrozen_water_content
from noahmp_clm.physics.clm_freezing import equilibrium_liquid_water

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve original evidence')
    textures=[('STAS1-sand',.339,2.79,.069),('STAS6-loam',.439,5.25,.355),('STAS9-clay-loam',.465,8.17,.263),('STAS12-clay',.468,11.55,.468)]
    records=[]
    for name,sat,b,psi in textures:
        t=np.concatenate([np.linspace(200.,273.14,401),[273.149,273.15,280.]])
        total,t=np.broadcast_arrays(sat*np.array([0.,.05,.2,.6,1.])[:,None],t[None,:])
        # Independent pressure-head form: positive suction in metres, without
        # the production smooth cap or its analytic derivative machinery.
        reference=np.minimum(total,sat*np.power(9.80616*np.minimum(t,273.15)*psi/
            (334000.*np.maximum(273.15-t,1e-12)),1./b))
        reference=np.where(t>=273.15,total,reference)
        old=unfrozen_water_content(total,t,SoilParams(smcmax=sat,bexp=b))
        new=equilibrium_liquid_water(total,t,sat,b,psi)['liquid']
        measured=reference>=1e-5
        error=float(np.max(abs(new[measured]-reference[measured])/reference[measured]))
        assert error<=.0103
        records.append(dict(texture=name,saturated_suction_m=psi,points=total.size,
            old_max_relative_error=float(np.max(abs(old[measured]-reference[measured])/reference[measured])),
            candidate_max_relative_error=error,old_max_absolute_error=float(np.max(abs(old-reference))),
            candidate_max_absolute_error=float(np.max(abs(new-reference)))))
    report=dict(scope='Independent NumPy component only; not yet integrated into extensive heat states or GPU backends. No model/year/observation acceptance claim.',
        reference='https://escomp.github.io/CTSM/release-clm5.0/tech_note/Soil_Snow_Temperatures/CLM50_Tech_Note_Soil_Snow_Temperatures.html#equation-6-52',
        constants=dict(freezing_temperature_k=273.15,fusion_jkg=334000.,gravity_ms2=9.80616),
        notes='Same project Tf/Lf used in independent analytic expression; no exact-constant identity with CTSM/Noah claimed. All404 temperatures and5 water levels retained per texture; no parameter fitting.',
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),Path('src/noahmp_clm/physics/clm_freezing.py')]},records=records)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(records),flush=True)

if __name__=='__main__':main()
