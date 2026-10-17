"""A-Ci fluorescence/gas-exchange consistency, with no fitting or light scoring.

Conditional estimates assume Ci represents chloroplast CO2, the existing Rd
and Gamma* laws, and that PSII electrons support carboxylation/photorespiration.
They are not direct measurements of absorptance or a new acceptance gate.
"""
from pathlib import Path
from dataclasses import asdict
import argparse,csv,gzip,hashlib,io,json,zipfile
import numpy as np
from noahmp_clm.physics.photosynthesis import LeafParameters,farquhar_rates,R_GAS,T_REFERENCE
import noahmp_clm.physics.photosynthesis as model

SHA='936cff54c647e84bb4fc59e380c6d234888f8e47722635b00d24e55068c83cd5'

def diagnose(archive,baseline):
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==SHA
    old=json.loads(gzip.decompress(baseline.read_bytes()))['multistart']['files'];by_name={x['file']:x for x in old};files=[]
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            raw=list(csv.DictReader(io.StringIO(z.read(name).decode('cp1252'))))[1:]
            # Every derived data row below is A-Ci; no light-response rows enter.
            rows=[r for r in raw if 11<=int(r['DataType'])<=25]
            data=np.asarray([[float(r[k]) for k in ['CO2i','PARi','Tleaf','AnetCO2','FmorFm.','FoorFs']] for r in rows])
            assert len(data)>0 and np.all(np.isfinite(data))
            ci,par,temp,observed,fm,fs=data.T;temp=temp+273.15
            p=LeafParameters(**by_name[name]['parameters']);rates=farquhar_rates(ci,.84*par,temp,p)
            np.testing.assert_allclose(np.sqrt(np.mean((rates['net']-observed)**2)),by_name[name]['calibration_rmse'],rtol=1e-10,atol=1e-10)
            gamma=42.75*np.exp(37830./(R_GAS*T_REFERENCE)*(1.-T_REFERENCE/temp))
            valid=(fm>0.)&(fs>=0.)&(fs<fm)&(par>0.)
            phi=np.divide(fm-fs,fm,out=np.zeros_like(fm),where=fm>0.)
            # The Ci>=2Gamma condition limits conditioning of the inverse,
            # not the input data or original calibration/validation masks.
            conditioned=ci>=2.*gamma
            light_limited=(rates['rubp']<=rates['rubisco'])&(rates['rubp']<=rates['tpu'])
            usable=valid&conditioned&light_limited&(observed+rates['respiration']>0.)
            gross=observed+rates['respiration']
            inferred=np.divide(gross*(4.*ci+8.*gamma),ci-gamma,out=np.zeros_like(ci),where=conditioned)
            photons=phi*par;ratios=np.divide(inferred,photons,out=np.zeros_like(ci),where=valid&conditioned)
            selected=np.flatnonzero(usable)
            inferred_product=None
            if len(selected):
                q=photons[usable];j=inferred[usable];inferred_product=float(np.dot(q,j)/np.dot(q,q))
            entries=[]
            for k,row in enumerate(rows):
                entries.append(dict(time=row['ObsDate']+'T'+row['HHMMSS'],ci_ppm=float(ci[k]),par_incident=float(par[k]),
                    observed_net=float(observed[k]),measured_psii_operating_efficiency=float(phi[k]) if valid[k] else None,
                    modeled_potential_electron_transport=float(rates['electron_transport'][k]),
                    fluorescence_electron_transport_under_original_optical_assumption=float(.42*photons[k]) if valid[k] else None,
                    assimilation_inferred_electron_demand=float(inferred[k]) if conditioned[k] else None,
                    conditioned_inverse=bool(conditioned[k]),modeled_rubp_limited=bool(light_limited[k]),selected_for_product_diagnostic=bool(usable[k]),
                    inferred_absorptance_times_psii_fraction=float(ratios[k]) if usable[k] else None))
            statistics=dict(selected_n=len(selected),assumed_absorptance_times_psii_fraction=.42,inferred_product=inferred_product)
            if len(selected):
                values=ratios[usable]
                statistics.update(product_min=float(values.min()),product_max=float(values.max()),product_mean=float(values.mean()),
                    product_coefficient_of_variation=float(values.std()/values.mean()),equivalent_absorptance_if_psii_fraction_half=2.*inferred_product,
                    equivalent_absorptance_in_physical_interval=bool(0.<2.*inferred_product<=1.))
            files.append(dict(file=name,calibration_n=len(data),parameters_from_old_aci_fit=asdict(p),summary=statistics,rows=entries))
    return dict(scope='A-Ci-only observational consistency diagnostic, not fitting, a light-response score, a direct absorptance observation, or proof of cause.',
        original_archive_sha256=SHA,baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        instrument_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_sha256=hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
        assumptions=['Measured Ci approximates chloroplast CO2; no mesophyll gradient inferred.',
            'Existing modeled daytime respiration and Gamma* used in the gas-exchange inversion.',
            'No extra electron sinks inferred; no direct leaf optical measurement available.',
            'Only original A-Ci-model RuBP-limited rows with Ci>=2Gamma*, positive gross uptake and valid fluorescence enter the product summary; all A-Ci rows retained.',
            'Original A-Ci/held-out light calibration and gates are unchanged.'],
        references=['https://escomp.github.io/CTSM/release-clm5.0/tech_note/Photosynthesis/CLM50_Tech_Note_Photosynthesis.html',
            'https://onlinelibrary.wiley.com/doi/10.1111/j.1365-3040.2007.01680.x',
            'https://www.leafweb.org/media/1086/leafweb_december_2022_release_user_guide_20221228.pdf'],files=files)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior output')
    result=diagnose(a.archive,a.baseline);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps([dict(file=v['file'],**v['summary']) for v in result['files']]),flush=True)
