"""Conditional variable-J diagnostics from original A-Ci rows only.

This does not change production photosynthesis, fit capacities, or score/read
light-response values. Optical/electron calibration is not measured here;
every estimate is conditional on that assumption and modeled Rd/Gamma*.
"""
from pathlib import Path
import argparse,csv,hashlib,io,json,os,time,zipfile
import numpy as np
from scipy.optimize import brentq,minimize_scalar
from noahmp_clm.physics.photosynthesis import LeafParameters,farquhar_rates
import noahmp_clm.physics.photosynthesis as model

ARCHIVE_SHA='936cff54c647e84bb4fc59e380c6d234888f8e47722635b00d24e55068c83cd5'
REFERENCES=['https://doi.org/10.1104/pp.98.4.1429',
    'https://doi.org/10.1111/j.1365-3040.2009.02016.x',
    'https://doi.org/10.1093/jxb/err288']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def inverse(ci,a,rd,gamma,j):
    """Return every row, including invalid or negative conditional estimates."""
    b=a+rd;den=j-4*b
    if not (np.isfinite([ci,a,rd,gamma,j]).all() and j>0 and b>0 and den>0):
        return dict(status='outside_positive_electron_limited_inverse',cc_ppm=None,gm_mol_m2_s=None,
            inverse_gm=None,cc_sensitivity_to_a=None,gm_relative_j_condition=None,brent_cc_difference_ppm=None)
    cc=gamma*(j+8*b)/den;draw=ci-cc
    gm=None if draw==0 else a/draw
    # Independent root of the original stoichiometric equation, not its inverse.
    def f(c):return j*(c-gamma)/(4*c+8*gamma)-b
    high=max(2*gamma,ci,1.)
    for _ in range(64):
        if f(high)>0:break
        high*=2
    reference=brentq(f,gamma,high,xtol=1e-10,rtol=2e-14)
    error=abs(cc-reference);assert error<1e-6*max(1.,abs(cc))
    sensitivity=12*gamma*j/den**2
    condition=None if draw==0 else abs(12*gamma*b*j/(den**2*draw))
    return dict(status='positive_gm' if gm is not None and gm>0 else 'nonpositive_or_singular_gm',
        cc_ppm=float(cc),gm_mol_m2_s=None if gm is None else float(gm),
        inverse_gm=None if a==0 else float(draw/a),cc_sensitivity_to_a=float(sensitivity),
        gm_relative_j_condition=None if condition is None else float(condition),
        harley_condition_10_to_50=bool(10<sensitivity<50),brent_cc_difference_ppm=float(error))

def constant_inverse_conductance(ci,a,rd,gamma,j):
    """Fit all originally selected rows, including ones with negative point gm.

    Fits only a single inverse conductance against training fluorescence J;
    beta=0 is the included infinite-gm endpoint. No data row is dropped based
    on the resulting gm, and no capacity or optical factor is fitted.
    """
    b=a+rd
    if len(ci)==0 or np.any(a<=0) or np.any(ci<=gamma):return dict(status='unavailable')
    upper=float(np.min((ci-gamma)/a))*(1-1e-6)
    def predicted(beta):
        cc=ci-a*beta
        return 4*b*(cc+2*gamma)/(cc-gamma)
    def objective(beta):return float(np.sum((predicted(beta)-j)**2))
    fit=minimize_scalar(objective,bounds=(0.,upper),method='bounded',options=dict(xatol=1e-11,maxiter=250))
    assert fit.success
    options=[(objective(0.),0.),(objective(float(fit.x)),float(fit.x)),(objective(upper),upper)]
    sse,beta=min(options)
    return dict(status='infinite_gm_endpoint' if beta==0 else 'finite_conditional_fit',
        inverse_gm=float(beta),gm_mol_m2_s=None if beta==0 else float(1/beta),
        j_rmse=float(np.sqrt(sse/len(ci))),j_relative_rmse=float(np.sqrt(sse/len(ci))/np.mean(j)),
        selected_n=len(ci),at_upper_endpoint=bool(beta==upper))

def diagnose(archive,baseline,output):
    assert sha(archive)==ARCHIVE_SHA
    old=json.loads(baseline.read_text());assert old['original_archive_sha256']==ARCHIVE_SHA
    assert old['model_sha256']==sha(model.__file__)
    protocol=dict(scope='A-Ci-only conditional measurement diagnostic; no production changes, capacity fit or light-curve scoring.',
        archive_sha256=sha(archive),baseline_diagnostic_sha256=sha(baseline),model_sha256=sha(model.__file__),
        diagnostic_sha256=sha(__file__),references=REFERENCES,
        optical_electron_products=[.35,.385,.42,.455,.49],rd_factors=[.5,1.,1.5],gamma_factors=[.9,1.,1.1],
        base=dict(optical_electron_product=.42,rd_factor=1.,gamma_factor=1.),
        selection='Exactly prior A-Ci fluorescence diagnostic mask; all original A-Ci rows retained; no light-response rows processed.',
        units='Ci/Cc/Gamma are CO2 mole fractions in ppm; gm=A/(Ci-Cc) in mol m-2 s-1. No conversion to pressure-based bar-1 conductance.',
        assumptions=['Optical product is absorptance times PSII partition times unmeasured electron/fluorescence calibration; .42=.84*.5 is an assumption.',
            'No independent non-photorespiratory calibration or mesophyll conductance measurements available in these five files.',
            'Existing dark-derived daytime respiration and Gamma temperature law remain assumptions.',
            'Harley conditioning is reported as metadata only; it never removes original training observations or changes acceptance.'])
    assert not output.exists() and not output.with_suffix('.protocol.json').exists()
    output.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    begun=time.monotonic();files=[]
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())=={v['file'] for v in old['files']}
        by_name={v['file']:v for v in old['files']}
        for name in sorted(z.namelist()):
            raw=list(csv.DictReader(io.StringIO(z.read(name).decode('cp1252'))))[1:]
            rows=[r for r in raw if 11<=int(r['DataType'])<=25]
            saved=by_name[name];assert len(rows)==saved['calibration_n']==len(saved['rows'])
            x=np.array([[float(r[k]) for k in ['CO2i','PARi','Tleaf','AnetCO2','FmorFm.','FoorFs']] for r in rows]);assert np.isfinite(x).all()
            ci,par,temp,a,fm,fs=x.T;temp=temp+273.15
            p=LeafParameters(**saved['parameters_from_old_aci_fit']);rates=farquhar_rates(ci,.84*par,temp,p)
            rd=rates['respiration'];gamma=42.75*np.exp(37830./(8.314462618*298.15)*(1-298.15/temp));phi=(fm-fs)/fm
            selected=np.array([r['selected_for_product_diagnostic'] for r in saved['rows']],bool)
            for k,(row,prior) in enumerate(zip(rows,saved['rows'])):
                assert row['ObsDate']+'T'+row['HHMMSS']==prior['time']
                assert ci[k]==prior['ci_ppm'] and par[k]==prior['par_incident'] and a[k]==prior['observed_net']
            scenarios=[];base_rows=None
            for product in protocol['optical_electron_products']:
                for rf in protocol['rd_factors']:
                    for gf in protocol['gamma_factors']:
                        result=[inverse(*v) for v in zip(ci,a,rd*rf,gamma*gf,product*phi*par)]
                        use=[v for v,k in zip(result,selected) if k];positive=[v['gm_mol_m2_s'] for v in use if v['status']=='positive_gm']
                        constant=constant_inverse_conductance(ci[selected],a[selected],rd[selected]*rf,gamma[selected]*gf,(product*phi*par)[selected])
                        scenarios.append(dict(optical_electron_product=product,rd_factor=rf,gamma_factor=gf,
                            selected_n=int(selected.sum()),positive_gm_n=len(positive),invalid_or_nonpositive_n=len(use)-len(positive),
                            positive_gm_min=None if not positive else min(positive),positive_gm_median=None if not positive else float(np.median(positive)),
                            positive_gm_max=None if not positive else max(positive),harley_conditioned_n=sum(v.get('harley_condition_10_to_50',False) for v in use),constant_fit=constant))
                        if product==.42 and rf==gf==1.:
                            base_rows=[dict(time=r['ObsDate']+'T'+r['HHMMSS'],ci_ppm=float(ci[k]),observed_net=float(a[k]),
                                rd=float(rd[k]),gamma=float(gamma[k]),fluorescence_efficiency=float(phi[k]),
                                prior_selected=bool(selected[k]),**v) for k,(r,v) in enumerate(zip(rows,result))]
            files.append(dict(file=name,original_aci_n=len(rows),base_rows=base_rows,scenarios=scenarios))
    result=dict(protocol=protocol,seconds=time.monotonic()-begun,node=os.uname().nodename,pbs_job=os.environ.get('PBS_JOBID'),files=files)
    assert sha(archive)==protocol['archive_sha256'] and sha(baseline)==protocol['baseline_diagnostic_sha256'] and sha(model.__file__)==protocol['model_sha256']
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(file=v['file'],original_aci_n=v['original_aci_n'],base=next(r for r in v['scenarios'] if r['optical_electron_product']==.42 and r['rd_factor']==r['gamma_factor']==1.)) for v in files]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    diagnose(args.archive,args.baseline,args.output)
