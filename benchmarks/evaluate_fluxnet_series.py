"""Evaluate complete CLM traces against explicit FLUXNET products and QC.

Instantaneous H/LE/NEE comparisons use each variable's own QC=0. GPP is a
partitioned product, not a direct measurement and has no invented QC. Daily
and monthly products require complete calendar periods and every reference
interval; no mean over missing intervals masquerades as a period integral.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta
import hashlib,json
from pathlib import Path
import numpy as np
from noahmp_clm.validation import compare_series

CARBON_G_PER_UMOL = 12.0107e-6
# Model carbon is interval-integrated g C/m2; reference carbon is umol CO2/m2/s.
PRODUCTS = {
    'H_F_MDS': ('h_wm2','h_wm2',False),
    'LE_F_MDS': ('le_wm2','le_wm2',False),
    'H_CORR': ('h_wm2','h_wm2',False),
    'LE_CORR': ('le_wm2','le_wm2',False),
    'GPP_NT_VUT_REF': ('gpp_gc_m2','gpp_gc_day',True),
    'GPP_DT_VUT_REF': ('gpp_gc_m2','gpp_gc_day',True),
    'NEE_VUT_REF': ('nee_gc_m2','nee_gc_day',True),
}
# FULLSET defines the original flux QC as the QC for its closure-corrected
# product too. Both products are reported; neither is chosen by model skill.
CORRECTED_QC = {'H_CORR': 'H_F_MDS', 'LE_CORR': 'LE_F_MDS'}


def evaluate(model,observed,*,utc_offset_hours,soil_operator=None,initial_soil=None):
    if not np.isfinite(utc_offset_hours) or not -12<=utc_offset_hours<=14:
        raise ValueError('Explicit fixed UTC offset required')
    starts=np.asarray(observed['time_start']);ends=np.asarray(observed['time_end'])
    if starts.ndim!=1 or not starts.size or ends.shape!=starts.shape:
        raise ValueError('Require nonempty matching one-dimensional times')
    for key in ('time_start','time_end'):
        if not np.array_equal(model[key],observed[key]):raise ValueError('Exact time alignment required')
    times=[datetime.strptime(str(x),'%Y%m%d%H%M') for x in starts]
    stops=[datetime.strptime(str(x),'%Y%m%d%H%M') for x in ends]
    dt=(stops[0]-times[0]).total_seconds()
    if dt not in (1800.,3600.) or any((b-a).total_seconds()!=dt for a,b in zip(times,stops)):
        raise ValueError('Only constant HH/HR intervals are supported')
    if any(b!=a for b,a in zip(stops[:-1],times[1:])):raise ValueError('Noncontiguous time axis')
    def vector(data,key,finite=False):
        array=np.asarray(data[key],float)
        if array.shape!=starts.shape:raise ValueError(f'Shape mismatch: {key}')
        if finite and not np.all(np.isfinite(array)):raise ValueError(f'Nonfinite model: {key}')
        return array
    prepared={};instant={}
    for field,(key,variable,carbon) in PRODUCTS.items():
        a=vector(model,key,True);b=vector(observed,'obs_'+field)
        qc=vector(observed,'qc_'+CORRECTED_QC.get(field,field))
        if carbon:a=a*86400/dt;b=b*CARBON_G_PER_UMOL*86400
        derived=field.startswith('GPP_')
        keep=np.isfinite(b) if derived else np.isfinite(b)&(qc==0)
        kind=('partitioned product; no independent QC supplied' if derived else
              'energy-balance-corrected product; underlying flux QC=0 intervals' if field in CORRECTED_QC else
              'own QC=0 intervals')
        instant[field]={'reference_kind':kind,
                        'qc_source':None if derived else CORRECTED_QC.get(field,field)+'_QC',
                        'metrics':compare_series(a,b,variable,mask=keep),
                        'available_reference_intervals':int(np.isfinite(b).sum()),'total_intervals':len(starts)}
        prepared[field]=(a,b,qc,variable,derived)
    soil_metadata=None
    if (soil_operator is None) != (initial_soil is None):
        raise ValueError('Soil evaluation requires both explicit geometry and initial state')
    if soil_operator is not None:
        from fluxnet_soil_operator import soil_products
        extra,soil_metadata=soil_products(model,observed,initial_soil,soil_operator)
        prepared.update(extra)
        for field,(a,b,qc,var,_) in extra.items():
            instant[field]={'reference_kind':'own QC=0; interval-mean state estimate at declared depth',
                'metrics':compare_series(a,b,var,mask=np.isfinite(b)&(qc==0)),
                'available_reference_intervals':int(np.isfinite(b).sum()),'total_intervals':len(starts)}
    groups={}
    for frequency in ('daily','monthly'):
        labels=np.array([x.strftime('%Y-%m-%d' if frequency=='daily' else '%Y-%m') for x in times])
        result={}
        for label in sorted(set(labels)):
            mask=labels==label;indices=np.flatnonzero(mask)
            begin=datetime.strptime(label,'%Y-%m-%d' if frequency=='daily' else '%Y-%m')
            end=begin+timedelta(days=1) if frequency=='daily' else (begin.replace(year=begin.year+1,month=1) if begin.month==12 else begin.replace(month=begin.month+1))
            complete=times[indices[0]]==begin and stops[indices[-1]]==end
            records={}
            for field,(a,b,qc,var,derived) in prepared.items():
                valid=np.isfinite(b[mask]);q=qc[mask]
                kind=('partitioned product' if derived else
                      'energy-balance-corrected product; observed and filled underlying flux retained' if field in CORRECTED_QC else
                      'gap-filled product; observed and filled intervals retained')
                record={'reference_kind':kind,
                        'qc_source':None if derived else CORRECTED_QC.get(field,field)+'_QC',
                        'expected_intervals':int((end-begin).total_seconds()/dt),'provided_intervals':len(indices),
                        'finite_reference_intervals':int(valid.sum()),'own_qc_counts':{
                            'zero':int(np.count_nonzero(q==0)),'nonzero':int(np.count_nonzero(np.isfinite(q)&(q!=0))),
                            'missing':int(np.count_nonzero(~np.isfinite(q)))},
                        'calendar_period_complete':complete}
                if complete and np.all(valid):
                    record['metrics']=compare_series(a[mask],b[mask],var)
                    # Explicit integrals, not the sum of unweighted mean fluxes.
                    if var not in ('tsoil_c','smc'):
                        factor=dt/86400 if var.endswith('_gc_day') else dt
                        record['model_integral']=float(np.sum(a[mask])*factor)
                        record['reference_integral']=float(np.sum(b[mask])*factor)
                        record['integral_units']='g C/m2' if carbon_variable(var) else 'J/m2'
                    else:
                        record['state_units']='deg C' if var=='tsoil_c' else 'm3/m3'
                        record['required_criterion']='mean bias; state correlation is descriptive'
                else:
                    record['metrics']={'status':'unmeasured','bias_pass':None,'correlation_pass':None,
                                       'reason':'partial calendar period' if not complete else 'missing reference intervals'}
                records[field]=record
            result[label]=records
        groups[frequency]=result
    return {'scope':'H/LE, partitioned carbon and optionally soil-state evaluation; not a complete site acceptance or spinup certificate',
            'time_convention':'local standard time, fixed UTC offset, no DST','utc_offset_hours':utc_offset_hours,
            'dt_seconds':dt,'instantaneous':instant,**groups,
            'aggregation_policy':'Complete calendar periods and finite reference at every interval; no infilling or partial-period extrapolation',
            'carbon_molar_mass_g_mol':12.0107,
            'soil_observation_operator':soil_metadata,
            'not_evaluated':['direct ET (LE is not an independent ET observation)','runoff','SWE']
                 + (['soil temperature/water depth mapping'] if soil_metadata is None else [])}


def carbon_variable(variable):
    return variable in ('gpp_gc_day','nee_gc_day')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','model-manifest','observations','observation-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--soil-operator',type=Path,help='Explicit sensor-depth and time-averaging JSON')
    args=parser.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    mm=json.loads(args.model_manifest.read_text(encoding='utf-8'));om=json.loads(args.observation_manifest.read_text(encoding='utf-8'))
    if mm.get('status')!='complete':raise ValueError('Only complete trajectories may be evaluated')
    if mm.get('source_unchanged') is not True:raise ValueError('Unchanged model/instrument source required')
    if om['prepared_npz_sha256']!=sha(args.observations) or mm['forcing_npz_sha256']!=sha(args.observations):
        raise ValueError('Observation/forcing source hash mismatch')
    if mm['model_npz_sha256']!=sha(args.model):raise ValueError('Model trace hash mismatch')
    if mm['utc_offset_hours']!=om['utc_offset_hours'] or mm['time_convention']!='local standard time':
        raise ValueError('Explicit matching time conventions required')
    soil_operator=initial_soil=None
    if args.soil_operator:
        from fluxnet_soil_operator import initial_soil_from_manifest
        soil_operator=json.loads(args.soil_operator.read_text(encoding='utf-8'))
        if soil_operator['site']!=mm['site']:raise ValueError('Soil operator/site mismatch')
        initial_soil=initial_soil_from_manifest(mm)
    with np.load(args.model,allow_pickle=False) as a,np.load(args.observations,allow_pickle=False) as b:
        if mm['steps_completed']!=mm['steps_requested'] or mm['steps_completed']!=len(a['time_start']):
            raise ValueError('Complete trajectory length disagrees with run manifest')
        result=evaluate(a,b,utc_offset_hours=om['utc_offset_hours'],soil_operator=soil_operator,initial_soil=initial_soil)
    result['source_sha256']={k:sha(p) for k,p in [('model',args.model),('observations',args.observations),
        ('model_manifest',args.model_manifest),('observation_manifest',args.observation_manifest),('instrument',Path(__file__))]}
    if args.soil_operator:
        import fluxnet_soil_operator as soil_module
        result['source_sha256'].update(soil_operator=sha(args.soil_operator),soil_instrument=sha(Path(soil_module.__file__)))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'daily_periods':len(result['daily']),'monthly_periods':len(result['monthly'])}))


if __name__=='__main__':main()
