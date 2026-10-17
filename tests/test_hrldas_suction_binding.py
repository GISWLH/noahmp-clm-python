"""Reject provenance/physical drift before comparing revised CLM with a reference."""
import copy,hashlib,json
import pytest
from test_hrldas_climate_prefix import inputs
from compare_hrldas_climate_prefix import compare_prefix
from compare_hrldas_suction_prefix import compare_suction_prefix

def case():
    trace,report,reference,prepared=inputs()
    old=dict(case='CN-Qia-initial-climate-cycle',scope='cold',parent_configuration_sha256='ancestor',
        column=dict(soil=dict(smcmax=.45),smc=[.28]*4),most_geometry=dict(wind_height=39.6),leaf_parameters=dict(vcmax25=60.))
    old_raw=json.dumps(old).encode();old_sha=hashlib.sha256(old_raw).hexdigest()
    new=copy.deepcopy(old);new.update(case='CN-Qia-suction-climate-cycle',scope='explicit suction',parent_configuration_sha256=old_sha,
        physics_revision=dict(law='CLM5Eq2.6.52 with explicit positive saturated suction head',saturated_suction_m=.355,
            old_checkpoints_reused=False,forcing_water_temperature_carbon_nitrogen_seed_policy_unchanged=True))
    new['column']['soil']['saturated_suction_m']=.355
    prepared['inputs_sha256'].update({'site-config.json':old_sha,'site.json':'metadata'})
    report.update(case=new['case'],forcing_manifest_sha256='metadata')
    return trace,report,reference,prepared,old_raw,new

def run(values):
    trace,report,reference,prepared,old_raw,new=values;new_raw=json.dumps(new).encode()
    report['configuration_sha256']=hashlib.sha256(new_raw).hexdigest()
    return compare_suction_prefix(trace,report,reference,prepared,old_raw,new_raw,'CN-Qia')

def test_derived_binding_keeps_original_manifest_and_incomplete_year():
    v=case();original=copy.deepcopy(v[3]);result=run(v)
    assert v[3]==original and result['original_model_steps_requested']==17520 and not result['scientific_acceptance_claim']
    assert result['diagnostic_all_daily_bias_criteria_met'] and result['diagnostic_flux_r_criteria_met']
    assert result['configuration_binding']['original_configuration_sha256']!=result['configuration_binding']['new_configuration_sha256']
    with pytest.raises(ValueError):compare_prefix(*v[:4])

@pytest.mark.parametrize('fault',['water','height','leaf','psi','parent','forcing','metadata','old_bytes','timezone','partial_reference'])
def test_changed_physics_or_identity_cannot_reuse_reference(fault):
    v=list(case());trace,report,reference,prepared,old_raw,new=v
    if fault=='water':new['column']['smc'][0]=.4
    elif fault=='height':new['most_geometry']['wind_height']=30.
    elif fault=='leaf':new['leaf_parameters']['vcmax25']=65.
    elif fault=='psi':new['column']['soil']['saturated_suction_m']=.5
    elif fault=='parent':new['parent_configuration_sha256']='unrelated'
    elif fault=='forcing':report['forcing_npz_sha256']='different'
    elif fault=='metadata':report['forcing_manifest_sha256']='different'
    elif fault=='old_bytes':v[4]=old_raw+b' '
    elif fault=='timezone':prepared['utc_offset_hours']=0
    else:reference['physical_intervals']=336
    with pytest.raises(ValueError):run(v)
