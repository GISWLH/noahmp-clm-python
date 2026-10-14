"""Preserve non-near-zero spectral absorption errors above the 10% gate."""
import importlib.util,json
from pathlib import Path
import numpy as np
from noahmp_clm.gpu.snicar import run_snow_shortwave_jax
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reference',root/'tests/test_gpu_snicar.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
data=ref.cases(np.float64,360);data=tuple(v.astype(np.float32) if v.dtype.kind=='f' else v for v in data)
out=run_snow_shortwave_jax(*data);failed=[]
for row in range(len(data[0])):
    expected=ref.reference(*(v[row] for v in data))['spectral_absorbed'];actual=np.asarray(out['spectral_absorbed'][row])
    bad=(np.abs(expected)>1.)&(np.abs(actual-expected)>.1*np.abs(expected))
    if bad.any():
        values=dict(row=row,cells=data[0][row].tolist(),count=int(data[1][row]),boundary=data[2][row].tolist(),
            expected=expected.tolist(),actual=actual.tolist(),failed_indices=np.argwhere(bad).tolist(),
            optics=np.asarray(out['optics'][row]).tolist())
        failed.append(values);print(json.dumps(values),flush=True)
Path('spectral-absorption-failures.json').write_text(json.dumps(failed,indent=2))
print('Failed columns',len(failed),flush=True)
