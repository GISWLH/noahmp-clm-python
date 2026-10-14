"""Convert checksum-pinned public UCAR snow input data to NumPy package tables.

No physical coefficients are fitted or regenerated. The full original files
and their provenance are listed in docs/CLM_SNOW_DESIGN.md. Requires SciPy only
for the offline NetCDF reader; the installed physics uses NumPy alone.
"""
from pathlib import Path
import argparse
import hashlib
import numpy as np
from scipy.io import netcdf_file

FILES={
    'optics':('snicar_optics_5bnd_c090915.nc','a17f1b42616ce0b5c0534ccc0f92c33f0020db574bcc2888d326c7da6b69c305'),
    'aging':('snicar_drdt_bst_fit_60_c070416.nc','75aa2dbb5e770cca38915ed2ed4f9c31e076857fd56c701ced2fef6bdb78e5e6')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();arrays={}
    for group,(name,digest) in FILES.items():
        path=args.input/name
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=digest: raise ValueError(f'Unexpected UCAR input identity: {name}')
        with netcdf_file(path,mmap=False) as f:
            for key,value in f.variables.items():
                arrays[group+'_'+key]=np.array(value[:],dtype=float)
        arrays[group+'_source_filename']=name
        arrays[group+'_source_sha256']=digest
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**arrays)
    print(args.output,hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__=='__main__': main()
