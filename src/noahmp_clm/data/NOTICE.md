# Snow lookup data

`snow_tables.npz` contains an exact numeric conversion of UCAR CLM/SNICAR input
tables, not project-authored physical coefficients or Fortran source code.
Original files are publicly served at
https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/lnd/clm2/snicardata/.

- `snicar_optics_5bnd_c090915.nc` (SNICAR optical properties; Mark Flanner).
- `snicar_drdt_bst_fit_60_c070416.nc` (Flanner and Zender snow-aging lookup).

Source filenames and SHA256 identities are embedded in the archive. Rebuild
with `benchmarks/prepare_snow_tables.py`; see `docs/CLM_SNOW_DESIGN.md` for source
checksums, metadata, scientific references and model simplifications. The
project's code license does not relicense these external scientific data.
