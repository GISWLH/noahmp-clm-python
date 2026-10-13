"""J3 first-step transition probe; preserve full-graph diagnostics before tracing.

Run on an allocated GPU only. The instrumented JAX copy appends existing canopy
intermediates, leaving production files untouched. Its original 12 diagnostics
must match the original full 24-step graph bitwise before accepting components.
CUDA components evaluate the original input/preparation at the actual JAX Ts.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import types

import numpy as np

from check_gpu_options import scenarios
from noahmp_clm.gpu.pack import column_state_to_vec, forcing_to_vec
from noahmp_clm.options import PhysicsOptions


EXTRA_NAMES = ("f_temp_square", "rs", "tr_le", "rn_canopy", "rho", "ce_initial")
CUDA_NAMES = ("f_temp_left", "f_temp_square", "rs", "tr_le", "rn_canopy", "rho", "ce_initial",
              "ts", "rnet", "h", "le", "g", "ecan_m", "esoil_m", "residual")
CUDA_PROBE = r'''
extern "C" __global__ void probe_transition_components(
    const float* __restrict__ fs,const int* __restrict__ ints,const float* __restrict__ us,
    const float* __restrict__ ts_values,float* __restrict__ output,int n) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=n)return;
    float s[70],u[8];int opts[33];
    for(int j=0;j<70;++j)s[j]=fs[col*70+j];
    for(int j=0;j<8;++j)u[j]=us[col*8+j];
    for(int j=0;j<33;++j)opts[j]=ints[col*33+j];
    int count=opts[32];
    float total=fmaxf(0.f,u[7])*fmaxf(0.f,u[0]);
    float snow=total*snow_fraction(u[1],opts[1]),rain=total-snow;
    s[30]+=fminf(rain,fmaxf(0.f,.5e-3f*s[31]-s[30]));
    snowfall(s,&count,snow,u,opts[23]);
    float ce=exchange(u,s[33],u[1],opts[4]);
    float rs=stomata(s,u,opts[2],opts[5]),t2=u[1]*u[1];
    float rncan=s[32]*((1.f-s[34])*u[5]+s[35]*(u[6]-SIGMA*(t2*t2)));
    float tr=penman(fmaxf(0.f,rncan),u,1.f/fmaxf(ce,1e-6f),rs);
    tr*=1.f-fminf(1.f,s[26]/.02f);
    float available=0.f;for(int k=0;k<4;++k)available+=fmaxf(0.f,s[4+k]-s[40])*DZ[k];
    float tm=tr/LV*u[0]/RW;if(tm>available)tr*=available/fmaxf(tm,1e-12f);
    float ts=ts_values[col];
    Energy e=energy(ts,s,u,s[26]>1e-4f,tr,opts[4],opts[3],opts[2],opts[9]);
    float a=298.f-u[1];
    float values[15]={clip(1.f-.0016f*a*a,.05f,1.f),clip(1.f-.0016f*(a*a),.05f,1.f),
                     rs,tr,rncan,air_density(u),ce,ts,e.rn,e.h,e.le,e.g,e.ecan,e.esoil,e.residual};
    for(int k=0;k<15;++k)output[col*15+k]=values[k];
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    import cupy as cp
    import jax
    jax.config.update("jax_enable_x64", True)
    from noahmp_clm.gpu import jax_core as jc
    from noahmp_clm.gpu.cuda import loader
    assert jax.default_backend() == "gpu" and not cp.cuda.runtime.is_hip
    cases = scenarios(24, replace(PhysicsOptions(), OptRainSnowPartition=3))
    packed = [column_state_to_vec(s) for _, s, _ in cases]
    f = np.array([p[0] for p in packed], np.float32)
    i = np.array([p[1] for p in packed], np.int32)
    u = np.array([[forcing_to_vec(v) for v in fs] for _, _, fs in cases], np.float32)
    compiled = jc._BATCH_JIT.lower(f, i, u).compile()
    args.output.with_suffix(".production.hlo.txt").write_text(compiled.as_text(), encoding="utf-8")
    expected = tuple(np.asarray(v) for v in compiled(f, i, u))
    actual = tuple(cp.asnumpy(v) for v in loader.run_columns_cuda(f, i, u))

    path = Path(jc.__file__)
    original_source = path.read_text(encoding="utf-8")
    marker = "    updated = fvec\n"
    assert original_source.count(marker) == 1
    extra = """    diag = jnp.concatenate((diag, jnp.array([
        jnp.clip(1.0 - 0.0016 * (298.0 - tair)**2, 0.05, 1.0),
        rs, tr_le, rn_canopy, rho, ce_initial], dtype=fvec.dtype)))
"""
    instrumented = types.ModuleType("noahmp_clm.gpu.jax_option_residual_probe")
    instrumented.__package__ = "noahmp_clm.gpu"
    exec(compile(original_source.replace(marker, extra+marker), "<jax_option_residual_probe>", "exec"),
         instrumented.__dict__)
    expanded = tuple(np.asarray(v) for v in instrumented.run_columns_jax(f, i, u))
    unchanged = (np.array_equal(expanded[0], expected[0]) and np.array_equal(expanded[1], expected[1])
                 and np.array_equal(expanded[2][:, :, :12], expected[2]))

    ts = expected[2][:, 0, 0].copy()
    module = cp.RawModule(code=loader.KERNEL_SOURCE+CUDA_PROBE,
        options=("--std=c++14", "--fmad=false", "--prec-div=false"),
        name_expressions=("probe_transition_components",))
    components = cp.empty((len(f), 15), cp.float32)
    module.get_function("probe_transition_components")((1,), (32,),
        (cp.asarray(f), cp.asarray(i), cp.asarray(u[:, 0].copy()), cp.asarray(ts), components, np.int32(len(f))))
    components = cp.asnumpy(components)
    report = {
        "jax_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cuda_source_sha256_normalized": hashlib.sha256(loader.KERNEL_SOURCE.encode()).hexdigest(),
        "extra_outputs_preserved_all_original_state_and_diagnostics_bitwise": unchanged,
        "column": 6, "step": 0, "scenario": cases[6][0],
        "jax_diagnostics": expected[2][6, 0].tolist(), "cuda_diagnostics": actual[2][6, 0].tolist(),
        "jax_components": dict(zip(EXTRA_NAMES, expanded[2][6, 0, 12:].tolist())),
        "cuda_components_at_jax_ts": dict(zip(CUDA_NAMES, components[6].tolist())),
        "fixed_ts_matches_original_cuda": bool(actual[2][6, 0, 0] == ts[6]),
        "instrumentation_accepted": unchanged,
    }
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".npz"), f=f, i=i, u=u,
                        jax_diagnostics=expected[2], cuda_diagnostics=actual[2],
                        extra_jax_diagnostics=expanded[2], cuda_components=components)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
