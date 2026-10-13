"""J2 diagnostic: identical float32 inputs for CUDA/JAX primitive comparisons.

This is a probe, not a replacement for the full-column parity gates. It compiles
the production CUDA source plus a read-only test entry point and reports all
component differences without masking residuals or asserting looser tolerances.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noahmp_clm.column import ColumnState, Forcing, default_soil_params
from noahmp_clm.gpu.pack import column_state_to_vec, forcing_to_vec


PROBE_SOURCE = r'''
extern "C" __global__ void probe_components(
    const float* states, const float* forcings, const float* temperatures,
    float* output, int n) {
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if (col>=n) return;
    float s[70], u[8];
    for(int j=0;j<70;++j) s[j]=states[col*70+j];
    for(int j=0;j<8;++j) u[j]=forcings[col*8+j];
    float t=temperatures[col], transp=17.0f;
    float* out=output+col*39;
    out[0]=esat(t); out[1]=qsat(t,u[4]); out[2]=air_density(u);
    out[3]=exchange(u,s[33],t); out[4]=ground_resistance(s);
    out[5]=beta(s[4],s); out[6]=stomata(s,u);
    out[7]=penman(123.0f,u,100.0f,150.0f);
    out[8]=liquid(s[4],s[0],s); out[9]=capacity(s[4],s[0],s);
    out[10]=soil_k(s[4],s); out[11]=snow_k(s[22]);
    Energy e=energy(t,s,u,false,transp);
    out[12]=e.rn; out[13]=e.h; out[14]=e.le; out[15]=e.g;
    out[16]=e.ecan; out[17]=e.esoil; out[18]=e.residual;
    Energy sn=energy(t,s,u,true,transp);
    out[19]=sn.rn; out[20]=sn.h; out[21]=sn.g; out[22]=sn.residual;
    out[23]=surface_temperature(s,u,false,transp);
    out[24]=surface_temperature(s,u,true,transp);
    float a[4]={0,-2,-3,-4}, b[4]={50,60,70,80}, c[4]={-2,-3,-4,0};
    float rhs[4]={s[0]*48,s[1]*55,s[2]*63,s[3]*76}, sol[4];
    thomas4(a,b,c,rhs,sol);
    for(int j=0;j<4;++j) out[25+j]=sol[j];
    soil_heat(s,u[0],12.345f);
    for(int j=0;j<4;++j) { out[29+j]=s[j]; out[33+j]=s[8+j]; }
    float t2=t*t;
    out[37]=SIGMA*t2*t2; out[38]=SIGMA*(t2*t2);
}
'''

NAMES = ("esat", "qsat", "rho", "exchange", "ground_resistance", "beta", "stomata",
         "penman", "liquid", "capacity", "soil_k", "snow_k", "bare_rn", "bare_h",
         "bare_le", "bare_g", "bare_ecan", "bare_esoil", "bare_residual",
         "snow_rn", "snow_h", "snow_g", "snow_residual", "bare_root", "snow_root",
         *[f"thomas_{k}" for k in range(4)], *[f"soil_t_{k}" for k in range(4)],
         *[f"soil_ice_{k}" for k in range(4)], "blackbody_left", "blackbody_pow4")


def inputs(n):
    states, forcings, temperatures = [], [], []
    for index in range(n):
        kind = index % 4
        state = ColumnState(soil=default_soil_params(("sand", "loam", "clay")[index % 3]),
                            lai=1.0 + index % 13 / 4,
                            tsoil=np.full(4, 268.0 if kind == 0 else 285.0),
                            swe=0.2 if kind == 0 else (0.008 if kind == 1 else 0.0),
                            snow_t=268.0 if kind == 0 else 273.15,
                            canw=0.0004 if kind == 2 else 0.0)
        tair, sw, lw, prcp = ((268., 100., 280., 1e-5), (295., 500., 350., 0.),
                            (285., 200., 330., 2e-5), (280., 300., 320., 0.))[kind]
        states.append(column_state_to_vec(state)[0])
        forcings.append(forcing_to_vec(Forcing(1800., tair, 0.004, 2., 1e5, sw, lw, prcp)))
        temperatures.append(tair + (index % 17 - 8) * 0.17)
    return (np.asarray(states, np.float32), np.asarray(forcings, np.float32),
            np.asarray(temperatures, np.float32))


def jax_components(s, u, t):
    import jax.numpy as jnp
    from noahmp_clm.gpu import jax_core as jc
    one = jnp.asarray(1, jnp.int32)
    option4 = jnp.asarray(4, jnp.int32)
    es, qs = jc._sat_vapor_pressure(t), jc._qsat(t, u[4])
    rho = jc._air_density(u[1], u[2], u[4])
    ce = jc._most_exchange_velocity(u[3], u[1], u[2], u[4], s[33], t)
    gr = jc._ground_resistance(s[4], jnp.float32(0.1), s[39], s[40], s[43], option4)
    _, beta = jc._soil_stress(s[4:8], s[12:16], s[39], s[40], s[41], s[43], one)
    rs = jc._stomatal_resistance(s[4:8], s[12:16], s[36], s[31], u[5], u[1], u[2], u[4],
                                s[39], s[40], s[41], s[43], one, one)
    pm = jc._penman_monteith(jnp.float32(123), u[1], u[2], u[4], jnp.float32(100), jnp.float32(150))
    liquid = jc._unfrozen_water(s[4], s[0], s[39], s[43], one)
    _, cap = jc._phase_and_capacity(s[4], s[0], s[39], s[43], one)
    kt = jc._soil_conductivity(s[4:5], s[39], s[44])[0]
    sk = jc._snow_conductivity(s[22], s[19], one)

    def energy(ts, snow):
        emiss, albedo = (jnp.float32(.99), s[28]) if snow else (s[35], s[34])
        rn = (1-albedo)*u[5]+emiss*(u[6]-jc.SIGMA*ts**4)
        exchange = jc._most_exchange_velocity(u[3], u[1], u[2], u[4], s[33], ts)
        h = rho*jc.CP*exchange*(ts-u[1])
        if snow:
            g = jc._snow_surface_flux(s[16:19], s[19:22], s[22:25], one, ts)
            return jnp.stack((rn, h, 0., g, 0., 0., rn-h-g))
        dq = jc._qsat(ts,u[4])-u[2]
        ra = 1/jnp.maximum(exchange,1e-6)
        eg = jnp.minimum(beta*rho*jc.LV*jnp.maximum(0.,dq)/(ra+gr),s[4]*.1*jc.LV*1000/u[0])
        ec = jnp.minimum(rho*jc.LV*exchange*dq*jnp.minimum(1.,s[30]/jnp.maximum(1e-9,.2*(.0005*s[31]))),
                         s[30]*jc.LV*1000/u[0])
        g = kt*(ts-s[0])/.05
        return jnp.stack((rn,h,eg+ec+17.,g,ec/jc.LV*u[0]/1000,eg/jc.LV*u[0]/1000,
                          rn-h-eg-ec-17.-g))

    bare, snow = energy(t,False), energy(t,True)
    bare_root = jc._secant_solve(lambda ts: energy(ts,False)[6],u[1],u[1]+1.)
    snow_root = jc._secant_solve(lambda ts: energy(ts,True)[6],u[1],u[1]+1.)
    th = jc._thomas4(jnp.array([0,-2,-3,-4],jnp.float32),jnp.array([50,60,70,80],jnp.float32),
                     jnp.array([-2,-3,-4,0],jnp.float32),s[:4]*jnp.array([48,55,63,76],jnp.float32))
    soil_t, ice = jc._soil_heat(s[:4],s[4:8],s[8:12],u[0],jnp.float32(12.345),s[39],s[43],s[44],one)
    blackbody = jc.SIGMA*t**4
    return jnp.concatenate((jnp.stack((es,qs,rho,ce,gr,beta,rs,pm,liquid,cap,kt,sk)),bare,
                            snow[jnp.array([0,1,3,6])],jnp.stack((bare_root,snow_root)),
                            th,soil_t,ice,jnp.stack((blackbody,blackbody))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=156)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-npz")
    parser.add_argument("--mode",default="all")
    args = parser.parse_args()
    import cupy as cp
    import jax
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        raise RuntimeError("GPU required")
    f, u, t = inputs(args.n)
    if args.input_npz:
        data=np.load(args.input_npz);f,u,t=data["f"],data["u"],data["t"];args.n=len(f)
    reference = np.asarray(jax.jit(jax.vmap(jax_components))(f,u,t))
    source = Path("src/noahmp_clm/gpu/cuda/column_kernel.cu").read_text(encoding="utf-8")
    rows = []
    for mode, flags in (("default",()),("fmad_false",("--fmad=false",)),
                        ("fmad_false_precdiv_false",("--fmad=false","--prec-div=false"))):
        if args.mode not in ("all",mode):continue
        module = cp.RawModule(code=source+PROBE_SOURCE,options=("--std=c++14",*flags),
                              name_expressions=("probe_components",))
        actual = cp.empty((args.n,len(NAMES)),cp.float32)
        module.get_function("probe_components")(((args.n+127)//128,),(128,),
            (cp.asarray(f),cp.asarray(u),cp.asarray(t),actual,np.int32(args.n)))
        actual = cp.asnumpy(actual)
        for col,name in enumerate(NAMES):
            delta=abs(actual[:,col]-reference[:,col])
            worst=int(np.argmax(delta))
            row={"mode":mode,"component":name,"max_abs":float(delta[worst]),
                 "failed_1e6":int(np.sum(~np.isclose(actual[:,col],reference[:,col],rtol=1e-6,atol=1e-6))),
                 "not_equal":int(np.sum(actual[:,col]!=reference[:,col])),
                 "worst_index":worst,"cuda":float(actual[worst,col]),"jax":float(reference[worst,col])}
            print(json.dumps(row),flush=True)
            rows.append(row)
    Path(args.output).write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
