// Default PhysicsOptions; float32 SoA ABI from STATE_SPEC.md.
// Corresponding reference: driver.step_column. CUDA parity is a separate gate.
#ifndef __CUDACC_RTC__
#include <math.h>
#endif

namespace {
constexpr float TF = 273.15f, SIGMA = 5.670374419e-8f;
constexpr float LV = 2.5e6f, LF = 3.34e5f, RW = 1000.0f, CP = 1004.0f;
__device__ __constant__ float DZ[4] = {0.1f, 0.3f, 0.6f, 1.0f};

// 对应普通版: energy.secant_solve。安全的标量界限。
__device__ __forceinline__ float clip(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}
// 对应普通版: soil._thomas。分母保护。
__device__ __forceinline__ float safe_den(float x) {
    return fabsf(x) < 1.0e-12f ? copysignf(1.0e-12f, x + 1.0e-20f) : x;
}
// 对应普通版: energy.sat_vapor_pressure。
__device__ __forceinline__ float esat(float t) {
    float tc = t - TF;
    return 611.2f * expf(17.67f * tc / (tc + 243.5f));
}
// 对应普通版: energy.qsat。
__device__ __forceinline__ float qsat(float t, float ps) {
    float es = esat(t);
    return 0.622f * es / fmaxf(1.0f, ps - 0.378f * es);
}
// 对应普通版: energy.air_density。
__device__ __forceinline__ float air_density(const float* u) {
    return u[4] / (287.0f * u[1] * (1.0f + 0.61f * u[2]));
}
// 对应普通版: surface_drag._psi_m/_psi_h。
__device__ void stability(float zeta, float* pm, float* ph) {
    if (zeta >= 0.0f) { *pm = *ph = -5.0f * zeta; return; }
    float x = powf(1.0f - 16.0f * zeta, 0.25f);
    *pm = 2.0f * logf((1.0f + x) / 2.0f)
        + logf((1.0f + x*x) / 2.0f) - 2.0f * atanf(x) + 1.5707963267948966f;
    *ph = 2.0f * logf((1.0f + x*x) / 2.0f);
}
// 对应普通版: surface_drag.most_exchange_velocity。固定20次，收敛后冻结。
__device__ float exchange(const float* u, float z0, float ts) {
    float wind = fmaxf(u[3], 0.3f), rho = air_density(u);
    z0 = clip(z0, 1.0e-5f, 5.0f);
    float lm = logf(10.0f / z0), lh = logf(10.0f / fmaxf(z0/10.0f, 1.0e-6f));
    float ce = 0.16f * wind / (lm*lh), ob = 1.0e30f;
    bool done = false;
    for (int it = 0; it < 20; ++it) {
        if (!done) {
            bool finite = fabsf(ob) < 1.0e29f;
            float pm, ph;
            stability(finite ? 10.0f/ob : 0.0f, &pm, &ph);
            float dm = fmaxf(1.0f, lm-pm), dh = fmaxf(1.0f, lh-ph);
            float us = 0.4f * wind / dm;
            float cn = fmaxf(1.0e-6f, 0.16f * wind / (dm*dh));
            float h = rho * CP * cn * (ts-u[1]);
            float on = fabsf(h) < 1.0e-8f ? 1.0e30f
                : clip(-rho*CP*u[1]*us*us*us/(0.4f*9.81f*h), -1.0e6f, 1.0e6f);
            bool next_finite = fabsf(on) < 1.0e29f;
            done = (finite && next_finite && fabsf(on-ob)/fmaxf(1.0f,fabsf(ob)) < 1.0e-4f)
                || (!finite && !next_finite);
            ce = cn; ob = on;
        }
    }
    return ce;
}
// 对应普通版: frozen_soil._niu_yang_liquid。Niu & Yang (2006)。
__device__ float liquid(float smc, float t, const float* s) {
    float total = fmaxf(0.0f, smc);
    if (t >= TF) return total;
    float clausius = LF/461.5f * fmaxf(0.0f,TF-t)/(TF*TF);
    float potential = s[39] * powf(fmaxf(clausius,1.0e-8f)/5.0e-5f, -1.0f/fmaxf(s[43],1.0e-6f));
    return fminf(total, potential);
}
// 对应普通版: frozen_soil.apparent_heat_capacity。
__device__ float capacity(float smc, float t, const float* s) {
    float derivative = (liquid(smc,t+0.02f,s)-liquid(smc,t-0.02f,s))/0.04f;
    return smc*4.18e6f+(1.0f-s[39])*2.0e6f+RW*LF*fmaxf(0.0f,derivative);
}
// 对应普通版: soil.thermal_conductivity。
__device__ float soil_k(float smc, const float* s) {
    float sr = clip(smc/s[39],0.0f,1.0f);
    float ke = sr > 0.1f ? clip(log10f(fmaxf(sr,1.0e-6f))+1.0f,0.0f,1.0f) : 0.0f;
    return 0.2f+(0.6f+1.6f*s[44]-0.2f)*ke;
}
// 对应普通版: stress.soil_water_stress。默认Noah水胁迫。
__device__ float beta(float smc, const float* s) {
    return clip((smc-s[40])/fmaxf(1.0e-9f,s[41]-s[40]),0.0f,1.0f);
}
// 对应普通版: canopy.ball_berry_rs。默认经验光响应Ball--Berry。
__device__ float stomata(const float* s, const float* u) {
    float water = 0.0f;
    for (int k=0;k<4;++k) water += s[12+k]*beta(s[4+k],s);
    water = fmaxf(0.02f,clip(water,0.0f,1.0f));
    float temp = clip(1.0f-0.0016f*(298.0f-u[1])*(298.0f-u[1]),0.05f,1.0f);
    float a = 25.0f*(1.0f-expf(-fmaxf(0.0f,u[5])*2.0f/450.0f))*temp*water;
    float ea = u[2]*u[4]/(0.622f+0.378f*u[2]);
    float rh = clip(ea/fmaxf(esat(u[1]),1.0f),0.05f,1.0f);
    float gs = (0.01f+9.0f*a*rh/400.0f)*8.314462618f*u[1]/fmaxf(u[4],1.0f);
    return clip(1.0f/fmaxf(gs,1.0e-6f),s[36],5000.0f);
}
// 对应普通版: canopy.penman_monteith。
__device__ float penman(float rn, const float* u, float ra, float rs) {
    float es = esat(u[1]), tc = u[1]-TF;
    float ea = u[2]*u[4]/(0.622f+0.378f*u[2]);
    float slope = es*17.67f*243.5f/((tc+243.5f)*(tc+243.5f));
    float gamma = CP*u[4]/(0.622f*LV);
    float numerator = slope*rn+air_density(u)*CP*fmaxf(0.0f,es-ea)/fmaxf(ra,1.0f);
    return fmaxf(0.0f,numerator/(slope+gamma*(1.0f+rs/fmaxf(ra,1.0f))));
}
// 对应普通版: ground_evap.ground_resistance。Opt=4的无雪Sakaguchi--Zeng。
__device__ float ground_resistance(const float* s) {
    float sat = fmaxf(s[39],1.0e-6f), fraction = fminf(1.0f,clip(s[4],0.0f,sat)/sat);
    float dry = DZ[0]*(expf((1.0f-fraction)*(1.0f-fraction))-1.0f)/1.718281828459045f;
    float diff = 2.2e-5f*sat*sat*powf(fmaxf(1.0e-6f,1.0f-s[40]/sat),2.0f+3.0f/fmaxf(s[43],1.0e-6f));
    return fmaxf(1.0e-3f,dry/diff);
}
// 对应普通版: soil._thomas。固定4x4追赶法。
__device__ void thomas4(const float a[4], const float b[4], const float c[4], const float d[4], float out[4]) {
    float cp0=c[0]/safe_den(b[0]), dp0=d[0]/safe_den(b[0]);
    float b1=b[1]-a[1]*cp0, cp1=c[1]/safe_den(b1), dp1=(d[1]-a[1]*dp0)/safe_den(b1);
    float b2=b[2]-a[2]*cp1, cp2=c[2]/safe_den(b2), dp2=(d[2]-a[2]*dp1)/safe_den(b2);
    float b3=b[3]-a[3]*cp2, dp3=(d[3]-a[3]*dp2)/safe_den(b3);
    out[3]=dp3; out[2]=dp2-cp2*out[3]; out[1]=dp1-cp1*out[2]; out[0]=dp0-cp0*out[1];
}
// 对应普通版: snow.snow_thermal_conductivity。默认Stieglitz/Yen。
__device__ float snow_k(float density) {
    return clip(2.2f*powf(clip(density,50.0f,917.0f)/1000.0f,1.88f),0.05f,2.5f);
}
// 对应普通版: snow.surface_snow_flux。
__device__ float snow_flux(const float* s, float ts) {
    float depth = fmaxf(s[16],1.0e-8f)*RW/fmaxf(s[22],50.0f);
    return snow_k(s[22])*(ts-s[19])/fmaxf(0.5f*depth,0.01f);
}
// 对应普通版: snow.add_snowfall/compact_snow。三层mask；默认压实Opt=2。
__device__ void snowfall(float* s, int* count, float amount, const float* u) {
    if (amount > 0.0f) {
        float total=amount, weighted_t=amount*u[1], weighted_d=amount*100.0f;
        for (int k=0;k<*count;++k) {
            total+=s[16+k]; weighted_t+=s[16+k]*s[19+k]; weighted_d+=s[16+k]*s[22+k];
        }
        *count = total <= 0.05f ? 1 : (total <= 0.15f ? 2 : 3);
        for (int k=0;k<3;++k) {
            s[16+k]=k<*count ? total/(*count) : 0.0f;
            s[19+k]=k<*count ? weighted_t/total : TF;
            s[22+k]=k<*count ? weighted_d/total : 300.0f;
        }
    }
    float total=0.0f;
    for (int k=0;k<3;++k) total+=s[16+k];
    if (total>1.0e-8f) {
        for (int k=0;k<*count;++k) {
            float over=0.0f;
            for (int j=k+1;j<*count;++j) over+=s[16+j];
            float warm=6.0e-8f*expf(clip(0.08f*(s[19+k]-258.15f),-4.0f,4.0f));
            s[22+k]=fminf(550.0f,s[22+k]+fmaxf(0.0f,u[0])*(warm*(550.0f-s[22+k])+1.5e-7f*over*1.0e4f));
        }
    }
    s[26]=total; s[27]=total>1.0e-8f ? s[19] : TF;
}
// 对应普通版: snow.apply_snow_heat/_implicit_heat_step。固定三层隐式系统。
__device__ float snow_heat(float* s, int* count, float dt, float top_flux) {
    float depth[3], cap[3], conduct[2], k[3], t[3];
    for (int j=0;j<3;++j) {
        depth[j]=j<*count ? s[16+j]*RW/fmaxf(s[22+j],50.0f) : 1.0e-4f;
        cap[j]=j<*count ? 630000.0f*depth[j]/fmaxf(dt,1.0e-6f) : 1.0e12f;
        k[j]=snow_k(s[22+j]);
    }
    for (int j=0;j<2;++j) conduct[j]=j+1<*count
        ? (2.0f/(1.0f/k[j]+1.0f/k[j+1]))/fmaxf(0.5f*(depth[j]+depth[j+1]),1.0e-4f) : 0.0f;
    float b0=cap[0]+conduct[0], c0=-conduct[0]/safe_den(b0);
    float d0=(cap[0]*s[19]+top_flux)/safe_den(b0);
    float b1=cap[1]+conduct[0]+conduct[1]+conduct[0]*c0;
    float c1=-conduct[1]/safe_den(b1), d1=(cap[1]*s[20]+conduct[0]*d0)/safe_den(b1);
    float b2=cap[2]+conduct[1]+conduct[1]*c1;
    t[2]=(cap[2]*s[21]+conduct[1]*d1)/safe_den(b2); t[1]=d1-c1*t[2]; t[0]=d0-c0*t[1];
    float melt=0.0f; int kept=0;
    for (int j=0;j<*count;++j) {
        float amount=fminf(s[16+j],fmaxf(t[j]-TF,0.0f)*630000.0f*depth[j]/(LF*RW));
        melt+=amount; float left=fmaxf(0.0f,s[16+j]-amount);
        if (left>1.0e-8f) {
            s[16+kept]=left; s[19+kept]=fminf(t[j],TF); s[22+kept]=s[22+j]; ++kept;
        }
    }
    for (int j=kept;j<3;++j) { s[16+j]=0.0f; s[19+j]=TF; s[22+j]=300.0f; }
    *count=kept>0 ? kept : 1;
    return melt;
}

struct Energy { float rn,h,le,g,ecan,esoil,residual; };
// 对应普通版: driver.step_column。LE包含土壤蒸发、截留蒸发和PM蒸腾。
__device__ Energy energy(float ts, const float* s, const float* u, bool snow, float transp) {
    Energy e;
    float emiss=snow ? 0.99f : s[35], albedo=snow ? s[28] : s[34];
    float t2=ts*ts, ce=exchange(u,s[33],ts), rho=air_density(u);
    e.rn=(1.0f-albedo)*u[5]+emiss*(u[6]-SIGMA*t2*t2);
    e.h=rho*CP*ce*(ts-u[1]); e.ecan=0.0f; e.esoil=0.0f; e.le=0.0f;
    if (snow) e.g=snow_flux(s,ts);
    else {
        float dq=qsat(ts,u[4])-u[2], ra=1.0f/fmaxf(ce,1.0e-6f);
        float eg=fminf(beta(s[4],s)*rho*LV*fmaxf(0.0f,dq)/(ra+ground_resistance(s)),s[4]*DZ[0]*LV*RW/u[0]);
        float ec=fminf(rho*LV*ce*dq*fminf(1.0f,s[30]/fmaxf(1.0e-9f,0.2f*0.5e-3f*s[31])),s[30]*LV*RW/u[0]);
        e.esoil=eg/LV*u[0]/RW; e.ecan=ec/LV*u[0]/RW;
        e.le=eg+ec+transp; e.g=soil_k(s[4],s)*(ts-s[0])/(0.5f*DZ[0]);
    }
    e.residual=e.rn-e.h-e.le-e.g;
    return e;
}
// 对应普通版: energy.secant_solve。固定25次割线，失败时扫描/二分回退。
__device__ float surface_temperature(const float* s, const float* u, bool snow, float transp) {
    float x0=u[1], x1=u[1]+1.0f, f0=energy(x0,s,u,snow,transp).residual;
    float best=x0, best_abs=fabsf(f0); bool stopped=false, converged=false;
    for (int it=0;it<25;++it) {
        if (!stopped) {
            float f1=energy(x1,s,u,snow,transp).residual;
            if (fabsf(f1)<best_abs) { best=x1; best_abs=fabsf(f1); }
            if (fabsf(f1)<1.0e-3f) { stopped=true; converged=true; }
            else if (fabsf(f1-f0)<1.0e-12f) stopped=true;
            else { float x2=clip(x1-f1*(x1-x0)/(f1-f0),200.0f,330.0f); x0=x1; f0=f1; x1=x2; }
        }
    }
    if (converged) return x1;
    float previous=energy(200.0f,s,u,snow,transp).residual;
    int bracket=-1;
    for (int j=0;j<130;++j) {
        float next=energy(201.0f+j,s,u,snow,transp).residual;
        if (bracket<0 && isfinite(previous) && isfinite(next) && previous*next<=0.0f) bracket=j;
        previous=next;
    }
    if (bracket<0) return best;
    float lo=200.0f+bracket, hi=lo+1.0f, flo=energy(lo,s,u,snow,transp).residual, result=lo;
    bool done=false;
    for (int j=0;j<40;++j) {
        if (!done) {
            float mid=0.5f*(lo+hi), fm=energy(mid,s,u,snow,transp).residual;
            if (fabsf(fm)<1.0e-3f) { result=mid; done=true; }
            else if (flo*fm<=0.0f) hi=mid;
            else { lo=mid; flo=fm; }
        }
    }
    return done ? result : 0.5f*(lo+hi);
}
// 对应普通版: soil.step_soil_water。入渗、根系水量限制、蒸发、Campbell排水。
__device__ float soil_water(float* s, const float* u, float incoming, float transp, float esoil, float* et) {
    float remaining=fmaxf(0.0f,incoming), runoff=0.0f; *et=0.0f;
    for (int k=0;k<4;++k) {
        float add=fminf(remaining,fmaxf(0.0f,(s[39]-s[4+k])*DZ[k]));
        s[4+k]+=add/DZ[k]; remaining-=add;
    }
    runoff+=fmaxf(0.0f,remaining);
    for (int k=0;k<4;++k) {
        float take=fminf(transp*s[12+k],fmaxf(0.0f,s[4+k]-s[40])*DZ[k]);
        s[4+k]-=take/DZ[k]; *et+=take;
    }
    float evap=fminf(esoil,fmaxf(0.0f,s[4]*DZ[0])); s[4]-=evap/DZ[0]; *et+=evap;
    for (int k=0;k<4;++k) {
        float q=s[42]*powf(clip(s[4+k]/s[39],0.0f,1.0f),2.0f*s[43]+3.0f);
        float drain=fminf(fmaxf(0.0f,s[4+k]-s[41])*DZ[k],q*u[0]);
        s[4+k]-=drain/DZ[k];
        if (k<3) s[5+k]+=drain/DZ[k+1]; else runoff+=drain;
    }
    for (int k=0;k<4;++k) {
        float excess=fmaxf(0.0f,s[4+k]-s[39])*DZ[k]; s[4+k]-=excess/DZ[k];
        if (k<3) s[5+k]+=excess/DZ[k+1]; else runoff+=excess;
    }
    return runoff;
}
// 对应普通版: soil.step_soil_heat/frozen_soil.update_soil_phase。底部零通量。
__device__ void soil_heat(float* s, float dt, float flux) {
    float cap[4], k[4], conduct[3], a[4], b[4], c[4], rhs[4], out[4];
    for (int j=0;j<4;++j) { cap[j]=capacity(s[4+j],s[j],s); k[j]=soil_k(s[4+j],s); }
    for (int j=0;j<3;++j) conduct[j]=(2.0f/(1.0f/k[j]+1.0f/k[j+1]))/(0.5f*(DZ[j]+DZ[j+1]));
    for (int j=0;j<4;++j) {
        float up=j>0 ? conduct[j-1] : 0.0f, down=j<3 ? conduct[j] : 0.0f;
        a[j]=-up; c[j]=-down; b[j]=cap[j]*DZ[j]/dt+up+down;
        rhs[j]=cap[j]*DZ[j]/dt*s[j]+(j==0 ? flux : 0.0f);
    }
    thomas4(a,b,c,rhs,out);
    for (int j=0;j<4;++j) { s[j]=out[j]; s[8+j]=fmaxf(0.0f,s[4+j]-liquid(s[4+j],s[j],s)); }
}
// 对应普通版: driver.step_column。默认选项组合全链路。
__device__ void step(float* s, int* count, const float* u, float* diag) {
    float total=fmaxf(0.0f,u[7])*fmaxf(0.0f,u[0]), delta=u[1]-TF;
    float fraction=delta>2.5f ? 0.0f : (delta<=0.5f ? 1.0f
        : (delta<=2.0f ? clip(1.0f-(-54.632f+0.2f*u[1]),0.0f,1.0f) : 0.6f));
    float snow=total*fraction, rain=total-snow, canmax=0.5e-3f*s[31];
    float add=fminf(rain,fmaxf(0.0f,canmax-s[30])); s[30]+=add;
    float through=rain-add;
    snowfall(s,count,snow,u);
    float t2=u[1]*u[1], rncan=s[32]*((1.0f-s[34])*u[5]+s[35]*(u[6]-SIGMA*t2*t2));
    float tr=penman(fmaxf(0.0f,rncan),u,1.0f/fmaxf(exchange(u,s[33],u[1]),1.0e-6f),stomata(s,u));
    tr*=1.0f-fminf(1.0f,s[26]/0.02f);
    float available=0.0f;
    for (int k=0;k<4;++k) available+=fmaxf(0.0f,s[4+k]-s[40])*DZ[k];
    float tr_m=tr/LV*u[0]/RW;
    if (tr_m>available) tr*=available/fmaxf(tr_m,1.0e-12f);
    tr_m=tr/LV*u[0]/RW;
    bool is_snow=s[26]>1.0e-4f;
    float ts=surface_temperature(s,u,is_snow,tr);
    Energy e=energy(ts,s,u,is_snow,tr); float melt=0.0f;
    if (is_snow) melt=snow_heat(s,count,u[0],e.g);
    else s[30]=fmaxf(0.0f,s[30]-e.ecan);
    float et, runoff=soil_water(s,u,through+melt,tr_m,e.esoil,&et);
    soil_heat(s,u[0],is_snow ? 0.0f : e.g);
    s[25]=ts; s[26]=s[16]+s[17]+s[18]; s[27]=s[26]>1.0e-8f ? s[19] : TF;
    if (s[26]<=1.0e-6f) { s[28]=0.8f; s[29]=0.0f; }
    else { s[29]=snow>1.0e-6f ? 0.0f : fmaxf(0.0f,s[29])+fmaxf(0.0f,u[0])/86400.0f;
        s[28]=clip(0.55f+0.30f*expf(-s[29]/3.0f),0.50f,0.90f); }
    diag[0]=ts; diag[1]=e.rn; diag[2]=e.h; diag[3]=e.le;
    diag[4]=is_snow && s[26]<=1.0e-4f ? 0.0f : e.g;
    diag[5]=(et+e.ecan)*1000.0f; diag[6]=u[7]*u[0]*1000.0f;
    diag[7]=runoff*1000.0f; diag[8]=melt*1000.0f; diag[9]=s[26]; diag[10]=s[30]; diag[11]=e.residual;
}
} // namespace

// 对应普通版: driver.run_columns。1 thread/column，SoA，grid-stride。
extern "C" __global__ void step_columns(
    const float* __restrict__ state_soa, const int* __restrict__ options_soa,
    const float* __restrict__ forcing_soa, float* __restrict__ state_out_soa,
    int* __restrict__ options_out_soa, float* __restrict__ diag_soa, int n_columns) {
    const int first=blockIdx.x*blockDim.x+threadIdx.x;
    const int stride=blockDim.x*gridDim.x;
    for (int column=first;column<n_columns;column+=stride) {
        float s[70], u[8], d[12];
        for (int j=0;j<70;++j) s[j]=state_soa[j*n_columns+column];
        for (int j=0;j<8;++j) u[j]=forcing_soa[j*n_columns+column];
        int count=options_soa[32*n_columns+column];
        step(s,&count,u,d);
        for (int j=0;j<70;++j) state_out_soa[j*n_columns+column]=s[j];
        for (int j=0;j<33;++j) options_out_soa[j*n_columns+column]=j==32 ? count : options_soa[j*n_columns+column];
        for (int j=0;j<12;++j) diag_soa[j*n_columns+column]=d[j];
    }
}
