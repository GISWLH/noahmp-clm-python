// 12 implemented PhysicsOptions selectors; float32 SoA ABI from STATE_SPEC.md.
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
__device__ __forceinline__ float vapor_exp(float t) {
    float tc = t - TF;
    return expf(17.67f * tc / (t + (243.5f-TF)));
}
__device__ __forceinline__ float esat(float t) {
    return 611.2f * vapor_exp(t);
}
// 对应普通版: energy.qsat。
__device__ __forceinline__ float qsat(float t, float ps) {
    // XLA folds consecutive scalar constants before multiplying exp().
    float exponent = vapor_exp(t);
    return (0.622f*611.2f)*exponent / fmaxf(1.0f, ps-(0.378f*611.2f)*exponent);
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
__device__ float exchange(const float* u, float z0, float ts, int option=1) {
    // 对应普通版: surface_drag.exchange_velocity。Chen/bulk option 2.
    if (option==2) {
        float rough=clip(z0,1.0e-5f,5.0f);
        return (0.16f/(logf(10.0f/rough)*logf(10.0f/(rough/10.0f))))*fmaxf(u[3],0.3f);
    }
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
            // Python precomputes KARMAN*GRAVITY before conversion to f32;
            // match that constant and the ustar**3 multiplication tree.
            float on = fabsf(h) < 1.0e-8f ? 1.0e30f
                : clip((-rho*CP*u[1])*(us*(us*us))/(3.924f*h), -1.0e6f, 1.0e6f);
            bool next_finite = fabsf(on) < 1.0e29f;
            done = (finite && next_finite && fabsf(on-ob)/fmaxf(1.0f,fabsf(ob)) < 1.0e-4f)
                || (!finite && !next_finite);
            ce = cn; ob = on;
        }
    }
    return ce;
}
// 对应普通版: frozen_soil._niu_yang_liquid。Niu & Yang (2006)。
__device__ float smooth_water_cap(float total, float potential) {
    float width=fmaxf(0.04f*total,1.0e-12f);
    float rounding=fmaxf(width-fabsf(total-potential),0.0f);
    return fmaxf(0.0f,fminf(total,potential)-rounding*rounding/(4.0f*width));
}
__device__ float liquid(float smc, float t, const float* s, int option=1) {
    float total = fmaxf(0.0f, smc);
    if (t >= TF) return total;
    // Match XLA's constant-folded coefficient and reciprocal scale. Keeping
    // the two constant divisions inside the float32 expression introduces
    // extra rounding that is amplified by apparent-capacity differencing.
    constexpr float phase_coefficient=(LF/461.5f)/(TF*TF);
    float clausius = phase_coefficient*fmaxf(0.0f,TF-t);
    float potential = s[39] * powf(fmaxf(clausius,1.0e-8f)*20000.0f, -1.0f/fmaxf(s[43],1.0e-6f));
    float result=smooth_water_cap(total,potential);
    // 对应普通版: frozen_soil.unfrozen_water_content。Koren's three updates.
    if (option==2) {
        for (int it=0;it<3;++it) result=0.5f*(result+fminf(smc,smooth_water_cap(result,potential)));
        result=fminf(smc,result);
    }
    return result;
}
// 对应普通版: frozen_soil.apparent_heat_capacity。
__device__ float capacity(float smc, float t, const float* s, int option=1) {
    float derivative = (liquid(smc,t+0.02f,s,option)-liquid(smc,t-0.02f,s,option))*25.0f;
    return smc*4.18e6f+(1.0f-s[39])*2.0e6f+RW*LF*fmaxf(0.0f,derivative);
}
// 对应普通版: soil.thermal_conductivity。
__device__ float soil_k(float smc, const float* s) {
    float sr = clip(smc/s[39],0.0f,1.0f);
    float ke = sr > 0.1f ? clip(log10f(fmaxf(sr,1.0e-6f))+1.0f,0.0f,1.0f) : 0.0f;
    return 0.2f+((0.6f-0.2f)+1.6f*s[44])*ke;
}
// 对应普通版: stress.soil_water_stress。Noah/CLM/SSiB water stress.
__device__ float beta(float smc, const float* s, int option=1) {
    switch (option) {
        case 2: case 3: {
            float sat=fmaxf(s[39],1.0e-9f);
            float psi=-0.1f*powf(clip(smc/sat,1.0e-4f,1.0f),-s[43]);
            float ref=-0.1f*powf(clip(s[41]/sat,1.0e-4f,1.0f),-s[43]);
            float matric=clip((-1500.0f-psi)/(-1500.0f-ref),0.0f,1.0f);
            return option==3 ? matric*matric : matric;
        }
        default: return clip((smc-s[40])/fmaxf(1.0e-9f,s[41]-s[40]),0.0f,1.0f);
    }
}
// 对应普通版: canopy.ball_berry_rs。默认经验光响应Ball--Berry。
__device__ float stomata(const float* s, const float* u, int water_option=1, int option=1) {
    float weights[4];
    for (int k=0;k<4;++k) weights[k]=s[12+k]*beta(s[4+k],s,water_option);
    // Match jax_core's explicit four-term cross-pair sum.
    float water=(weights[0]+weights[2])+(weights[1]+weights[3]);
    water = fmaxf(0.02f,clip(water,0.0f,1.0f));
    // Match the production JAX HLO: square the temperature delta before scaling.
    // Left-associated coefficient multiplication changes stomatal LE near 274.2 K.
    float temperature_delta = 298.0f-u[1];
    float temp = clip(1.0f-0.0016f*(temperature_delta*temperature_delta),0.05f,1.0f);
    // 对应普通版: canopy.jarvis_rs。Light, temperature, VPD and root stress.
    if (option==2) {
        float ea=u[2]*u[4]/(0.622f+0.378f*u[2]);
        float vpd=fmaxf(0.0f,esat(u[1])-ea);
        float light=clip(u[5]/500.0f,0.02f,1.0f), fvpd=clip(1.0f-vpd/4000.0f,0.10f,1.0f);
        return clip(s[36]/(light*temp*fvpd*water),s[36],5000.0f);
    }
    float a = 25.0f*(1.0f-expf(-fmaxf(0.0f,u[5])*2.0f/450.0f))*temp*water;
    // XLA combines (q*ps / humidity_denominator) / es into one divide.
    float rh = clip((u[2]*u[4])/((0.622f+0.378f*u[2])*fmaxf(esat(u[1]),1.0f)),0.05f,1.0f);
    float gs = (0.01f+9.0f*a*rh/400.0f)*8.314462618f*u[1]/fmaxf(u[4],1.0f);
    return clip(1.0f/fmaxf(gs,1.0e-6f),s[36],5000.0f);
}
// 对应普通版: canopy.penman_monteith。
__device__ float penman(float rn, const float* u, float ra, float rs) {
    float exponent=vapor_exp(u[1]), es=611.2f*exponent;
    float vapor_denominator=u[1]+(243.5f-TF);
    float ea = u[2]*u[4]/(0.622f+0.378f*u[2]);
    float slope = (611.2f*17.67f*243.5f)*exponent/(vapor_denominator*vapor_denominator);
    float gamma = (CP/(0.622f*LV))*u[4];
    float numerator = slope*rn+air_density(u)*CP*fmaxf(0.0f,es-ea)/fmaxf(ra,1.0f);
    return fmaxf(0.0f,numerator/(slope+gamma*(1.0f+rs/fmaxf(ra,1.0f))));
}
// 对应普通版: ground_evap.ground_resistance。Opt=4的无雪Sakaguchi--Zeng。
__device__ float ground_resistance(const float* s, int option=4) {
    float sat = fmaxf(s[39],1.0e-6f), fraction = fminf(1.0f,clip(s[4],0.0f,sat)/sat);
    switch (option) {
        case 2: return expf(8.25f-4.225f*fmaxf(0.0f,s[4]/sat));
        case 3: return expf(8.25f-6.0f*fmaxf(0.0f,s[4]/sat));
        default: break; // 1 and 4 share the no-snow Sakaguchi--Zeng branch.
    }
    // DZ is a captured array in the full JAX graph: its multiplication must
    // round before the constant reciprocal, even though an isolated scalar
    // probe can fold both constants together.
    float dry = (DZ[0]*(expf((1.0f-fraction)*(1.0f-fraction))-1.0f))*(1.0f/1.718281828459045f);
    float diff = 2.2e-5f*(sat*sat)*powf(fmaxf(1.0e-6f,1.0f-s[40]/sat),2.0f+3.0f/fmaxf(s[43],1.0e-6f));
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
// 对应普通版: snow.snow_thermal_conductivity。Five implemented schemes.
__device__ float snow_k(float density, float temperature=TF, int option=1) {
    if (option==1) return clip(2.2f*powf(clip(density,50.0f,917.0f)/1000.0f,1.88f),0.05f,2.5f);
    float ratio=clip(density,50.0f,917.0f)/1000.0f, result;
    switch (option) {
        case 2: result=2.9f*(ratio*ratio); break;
        case 3: result=0.30f; break;
        case 4: result=(0.023f+0.234f*(ratio*ratio))*(1.0f+0.05f*clip(temperature-268.15f,0.0f,5.0f)/5.0f); break;
        default: result=(2.2f*powf(ratio,1.88f))*(1.0f+0.1f*ratio); break;
    }
    return clip(result,0.05f,2.5f);
}
// 对应普通版: snow.surface_snow_flux。
__device__ float snow_flux(const float* s, float ts, int option=1) {
    float depth = fmaxf(s[16],1.0e-8f)*RW/fmaxf(s[22],50.0f);
    return snow_k(s[22],s[19],option)*(ts-s[19])/fmaxf(0.5f*depth,0.01f);
}
// 对应普通版: snow.add_snowfall/compact_snow。三层mask；默认压实Opt=2。
__device__ void snowfall(float* s, int* count, float amount, const float* u, int option=2) {
    if (amount > 0.0f) {
        // Match _add_snowfall: reduce the old layers before adding new snow.
        float total=0.0f, weighted_t=0.0f, weighted_d=0.0f;
        for (int k=0;k<*count;++k) {
            total+=s[16+k]; weighted_t+=s[16+k]*s[19+k]; weighted_d+=s[16+k]*s[22+k];
        }
        total+=amount; weighted_t+=amount*u[1]; weighted_d+=amount*100.0f;
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
            if (option==1) {
                warm=1.0e-7f*expf(clip(0.08f*(s[19+k]-258.15f),-4.0f,4.0f));
                s[22+k]=fminf(550.0f,s[22+k]+fmaxf(0.0f,u[0])*(warm*(550.0f-s[22+k])+2.5e-7f*over*1.0e4f));
            } else s[22+k]=fminf(550.0f,s[22+k]+fmaxf(0.0f,u[0])*(warm*(550.0f-s[22+k])+1.5e-7f*over*1.0e4f));
        }
    }
    s[26]=total; s[27]=total>1.0e-8f ? s[19] : TF;
}
// 对应普通版: snow.apply_snow_heat/_implicit_heat_step。固定三层隐式系统。
__device__ float snow_heat(float* s, int* count, float dt, float top_flux, float* soil_energy, int option=1) {
    float depth[3], cap[3], conduct[2], k[3], t[3];
    for (int j=0;j<3;++j) {
        depth[j]=j<*count ? s[16+j]*RW/fmaxf(s[22+j],50.0f) : 1.0e-4f;
        cap[j]=j<*count ? 630000.0f*depth[j]/fmaxf(dt,1.0e-6f) : 1.0e12f;
        k[j]=snow_k(s[22+j],s[19+j],option);
    }
    for (int j=0;j<2;++j) conduct[j]=j+1<*count
        ? (2.0f/(1.0f/k[j]+1.0f/k[j+1]))/fmaxf(0.5f*(depth[j]+depth[j+1]),1.0e-4f) : 0.0f;
    float b0=cap[0]+conduct[0], c0=-conduct[0]/safe_den(b0);
    float d0=(cap[0]*s[19]+top_flux)/safe_den(b0);
    float b1=cap[1]+conduct[0]+conduct[1]+conduct[0]*c0;
    float c1=-conduct[1]/safe_den(b1), d1=(cap[1]*s[20]+conduct[0]*d0)/safe_den(b1);
    float b2=cap[2]+conduct[1]+conduct[1]*c1;
    t[2]=(cap[2]*s[21]+conduct[1]*d1)/safe_den(b2); t[1]=d1-c1*t[2]; t[0]=d0-c0*t[1];
    float melt=0.0f, remaining_heat=0.0f; int kept=0;
    for (int j=0;j<*count;++j) {
        float capacity=630000.0f*depth[j];
        float excess=(t[j]-TF)*capacity+remaining_heat;
        float amount=fminf(s[16+j],fmaxf(excess,0.0f)/(LF*RW));
        float temperature=remaining_heat>0.0f ? TF+fminf(excess,0.0f)/capacity : fminf(t[j],TF);
        remaining_heat=fmaxf(0.0f,excess-amount*(LF*RW));
        melt+=amount; float left=fmaxf(0.0f,s[16+j]-amount);
        if (left>1.0e-8f) {
            s[16+kept]=left; s[19+kept]=temperature; s[22+kept]=s[22+j]; ++kept;
        }
    }
    for (int j=kept;j<3;++j) { s[16+j]=0.0f; s[19+j]=TF; s[22+j]=300.0f; }
    *count=kept>0 ? kept : 1;
    *soil_energy=remaining_heat;
    return melt;
}

struct SnowBoundary { float reference,effective_k,surface_k; };
// Implicit top boundary: eliminate the same snow heat matrix used below.
__device__ SnowBoundary snow_response(const float* s, float dt, int option) {
    int count=s[18]>1.0e-8f ? 3 : (s[17]>1.0e-8f ? 2 : 1);
    float depth[3],cap[3],k[3],g[2];
    for(int j=0;j<3;++j) {
        depth[j]=j<count ? s[16+j]*RW/fmaxf(s[22+j],50.0f) : 1.0e-4f;
        cap[j]=j<count ? 630000.0f*depth[j]/fmaxf(dt,1.0e-6f) : 1.0e12f;
        k[j]=snow_k(s[22+j],s[19+j],option);
    }
    for(int j=0;j<2;++j) g[j]=j+1<count
        ? (2.0f/(1.0f/k[j]+1.0f/k[j+1]))/fmaxf(0.5f*(depth[j]+depth[j+1]),1.0e-4f) : 0.0f;
    float factor1=g[1]/fmaxf(cap[2]+g[1],1.0e-12f);
    float diagonal1=(cap[1]+g[0]+g[1])-factor1*g[1];
    float rhs1=cap[1]*(s[20]-s[19])+factor1*(cap[2]*(s[21]-s[19]));
    float factor0=g[0]/fmaxf(diagonal1,1.0e-12f);
    float admittance=fmaxf((cap[0]+g[0])-factor0*g[0],1.0e-12f);
    SnowBoundary b;
    b.reference=s[19]+(factor0*rhs1)/admittance;
    b.surface_k=k[0]/fmaxf(0.5f*depth[0],0.01f);
    b.effective_k=b.surface_k/(1.0f+b.surface_k/admittance);
    return b;
}
__device__ float coupled_snow_flux(float ts, SnowBoundary b) {
    return fmaxf(b.effective_k*(ts-b.reference),b.surface_k*(ts-TF));
}
struct Energy { float rn,h,le,g,ecan,esoil,residual; };
// 对应普通版: driver.step_column。LE包含土壤蒸发、截留蒸发和PM蒸腾。
__device__ Energy energy(float ts, const float* s, const float* u, bool snow, float transp,
                         int drag=1, int ground=4, int water=1, int conductivity=1,
                         SnowBoundary boundary={0.0f,0.0f,0.0f}) {
    Energy e;
    float emiss=snow ? 0.99f : s[35], albedo=snow ? s[28] : s[34];
    float t2=ts*ts, ce=exchange(u,s[33],ts,drag), rho=air_density(u);
    // jax_core uses SIGMA * ts**4: square twice before scaling, so the
    // float32 rounding tree does not depend on left-associated products.
    e.rn=(1.0f-albedo)*u[5]+emiss*(u[6]-SIGMA*(t2*t2));
    e.h=rho*CP*ce*(ts-u[1]); e.ecan=0.0f; e.esoil=0.0f; e.le=0.0f;
    if (snow) {
        if(boundary.surface_k==0.0f) boundary=snow_response(s,u[0],conductivity);
        e.g=coupled_snow_flux(ts,boundary); e.residual=e.rn-e.h-e.g;
    }
    else {
        float dq=qsat(ts,u[4])-u[2], ra=1.0f/fmaxf(ce,1.0e-6f);
        float eg=fminf(beta(s[4],s,water)*rho*LV*fmaxf(0.0f,dq)/(ra+ground_resistance(s,ground)),s[4]*DZ[0]*LV*RW/u[0]);
        float ec=fminf(rho*LV*ce*dq*fminf(1.0f,s[30]/fmaxf(1.0e-9f,0.2f*0.5e-3f*s[31])),s[30]*LV*RW/u[0]);
        e.esoil=eg/LV*u[0]/RW; e.ecan=ec/LV*u[0]/RW;
        e.le=eg+ec+transp; e.g=soil_k(s[4],s)*(ts-s[0])/(0.5f*DZ[0]);
        // Match the reference/JAX root function's ordered subtractions.
        e.residual=e.rn-e.h-eg-ec-transp-e.g;
    }
    return e;
}
// 对应普通版: energy.secant_solve。固定25次割线，失败时扫描/二分回退。
__device__ float surface_temperature(const float* s, const float* u, bool snow, float transp,
                                    int drag=1, int ground=4, int water=1, int conductivity=1) {
    SnowBoundary boundary=snow ? snow_response(s,u[0],conductivity) : SnowBoundary{0.0f,0.0f,0.0f};
    float x0=u[1], x1=u[1]+1.0f, f0=energy(x0,s,u,snow,transp,drag,ground,water,conductivity,boundary).residual;
    float best=x0, best_abs=fabsf(f0); bool stopped=false, converged=false;
    for (int it=0;it<25;++it) {
        if (!stopped) {
            float f1=energy(x1,s,u,snow,transp,drag,ground,water,conductivity,boundary).residual;
            if (fabsf(f1)<best_abs) { best=x1; best_abs=fabsf(f1); }
            if (fabsf(f1)<1.0e-3f) { stopped=true; converged=true; }
            else if (fabsf(f1-f0)<1.0e-12f) stopped=true;
            else { float x2=clip(x1-f1*(x1-x0)/(f1-f0),200.0f,330.0f); x0=x1; f0=f1; x1=x2; }
        }
    }
    if (converged) return x1;
    float previous=energy(200.0f,s,u,snow,transp,drag,ground,water,conductivity,boundary).residual;
    int bracket=-1;
    for (int j=0;j<130;++j) {
        float next=energy(201.0f+j,s,u,snow,transp,drag,ground,water,conductivity,boundary).residual;
        if (bracket<0 && isfinite(previous) && isfinite(next) && previous*next<=0.0f) bracket=j;
        previous=next;
    }
    if (bracket<0) return best;
    float lo=200.0f+bracket, hi=lo+1.0f, flo=energy(lo,s,u,snow,transp,drag,ground,water,conductivity,boundary).residual, result=lo;
    bool done=false;
    for (int j=0;j<40;++j) {
        if (!done) {
            float mid=0.5f*(lo+hi), fm=energy(mid,s,u,snow,transp,drag,ground,water,conductivity,boundary).residual;
            if (fabsf(fm)<1.0e-3f) { result=mid; done=true; }
            else if (flo*fm<=0.0f) hi=mid;
            else { lo=mid; flo=fm; }
        }
    }
    return done ? result : 0.5f*(lo+hi);
}
struct Runoff { float surface, subsurface; };
// 对应普通版: runoff.topmodel_runoff/bats_runoff。Independent outlet components.
__device__ Runoff runoff_scheme(float incoming, const float* s, int option) {
    incoming=fmaxf(0.0f,incoming);
    float surface=0.0f;
    switch (option) {
        case 1: case 2: {
            float deficits[4];
            for(int k=0;k<4;++k) deficits[k]=fmaxf(0.0f,s[41]-s[4+k])*DZ[k];
            float deficit=(deficits[0]+deficits[2])+(deficits[1]+deficits[3]);
            float terrain=expf(-s[37]);
            if(option==2) deficit=fmaxf(deficit,fmaxf(0.0f,s[38])*s[39]*terrain);
            deficit=fmaxf(0.0f,deficit*terrain);
            float wetness=clip(expf(s[37])*expf(-fmaxf(0.0f,s[38])/2.0f),0.0f,1.0f);
            surface=fmaxf(0.0f,incoming-deficit)*wetness;
            break;
        }
        case 4: {
            float wet[4];
            for(int k=0;k<4;++k) wet[k]=clip(s[4+k]/fmaxf(s[39],1.0e-6f),0.0f,1.0f);
            float mean=((wet[0]+wet[2])+(wet[1]+wet[3]))*0.25f;
            surface=incoming*clip(mean*(mean*mean),0.0f,1.0f);
            break;
        }
        default: break; // 3: original free drainage.
    }
    surface=clip(surface,0.0f,incoming);
    return {surface,incoming-surface};
}
// 对应普通版: runoff.route_runoff。The second selector sees the first remainder.
__device__ Runoff route_runoff(float incoming, const float* s, int surface_option, int subsurface_option) {
    Runoff first=runoff_scheme(incoming,s,surface_option);
    if(surface_option==subsurface_option) return first;
    Runoff second=runoff_scheme(first.subsurface,s,subsurface_option);
    return {first.surface+second.surface,second.subsurface};
}
// 对应普通版: soil.step_soil_water。入渗、根系水量限制、蒸发、Campbell排水。
__device__ float soil_water(float* s, const float* u, float incoming, float transp, float esoil, float* et,
                            int permeability=1, int supercool=1, int surface_runoff=3, int subsurface_runoff=3) {
    float remaining=fmaxf(0.0f,incoming), runoff=0.0f; *et=0.0f;
    for (int k=0;k<4;++k) {
        float add=fminf(remaining,fmaxf(0.0f,(s[39]-s[4+k])*DZ[k]));
        s[4+k]+=add/DZ[k]; remaining-=add;
    }
    runoff+=fmaxf(0.0f,remaining);
    // Corresponding reference: frozen_soil.permeability_factor. Snapshot phase
    // after infiltration and before evaporation/extraction, as in both references.
    float frozen_factor[4];
    if(permeability==2) for(int k=0;k<4;++k) {
        float ice=fmaxf(0.0f,s[4+k]-liquid(s[4+k],s[k],s,supercool));
        frozen_factor[k]=expf(-7.0f*ice/fmaxf(s[39],1.0e-6f));
    }
    for (int k=0;k<4;++k) {
        float take=fminf(transp*s[12+k],fmaxf(0.0f,s[4+k]-s[40])*DZ[k]);
        s[4+k]-=take/DZ[k]; *et+=take;
    }
    float evap=fminf(esoil,fmaxf(0.0f,s[4]*DZ[0])); s[4]-=evap/DZ[0]; *et+=evap;
    for (int k=0;k<4;++k) {
        float q=s[42]*powf(clip(s[4+k]/s[39],0.0f,1.0f),2.0f*s[43]+3.0f);
        if(permeability==2) q=(s[42]*frozen_factor[k])*powf(clip(s[4+k]/s[39],0.0f,1.0f),2.0f*s[43]+3.0f);
        float drain=fminf(fmaxf(0.0f,s[4+k]-s[41])*DZ[k],q*u[0]);
        s[4+k]-=drain/DZ[k];
        if (k<3) s[5+k]+=drain/DZ[k+1]; else runoff+=drain;
    }
    for (int k=0;k<4;++k) {
        float excess=fmaxf(0.0f,s[4+k]-s[39])*DZ[k]; s[4+k]-=excess/DZ[k];
        if (k<3) s[5+k]+=excess/DZ[k+1]; else runoff+=excess;
    }
    // Preserve the default sum's arithmetic; the default scheme is (0, runoff).
    if(surface_runoff==3 && subsurface_runoff==3) return runoff;
    Runoff routed=route_runoff(runoff,s,surface_runoff,subsurface_runoff);
    return routed.surface+routed.subsurface;
}
// 对应普通版: soil.step_soil_heat/frozen_soil.update_soil_phase。底部零通量。
__device__ void soil_heat(float* s, float dt, float flux, int supercool=1) {
    if (flux==0.0f && s[0]==s[1] && s[0]==s[2] && s[0]==s[3]) {
        for (int j=0;j<4;++j) s[8+j]=fmaxf(0.0f,s[4+j]-liquid(s[4+j],s[j],s,supercool));
        return;
    }
    float cap[4], k[4], conduct[3], a[4], b[4], c[4], rhs[4], out[4];
    for (int j=0;j<4;++j) { cap[j]=capacity(s[4+j],s[j],s,supercool); k[j]=soil_k(s[4+j],s); }
    for (int j=0;j<3;++j) conduct[j]=(2.0f/(1.0f/k[j]+1.0f/k[j+1]))/(0.5f*(DZ[j]+DZ[j+1]));
    for (int j=0;j<4;++j) {
        float up=j>0 ? conduct[j-1] : 0.0f, down=j<3 ? conduct[j] : 0.0f;
        a[j]=-up; c[j]=-down; b[j]=cap[j]*DZ[j]/dt+up+down;
        rhs[j]=cap[j]*DZ[j]/dt*s[j]+(j==0 ? flux : 0.0f);
    }
    thomas4(a,b,c,rhs,out);
    for (int j=0;j<4;++j) { s[j]=out[j]; s[8+j]=fmaxf(0.0f,s[4+j]-liquid(s[4+j],s[j],s,supercool)); }
}
// Wang (2019) equations 1-2, fixed dry-bulb temperature in the depression equation.
__device__ void wetbulb_values(float t,float q,float p,float *tw,float *fraction) {
    float tc=t-273.15f,vapor=q*p/(.622f+.378f*q);
    float latent=t>273.15f?2500000.f:2834000.f;
    float gamma=1004.f*p/(.622f*latent),lower=-150.f,upper=100.f;
    for(int k=0;k<60;k++) {
        float middle=.5f*(lower+upper);
        float es=610.8f*expf(17.27f*middle/(237.3f+middle));
        float residual=(es-vapor)+gamma*(middle-tc);
        if(residual<0.f)lower=middle;else upper=middle;
    }
    float offset=.5f*(lower+upper);
    *tw=273.15f+offset;
    float argument=(float)log(6.99e-5)+2.f*(offset+3.97f);
    float tail=expf(-fabsf(argument)),den=1.f+tail;
    *fraction=argument>=0.f?tail/den:1.f/den;
}
// 对应普通版: precip.partition_precipitation。Jordan, BATS, freezing, wet bulb.
__device__ float snow_fraction(float tair, int option=1,float qair=0.f,float pressure=101325.f) {
    switch(option) {
        case 5: {float tw,fraction;wetbulb_values(tair,qair,pressure,&tw,&fraction);return fraction;}
        case 2: return tair<(273.15f+2.2f) ? 1.0f : 0.0f;
        case 3: return tair<TF ? 1.0f : 0.0f;
        default: {
            float delta=tair-TF;
            return delta>2.5f ? 0.0f : (delta<=0.5f ? 1.0f
                : (delta<=2.0f ? clip(1.0f-(-54.632f+0.2f*tair),0.0f,1.0f) : 0.6f));
        }
    }
}
// 对应普通版: snow.update_snow_albedo。BATS/CLASS share the snow-age update.
__device__ void snow_albedo(float* s, float snow, float dt, int option=1) {
    if (s[26]<=1.0e-6f) { s[28]=0.8f; s[29]=0.0f; }
    else {
        s[29]=snow>1.0e-6f ? 0.0f : fmaxf(0.0f,s[29])+fmaxf(0.0f,dt)/86400.0f;
        if(option==2) {
            float fresh=s[27]<271.15f ? 0.84f : 0.70f;
            float rate=s[27]<268.15f ? 0.02f : 0.06f;
            float floor=s[27]<271.15f ? 0.65f : 0.50f;
            s[28]=clip(fresh-rate*sqrtf(s[29]),floor,fresh);
        } else s[28]=clip(0.55f+0.30f*expf(-s[29]/3.0f),0.50f,0.90f);
    }
}
// 对应普通版: driver.step_column。Default specialization shares the exact
// expressions with runtime option dispatch; no second copy of the physics.
template<bool Defaults>
__device__ void step_selected(float* s, int* count, const float* u, float* diag, const int* opts) {
    const int phase=Defaults ? 1 : opts[1], water=Defaults ? 1 : opts[2];
    const int ground=Defaults ? 4 : opts[3], drag=Defaults ? 1 : opts[4], stomatal=Defaults ? 1 : opts[5];
    const int albedo=Defaults ? 1 : opts[6], conductivity=Defaults ? 1 : opts[9], supercool=Defaults ? 1 : opts[11];
    const int surface=Defaults ? 3 : opts[12], subsurface=Defaults ? 3 : opts[13];
    const int permeability=Defaults ? 1 : opts[14], compaction=Defaults ? 2 : opts[23];
    float total=fmaxf(0.0f,u[7])*fmaxf(0.0f,u[0]), fraction=snow_fraction(u[1],phase,u[2],u[4]);
    float snow=total*fraction, rain=total-snow, canmax=0.5e-3f*s[31];
    float add=fminf(rain,fmaxf(0.0f,canmax-s[30])); s[30]+=add;
    float through=rain-add;
    snowfall(s,count,snow,u,compaction);
    float t2=u[1]*u[1], rncan=s[32]*((1.0f-s[34])*u[5]+s[35]*(u[6]-SIGMA*(t2*t2)));
    float tr=penman(fmaxf(0.0f,rncan),u,1.0f/fmaxf(exchange(u,s[33],u[1],drag),1.0e-6f),stomata(s,u,water,stomatal));
    tr*=1.0f-fminf(1.0f,s[26]/0.02f);
    float available=0.0f;
    for (int k=0;k<4;++k) available+=fmaxf(0.0f,s[4+k]-s[40])*DZ[k];
    float tr_m=tr/LV*u[0]/RW;
    if (tr_m>available) tr*=available/fmaxf(tr_m,1.0e-12f);
    tr_m=tr/LV*u[0]/RW;
    bool is_snow=s[26]>1.0e-4f;
    float ts=surface_temperature(s,u,is_snow,tr,drag,ground,water,conductivity);
    Energy e=energy(ts,s,u,is_snow,tr,drag,ground,water,conductivity); float melt=0.0f;
    float soil_energy=0.0f;
    if (is_snow) melt=snow_heat(s,count,u[0],e.g,&soil_energy,conductivity);
    else s[30]=fmaxf(0.0f,s[30]-e.ecan);
    float et, runoff=soil_water(s,u,through+melt,tr_m,e.esoil,&et,permeability,supercool,surface,subsurface);
    soil_heat(s,u[0],is_snow ? soil_energy/u[0] : e.g,supercool);
    s[25]=ts; s[26]=s[16]+s[17]+s[18]; s[27]=s[26]>1.0e-8f ? s[19] : TF;
    snow_albedo(s,snow,u[0],albedo);
    diag[0]=ts; diag[1]=e.rn; diag[2]=e.h; diag[3]=e.le;
    diag[4]=e.g;
    diag[5]=(et+e.ecan)*1000.0f; diag[6]=u[7]*u[0]*1000.0f;
    diag[7]=runoff*1000.0f; diag[8]=melt*1000.0f; diag[9]=s[26]; diag[10]=s[30]; diag[11]=e.residual;
}
// 对应普通版: driver.step_column。Keep the default probe entry point and tree.
__device__ void step(float* s, int* count, const float* u, float* diag) {
    step_selected<true>(s,count,u,diag,nullptr);
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
        int opts[32];
        for(int j=0;j<32;++j) opts[j]=options_soa[j*n_columns+column];
        int count=options_soa[32*n_columns+column];
        bool defaults=opts[1]==1 && opts[2]==1 && opts[3]==4 && opts[4]==1 && opts[5]==1
            && opts[6]==1 && opts[9]==1 && opts[11]==1 && opts[12]==3 && opts[13]==3
            && opts[14]==1 && opts[23]==2;
        if(defaults) step(s,&count,u,d);
        else step_selected<false>(s,&count,u,d,opts);
        for (int j=0;j<70;++j) state_out_soa[j*n_columns+column]=s[j];
        for (int j=0;j<33;++j) options_out_soa[j*n_columns+column]=j==32 ? count : options_soa[j*n_columns+column];
        for (int j=0;j<12;++j) diag_soa[j*n_columns+column]=d[j];
    }
}
