// Developing CLM driver. Helpers from column_kernel.cu remain shared; the new
// canopy solver runs in its own device launch with a checked convergence flag.
extern "C" __global__ void prepare_clm_columns(const float* states,const int* options,
    const float* forcing,const float* sites,float* areas,float* inputs,float* factors,
    float* soils,float* initials,float* prep,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x) {
        const float* s=states+row*70;const int* o=options+row*CLM_INT_SIZE;
        const float* f=forcing+row*8;const float* v=sites+row*5;
        float* a=areas+row*10;float* u=inputs+row*23;float* fac=factors+row*20;
        float* soil=soils+row*10;float* initial=initials+row*11;float* p=prep+row*4;
        float liq[4],weights[4],view[70];
        for(int j=0;j<70;++j)view[j]=s[j];
        for(int j=0;j<4;++j) {
            float ice=fmaxf(0.f,s[4+j]-liquid(s[4+j],s[j],s,o[CLM_OptSoilSupercoolWater]));
            liq[j]=fmaxf(s[4+j]-ice,0.f);view[4+j]=liq[j];
            weights[j]=s[12+j]*beta(liq[j],s,o[CLM_OptSoilWaterTranspiration]);
        }
        float stress=clip((weights[0]+weights[2])+(weights[1]+weights[3]),0.f,1.f);
        float rain=fmaxf(0.f,f[7])*f[0],capacity=.0005f*s[32]*s[31];
        float intercepted=s[31]>0.f?fminf(s[32]*rain,fmaxf(0.f,capacity-s[30])):0.f;
        float canw=s[30]+intercepted;
        float wet=capacity>0.f?fminf(1.f,powf(canw/fmaxf(capacity,1e-30f),(float)(2./3.))):0.f;
        float ce=exchange(f,s[33],s[25],o[CLM_OptSurfaceDrag]);
        float z0=clip(s[33],1e-5f,5.f),ustar=.4f*fmaxf(f[3],.3f)/logf(10.f/z0);
        float leaf=.01f*sqrtf(ustar/v[3]);
        float bare=(float)(.4/.13)*powf(z0*ustar/1.5e-5f,-.45f);
        float ground=(bare*expf(-s[31])+.004f*(1.f-expf(-s[31])))*ustar;
        float conduct=soil_k(s[4],s)/(.5f*DZ[0]);
        float resistance=ground_resistance(view,o[CLM_OptGroundResistanceEvap]);
        if(o[CLM_OptGroundResistanceEvap]==1||o[CLM_OptGroundResistanceEvap]==4) {
            // Here DZ is a compile-time JAX constant (unlike the old full
            // column graph). Match its folded 0.1/(e-1) coefficient locally.
            float sat=fmaxf(s[39],1e-6f),fraction=fminf(1.f,clip(liq[0],0.f,sat)/sat);
            float dry=(expf((1.f-fraction)*(1.f-fraction))-1.f)*(.1f*(float)(1./1.718281828459045));
            float diff=2.2e-5f*(sat*sat)*powf(fmaxf(1e-6f,1.f-s[40]/sat),2.f+3.f/fmaxf(s[43],1e-6f));
            resistance=fmaxf(1e-3f,dry/diff);
        }
        float values[23]={f[1],f[2],f[4],ce,ground,leaf,s[0],conduct,v[0],resistance,0.f,
            f[5],f[6],s[32],wet,canw*1000.f/f[0],__int_as_float(0x7f800000),__int_as_float(0x7f800000),s[35],s[34],s[34],v[2],v[1]};
        for(int j=0;j<23;++j)u[j]=values[j];
        for(int j=0;j<10;++j){a[j]=s[31]/10.f;fac[j*2]=v[4];fac[j*2+1]=stress;}
        for(int j=0;j<11;++j)initial[j]=f[1];
        soil[0]=1.f;
        for(int j=0;j<4;++j){soil[1+j]=fmaxf(liq[j]-s[40],0.f)*DZ[j]*1000.f/f[0];soil[5+j]=s[12+j];}
        soil[9]=liq[0]*DZ[0]*1000.f/f[0];
        p[0]=canw;p[1]=rain-intercepted;p[2]=capacity;p[3]=rain;
    }
}

extern "C" __global__ void finish_clm_columns(const float* states,const int* options,
    const float* forcing,const float* prepared,const float* canopy,float* outputs,float* diagnostics,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x) {
        const float* old=states+row*70;const int* o=options+row*CLM_INT_SIZE;
        const float* f=forcing+row*8;const float* p=prepared+row*4;const float* c=canopy+row*195;
        float s[70];for(int j=0;j<70;++j)s[j]=old[j];
        float ecan=ca_sum(c+30,10)*f[0]/1000.f,esoil=c[50]*f[0]/1000.f,roots[4];
        for(int j=0;j<4;++j)roots[j]=c[51+j]*f[0]/1000.f;
        float transp=ca_sum(roots,4),canw=p[0]-ecan;
        bool valid=canw>=-2e-8f;
        canw=fmaxf(0.f,canw);float drip=fmaxf(0.f,canw-p[2]);canw-=drip;
        float remaining=p[1]+drip;
        for(int j=0;j<4;++j) {
            float add=fminf(remaining,fmaxf(0.f,(s[39]-s[4+j])*DZ[j]));
            s[4+j]+=add/DZ[j];remaining-=add;
        }
        float runoff=fmaxf(0.f,remaining),ice[4],perm[4];
        for(int j=0;j<4;++j) {
            ice[j]=fmaxf(0.f,s[4+j]-liquid(s[4+j],s[j],s,o[CLM_OptSoilSupercoolWater]));
            valid=valid&&(roots[j]<=fmaxf(s[4+j]-ice[j]-s[40],0.f)*DZ[j]+2e-8f);
            s[4+j]-=roots[j]/DZ[j];
            perm[j]=o[CLM_OptSoilPermeabilityFrozen]==2?expf(-7.f*ice[j]/fmaxf(s[39],1e-6f)):1.f;
        }
        valid=valid&&(esoil<=fmaxf(0.f,s[4]-ice[0])*DZ[0]+2e-8f);
        s[4]-=esoil/DZ[0];
        for(int j=0;j<4;++j) {
            float q=s[42]*perm[j]*powf(clip(s[4+j]/s[39],0.f,1.f),2.f*s[43]+3.f);
            float drain=fminf(fmaxf(0.f,s[4+j]-s[41])*DZ[j],q*f[0]);
            s[4+j]-=drain/DZ[j];if(j<3)s[5+j]+=drain/DZ[j+1];else runoff+=drain;
        }
        for(int j=0;j<4;++j) {
            float excess=fmaxf(0.f,s[4+j]-s[39])*DZ[j];if(excess>0.f)s[4+j]=s[39];
            if(j<3)s[5+j]+=excess/DZ[j+1];else runoff+=excess;
        }
        Runoff routed=route_runoff(runoff,s,o[CLM_OptRunoffSurface],o[CLM_OptRunoffSubsurface]);
        runoff=routed.surface+routed.subsurface;
        soil_heat(s,f[0],c[28],o[CLM_OptSoilSupercoolWater]);s[25]=c[10];s[30]=canw;
        float et=ecan+transp+esoil;
        float storage0=ca_dot(old+4,DZ,4)+old[30],storage1=ca_dot(s+4,DZ,4)+canw;
        float d[22]={c[10],c[29],c[26],c[27],c[28],et*1000.f,p[3]*1000.f,runoff*1000.f,
            0.f,s[26],canw,c[66],c[13],c[13]*12.0107e-6f*f[0],c[14],c[11],c[12],
            storage1-storage0-p[3]+et+runoff,c[27]-2.5e6f*et*1000.f/f[0],c[67],c[68],valid?1.f:0.f};
        for(int j=0;j<70;++j)outputs[row*70+j]=s[j];
        for(int j=0;j<22;++j)diagnostics[row*22+j]=d[j];
    }
}
