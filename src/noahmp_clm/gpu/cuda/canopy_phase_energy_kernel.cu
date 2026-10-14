// Coupled extensive canopy phase state. Every Newton operation runs on GPU.
// Uses the validated leaf and radiation kernels, with deterministic reductions.
struct PhaseContext {
    const float *leaf,*u,*f,*soil,*c,*b,*p,*rad;
    float area[10],fw[10],dry[10],vc[10],weights[11],sw[11],lw[132];
    float fv,rho,mol,atm,bare,hden,trlimit,lg;
};
__device__ void phase_prepare(const float* leaf,const float* u,const float* f,const float* soil,
    const float* stem,const float* c,const float* b,const float* p,const float* rad,PhaseContext& v) {
    v.leaf=leaf;v.u=u;v.f=f;v.soil=soil;v.c=c;v.b=b;v.p=p;v.rad=rad;
    for(int i=0;i<10;++i)v.area[i]=leaf[i]+stem[i];
    v.fv=ca_sum(v.area,10)>0.f?u[13]:0.f;v.rho=u[2]/((287.f*u[0])*(1.f+.61f*u[1]));v.mol=u[2]/(8.314462618f*u[0]);
    v.atm=((v.rho*1004.f)*v.fv)*u[3];v.bare=((v.rho*1004.f)*(1.f-v.fv))*u[3];
    for(int i=0;i<10;++i) {
        v.weights[i]=(v.rho*1004.f)*((v.fv*v.area[i])*u[5]);v.vc[i]=((v.rho*v.fv)*v.area[i])*u[5];
        float cap=(.1f*v.fv)*v.area[i];v.fw[i]=cap>0.f?fminf(1.f,powf(c[2*i]/cap,2.f/3.f)):0.f;
        v.dry[i]=(1.f-v.fw[i])*(leaf[i]/(v.area[i]>0.f?v.area[i]:1.f));v.sw[i]=rad[i];
    }
    v.weights[10]=(v.rho*1004.f)*(v.fv*u[4]);v.hden=ca_sum(v.weights,11)+v.atm;
    v.sw[10]=ca_sum(rad+20,5)+rad[25];ca_longwave(v.area,u[18],v.fv,v.lw);
    v.trlimit=soil[0]>0.f?fminf(u[16],ca_root_capacity(soil)):0.f;v.lg=2.5e6f+(b[2]>0.f?334000.f:0.f);
}

__device__ __noinline__ void phase_evaluate(const PhaseContext& v,const float* z,float* residual,float* out) {
    const float *u=v.u,*b=v.b,*c=v.c;float t[11],thermal[10],qs[11],qi[11],es[11],gas[120];
    float mass[10],heat[10],adv[10],vapor[10],iv[10],liq[10],wetle[10],unmet[10],wet[10],tr[10],take[4];
    const float plateau=79.90430622009569f;float q=z[11]*.001f;
    for(int i=0;i<10;++i) {
        t[i]=c[2*i]>0.f?273.15f+((fminf(z[i],0.f)*4180.f/2100.f)+fmaxf(z[i]-plateau,0.f)):273.15f+z[i];
        thermal[i]=(c[2*i]*4180.f)*z[i];
    }
    t[10]=z[10];
    for(int i=0;i<11;++i) {
        float tc=t[i]-273.15f;es[i]=611.2f*expf((17.67f*tc)/(tc+243.5f));qs[i]=(.622f*es[i])/(u[2]-.378f*es[i]);
        float esi=expf(((9.550426f-5723.265f/t[i])+3.53068f*logf(t[i]))-.00728332f*t[i]);qi[i]=(.622f*esi)/(u[2]-.378f*esi);
    }
    for(int i=0;i<10;++i) {
        float m=c[2*i],frac=m>0.f?fminf(m,fmaxf(0.f,thermal[i]/334000.f))/m:(t[i]>273.15f?1.f:0.f);
        float sat=frac*qs[i]+(1.f-frac)*qi[i],rh=((q*u[2])/(.622f+.378f*q))/es[i];
        float input[7]={v.rad[10+i],t[i],u[8],rh,u[5]*v.mol,v.f[2*i]*(v.trlimit>0.f?v.f[2*i+1]:0.f),v.f[2*i+1]};
        leaf_gas_exchange_device(input,v.p,gas+12*i);
        float amount=b[4]*((v.vc[i]*(sat>q?v.fw[i]:1.f))*(sat-q));
        float loss=fminf(fmaxf(amount,0.f),m),deposit=fmaxf(-amount,0.f),remaining=m-loss;
        float retained=(remaining/(m>0.f?m:1.f))*thermal[i];
        float added=deposit*((1.f-frac)*(2100.f*(t[i]-273.15f))+frac*(334000.f+4180.f*(t[i]-273.15f)));
        mass[i]=remaining+deposit;heat[i]=retained+added;adv[i]=(thermal[i]-retained)-added;
        vapor[i]=loss-deposit;liq[i]=vapor[i]*frac;iv[i]=vapor[i]-liq[i];wetle[i]=2.5e6f*vapor[i]+334000.f*iv[i];
        wet[i]=vapor[i]/b[4];unmet[i]=fmaxf(amount-loss,0.f);
        tr[i]=(((v.vc[i]*v.dry[i])*gas[12*i+9])/(u[5]*v.mol+gas[12*i+9]))*fmaxf(qs[i]-q,0.f);
    }
    float total=ca_sum(tr,10),scale=total>0.f?fminf(1.f,v.trlimit/fmaxf(total,1e-30f)):1.f;
    for(int i=0;i<10;++i)tr[i]*=scale;
    ca_allocate(ca_sum(tr,10),v.soil,take);if(v.soil[0]==0.f)for(int i=0;i<4;++i)take[i]=0.f;
    float limit=v.soil[0]>0.f&&b[3]>0.f?fminf(u[17],fmaxf(0.f,v.soil[9]-take[0])):u[17];
    float gqs=b[2]>0.f?qi[10]:qs[10];
    float ge[2]={(v.rho*(v.fv*(gqs-q)))/(1.f/u[4]+u[9]),(v.rho*((1.f-v.fv)*(gqs-u[1])))/(1.f/u[3]+u[9])};
    scale=ca_scale(ge,2,limit);for(int i=0;i<2;++i)ge[i]=fminf(ge[i],0.f)+fmaxf(ge[i],0.f)*scale;
    float offset=0.f;for(int i=0;i<11;++i)offset+=v.weights[i]*(t[i]-u[0]);offset=v.hden>0.f?offset/fmaxf(v.hden,1e-30f):0.f;
    float sensible[11],emission[12],rn[11],latent[11],raw[11],storage[10],trheat[10],physical[11];
    float taken=ca_sum(take,4),root=taken>0.f?ca_dot(take,b+5,4)/fmaxf(taken,1e-30f):0.f;
    for(int i=0;i<11;++i){sensible[i]=v.weights[i]*((t[i]-u[0])-offset);emission[i]=5.670374419e-8f*((t[i]*t[i])*(t[i]*t[i]));}
    sensible[10]+=v.bare*(t[10]-u[0]);emission[11]=u[12];
    for(int i=0;i<11;++i) {
        rn[i]=v.sw[i]+ca_dot(v.lw+12*i,emission,12);
        latent[i]=i<10?wetle[i]/b[4]+2.5e6f*tr[i]:v.lg*ca_sum(ge,2);
        raw[i]=(rn[i]-sensible[i])-latent[i];
    }
    float gh=b[0]>0.f?raw[10]:u[7]*(t[10]-u[6])+u[10];
    for(int i=0;i<10;++i) {
        trheat[i]=tr[i]*((334000.f+4180.f*(t[i]-273.15f))-root);storage[i]=(heat[i]-c[2*i+1])/b[4];
        physical[i]=raw[i]-((storage[i]+adv[i]/b[4])+trheat[i]);
        residual[i]=v.fv*v.area[i]>0.f?raw[i]-((thermal[i]-c[2*i+1])/b[4]+trheat[i]):u[0]-t[i];
    }
    physical[10]=raw[10]-gh;residual[10]=b[0]>0.f?b[1]-t[10]:raw[10]-gh;
    float airwater=((ca_sum(wet,10)+ca_sum(tr,10))+ge[0])-((v.rho*v.fv)*u[3])*(q-u[1]);
    residual[11]=v.fv>0.f?airwater*2.5e6f:(q-u[1])*1e5f;
    if(!out)return;
    for(int i=0;i<300;++i)out[i]=0.f;
    for(int i=0;i<10;++i) {
        out[i]=t[i];out[15+i]=sensible[i];out[30+i]=wet[i];out[40+i]=tr[i];
        out[195+i]=mass[i];out[205+i]=heat[i];out[215+i]=thermal[i];out[225+i]=iv[i];out[235+i]=liq[i];
        out[245+i]=adv[i];out[255+i]=wetle[i];out[265+i]=unmet[i];out[275+i]=z[i];
        out[13]+=v.leaf[i]*(v.fv*v.area[i]>0.f?gas[12*i]:0.f);out[14]+=v.leaf[i]*(v.fv*v.area[i]>0.f?gas[12*i+5]:0.f);
        out[69]+=(mass[i]-c[2*i])+vapor[i];
    }
    out[13]*=v.fv;out[14]*=v.fv;out[10]=t[10];out[11]=u[0]+offset;out[12]=q;out[25]=sensible[10];
    out[26]=ca_sum(sensible,11);out[27]=ca_sum(latent,11);out[28]=gh;out[29]=ca_sum(rn,11);out[50]=ca_sum(ge,2);
    for(int i=0;i<4;++i)out[51+i]=take[i];for(int i=0;i<11;++i)out[55+i]=physical[i];out[66]=ca_sum(physical,11);
    out[71]=(ca_sum(sensible,11)-v.atm*offset)-v.bare*(t[10]-u[0]);out[72]=airwater;out[73]=v.rad[26];
    float lwtotal=0.f;for(int i=0;i<11;++i)lwtotal+=rn[i]-v.sw[i];out[74]=u[12]-lwtotal;
    for(int i=0;i<120;++i)out[75+i]=gas[i];
    out[285]=ca_sum(storage,10);out[286]=ca_sum(adv,10)/b[4];
    for(int i=0;i<10;++i)out[287]+=tr[i]*(334000.f+4180.f*(t[i]-273.15f));out[288]=gh-v.sw[10];
    out[289]=v.rad[25];out[290]=v.lg;for(int i=0;i<12;++i)out[292]=fmaxf(out[292],fabsf(residual[i]));
    for(int i=0;i<5;++i)out[293+i]=v.rad[20+i];
    out[298]=ca_dot(take,b+5,4);out[299]=ca_sum(trheat,10);
}

__device__ void phase_linear(float* a,float* b,float* x) {
    for(int k=0;k<12;++k) {
        int pivot=k;for(int i=k+1;i<12;++i)if(fabsf(a[i*12+k])>fabsf(a[pivot*12+k]))pivot=i;
        for(int i=0;i<12;++i){float old=a[k*12+i];a[k*12+i]=a[pivot*12+i];a[pivot*12+i]=old;}
        float old=b[k];b[k]=b[pivot];b[pivot]=old;
        for(int i=k+1;i<12;++i) {
            float factor=a[i*12+k]/a[k*12+k];for(int j=k+1;j<12;++j)a[i*12+j]-=factor*a[k*12+j];b[i]-=factor*b[k];
        }
    }
    for(int i=11;i>=0;--i){float value=b[i];for(int k=i+1;k<12;++k)value-=a[i*12+k]*x[k];x[i]=value/a[i*12+i];}
}
__device__ float phase_max(const float* v,int n){float x=0.f;for(int i=0;i<n;++i)x=fmaxf(x,fabsf(v[i]));return x;}

__device__ void phase_solve(const PhaseContext& v,float* out,int max_iterations,float tolerance) {
    float x[12],ambient[12],alternative[12],lower[12],upper[12],r[12],ra[12];const float plateau=79.90430622009569f;
    for(int i=0;i<10;++i) {
        float mass=v.c[2*i];x[i]=mass>0.f?v.c[2*i+1]/(mass*4180.f):v.u[0]-273.15f;
        ambient[i]=mass>0.f?(v.u[0]<273.15f?(v.u[0]-273.15f)*2100.f/4180.f:plateau+(v.u[0]-273.15f)):v.u[0]-273.15f;
        if(v.u[0]==273.15f)ambient[i]=x[i];
        lower[i]=mass>0.f?-36.75f:-73.15f;upper[i]=mass>0.f?(plateau+350.f)-273.15f:76.85f;
    }
    x[10]=v.b[0]>0.f?v.b[1]:v.u[0];x[11]=v.u[1]*1000.f;ambient[10]=x[10];ambient[11]=x[11];
    lower[10]=200.f;lower[11]=0.f;upper[10]=350.f;upper[11]=190.f;
    phase_evaluate(v,x,r,nullptr);phase_evaluate(v,ambient,ra,nullptr);bool choose=ca_dot(ra,ra,12)<ca_dot(r,r,12),available=false;
    for(int i=0;i<12;++i){alternative[i]=choose?x[i]:ambient[i];x[i]=choose?ambient[i]:x[i];available|=x[i]!=alternative[i];}
    int iteration=0,restarts=0;bool failed=false;
    while(iteration<max_iterations&&!failed) {
        phase_evaluate(v,x,r,out);if(phase_max(r,12)<=tolerance)break;
        for(int i=0;i<10;++i)if(v.c[2*i]>0.f&&out[195+i]==0.f&&x[i]>0.f&&x[i]<plateau&&fabsf(r[i])>tolerance)x[i]=r[i]>0.f?plateau:0.f;
        phase_evaluate(v,x,r,nullptr);float jac[144],delta[12],rhs[12];
        for(int k=0;k<12;++k) {
            float lo[12],hi[12],rl[12],rh[12],step=k==11?.002f:.02f;
            for(int i=0;i<12;++i){lo[i]=x[i];hi[i]=x[i];}
            lo[k]=fmaxf(lower[k],x[k]-step);hi[k]=fminf(upper[k],x[k]+step);
            phase_evaluate(v,lo,rl,nullptr);phase_evaluate(v,hi,rh,nullptr);
            for(int i=0;i<12;++i)jac[i*12+k]=(rh[i]-rl[i])/(hi[k]-lo[k]);
        }
        for(int i=0;i<12;++i)rhs[i]=-r[i];phase_linear(jac,rhs,delta);
        float scale=1.f;for(int i=0;i<12;++i)scale=fmaxf(scale,fabsf(delta[i])/(i<10?20.f:5.f));
        for(int i=0;i<12;++i)delta[i]/=scale;
        float norm=ca_dot(r,r,12),chosen[12];bool accepted=false;
        for(int i=0;i<12;++i)chosen[i]=x[i];
        for(int k=0;k<24;++k) {
            float trial[12],rr[12];for(int i=0;i<12;++i)trial[i]=fminf(upper[i],fmaxf(lower[i],x[i]+delta[i]*ldexpf(1.f,-k)));
            phase_evaluate(v,trial,rr,nullptr);
            if(!accepted&&ca_dot(rr,rr,12)<norm){for(int i=0;i<12;++i)chosen[i]=trial[i];accepted=true;}
        }
        bool retry=!accepted&&available;
        for(int i=0;i<12;++i)x[i]=retry?alternative[i]:chosen[i];
        if(retry){available=false;++restarts;}failed=!(accepted||retry);++iteration;
    }
    phase_evaluate(v,x,r,out);out[67]=iteration+1;out[68]=!failed&&phase_max(r,12)<=tolerance&&phase_max(out+55,11)<=tolerance*1.05f;
    out[291]=restarts;
}
extern "C" __global__ void run_canopy_phase(const float* leaf,const float* u,const float* f,const float* soil,
    const float* stem,const float* c,const float* b,const float* radiation,const float* p,float* out,int n,int max_iterations,float tolerance) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x) {
        PhaseContext v;phase_prepare(leaf+row*10,u+row*23,f+row*20,soil+row*10,stem+row*10,c+row*20,b+row*9,p,radiation+row*584,v);
        phase_solve(v,out+row*300,max_iterations,tolerance);
    }
}
