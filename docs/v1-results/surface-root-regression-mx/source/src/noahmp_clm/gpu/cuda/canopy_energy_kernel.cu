// Ten leaf temperatures + ground, one diagnostic air box, four root layers.
// Compiled with leaf_kernel.cu and radiation_kernel.cu in the same translation unit.
__device__ float ca_sum(const float* v,int n) {
    float s=0.f;for(int j=0;j<n;++j)s+=v[j];return s;
}
__device__ float ca_dot(const float* a,const float* b,int n) {
    float s=0.f;for(int j=0;j<n;++j)s+=a[j]*b[j];return s;
}
__device__ float ca_root_capacity(const float* soil) {
    float total=0.f;for(int j=0;j<4;++j)total+=soil[5+j]>0.f?soil[1+j]:0.f;return total;
}
__device__ void ca_allocate(float demand,const float* soil,float* take) {
    float target=fminf(demand,ca_root_capacity(soil));
    for(int j=0;j<4;++j)take[j]=0.f;
    for(int iter=0;iter<4;++iter) {
        float remaining=fmaxf(0.f,target-ca_sum(take,4)),active[4];
        for(int j=0;j<4;++j)active[j]=take[j]<soil[1+j]?soil[5+j]:0.f;
        float denom=ca_sum(active,4);
        for(int j=0;j<4;++j)take[j]+=fminf(remaining*active[j]/fmaxf(denom,1e-30f),soil[1+j]-take[j]);
    }
}
__device__ float ca_scale(const float* v,int n,float limit) {
    float pos=0.f,neg=0.f;
    for(int j=0;j<n;++j) {pos+=fmaxf(v[j],0.f);neg+=fminf(v[j],0.f);}
    return pos>0.f?fminf(1.f,(limit-neg)/fmaxf(pos,1e-30f)):1.f;
}
struct CanopyWater {
    float wet[10],tr[10],kleaf[10],eg,eb,kunder,kbare,residual,uptake[4];
};
__device__ void ca_water(float q,const float* qs,const float* gs,const float* area,
                         float fv,float rho,const float* u,const float* soil,CanopyWater& w,bool shares=true) {
    float kw[10],kt[10],raw_wet[10],raw_tr[10];
    for(int j=0;j<10;++j) {
        float boundary=rho*fv*area[j]*u[5];
        bool drying=qs[j]>q;
        kw[j]=boundary*(drying?u[14]:1.f);
        kt[j]=boundary*(drying?1.f-u[14]:0.f)*gs[j]/(u[5]+gs[j]);
        raw_wet[j]=kw[j]*(qs[j]-q);
        raw_tr[j]=kt[j]*(qs[j]-q);
    }
    float wet_scale=ca_scale(raw_wet,10,u[15]);
    float tr_limit=soil[0]>0.f?fminf(u[16],ca_root_capacity(soil)):u[16];
    float tr_scale=ca_scale(raw_tr,10,tr_limit);
    for(int j=0;j<10;++j) {
        kw[j]*=raw_wet[j]>0.f?wet_scale:1.f;
        kt[j]*=tr_scale;
        w.tr[j]=kt[j]*(qs[j]-q);
    }
    ca_allocate(ca_sum(w.tr,10),soil,w.uptake);
    if(soil[0]==0.f)for(int j=0;j<4;++j)w.uptake[j]=0.f;
    float glimit=soil[0]>0.f&&shares?fminf(u[17],fmaxf(0.f,soil[9]-w.uptake[0])):u[17];
    float under=rho*fv/(1.f/u[4]+u[9]),bare=rho*(1.f-fv)/(1.f/u[3]+u[9]);
    float ground[2]={under*(qs[10]-q),bare*(qs[10]-u[1])};
    float ground_scale=ca_scale(ground,2,glimit);
    under*=ground[0]>0.f?ground_scale:1.f;
    bare*=ground[1]>0.f?ground_scale:1.f;
    for(int j=0;j<10;++j){w.wet[j]=kw[j]*(qs[j]-q);w.kleaf[j]=kw[j]+kt[j];}
    w.eg=under*(qs[10]-q);w.eb=bare*(qs[10]-u[1]);w.kunder=under;w.kbare=bare;
    w.residual=ca_sum(w.wet,10)+ca_sum(w.tr,10)+w.eg-rho*fv*u[3]*(q-u[1]);
}
__device__ void ca_longwave(const float* area,float eps,float fv,float* matrix) {
    float down[132],up[132],t[10];
    for(int j=0;j<132;++j){down[j]=0.f;up[j]=0.f;}
    down[11]=1.f;
    for(int j=0;j<10;++j) {
        t[j]=expf(-area[j]);
        for(int k=0;k<12;++k)down[(j+1)*12+k]=t[j]*down[j*12+k];
        down[(j+1)*12+j]+=1.f-t[j];
    }
    for(int k=0;k<12;++k)up[120+k]=(1.f-eps)*down[120+k];
    up[130]+=eps;
    for(int j=9;j>=0;--j) {
        for(int k=0;k<12;++k)up[j*12+k]=t[j]*up[(j+1)*12+k];
        up[j*12+j]+=1.f-t[j];
    }
    for(int j=0;j<10;++j)for(int k=0;k<12;++k)
        matrix[j*12+k]=fv*(down[j*12+k]+up[(j+1)*12+k]-down[(j+1)*12+k]-up[j*12+k]);
    for(int k=0;k<12;++k)matrix[120+k]=fv*(down[120+k]-up[120+k]);
    matrix[130]+=-(1.f-fv)*eps;matrix[131]+=(1.f-fv)*eps;
}
__device__ void ca_linear_solve(float* a,float* b,float* x) {
    for(int k=0;k<11;++k) {
        int pivot=k;
        for(int j=k+1;j<11;++j)if(fabsf(a[j*11+k])>fabsf(a[pivot*11+k]))pivot=j;
        for(int j=0;j<11;++j){float old=a[k*11+j];a[k*11+j]=a[pivot*11+j];a[pivot*11+j]=old;}
        float old=b[k];b[k]=b[pivot];b[pivot]=old;
        for(int j=k+1;j<11;++j) {
            float factor=a[j*11+k]/a[k*11+k];
            for(int i=k+1;i<11;++i)a[j*11+i]-=factor*a[k*11+i];
            b[j]-=factor*b[k];
        }
    }
    for(int j=10;j>=0;--j) {
        float value=b[j];for(int k=j+1;k<11;++k)value-=a[j*11+k]*x[k];
        x[j]=value/a[j*11+j];
    }
}
__device__ void canopy_energy_device(const float* area,const float* u,const float* f,
    const float* soil,const float* initial,const float* p,float* out,
    int max_iterations,float tolerance,float q_tolerance,const float* boundary=nullptr,const float* radiation=nullptr) {
        float fv=ca_sum(area,10)>0.f?u[13]:0.f;
        bool active[10];for(int j=0;j<10;++j)active[j]=fv*area[j]>0.f;
        float bands[2][46];
        for(int i=0;i<2;++i) {
            float optics[6]={u[11]*.5f*(1.f-u[21]),u[11]*.5f*u[21],u[19+i],u[22],
                i==0?.1f:.45f,i==0?.05f:.25f};
            shortwave_device(area,optics,bands[i]);
        }
        float bare_sw=u[11]*(1.f-.5f*(u[19]+u[20]));
        float sw[11],par[10],lw[132];
        for(int j=0;j<10;++j) {
            sw[j]=fv*(bands[0][j]+bands[1][j]);
            par[j]=area[j]>0.f?bands[0][j]*4.6f/fmaxf(area[j],1e-30f):0.f;
        }
        sw[10]=fv*(bands[0][44]+bands[1][44])+(1.f-fv)*bare_sw;
        float outgoing_sw=fv*(bands[0][43]+bands[1][43])+(1.f-fv)*(u[11]-bare_sw);
        float latent[11],weights[10];
        for(int j=0;j<11;++j)latent[j]=2.5e6f;
        for(int j=0;j<10;++j)weights[j]=area[j]/fmaxf(ca_sum(area,10),1e-30f);
        if(boundary) {
            for(int j=0;j<10;++j){sw[j]=radiation[j];par[j]=radiation[10+j];}
            sw[10]=ca_sum(radiation+20,5)+radiation[25];outgoing_sw=radiation[26];
            if(boundary[1]>0.f)latent[10]+=334000.f;
        }
        ca_longwave(area,u[18],fv,lw);
        float rho=u[2]/(287.f*u[0]*(1.f+.61f*u[1]));
        float hcan[11],hjac[121];
        for(int j=0;j<10;++j)hcan[j]=rho*1004.f*(fv*area[j]*u[5]);
        hcan[10]=rho*1004.f*(fv*u[4]);
        float hatm=rho*1004.f*fv*u[3],hbare=rho*1004.f*(1.f-fv)*u[3];
        float hden=ca_sum(hcan,11)+hatm;
        for(int j=0;j<11;++j)for(int k=0;k<11;++k)
            hjac[j*11+k]=(j==k?hcan[j]:0.f)-(hden>0.f?hcan[j]*hcan[k]/fmaxf(hden,1e-30f):0.f);
        hjac[120]+=hbare;
        float mol_density=u[2]/(8.314462618f*u[0]),temp[11],qcan=u[1];
        for(int j=0;j<11;++j)temp[j]=(j<10&&!active[j])?u[0]:initial[j];
        if(boundary)temp[10]=boundary[0];
        bool dry=u[16]==0.f||(soil[0]>0.f&&ca_root_capacity(soil)==0.f);
        for(int count=0;count<max_iterations;++count) {
            float qs[11],dq[11],es[11],gas[120],gs[10];
            for(int j=0;j<11;++j) {
                float tc=temp[j]-273.15f;
                es[j]=611.2f*expf(17.67f*tc/(tc+243.5f));
                float den=u[2]-.378f*es[j];
                qs[j]=.622f*es[j]/den;
                dq[j]=.622f*u[2]*es[j]*17.67f*243.5f/(den*den*(tc+243.5f)*(tc+243.5f));
            }
            if(boundary&&boundary[1]>0.f) {
                float t=temp[10];
                es[10]=expf(9.550426f-5723.265f/t+3.53068f*logf(t)-.00728332f*t);
                float slope=5723.265f/(t*t)+3.53068f/t-.00728332f,den=u[2]-.378f*es[10];
                qs[10]=.622f*es[10]/den;dq[10]=.622f*u[2]*es[10]*slope/(den*den);
            }
            float vapor=qcan*u[2]/(.622f+.378f*qcan);
            for(int j=0;j<10;++j) {
                float leaf[7]={par[j],temp[j],u[8],vapor/es[j],u[5]*mol_density,
                    f[j*2],dry?0.f:f[j*2+1]};
                leaf_gas_exchange_device(leaf,p,gas+j*12);
                gs[j]=gas[j*12+9]/mol_density;
            }
            float left=u[1],right=u[1];
            for(int j=0;j<11;++j){left=fminf(left,qs[j]);right=fmaxf(right,qs[j]);}
            CanopyWater water;
            for(int iter=0;iter<42;++iter) {
                float middle=.5f*(left+right);
                ca_water(middle,qs,gs,area,fv,rho,u,soil,water,boundary?boundary[2]>0.f:true);
                if(water.residual>0.f)left=middle;else right=middle;
            }
            float qnew=fv>0.f?.5f*(left+right):u[1];
            ca_water(qnew,qs,gs,area,fv,rho,u,soil,water,boundary?boundary[2]>0.f:true);
            float relative_temp[11];for(int j=0;j<11;++j)relative_temp[j]=temp[j]-u[0];
            float air_offset=hden>0.f?ca_dot(hcan,relative_temp,11)/fmaxf(hden,1e-30f):0.f;
            float tair=u[0]+air_offset;
            float sensible[11],evaporation[11],net[11],blackbody[12],residual[11];
            for(int j=0;j<11;++j) {
                sensible[j]=hcan[j]*((temp[j]-u[0])-air_offset);
                float square=temp[j]*temp[j];blackbody[j]=5.670374419e-8f*(square*square);
            }
            blackbody[11]=u[12];sensible[10]+=hbare*(temp[10]-u[0]);
            for(int j=0;j<10;++j)evaporation[j]=water.wet[j]+water.tr[j];
            evaporation[10]=water.eg+water.eb;
            float ground=u[7]*(temp[10]-u[6])+u[10];
            for(int j=0;j<11;++j) {
                net[j]=sw[j]+ca_dot(lw+j*12,blackbody,12);
                residual[j]=net[j]-sensible[j]-2.5e6f*evaporation[j];
            }
            residual[10]+=-ground;
            bool enabled=boundary&&boundary[3]>0.f;
            float mass=0.f,enthalpy=0.f,storage=0.f,wet_adv=0.f,tr_adv=0.f;
            if(boundary) {
                for(int j=0;j<11;++j)residual[j]=net[j]-sensible[j]-latent[j]*evaporation[j];
                ground=residual[10];residual[10]=0.f;
                if(enabled) {
                    mass=fmaxf(0.f,boundary[4]-boundary[6]*ca_sum(water.wet,10));
                    float heat[10];for(int j=0;j<10;++j)heat[j]=334000.f+4180.f*(temp[j]-273.15f);
                    enthalpy=mass*ca_dot(weights,heat,10);storage=(enthalpy-boundary[5])/boundary[6];
                    float uptake=ca_sum(water.uptake,4);
                    float root_mean=uptake>0.f?ca_dot(water.uptake,boundary+7,4)/fmaxf(uptake,1e-30f):0.f;
                    wet_adv=ca_dot(water.wet,heat,10);tr_adv=ca_dot(water.tr,heat,10);
                    for(int j=0;j<10;++j)residual[j]-=weights[j]*storage+water.wet[j]*heat[j]+water.tr[j]*(heat[j]-root_mean);
                }
            }
            float maxres=0.f;for(int j=0;j<11;++j)maxres=fmaxf(maxres,fabsf(residual[j]));
            float qerror=fabsf(qnew-qcan);
            bool energy_balanced=maxres<=tolerance;
            bool converged=energy_balanced&&qerror<=q_tolerance;
            float h=ca_sum(sensible,11),evap=ca_sum(evaporation,11),gross[10],resp[10],netlw[11];
            for(int j=0;j<10;++j){gross[j]=active[j]?gas[j*12]:0.f;resp[j]=active[j]?gas[j*12+5]:0.f;}
            for(int j=0;j<11;++j){out[j]=temp[j];netlw[j]=net[j]-sw[j];}
            out[11]=tair;out[12]=qnew;out[13]=fv*ca_dot(area,gross,10);out[14]=fv*ca_dot(area,resp,10);
            for(int j=0;j<10;++j){out[15+j]=sensible[j];out[30+j]=water.wet[j];out[40+j]=water.tr[j];}
            out[25]=sensible[10];out[26]=h;out[27]=2.5e6f*evap;out[28]=ground;out[29]=ca_sum(net,11);
            out[50]=water.eg+water.eb;for(int j=0;j<4;++j)out[51+j]=water.uptake[j];
            for(int j=0;j<11;++j)out[55+j]=residual[j];
            out[66]=ca_sum(residual,11);out[67]=(float)(count+1);out[68]=converged?1.f:0.f;
            out[69]=water.residual;out[70]=qerror;
            out[71]=h-(hatm*air_offset+hbare*(temp[10]-u[0]));
            out[72]=evap-(rho*fv*u[3]*(qnew-u[1])+water.eb);
            out[73]=outgoing_sw;out[74]=u[12]-ca_sum(netlw,11);
            for(int j=0;j<120;++j)out[75+j]=gas[j];
            if(boundary) {
                out[27]=ca_dot(latent,evaporation,11);out[195]=ground-sw[10];
                for(int j=0;j<6;++j)out[196+j]=radiation[20+j];
                out[202]=latent[10];out[203]=storage;out[204]=mass;out[205]=enthalpy;out[206]=wet_adv;out[207]=tr_adv;
            }
            if(converged)break;
            if(energy_balanced){qcan=qnew;continue;}
            float kcan[11],kd[11],jac[121],rhs[11],delta[11];
            for(int j=0;j<10;++j)kcan[j]=water.kleaf[j];kcan[10]=water.kunder;
            for(int j=0;j<11;++j)kd[j]=kcan[j]*dq[j];
            float kden=ca_sum(kcan,11)+rho*fv*u[3];
            for(int j=0;j<11;++j)for(int k=0;k<11;++k) {
                float ejac=(j==k?kd[j]:0.f)-(kden>0.f?kcan[j]*kd[k]/fmaxf(kden,1e-30f):0.f);
                if(j==10&&k==10)ejac+=water.kbare*dq[10];
                jac[j*11+k]=lw[j*12+k]*((float)(4.*5.670374419e-8)*(temp[k]*temp[k]*temp[k]))-hjac[j*11+k]-2.5e6f*ejac;
                if(j==10&&k==10)jac[j*11+k]+=-u[7];
                if(boundary) {
                    jac[j*11+k]=lw[j*12+k]*((float)(4.*5.670374419e-8)*(temp[k]*temp[k]*temp[k]))-hjac[j*11+k]-latent[j]*ejac;
                    if(enabled&&j<10&&k<10)jac[j*11+k]-=4180.f*mass/boundary[6]*(weights[j]*weights[k]);
                    if(enabled&&j<10&&j==k)jac[j*11+k]-=4180.f*(water.wet[j]+water.tr[j]);
                }
                if(j<10&&!active[j])jac[j*11+k]=j==k?-1.f:0.f;
            }
            for(int j=0;j<11;++j)rhs[j]=-((j<10&&!active[j])?u[0]-temp[j]:residual[j]);
            if(boundary) {for(int k=0;k<11;++k)jac[110+k]=k==10?-1.f:0.f;rhs[10]=0.f;}
            ca_linear_solve(jac,rhs,delta);
            for(int j=0;j<11;++j)temp[j]=fminf(350.f,fmaxf(200.f,temp[j]+.8f*fminf(5.f,fmaxf(-5.f,delta[j]))));
            qcan=qnew;
        }
}

extern "C" __global__ void run_canopy_energy(const float* areas,const float* inputs,
    const float* factors,const float* soils,const float* initials,const float* p,float* outputs,
    int n,int max_iterations,float tolerance,float q_tolerance) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        canopy_energy_device(areas+row*10,inputs+row*23,factors+row*20,soils+row*10,initials+row*11,p,outputs+row*195,
            max_iterations,tolerance,q_tolerance);
}
extern "C" __global__ void run_canopy_boundary(const float* areas,const float* inputs,
    const float* factors,const float* soils,const float* initials,const float* boundaries,const float* radiation,
    const float* p,float* outputs,int n,int max_iterations,float tolerance,float q_tolerance) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        canopy_energy_device(areas+row*10,inputs+row*23,factors+row*20,soils+row*10,initials+row*11,p,outputs+row*208,
            max_iterations,tolerance,q_tolerance,boundaries+row*11,radiation+row*584);
}
