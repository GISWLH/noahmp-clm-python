// Shared extensive surface coupling. The solver adapter preserves each canopy ABI.
struct LiquidSurfaceCanopy {
    static const int output_size=441;
    const float *a,*f,*leaf,*radiation,*v;
    __device__ void solve(float temp,const float* u,const float* supply,const float* root_h,int n,float* out) const {
        float initial[11],boundary[11]={temp,temp<273.15f?1.f:0.f,n==0?1.f:0.f,1.f,v[1],v[2],v[0]};
        for(int i=0;i<11;++i)initial[i]=u[0];for(int i=0;i<4;++i)boundary[7+i]=root_h[i];
        canopy_energy_device(a,u,f,supply,initial,leaf,out+58,100,.005f,1e-8f,boundary,radiation);
    }
    __device__ float latent(const float* out) const {
        return 2500000.f*(ca_sum(out+98,10)+ca_sum(out+88,10))+out[350];
    }
};
template<class Solver> __device__ void surface_evaluate(const float* c,const float* g,const float* p,int n,int phase,
    const float* a,const float* u,const float* f,const float* v,const float* radiation,const float* leaf,
    const float* supply,const float* root_h,bool inverse_ok,float temp,float* out,const Solver& solver) {
    float dt=v[0];solver.solve(temp,u,supply,root_h,n,out);
    float* canopy=out+58;
    float roots[4],carried[4],demand[7]={0.f,0.f,dt},tr[47];int operation[3]={1,phase,1};
    float gm=canopy[50]*dt,wet=ca_sum(canopy+30,10)*dt;
    for(int i=0;i<4;++i){roots[i]=canopy[51+i]*dt;carried[i]=roots[i]*root_h[i];demand[3+i]=roots[i];}
    if(n==0&&gm>0.f)demand[3]+=gm;
    column_soil_transport_device(g,p,operation,demand,tr);
    float ground[12];for(int i=0;i<12;++i)ground[i]=tr[i];
    float gh=ca_sum(tr+32,4)-ca_sum(carried,4),latent,runoff=0.f,runoff_h=0.f,residue[5]={0.f};
    float sw[5],soil_sw=canopy[201];for(int i=0;i<5;++i)sw[i]=canopy[196+i];
    int nn=n;bool valid=tr[46]>0.f;
    if(n>0) {
        float vapor[56],args[3]={gm,temp-273.15f,0.f};snow_vapor_device(c,n,args,vapor);
        nn=(int)vapor[45];for(int i=0;i<45;++i)out[i]=vapor[i];
        gm=vapor[46];gh=vapor[47];latent=vapor[55]/dt;
        for(int i=0;i<5;++i)residue[i]=vapor[48+i];
        if(nn<n){float removed=sw[0];for(int i=0;i<4;++i)sw[i]=sw[i+1];sw[4]=0.f;if(nn>0)sw[0]+=removed;else soil_sw+=removed;}
    } else {
        for(int i=0;i<45;++i)out[i]=c[i];
        float offset=temp-273.15f;
        if(gm<0.f)gh=gm*(temp<273.15f?2100.f*offset:334000.f+4180.f*offset);
        latent=((gm<0.f&&temp<273.15f)?2834000.f:2500000.f)*gm/dt;
        float args[7]={fmaxf(-gm,0.f),gm<0.f?-gh:0.f,dt,0.f,0.f,0.f,0.f};operation[0]=0;
        column_soil_transport_device(ground,p,operation,args,tr);valid=valid&&tr[46]>0.f;
        for(int i=0;i<12;++i)ground[i]=tr[i];runoff=tr[26];runoff_h=tr[27];
    }
    out[45]=(float)nn;
    for(int i=0;i<5;++i){float t,liquid;snow_mass_phase(out[9*i],out[9*i+1],t,out[343+i],liquid);}
    float top=(canopy[195]+canopy[202]*canopy[50])-latent;
    float cells[45]={0.f},props[4]={p[0],p[1],p[4],p[6]},bc[4]={dt,top,0.f,0.f};int options[2]={nn,phase};
    for(int k=0;k<9;++k) {
        if(k<nn){cells[k*5]=out[k*9];cells[k*5+1]=out[k*9+1];cells[k*5+2]=out[k*9+2];cells[k*5+4]=sw[k];}
        else if(k<nn+4){int j=k-nn;for(int i=0;i<3;++i)cells[k*5+i]=ground[j*3+i];if(j==0)cells[k*5+4]=soil_sw;}
        else {cells[k*5+2]=1.f;cells[k*5+3]=1.f;}
    }
    float* heat=out+266;joint_heat_device(cells,props,options,bc,heat);
    for(int k=0;k<5;++k)out[k*9+1]=k<nn?heat[k]:0.f;
    for(int k=0;k<4;++k)ground[k*3+1]=heat[nn+k];for(int i=0;i<12;++i)out[46+i]=ground[i];
    out[348]=gm;out[349]=gh;out[350]=latent;out[351]=wet;out[352]=ca_sum(roots,4);
    out[353]=runoff;out[354]=runoff_h;for(int i=0;i<5;++i)out[355+i]=residue[i];
    out[360]=temp;out[361]=heat[18]-(temp-273.15f);
    out[362]=valid&&canopy[68]>0.f&&heat[75]>0.f&&inverse_ok?1.f:0.f;
}

template<class Solver> __device__ void surface_finish(const float* old_snow,const float* old_soil,const float* v,
    float* out,int count,bool converged,const float* aging,const float* p,int phase,const Solver& solver,const float* vic=nullptr) {
    const float* d=out;const float* canopy=d+58;const float* heat=d+266;int n=(int)d[45];float dt=v[0];
    float z[6],centers[6],temperatures[6],args[23]={0.f};args[0]=dt;
    for(int i=0;i<6;++i){z[i]=i<n?d[9*i+2]:d[48];temperatures[i]=i<n?heat[18+i]:heat[18+n];}
    for(int i=0;i<6;++i){float sum=0.f;for(int k=0;k<6;++k)sum+=k<=i?z[k]:0.f;centers[i]=sum-.5f*z[i];}
    for(int i=0;i<5;++i) {
        int prev=max(i-1,0);float h1=i==0?1.f:centers[i]-centers[prev],h2=centers[i+1]-centers[i];
        float left=(temperatures[i]-temperatures[prev])/h1,right=(temperatures[i+1]-temperatures[i])/h2;
        args[1+i]=i<n?fabsf(i==0?right:(h2*left+h1*right)/(h1+h2)):0.f;
        args[6+i]=d[343+i];float offset,ice,liquid;snow_mass_phase(d[i*9],d[i*9+1],offset,ice,liquid);
        args[11+i]=fmaxf(0.f,ice-d[343+i]);
    }
    int opt[5]={n,3,1,1,0};float evolved[73];snow_evolution_device(d,opt,args,aging,evolved);
    float drainargs[16]={0.f,0.f,100.f,54.5f,.033f,.05f,.2f,.02f,.02f,.01f,.01f,0.f,0.f,0.f,0.f,0.f},drain[74];
    snow_transfer_device(evolved,n,3,drainargs,drain);
    for(int i=0;i<45;++i)out[363+i]=drain[i];out[408]=drain[45];
    float admitargs[7]={drain[53],drain[54],dt,0.f,0.f,0.f,0.f},admission[47];int so[3]={0,phase,1};
    column_soil_transport_device(d+46,p,so,admitargs,admission,vic);for(int i=0;i<12;++i)out[409+i]=admission[i];
    float runoff=(d[353]+drain[60])+admission[26],runoff_h=(d[354]+drain[61])+admission[27];
    float advected=d[349]+dt*(canopy[206]+canopy[207]),et=(d[352]+d[348])+d[351];
    float le=solver.latent(out);
    float atmospheric=((canopy[29]-canopy[26])-le)*dt;
    float old_hs=0.f,new_hs=0.f,old_hg=0.f,new_hg=0.f,old_ms=0.f,new_ms=0.f,old_mg=0.f,new_mg=0.f;
    for(int i=0;i<5;++i){old_hs+=old_snow[i*9+1];new_hs+=drain[i*9+1];old_ms+=old_snow[i*9];new_ms+=drain[i*9];}
    for(int i=0;i<4;++i){old_hg+=old_soil[i*3+1];new_hg+=admission[i*3+1];old_mg+=old_soil[i*3];new_mg+=admission[i*3];}
    float old_h=(old_hs+old_hg)+v[2],new_h=(new_hs+new_hg)+canopy[205];
    float old_m=(old_ms+old_mg)+v[1],new_m=(new_ms+new_mg)+canopy[204];
    out[421]=et;out[422]=runoff;out[423]=runoff_h;out[424]=advected;out[425]=atmospheric;
    out[426]=(((new_h-old_h)-atmospheric)+advected)+runoff_h;out[427]=((new_m-old_m)+et)+runoff;
    out[428]=drain[53];out[429]=drain[54];out[430]=le;out[431]=heat[71]/dt;out[432]=(float)count;
    out[433]=converged&&admission[46]>0.f?1.f:0.f;
    for(int i=0;i<5;++i)out[434+i]=(d[355+i]+drain[62+i])+drain[55+i];
    if(d[348]!=0.f)out[260]=(d[350]*dt)/d[348];
}

template<class Solver> __device__ void surface_device(const float* c,const float* g,const float* p,const int* o,
    const float* a,const float* input,const float* f,const float* v,const float* radiation,
    const float* leaf,const float* aging,float* out,const Solver& solver,const float* vic=nullptr) {
    int n=o[0],phase=o[1];float u[23],supply[10],root_h[4],offset0=0.f;bool inverse_ok=true;
    for(int i=0;i<23;++i)u[i]=input[i];u[15]=v[1]/v[0];u[17]=n>0?c[0]/v[0]:__int_as_float(0x7f800000);supply[0]=1.f;
    for(int i=0;i<4;++i) {
        float offset,h,cap,liquid,count;inverse_ok=invert_soil_enthalpy(g[i*3],g[i*3+1],g[i*3+2],p[0],p[1],p[6],phase,offset,h,cap,liquid,count)&&inverse_ok;
        root_h[i]=334000.f+4180.f*offset;supply[1+i]=fmaxf(0.f,liquid-(p[5]*1000.f)*g[i*3+2])/v[0];supply[5+i]=v[3+i];
        if(i==0){supply[9]=liquid/v[0];offset0=offset;}
    }
    if(n>0){float ice,liq;snow_mass_phase(c[0],c[1],offset0,ice,liq);}
    float center=fminf(350.f,fmaxf(200.f,offset0+273.15f));
    surface_evaluate(c,g,p,n,phase,a,u,f,v,radiation,leaf,supply,root_h,inverse_ok,center,out,solver);
    const float tol=2.f*1.1920928955078125e-7f*273.15f;
    int count=1;float left=center,right=center,fl=out[361],fr=fl,width=2.f,unused[Solver::output_size];
    if(fabsf(out[361])>tol)for(int k=0;k<8;++k) {
        if(fl<0.f&&left>200.f){left=fmaxf(200.f,center-width);surface_evaluate(c,g,p,n,phase,a,u,f,v,radiation,leaf,supply,root_h,inverse_ok,left,unused,solver);fl=unused[361];++count;}
        if(fr>0.f&&right<350.f){right=fminf(350.f,center+width);surface_evaluate(c,g,p,n,phase,a,u,f,v,radiation,leaf,supply,root_h,inverse_ok,right,unused,solver);fr=unused[361];++count;}
        width*=2.f;
    }
    bool bracketed=fl>=0.f&&fr<=0.f;
    // Stay on the continuous branch connected to the initial snow surface
    // when the expanding bracket crosses removal of its vapour-supply cell.
    if(n>0&&(int)out[45]==n&&out[361]>0.f&&bracketed) {
        surface_evaluate(c,g,p,n,phase,a,u,f,v,radiation,leaf,supply,root_h,inverse_ok,right,unused,solver);++count;
        if((int)unused[45]<n) {
            float event_left=center,event_right=right;
            for(int k=0;k<60&&(event_right-event_left)>tol;++k) {
                float trial=event_left+.5f*(event_right-event_left);
                surface_evaluate(c,g,p,n,phase,a,u,f,v,radiation,leaf,supply,root_h,inverse_ok,trial,unused,solver);++count;
                if((int)unused[45]==n) {
                    if(unused[361]<=0.f){right=trial;fr=unused[361];break;}
                    event_left=trial;
                }else event_right=trial;
            }
        }
    }
    for(int k=0;k<60&&fabsf(out[361])>tol&&bracketed&&(right-left)>tol;++k) {
        float span=right-left,trial=left-(fl*span)/(fr!=fl?fr-fl:1.f);
        trial=fminf(right-.05f*span,fmaxf(left+.05f*span,trial));
        // A percentage margin may round back to the endpoint. Contract it.
        if(trial<=left||trial>=right)trial=left+.5f*span;
        surface_evaluate(c,g,p,n,phase,a,u,f,v,radiation,leaf,supply,root_h,inverse_ok,trial,out,solver);++count;
        if(out[361]>0.f){left=trial;fl=out[361];}else{right=trial;fr=out[361];}
    }
    if(out[45]>0.f&&out[284]==0.f&&out[360]!=273.15f) {
        surface_evaluate(c,g,p,n,phase,a,u,f,v,radiation,leaf,supply,root_h,inverse_ok,273.15f,unused,solver);++count;
        if(fabsf(unused[361])<=tol)for(int i=0;i<363;++i)out[i]=unused[i];
        if(fabsf(unused[361])<=tol)for(int i=441;i<Solver::output_size;++i)out[i]=unused[i];
    }
    bool converged=(fabsf(out[361])<=tol||(bracketed&&(right-left)<=tol))&&out[362]>0.f;
    surface_finish(c,g,v,out,count,converged,aging,p,phase,solver,vic);
    out[439]=left;out[440]=right;
}

__device__ void canopy_surface_device(const float* c,const float* g,const float* p,const int* o,
    const float* a,const float* input,const float* f,const float* v,const float* radiation,
    const float* leaf,const float* aging,float* out) {
    LiquidSurfaceCanopy solver{a,f,leaf,radiation,v};
    surface_device(c,g,p,o,a,input,f,v,radiation,leaf,aging,out,solver);
}

extern "C" __global__ void run_canopy_surface(const float* c,const float* g,const float* p,const int* o,
    const float* a,const float* u,const float* f,const float* v,const float* radiation,
    const float* leaf,const float* aging,float* out,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        canopy_surface_device(c+45*row,g+12*row,p+7*row,o+2*row,a+10*row,u+23*row,f+20*row,v+7*row,
            radiation+584*row,leaf,aging,out+441*row);
}
