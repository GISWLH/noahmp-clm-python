// Extensive development column. The released kernel supplies drag and stress
// helpers only; no released temperature or water time step is called here.
__device__ float ec_sum(const float* x,int n,int stride=1){float v=0.f;for(int k=0;k<n;++k)v+=x[k*stride];return v;}

__device__ bool ec_route_water(float* snow,float* soil,int n,const float* p,int phase,float dt,
                               float mass,float heat,float* runoff,const float* vic=nullptr) {
    runoff[0]=runoff[1]=0.f;
    if(mass<=0.f)return true;
    if(n>0){snow[0]+=mass;snow[1]+=heat;return true;}
    int opt[3]={0,phase,1};float u[7]={mass,heat,dt,0.f,0.f,0.f,0.f},out[47];
    column_soil_transport_device(soil,p,opt,u,out,vic);for(int k=0;k<12;++k)soil[k]=out[k];
    runoff[0]=out[26];runoff[1]=out[27];return out[46]>0.f;
}

__device__ bool ec_collapse_molten_snow(float* c,float* g,int& n,const float* p,int phase,float dt,
                                       float* runoff,float* impurity,const float* vic=nullptr) {
    runoff[0]=runoff[1]=0.f;for(int k=0;k<5;++k)impurity[k]=0.f;
    bool molten=n>0;for(int k=0;k<n;++k)molten=molten&&(c[k*9+1]>=334000.f*c[k*9]);
    if(!molten)return true;
    float mass=ec_sum(c,5,9),heat=ec_sum(c+1,5,9);
    for(int k=0;k<5;++k)impurity[k]=ec_sum(c+4+k,5,9);
    float u[7]={mass,heat,dt,0.f,0.f,0.f,0.f},out[47];int opt[3]={0,phase,1};
    column_soil_transport_device(g,p,opt,u,out,vic);
    for(int k=0;k<45;++k)c[k]=0.f;for(int k=0;k<5;++k)c[k*9+3]=54.5f;n=0;
    for(int k=0;k<12;++k)g[k]=out[k];runoff[0]=out[26];runoff[1]=out[27];return out[46]>0.f;
}

__device__ float ec_ground_resistance(float liquid,float dz,const float* p,int option,bool snow) {
    if(option==4&&snow)return 1.f;
    float sat=fmaxf(p[0],1e-6f),ratio=fmaxf(0.f,liquid/sat);
    if(option==2)return expf(8.25f-4.225f*ratio);
    if(option==3)return expf(8.25f-6.f*ratio);
    float fraction=fminf(1.f,clip(liquid,0.f,sat)/sat),y=1.f-fraction;
    float dry=(dz*(expf(y*y)-1.f))*(float)(1./1.718281828459045);
    float diffusion=(2.2e-5f*(sat*sat))*powf(fmaxf(1e-6f,1.f-p[5]/sat),2.f+3.f/fmaxf(p[1],1e-6f));
    return fmaxf(1e-3f,dry/diffusion);
}

extern "C" __global__ void prepare_enthalpy_columns(const float* snow_in,const float* soil_in,const float* params,
    const int* options,const float* sites,const float* storage,const float* roots_in,const float* forcing,
    float* snow_out,float* soil_out,int* surface_options,float* areas,float* inputs,float* factors,
    float* surface_storage,float* prepared,int* validity,float* pre_impurity,const float* vic_parameters,int size) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<size;row+=blockDim.x*gridDim.x) {
        const float* p=params+row*7;const int* o=options+row*9;const float* site=sites+row*12;
        const float* store=storage+row*3;const float* roots=roots_in+row*4;const float* f=forcing+row*8;
        float c[45],g[12];for(int k=0;k<45;++k)c[k]=snow_in[row*45+k];for(int k=0;k<12;++k)g[k]=soil_in[row*12+k];
        int n=o[0],phase=o[1];float dt=f[0],tair=f[1],capacity=(.5f*site[1])*site[0],mass=store[0],heat=store[1];
        float total=f[7]*dt,snow_m=total*snow_fraction(tair,o[2],f[2],f[4]),rain=(total-snow_m)*1000.f,snowfall=snow_m*1000.f;
        float rain_h=334000.f+4180.f*(tair-273.15f),ice_h=2100.f*(fminf(tair,273.15f)-273.15f);
        float precipitation_h=rain*rain_h+snowfall*ice_h;
        float drip=fmaxf(0.f,mass-capacity),drip_h=(drip*heat)/(mass>0.f?mass:1.f),runoff[2];
        mass-=drip;heat-=drip_h;bool valid=ec_route_water(c,g,n,p,phase,dt,drip,drip_h,runoff);
        float interception=fminf(site[1]*rain,fmaxf(0.f,capacity-mass));mass+=interception;heat+=interception*rain_h;
        if(snowfall>0.f) {
            float fresh[2];snow_fresh_properties(tair-273.15f,f[3],fresh);
            float u[16]={snowfall,fminf(tair,273.15f)-273.15f,fresh[0],fresh[1],.033f,.05f,.2f,.02f,.02f,.01f,.01f,0.f,0.f,0.f,0.f,0.f},out[74];
            snow_transfer_device(c,n,1,u,out);for(int k=0;k<45;++k)c[k]=out[k];n=(int)out[45];
        }
        float through=rain-interception,flow[2];valid=ec_route_water(c,g,n,p,phase,dt,through,through*rain_h,flow)&&valid;
        runoff[0]+=flow[0];runoff[1]+=flow[1];
        bool good=ec_collapse_molten_snow(c,g,n,p,phase,dt,flow,pre_impurity+row*5,vic_parameters?vic_parameters+row*8:nullptr);
        runoff[0]+=flow[0];runoff[1]+=flow[1];valid=valid&&good;
        float liq[4],t[4],weighted[4],view[70]={0.f};view[39]=p[0];view[40]=p[5];view[41]=p[2];view[42]=p[3];view[43]=p[1];view[44]=p[4];
        for(int k=0;k<4;++k) {
            float reconstructed,cap,count,liquid;valid=invert_soil_enthalpy(g[k*3],g[k*3+1],g[k*3+2],p[0],p[1],p[6],phase,t[k],reconstructed,cap,liquid,count)&&valid;
            liq[k]=liquid/(1000.f*g[k*3+2]);
            weighted[k]=roots[k]*(o[3]==1?beta(liq[k],view,o[3]):clm_soil_beta(liq[k],p,o[3]));
        }
        float stress=clip((weighted[0]+weighted[2])+(weighted[1]+weighted[3]),0.f,1.f);
        float velocity=exchange(f,site[2],store[2],o[5]),z0=clip(site[2],1e-5f,5.f);
        float ustar=(.4f*fmaxf(f[3],.3f))/logf(10.f/z0),leaf=.01f*sqrtf(ustar/site[10]);
        float coefficient=(float)(.4/.13)*powf((z0*ustar)/1.5e-5f,-.45f),decay=expf(-site[0]);
        float ground=(coefficient*decay+.004f*(1.f-decay))*ustar;
        float resistance=ec_ground_resistance(liq[0],g[2],p,o[4],n>0);
        float wet=capacity>0.f?fminf(1.f,powf(mass/fmaxf(capacity,1e-30f),(float)(2./3.))):0.f;
        float inf=__int_as_float(0x7f800000);
        float u[23]={tair,f[2],f[4],velocity,ground,leaf,t[0]+273.15f,0.f,site[7],resistance,0.f,
            f[5],f[6],site[1],wet,mass/dt,inf,inf,site[4],site[3],site[3],site[9],site[8]};
        for(int k=0;k<45;++k)snow_out[row*45+k]=c[k];for(int k=0;k<12;++k)soil_out[row*12+k]=g[k];
        surface_options[row*2]=n;surface_options[row*2+1]=phase;
        for(int k=0;k<10;++k){areas[row*10+k]=site[0]/10.f;factors[row*20+k*2]=site[11];factors[row*20+k*2+1]=stress;}
        for(int k=0;k<23;++k)inputs[row*23+k]=u[k];
        float* v=surface_storage+row*7;v[0]=dt;v[1]=mass;v[2]=heat;for(int k=0;k<4;++k)v[3+k]=roots[k];
        float* prep=prepared+row*5;prep[0]=rain+snowfall;prep[1]=precipitation_h;prep[2]=runoff[0];prep[3]=runoff[1];prep[4]=capacity;
        validity[row]=valid?1:0;
    }
}

__device__ void ec_runoff_scheme(float incoming,const float* g,const float* p,const float* site,int option,float* result) {
    float smc[4],deficit[4],saturation[4];
    for(int k=0;k<4;++k){smc[k]=g[k*3]/(1000.f*g[k*3+2]);deficit[k]=fmaxf(0.f,p[2]-smc[k])*g[k*3+2];saturation[k]=clip(smc[k]/fmaxf(p[0],1e-6f),0.f,1.f);}
    float terrain=expf(-site[5]),missing=ec_sum(deficit,4);
    if(option==2)missing=fmaxf(missing,(fmaxf(0.f,site[6])*p[0])*terrain);
    missing=fmaxf(0.f,missing*terrain);
    float wet=clip(expf(site[5])*expf(-fmaxf(0.f,site[6])/2.f),0.f,1.f);
    float top=fmaxf(0.f,incoming-missing)*wet,average=ec_sum(saturation,4)*.25f;
    float bats=incoming*((average*average)*average);
    float surface=option==3?0.f:(option==4?bats:top);surface=clip(surface,0.f,incoming);
    result[0]=surface;result[1]=incoming-surface;
}

extern "C" __global__ void finish_enthalpy_columns(const float* old_snow,const float* old_soil,const float* params,
    const int* options,const float* sites,const float* old_storage,const float* forcing,const float* deposition,
    const float* prep_in,const float* surface_snow,const float* surface_soil,const int* counts,const float* canopy,
    const float* flux_in,const float* surface_impurity,float* snow_out,float* soil_out,int* options_out,
    float* storage_out,float* diag_out,float* impurity_out,const float* vic_parameters,int size) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<size;row+=blockDim.x*gridDim.x) {
        const float* p=params+row*7;const int* o=options+row*9;const float* site=sites+row*12;const float* old_store=old_storage+row*3;
        const float* prep=prep_in+row*5;const float* flux=flux_in+row*15;float dt=forcing[row*8];int phase=o[1],n=counts[row];
        const float* vic=vic_parameters?vic_parameters+row*8:nullptr;
        float c[45],g[12],impurity[5];for(int k=0;k<45;++k)c[k]=surface_snow[row*45+k];for(int k=0;k<12;++k)g[k]=surface_soil[row*12+k];
        for(int k=0;k<5;++k)impurity[k]=surface_impurity[row*5+k];
        float mass=canopy[row*2],heat=canopy[row*2+1],runoff[2]={prep[2]+flux[1],prep[3]+flux[2]};bool valid=true;
        float drip=fmaxf(0.f,mass-prep[4]),drip_h=(drip*heat)/(mass>0.f?mass:1.f);mass-=drip;heat-=drip_h;
        if(drip>0.f) {
            float flow[2];valid=ec_route_water(c,g,n,p,phase,dt,drip,drip_h,flow,vic);
            float u[16]={0.f,0.f,100.f,54.5f,.033f,.05f,.2f,.02f,.02f,.01f,.01f,0.f,0.f,0.f,0.f,0.f},drained[74];
            snow_transfer_device(c,n,3,u,drained);for(int k=0;k<45;++k)c[k]=drained[k];n=(int)drained[45];
            float su[7]={drained[53],drained[54],dt,0.f,0.f,0.f,0.f},admitted[47];int so[3]={0,phase,1};
            column_soil_transport_device(g,p,so,su,admitted,vic);for(int k=0;k<12;++k)g[k]=admitted[k];valid=valid&&admitted[46]>0.f;
            runoff[0]+=(flow[0]+drained[60])+admitted[26];runoff[1]+=(flow[1]+drained[61])+admitted[27];
            for(int k=0;k<5;++k)impurity[k]+=drained[62+k]+drained[55+k];
        }
        float su[7]={0.f,0.f,dt,0.f,0.f,0.f,0.f},drained[47];int so[3]={2,phase,o[8]};
#ifdef CLM_RICHARDS_ZERO_FLUX
        // Explicit Richards follows this finish stage. Zero-input mode
        // copies actual stores and diagnoses phase without gravity drainage.
        so[0]=0;
#endif
        column_soil_transport_device(g,p,so,su,drained,vic);for(int k=0;k<12;++k)g[k]=drained[k];valid=valid&&drained[46]>0.f;
        runoff[0]+=drained[44];runoff[1]+=drained[45];
        for(int k=0;k<5;++k){float addition=deposition[row*5+k]*dt;if(n>0)c[4+k]+=addition;else impurity[k]+=addition;}
        float offset=drained[12];if(n>0){float ice,liq;snow_mass_phase(c[0],c[1],offset,ice,liq);}float temp=offset+273.15f;
        float incoming=runoff[0]/1000.f,one[2],two[2];ec_runoff_scheme(incoming,g,p,site,o[6],one);ec_runoff_scheme(one[1],g,p,site,o[7],two);
        float surface=clip(o[6]==o[7]?one[0]:one[0]+two[0],0.f,incoming),deep=incoming-surface;
        float water0=(ec_sum(old_snow+row*45,5,9)+ec_sum(old_soil+row*12,4,3))+old_store[0];
        float water1=(ec_sum(c,5,9)+ec_sum(g,4,3))+mass;
        float h0=(ec_sum(old_snow+row*45+1,5,9)+ec_sum(old_soil+row*12+1,4,3))+old_store[1];
        float h1=(ec_sum(c+1,5,9)+ec_sum(g+1,4,3))+heat;
        float water_res=(((water1-water0)-prep[0])+flux[0])+runoff[0];
        float energy_res=((((h1-h0)-prep[1])-flux[3])+flux[4])+runoff[1];
        float d[26]={temp,flux[9],flux[6],flux[7],flux[8],flux[0],prep[0],runoff[0],surface*1000.f,deep*1000.f,
            ec_sum(c,5,9)/1000.f,mass/1000.f,flux[5],(flux[5]*12.0107e-6f)*dt,flux[10],flux[12],flux[13],flux[14],
            prep[1],runoff[1],flux[4],flux[3],water_res,energy_res,flux[11],valid?1.f:0.f};
        if(vic&&(vic[0]>0.f||vic[1]>0.f)){d[8]=runoff[0]-drained[44];d[9]=drained[44];}
        for(int k=0;k<45;++k)snow_out[row*45+k]=c[k];for(int k=0;k<12;++k)soil_out[row*12+k]=g[k];
        for(int k=0;k<9;++k)options_out[row*9+k]=k==0?n:o[k];
        storage_out[row*3]=mass;storage_out[row*3+1]=heat;storage_out[row*3+2]=temp;
        for(int k=0;k<26;++k)diag_out[row*26+k]=d[k];for(int k=0;k<5;++k)impurity_out[row*5+k]=impurity[k];
    }
}
