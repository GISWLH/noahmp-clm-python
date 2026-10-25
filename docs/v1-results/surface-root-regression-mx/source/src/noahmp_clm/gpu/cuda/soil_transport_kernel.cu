// Four liquid-equivalent soil reservoirs; H remains authoritative.
// Compiled after enthalpy_kernel.cu, independent of the released soil scheme.
// Keep the transfer call boundary when reused in the fused nonlinear surface.
// NVRTC's inlined version corrupts local soil masses in that caller (the
// standalone transfer is unaffected); full-field regression covers both.
__device__ float vic_admission(float incoming,float water,float capacity,float shape) {
    float deficit=fmaxf(0.f,capacity-water);
    if(deficit==0.f)return 0.f;
    if(shape==0.f)return fminf(incoming,deficit);
    float exponent=1.f+shape,log_x=logf(deficit/capacity)/exponent;
    float limit=(exponent*capacity)*expf(log_x);
    float value=incoming>=limit?deficit:-deficit*expm1f(exponent*log1pf(-incoming/limit));
    return fminf(incoming,fmaxf(0.f,value));
}

__device__ float vic_baseflow(float liquid,float capacity,float dt,const float* v) {
    float residual=v[7]*capacity,available=fmaxf(0.f,liquid-residual);
    float wet=available/(capacity-residual),linear=v[3]/v[5];
    float excess=fmaxf(0.f,wet-v[5])/(1.f-v[5]);
    float fraction=linear*wet+(1.f-linear)*powf(excess,v[6]);
    return fminf(available,(v[4]*(dt/86400.f))*fraction);
}

__device__ __noinline__ void soil_transport_device(const float* cells,const float* p,const int* option,
                                      const float* u,float* out,const float* vic=nullptr) {
    float water[4],heat[4],z[4],offset[4],liquid[4];bool valid=true;
    float por=p[0],b=p[1],field=p[2],satdk=p[3],psi=p[4];int mode=option[0],phase=option[1],permeability=option[2];
    for(int j=0;j<47;++j)out[j]=0.f;
    for(int j=0;j<4;++j) {
        water[j]=cells[j*3];heat[j]=cells[j*3+1];z[j]=cells[j*3+2];
        float h,c,count;
        valid=invert_soil_enthalpy(water[j],heat[j],z[j],por,b,psi,phase,offset[j],h,c,liquid[j],count)&&valid;
    }
    if(mode==0) {
        float specific=u[1]/(u[0]>0.f?u[0]:1.f),left=u[0],surface=0.f;
        if(vic&&vic[0]>0.f) {
            float capacity=(1000.f*por)*(z[0]+z[1]),upper_water=fminf(capacity,water[0]+water[1]);
            left=vic_admission(u[0],upper_water,capacity,vic[2]);surface=u[0]-left;
        }
        out[24]=u[0];out[25]=u[1];
        for(int j=0;j<4;++j) {
            float accepted=fminf(left,fmaxf(0.f,(1000.f*por)*z[j]-water[j]));
            water[j]+=accepted;heat[j]+=accepted*specific;left-=accepted;
        }
        left+=surface;out[26]=left;out[27]=left*specific;
    } else if(mode==1) {
        for(int j=0;j<4;++j) {
            float demand=u[3+j],rounding=8.f*1.1920928955078125e-7f*fmaxf(water[j],1.f);
            valid=valid&&demand<=liquid[j]+rounding&&demand<=water[j];
            float carried=demand*(334000.f+4180.f*offset[j]);
            water[j]-=demand;heat[j]-=carried;out[28+j]=demand;out[32+j]=carried;
        }
    } else {
        for(int j=0;j<4;++j) {
            float offset,h,c,liquid,count;
            valid=invert_soil_enthalpy(water[j],heat[j],z[j],por,b,psi,phase,offset,h,c,liquid,count)&&valid;
            float ice=water[j]-liquid,capacity=(1000.f*por)*z[j];
            float wet=fminf(1.f,fmaxf(0.f,water[j]/capacity));
            float frozen=permeability==1?1.f:expf((-7.f*ice)/capacity);
            float hydraulic=(((1000.f*u[2])*satdk)*frozen)*powf(wet,2.f*b+3.f);
            float excess=fmaxf(0.f,water[j]-(1000.f*field)*z[j]);
            float moved=fminf(fminf(hydraulic,excess),liquid);
            if(j==3&&vic&&vic[1]>0.f)moved=vic_baseflow(liquid,capacity,u[2],vic)*frozen;
            if(j<3)moved=fminf(moved,fmaxf(0.f,(1000.f*por)*z[j+1]-water[j+1]));
            float carried=moved*(334000.f+4180.f*offset);
            water[j]-=moved;heat[j]-=carried;
            if(j<3){water[j+1]+=moved;heat[j+1]+=carried;}
            out[36+j]=moved;out[40+j]=carried;
        }
        out[44]=out[39];out[45]=out[43];
    }
    for(int j=0;j<4;++j) {
        float h,c,count;
        valid=invert_soil_enthalpy(water[j],heat[j],z[j],por,b,psi,phase,offset[j],h,c,liquid[j],count)&&valid;
        valid=valid&&water[j]>=0.f;
        out[j*3]=water[j];out[j*3+1]=heat[j];out[j*3+2]=z[j];
        out[12+j]=offset[j];out[16+j]=liquid[j];out[20+j]=water[j]-liquid[j];
    }
    out[46]=valid?1.f:0.f;
}

__device__ void column_soil_transport_device(const float* cells,const float* full,const int* options,
        const float* inputs,float* output,const float* vic=nullptr) {
    float compact[5]={full[0],full[1],full[2],full[3],full[6]};
    soil_transport_device(cells,compact,options,inputs,output,vic);
}

extern "C" __global__ void run_vic_transport(const float* cells,const float* p,const int* options,
                                             const float* inputs,const float* vic,float* output,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        soil_transport_device(cells+row*12,p+row*5,options+row*3,inputs+row*7,output+row*47,vic+row*8);
}

extern "C" __global__ void run_soil_transport(const float* cells,const float* p,const int* options,
                                             const float* inputs,float* output,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        soil_transport_device(cells+row*12,p+row*5,options+row*3,inputs+row*7,output+row*47);
}
