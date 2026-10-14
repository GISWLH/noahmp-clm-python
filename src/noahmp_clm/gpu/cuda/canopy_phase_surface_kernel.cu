// Layered phase canopy adapter for the shared extensive surface solver.
struct PhaseSurfaceCanopy {
    static const int output_size=741;
    const float *a,*f,*leaf,*radiation,*v,*condensate;
    const float *extension=nullptr;
    __device__ void solve(float temp,const float* u,const float* supply,const float* root_h,int n,float* out) const {
        float stem[10]={0.f},b[9]={1.f,temp,temp<273.15f?1.f:0.f,n==0?1.f:0.f,v[0]};
        for(int i=0;i<4;++i)b[5+i]=root_h[i];
        PhaseContext context;phase_prepare(a,u,f,supply,stem,condensate,b,leaf,radiation,context);
        if(extension){context.water_velocity=extension[0];context.heat_temperature=extension[1];}
        float* full=out+441;phase_solve(context,full,100,.005f);
        float* canopy=out+58;
        for(int i=0;i<195;++i)canopy[i]=full[i];
        canopy[195]=full[288];for(int i=0;i<5;++i)canopy[196+i]=full[293+i];
        canopy[201]=full[289];canopy[202]=full[290];canopy[203]=full[285];
        canopy[204]=ca_sum(full+195,10);canopy[205]=ca_sum(full+205,10);
        canopy[206]=full[286];canopy[207]=full[287];
    }
    __device__ float latent(const float* out) const {
        return (ca_sum(out+696,10)/v[0]+2500000.f*ca_sum(out+98,10))+out[350];
    }
};

__device__ void canopy_phase_surface_device(const float* c,const float* g,const float* p,const int* o,
    const float* a,const float* u,const float* f,const float* input,const float* condensate,
    const float* radiation,const float* leaf,const float* aging,float* out) {
    float v[7];for(int i=0;i<7;++i)v[i]=input[i];v[1]=0.f;v[2]=0.f;
    for(int i=0;i<10;++i){v[1]+=condensate[2*i];v[2]+=condensate[2*i+1];}
    PhaseSurfaceCanopy solver{a,f,leaf,radiation,v,condensate};
    surface_device(c,g,p,o,a,u,f,v,radiation,leaf,aging,out,solver);
}

extern "C" __global__ void run_canopy_phase_surface_extended(const float* c,const float* g,const float* p,const int* o,
    const float* a,const float* u,const float* f,const float* input,const float* condensate,const float* extension,
    const float* radiation,const float* leaf,const float* aging,float* out,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x){
        float v[7];for(int i=0;i<7;++i)v[i]=input[7*row+i];v[1]=0.f;v[2]=0.f;
        for(int i=0;i<10;++i){v[1]+=condensate[20*row+2*i];v[2]+=condensate[20*row+2*i+1];}
        PhaseSurfaceCanopy solver{a+10*row,f+20*row,leaf,radiation+584*row,v,condensate+20*row,extension+2*row};
        surface_device(c+45*row,g+12*row,p+6*row,o+2*row,a+10*row,u+23*row,f+20*row,v,
            radiation+584*row,leaf,aging,out+741*row,solver);
    }
}

extern "C" __global__ void run_canopy_phase_surface(const float* c,const float* g,const float* p,const int* o,
    const float* a,const float* u,const float* f,const float* v,const float* condensate,const float* radiation,
    const float* leaf,const float* aging,float* out,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        canopy_phase_surface_device(c+45*row,g+12*row,p+6*row,o+2*row,a+10*row,u+23*row,f+20*row,v+7*row,
            condensate+20*row,radiation+584*row,leaf,aging,out+741*row);
}

extern "C" __global__ void run_canopy_phase_surface_vic(const float* c,const float* g,const float* p,const int* o,
    const float* a,const float* u,const float* f,const float* input,const float* condensate,const float* extension,
    const float* vic,const float* radiation,const float* leaf,const float* aging,float* out,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x){
        float v[7];for(int i=0;i<7;++i)v[i]=input[7*row+i];v[1]=0.f;v[2]=0.f;
        for(int i=0;i<10;++i){v[1]+=condensate[20*row+2*i];v[2]+=condensate[20*row+2*i+1];}
        PhaseSurfaceCanopy solver{a+10*row,f+20*row,leaf,radiation+584*row,v,condensate+20*row,
            extension?extension+2*row:nullptr};
        surface_device(c+45*row,g+12*row,p+6*row,o+2*row,a+10*row,u+23*row,f+20*row,v,
            radiation+584*row,leaf,aging,out+741*row,solver,vic+8*row);
    }
}
