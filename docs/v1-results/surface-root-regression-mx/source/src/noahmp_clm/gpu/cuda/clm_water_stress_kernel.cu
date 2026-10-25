// Independently implemented Noah-MP wetness equations, heads in metres.
__device__ float clm_soil_beta(float theta,const float* p,int option) {
    float beta;
    if (option==1) beta=(theta-p[5])/fmaxf(1.e-9f,p[2]-p[5]);
    else {
        float log_wilt=logf(150.f);
        float log_head=logf(p[6])-p[1]*logf(fmaxf(.01f,theta)/p[0]);
        float potential=log_head>=log_wilt?-150.f:-expf(fminf(log_head,log_wilt));
        if (option==2) beta=(1.f-potential/-150.f)/(1.f+p[6]/-150.f);
        else beta=-expm1f(-5.8f*logf(-150.f/potential));
    }
    return fminf(1.f,fmaxf(0.f,beta));
}

extern "C" __global__ void run_clm_water_stress(
    const float* liquid, const float* roots, const float* parameters,
    const int* options, float* out, int count) {
    for (int row=blockIdx.x*blockDim.x+threadIdx.x;row<count;row+=blockDim.x*gridDim.x) {
        const float* p=parameters+row*7;
        float weighted[4];
        for (int k=0;k<4;++k) {
            float beta=clm_soil_beta(liquid[row*4+k],p,options[row]);
            out[row*5+1+k]=beta;weighted[k]=beta*roots[row*4+k];
        }
        out[row*5]=fminf(1.f,fmaxf(0.f,(weighted[0]+weighted[2])+(weighted[1]+weighted[3])));
    }
}
