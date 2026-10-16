// Independent CUDA implementation of simplified crop thermal/management step.
// NVRTC provides CUDA math intrinsics; no host C library headers are needed.

__device__ float crop_c(const float* b) { return ((b[0]+b[1])+b[2])+b[6]; }
__device__ float crop_n(const float* b) { return ((b[3]+b[4])+b[5])+b[7]; }
__device__ float crop_sum7(const float* a) {
    float v=0.f; for(int k=0;k<7;k++)v=v+a[k]; return v;
}

extern "C" __global__ void run_crop(const float* states,const int* phases,const float* plants,
    const float* climates,const int* events,const float* parameters,const float* plant_parameters,float* output,int n) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
        const float* old=states+5*i;const float* b=plants+8*i;const float* u=climates+4*i;
        const float* p=parameters+9*i;const float* pp=plant_parameters+21*i;const int* e=events+2*i;
        float* out=output+43*i;for(int k=0;k<43;k++)out[k]=0.f;
        float oldc=crop_c(b)+old[3],oldn=crop_n(b)+old[4];bool planted=e[0]>0;
        int phase=planted?1:phases[i];bool active=phase>0;
        float ic=planted?p[6]:0.f,inn=planted?p[6]/pp[3]:0.f;
        float seedc=planted?ic:old[3],seedn=planted?inn:old[4];
        float air=planted?0.f:old[0],soil=planted?0.f:old[1],age=planted?0.f:old[2];
        float da=(fminf(fmaxf(u[0]-p[0],0.f),p[1])*u[2])/86400.f;
        float ds=(fminf(fmaxf(u[1]-p[0],0.f),p[1])*u[2])/86400.f;
        air=air+(active?da:0.f);soil=soil+(active?ds:0.f);age=age+(active?u[2]/86400.f:0.f);
        int reason=active?(e[1]>0?3:(air>=p[2]?1:(age>=p[5]?2:0))):0;
        bool harvest=reason>0,emerge=phase==1&&!harvest&&soil>=p[3]*p[2];
        float ec=harvest?p[7]*(b[0]+b[1]):0.f,en=harvest?p[7]*(b[3]+b[4]):0.f;
        float lc=harvest?(crop_c(b)+seedc)-ec:0.f,ln=harvest?(crop_n(b)+seedn)-en:0.f;
        for(int k=0;k<3;k++){out[14+k]=lc*pp[9+k];out[21+k]=ln*pp[9+k];}
        for(int k=0;k<8;k++)out[6+k]=harvest?0.f:b[k];
        if(emerge){out[6]=b[0]+seedc;out[9]=b[3]+seedn;}
        seedc=harvest||emerge?0.f:seedc;seedn=harvest||emerge?0.f:seedn;
        phase=harvest?0:(emerge?2:phase);
        bool grain=phase==2&&(air>=p[4]*p[2]||(out[6]*pp[19])/u[3]>=p[8]);
        phase=grain?3:phase;bool growing=phase==2||phase==3;
        float newc=crop_c(out+6)+seedc,newn=crop_n(out+6)+seedn;
        float cr=(((newc-oldc)+ec)+crop_sum7(out+14))-ic,nr=(((newn-oldn)+en)+crop_sum7(out+21))-inn;
        const float eps=1.1920928955078125e-7f;
        float cb=128.f*eps*fmaxf((((newc+oldc)+ec)+crop_sum7(out+14))+ic,1.f);
        float nb=128.f*eps*fmaxf((((newn+oldn)+en)+crop_sum7(out+21))+inn,1.f);
        bool valid=fabsf(cr)<=cb&&fabsf(nr)<=nb&&seedc>=0.f&&seedn>=0.f;
        for(int k=0;k<8;k++)valid=valid&&out[6+k]>=0.f;
        out[0]=air;out[1]=soil;out[2]=age;out[3]=seedc;out[4]=seedn;out[5]=phase;
        out[28]=ic;out[29]=inn;out[30]=ec;out[31]=en;out[32]=planted;out[33]=emerge;
        out[34]=harvest;out[35]=reason;out[36]=growing;out[37]=growing?p[8]:0.f;
        out[38]=cr;out[39]=nr;out[40]=cb;out[41]=nb;out[42]=valid;
    }
}
