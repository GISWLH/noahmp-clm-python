// Four-layer moisture trigger and explicit supply, all calculations on device.
extern "C" __global__ void run_irrigation(const float* old,const float* layers,const float* climate,
    const int* flags,const float* params,float* result,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x) {
        const float* o=old+2*row;const float* l=layers+12*row;const float* u=climate+4*row;
        const float* p=params+5*row;float* out=result+16*row;
        float top=0.f,depth=0.f,available=0.f,frozen=0.f;
        for(int k=0;k<4;++k){
            float thickness=fminf(l[3*k+2],fmaxf(p[0]-top,0.f)),fraction=thickness/l[3*k+2];
            depth+=thickness;available+=l[3*k]*fraction;frozen+=l[3*k+1]*fraction;top+=l[3*k+2];
        }
        float target=(1000.f*p[1])*depth,wilt=(1000.f*p[2])*depth;
        float threshold=p[3]*(target-wilt)+wilt,deficit=fmaxf(threshold-available,0.f);
        bool eligible=u[1]>0.f&&frozen<=1e-9f;float cancelled=eligible?0.f:o[0];
        float requested=flags[row]&&eligible?deficit:0.f;
        float pending=requested>0.f?requested:(eligible?o[0]:0.f),seconds=requested>0.f?p[4]:(eligible?o[1]:0.f),quota=0.f;
        if(pending>0.f){if(u[0]>=seconds){quota=pending;pending=seconds=0.f;}
            else {quota=pending*(u[0]/seconds);pending-=quota;seconds-=u[0];}}
        float applied=fminf(quota,u[2]),unmet=quota-applied,heat=applied*(334000.f+4180.f*(u[3]-273.15f));
        float residual=((((pending-o[0])+applied)+unmet)+cancelled)-requested;
        float scale=(((((pending+o[0])+applied)+unmet)+cancelled)+requested);
        float bound=128.f*1.1920928955078125e-7f*fmaxf(scale,1.f);
        float values[16]={pending,seconds,available,threshold,deficit,frozen,(float)eligible,(float)(requested>0.f),requested,
            applied,heat,unmet,cancelled,residual,bound,(float)(fabsf(residual)<=bound&&pending>=0.f&&seconds>=0.f)};
        for(int k=0;k<16;++k)out[k]=values[k];
    }
}
