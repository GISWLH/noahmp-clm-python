// NVRTC supplies device math intrinsics; no host C-library headers are needed.
extern "C" __global__ void run_clm_freezing(const float *a,float *out,int n) {
    for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
        const float *x=a+5*i;
        float total=x[0],t=x[1],sat=x[2],b=x[3],psi=x[4];
        float deficit=fmaxf(273.15f-t,0.f);
        float coefficient=334000.f/(9.80616f*psi);
        float raw=(coefficient*deficit)/t,ratio=fmaxf(raw,1e-12f);
        float potential=expf(fminf(logf(sat)-logf(ratio)/b,logf(2.f*sat)));
        float denominator=((b*t)*t)*ratio;
        float dp=raw>1e-12f?((potential*coefficient)*273.15f)/denominator:0.f;
        float four=.04f*total,width=fmaxf(four,1e-12f),dw=four>1e-12f?.04f:0.f;
        float difference=total-potential,sign=difference>0.f?1.f:(difference<0.f?-1.f:0.f);
        float rounded=fmaxf(width-fabsf(difference),0.f);
        float liquid=fmaxf(fminf(total,potential)-(rounded*rounded)/(4.f*width),0.f);
        float dtotal=difference<0.f?1.f:(difference>0.f?0.f:.5f);
        dtotal=dtotal-(rounded*(dw-sign))/(2.f*width);
        dtotal=dtotal+((rounded*rounded)*dw)/((4.f*width)*width);
        float dcap=(difference>0.f?1.f:(difference<0.f?0.f:.5f))-(rounded*sign)/(2.f*width);
        float dt=liquid>0.f?dcap*dp:0.f;
        dtotal=liquid>0.f?dtotal:0.f;
        out[3*i]=t>=273.15f?total:liquid;
        out[3*i+1]=t>=273.15f?0.f:dt;
        out[3*i+2]=t>=273.15f?1.f:dtotal;
    }
}
