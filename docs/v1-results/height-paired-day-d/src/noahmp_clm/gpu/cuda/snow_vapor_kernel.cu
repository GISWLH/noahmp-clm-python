// Requires snow_mass_kernel.cu: phase diagnosis and conservative remapping.
__device__ void snow_vapor_device(const float* c,int n,const float* u,float* out) {
    float cells[54]={0.f};for(int i=0;i<45;++i)cells[i]=c[i];
    float amount=0.f,ice=0.f,liquid=0.f,heat=0.f,residue[5]={0.f};
    if(u[0]>0.f) {
        float supply=c[0],bound=(64.f*1.1920928955078125e-7f)*fmaxf(u[0],supply);
        amount=fabsf(u[0]-supply)<=bound?supply:u[0];bool full=amount==supply;
        float fraction=full?1.f:amount/fmaxf(supply,1e-30f),offset,liquid0;
        snow_mass_phase(supply,c[1],offset,ice,liquid0);
        ice=full?ice:(amount*ice)/fmaxf(supply,1e-30f);liquid=amount-ice;heat=fraction*c[1];
        cells[0]=supply-amount;cells[1]=c[1]-heat;cells[2]=c[2]*(1.f-fraction);
        if(full) {
            for(int p=0;p<5;++p) {if(n>1)cells[13+p]+=cells[4+p];else residue[p]=cells[4+p];}
            --n;
            for(int i=0;i<5;++i)for(int p=0;p<9;++p)
                cells[9*i+p]=i<n?cells[9*(i+1)+p]:(p==3?54.5f:0.f);
        }
    } else if(u[0]<0.f) {
        amount=u[0];ice=u[1]<0.f?amount:0.f;liquid=amount-ice;
        heat=(ice*2100.f)*u[1]+liquid*(334000.f+4180.f*u[1]);
        cells[0]=c[0]-amount;cells[1]=c[1]-heat;float offset,frozen,melt;
        snow_mass_phase(cells[0],cells[1],offset,frozen,melt);
        cells[2]=fmaxf(c[2]-ice/200.f,frozen/917.f);n=max(n,1);
    }
    if(u[2]>0.f&&u[0]!=0.f&&n>0)n=snow_mass_regrid(cells,out);
    else for(int i=0;i<45;++i)out[i]=cells[i];
    out[45]=(float)n;out[46]=amount;out[47]=heat;
    for(int p=0;p<5;++p)out[48+p]=residue[p];
    out[53]=ice;out[54]=liquid;out[55]=2.5e6f*amount+334000.f*ice;
}

extern "C" __global__ void run_snow_vapor(const float* cells,const int* counts,const float* inputs,float* output,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        snow_vapor_device(cells+row*45,counts[row],inputs+row*3,output+row*56);
}
