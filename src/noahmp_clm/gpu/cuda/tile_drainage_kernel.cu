// Field-capacity excess at one tube depth; the frozen drain remains closed.
extern "C" __global__ void run_tile_drainage(const float* layers,const float* dt,const float* params,float* result,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x) {
        const float* l=layers+16*row;const float* p=params+3*row;float* out=result+10*row;
        float bottom=0.f;int index=0;
        for(int k=0;k<4;++k){bottom+=l[4*k+2];index+=(p[0]>bottom);}
        float water=l[4*index],ice=l[4*index+1],z=l[4*index+2],t=l[4*index+3];
        float retained=fmaxf(0.f,(1000.f*p[1])*z-ice),excess=fminf(water,fmaxf(0.f,water-retained));
        bool eligible=t>=273.15f&&ice<=1e-9f;float capacity=p[2]*(dt[row]/86400.f);
        float removed=eligible?fminf(excess,capacity):0.f,heat=removed*(334000.f+4180.f*(t-273.15f));
        for(int k=0;k<4;++k)out[k]=k==index?removed:0.f;
        out[4]=index;out[5]=(float)eligible;out[6]=excess;out[7]=capacity;out[8]=removed;out[9]=heat;
    }
}
