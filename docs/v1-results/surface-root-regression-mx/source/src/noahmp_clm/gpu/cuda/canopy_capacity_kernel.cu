// Original C3 capacity/profile and exact retained-window update equations.
// One thread owns a column; every arithmetic output is retained unrounded.
extern "C" __global__ void canopy_capacity(const float* leaf,const float* env,
    const float* traits,const float* source,float* output,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x) {
        const float* a=leaf+10*row;const float* u=env+3*row;
        const float* p=traits+4*row;const float* lp=source+10*row;float* out=output+126*row;
        float nitrogen=1.f/(p[0]*p[1]);
        float unscaled=(((nitrogen*p[2])*7.16f)*60.f)*p[3];
        float kn=expf(.00963f*unscaled-2.43f);
        float ratio=u[0]/u[1];float factor=fmaxf(.01f,ratio*ratio);
        float growth=fminf(fmaxf(u[2]-273.15f,11.f),35.f);float top=unscaled*factor;
        float cumulative=0.f;
        for(int k=0;k<10;++k) {
            cumulative=cumulative+a[k];float midpoint=cumulative-.5f*a[k];
            float scale=a[k]>0.f?expf(-(kn*midpoint)):0.f;float vcmax=top*scale;
            for(int j=0;j<10;++j)out[10*k+j]=lp[j];
            out[10*k]=vcmax;out[10*k+1]=(2.59f-.035f*growth)*vcmax;
            out[10*k+2]=.167f*vcmax;out[10*k+3]=.015f*vcmax;out[10*k+8]=growth+273.15f;
            out[100+k]=scale;out[110+k]=nitrogen*scale;
        }
        out[120]=unscaled;out[121]=top;out[122]=kn;out[123]=factor;out[124]=u[2];out[125]=growth+273.15f;
    }
}

extern "C" __global__ void temperature_memory(const float* values,const int* controls,
    const float* forcing,float* after,int* next_controls,float* means,int n,int width) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x) {
        int count=controls[2*row],cursor=controls[2*row+1];
        int next_count=count+1<width?count+1:width;float old=0.f,updated=0.f;
        for(int k=0;k<width;++k) {
            float value=values[row*width+k];
            float next=k==cursor?forcing[3*row+2]:value;after[row*width+k]=next;
            old=old+(k<count?value-273.15f:0.f);
            updated=updated+(k<next_count?next-273.15f:0.f);
        }
        means[2*row]=count>0?273.15f+old/(float)count:forcing[3*row+1];
        means[2*row+1]=273.15f+updated/(float)next_count;
        next_controls[2*row]=next_count;next_controls[2*row+1]=(cursor+1)%width;
    }
}
