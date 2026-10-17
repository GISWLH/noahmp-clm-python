__device__ int mix_glacier_liquid(float* cells,int n,float* parcel) {
    int first=0;
    while(first<n) {
        bool any=parcel[0]>0.f;for(int a=2;a<7;++a)any=any||parcel[a]>0.f;if(!any)break;
        float v[7];for(int a=0;a<7;++a)v[a]=cells[7*first+a]+parcel[a];
        float liquid=fminf(v[0],fmaxf(v[1],0.f)/334000.f),mass=v[0]-liquid,ratio=liquid/v[0];
        parcel[0]=liquid;parcel[1]=fmaxf(v[1],0.f);cells[7*first]=mass;cells[7*first+1]=fminf(v[1],0.f);
        for(int a=2;a<7;++a){parcel[a]=v[a]*ratio;cells[7*first+a]=v[a]-parcel[a];}
        if(mass>0.f)break;++first;
    }
    for(int i=0;i<4;++i)for(int a=0;a<7;++a)cells[7*i+a]=i<n-first?cells[7*(i+first)+a]:0.f;
    return n-first;
}

__device__ void glacier_transfer_device(const float* input,int n,int mode,const float* u,float* out) {
    float cells[28],run=0.f,runh=0.f,amount=0.f,heat=0.f,latent=0.f,tracer[5]={0.f};
    for(int i=0;i<28;++i)cells[i]=input[i];
    if(mode==0) {
        float parcel[7];for(int i=0;i<7;++i)parcel[i]=u[i];n=mix_glacier_liquid(cells,n,parcel);
        run=parcel[0];runh=parcel[1];for(int a=0;a<5;++a)tracer[a]=parcel[2+a];
    } else if(u[0]>0.f) {
        float bound=(64.f*1.1920928955078125e-7f)*fmaxf(u[0],input[0]);
        amount=fabsf(u[0]-input[0])<=bound?input[0]:u[0];float ratio=amount/input[0];
        heat=ratio*input[1];latent=2834000.f*amount;cells[0]-=amount;cells[1]-=heat;
        if(cells[0]==0.f) {
            for(int a=0;a<5;++a){if(n>1)cells[9+a]+=cells[2+a];else tracer[a]=cells[2+a];}
            for(int i=0;i<4;++i)for(int a=0;a<7;++a)cells[7*i+a]=i<n-1?cells[7*(i+1)+a]:0.f;
            --n;
        }
    } else if(u[0]<0.f) {
        amount=u[0];
        if(u[1]<0.f) {heat=(amount*2100.f)*u[1];latent=2834000.f*amount;cells[0]-=amount;cells[1]-=heat;}
        else {
            heat=amount*(334000.f+4180.f*u[1]);latent=2.5e6f*amount;
            float parcel[7]={-amount,-heat,0.f,0.f,0.f,0.f,0.f};n=mix_glacier_liquid(cells,n,parcel);
            run=parcel[0];runh=parcel[1];for(int a=0;a<5;++a)tracer[a]=parcel[2+a];
        }
    }
    for(int i=0;i<28;++i)out[i]=cells[i];out[28]=(float)n;
    out[29]=run;out[30]=runh;out[31]=amount;out[32]=heat;out[33]=latent;
    for(int a=0;a<5;++a)out[34+a]=tracer[a];
}

extern "C" __global__ void run_glacier_transfer(const float* cells,const int* counts,const int* modes,
        const float* inputs,float* out,int size) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<size;row+=blockDim.x*gridDim.x)
        glacier_transfer_device(cells+28*row,counts[row],modes[row],inputs+7*row,out+39*row);
}
