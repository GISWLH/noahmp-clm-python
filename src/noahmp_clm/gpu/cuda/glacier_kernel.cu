// Included after enthalpy, joint heat, glacier heat and snow mass kernels.
// Output: snow 45, ice 28, counts 2, diagnostics 15, thermal equation 77.
__device__ void glacier_device(const float* snow,const float* ice,const int* count,
        const float* bc,const float* p,const float* q,float* out) {
    int ns=count[0],ni=count[1],n=ns+ni;
    float cells[45]={0.f},heated[45],thermal[77],transfer[74];
    for(int i=0;i<9;++i) {
        float m=0.f,h=0.f,z=1.f,k=1.f,solar=0.f;
        if(i<ns) {
            m=snow[9*i];h=snow[9*i+1];z=snow[9*i+2];solar=q[i];
            float rho=fminf(m/z,917.f);
            k=.023f+((7.75e-5f*rho)+(1.105e-6f*rho)*rho)*(float)(2.29-.023);
        } else if(i<n) {
            m=ice[7*(i-ns)];h=ice[7*(i-ns)+1];z=m/917.f;k=p[0];solar=q[5+i-ns];
        }
        if(i==n-1)solar+=p[1];
        cells[5*i]=m;cells[5*i+1]=h;cells[5*i+2]=z;cells[5*i+3]=k;cells[5*i+4]=solar;
    }
    glacier_heat_device(cells,n,bc,thermal);
    for(int i=0;i<45;++i)heated[i]=snow[i];
    for(int i=0;i<5;++i)heated[9*i+1]=i<ns?thermal[i]:0.f;
    float u[16]={0.f,0.f,100.f,54.5f,.033f,.05f,.2f,.02f,.02f,.01f,.01f,0.f,0.f,0.f,0.f,0.f};
    snow_transfer_device(heated,ns,3,u,transfer);
    float next[28]={0.f},im[7]={0.f};int nice=0;
    for(int i=0;i<ni;++i) {
        float m=ice[7*i],h=thermal[ns+i],liquid=fminf(m,fmaxf(h,0.f)/334000.f);
        float remain=m-liquid,ratio=liquid/m;
        im[0]+=liquid;im[1]+=fmaxf(h,0.f);
        if(remain>0.f){next[7*nice]=remain;next[7*nice+1]=fminf(h,0.f);}
        for(int a=0;a<5;++a) {
            float tracer=ice[7*i+2+a]*ratio;im[2+a]+=tracer;
            if(remain>0.f)next[7*nice+2+a]=ice[7*i+2+a]-tracer;
        }
        if(remain>0.f)++nice;
    }
    float parcel[7];for(int a=0;a<7;++a)parcel[a]=transfer[53+a];int first=0;
    while(first<nice) {
        bool any=parcel[0]>0.f;for(int a=2;a<7;++a)any=any||parcel[a]>0.f;
        if(!any)break;
        float v[7];for(int a=0;a<7;++a)v[a]=next[7*first+a]+parcel[a];
        float liquid=fminf(v[0],fmaxf(v[1],0.f)/334000.f),remain=v[0]-liquid,ratio=liquid/v[0];
        parcel[0]=liquid;parcel[1]=fmaxf(v[1],0.f);next[7*first]=remain;next[7*first+1]=fminf(v[1],0.f);
        for(int a=2;a<7;++a){parcel[a]=v[a]*ratio;next[7*first+a]=v[a]-parcel[a];}
        if(remain>0.f)break;
        ++first;
    }
    for(int a=0;a<45;++a)out[a]=transfer[a];
    for(int i=0;i<4;++i)for(int a=0;a<7;++a)out[45+7*i+a]=i<nice-first?next[7*(i+first)+a]:0.f;
    out[73]=transfer[45];out[74]=(float)(nice-first);
    for(int a=0;a<7;++a)out[75+a]=(im[a]+transfer[60+a])+parcel[a];
    float old[2]={0.f},now[2]={0.f},abs_h=0.f,newice=0.f,oldice=0.f;
    for(int i=0;i<5;++i)for(int a=0;a<2;++a){old[a]+=snow[9*i+a];now[a]+=out[9*i+a];}
    for(int i=0;i<4;++i)for(int a=0;a<2;++a){old[a]+=ice[7*i+a];now[a]+=out[45+7*i+a];}
    for(int i=0;i<9;++i)abs_h+=fabsf(cells[5*i+1]);
    for(int i=0;i<4;++i){newice+=out[45+7*i];oldice+=ice[7*i];}
    out[82]=im[0];out[83]=transfer[53];out[84]=newice-(oldice-im[0]);out[85]=p[1]*bc[0];
    out[86]=(now[0]-old[0])+out[75];out[87]=((now[1]-old[1])+out[76])-thermal[71];
    out[88]=(64.f*1.1920928955078125e-7f)*fmaxf(old[0]+out[75],1.f);
    out[89]=(64.f*1.1920928955078125e-7f)*fmaxf((abs_h+fabsf(out[76]))+fabsf(thermal[71]),1.f)+(float)n*thermal[76];
    for(int i=0;i<77;++i)out[90+i]=thermal[i];
}

extern "C" __global__ void run_glacier(const float* snow,const float* ice,const int* count,
        const float* bc,const float* params,const float* absorbed,float* out,int size) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<size;row+=blockDim.x*gridDim.x)
        glacier_device(snow+45*row,ice+28*row,count+2*row,bc+4*row,params+4*row,absorbed+9*row,out+167*row);
}
