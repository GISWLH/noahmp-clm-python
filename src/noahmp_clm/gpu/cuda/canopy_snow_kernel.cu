// Included after radiation_kernel.cu and snicar_kernel.cu.
// Reuse linear unit snow responses for downward direct/diffuse spectra.
__device__ void canopy_snow_device(const float* lai,const float* cells,int count,const float* inputs,
                                   const float* ice,const float* impurities,float* out) {
    float sw=inputs[0],fv=inputs[1],diff=inputs[2],mu=inputs[3];
    float boundary[7]={1.f,1.f,1.f,1.f,inputs[4],inputs[5],mu},unit[460];
    snow_shortwave_device(cells,count,boundary,ice,impurities,unit);
    float ad[2]={unit[50+23],0.f},af[2]={unit[50+5*26+23],0.f};
    for(int band=1;band<5;++band){ad[1]+=unit[50+band*26+23];af[1]+=unit[50+(5+band)*26+23];}
    float beam=(.5f*sw)*(1.f-diff),sky=(.5f*sw)*diff,canopy[92];
    for(int band=0;band<2;++band) {
        float u[6]={beam,sky,af[band],mu,band==0?.1f:.45f,band==0?.05f:.25f};
        shortwave_device_albedos(lai,u,ad[band],true,canopy+band*46);
        boundary[band]=fv*canopy[band*46+20]+(1.f-fv)*beam;
        boundary[2+band]=fv*canopy[band*46+31]+(1.f-fv)*sky;
    }
    float* snow=out+124;
    for(int i=0;i<50;++i)snow[i]=0.f;
    for(int i=45;i<50;++i)snow[i]=unit[i];
    for(int i=310;i<460;++i)snow[i]=unit[i];
    for(int mode=0;mode<2;++mode)for(int band=0;band<5;++band) {
        float incoming=boundary[2*mode+(band==0?0:1)];int start=50+(mode*5+band)*26;
        for(int k=0;k<26;++k)snow[start+k]=unit[start+k]*incoming;
        for(int layer=0;layer<5;++layer)snow[5+layer*5+band]+=snow[start+layer];
        snow[30+band]+=snow[start+24];snow[35+band]+=snow[start+23];
    }
    float absorbed=0.f,ground=0.f,reflected_snow=0.f;
    for(int i=0;i<5;++i) {
        for(int band=0;band<5;++band)snow[i]+=snow[5+i*5+band];
        absorbed+=snow[i];ground+=snow[30+i];reflected_snow+=snow[35+i];
    }
    float incident=((boundary[0]+boundary[1])+boundary[2])+boundary[3];
    snow[40]=ground;snow[41]=reflected_snow;snow[42]=incident>0.f?reflected_snow/fmaxf(incident,1e-30f):0.f;
    snow[43]=((incident-reflected_snow)-absorbed)-ground;snow[44]=incident>0.f?1.f:0.f;
    float total=0.f;
    for(int i=0;i<10;++i) {
        out[i]=fv*(canopy[i]+canopy[46+i]);total+=out[i];
        out[10+i]=lai[i]>0.f?(canopy[i]*4.6f)/fmaxf(lai[i],1e-30f):0.f;
    }
    float bare0=ad[0]*beam+af[0]*sky,bare1=ad[1]*beam+af[1]*sky;
    float reflected=fv*(canopy[43]+canopy[46+43])+(1.f-fv)*(bare0+bare1);
    for(int i=0;i<5;++i)out[20+i]=snow[i];
    out[25]=ground;out[26]=reflected;out[27]=((sw-reflected)-total)-(absorbed+ground);
    for(int i=0;i<2;++i){out[28+i]=ad[i];out[30+i]=af[i];}
    for(int i=0;i<92;++i)out[32+i]=canopy[i];
}

extern "C" __global__ void run_canopy_snow(const float* area,const float* cells,const int* counts,
                                          const float* inputs,const float* ice,const float* impurities,float* output,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        canopy_snow_device(area+row*10,cells+row*35,counts[row],inputs+row*6,ice,impurities,output+row*584);
}
