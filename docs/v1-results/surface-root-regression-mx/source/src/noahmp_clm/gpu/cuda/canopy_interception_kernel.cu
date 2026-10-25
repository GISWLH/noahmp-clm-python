// Original extensive canopy rain/snow and vapour kernel; kg/m2, J/m2, K.
__device__ float cw_sum(const float* a){float x=0.f;for(int k=0;k<10;++k)x+=a[k];return x;}
__device__ void cw_phase(float m,float h,float& t,float& ice,float& liquid){
    liquid=fminf(m,fmaxf(0.f,h/334000.f));ice=m-liquid;
    float den=m>0.f?m:1.f,offset=h<0.f?h/(2100.f*den):(h>m*334000.f?(h-m*334000.f)/(4180.f*den):0.f);
    t=273.15f+(m>0.f?offset:0.f);
}
__device__ void canopy_water_device(const float* c,const float* area,const float* u,const float* v,int mode,float* out){
    for(int k=0;k<182;++k)out[k]=0.f;
    float mass[10],heat[10],m[10],h[10];for(int k=0;k<10;++k){mass[k]=c[2*k];heat[k]=c[2*k+1];}
    if(mode==0){
        float a[10];for(int k=0;k<10;++k)a[k]=area[k*2]+area[k*2+1];float total=cw_sum(a);
        // Same stable tanh identity as JAX; avoid backend-specific tanh ulps.
        float em=expm1f(-2.f*total),rain_fraction=(-em)/(2.f+em);
        float fi=u[4]*rain_fraction,fs=u[4]*(-expm1f(-.5f*total));
        float ri=fi*u[0],si=fs*u[1],rh=334000.f+4180.f*(u[2]-273.15f),ih=2100.f*(fminf(u[2],273.15f)-273.15f);
        float ph=u[0]*rh+u[1]*ih,dh[10],sh[10],drip[10],snowout[10];
        for(int k=0;k<10;++k){
            float weight=a[k]/(total>0.f?total:1.f),mm=mass[k]+weight*(ri+si),hh=heat[k]+weight*(ri*rh+si*ih);
            float t,ice,liq;cw_phase(mm,hh,t,ice,liq);float lh=hh>0.f?hh:0.f,hi=hh-lh;
            float cap=(.1f*u[4])*a[k],icap=(6.f*u[4])*a[k],kept_liquid=fminf(liq,cap),kept=fminf(ice,icap);drip[k]=liq-kept_liquid;
            float fall=ice-kept,rate=u[5]/1.56e5f+fmaxf(0.f,u[2]-u[6])/1.87e5f;
            float unload=kept*fminf(1.f,u[3]*rate);snowout[k]=fall+unload;
            float left=kept-unload,hl=(kept_liquid/(liq>0.f?liq:1.f))*lh,hs=(left/(ice>0.f?ice:1.f))*hi;
            dh[k]=lh-hl;sh[k]=hi-hs;m[k]=kept_liquid+left;h[k]=hl+hs;
            float ni,nl;cw_phase(m[k],h[k],t,ni,nl);
            float wet=cap>0.f?fminf(.05f,powf(m[k]/cap,(float)(2./3.))):0.f;
            out[50+k]=wet;out[60+k]=(1.f-wet)*(area[k*2]/(a[k]>0.f?a[k]:1.f));
            out[70+k]=icap>0.f?fminf(1.f,powf(ni/icap,.15f)):0.f;
            out[80+k]=drip[k];out[90+k]=fall;out[100+k]=unload;
        }
        float tr=u[0]-ri,ts=u[1]-si,gm=tr+cw_sum(drip),gi=ts+cw_sum(snowout),gh=tr*rh+cw_sum(dh),gih=ts*ih+cw_sum(sh);
        float water=((((cw_sum(m)-cw_sum(mass))-u[0])-u[1])+gm)+gi;
        float energy=(((cw_sum(h)-cw_sum(heat))-ph)+gh)+gih;
        float d[12]={ri,si,tr,ts,gm,gi,gh,gih,ph,water,energy,1.f};for(int k=0;k<12;++k)out[170+k]=d[k];
    }else{
        float actual[10],transport[10];
        for(int k=0;k<10;++k){
            float t,ice,liq;cw_phase(mass[k],heat[k],t,ice,liq);bool cold=v[k*2+1]<=273.15f;
            float available=cold?ice:liq,loss=fminf(fmaxf(v[k*2],0.f),available),remaining=available-loss;
            float deposit=fmaxf(-v[k*2],0.f),di=cold?deposit:0.f,dl=deposit-di;
            float oi=(cold?loss:0.f)-di,ol=(cold?0.f:loss)-dl,source=cold?fminf(heat[k],0.f):fmaxf(heat[k],0.f);
            float retained=(remaining/(available>0.f?available:1.f))*source;
            float added=(di*2100.f)*(v[k*2+1]-273.15f)+dl*(334000.f+4180.f*(v[k*2+1]-273.15f));
            float hx=(source-retained)-added;
            actual[k]=loss-deposit;transport[k]=hx;m[k]=(remaining+(cold?liq:ice))+deposit;h[k]=((heat[k]-source)+retained)+added;
            out[110+k]=actual[k];out[120+k]=oi;out[130+k]=ol;out[140+k]=hx;
            out[150+k]=2.5e6f*(oi+ol)+334000.f*oi;out[160+k]=fmaxf(v[k*2]-loss,0.f);
        }
        out[179]=(cw_sum(m)-cw_sum(mass))+cw_sum(actual);out[180]=(cw_sum(h)-cw_sum(heat))+cw_sum(transport);out[181]=1.f;
    }
    for(int k=0;k<10;++k){float t,ice,liq;cw_phase(m[k],h[k],t,ice,liq);out[2*k]=m[k];out[2*k+1]=h[k];out[20+k]=t;out[30+k]=ice;out[40+k]=liq;}
}
extern "C" __global__ void canopy_water_columns(const float* c,const float* area,const float* u,const float* v,const int* mode,float* out,int n){
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<n;k+=blockDim.x*gridDim.x)canopy_water_device(c+k*20,area+k*20,u+k*7,v+k*20,mode[k],out+k*182);
}
