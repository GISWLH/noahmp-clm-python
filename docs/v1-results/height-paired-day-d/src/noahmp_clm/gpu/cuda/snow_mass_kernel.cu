// Extensive snow transfer, fixed five-cell output and six-cell insertion.
__device__ void snow_mass_phase(float m,float h,float& offset,float& ice,float& liquid) {
    liquid=fminf(m,fmaxf(0.f,h/334000.f));ice=m-liquid;
    float safe=fmaxf(m,1e-30f);
    offset=h<0.f?h/(2100.f*safe):(h>334000.f*m?(h-334000.f*m)/(4180.f*safe):0.f);
}

__device__ int snow_mass_regrid(const float* cells,float* out) {
    float old[7]={0.f},edges[6]={0.f},z[5]={0.f};
    for(int i=0;i<6;++i)old[i+1]=old[i]+cells[9*i+2];
    float depth=old[6];int n=0;
    if(depth<=.03f)z[0]=depth;
    else if(depth<=.04f)z[0]=z[1]=depth/2.f;
    else if(depth<=.07f){z[0]=.02f;z[1]=depth-.02f;}
    else if(depth<=.12f){z[0]=.02f;z[1]=z[2]=(depth-.02f)/2.f;}
    else if(depth<=.18f){z[0]=.02f;z[1]=.05f;z[2]=depth-.07f;}
    else if(depth<=.29f){z[0]=.02f;z[1]=.05f;z[2]=z[3]=(depth-.07f)/2.f;}
    else if(depth<=.41f){z[0]=.02f;z[1]=.05f;z[2]=.11f;z[3]=depth-.18f;}
    else if(depth<=.64f){z[0]=.02f;z[1]=.05f;z[2]=.11f;z[3]=z[4]=(depth-.18f)/2.f;}
    else {z[0]=.02f;z[1]=.05f;z[2]=.11f;z[3]=.23f;z[4]=depth-.41f;}
    for(int i=0;i<5;++i){n+=z[i]>0.f;edges[i+1]=edges[i]+z[i];}
    edges[n]=depth;
    for(int p=0;p<45;++p)out[p]=0.f;
    for(int k=0;k<6;++k) {
        const float* v=cells+9*k;float span[5],total=0.f;int last=-1,containing=0;
        for(int i=0;i<5;++i) {
            span[i]=i<n?fmaxf(0.f,fminf(edges[i+1],old[k+1])-fmaxf(edges[i],old[k])):0.f;
            total+=span[i];if(span[i]>0.f)last=i;if(i<n&&edges[i+1]<=old[k])++containing;
        }
        if(last<0)last=min(containing,max(n-1,0));
        float content[9],remaining[9];
        for(int p=0;p<9;++p)content[p]=v[p];
        content[2]=0.f;content[3]=v[0]/fmaxf(v[3],1e-30f);
        for(int p=0;p<9;++p)remaining[p]=content[p];
        for(int i=0;i<5;++i)for(int p=0;p<9;++p) {
            float part=i==last?remaining[p]:(span[i]>0.f&&i<last?content[p]*(span[i]/fmaxf(total,1e-30f)):0.f);
            if(v[0]<=0.f)part=0.f;
            out[9*i+p]+=part;remaining[p]-=part;
        }
    }
    for(int i=0;i<5;++i){out[9*i+2]=z[i];out[9*i+3]=i<n?out[9*i]/fmaxf(out[9*i+3],1e-30f):54.5f;}
    return n;
}

__device__ float snow_receiver_volume(const float* v,float amount,float specific) {
    float offset,ice,liquid;snow_mass_phase(v[0]+amount,v[1]+amount*specific,offset,ice,liquid);
    return ice/917.f+liquid/1000.f;
}

__device__ void snow_transfer_device(const float* input,int n,int mode,const float* u,float* out) {
    float cells[54]={0.f},ledger[28]={0.f};
    for(int i=0;i<45;++i)cells[i]=input[i];
    bool remap=mode!=2;
    if(mode==1) {
        if(u[0]>0.f) {
            for(int i=44;i>=0;--i)cells[i+9]=cells[i];
            cells[0]=u[0];cells[1]=(u[0]*2100.f)*u[1];cells[2]=u[0]/u[2];cells[3]=u[3];
            ledger[0]=cells[0];ledger[1]=cells[1];
            for(int i=0;i<5;++i)cells[4+i]=ledger[2+i]=u[11+i];
        } else remap=false;
    } else if(mode==2) {
        float heat=u[0]*(334000.f+4180.f*u[1]);
        cells[0]+=u[0];cells[1]+=heat;ledger[0]=u[0];ledger[1]=heat;
    } else if(mode==3) {
        float water=0.f,heat=0.f,tracer[5]={0.f};
        for(int k=0;k<n;++k) {
            float* v=cells+9*k;v[0]+=water;v[1]+=heat;
            for(int p=0;p<5;++p)v[4+p]+=tracer[p];
            float offset,ice,liquid;snow_mass_phase(v[0],v[1],offset,ice,liquid);
            v[2]=fmaxf(v[2],ice/917.f);
            float pore=fmaxf(0.f,1.f-ice/(917.f*fmaxf(v[2],1e-30f)));
            float cap=((u[4]*pore)*v[2])*1000.f;
            float potential=ice==0.f?liquid:(pore>=u[5]?fmaxf(0.f,liquid-cap):0.f);
            float specific=334000.f+4180.f*offset;water=potential;
            if(k+1<n&&potential>0.f) {
                const float* next=cells+9*(k+1);float nt,ni,nl;
                snow_mass_phase(next[0],next[1],nt,ni,nl);
                if(1.f-ni/(917.f*fmaxf(next[2],1e-30f))<u[5])water=0.f;
                else if(snow_receiver_volume(next,potential,specific)>next[2]) {
                    float low=0.f,high=potential;
                    for(int j=0;j<48;++j) {
                        float mid=(low+high)*.5f;
                        if(snow_receiver_volume(next,mid,specific)<=next[2])low=mid;else high=mid;
                    }
                    water=low;
                }
            }
            float left=liquid-water,occupied=ice/917.f+left/1000.f;
            float overflow=ice==0.f?left:(occupied>v[2]?fmaxf(0.f,left-(v[2]-ice/917.f)*1000.f):0.f);
            float total=ice==0.f?liquid:water+overflow;heat=water*specific;
            for(int p=0;p<5;++p) {
                float fraction=fminf(1.f,(u[6+p]*total)/fmaxf(v[0],1e-30f));
                float exported=total==v[0]?v[4+p]:v[4+p]*fraction;
                float split=fminf(1.f,fmaxf(0.f,water/fmaxf(total,1e-30f)));
                tracer[p]=water==total?exported:(total>0.f?exported*split:0.f);
                ledger[16+p]+=exported-tracer[p];v[4+p]-=exported;
            }
            ledger[14]+=overflow;ledger[15]+=overflow*specific;
            v[0]-=total;v[1]-=total*specific;
            if(v[0]==0.f){v[1]=0.f;v[2]=0.f;}
        }
        ledger[7]=water;ledger[8]=heat;for(int p=0;p<5;++p)ledger[9+p]=tracer[p];
    } else if(mode==4) {
        float remaining=u[0],actual=0.f,energy=0.f,residue[5]={0.f};
        for(int k=0;k<n;++k) {
            float* v=cells+9*k;float offset,ice,liquid;
            snow_mass_phase(v[0],v[1],offset,ice,liquid);
            float loss=fminf(remaining,ice),heat=(loss*2100.f)*offset,mass=v[0]-loss;
            v[2]*=mass/fmaxf(v[0],1e-30f);v[0]=mass;v[1]=mass==0.f?0.f:v[1]-heat;
            for(int p=0;p<5;++p) {
                float a=v[4+p]+residue[p];residue[p]=mass==0.f?a:0.f;v[4+p]=mass==0.f?0.f:a;
            }
            remaining-=loss;actual+=loss;energy+=heat;
        }
        ledger[21]=actual;ledger[22]=energy;for(int p=0;p<5;++p)ledger[23+p]=residue[p];
    }
    if(remap)n=snow_mass_regrid(cells,out);
    else for(int i=0;i<45;++i)out[i]=cells[i];
    out[45]=(float)n;for(int i=0;i<28;++i)out[46+i]=ledger[i];
}

extern "C" __global__ void run_snow_transfer(const float* cells,const int* counts,const int* modes,
                                              const float* inputs,float* output,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        snow_transfer_device(cells+row*45,counts[row],modes[row],inputs+row*16,output+row*74);
}
