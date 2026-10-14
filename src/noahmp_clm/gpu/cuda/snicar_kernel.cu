// Native five-band SNICAR approximation; original thin-slab adding solver.
// Constants/tables are supplied data, not host-evaluated column optics.
__device__ void snow_slab(float tau,float omega,float asym,float mu,bool eddington,float* out) {
    float divisor=1.f-(omega*asym)*asym;
    tau*=divisor;float loss=(1.f-omega)/divisor,w=1.f-loss,g=asym/(1.f+asym);
    float a=eddington?(7.f-w*(4.f+3.f*g))/4.f:2.f-w*(1.f+g);
    float b=eddington?(w*(4.f-3.f*g)-1.f)/4.f:w*(1.f-g);
    float beta=eddington?(2.f-(3.f*g)*mu)/4.f:(1.f-g*mu)/2.f;
    float matrix[9]={-a,b,w*(1.f-beta)/mu,-b,a,-(w*beta)/mu,0.f,0.f,-1.f/mu};
    float norm=0.f;
    for(int i=0;i<3;++i)norm=fmaxf(norm,(fabsf(matrix[3*i])+fabsf(matrix[3*i+1]))+fabsf(matrix[3*i+2]));
    float magnitude=tau*norm/.1f;int exponent;
    float fraction=frexpf(fmaxf(magnitude,1.f),&exponent);
    int count=exponent-(fraction==.5f?1:0);float width=ldexpf(tau,-count);
    float term[9],transfer[9],small[9],next[9];
    float abs_term[3]={0.f,0.f,0.f},integral[3]={0.f,0.f,0.f},abs_next[3];
    for(int i=0;i<9;++i){term[i]=transfer[i]=i%4==0?1.f:0.f;small[i]=matrix[i]*width;}
    float source[3]={(2.f*loss)*width,(2.f*loss)*width,-loss*small[8]};
    for(int order=1;order<13;++order) {
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            next[3*i+j]=((term[3*i]*small[j]+term[3*i+1]*small[3+j])+term[3*i+2]*small[6+j])/(float)order;
        for(int j=0;j<3;++j)abs_next[j]=order==1?source[j]:((abs_term[0]*small[j]+abs_term[1]*small[3+j])+abs_term[2]*small[6+j])/(float)order;
        for(int j=0;j<3;++j){abs_term[j]=abs_next[j];integral[j]+=abs_next[j];}
        for(int i=0;i<9;++i){term[i]=next[i];transfer[i]+=next[i];}
    }
    float r=-transfer[3]/transfer[4],t=1.f/transfer[4],u=-transfer[5]/transfer[4];
    float d=transfer[2]+transfer[1]*u,beam=expf(-width/mu);
    float ad=integral[0]+integral[1]*r,ab=integral[2]+integral[1]*u;
    t=(1.f-r)-ad;d=((1.f-beam)-u)-ab;
    for(int i=0;i<count;++i) {
        float den=1.f-r*r,tt=t*t;
        float nr=r+tt*r/den;
        float nu=u+t*(r*d+u*beam)/den;
        float nb=beam*beam;
        float middle=(d+(r*u)*beam)/den;
        float nad=ad+ad*((1.f+r)*t)/den;
        float nab=ab*(1.f+beam)+ad*((1.f+r)*middle+u*beam);
        r=nr;t=(1.f-nr)-nad;u=nu;d=((1.f-nb)-nu)-nab;beam=nb;ad=nad;ab=nab;
    }
    out[0]=r;out[1]=t;out[2]=u;out[3]=d;out[4]=beam;
}

__device__ void snow_spectrum(const float* optics,const float* boundary,bool eddington,float* out) {
    float r[5],t[5],su[5],sd[5],beam[6],down[6],up[6],re[6],so[6];
    beam[0]=boundary[0];
    for(int i=0;i<5;++i) {
        float slab[5];snow_slab(optics[3*i],optics[3*i+1],optics[3*i+2],boundary[3],eddington,slab);
        r[i]=slab[0];t[i]=slab[1];su[i]=slab[2]*beam[i];sd[i]=slab[3]*beam[i];beam[i+1]=beam[i]*slab[4];
    }
    re[5]=boundary[2];so[5]=boundary[2]*beam[5];
    for(int i=4;i>=0;--i) {
        float den=1.f-r[i]*re[i+1];
        re[i]=r[i]+(t[i]*t[i])*re[i+1]/den;
        so[i]=su[i]+t[i]*(so[i+1]+re[i+1]*sd[i])/den;
    }
    down[0]=boundary[1];
    for(int i=0;i<5;++i)down[i+1]=((t[i]*down[i]+r[i]*so[i+1])+sd[i])/(1.f-r[i]*re[i+1]);
    for(int i=0;i<6;++i)up[i]=re[i]*down[i]+so[i];
    float total=0.f;
    for(int i=0;i<5;++i){out[i]=optics[3*i]==0.f?0.f:((((beam[i]+down[i])+up[i+1])-beam[i+1])-down[i+1])-up[i];total+=out[i];}
    for(int i=0;i<6;++i){out[5+i]=beam[i];out[11+i]=down[i];out[17+i]=up[i];}
    float ground=(1.f-boundary[2])*(beam[5]+down[5]);
    out[23]=up[0];out[24]=ground;out[25]=((boundary[0]+boundary[1]-up[0])-total)-ground;
}

__device__ void snow_shortwave_device(const float* cells,int count,const float* boundary,
                                       const float* ice,const float* impurities,float* out) {
    for(int i=0;i<50;++i)out[i]=0.f;
    const float weights[10]={1.f,.494f,.181f,.121f,.204f,1.f,.586f,.202f,.109f,.103f};
    for(int mode=0;mode<2;++mode)for(int band=0;band<5;++band) {
        float optical[15];
        for(int layer=0;layer<5;++layer) {
            const float* cell=cells+layer*7;
            float radius=fminf(1500.f,fmaxf(30.f,cell[1]));
            int lower=min(1469,(int)floorf(radius-30.f));float weight=radius-((float)lower+30.f),lookup[3];
            for(int prop=0;prop<3;++prop) {
                const float* table=ice+((mode*3+prop)*5+band)*1471;
                lookup[prop]=table[lower]+weight*(table[lower+1]-table[lower]);
            }
            float tau=cell[0]*lookup[0],scatter=tau*lookup[1],moment=scatter*lookup[2];
            for(int species=0;species<5;++species) {
                const float* data=impurities+species*15;
                float mass=cell[species+2]*data[band],scattering=mass*data[5+band];
                tau+=mass;scatter+=scattering;moment+=scattering*data[10+band];
            }
            optical[3*layer]=tau;optical[3*layer+1]=tau>0.f?scatter/fmaxf(tau,1e-30f):0.f;
            optical[3*layer+2]=scatter>0.f?moment/fmaxf(scatter,1e-30f):0.f;
            for(int prop=0;prop<3;++prop)out[310+((mode*5+band)*3+prop)*5+layer]=optical[3*layer+prop];
            out[45+layer]=layer<count&&cell[1]!=radius?1.f:0.f;
        }
        float flux=boundary[2*mode+(band==0?0:1)]*weights[5*mode+band];
        float bc[4]={mode==0?flux:0.f,mode==1?flux:0.f,boundary[4+(band==0?0:1)],boundary[6]};
        float* result=out+50+(mode*5+band)*26;
        snow_spectrum(optical,bc,band==0,result);
        for(int layer=0;layer<5;++layer)out[5+layer*5+band]+=result[layer];
        out[30+band]+=result[24];out[35+band]+=result[23];
    }
    float absorbed=0.f,ground=0.f,reflection=0.f;
    for(int i=0;i<5;++i) {
        for(int band=0;band<5;++band)out[i]+=out[5+i*5+band];
        absorbed+=out[i];ground+=out[30+i];reflection+=out[35+i];
    }
    float incoming=((boundary[0]+boundary[1])+boundary[2])+boundary[3];
    out[40]=ground;out[41]=reflection;out[42]=incoming>0.f?reflection/fmaxf(incoming,1e-30f):0.f;
    out[43]=((incoming-reflection)-absorbed)-ground;out[44]=incoming>0.f?1.f:0.f;
}

extern "C" __global__ void run_snow_shortwave(const float* cells,const int* counts,const float* boundary,
                                              const float* ice,const float* impurities,float* output,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        snow_shortwave_device(cells+row*35,counts[row],boundary+row*7,ice,impurities,output+row*460);
}

extern "C" __global__ void run_snow_spectrum(const float* optics,const float* boundary,const int* eddington,float* output,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        snow_spectrum(optics+row*15,boundary+row*4,eddington[row]!=0,output+row*26);
}
