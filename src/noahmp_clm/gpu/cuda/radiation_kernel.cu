// Ten-layer two-stream component, independent of the NumPy and JAX code.
// NVRTC device math is built in; no system includes are needed.
__device__ void shortwave_device_albedos(const float* lai, const float* u,
                                         float direct_albedo,bool separate_albedo,float* out,bool track_absorption=false) {
        float beam0=u[0], diffuse0=u[1], ag=u[2], mu=u[3], refl=u[4], tran=u[5];
        float k=.5f/mu, omega=refl+tran, back=.625f*refl+.375f*tran;
        // Explicit .5/k == mu avoids a second, separately rounded reciprocal.
        float beta=(1.f+k)*mu*(1.f-mu*logf(1.f+1.f/mu));
        float a=1.f-omega+back;
        float r[10],t[10],su[10],sd[10],bt[10],beam[11],da[10],ba[10];
        for(int j=0;j<10;++j) {
            float width=lai[j]/1024.f;
            t[j]=1.f/(1.f+a*width);
            r[j]=back*width*t[j];
            float intercepted=-expm1f(-k*width);
            bt[j]=1.f-intercepted;
            su[j]=omega*beta*intercepted;
            sd[j]=omega*(1.f-beta)*intercepted;
            da[j]=(1.f-omega)*width*t[j];ba[j]=(1.f-omega)*intercepted;
            for(int iter=0;iter<10;++iter) {
                float den=1.f-r[j]*r[j];
                float nr=r[j]+t[j]*t[j]*r[j]/den;
                float nt=t[j]*t[j]/den;
                float nu=su[j]+t[j]*(r[j]*sd[j]+su[j]*bt[j])/den;
                float nd=sd[j]*bt[j]+t[j]*(sd[j]+r[j]*su[j]*bt[j])/den;
                float nb=bt[j]*bt[j];
                if(track_absorption) {
                    float nda=da[j]*(1.f+t[j]/(1.f-r[j]));
                    float nba=ba[j]*(1.f+bt[j])+da[j]*(sd[j]+su[j]*bt[j])/(1.f-r[j]);
                    intercepted=intercepted*(2.f-intercepted);
                    nt=(1.f-nr)-nda;nb=1.f-intercepted;nd=(intercepted-nu)-nba;
                    da[j]=nda;ba[j]=nba;
                }
                r[j]=nr;t[j]=nt;su[j]=nu;sd[j]=nd;bt[j]=nb;
                if(omega==1.f) {t[j]=1.f-r[j];sd[j]=1.f-bt[j]-su[j];}
            }
        }
        beam[0]=beam0;
        for(int j=0;j<10;++j) {
            beam[j+1]=beam[j]*bt[j];
            su[j]*=beam[j];sd[j]*=beam[j];
        }
        float re[11],so[11],down[11],up[11],absorbed[10];
        re[10]=ag;so[10]=direct_albedo*beam[10];
        for(int j=9;j>=0;--j) {
            float den=1.f-r[j]*re[j+1];
            re[j]=r[j]+t[j]*t[j]*re[j+1]/den;
            so[j]=su[j]+t[j]*(so[j+1]+re[j+1]*sd[j])/den;
        }
        down[0]=diffuse0;
        for(int j=0;j<10;++j) {
            down[j+1]=(t[j]*down[j]+r[j]*so[j+1]+sd[j])/(1.f-r[j]*re[j+1]);
        }
        for(int j=0;j<11;++j)up[j]=re[j]*down[j]+so[j];
        float total=0.f;
        for(int j=0;j<10;++j) {
            absorbed[j]=beam[j]+down[j]+up[j+1]-beam[j+1]-down[j+1]-up[j];
            if(track_absorption)absorbed[j]=da[j]*(down[j]+up[j+1])+ba[j]*beam[j];
            total+=absorbed[j];
        }
        float ground=separate_albedo?(1.f-direct_albedo)*beam[10]+(1.f-ag)*down[10]:(1.f-ag)*(beam[10]+down[10]);
        float residual=beam0+diffuse0-up[0]-total-ground;
        for(int j=0;j<10;++j)out[j]=absorbed[j];
        for(int j=0;j<11;++j) {out[10+j]=beam[j];out[21+j]=down[j];out[32+j]=up[j];}
        out[43]=up[0];out[44]=ground;out[45]=residual;
}

__device__ void shortwave_device(const float* lai,const float* u,float* out) {
    shortwave_device_albedos(lai,u,u[2],false,out);
}

extern "C" __global__ void run_shortwave(const float* leaf, const float* inputs,
                                         float* output, int n) {
    for (int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        shortwave_device(leaf+10*row,inputs+6*row,output+46*row);
}

extern "C" __global__ void run_longwave(const float* leaf, const float* temperature,
                                        const float* inputs,float* output,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x) {
        const float* lai=leaf+10*row;
        const float* temp=temperature+10*row;
        const float* u=inputs+3*row;
        float transmission[10],emission[10],down[11],up[11];
        for(int j=0;j<10;++j) {
            transmission[j]=expf(-lai[j]);
            float square=temp[j]*temp[j];
            emission[j]=(1.f-transmission[j])*5.670374419e-8f*(square*square);
        }
        down[0]=u[0];
        for(int j=0;j<10;++j)down[j+1]=transmission[j]*down[j]+emission[j];
        float ground_square=u[1]*u[1];
        up[10]=(1.f-u[2])*down[10]+u[2]*5.670374419e-8f*(ground_square*ground_square);
        for(int j=9;j>=0;--j)up[j]=transmission[j]*up[j+1]+emission[j];
        float total=0.f;
        float* out=output+35*row;
        for(int j=0;j<10;++j) {
            out[j]=down[j]+up[j+1]-down[j+1]-up[j];
            total+=out[j];
        }
        for(int j=0;j<11;++j){out[10+j]=down[j];out[21+j]=up[j];}
        out[32]=down[10]-up[10];out[33]=up[0];out[34]=u[0]-up[0]-total-out[32];
    }
}
