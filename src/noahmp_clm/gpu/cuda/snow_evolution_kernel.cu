// Native compaction, trilinear aging table, dry/wet growth and deposition.
__device__ void snow_aging_parameters(float offset,float gradient,float density,const float* table,float* result) {
    float coord[3]={(fminf(0.f,fmaxf(-50.f,offset))+50.f)/5.f,
        fminf(300.f,fmaxf(0.f,gradient))/10.f,(fminf(400.f,fmaxf(50.f,density))-50.f)/50.f};
    float physical[3]={273.15f+offset,gradient,density},spacing[3]={5.f,10.f,50.f};
    for(int d=0;d<3;++d) {
        float tolerance=((8.f*1.1920928955078125e-7f)*fmaxf(fabsf(physical[d]),1.f))/spacing[d],nearest=nearbyintf(coord[d]);
        if(fabsf(coord[d]-nearest)<=tolerance)coord[d]=nearest;
    }
    int low[3]={min((int)floorf(coord[0]),9),min((int)floorf(coord[1]),29),min((int)floorf(coord[2]),6)};
    float f[3];for(int d=0;d<3;++d)f[d]=coord[d]-(float)low[d];
    for(int p=0;p<3;++p)result[p]=0.f;
    for(int corner=0;corner<8;++corner) {
        int hi[3]={corner&1,(corner>>1)&1,(corner>>2)&1};float w[3];
        for(int d=0;d<3;++d)w[d]=hi[d]?f[d]:1.f-f[d];
        float weight=(w[0]*w[1])*w[2];int index=((low[0]+hi[0])*31+low[1]+hi[1])*8+low[2]+hi[2];
        for(int p=0;p<3;++p)result[p]+=weight*table[p*2728+index];
    }
}

__device__ float snow_dry_growth(float radius,float dt,const float* parameters) {
    if(dt==0.f)return radius;
    float tau=parameters[0],kappa=parameters[1],rate=parameters[2];
    float y=(radius-54.5f)+tau,p=1.f+1.f/kappa,hours=dt/7200.f;
    float inc=(logf(fmaxf((p*rate)*hours,1e-30f))-logf(y))+logf(tau/y)/kappa;
    float soft=fmaxf(inc,0.f)+log1pf(expf(-fabsf(inc)));
    return radius+y*expm1f(soft/p);
}

__device__ void snow_fresh_properties(float offset,float wind,float* out) {
    float thermal=offset>-15.f?50.f+1.7f*powf(fmaxf(fminf(offset,2.f)+15.f,0.f),1.5f):
        (-3.833f*offset)-(.0333f*offset)*offset;
    float loading=wind>.1f?266.861f*powf((1.f+tanhf(wind/5.f))/2.f,8.8f):0.f;
    out[0]=fminf(917.f,fmaxf(30.f,thermal+loading));
    out[1]=54.5f+5.f*fminf(30.f,fmaxf(0.f,offset+30.f));
}

__device__ void snow_evolution_device(const float* c,const int* o,const float* u,const float* table,float* out) {
    int n=o[0],mode=o[1];bool do_age=mode==1||mode==3;
    float cells[54]={0.f},overburden=0.f;
    for(int i=0;i<45;++i)cells[i]=c[i];
    for(int i=46;i<73;++i)out[i]=0.f;
    for(int k=0;k<n;++k) {
        const float* v=c+9*k;float m=v[0],h=v[1],z=fmaxf(v[2],1e-30f),offset,ice,liquid;
        snow_mass_phase(m,h,offset,ice,liquid);float rho=m/z,safe=fmaxf(m,1e-30f);
        overburden+=m;
        if(mode==0||mode==3) {
            float c1=expf(-.046f*fmaxf(ice/z-175.f,0.f)),c2=liquid/z>.01f?2.f:1.f;
            float destructive=((2.777e-6f*c1)*c2)*expf(.04f*offset);
            float over=overburden-.5f*m;
            float viscosity=((4.f*7.62237e6f)*(rho/450.f))*expf((-.1f*offset)+.023f*rho)/(1.f+(60.f*liquid)/(1000.f*z));
            float rate=destructive+over/fmaxf(viscosity,1e-30f),volume=ice/917.f+liquid/1000.f;
            bool allowed=1.f-volume/z>.001f&&ice>.1f;
            float factor=expf(-u[0]*(allowed?rate:0.f));
            float melt=u[6+k]>0.f?ice/fmaxf(u[6+k],1e-30f):1.f;
            factor*=o[2]&&allowed?fminf(1.f,melt):1.f;
            cells[9*k+2]=fmaxf(v[2]*factor,volume);
        }
        float aging_density=m/fmaxf(cells[9*k+2],1e-30f);
        if(do_age) {
            float* parameters=out+51+3*k;snow_aging_parameters(offset,u[1+k],fmaxf(aging_density,1.f),table,parameters);
            float radius=snow_dry_growth(fmaxf(v[3],54.5f),u[0],parameters);
            float fraction=fminf(.1f,liquid/safe),wet=((422000.f*fraction)*fraction)*fraction/12.56637061435917295385f;
            radius=cbrtf((radius*radius)*radius+(3.f*wet)*u[0]);
            radius=snow_dry_growth(radius,u[0],parameters);
            if(o[3]){float frozen=fminf(u[11+k],ice)/safe;radius=1.f/((1.f-frozen)/radius+frozen/1000.f);}
            cells[9*k+3]=fminf(1500.f,fmaxf(30.f,radius));
            out[68+k]=offset<-50.f||offset>0.f||u[1+k]>300.f||aging_density<50.f||aging_density>400.f?1.f:0.f;
        }
    }
    if(mode==2)for(int p=0;p<5;++p) {
        float addition=u[16+p]*u[0];if(n>0)cells[4+p]+=addition;else out[46+p]=addition;
    }
    if(o[4]&&(mode==0||mode==3)&&n>0)n=snow_mass_regrid(cells,out);
    else for(int i=0;i<45;++i)out[i]=cells[i];
    out[45]=(float)n;snow_fresh_properties(u[21],u[22],out+66);
}

extern "C" __global__ void run_snow_evolution(const float* cells,const int* options,const float* inputs,
                                               const float* table,float* output,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        snow_evolution_device(cells+row*45,options+row*5,inputs+row*23,table,output+row*73);
}
