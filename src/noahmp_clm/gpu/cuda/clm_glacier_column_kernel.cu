// Full finite glacier meteorology. Stages retain extensive stores on device.
extern "C" __global__ void prepare_glacier_column(const float* cs,const float* gs,const float* ps,const int* os,
        const float* sites,const float* fs,float* pcs,float* pgs,int* pos,float* preps,float* vels,int size) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<size;row+=blockDim.x*gridDim.x) {
        const float *c=cs+45*row,*g=gs+28*row,*f=fs+8*row,*site=sites+6*row;const int* o=os+4*row;
        float *pc=pcs+45*row,*pg=pgs+28*row,*prep=preps+9*row,*vel=vels+3*row;int* po=pos+4*row;
        for(int i=0;i<45;++i)pc[i]=c[i];for(int i=0;i<28;++i)pg[i]=g[i];for(int i=0;i<4;++i)po[i]=o[i];
        float total=f[7]*f[0],snow=total*snow_fraction(f[1],o[2],f[2],f[4]);
        float rain=(total-snow)*1000.f;snow*=1000.f;float offset=f[1]-273.15f;
        float rainh=rain*(334000.f+4180.f*offset),snowh=snow*(2100.f*fminf(offset,0.f));
        if(snow>0.f) {
            float fresh[2],transfer[74];snow_fresh_properties(offset,f[3],fresh);
            float u[16]={snow,fminf(offset,0.f),fresh[0],fresh[1],.033f,.05f,.2f,.02f,.02f,.01f,.01f,0.f,0.f,0.f,0.f,0.f};
            snow_transfer_device(pc,po[0],1,u,transfer);for(int i=0;i<45;++i)pc[i]=transfer[i];po[0]=(int)transfer[45];
        }
        float parcel[7]={0.f};
        if(rain>0.f) {
            if(po[0]>0){pc[0]+=rain;pc[1]+=rainh;}
            else {parcel[0]=rain;parcel[1]=rainh;po[1]=mix_glacier_liquid(pg,po[1],parcel);}
        }
        prep[0]=rain+snow;prep[1]=rainh+snowh;for(int i=0;i<7;++i)prep[2+i]=parcel[i];
        float momentum=logf(site[1]/site[0]),rough=site[0]*.1f,numerator=.16f*fmaxf(f[3],.3f);
        vel[0]=numerator/(momentum*logf(site[2]/rough));vel[1]=numerator/(momentum*logf(site[3]/rough));vel[2]=f[1];
    }
}

__device__ void glacier_surface_evaluate(const float* c,const float* g,const float* p,const int* o,
        const float* f,const float* vel,const float* q,float reflected,float t,float* out) {
    int ns=o[0],ni=o[1],count[2]={ns,ni};float dt=f[0],offset=t-273.15f;
    float rho=f[4]/((287.f*f[1])*(1.f+.61f*f[2]));
    float es=t<273.15f?expf(((9.550426f-5723.265f/t)+3.53068f*logf(t))-.00728332f*t):
        611.2f*expf((17.67f*offset)/(offset+243.5f));
    float humidity=(.622f*es)/(f[4]-.378f*es),supply=ns>0?c[0]:g[0];
    float demand=fminf(((rho*vel[1])*(humidity-f[2]))*dt,supply);
    float snow[45],ice[28],solar[9],tracer[5],gm,gh,le,flow[2]={0.f};
    for(int i=0;i<45;++i)snow[i]=c[i];for(int i=0;i<28;++i)ice[i]=g[i];for(int i=0;i<9;++i)solar[i]=q[i];
    if(ns>0) {
        float u[3]={demand,offset,0.f},v[56];snow_vapor_device(c,ns,u,v);count[0]=(int)v[45];
        for(int i=0;i<45;++i)snow[i]=v[i];gm=v[46];gh=v[47];le=v[55]/dt;
        for(int a=0;a<5;++a)tracer[a]=v[48+a];
        if(count[0]<ns) {
            for(int i=0;i<5;++i)solar[i]=i<4?q[i+1]:0.f;
            if(count[0]>0)solar[0]+=q[0];else solar[5]+=q[0];
        }
    } else {
        float u[7]={demand,offset,0.f,0.f,0.f,0.f,0.f},v[39];glacier_transfer_device(g,ni,1,u,v);count[1]=(int)v[28];
        for(int i=0;i<28;++i)ice[i]=v[i];gm=v[31];gh=v[32];le=v[33]/dt;flow[0]=v[29];flow[1]=v[30];
        for(int a=0;a<5;++a)tracer[a]=v[34+a];
        if(count[1]<ni){for(int i=5;i<9;++i)solar[i]=i<8?q[i+1]:0.f;if(count[1]>0)solar[5]+=q[5];}
    }
    float h=((rho*1004.f)*vel[0])*(t-vel[2]),t2=t*t,lw=p[3]*(f[6]-5.670374419e-8f*(t2*t2));
    float boundary[4]={dt,(lw-h)-le,0.f,0.f};
    if(count[1]>0)glacier_device(snow,ice,count,boundary,p,solar,out);
    else {for(int i=0;i<167;++i)out[i]=0.f;out[99]=nanf("");}
    out[167]=t;out[168]=out[99]-t;out[169]=count[1]>0&&out[165]>0.f?1.f:0.f;
    out[170]=gm;out[171]=gh;out[172]=le;out[173]=h;out[174]=(f[5]-reflected)+lw;
    out[175]=flow[0]+out[75];out[176]=flow[1]+out[76];
    for(int a=0;a<5;++a)out[177+a]=tracer[a]+out[77+a];out[182]=vel[0];out[183]=vel[1];
}

extern "C" __global__ void solve_glacier_surface(const float* cs,const float* gs,const float* ps,const int* os,
        const float* fs,const float* vels,const float* qs,const float* reflected,float* outputs,int size) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<size;row+=blockDim.x*gridDim.x) {
        const float *c=cs+45*row,*g=gs+28*row,*p=ps+4*row,*f=fs+8*row,*vel=vels+3*row,*q=qs+9*row;const int* o=os+4*row;
        float *out=outputs+188*row,scratch[184],offset,ice,liquid;
        if(o[0]>0)snow_mass_phase(c[0],c[1],offset,ice,liquid);else offset=g[1]/(2100.f*fmaxf(g[0],1e-30f));
        float center=fminf(350.f,fmaxf(200.f,offset+273.15f));
        glacier_surface_evaluate(c,g,p,o,f,vel,q,reflected[row],center,out);int evaluations=1;
        float resolution=(2.f*1.1920928955078125e-7f)*273.15f,tol=fmaxf(2e-7f,resolution);
        float left=center,right=center,fl=out[168],fr=fl,width=2.f;
        if(fabsf(out[168])>tol)for(int i=0;i<8;++i) {
            if(fl<0.f&&left>200.f){left=fmaxf(200.f,center-width);glacier_surface_evaluate(c,g,p,o,f,vel,q,reflected[row],left,scratch);fl=scratch[168];++evaluations;}
            if(fr>0.f&&right<350.f){right=fminf(350.f,center+width);glacier_surface_evaluate(c,g,p,o,f,vel,q,reflected[row],right,scratch);fr=scratch[168];++evaluations;}
            width*=2.f;
        }
        bool bracketed=fl>=0.f&&fr<=0.f;
        for(int i=0;i<60&&fabsf(out[168])>tol&&bracketed&&right-left>resolution;++i) {
            width=right-left;float trial=left-(fl*width)/(fr!=fl?fr-fl:1.f);
            trial=fminf(right-.05f*width,fmaxf(left+.05f*width,trial));
            if(trial<=left||trial>=right)trial=left+.5f*width;
            glacier_surface_evaluate(c,g,p,o,f,vel,q,reflected[row],trial,out);++evaluations;
            if(out[168]>0.f){left=trial;fl=out[168];}else {right=trial;fr=out[168];}
        }
        if(out[108]==0.f&&out[167]!=273.15f) {
            glacier_surface_evaluate(c,g,p,o,f,vel,q,reflected[row],273.15f,scratch);++evaluations;
            if(fabsf(scratch[168])<=tol)for(int i=0;i<184;++i)out[i]=scratch[i];
        }
        out[184]=(float)evaluations;out[185]=left;out[186]=right;
        out[187]=((fabsf(out[168])<=tol||(bracketed&&right-left<=resolution))&&out[169]>0.f)?1.f:0.f;
    }
}

extern "C" __global__ void finish_glacier_column(const float* cs,const float* gs,const float* ps,const int* os,
        const float* fs,const float* deps,const float* preps,const float* surfaces,const float* aging,
        float* news,float* newg,int* newo,float* diagnostics,float* impurities,int size) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<size;row+=blockDim.x*gridDim.x) {
        const float *c=cs+45*row,*g=gs+28*row,*p=ps+4*row,*f=fs+8*row,*dep=deps+5*row,*prep=preps+9*row,*d=surfaces+188*row;
        float *snow=news+45*row,*ice=newg+28*row,*diag=diagnostics+22*row,*tracer=impurities+5*row;int* o=newo+4*row;
        int ns=(int)d[73],ni=(int)d[74];for(int i=0;i<4;++i)o[i]=os[4*row+i];o[0]=ns;o[1]=ni;
        for(int i=0;i<28;++i)ice[i]=d[45+i];float z[6],t[6],center[6],depth=0.f;
        for(int i=0;i<6;++i) {
            if(i<ns){z[i]=d[9*i+2];float frozen,liquid;snow_mass_phase(d[9*i],d[9*i+1],t[i],frozen,liquid);}
            else {z[i]=fmaxf(ice[0]/917.f,1e-30f);t[i]=ice[1]/(2100.f*fmaxf(ice[0],1e-30f));}
            depth+=z[i];center[i]=depth-.5f*z[i];
        }
        float u[23]={0.f},ev[73];int opt[5]={ns,3,0,0,0};u[0]=f[0];
        for(int i=0;i<5;++i) {
            int lo=max(i-1,0),hi=min(i+1,5);float h1=i==0?1.f:center[i]-center[lo],h2=fmaxf(center[hi]-center[i],1e-30f);
            float left=(t[i]-t[lo])/h1,right=(t[hi]-t[i])/h2,middle=(h2*left+h1*right)/(h1+h2);
            float value=i==0?right:(i==ns+(ni>0?1:0)-1?left:middle);
            u[1+i]=i<ns&&ns+(ni>0?1:0)>1?fabsf(value):0.f;
        }
        snow_evolution_device(d,opt,u,aging,ev);for(int i=0;i<45;++i)snow[i]=ev[i];
        for(int a=0;a<5;++a) {
            float addition=dep[a]*f[0];tracer[a]=d[177+a]+prep[4+a];
            snow[4+a]+=ns>0?addition:0.f;ice[2+a]+=ns==0&&ni>0?addition:0.f;
            tracer[a]+=ns==0&&ni==0?addition:0.f;
        }
        float ms0=0.f,mi0=0.f,hs0=0.f,hi0=0.f,ms1=0.f,mi1=0.f,hs1=0.f,hi1=0.f;
        for(int i=0;i<5;++i){ms0+=c[9*i];hs0+=c[9*i+1];ms1+=snow[9*i];hs1+=snow[9*i+1];}
        for(int i=0;i<4;++i){mi0+=g[7*i];hi0+=g[7*i+1];mi1+=ice[7*i];hi1+=ice[7*i+1];}
        float mass0=ms0+mi0,heat0=hs0+hi0,mass1=ms1+mi1,heat1=hs1+hi1;
        float run=prep[2]+d[175],runh=prep[3]+d[176],atm=((d[174]-d[173])-d[172])*f[0],geo=p[1]*f[0];
        float water=(((mass1-mass0)-prep[0])+d[170])+run,heat=(((((heat1-heat0)-prep[1])-atm)-geo)+d[171])+runh;
        float wm=(128.f*1.1920928955078125e-7f)*fmaxf((mass0+prep[0])+fabsf(d[170]),1.f),scale=0.f;
        for(int i=0;i<5;++i)scale+=fabsf(c[9*i+1]);for(int i=0;i<4;++i)scale+=fabsf(g[7*i+1]);
        scale+=fabsf(prep[1]);scale+=fabsf(atm);scale+=fabsf(geo);scale+=fabsf(d[171]);scale+=fabsf(runh);
        float wh=(128.f*1.1920928955078125e-7f)*fmaxf(scale,1.f)+d[89],ts=0.f;
        if(ns>0){float offset,frozen,liquid;snow_mass_phase(snow[0],snow[1],offset,frozen,liquid);ts=offset+273.15f;}
        else if(ni>0)ts=ice[1]/(2100.f*fmaxf(ice[0],1e-30f))+273.15f;
        float values[22]={ts,d[174],d[173],d[172],d[161]/f[0]-p[1],d[170],prep[0],run,ms1/1000.f,d[167],
            prep[1],runh,d[171],atm,geo,water,heat,wm,wh,d[184],d[182],d[183]};
        for(int i=0;i<22;++i)diag[i]=values[i];
    }
}
