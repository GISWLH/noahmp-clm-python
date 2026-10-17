// Native CUDA extensive soil thermodynamics; physical T is TF + offset.
// This is independent of the released AHC/Thomas temperature solver.
__device__ void cap_partials(float total,float potential,float &v,float &ds,float &dp) {
    float width=fmaxf(.04f*total,1e-12f),dw=.04f*total>1e-12f?.04f:0.f;
    float diff=total-potential,sign=diff>0.f?1.f:(diff<0.f?-1.f:0.f);
    float rounded=fmaxf(width-fabsf(diff),0.f);
    v=fminf(total,potential)-rounded*rounded/(4.f*width);
    ds=(diff<0.f?1.f:(diff>0.f?0.f:.5f))-rounded*(dw-sign)/(2.f*width);
    ds+=rounded*rounded*dw/(4.f*width*width);
    dp=(diff>0.f?1.f:(diff<0.f?0.f:.5f))-rounded*sign/(2.f*width);
    if(v<=0.f) {v=0.f;ds=0.f;dp=0.f;}
}

__device__ void soil_thermodynamics(float water,float offset,float dz,float por,float b,int option,
                                   float &h,float &c,float &liquid) {
    const float clausius=(float)(334000./461.5/(273.15*273.15));
    float theta=water/(1000.f*dz),suction=fmaxf(clausius*fmaxf(-offset,0.f),1e-8f);
    float potential=por*powf(suction/5e-5f,-1.f/b);
    float dp=clausius*fmaxf(-offset,0.f)>1e-8f?potential*clausius/(b*suction):0.f;
    float unused,dpot;cap_partials(theta,potential,liquid,unused,dpot);
    float derivative=dpot*dp;
    if(option==2) for(int j=0;j<3;++j) {
        float limited,ds,dl;cap_partials(liquid,potential,limited,ds,dl);
        float slope=ds*derivative+dl*dp;
        if(limited>=theta) slope=0.f;
        liquid=.5f*(liquid+fminf(theta,limited));derivative=.5f*(derivative+slope);
    }
    liquid=offset>=0.f?water:fminf(water,fmaxf(0.f,liquid*(1000.f*dz)));
    derivative=(offset>=0.f?0.f:derivative)*(1000.f*dz);
    float dry=((1.f-por)*2e6f)*dz;
    float sensible=(dry+2100.f*water)+2080.f*liquid;
    h=sensible*offset+334000.f*liquid;
    c=sensible+(2080.f*offset+334000.f)*derivative;
}

__device__ bool invert_soil_enthalpy(float water,float target,float dz,float por,float b,int option,
                                    float &t,float &h,float &c,float &liquid,float &count) {
    float lo=-123.15f,hi=126.85f,hlo,hhi;
    soil_thermodynamics(water,lo,dz,por,b,option,hlo,c,liquid);
    soil_thermodynamics(water,hi,dz,por,b,option,hhi,c,liquid);
    bool valid=target>=hlo&&target<=hhi;
    float dry=((1.f-por)*2e6f)*dz;
    t=fminf(hi,fmaxf(lo,(target-334000.f*water)/(dry+4180.f*water)));
    count=0.f;
    for(int iteration=0;iteration<64;++iteration) {
        soil_thermodynamics(water,t,dz,por,b,option,h,c,liquid);count+=1.f;
        float err=h-target;
        float tol=fmaxf(2e-8f,8.f*1.1920928955078125e-7f*fmaxf(1.f,fmaxf(fabsf(target),fabsf(334000.f*liquid))));
        if(fabsf(err)<=tol) return valid;
        if(err<0.f)lo=t;else if(err>0.f)hi=t;
        float trial=t-err/c;
        if(!(trial>lo&&trial<hi)) trial=.5f*(lo+hi);
        t=trial;
    }
    soil_thermodynamics(water,t,dz,por,b,option,h,c,liquid);
    float tol=fmaxf(2e-8f,8.f*1.1920928955078125e-7f*fmaxf(1.f,fmaxf(fabsf(target),fabsf(334000.f*liquid))));
    return valid&&fabsf(h-target)<=tol;
}

extern "C" __global__ void run_soil_thermodynamics(const float* inputs,const int* option,
                                                  float* output,int inverse,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x) {
        const float* u=inputs+5*row;float* out=output+9*row;
        float t=u[1],h,c,liq,count=0.f;bool ok=true;
        if(inverse) ok=invert_soil_enthalpy(u[0],u[1],u[2],u[3],u[4],option[row],t,h,c,liq,count);
        else soil_thermodynamics(u[0],t,u[2],u[3],u[4],option[row],h,c,liq);
        out[0]=t;out[1]=t+273.15f;out[2]=liq;out[3]=u[0]-liq;
        out[4]=h;out[5]=c;out[6]=inverse?h-u[1]:0.f;out[7]=count;out[8]=ok?1.f:0.f;
    }
}

// Included after enthalpy_kernel.cu; one thread advances one independent column.
__device__ bool heat_evaluate(const float* h,const float* old,const float* mass,const float* z,
                             const float* solar,const float* conduct,int ns,const float* props,const float* bc,
                             int phase,float* residual,float* t,float* derivative,float* liquid,float* flux,float &top,float* rounding) {
    bool valid=true;
    for(int i=0;i<9;++i) {
        t[i]=0.f;derivative[i]=0.f;liquid[i]=0.f;rounding[i]=0.f;
        if(i<ns) {
            float upper=334000.f*mass[i];
            if(h[i]<0.f) {t[i]=h[i]/(2100.f*mass[i]);derivative[i]=1.f/(2100.f*mass[i]);}
            else if(h[i]>upper) {t[i]=(h[i]-upper)/(4180.f*mass[i]);derivative[i]=1.f/(4180.f*mass[i]);}
            liquid[i]=fminf(mass[i],fmaxf(0.f,h[i]/334000.f));
            rounding[i]=(1.1920928955078125e-7f*(fabsf(h[i])+(h[i]>upper?upper:0.f)))*derivative[i]+
                (2.f*1.1920928955078125e-7f)*fabsf(t[i]);
        } else if(i<ns+4) {
            float reconstructed,capacity,count;
            bool ok=invert_soil_enthalpy(mass[i],h[i],z[i],props[0],props[1],phase,t[i],reconstructed,capacity,liquid[i],count);
            derivative[i]=1.f/capacity;valid=valid&&ok;
            rounding[i]=fabsf(reconstructed-h[i])/capacity+(2.f*1.1920928955078125e-7f)*fabsf(t[i]);
        }
    }
    for(int i=0;i<8;++i) flux[i]=conduct[i]*(t[i]-t[i+1]);
    top=bc[1]+bc[3]*(bc[2]-t[0]);
    for(int i=0;i<9;++i) {
        float from=i==0?top:flux[i-1],to=i==8?0.f:flux[i];
        residual[i]=i<ns+4?(h[i]-old[i])-bc[0]*((solar[i]+from)-to):0.f;
    }
    return valid;
}

__device__ float heat_tolerance(int i,const float* h,const float* tol,const float* rounding,const float* conduct,const float* bc) {
    float up=i==0?0.f:conduct[i-1],down=i==8?0.f:conduct[i];
    float err=up*(rounding[i]+(i==0?0.f:rounding[i-1]))+down*(rounding[i]+(i==8?0.f:rounding[i+1]));
    err+=i==0?bc[3]*rounding[i]:0.f;
    return fmaxf(tol[i],8.f*1.1920928955078125e-7f*fabsf(h[i]))+(2.f*bc[0])*err;
}

__device__ float heat_norm(const float* r,const float* h,const float* tol,const float* rounding,const float* conduct,const float* bc) {
    float v=0.f;for(int i=0;i<9;++i)v=fmaxf(v,fabsf(r[i])/heat_tolerance(i,h,tol,rounding,conduct,bc));return v;
}

__device__ void heat_linear(const float* d,const float* r,const float* conduct,const float* bc,float* delta) {
    float diag[9],upper[9],lower[9],rhs[9];
    for(int i=0;i<9;++i) {
        float up=i==0?0.f:conduct[i-1],down=i==8?0.f:conduct[i];
        diag[i]=1.f+(bc[0]*((up+down)+(i==0?bc[3]:0.f)))*d[i];
        lower[i]=-(bc[0]*up)*(i==0?0.f:d[i-1]);
        upper[i]=-(bc[0]*down)*(i==8?0.f:d[i+1]);rhs[i]=-r[i];
    }
    for(int i=1;i<9;++i) {
        float ratio=lower[i]/diag[i-1];diag[i]-=ratio*upper[i-1];rhs[i]-=ratio*rhs[i-1];
    }
    delta[8]=rhs[8]/diag[8];
    for(int i=7;i>=0;--i) delta[i]=(rhs[i]-upper[i]*delta[i+1])/diag[i];
}

__device__ void joint_heat_device(const float* cells,const float* props,const int* options,const float* bc,float* out) {
    int ns=options[0],phase=options[1];float por=props[0],quartz=props[2];
    float mass[9],old[9],h[9],z[9],solar[9],k[9],conduct[8],upperH[9];
    float tolerance[9],max_tolerance=0.f;
    for(int i=0;i<9;++i) {
        mass[i]=cells[5*i];old[i]=h[i]=cells[5*i+1];z[i]=cells[5*i+2];solar[i]=cells[5*i+4];
        upperH[i]=334000.f*mass[i];k[i]=cells[5*i+3];
        if(k[i]==0.f) {
            if(i<ns) {
                float rho=fminf(mass[i]/z[i],917.f);
                k[i]=.023f+((7.75e-5f*rho)+(1.105e-6f*rho)*rho)*(float)(2.29-.023);
            } else {
                float saturation=fminf(1.f,fmaxf(0.f,mass[i]/((1000.f*z[i])*por)));
                float kersten=saturation>.1f?fminf(1.f,fmaxf(0.f,logf(fmaxf(saturation,1e-6f))*(float).4342944819032518+1.f)):0.f;
                k[i]=.2f+((.6f+1.6f*quartz)-.2f)*kersten;
            }
        }
        float scale=fmaxf(fabsf(h[i]),fmaxf(fabsf(bc[0]*(solar[i]+(i==0?bc[1]:0.f))),1.f));
        tolerance[i]=fmaxf(2e-7f,8.f*1.1920928955078125e-7f*scale);max_tolerance=fmaxf(max_tolerance,tolerance[i]);
    }
    for(int i=0;i<8;++i) conduct[i]=i<ns+3?1.f/((.5f*z[i])/k[i]+(.5f*z[i+1])/k[i+1]):0.f;
    float residual[9],t[9],d[9],liq[9],flux[8],top,rounding[9];
    bool valid=heat_evaluate(h,old,mass,z,solar,conduct,ns,props,bc,phase,residual,t,d,liq,flux,top,rounding);
    bool done=heat_norm(residual,h,tolerance,rounding,conduct,bc)<=1.f;int count=0;
    while(count<80&&!done&&valid) {
        float merit=0.f,weights[9],delta[9],f0[9],f1[9];
        bool at0[9],at1[9],ev0[9],ev1[9];
        for(int i=0;i<9;++i) {
            // Hold merit weights fixed across a phase-event line search.
            weights[i]=heat_tolerance(i,h,tolerance,rounding,conduct,bc);
            merit+=fabsf(residual[i])/weights[i];
            at0[i]=i<ns&&h[i]==0.f;at1[i]=i<ns&&h[i]==upperH[i];
        }
        for(int a=0;a<12;++a) {
            heat_linear(d,residual,conduct,bc,delta);
            for(int i=0;i<ns;++i) {
                if(at0[i]) d[i]=delta[i]<0.f?1.f/(2100.f*mass[i]):0.f;
                if(at1[i]) d[i]=delta[i]>0.f?1.f/(4180.f*mass[i]):0.f;
            }
        }
        heat_linear(d,residual,conduct,bc,delta);float step=1.f;
        for(int i=0;i<9;++i) {
            f0[i]=-h[i]/(delta[i]!=0.f?delta[i]:1.f);f1[i]=(upperH[i]-h[i])/(delta[i]!=0.f?delta[i]:1.f);
            ev0[i]=i<ns&&!at0[i]&&delta[i]!=0.f&&f0[i]>0.f&&f0[i]<=1.f;
            ev1[i]=i<ns&&!at1[i]&&delta[i]!=0.f&&f1[i]>0.f&&f1[i]<=1.f;
            if(ev0[i])step=fminf(step,f0[i]);if(ev1[i])step=fminf(step,f1[i]);
        }
        float candidate[9],trial_r[9],trial_t[9],trial_d[9],trial_liq[9],trial_f[8],trial_top,trial_rounding[9];
        bool accepted=false;
        for(int search=0;search<40&&!accepted;++search) {
            for(int i=0;i<9;++i) {
                candidate[i]=h[i]+step*delta[i];
                if(ev0[i]&&step==f0[i])candidate[i]=0.f;
                else if(ev1[i]&&step==f1[i])candidate[i]=upperH[i];
            }
            bool good=heat_evaluate(candidate,old,mass,z,solar,conduct,ns,props,bc,phase,
                trial_r,trial_t,trial_d,trial_liq,trial_f,trial_top,trial_rounding);
            float trial_merit=0.f;for(int i=0;i<9;++i)trial_merit+=fabsf(trial_r[i])/weights[i];
            accepted=good&&(trial_merit<merit||heat_norm(trial_r,candidate,tolerance,trial_rounding,conduct,bc)<=1.f);
            if(accepted)for(int i=0;i<9;++i)h[i]=candidate[i];
            step*=.5f;
        }

        if(!accepted) {
            int worst=0;float peak=-1.f;
            for(int i=0;i<ns+4;++i){float ratio=fabsf(residual[i])/weights[i];if(ratio>peak){peak=ratio;worst=i;}}
            float derivative=d[worst];
            if(at0[worst])derivative=residual[worst]>0.f?1.f/(2100.f*mass[worst]):0.f;
            if(at1[worst])derivative=residual[worst]<0.f?1.f/(4180.f*mass[worst]):0.f;
            float up=worst==0?0.f:conduct[worst-1],down=worst==8?0.f:conduct[worst];
            float diagonal=1.f+(bc[0]*((up+down)+(worst==0?bc[3]:0.f)))*derivative;
            float correction=-residual[worst]/diagonal;
            float scalar_step=1.f;
            float e0=-h[worst]/(correction!=0.f?correction:1.f),e1=(upperH[worst]-h[worst])/(correction!=0.f?correction:1.f);
            bool land0=worst<ns&&!at0[worst]&&correction!=0.f&&e0>0.f&&e0<=1.f;
            bool land1=worst<ns&&!at1[worst]&&correction!=0.f&&e1>0.f&&e1<=1.f;
            if(land0)scalar_step=fminf(scalar_step,e0);if(land1)scalar_step=fminf(scalar_step,e1);
            for(int search=0;search<40&&!accepted;++search) {
                for(int i=0;i<9;++i)candidate[i]=h[i];
                candidate[worst]=h[worst]+scalar_step*correction;
                if(land0&&scalar_step==e0)candidate[worst]=0.f;
                else if(land1&&scalar_step==e1)candidate[worst]=upperH[worst];
                bool good=heat_evaluate(candidate,old,mass,z,solar,conduct,ns,props,bc,phase,
                    trial_r,trial_t,trial_d,trial_liq,trial_f,trial_top,trial_rounding);
                float trial_merit=0.f;for(int i=0;i<9;++i)trial_merit+=fabsf(trial_r[i])/weights[i];
                accepted=good&&(trial_merit<merit||heat_norm(trial_r,candidate,tolerance,trial_rounding,conduct,bc)<=1.f);
                if(accepted)for(int i=0;i<9;++i)h[i]=candidate[i];
                scalar_step*=.5f;
            }
        }
        bool good=heat_evaluate(h,old,mass,z,solar,conduct,ns,props,bc,phase,residual,t,d,liq,flux,top,rounding);
        done=heat_norm(residual,h,tolerance,rounding,conduct,bc)<=1.f;valid=good&&(accepted||done);count++;
    }
    float sum_sw=0.f,change=0.f;
    for(int i=0;i<9;++i) {
        sum_sw+=i<ns+4?solar[i]:0.f;change+=h[i]-old[i];
        out[i]=h[i];out[9+i]=i<ns+4?t[i]+273.15f:0.f;out[18+i]=t[i];
        out[27+i]=liq[i];out[36+i]=mass[i]-liq[i];out[45+i]=residual[i];
        out[62+i]=heat_tolerance(i,h,tolerance,rounding,conduct,bc);max_tolerance=fmaxf(max_tolerance,out[62+i]);
    }
    for(int i=0;i<8;++i)out[54+i]=flux[i];
    float incoming=bc[0]*(top+sum_sw);
    out[71]=incoming;out[72]=change-incoming;out[73]=top;out[74]=(float)(count+1);
    out[75]=done&&valid?1.f:0.f;out[76]=max_tolerance;
}

extern "C" __global__ void run_joint_heat(const float* cells,const float* props,const int* options,
                                          const float* boundary,float* out,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x)
        joint_heat_device(cells+45*row,props+3*row,options+2*row,boundary+4*row,out+77*row);
}
