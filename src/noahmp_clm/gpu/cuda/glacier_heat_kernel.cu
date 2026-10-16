// Included after column_heat_kernel.cu: reuse its linear solve/error-bound helpers.
__device__ bool glacier_heat_evaluate(const float* h,const float* old,const float* mass,const float* z,
        const float* solar,const float* conduct,int n,const float* bc,
        float* residual,float* t,float* derivative,float* liquid,float* flux,float &top,float* rounding) {
    for(int i=0;i<9;++i) {
        t[i]=0.f;derivative[i]=0.f;liquid[i]=0.f;rounding[i]=0.f;
        if(i<n) {
            float upper=334000.f*mass[i];
            if(h[i]<0.f) {t[i]=h[i]/(2100.f*mass[i]);derivative[i]=1.f/(2100.f*mass[i]);}
            else if(h[i]>upper) {t[i]=(h[i]-upper)/(4180.f*mass[i]);derivative[i]=1.f/(4180.f*mass[i]);}
            liquid[i]=fminf(mass[i],fmaxf(0.f,h[i]/334000.f));
            rounding[i]=(1.1920928955078125e-7f*(fabsf(h[i])+(h[i]>upper?upper:0.f)))*derivative[i]+
                (2.f*1.1920928955078125e-7f)*fabsf(t[i]);
        }
    }
    for(int i=0;i<8;++i) flux[i]=conduct[i]*(t[i]-t[i+1]);
    top=bc[1]+bc[3]*(bc[2]-t[0]);
    for(int i=0;i<9;++i) {
        float from=i==0?top:flux[i-1],to=i==8?0.f:flux[i];
        residual[i]=i<n?(h[i]-old[i])-bc[0]*((solar[i]+from)-to):0.f;
    }
    return true;
}

__device__ void glacier_heat_device(const float* cells,int n,const float* bc,float* out) {
    float mass[9],old[9],h[9],z[9],solar[9],k[9],conduct[8],upperH[9];
    float tolerance[9],max_tolerance=0.f;
    for(int i=0;i<9;++i) {
        mass[i]=cells[5*i];old[i]=h[i]=cells[5*i+1];z[i]=cells[5*i+2];solar[i]=cells[5*i+4];
        upperH[i]=334000.f*mass[i];k[i]=cells[5*i+3];
        float scale=fmaxf(fabsf(h[i]),fmaxf(fabsf(bc[0]*(solar[i]+(i==0?bc[1]:0.f))),1.f));
        tolerance[i]=fmaxf(2e-7f,8.f*1.1920928955078125e-7f*scale);max_tolerance=fmaxf(max_tolerance,tolerance[i]);
    }
    for(int i=0;i<8;++i) conduct[i]=i<n-1?1.f/((.5f*z[i])/k[i]+(.5f*z[i+1])/k[i+1]):0.f;
    float residual[9],t[9],d[9],liq[9],flux[8],top,rounding[9];
    bool valid=glacier_heat_evaluate(h,old,mass,z,solar,conduct,n,bc,residual,t,d,liq,flux,top,rounding);
    bool done=heat_norm(residual,h,tolerance,rounding,conduct,bc)<=1.f;int count=0;
    while(count<80&&!done&&valid) {
        float merit=0.f,weights[9],delta[9],f0[9],f1[9];
        bool at0[9],at1[9],ev0[9],ev1[9];
        for(int i=0;i<9;++i) {
            // Hold merit weights fixed across a phase-event line search.
            weights[i]=heat_tolerance(i,h,tolerance,rounding,conduct,bc);
            merit+=fabsf(residual[i])/weights[i];
            at0[i]=i<n&&h[i]==0.f;at1[i]=i<n&&h[i]==upperH[i];
        }
        for(int a=0;a<12;++a) {
            heat_linear(d,residual,conduct,bc,delta);
            for(int i=0;i<n;++i) {
                if(at0[i]) d[i]=delta[i]<0.f?1.f/(2100.f*mass[i]):0.f;
                if(at1[i]) d[i]=delta[i]>0.f?1.f/(4180.f*mass[i]):0.f;
            }
        }
        heat_linear(d,residual,conduct,bc,delta);float step=1.f;
        for(int i=0;i<9;++i) {
            f0[i]=-h[i]/(delta[i]!=0.f?delta[i]:1.f);f1[i]=(upperH[i]-h[i])/(delta[i]!=0.f?delta[i]:1.f);
            ev0[i]=i<n&&!at0[i]&&delta[i]!=0.f&&f0[i]>0.f&&f0[i]<=1.f;
            ev1[i]=i<n&&!at1[i]&&delta[i]!=0.f&&f1[i]>0.f&&f1[i]<=1.f;
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
            bool good=glacier_heat_evaluate(candidate,old,mass,z,solar,conduct,n,bc,
                trial_r,trial_t,trial_d,trial_liq,trial_f,trial_top,trial_rounding);
            float trial_merit=0.f;for(int i=0;i<9;++i)trial_merit+=fabsf(trial_r[i])/weights[i];
            accepted=good&&(trial_merit<merit||heat_norm(trial_r,candidate,tolerance,trial_rounding,conduct,bc)<=1.f);
            if(accepted)for(int i=0;i<9;++i)h[i]=candidate[i];
            step*=.5f;
        }
        bool good=glacier_heat_evaluate(h,old,mass,z,solar,conduct,n,bc,residual,t,d,liq,flux,top,rounding);
        done=heat_norm(residual,h,tolerance,rounding,conduct,bc)<=1.f;valid=good&&(accepted||done);count++;
    }
    float sum_sw=0.f,change=0.f;
    for(int i=0;i<9;++i) {
        sum_sw+=i<n?solar[i]:0.f;change+=h[i]-old[i];
        out[i]=h[i];out[9+i]=i<n?t[i]+273.15f:0.f;out[18+i]=t[i];
        out[27+i]=liq[i];out[36+i]=mass[i]-liq[i];out[45+i]=residual[i];
        out[62+i]=heat_tolerance(i,h,tolerance,rounding,conduct,bc);max_tolerance=fmaxf(max_tolerance,out[62+i]);
    }
    for(int i=0;i<8;++i)out[54+i]=flux[i];
    float incoming=bc[0]*(top+sum_sw);
    out[71]=incoming;out[72]=change-incoming;out[73]=top;out[74]=(float)(count+1);
    out[75]=done&&valid?1.f:0.f;out[76]=max_tolerance;
}
