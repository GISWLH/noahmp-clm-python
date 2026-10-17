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

__device__ void soil_thermodynamics(float water,float offset,float dz,float por,float b,float psi,int option,
                                   float &h,float &c,float &liquid) {
    float temperature=273.15f+offset,theta=water/(1000.f*dz);
    float coefficient=334000.f/(9.80616f*psi);
    float raw=(coefficient*fmaxf(-offset,0.f))/temperature,ratio=fmaxf(raw,1e-12f);
    float potential=expf(fminf(logf(por)-logf(ratio)/b,logf(2.f*por)));
    float denominator=((b*temperature)*temperature)*ratio;
    float dp=raw>1e-12f?((potential*coefficient)*273.15f)/denominator:0.f;
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

__device__ bool invert_soil_enthalpy(float water,float target,float dz,float por,float b,float psi,int option,
                                    float &t,float &h,float &c,float &liquid,float &count) {
    float lo=-123.15f,hi=126.85f,hlo,hhi;
    soil_thermodynamics(water,lo,dz,por,b,psi,option,hlo,c,liquid);
    soil_thermodynamics(water,hi,dz,por,b,psi,option,hhi,c,liquid);
    bool valid=target>=hlo&&target<=hhi;
    float dry=((1.f-por)*2e6f)*dz;
    t=fminf(hi,fmaxf(lo,(target-334000.f*water)/(dry+4180.f*water)));
    count=0.f;
    for(int iteration=0;iteration<64;++iteration) {
        soil_thermodynamics(water,t,dz,por,b,psi,option,h,c,liquid);count+=1.f;
        float err=h-target;
        float tol=fmaxf(2e-8f,8.f*1.1920928955078125e-7f*fmaxf(1.f,fmaxf(fabsf(target),fabsf(334000.f*liquid))));
        if(fabsf(err)<=tol) return valid;
        if(err<0.f)lo=t;else if(err>0.f)hi=t;
        float trial=t-err/c;
        if(!(trial>lo&&trial<hi)) trial=.5f*(lo+hi);
        t=trial;
    }
    soil_thermodynamics(water,t,dz,por,b,psi,option,h,c,liquid);
    float tol=fmaxf(2e-8f,8.f*1.1920928955078125e-7f*fmaxf(1.f,fmaxf(fabsf(target),fabsf(334000.f*liquid))));
    return valid&&fabsf(h-target)<=tol;
}

extern "C" __global__ void run_soil_thermodynamics(const float* inputs,const int* option,
                                                  float* output,int inverse,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x) {
        const float* u=inputs+6*row;float* out=output+9*row;
        float t=u[1],h,c,liq,count=0.f;bool ok=true;
        if(inverse) ok=invert_soil_enthalpy(u[0],u[1],u[2],u[3],u[4],u[5],option[row],t,h,c,liq,count);
        else soil_thermodynamics(u[0],t,u[2],u[3],u[4],u[5],option[row],h,c,liq);
        out[0]=t;out[1]=t+273.15f;out[2]=liq;out[3]=u[0]-liq;
        out[4]=h;out[5]=c;out[6]=inverse?h-u[1]:0.f;out[7]=count;out[8]=ok?1.f:0.f;
    }
}
