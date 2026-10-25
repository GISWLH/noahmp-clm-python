// Independent closed-column liquid Richards step.
// Storage is actual kg/m2. Coefficient limits never add stored water.
extern "C" __global__ void richards_controller_reciprocal(const float* value,float* result,int n) {
    int i=threadIdx.x;if(i<n)result[i]=1.f/value[i];
}

__device__ void richards_system(const float* u,const float* old,const float* p,
                               float dt,int n,float* residual,float* q,float* a) {
    float water[10],dw[10],head[10],dh[10],z[10],sum=0.f;
    for(int i=0;i<n*n;++i)a[i]=0.f;
    for(int i=0;i<n;++i) {
        const float* v=p+6*i;float dz=v[0],por=v[1],psi=v[3],b=v[4],ref=v[5];
        float maximum=(1000.f*dz)*por,safe=maximum>0.f?maximum:1.f;
        bool saturated=u[i]>maximum;water[i]=fminf(u[i],maximum);dw[i]=saturated?0.f:1.f;
        float relative=water[i]/safe,limited=fmaxf(relative,(.01f*ref)/(por>0.f?por:1.f));
        float raw=psi*powf(limited,-b),entry=fminf(psi,1e5f);
        bool floor=relative<limited||raw>=1e5f;
        head[i]=saturated?-entry+(((u[i]-maximum)*b)*entry)/safe:-fminf(raw,1e5f);
        dh[i]=saturated?(b*entry)/safe:(floor?0.f:(b*raw)/(water[i]>0.f?water[i]:1.f));
        sum+=dz;z[i]=sum-.5f*dz;a[i*n+i]=dw[i];residual[i]=water[i]-old[i];
    }
    for(int i=0;i<n-1;++i) {
        const float* v=p+6*i;const float* w=v+6;
        float theta=water[i]/(1000.f*v[0]),next=water[i+1]/(1000.f*w[0]);
        float dtheta=dw[i]/(1000.f*v[0]),dnext=dw[i+1]/(1000.f*w[0]);
        float reference=v[5]+w[5],relative=(theta+next)/reference,power=(v[4]+w[4])+3.f;
        float material=sqrtf(v[2])*sqrtf(w[2]),face=material*powf(relative,power);
        float dk=((material*power)*powf(relative,power-1.f))/reference;
        float distance=z[i+1]-z[i],gradient=1.f-(head[i+1]-head[i])/distance;
        q[i]=face*gradient;
        float left=((dk*dtheta)*gradient)+(face*dh[i])/distance;
        float right=((dk*dnext)*gradient)-(face*dh[i+1])/distance;
        float l=(1000.f*dt)*left,r=(1000.f*dt)*right;
        a[i*n+i]+=l;a[i*n+i+1]=r;a[(i+1)*n+i]=-l;a[(i+1)*n+i+1]-=r;
    }
    for(int i=0;i<n;++i) {
        float in=i>0?q[i-1]:0.f,out=i<n-1?q[i]:0.f;
        residual[i]+=(1000.f*dt)*(out-in);
    }
}

__device__ bool richards_gaussian(float* a,float* r,int n,float* x) {
    bool ok=true;
    for(int k=0;k<n;++k) {
        int pivot=k;
        for(int i=k+1;i<n;++i)if(fabsf(a[i*n+k])>fabsf(a[pivot*n+k]))pivot=i;
        for(int i=0;i<n;++i){float old=a[k*n+i];a[k*n+i]=a[pivot*n+i];a[pivot*n+i]=old;}
        float old=r[k];r[k]=r[pivot];r[pivot]=old;
        bool good=fabsf(a[k*n+k])>1e-30f;ok=ok&&good;
        float den=good?a[k*n+k]:1.f;
        for(int i=k+1;i<n;++i) {
            float factor=a[i*n+k]/den;
            for(int z=0;z<n;++z)a[i*n+z]-=factor*a[k*n+z];
            r[i]-=factor*r[k];
        }
    }
    for(int i=n-1;i>=0;--i) {
        float value=r[i];
        for(int z=i+1;z<n;++z)value-=a[i*n+z]*x[z];
        x[i]=value/(fabsf(a[i*n+i])>1e-30f?a[i*n+i]:1.f);ok=ok&&isfinite(x[i]);
    }
    return ok;
}

extern "C" __global__ void richards_implicit_step(const float* old,const float* guess,const float* p,
                                                float dt,float* output,int n) {
    if(blockIdx.x||threadIdx.x)return;
    float u[10],maximum[10],residual[10],q[10],a[100],total=0.f,oldtotal=0.f,scale=1.f;
    for(int i=0;i<n;++i){u[i]=guess[i];maximum[i]=(1000.f*p[6*i])*p[6*i+1];total+=maximum[i];oldtotal+=old[i];scale=fmaxf(scale,fabsf(old[i]));}
    float tolerance=fmaxf(1e-10f,(8.f*1.1920928955078125e-7f)*scale),norm=0.f;
    bool good=true,done=false;int count=0;
    for(int iteration=0;iteration<45;++iteration) {
        count=iteration+1;richards_system(u,old,p,dt,n,residual,q,a);norm=0.f;
        for(int i=0;i<n;++i)norm=fmaxf(norm,fabsf(residual[i]));
        if(norm<=tolerance){done=true;break;}
        float rhs[10],direction[10];for(int i=0;i<n;++i)rhs[i]=-residual[i];
        bool linear=richards_gaussian(a,rhs,n,direction),found=false;float factor=1.f;
        for(int line=0;line<35;++line) {
            float candidate[10];bool allowed=linear,all_sat=true;
            for(int i=0;i<n;++i){candidate[i]=u[i]+factor*direction[i];allowed=allowed&&isfinite(candidate[i])&&candidate[i]>=0.f;all_sat=all_sat&&candidate[i]>=maximum[i];}
            allowed=allowed&&!(all_sat&&oldtotal<total-1e-8f);
            if(allowed) {
                float trial[10],tq[10],ta[100],error=0.f;
                richards_system(candidate,old,p,dt,n,trial,tq,ta);
                for(int i=0;i<n;++i)error=fmaxf(error,fabsf(trial[i]));
                if(error<norm){for(int i=0;i<n;++i)u[i]=candidate[i];found=true;break;}
            }
            factor*=.5f;
        }
        if(!found){good=false;break;}
    }
    richards_system(u,old,p,dt,n,residual,q,a);norm=0.f;
    for(int i=0;i<n-1;++i)output[2*n+i]=(1000.f*dt)*q[i];
    for(int i=0;i<n;++i) {
        float in=i>0?output[2*n+i-1]:0.f,out=i<n-1?output[2*n+i]:0.f;
        float water=(old[i]+in)-out;output[i]=water;output[n+i]=u[i];
        float rounding=(8.f*1.1920928955078125e-7f)*fmaxf(maximum[i],1.f);
        good=good&&isfinite(water)&&water>=0.f&&water<=maximum[i]+rounding;norm=fmaxf(norm,fabsf(residual[i]));
    }
    output[3*n-1]=norm;output[3*n]=float(count);output[3*n+1]=good&&done?1.f:0.f;
}
