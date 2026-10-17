// Published CLM5 MOST profile integrals; one independent column per thread.
__device__ float cm_psi(float z,bool momentum){
    float x=powf(1.f-16.f*z,.25f);
    return momentum?2.f*logf((1.f+x)/2.f)+logf((1.f+x*x)/2.f)-2.f*atanf(x)+1.5707963267948966f:
        2.f*logf((1.f+x*x)/2.f);
}
__device__ float cm_primitive(float z,float inverse,bool momentum){
    float x=z*inverse,correction;
    if(x>=0.f)correction=x<=1.f?5.f*x:4.f+x+4.f*logf(x);
    else{
        float edge=momentum?-1.574f:-.465f;
        if(x>=edge)correction=-cm_psi(x,momentum);
        else{
            // Constants evaluated in double then rounded match the published
            // NumPy/JAX edge values; only the varying argument uses powf.
            float tail=momentum?1.14f*(powf(-x,1.f/3.f)-(float)pow(1.574,1./3.)):
                .8f*((float)pow(.465,-1./3.)-powf(-x,-1.f/3.f));
            correction=logf(edge/x)-cm_psi(edge,momentum)+tail;
        }
    }
    return logf(z)+correction;
}
__device__ void canopy_most_device(const float* g,const float* a,const float* s,float* out){
    float inverse=s[0]/(g[0]-g[3]),integral[3];
    for(int k=0;k<3;k++)integral[k]=cm_primitive(g[k]-g[3],inverse,k==0)-cm_primitive(g[k+4],inverse,k==0);
    float ustar=.4f*s[1]/integral[0],leaf=.01f*sqrtf(ustar/s[2]),sparse=expf(-s[3]);
    float coefficient=(float)(.4/.13)*powf(g[8]*ustar/1.5e-5f,-.45f);
    float ground=ustar*(coefficient*sparse+.004f*(1.f-sparse));
    float theta=a[0]+.0098f*g[1],virtual_t=theta*(1.f+.61f*a[1]);
    float pv=a[1]*a[2]/(.622f+.378f*a[1]),rho=(a[2]-.378f*pv)/(287.f*a[0]);
    float vf=((1.f+.61f*a[1])*a[4]/1004.f+.61f*theta*a[5])/rho;
    float buoyancy=9.81f*vf/virtual_t,conv=powf(fmaxf(0.f,buoyancy*g[7]),1.f/3.f);
    float wind=fmaxf(1.f,sqrtf(a[3]*a[3]+conv*conv));
    float raw=-(g[0]-g[3])*.4f*buoyancy/(a[6]*a[6]*a[6]);
    out[0]=fmaxf(-100.f,fminf(2.f,raw));out[1]=raw;out[2]=buoyancy;out[3]=conv;out[4]=wind;
    out[5]=rho;out[6]=theta;out[7]=ustar;out[8]=.4f*ustar/integral[1];out[9]=.4f*ustar/integral[2];
    out[10]=leaf;out[11]=ground;for(int k=0;k<3;k++)out[12+k]=integral[k];
}
extern "C" __global__ void run_canopy_most(const float* g,const float* a,const float* s,float* out,int n){
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<n;k+=blockDim.x*gridDim.x)
        canopy_most_device(g+9*k,a+7*k,s+4*k,out+15*k);
}
