// Original nonstructural reserves, maintenance, tissue allocation and litter.
__device__ float plant_sum(const float* a,int n){float v=0.f;for(int k=0;k<n;k++)v+=a[k];return v;}
__device__ void plant_budget(const float* old,const float* now,float ci,float co,float ni,float no,float* d){
    float oc=plant_sum(old,3)+old[6],nc=plant_sum(now,3)+now[6];
    float on=plant_sum(old+3,3)+old[7],nn=plant_sum(now+3,3)+now[7];
    float cr=((nc-oc)-ci)+co,nr=((nn-on)-ni)+no;
    const float eps=1.1920928955078125e-7f;
    float cb=64.f*eps*fmaxf(((oc+nc)+ci)+co,1.f),nb=64.f*eps*fmaxf(((on+nn)+ni)+no,1.f);
    bool valid=fabsf(cr)<=cb&&fabsf(nr)<=nb;
    for(int k=0;k<8;k++)valid=valid&&now[k]>=0.f&&isfinite(now[k]);
    d[0]=cr;d[1]=nr;d[2]=cb;d[3]=nb;d[4]=valid?1.f:0.f;
}
extern "C" __global__ void run_plant_turnover(const float* states,const float* inputs,const float* parameters,float* output,int count){
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<count;row+=blockDim.x*gridDim.x){
        const float* old=states+8*row;const float* u=inputs+2*row;const float* p=parameters+21*row;float* out=output+34*row;
        float fc[3],fn[3];
        for(int k=0;k<3;k++){
            float loss=-expm1f(-u[0]/(p[6+k]*86400.f));if(k==0)loss=1.f-(1.f-loss)*u[1];
            fc[k]=old[k]*loss;fn[k]=old[3+k]*loss;out[k]=old[k]-fc[k];out[3+k]=old[3+k]-fn[k];
            out[22+k]=fc[k];out[25+k]=fn[k];
        }
        float recovered=fmaxf(0.f,fn[0]-fc[0]/(p[20]*p[3]));out[6]=old[6];out[7]=old[7]+recovered;
        for(int k=0;k<7;k++){out[8+k]=0.f;out[15+k]=0.f;}
        for(int k=0;k<3;k++){out[8+k]=(fc[0]+fc[2])*p[9+k];out[15+k]=((fn[0]-recovered)+fn[2])*p[9+k];}
        out[11]=fc[1];out[18]=fn[1];out[28]=recovered;
        plant_budget(old,out,0.f,plant_sum(out+8,7),0.f,plant_sum(out+15,7),out+29);
    }
}
extern "C" __global__ void run_plant_plan(const float* states,const float* inputs,const float* parameters,float* output,int count){
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<count;row+=blockDim.x*gridDim.x){
        const float* old=states+8*row;const float* u=inputs+7*row;const float* p=parameters+21*row;float* out=output+24*row;
        float qair=expf(logf(p[14])*((u[2]-293.15f)/10.f)),qroot=expf(logf(p[14])*((u[3]-293.15f)/10.f));
        float demand[3]={u[1],(((old[4]*p[15])*p[13])*u[4])*qair,((old[5]*p[13])*u[4])*qroot};
        float cash=old[6]+u[0],total=plant_sum(demand,3);
        float fraction=total>0.f?fminf(1.f,cash/fmaxf(total,1.1754943508222875e-38f)):1.f;
        for(int k=0;k<8;k++)out[k]=old[k];
        for(int k=0;k<3;k++){out[17+k]=demand[k]*fraction;out[20+k]=demand[k]-out[17+k];}
        float available=fmaxf(0.f,cash-total),current=fmaxf(0.f,u[0]-total);
        float stored=fmaxf(0.f,available-current),release=-expm1f(-u[4]/(p[18]*86400.f));
        float budget=u[6]>0.f?((1.f-p[17])*current+stored*release):0.f,target[3];
        for(int k=0;k<3;k++){
            out[11+k]=p[k]*(budget/(1.f+p[12]));if(k==0)out[11]=fminf(out[11],fmaxf(0.f,u[5]-old[0]));
            out[14+k]=fmaxf(old[k]/p[3+k]-old[3+k],0.f);target[k]=out[11+k]/p[3+k];
        }
        out[8]=u[0];out[9]=u[4];out[10]=available;
        out[23]=fmaxf(0.f,(plant_sum(out+14,3)+plant_sum(target,3))-old[7]);
    }
}
extern "C" __global__ void run_plant_allocation(const float* plans,const float* awards,const float* parameters,float* output,int count){
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<count;row+=blockDim.x*gridDim.x){
        const float* plan=plans+24*row;const float* p=parameters+21*row;float award=awards[row];float* out=output+38*row;
        const float tiny=1.1754943508222875e-38f;
        float supply=plan[7]+award,deficit=plant_sum(plan+14,3),target[3];
        float repair_fraction=deficit>0.f?fminf(1.f,supply/fmaxf(deficit,tiny)):0.f;
        for(int k=0;k<3;k++){out[14+k]=plan[14+k]*repair_fraction;target[k]=plan[11+k]/p[3+k];}
        float available=fmaxf(0.f,supply-deficit),minimum=plant_sum(target,3)/p[16];
        float gf=minimum>0.f?fminf(1.f,available/fmaxf(minimum,tiny)):1.f;
        for(int k=0;k<3;k++){out[8+k]=plan[11+k]*gf;target[k]=out[8+k]/p[3+k];}
        float total=plant_sum(target,3),nf=total>0.f?fminf(1.f,available/fmaxf(total,tiny)):0.f;
        for(int k=0;k<3;k++){
            out[11+k]=target[k]*nf;out[20+k]=p[12]*out[8+k];out[17+k]=plan[17+k];out[23+k]=plan[20+k];
            out[k]=plan[k]+out[8+k];out[3+k]=(plan[3+k]+out[14+k])+out[11+k];
        }
        out[7]=fmaxf(0.f,available-total);out[6]=fmaxf(0.f,plan[10]-(plant_sum(out+8,3)+plant_sum(out+20,3)));
        float ra=plant_sum(plan+17,3)+plant_sum(out+20,3),factor=out[0]>0.f?fminf(1.f,(out[3]*p[3])/fmaxf(out[0],tiny)):0.f;
        out[26]=plan[8];out[27]=plan[8]-ra;out[28]=ra;out[29]=plan[23];out[30]=award;out[31]=factor;out[32]=plan[19]+out[22];
        plant_budget(plan,out,plan[8],ra,award,0.f,out+33);
    }
}
