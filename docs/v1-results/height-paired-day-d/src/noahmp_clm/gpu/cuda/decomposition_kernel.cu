// Original extensive CENTURY-style C/N transfers; no host tendencies.
extern "C" __global__ void run_decomposition(const float* states,const float* inputs,const float* parameters,float* output,int count) {
    const int donor[10]={0,1,2,3,3,4,4,5,5,6},receiver[10]={4,4,5,1,2,5,6,4,6,4};
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<count;row+=blockDim.x*gridDim.x) {
        const float* old=states+15*row;const float* u=inputs+20*row;const float* p=parameters+16*row;
        float* result=output+70*row;float c[7],n[7],fraction[7],nc[7],nn[7];
        float mineral=old[14]+u[18],tf=expf(logf(p[12])*((u[0]-p[13])/10.f));
        float frozen=(-334000.f*fmaxf(273.15f-u[0],0.f))/(1000.f*u[0]);float psi=fminf(u[1],frozen);
        float wf=psi<=p[14]?0.f:(psi>=p[15]?1.f:logf(p[14]/fminf(psi,p[15]))/logf(p[14]/p[15]));
        float modifier=(tf*wf)*u[3];float rf=.17f+.68f*p[10],share=.004f/(1.f-rf);
        float path[10]={1.f,1.f,1.f,p[11],1.f-p[11],1.f-share,share,float(.42/.45),float(.03/.45),1.f};
        float resp[10]={.55f,.5f,.5f,0.f,0.f,rf,rf,.55f,.55f,.55f};
        float co[10],ci[10],no[10],ni[10],demand[10],potential=0.f;
        for(int k=0;k<7;k++){c[k]=old[k]+u[4+k];n[k]=old[7+k]+u[11+k];nc[k]=c[k];nn[k]=n[k];fraction[k]=-expm1f((-modifier*u[2])/(p[k]*31536000.f));}
        for(int k=0;k<10;k++){
            int a=donor[k],b=receiver[k];co[k]=(c[a]*fraction[a])*path[k];no[k]=(n[a]*fraction[a])*path[k];
            ci[k]=co[k]*(1.f-resp[k]);ni[k]=b>=4?ci[k]/p[b+3]:no[k];demand[k]=ni[k]-no[k];potential+=fmaxf(demand[k],0.f);
        }
        float total=potential+u[19],limitation=total>0.f?fminf(1.f,mineral/fmaxf(total,1.1754943508222875e-38f)):1.f;
        float immob=0.f,nmin=0.f,rh=0.f,uptake=u[19]*limitation;
        for(int k=0;k<10;k++){
            float scale=demand[k]>0.f?limitation:1.f;co[k]*=scale;ci[k]*=scale;no[k]*=scale;ni[k]*=scale;
            float dn=ni[k]-no[k];immob+=fmaxf(dn,0.f);nmin+=fmaxf(-dn,0.f);rh+=co[k]-ci[k];
            nc[donor[k]]-=co[k];nc[receiver[k]]+=ci[k];nn[donor[k]]-=no[k];nn[receiver[k]]+=ni[k];
            result[15+k]=co[k];result[25+k]=ci[k];result[35+k]=no[k];result[45+k]=ni[k];
        }
        float sc=0.f,sn=0.f,oc=0.f,on=0.f,ic=0.f,in=0.f,bc=0.f,bn=0.f;bool valid=true;
        for(int k=0;k<7;k++){bc+=c[k];bn+=n[k];}
        const float eps=1.1920928955078125e-7f;
        for(int k=0;k<7;k++){
            valid=valid&&(nc[k]>=-64.f*eps*fmaxf(bc,1.f))&&(nn[k]>=-64.f*eps*fmaxf(bn,1.f));
            nc[k]=fmaxf(nc[k],0.f);nn[k]=fmaxf(nn[k],0.f);result[k]=nc[k];result[7+k]=nn[k];
            sc+=nc[k];sn+=nn[k];oc+=old[k];on+=old[7+k];ic+=u[4+k];in+=u[11+k];
        }
        float nm=fmaxf(0.f,(mineral-immob)-uptake)+nmin;result[14]=nm;
        float cr=((sc-oc)-ic)+rh,nr=(((((sn+nm)-on)-old[14])-in)-u[18])+uptake;
        float cb=64.f*eps*fmaxf(((sc+oc)+ic)+rh,1.f),nb=64.f*eps*fmaxf((((((sn+nm)+on)+old[14])+in)+u[18])+uptake,1.f);
        float d[15]={rh,rh/(12.0107e-6f*u[2]),tf,wf,modifier,potential,immob,nmin,uptake,limitation,cr,nr,cb,nb,valid?1.f:0.f};
        for(int k=0;k<15;k++)result[55+k]=d[k];
    }
}
