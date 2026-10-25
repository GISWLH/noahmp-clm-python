extern "C" __global__ void structure_columns(const float *inputs,const float *params,
        const int *flags,float *outputs,int n){
    for(int k=blockIdx.x*blockDim.x+threadIdx.x;k<n;k+=blockDim.x*gridDim.x){
        const float *c=inputs+6*k,*p=params+2*k;const int *o=flags+6*k;float *r=outputs+14*k;
        const int kind=o[0];const bool cn=kind==1||kind==2;
        const float loss=fmaxf(0.f,c[1]-c[2]);
        const float rate=kind==1?.5f:1.f,minimum=cn?(kind==1?.5f:.05f):0.f;
        const float factor=cn?1.f-(rate*c[3])/2592000.f:1.f;
        float sai;
        if(kind==0)sai=o[2]?c[5]:c[0];
        else if(kind==3){
            sai=.2f*c[2];if(o[3]||(o[4]&&c[0]>0.f))sai=.25f;
            if(o[5]&&c[2]==0.f)sai=0.f;
        }else sai=fmaxf(minimum,factor*c[0]+loss);
        const float height=p[0]-p[1],hidden=fminf(fmaxf(c[4]-p[1],0.f),height);
        const float short_fraction=c[4]>=.2f?0.f:1.f-c[4]/.2f;
        const float tall_fraction=c[4]>=p[0]?0.f:(c[4]<=p[1]?1.f:1.f-hidden/height);
        const float fraction=o[1]?short_fraction:tall_fraction;
        r[0]=c[0];r[1]=sai;r[2]=c[1];r[3]=c[2];r[4]=loss;r[5]=factor;r[6]=minimum;
        r[7]=c[2];r[8]=sai;r[9]=c[2]*fraction;r[10]=sai*fraction;
        r[11]=(c[2]+sai)*fraction;r[12]=fraction;r[13]=c[4];
    }
}
