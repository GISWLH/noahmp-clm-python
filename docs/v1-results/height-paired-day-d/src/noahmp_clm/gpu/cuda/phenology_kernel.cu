// Seasonal-deciduous control; every state update is on the device.
extern "C" __global__ void run_phenology(const float* states,const int* options,const float* climate,
    const int* calendar,const float* parameters,float* state_out,int* options_out,float* diagnostics,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x) {
        const float* old=states+3*row;const float* u=climate+3*row;const float* p=parameters+6*row;
        float gdd=old[0],leaf=old[1],remaining=old[2];int phase=options[2*row],armed=options[2*row+1];
        bool increasing=calendar[2*row]>0,winter=calendar[2*row+1]>0;
        bool dormant=phase==0,accumulate=dormant&&increasing&&(armed||winter);
        if(dormant) {if(winter||!increasing)gdd=0.f;if(!increasing)armed=0;else if(winter)armed=1;}
        float increment=accumulate?fmaxf(0.f,u[0]-273.15f)*(u[2]/86400.f):0.f;
        gdd+=increment;bool onset=accumulate&&gdd>p[0];
        if(onset){phase=1;remaining=p[1]*86400.f;gdd=0.f;armed=0;}
        bool offset=phase==2&&!increasing&&u[1]<p[3];
        if(offset){phase=3;remaining=p[2]*86400.f;}
        if(phase==1||phase==3)remaining=fmaxf(0.f,remaining-u[2]);
        if(phase==1)leaf=1.f-remaining/(p[1]*86400.f);else if(phase==3)leaf=remaining/(p[2]*86400.f);
        if(phase==1&&remaining==0.f)phase=2;else if(phase==3&&remaining==0.f)phase=0;
        state_out[3*row]=gdd;state_out[3*row+1]=leaf;state_out[3*row+2]=remaining;
        options_out[2*row]=phase;options_out[2*row+1]=armed;
        float* d=diagnostics+6*row;d[0]=p[4]+(p[5]-p[4])*leaf;d[1]=fmaxf(0.f,leaf-old[1]);
        d[2]=fmaxf(0.f,old[1]-leaf);d[3]=increment;d[4]=onset?1.f:0.f;d[5]=offset?1.f:0.f;
    }
}
