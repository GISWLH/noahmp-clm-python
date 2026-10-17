// One CH4-C pool, funded production and explicit competing sinks.
extern "C" __global__ void run_wetland(const float* old,const float* inputs,const float* params,float* output,int n) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=blockDim.x*gridDim.x) {
        const float* u=inputs+7*row;const float* p=params+13*row;float* out=output+18*row;
        float air=fmaxf((u[4]-u[2])-u[3],0.f),sat=fminf(u[2]/u[4],1.f);
        float anaerobic=u[2]>=u[4]?1.f:fmaxf((sat-p[1])/(1.f-p[1]),0.f);
        float ratio=p[2]*expf(logf(p[3]/p[4])*((u[1]-p[5])/10.f));
        float fraction=u[1]>273.15f?(anaerobic*ratio)/(1.f+ratio):0.f;
        float capacity=(u[2]+air/p[6])*p[0],aqueous=(p[7]*u[2])*u[2];
        float gaseous=(p[8]*expf(logf(air)*float(10./3.)))/(u[4]*u[4]);
        float conductance=(aqueous+gaseous/p[6])/(.5f*p[0]),kd=capacity>0.f?conductance/capacity:0.f;
        float ko=u[1]>273.15f&&u[2]>0.f?(p[9]*(1.f-anaerobic))*expf(logf(p[10])*((u[1]-p[11])/10.f)):0.f;
        float limit=(((12.f*p[6])*p[12])*u[5])/(float(8.31446261815324)*u[1]),maximum=limit*capacity;
        bool active=u[1]>273.15f&&anaerobic>0.f&&u[2]>0.f;
        float produced=u[0]*fraction,k=ko+kd,x=k*u[6],survival=expf(-x),lost_old=-expm1f(-x);
        float lost_new=x<1e-4f?x*(.5f+x*(float(-1./6.)+x*(float(1./24.)-x/120.f))):1.f-lost_old/x;
        float end=old[row]*survival+produced*(1.f-lost_new),loss=old[row]*lost_old+produced*lost_new;
        float oxidized=k>0.f?(loss*ko)/k:0.f,diffused=loss-oxidized;
        float bubble=active?fmaxf(end-maximum,0.f):0.f;end-=bubble;float emitted=diffused+bubble;
        float co2=(u[0]-produced)+oxidized;
        float cr=(((end-old[row])+co2)+emitted)-u[0],mr=(((end-old[row])+oxidized)+emitted)-produced;
        float scale=(((end+old[row])+co2)+emitted)+u[0],bound=128.f*1.1920928955078125e-7f*fmaxf(scale,1.f);
        bool valid=fabsf(cr)<=bound&&fabsf(mr)<=bound&&end>=0.f;
        float values[18]={end,anaerobic,fraction,kd,ko,maximum,(float)active,produced,oxidized,diffused,bubble,co2,emitted,
            emitted*float(16000./12.),cr,mr,bound,(float)valid};
        for(int k=0;k<18;++k)out[k]=values[k];
    }
}
