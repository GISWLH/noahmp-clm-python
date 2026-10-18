// Layered interception adapters. Shared enthalpy-column helpers compile first.
__device__ void lc_interception(const float* cc,const float* site,const float* forcing,float rain,float snow,float* out,const float* geometry=nullptr) {
    float area[20]={0.f},vapor[20]={0.f};for(int k=0;k<10;++k){
        area[k*2]=(geometry?geometry[0]:site[0])/10.f;area[k*2+1]=geometry?geometry[1]/10.f:0.f;}
    float u[7]={rain,snow,forcing[1],forcing[0]/2.f,site[1],forcing[3],270.15f};
    canopy_water_device(cc,area,u,vapor,0,out);
}
__device__ bool lc_route(float* c,float* g,int& n,const float* p,int phase,const float* forcing,const float* ledger,float* runoff,const float* vic=nullptr) {
    if(ledger[5]>0.f) {
        float fresh[2];snow_fresh_properties(forcing[1]-273.15f,forcing[3],fresh);
        float offset=ledger[7]/(ledger[5]*2100.f);
        float u[16]={ledger[5],offset,fresh[0],fresh[1],.033f,.05f,.2f,.02f,.02f,.01f,.01f,0.f,0.f,0.f,0.f,0.f},out[74];
        snow_transfer_device(c,n,1,u,out);for(int k=0;k<45;++k)c[k]=out[k];n=(int)out[45];
    }
    return ec_route_water(c,g,n,p,phase,forcing[0],ledger[4],ledger[6],runoff,vic);
}

extern "C" __global__ void prepare_layered_transfer(const float* old_c,const float* old_g,const float* params,
    const int* options,const float* sites,const float* storage,const float* condensate,const float* forcing,
    float* snow_out,float* soil_out,int* options_out,float* canopy_out,float* totals,float* prepared,
    float* ledger_out,float* old_storage,int* validity,const float* vic_parameters,const float* ground_input,const float* geometry,int size) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<size;row+=blockDim.x*gridDim.x) {
        const float* p=params+7*row;const int* o=options+9*row;const float* site=sites+12*row;
        const float* f=forcing+8*row;const float* cc=condensate+20*row;
        float total=f[7]*f[0],sm=total*snow_fraction(f[1],o[2],f[2],f[4]),rain=(total-sm)*1000.f,snow=sm*1000.f;
        float* intercepted=ledger_out+182*row;lc_interception(cc,site,f,rain,snow,intercepted,geometry?geometry+2*row:nullptr);
        float c[45],g[12],runoff[2];for(int k=0;k<45;++k)c[k]=old_c[row*45+k];for(int k=0;k<12;++k)g[k]=old_g[row*12+k];
        int n=o[0];bool valid=lc_route(c,g,n,p,o[1],f,intercepted+170,runoff,vic_parameters?vic_parameters+8*row:nullptr);
        if(ground_input){float extra[2];bool good=ec_route_water(c,g,n,p,o[1],f[0],ground_input[2*row],ground_input[2*row+1],extra,vic_parameters?vic_parameters+8*row:nullptr);
            runoff[0]+=extra[0];runoff[1]+=extra[1];valid=valid&&good;}
        for(int k=0;k<45;++k)snow_out[row*45+k]=c[k];for(int k=0;k<12;++k)soil_out[row*12+k]=g[k];
        for(int k=0;k<9;++k)options_out[row*9+k]=k==0?n:o[k];
        for(int k=0;k<20;++k)canopy_out[row*20+k]=intercepted[k];
        totals[row*2]=ec_sum(intercepted,10,2);totals[row*2+1]=ec_sum(intercepted+1,10,2);
        float* prep=prepared+row*5;prep[0]=rain+snow;prep[1]=intercepted[178];prep[2]=runoff[0];prep[3]=runoff[1];prep[4]=0.f;
        if(ground_input){prep[0]+=ground_input[2*row];prep[1]+=ground_input[2*row+1];}
        old_storage[row*3]=ec_sum(cc,10,2);old_storage[row*3+1]=ec_sum(cc+1,10,2);old_storage[row*3+2]=storage[row*3+2];
        validity[row]=valid?1:0;
    }
}

extern "C" __global__ void finish_layered_transfer(const float* snow,const float* soil,const float* params,
    const int* options,const float* sites,const float* forcing,const float* prep_in,const int* counts,
    const float* condensate,const float* impurity_in,float* snow_out,float* soil_out,int* counts_out,
    float* canopy_out,float* totals,float* prep_out,float* impurity_out,float* ledger_out,int* validity,const float* vic_parameters,const float* geometry,int size) {
    for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<size;row+=blockDim.x*gridDim.x) {
        const float* p=params+row*7;const int* o=options+row*9;const float* f=forcing+row*8;
        const float* vic=vic_parameters?vic_parameters+8*row:nullptr;
        float* post=ledger_out+row*182;lc_interception(condensate+row*20,sites+row*12,f,0.f,0.f,post,geometry?geometry+2*row:nullptr);
        float c[45],g[12],runoff[2];for(int k=0;k<45;++k)c[k]=snow[row*45+k];for(int k=0;k<12;++k)g[k]=soil[row*12+k];
        int n=counts[row];bool valid=lc_route(c,g,n,p,o[1],f,post+170,runoff,vic);
        float u[16]={0.f,0.f,100.f,54.5f,.033f,.05f,.2f,.02f,.02f,.01f,.01f,0.f,0.f,0.f,0.f,0.f},drained[74];
        snow_transfer_device(c,n,3,u,drained);
        float su[7]={drained[53],drained[54],f[0],0.f,0.f,0.f,0.f},admit[47];int so[3]={0,o[1],1};
        column_soil_transport_device(g,p,so,su,admit,vic);valid=valid&&admit[46]>0.f;
        runoff[0]=(runoff[0]+drained[60])+admit[26];runoff[1]=(runoff[1]+drained[61])+admit[27];
        for(int k=0;k<5;++k)impurity_out[row*5+k]=impurity_in[row*5+k]+(drained[62+k]+drained[55+k]);
        for(int k=0;k<45;++k)snow_out[row*45+k]=drained[k];for(int k=0;k<12;++k)soil_out[row*12+k]=admit[k];
        counts_out[row]=(int)drained[45];for(int k=0;k<20;++k)canopy_out[row*20+k]=post[k];
        float m=ec_sum(post,10,2),h=ec_sum(post+1,10,2);totals[row*2]=m;totals[row*2+1]=h;
        float* prep=prep_out+row*5;for(int k=0;k<5;++k)prep[k]=prep_in[row*5+k];
        prep[2]+=runoff[0];prep[3]+=runoff[1];prep[4]=m;validity[row]=valid?1:0;
    }
}
