// Component entry uses the same device function as the full column.
extern "C" __global__ void run_wetbulb(const float *a,float *out,int n) {
    for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)
        wetbulb_values(a[3*i],a[3*i+1],a[3*i+2],out+2*i,out+2*i+1);
}
