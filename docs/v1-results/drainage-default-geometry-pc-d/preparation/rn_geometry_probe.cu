// Independent hardware RN diagnostic; not registered in the physical model.
static_assert(sizeof(unsigned int)==4,"Exact uint32 output ABI required");
extern "C" __global__ void rn_geometry_probe(const float* cs,const float* ps,const float* observed,
    unsigned int* observed_bits,unsigned int* strict_bits,unsigned int* operand_bits,
    unsigned int* strict_equal,unsigned int* operand_equal,unsigned int* valid_rows,
    unsigned int* group_counts,int groups){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=groups*4)return;
  int group=i/4;const float* c=cs+24*group+6*(i%4);
  const float* p=ps+12*group;const float* a=observed+9*i;
  bool valid=isfinite(c[0])&&isfinite(c[1])&&isfinite(c[2])&&isfinite(p[0])
      &&c[0]>=0.f&&c[1]>=0.f&&c[2]>0.f&&p[0]>0.f;
  for(int k=0;k<9;k++)valid=valid&&isfinite(a[k]);
  valid=valid&&a[3]>0.f&&a[6]>0.f;
  float strict[9],local[9];
  strict[0]=__fdiv_rn(c[0],1000.f);
  strict[1]=__fdiv_rn(c[1],917.f);
  strict[2]=__fadd_rn(strict[0],strict[1]);
  strict[3]=__fmul_rn(c[2],p[0]);
  strict[4]=__fdiv_rn(strict[2],strict[3]);
  strict[5]=__fmul_rn(917.f,c[2]);
  strict[6]=__fmul_rn(strict[5],p[0]);
  strict[7]=__fdiv_rn(c[1],strict[6]);
  strict[8]=1.f<strict[7]?1.f:strict[7];
  // Each local reference uses the actual captured backend operands, so an
  // upstream difference is not silently attributed to a later quotient.
  local[0]=__fdiv_rn(c[0],1000.f);
  local[1]=__fdiv_rn(c[1],917.f);
  local[2]=__fadd_rn(a[0],a[1]);
  local[3]=__fmul_rn(c[2],p[0]);
  local[4]=__fdiv_rn(a[2],a[3]);
  local[5]=__fmul_rn(917.f,c[2]);
  local[6]=__fmul_rn(a[5],p[0]);
  local[7]=__fdiv_rn(c[1],a[6]);
  local[8]=1.f<a[7]?1.f:a[7];
  unsigned int strict_misses=0,operand_misses=0;
  for(int k=0;k<9;k++){
    int j=9*i+k;unsigned int actual=__float_as_uint(a[k]);
    unsigned int sr=__float_as_uint(strict[k]),lr=__float_as_uint(local[k]);
    observed_bits[j]=actual;strict_bits[j]=sr;operand_bits[j]=lr;
    strict_equal[j]=actual==sr;operand_equal[j]=actual==lr;
    strict_misses+=actual!=sr;operand_misses+=actual!=lr;
  }
  valid_rows[i]=valid;
  atomicAdd(group_counts+3*group,strict_misses);
  atomicAdd(group_counts+3*group+1,operand_misses);
  atomicAdd(group_counts+3*group+2,valid?0u:1u);
}
