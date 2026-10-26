// Independent diagnostic only; header is mechanically prepended by the worker.
extern "C" __global__ void geometry_default_probe(const float* cs,const float* ps,
    const float* previous,float* candidate,unsigned int* input_bits,
    unsigned int* candidate_bits,unsigned int* previous_bits,unsigned int* previous_equal,
    unsigned int* input_valid,unsigned int* output_finite,unsigned int* constant_bits,
    unsigned int* constant_equal,unsigned int* group_counts,int groups,int original_groups){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=4*groups)return;
  int group=i/4;const float* c=cs+24*group+6*(i%4);const float* p=ps+12*group;
  bool valid=isfinite(c[0])&&isfinite(c[1])&&isfinite(c[2])&&isfinite(p[0])
      &&c[0]>=0.f&&c[1]>=0.f&&c[2]>0.f&&p[0]>0.f;
  input_bits[4*i]=__float_as_uint(c[0]);input_bits[4*i+1]=__float_as_uint(c[1]);
  input_bits[4*i+2]=__float_as_uint(c[2]);input_bits[4*i+3]=__float_as_uint(p[0]);
  float values[9];geometry_default_f32(c[0],c[1],c[2],p[0],values);
  unsigned int mismatch=0;bool finite=true;
  for(int k=0;k<9;k++){
    int j=9*i+k;candidate[j]=values[k];candidate_bits[j]=__float_as_uint(values[k]);
    finite=finite&&isfinite(values[k]);
    unsigned int before=group<original_groups?__float_as_uint(previous[j]):0u;
    previous_bits[j]=before;
    previous_equal[j]=group<original_groups?(before==candidate_bits[j]?1u:0u):2u;
    if(group<original_groups)mismatch+=before!=candidate_bits[j];
  }
  unsigned int k1000=__float_as_uint(0.001f),k917=__float_as_uint(0.00109051249f);
  unsigned int r1000=__float_as_uint(__fdiv_rn(1.f,1000.f)),r917=__float_as_uint(__fdiv_rn(1.f,917.f));
  constant_bits[4*i]=k1000;constant_bits[4*i+1]=r1000;
  constant_bits[4*i+2]=k917;constant_bits[4*i+3]=r917;
  constant_equal[2*i]=k1000==r1000;constant_equal[2*i+1]=k917==r917;
  input_valid[i]=valid;output_finite[i]=finite;
  atomicAdd(group_counts+4*group,mismatch);
  atomicAdd(group_counts+4*group+1,valid?0u:1u);
  atomicAdd(group_counts+4*group+2,finite?0u:1u);
  atomicAdd(group_counts+4*group+3,(k1000!=r1000)+(k917!=r917));
}
