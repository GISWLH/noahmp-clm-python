// Proposal-only float32 geometry; no new divider implementation or FP64.
__device__ __forceinline__ float geometry_div_full_f32(float a,float b){
  float result;
  asm volatile("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(a),"f"(b));
  return result;
}
__device__ __forceinline__ void geometry_default_f32(float liquid,float ice,
    float dz,float porosity,float* out){
  // Decimal literals are the actual default-JAX optimized-HLO f32 constants.
  // The diagnostic separately checks their bits against hardware RN(1/rho).
  out[0]=__fmul_rn(liquid,0.001f);
  out[1]=__fmul_rn(ice,0.00109051249f);
  out[2]=__fadd_rn(out[0],out[1]);
  out[3]=__fmul_rn(dz,porosity);
  out[4]=geometry_div_full_f32(out[2],out[3]);
  out[5]=__fmul_rn(917.f,dz);
  out[6]=__fmul_rn(out[5],porosity);
  out[7]=geometry_div_full_f32(ice,out[6]);
  out[8]=1.f<out[7]?1.f:out[7];
}
