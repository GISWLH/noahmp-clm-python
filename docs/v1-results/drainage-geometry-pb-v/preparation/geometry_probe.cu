// Independent diagnostic appended to the unchanged PB-t helper text.
// No production kernel, helper, option or arithmetic flag is replaced.
extern "C" __global__ void geometry_probe(const float* cs,const float* ps,
                                         float* ds,float* os,int n){
  int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
  typedef float T;
  const T* c=cs+24*row;const T* p=ps+12*row;
  T e[5]={0},ice[4];
  diagnosis(c,p,ds+29*row,e,ice);
  for(int k=0;k<4;k++){
    T* o=os+36*row+9*k;
    T liquid_divided=c[6*k]/T(1000);
    T ice_divided=c[6*k+1]/T(917);
    T saturation_numerator=liquid_divided+ice_divided;
    T saturation_denominator=c[6*k+2]*p[0];
    T saturation=saturation_numerator/saturation_denominator;
    T density_thickness=T(917)*c[6*k+2];
    T ice_denominator=density_thickness*p[0];
    T ice_quotient=c[6*k+1]/ice_denominator;
    T ice_fraction=mn(T(1),ice_quotient);
    o[0]=liquid_divided;o[1]=ice_divided;o[2]=saturation_numerator;
    o[3]=saturation_denominator;o[4]=saturation;o[5]=density_thickness;
    o[6]=ice_denominator;o[7]=ice_quotient;o[8]=ice_fraction;
  }
}
