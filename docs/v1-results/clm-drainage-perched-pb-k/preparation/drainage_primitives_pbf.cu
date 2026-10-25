// Isolated PB-f primitives. No column registration, phase inversion or stepping loop.
#include <math.h>
#include <float.h>
template<class T> __device__ T mn(T a,T b){return a<b?a:b;}
template<class T> __device__ T mx(T a,T b){return a>b?a:b;}
template<class T> __device__ T ab(T a){return a<0?-a:a;}
template<class T> __device__ T yield(T z,const T* p){
  return mx(p[9],-p[0]*expm1(-log1p(z/p[1])/p[2]));
}
template<class T> __device__ T integral(T a,T b,const T* p){
  if(b==a)return T(0);
  T lf=-p[2]*log1p(-p[9]/p[0]);
  if(lf>=log1p(b/p[1]))return p[9]*(b-a);
  T fd=p[1]*expm1(lf),split=mn(b,mx(a,fd)),len=b-split,power=T(1)-T(1)/p[2];
  T lr=log1p(len/(p[1]+split));
  T scaled=power==T(0)?lr:expm1(power*lr)/power;
  T retention=(exp(-log1p(split/p[1])/p[2])*(p[1]+split))*scaled;
  return p[9]*(split-a)+p[0]*(len-retention);
}
template<class T> __device__ T level(const T* sat,const T* z,const T* e,int count,
                                    T bottom,T threshold,bool main){
  int dry=-1;
  for(int k=0;k<4;k++)if(k<count&&sat[k]<=threshold)dry=k;
  if(main&&dry<=0)return mn(e[1],bottom);
  if(!main&&dry<0)return T(0);
  if(dry==count-1)return bottom;
  int j=dry<0?0:(dry>2?2:dry);T delta=sat[j+1]-sat[j];
  T f=(threshold-sat[j])/(delta==0?T(1):delta);
  return mn(bottom,mx(T(0),z[j]+f*(z[j+1]-z[j])));
}
template<class T> __device__ T rate(const T* c,const T* p,const T* e,const T* ice,
                                  T front,T bottom,bool perched,T& factor,T* w,int& fi){
  fi=0;int bi=0;for(int k=0;k<4;k++){fi+=e[k+1]<front;bi+=e[k+1]<bottom;}
  T num=0,den=0,thickness=bottom-front;
  for(int k=0;k<4;k++){
    bool chosen=perched?(k>=fi&&k<=bi):(k>=(fi>1?fi-1:0));
    w[k]=chosen?c[6*k+2]:T(0);
    T v=perched?pow(T(10),-p[11]*(T(.5)*(ice[k]+ice[k<3?k+1:3]))):ice[k];
    num=num+w[k]*v;den=den+w[k];
  }
  T avg=num/(den==0?T(1):den);
  factor=thickness==0?T(1):(perched?avg:pow(T(10),-p[11]*avg));
  if(perched){
    if(fi>=bi)return T(0);
    return (((T(1000)*p[7])*sin(p[5]))*(p[3]*factor))*thickness;
  }
  return ((p[6]*tan(p[5]))*thickness)*factor;
}
template<class T> __global__ void drainage(const T* cs,const T* ps,const T* us,
                                         T* ds,T* ts,int n){
  int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
  const T* c=cs+24*row;const T* p=ps+12*row;const T* u=us+3*row;
  T* d=ds+29*row;T* t=ts+26*row;
  T e[5]={0},z[4],sat[4],ice[4];int active=0;
  T eps=sizeof(T)==sizeof(float)?T(FLT_EPSILON):T(DBL_EPSILON);
  for(int k=0;k<4;k++)e[k+1]=e[k]+c[6*k+2];
  for(int k=0;k<4;k++){
    z[k]=e[k]+T(.5)*c[6*k+2];
    sat[k]=(c[6*k]/T(1000)+c[6*k+1]/T(917))/(c[6*k+2]*p[0]);
    ice[k]=mn(T(1),c[6*k+1]/((T(917)*c[6*k+2])*p[0]));
    active+=e[k+1]<=p[4]+T(16)*eps*mx(e[4],T(1));
  }
  T main=level(sat,z,e,active,p[4],p[8],true);int fi=-1,count=0;
  for(int k=1;k<4;k++)if(k<active&&c[6*(k-1)+3]>T(273.15)&&c[6*k+3]<=T(273.15)){
    if(fi<0)fi=k;count++;
  }
  T frost=fi>=0?z[fi]:T(0);
  T candidate=level(sat,z,e,fi+1,frost,p[8],false);
  bool has=fi>=0&&main>=frost&&candidate<frost;T perch=has?candidate:T(0);
  T mf,pf,mw[4],pw[4];int mi,pi;
  T mq=rate(c,p,e,ice,main,p[4],false,mf,mw,mi);
  T pq=rate(c,p,e,ice,perch,frost,true,pf,pw,pi);
  d[0]=main;d[1]=frost;d[2]=perch;d[3]=fi>=0;d[4]=has;d[5]=count;d[6]=active;
  d[7]=mq;d[8]=has?pq:T(0);d[9]=mf;d[10]=has?pf:T(1);
  d[11]=yield(main,p);d[12]=has?pi+1:-1;
  for(int k=0;k<4;k++){d[13+k]=mw[k];d[17+k]=has?pw[k]:T(0);d[21+k]=ice[k];d[25+k]=sat[k];}
  T factor,weights[4];int front_layer;
  T q=rate(c,p,e,ice,u[1],u[2],false,factor,weights,front_layer);
  T requested=q*u[0],remaining=requested,position=u[1];bool enabled=true,limited=false;
  for(int k=0;k<4;k++){
    T left=mx(position,e[k]),right=mn(u[2],e[k+1]);
    bool eligible=enabled&&right>left&&remaining>T(0);T safe_right=mx(right,left);
    T cap=T(1000)*integral(left,safe_right,p);
    T take=eligible?mn(remaining,mn(cap,mx(T(0),c[6*k]-p[10]))):T(0);
    T lo=left,hi=safe_right;
    for(int b=0;b<64;b++){
      T mid=T(.5)*(lo+hi),mass=T(1000)*integral(left,mid,p);
      if(mass<take)lo=mid;else hi=mid;
    }
    T el=ab(T(1000)*integral(left,lo,p)-take),eh=ab(T(1000)*integral(left,hi,p)-take);
    T moved=el<=eh?lo:hi;
    if(take==0)moved=left;else if(take>=cap)moved=safe_right;
    remaining=remaining-take;bool stop=eligible&&take<cap;
    limited=limited||(stop&&remaining>0);enabled=enabled&&!stop;
    if(eligible)position=moved;
    T heat=take*(T(334000)+T(4180)*(c[6*k+3]-T(273.15)));
    t[10+k]=take;t[14+k]=heat;t[18+k]=c[6*k+4]-take;t[22+k]=c[6*k+5]-heat;
  }
  T total=0,mr=0,hr=0,wm=0,hm=0;
  for(int k=0;k<4;k++){
    total=total+t[10+k];mr=mr+((t[18+k]-c[6*k+4])+t[10+k]);
    hr=hr+((t[22+k]-c[6*k+5])+t[14+k]);
    wm=wm+((ab(c[6*k+4])+ab(t[18+k]))+ab(t[10+k]));
    hm=hm+((ab(c[6*k+5])+ab(t[22+k]))+ab(t[14+k]));
  }
  T unmet=mx(T(0),requested-total);
  t[0]=position;t[1]=requested;t[2]=total;t[3]=limited?unmet:T(0);
  t[4]=limited?T(0):unmet;t[5]=limited;t[6]=mr;t[7]=hr;
  t[8]=T(64)*eps*mx(wm,T(1));t[9]=T(64)*eps*mx(hm,T(1));
}
