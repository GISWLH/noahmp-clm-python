// New perched kernel; compile after unchanged F helper source.
template<class T> __global__ void perched(const T* cs,const T* ps,const T* us,
                                         T* ds,T* ts,int n){
  int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
  const T* c=cs+24*row;const T* p=ps+12*row;const T* u=us+3*row;
  T* d=ds+29*row;T* t=ts+29*row;
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

  T factor,weights[4];int front_layer,bi=0;
  T q=rate(c,p,e,ice,u[1],u[2],true,factor,weights,front_layer);
  for(int k=0;k<4;k++)bi+=e[k+1]<u[2];
  T sy=yield(main,p),requested=q*u[0],capacity=(T(1000)*sy)*(u[2]-u[1]);
  T desired=mn(requested,capacity),remaining=desired;
  for(int k=0;k<4;k++){
    bool eligible=q>T(0)&&k>front_layer&&k<=bi&&remaining>T(0);
    T take=eligible?mn(remaining,mx(T(0),c[6*k]-p[10])):T(0);
    remaining=remaining-take;
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
  t[0]=mn(u[2],u[1]+total/(T(1000)*sy));t[1]=requested;t[2]=total;
  t[3]=mx(T(0),desired-total);t[4]=mx(T(0),requested-desired);t[5]=total<desired;
  t[6]=mr;t[7]=hr;t[8]=T(64)*eps*mx(wm,T(1));t[9]=T(64)*eps*mx(hm,T(1));
  t[26]=main;t[27]=sy;t[28]=front_layer+1;
}
