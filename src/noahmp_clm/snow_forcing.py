"""Monthly atmospheric BC/dust climatology mapped to a single snow column.

No aerosol transport is simulated. Reading the public UCAR NetCDF is optional
preprocessing (SciPy); interpolation and runtime deposition need only NumPy.
The resulting rates are kg m-2 s-1, total wet+dry deposition.
"""
from dataclasses import dataclass
from pathlib import Path
import calendar
import hashlib
import numpy as np


@dataclass(frozen=True)
class SnowDepositionClimatology:
    latitude: np.ndarray
    longitude: np.ndarray
    rates: np.ndarray  # month, latitude, longitude, species (BC, dust01..04)
    source: str = 'caller supplied'
    source_sha256: str | None = None

    def __post_init__(self):
        lat=np.array(self.latitude,float,copy=True);lon=np.array(self.longitude,float,copy=True)
        rates=np.array(self.rates,float,copy=True)
        if (lat.ndim!=1 or lon.ndim!=1 or len(lat)<2 or len(lon)<2
                or rates.shape!=(12,len(lat),len(lon),5)
                or not all(np.all(np.isfinite(x)) for x in (lat,lon,rates))
                or np.any(np.diff(lat)<=0) or np.any(np.diff(lon)<=0)
                or lat[0]<-90 or lat[-1]>90 or lon[0]<0 or lon[-1]>=360 or np.any(rates<0)):
            raise ValueError('Expected ascending latitude/0..360 longitude and nonnegative 12-month rates')
        for name,array in (('latitude',lat),('longitude',lon),('rates',rates)):
            array.setflags(write=False);object.__setattr__(self,name,array)

    @classmethod
    def from_netcdf(cls,path,*,expected_sha256=None):
        from scipy.io import netcdf_file
        path=Path(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest!=expected_sha256:
            raise ValueError('Snow deposition source checksum mismatch')
        names=[('BCDEPWET','BCPHIDRY','BCPHODRY')]+[
            (f'DSTX{j:02d}WD',f'DSTX{j:02d}DD') for j in range(1,5)]
        with netcdf_file(path,mmap=False) as data:
            def flux(name):
                v=data.variables[name];units=getattr(v,'units',b'')
                if isinstance(units,bytes): units=units.decode('utf-8')
                if units.strip() not in ('kg/m2/s','kg m-2 s-1'):
                    raise ValueError(f'Unsupported deposition units for {name}: {units}')
                return np.array(v[:],float)
            rates=np.stack([sum(flux(name) for name in group) for group in names],axis=-1)
            lat=np.array(data.variables['lat'][:],float);lon=np.mod(data.variables['lon'][:],360.)
        ilat=np.argsort(lat);ilon=np.argsort(lon)
        return cls(lat[ilat],lon[ilon],rates[:,ilat][:,:,ilon],str(path),digest)

    def at_site(self,latitude,longitude,day_of_year,year):
        """Bilinear space and cyclic linear monthly-midpoint interpolation.

        Day zero is Jan 1 00:00, fractional days allowed; leap years use their
        actual month lengths. Longitudes wrap at 360; latitude is held at the
        nearest data row outside the supplied grid. No extrapolated negatives.
        """
        if (not np.all(np.isfinite([latitude,longitude,day_of_year]))
                or not -90<=latitude<=90 or not isinstance(year,(int,np.integer))
                or not 1<=year<=9999):
            raise ValueError('Invalid deposition site, date or year')
        lat=np.clip(latitude,self.latitude[0],self.latitude[-1])
        ilat=int(np.clip(np.searchsorted(self.latitude,lat,side='right')-1,0,len(self.latitude)-2))
        fy=(lat-self.latitude[ilat])/(self.latitude[ilat+1]-self.latitude[ilat])
        # Shift the longitude relative to the first supplied center so a
        # nonzero-origin grid also wraps correctly across its last cell.
        lon=(longitude-self.longitude[0])%360.+self.longitude[0]
        extended=np.r_[self.longitude,self.longitude[0]+360.]
        j=int(np.searchsorted(extended,lon,side='right')-1);jp=(j+1)%len(self.longitude)
        fx=(lon-extended[j])/(extended[j+1]-extended[j])
        monthly=((1-fy)*((1-fx)*self.rates[:,ilat,j]+fx*self.rates[:,ilat,jp])+
            fy*((1-fx)*self.rates[:,ilat+1,j]+fx*self.rates[:,ilat+1,jp]))
        lengths=np.array([calendar.monthrange(year,m)[1] for m in range(1,13)])
        ends=np.cumsum(lengths);centers=ends-.5*lengths;days=int(ends[-1])
        day=day_of_year%days
        times=np.r_[centers[-1]-days,centers,centers[0]+days]
        values=np.concatenate((monthly[-1:],monthly,monthly[:1]))
        return np.array([np.interp(day,times,values[:,k]) for k in range(5)])
