"""Analytic constraints and the independently archived EHPC393 producer inputs."""
import math
import numpy as np
import pytest
from noahmp_clm.physics.finite_volume import bounded_interface_transfers,close_integrated_transport


CASES=[([5,9],[10,10],[2],[1],[4,10]),
       ([1,5],[10,10],[2],[1],[0,6]),
       ([9,5],[10,10],[-2],[-1],[10,4]),
       ([5,10,10],[10]*3,[2,2],[0,0],[5,10,10]),
       ([0,0,5],[10]*3,[2,2],[0,0],[0,0,5]),
       ([10,0,10],[10]*3,[-1,1],[0,0],[10,0,10]),
       ([5,10,5],[10]*3,[1,-1],[0,0],[5,10,5]),
       ([1,2,3],[10]*3,[.125,-.25],[.125,-.25],[.875,2.375,2.75]),
       ([0],[1],[],[],[0])]


def update(old,faces):
    z=np.zeros(1,old.dtype)
    return (old+np.r_[z,faces])-np.r_[faces,z]


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('case',CASES)
def test_directed_capacity_and_dry_constraints(case,dtype):
    old,capacity,net,expected,water=[np.array(v,dtype) for v in case]
    original=[v.copy() for v in (old,capacity,net)]
    fixed=bounded_interface_transfers(old,capacity,net,roundoff_only=False)
    np.testing.assert_array_equal(fixed,expected)
    np.testing.assert_array_equal(update(old,fixed),water)
    assert math.fsum(water)==math.fsum(old)
    for left,right in zip(original,(old,capacity,net)):np.testing.assert_array_equal(left,right)
    if np.array_equal(net,expected):assert fixed.tobytes()==net.tobytes()


@pytest.mark.parametrize('direction',[1.,-1.])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_returned_parcel_keeps_its_original_donor_heat(direction,dtype):
    old=np.array([5,9] if direction>0 else [9,5],dtype)
    cap=np.array([10,10],dtype);net=np.array([2*direction],dtype)
    forward=np.maximum(net,0);reverse=np.maximum(-net,0)
    hf=forward*340000.;hr=reverse*350000.;heat=hf-hr
    fixed,f,r,h=close_integrated_transport(old,cap,net,forward,reverse,
        heat=heat,forward_heat=hf,reverse_heat=hr,roundoff_only=False)
    np.testing.assert_array_equal(fixed,[direction])
    np.testing.assert_array_equal(f,[max(direction,0)])
    np.testing.assert_array_equal(r,[max(-direction,0)])
    expected=[1660000.,3340000.] if direction>0 else [2350000.,2650000.]
    energy=update(np.array([2e6,3e6],dtype),h)
    np.testing.assert_array_equal(energy,expected)
    assert math.fsum(energy)==5e6


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_large_bad_solve_cannot_be_hidden_as_roundoff(dtype):
    with pytest.raises(RuntimeError,match='exceeds nonlinear/roundoff'):
        bounded_interface_transfers(np.array([5,9],dtype),np.array([10,10],dtype),np.array([2],dtype))
    with pytest.raises(ValueError,match='valid original'):
        bounded_interface_transfers(np.array([5,11],dtype),np.array([10,10],dtype),np.array([0],dtype))


@pytest.mark.parametrize('old,net,expected',[
    ([40.73590850830078,115.50540161132812,235.37063598632812,449.9999694824219],
     [2.705265998840332,2.0867185592651367,4.6519908210029826e-5],
     [2.705265998840332,2.0867185592651367,1.6002330085029826e-5]),
    ([39.119747161865234,113.99974822998047,229.94784545898438,449.6479797363281],
     [1.721972107887268,1.669431447982788,.3520396053791046],
     [1.721972107887268,1.669431447982788,.3520090878009796]),
])
def test_original_cuda_producer_faces_against_independent_ehpc393(old,net,expected):
    old=np.array(old,np.float32);capacity=np.array([45,135,270,450],np.float32)
    net=np.array(net,np.float32);fixed=bounded_interface_transfers(old,capacity,net)
    np.testing.assert_array_equal(fixed,np.array(expected,np.float32))
    water=update(old,fixed)
    assert np.all((water>=0)&(water<=capacity))
    exact=old.astype(np.longdouble)+np.r_[np.longdouble(0),fixed.astype(np.longdouble)]-np.r_[fixed.astype(np.longdouble),np.longdouble(0)]
    assert np.all((exact>=0)&(exact<=capacity))
    bound=32*np.finfo(old.dtype).eps*max(math.fsum(abs(water))+math.fsum(abs(old)),1.)
    assert abs(math.fsum(water)-math.fsum(old))<=bound
